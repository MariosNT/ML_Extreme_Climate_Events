"""

"""

import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal
import pylab as plt
from Sampler import EllipticalSliceSampling
import sys
#import joblib
from joblib import Parallel, delayed

#X = np.random.normal(size=(100, 5), loc=1, scale=1) # Returns a 100x2 matrix of random (normal) elements
# Model fields
X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff.npy')
# Calculating transpose such that each row corresponds for a day
X = np.transpose(X)
# Calculating windspeed and consider that as avraible
X = np.concatenate((X[:,[0,3,4,5]],np.sqrt(pow(X[:,1],2)+pow(X[:,2],2)).reshape(-1,1)), axis=1)
# Standardize data (making each column having 0 mean and stdev 1)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# Rain fall
Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\rainfall_Cardiff_1979.npy')
print(Y.shape)



##### Defining the priors from Sherman's paper .... without prior on sigmas, so just taking mean for them
theta_0 = np.concatenate(([-0.46, 0, 0, 0, 0, 0, 1.44, 0, 0, 0, 0, 0, -0.45, 0, 0, 0, 0, 0], np.zeros(shape=(20,))))
Sigma_0 = np.diag(np.concatenate(((1/6)*np.ones(shape=(18,)), (1/(1.3*65))*np.ones(shape=(20,)))))


## Realistic priors ##
# Sampling from prior to define a true_theta

beta_lambda, beta_mu, beta_omega = np.random.normal(size=(6,), loc=[-0.46, 0, 0, 0, 0, 0], scale=1/6), \
                                   np.random.normal(size=(6,), loc=[1.44, 0, 0, 0, 0, 0], scale=1/6), \
                                   np.random.normal(size=(6,), loc=[-0.45, 0, 0, 0, 0, 0], scale=1/6)
phi_lambda, phi_mu, gamma_lambda, gamma_mu = np.random.normal(size=(5,), scale=1/(1.3*65)),\
                                             np.random.normal(size=(5,), scale=1/(1.3*65)),\
                                             np.random.normal(size=(5,), scale=1/(1.3*65)),\
                                             np.random.normal(size=(5,), scale=1/(1.3*65))
true_theta = np.array([])
for array in [beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu]:
    true_theta = np.concatenate([true_theta, array])


#### Simulated data
z, y = cptimeseries(true_theta).simulate(X)
print(cptimeseries(true_theta).loglikelihood(z, y, X))


#### Now we want to implment a Gibbs sample where we update theta and z one after another

# number of steps Gibbs we want to use
n_step_Gibbs = 1

### Lists to store the samples
Theta, Z = [], []

# Extract zero/non-zero indices of y
zero_y_indices = [i for i, e in enumerate(Y) if e == 0] #"Maybe faster with numpy boolean, but maybe this way useful for later"
nonzero_y_indices = [i for i, e in enumerate(Y) if e != 0]

## Lets first initialize theta and z for a Markov chain ##

#### For non-zero y, get distribution of rainfalls and calculate quantiles
#### Then use this to initialise z (1, 2, 3, 4), based on the quantiles
z_state = np.ones(shape=Y.shape)
z_state[zero_y_indices] = 0 #z_state an array of 0, 1
theta_state = theta_0
# Add to stored samples
Theta.append(copy.deepcopy(theta_state))
Z.append(copy.deepcopy(z_state))

#### Serial case

"""
for ind_Gibbs in range(n_step_Gibbs):
    #print(ind_Gibbs)
    theta_state = copy.deepcopy(Theta[-1])
    z_state = copy.deepcopy(Z[-1])
    while True:
        try:
            #### First sample theta using Elliptic Slice Sampler ###
            # define conditional likelihood for theta
            loglikelihood_theta = lambda theta: cptimeseries(theta).loglikelihood(z_state, Y, X)
            # Sample/Update theta
            ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
            ## f_0 defines the present state of the Markov chain
            Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=theta_0, Sigma=Sigma_0,
                                              f_0=theta_state)
            theta_state = Samples[-1]
            # define conditional likelihood for z
            loglikelihood_z = lambda z: cptimeseries(theta_state).loglikelihood(z, Y, X)
            # Sample/Update z
            possible_z = z_state
            ### Check to delete some subset of the loop
            for ind_nonzero in nonzero_y_indices:
                prob_z = np.zeros(9)
                for ind_z in range(9):  ### Why 9 here? Times it rains/day
                    possible_z[ind_nonzero] = ind_z + 1
                    prob_z[ind_z] = loglikelihood_z(possible_z)
                print(prob_z)
                finite_indices = np.isfinite(prob_z)
                prob_z = np.exp(prob_z[finite_indices] - np.min(prob_z[finite_indices]))
                possible_z[ind_nonzero] = np.random.choice(a=np.arange(1, 10)[finite_indices],
                                                           p=prob_z / np.sum(prob_z))
            z_state = possible_z
        except (RuntimeError, ValueError, TypeError, NameError, ZeroDivisionError, OSError):
            continue
        break
    print(str(ind_Gibbs)+'-st/th iteration successfully finished' )
    # Add to stored samples
    Theta.append(copy.deepcopy(theta_state))
    Z.append(copy.deepcopy(z_state))
    
    # Shapes
    # Theta -> (N_Gibbs, 38)
    # Z -> (N_Gibbs, 365)

    print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],Y, X)))
"""


#np.savez('timeseries_samples', Z=Z, Theta=Theta)


# plt.figure(figsize=(10, 8))
# plt.plot(time, z_new, linestyle = '-', color = 'b')
# plt.plot(time, Y, marker='+', linestyle='', color = 'black')
# plt.fill_between(time, z_new-y_new, z_new+y_new)
# plt.ylim(0, np.max(z_new+y_new)+1)


#### Parallel Case

def parallel_indices(ind_non, ind_z, possible_z, loglikelihood_z):
    possible_z[ind_non] = ind_z + 1
    prob_z[ind_z] = loglikelihood_z(possible_z)
    return prob_z

"""
for ind_Gibbs in range(n_step_Gibbs):
    #print(ind_Gibbs)
    theta_state = copy.deepcopy(Theta[-1])
    z_state = copy.deepcopy(Z[-1])
    while True:
        try:
            #### First sample theta using Elliptic Slice Sampler ###
            # define conditional likelihood for theta
            loglikelihood_theta = lambda theta: cptimeseries(theta).loglikelihood(z_state, Y, X)
            # Sample/Update theta
            ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
            ## f_0 defines the present state of the Markov chain
            Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=theta_0, Sigma=Sigma_0,
                                              f_0=theta_state)
            theta_state = Samples[-1]
            # define conditional likelihood for z
            loglikelihood_z = lambda z: cptimeseries(theta_state).loglikelihood(z, Y, X)
            # Sample/Update z
            possible_z = z_state
            for ind_nonzero in nonzero_y_indices:
                prob_z = np.zeros(9)
                prob_z = Parallel(n_jobs=4)(delayed(parallel_indices)(ind_nonzero, ind_z, possible_z, loglikelihood_z)\
                       for ind_z in range(9))
                prob_z = np.sum(prob_z, axis=0)
                print(prob_z)
                finite_indices = np.isfinite(prob_z)
                prob_z = np.exp(prob_z[finite_indices] - np.min(prob_z[finite_indices]))
                possible_z[ind_nonzero] = np.random.choice(a=np.arange(1, 10)[finite_indices],
                                                           p=prob_z / np.sum(prob_z))
            z_state = possible_z
        except (RuntimeError, ValueError, TypeError, NameError, ZeroDivisionError, OSError):
            continue
        break
    print(str(ind_Gibbs)+'-st/th iteration successfully finished' )
    # Add to stored samples
    Theta.append(copy.deepcopy(theta_state))
    Z.append(copy.deepcopy(z_state))
    print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],Y, X)))
"""