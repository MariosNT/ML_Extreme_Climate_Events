"""
Improved sampling code
"""

import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal
import pylab as plt
from Sampler import EllipticalSliceSampling
from timeseries_v3 import cptimeseries
from timeseries_extreme import cptimeseries_extreme
import sys
from joblib import Parallel, delayed

year = 2000 #For now, we're focusing on a single year
extreme_case = True

location = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\code\\images\\year_'+str(year)+"\\"

# Model fields
X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}.npy'.format(year))
# Rain fall
Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\Rainfall_Cardiff_{}.npy'.format(year))
print(Y.shape)



##### Defining the priors from Sherman's paper .... without prior on sigmas, so just taking mean for them
if extreme_case:
    theta_0 = np.concatenate(([-0.46, 0, 0, 0, 0, 0, 1.44, 0, 0, 0, 0, 0, -0.45, 0, 0, 0, 0, 0], np.zeros(shape=(32,))))
    true_theta = theta_0
    Sigma_0 = np.diag(np.concatenate(((1/6)*np.ones(shape=(30,)), (1/(1.3*65))*np.ones(shape=(20,)))))

else:
    theta_0 = np.concatenate(([-0.46, 0, 0, 0, 0, 0, 1.44, 0, 0, 0, 0, 0, -0.45, 0, 0, 0, 0, 0], np.zeros(shape=(20,))))
    
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

    Sigma_0 = np.diag(np.concatenate(((1/6)*np.ones(shape=(18,)), (1/(1.3*65))*np.ones(shape=(20,)))))


#### Simulated data
if extreme_case:
    z, y, lambda_t, _, _ = cptimeseries_extreme(true_theta).simulate(X)
    print(cptimeseries_extreme(true_theta).loglikelihood(z, y, X)[1])
else:
    z, y, lambda_t, _, _ = cptimeseries(true_theta).simulate(X)
    print(cptimeseries(true_theta).loglikelihood(z, y, X)[1])


#### Now we want to implment a Gibbs sample where we update theta and z one after another

# number of steps Gibbs we want to use
n_step_Gibbs = 1

### Lists to store the samples
Theta, Z = [], []

# Extract zero/non-zero indices of y
en = np.arange(len(Y))
bool_y_zero = (Y==0)

zero_y_indices = en[bool_y_zero]
nonzero_y_indices = en[np.invert(bool_y_zero)]

## Lets first initialize theta and z for a Markov chain ##

#### For non-zero y, get distribution of rainfalls and calculate quantiles
#### Then use this to initialise z (1, 2, 3, 4), based on the quantiles
y_non_zero = Y[Y>0]
edge1 = np.quantile(y_non_zero, 0.25)
edge2 = np.quantile(y_non_zero, 0.5)
edge3 = np.quantile(y_non_zero, 0.75)
edge4 = np.max(Y)

bin_2 = (edge1<=Y) & (Y<=edge2)
bin_3 = (edge2<Y) & (Y<=edge3)
bin_4 = (edge3<Y) & (Y<=edge4)

z_state = np.ones(shape=Y.shape)
z_state[bin_2] = 2
z_state[bin_3] = 3
z_state[bin_4] = 4

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
    prob_z[ind_z] = loglikelihood_z(possible_z)[0]*np.random.poisson(loglikelihood_z(possible_z)[1])
    return prob_z

perc = 0.5


for ind_Gibbs in range(n_step_Gibbs):
    #print(ind_Gibbs)
    theta_state = copy.deepcopy(Theta[-1])
    z_state = copy.deepcopy(Z[-1])
    while True:
        try:
            #### First sample theta using Elliptic Slice Sampler ###
            if extreme_case:
                loglikelihood_theta = lambda theta: cptimeseries_extreme(theta).loglikelihood(z_state, Y, X)
                Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=theta_0, Sigma=Sigma_0,
                                                  f_0=theta_state)
                theta_state = Samples[-1]
                loglikelihood_z = lambda z: cptimeseries_extreme(theta_state).loglikelihood(z, Y, X)
            else:
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
            nonzero_y = np.random.choice(nonzero_y_indices, size=int(perc*len(nonzero_y_indices)))
            for ind_nonzero in nonzero_y:
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
    if extreme_case:
        print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries_extreme(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],Y, X)[0]))
    else:
        print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],Y, X)[0]))
