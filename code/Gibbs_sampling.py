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

year = 1 #For now, we're focusing on a single year
extreme_case = False

location = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\code\\images\\year_'+str(year)+"\\"

# Model fields
multiple = True

if multiple:
    X1 = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}_wv.npy'.format(year))
    X2 = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}_hum.npy'.format(year))

    X2 = X2[:,5].reshape(len(X2),1)

    X = np.hstack([X1, X2])
    
else:
    X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}_wv.npy'.format(year))

x_size = X.shape[1]+1

# Rain fall
Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\Rainfall_Cardiff_{}.npy'.format(year))
print(Y.shape)



##### Defining the priors from Sherman's paper .... without prior on sigmas, so just taking mean for them
diff = x_size-6
if extreme_case:
    true_theta = np.concatenate(([-0.46, 0, 0, 0, 0, 0], np.zeros(diff), [1.44, 0, 0, 0, 0, 0], np.zeros(diff),\
                                 [-0.45, 0, 0, 0, 0, 0], np.zeros(diff), np.zeros(shape=(32+diff*3,))))
    Sigma_0 = np.diag(np.concatenate(((1/6)*np.ones(shape=(30+diff*3,)), (1/(1.3*65))*np.ones(shape=(20+diff*3,)))))

else:   
    ## Realistic priors ##
    # Sampling from prior to define a true_theta
    loc1 = np.concatenate(([-0.46, 0, 0, 0, 0, 0,], np.zeros(diff)))
    loc2 = np.concatenate(([1.44, 0, 0, 0, 0, 0,], np.zeros(diff)))
    loc3 = np.concatenate(([-0.45, 0, 0, 0, 0, 0,], np.zeros(diff)))
    
    
    beta_lambda, beta_mu, beta_omega = np.random.normal(size=(x_size,), loc=loc1, scale=1/6), \
                                       np.random.normal(size=(x_size,), loc=loc2, scale=1/6), \
                                       np.random.normal(size=(x_size,), loc=loc3, scale=1/6)
    phi_lambda, phi_mu, gamma_lambda, gamma_mu = np.random.normal(size=(5,), scale=1/(1.3*65)),\
                                                 np.random.normal(size=(5,), scale=1/(1.3*65)),\
                                                 np.random.normal(size=(5,), scale=1/(1.3*65)),\
                                                 np.random.normal(size=(5,), scale=1/(1.3*65))
    true_theta = np.array([])
    for array in [beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu]:
        true_theta = np.concatenate([true_theta, array])

    Sigma_0 = np.diag(np.concatenate(((1/6)*np.ones(shape=(18+diff*3,)), (1/(1.3*65))*np.ones(shape=(20,)))))


#### Simulated data
if extreme_case:
    z, y, lambda_t, _, _ = cptimeseries_extreme(true_theta, k=x_size).simulate(X)
    print(cptimeseries_extreme(true_theta, k=x_size).loglikelihood(z, y, X))
else:
    z, y, lambda_t, _, _ = cptimeseries(true_theta, k=x_size).simulate(X)
    print(cptimeseries(true_theta, k=x_size).loglikelihood(z, y, X))


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
theta_state = true_theta

# Add to stored samples
Theta.append(copy.deepcopy(theta_state))
Z.append(copy.deepcopy(z_state))

#### Parallel Case

def parallel_indices(ind_non, ind_z, possible_z, loglikelihood_z):
    possible_z[ind_non] = ind_z + 1
    #### This is wrong - include the prior in z inside the loglikelihood (final step)
    prob_z[ind_z] = loglikelihood_z(possible_z) #[0]*np.log(np.random.poisson(loglikelihood_z(possible_z)[1]))
    return prob_z

perc = 0.1


for ind_Gibbs in range(n_step_Gibbs):
    #print(ind_Gibbs)
    theta_state = copy.deepcopy(Theta[-1])
    z_state = copy.deepcopy(Z[-1])
    while True:
        try:
            #### First sample theta using Elliptic Slice Sampler ###
            if extreme_case:
                # define conditional likelihood for theta
                loglikelihood_theta = lambda theta: cptimeseries_extreme(theta, k=x_size).loglikelihood(z_state, Y, X)
                # Sample/Update theta
                ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
                ## f_0 defines the present state of the Markov chain
                Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=true_theta, Sigma=Sigma_0,
                                                  f_0=theta_state)
                theta_state = Samples[-1]
                # define conditional likelihood for z
                loglikelihood_z = lambda z: cptimeseries_extreme(theta_state, k=x_size).loglikelihood(z, Y, X)
            else:
                # define conditional likelihood for theta
                loglikelihood_theta = lambda theta: cptimeseries(theta, k=x_size).loglikelihood(z_state, Y, X)
                # Sample/Update theta
                ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
                ## f_0 defines the present state of the Markov chain
                Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=true_theta, Sigma=Sigma_0,
                                                  f_0=theta_state)
                theta_state = Samples[-1]
                # define conditional likelihood for z
                loglikelihood_z = lambda z: cptimeseries(theta_state, k=x_size).loglikelihood(z, Y, X)
            # Sample/Update z
            possible_z = z_state
            nonzero_y = np.random.choice(nonzero_y_indices, size=int(perc*len(nonzero_y_indices)))
            for ind_nonzero in nonzero_y:
                prob_z = np.zeros(9)
                prob_z = Parallel(n_jobs=4, prefer="threads")(delayed(parallel_indices)(ind_nonzero, ind_z, possible_z, loglikelihood_z)\
                       for ind_z in range(9))
                prob_z = np.sum(prob_z, axis=0)
                #print(prob_z)
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
        print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries_extreme(Theta[ind_Gibbs], k=x_size).loglikelihood(Z[ind_Gibbs],Y, X)))
    else:
        print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries(Theta[ind_Gibbs], k=x_size).loglikelihood(Z[ind_Gibbs],Y, X)))
