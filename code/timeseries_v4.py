import numpy as np
from scipy.stats import gamma
import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal
import pylab as plt
from Sampler import EllipticalSliceSampling
import sys
from joblib import Parallel, delayed
from timeit import default_timer as timer


class cptimeseries():
    def __init__(self, theta, k=6, p=5):
        self.beta_lambda = theta[:k,]
        self.beta_mu = theta[k:2*k,]
        self.beta_omega = theta[2*k:3*k,]
        self.phi_lambda = theta[3*k:3*k+p,]
        self.phi_mu = theta[3*k+p:3*k+2*p,]
        self.gamma_lambda = theta[3*k+2*p:3*k+3*p,]
        self.gamma_mu = theta[3*k+3*p:,]


    def simulate(self, X):

        num_model_field, T = X.shape[1], X.shape[0]
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1) #Adds a column of 1s at the beginning of X
        # Only linear regression term
        omega_t = np.exp(np.dot(XX, self.beta_omega)) #Eq. 60 - k=0
        lambda_t = np.exp(np.dot(XX, self.beta_lambda)) #Eq. 58 - Φ, Γ, k=0
        mu_t = np.exp(np.dot(XX, self.beta_mu)) #Eq. 59 - Φ, Γ, k=0
        # Add ARMA term
        z_t, y_t = np.zeros(shape=(T, )), np.zeros(shape=(T, )) #Rows of T zeros
        for ind_t in range(T): # We loop for all values of model fields
            if ind_t == 0:
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
                # Simulate y_t - Eq. 4, 31, 32
                if z_t[ind_t] == 0:
                    y_t[ind_t] = 0
                else:
                    y_t[ind_t] = np.random.gamma(shape=z_t[ind_t] / omega_t[ind_t], scale=1 / (omega_t[ind_t] * mu_t[ind_t]))
            else:
                # calculate lambda_t - Eq. 34-41 or/and 64, 65
                lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + np.sum(self.phi_lambda[-min(ind_t, 5):] *\
                                        (np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_lambda[0])) \
                                         + np.sum(self.gamma_lambda[-min(ind_t, 5):] *\
                                        ((z_t[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]) /\
                                         np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t]))))
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
                # Calculate mu_t
                num = np.nan_to_num(y_t[ind_t - (min(ind_t, 5)):ind_t] - (z_t[ind_t - (min(ind_t, 5)):ind_t] *
                       mu_t[ind_t - (min(ind_t, 5)):ind_t]))
                deno = np.nan_to_num(mu_t[ind_t - (min(ind_t,5)):ind_t] *
                       np.sqrt(z_t[ind_t - (min(ind_t, 5)):ind_t] * omega_t[ind_t - (min(ind_t, 5)):ind_t] ))
                MA_comp = np.sum(self.gamma_mu[-min(ind_t, 5):][z_t[ind_t - (min(ind_t, 5)):ind_t]!=0]
                                 * (num[z_t[ind_t - (min(ind_t, 5)):ind_t]!=0] / deno[z_t[ind_t - (min(ind_t, 5)):ind_t]!=0]))
                mu_t[ind_t] = np.exp(np.log(mu_t[ind_t]) + np.sum(self.phi_mu[-min(ind_t, 5):] *\
                                    (np.log(mu_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_mu[0])) + MA_comp)
                # Simulate y_t
                if z_t[ind_t] == 0:
                    y_t[ind_t] = 0
                else:
                    y_t[ind_t] = np.random.gamma(shape= z_t[ind_t] / omega_t[ind_t], scale = 1 / (omega_t[ind_t] * mu_t[ind_t]))

        return z_t, y_t

    def loglikelihood(self, z, y, X):

        num_model_field, T = X.shape[1], X.shape[0]
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1)
        ## We check whether there are some days when z and y are not both 0, if so then llhd is -np.inf, ow we calculate llhd
        check = True
        for ind in range(T):
            if y[ind] == 0 and z[ind] !=0 or y[ind] != 0 and z[ind] ==0:
                check = False
        if check == True:
            ### Initialize likelihood ####
            llhd = 0

            # Only linear regression term
            omega_t = np.exp(np.dot(XX, self.beta_omega))
            lambda_t = np.exp(np.dot(XX, self.beta_lambda))
            mu_t = np.exp(np.dot(XX, self.beta_mu))

            for ind_t in range(T):
                ####### Loop over data for every timepoint ########
                if ind_t > 0:
                    ########### Update lambda_t and mu_t if ind_t > 0 -- Add ARMA term #############
                    #### nonzero zs
                    nonzero_z = z[ind_t - (min(ind_t, 5)):ind_t] != 0
                    #### Update lambda_t
                    num = z[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]
                    deno = np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t])
                    MA_comp = np.sum(self.gamma_lambda[-min(ind_t, 5):][nonzero_z] * (num[nonzero_z] / deno[nonzero_z]) )
                    # update lambda_t
                    lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + np.sum(self.phi_lambda[-min(ind_t, 5):] *\
                                      (np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_lambda[0]))
                                             + MA_comp)
                    #print('lambda_t :' +str(lambda_t[ind_t]))
                    #### Update mu_t
                    num = (y[ind_t - (min(ind_t, 5)):ind_t] - z[ind_t - (min(ind_t, 5)):ind_t] *
                           mu_t[ind_t - (min(ind_t, 5)):ind_t])
                    deno = (mu_t[ind_t - (min(ind_t, 5)):ind_t] *
                            np.sqrt(z[ind_t - (min(ind_t, 5)):ind_t] * omega_t[ind_t - (min(ind_t, 5)):ind_t]))
                    MA_comp = np.sum(self.gamma_mu[-min(ind_t, 5):][nonzero_z]
                                     * (num[nonzero_z] / deno[nonzero_z]))
                    mu_t[ind_t] = np.exp(np.log(mu_t[ind_t]) + np.sum(self.phi_mu[-min(ind_t, 5):] *\
                                        (np.log(mu_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_mu[0])) \
                                         + MA_comp)
                    #print('mu_t :' +str(mu_t[ind_t]))

                ################ Compute likelihood
                if y[ind_t] > 0 and z[ind_t]>0:
                    llhd += gamma.logpdf(y[ind_t], a = z[ind_t] / omega_t[ind_t], scale = 1 / (omega_t[ind_t] * mu_t[ind_t]))
                elif y[ind_t] == 0 and z[ind_t] == 0:
                    llhd += - lambda_t[ind_t]
            if np.isnan(llhd):
                # Return -ve inf for the theta and z, the timeseries diverges
                final = -np.inf
            else:
                final = llhd
        else:
            # Return -ve inf when y and z are not both zero
            final = -np.inf
        return final



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
n_step_Gibbs = 5

### Lists to store the samples
Theta, Z = [], []

# Extract zero/non-zero indices of y
# zero_y_indices = [i for i, e in enumerate(Y) if e == 0] #"Maybe faster with numpy boolean, but maybe this way useful for later"
# nonzero_y_indices = [i for i, e in enumerate(Y) if e != 0]

en = np.arange(len(Y))
bool_y_zero = (Y==0)

zero_y_indices = en[bool_y_zero]
nonzero_y_indices = en[np.invert(bool_y_zero)]

## Lets first initialize theta and z for a Markov chain ##

#### For non-zero y, get distribution of rainfalls and calculate quantiles
#### Then use this to initialise z (1, 2, 3, 4), based on the quantiles
y_max = np.max(Y)
edges = np.linspace(0, y_max, 5)

bin_2 = (edges[1]<=Y) & (Y<=edges[2])
bin_3 = (edges[2]<Y) & (Y<=edges[3])
bin_4 = (edges[3]<Y) & (Y<=edges[4])

z_state = np.ones(shape=Y.shape)
z_state[bin_2] = 2
z_state[bin_3] = 3
z_state[bin_4] = 4

z_state[zero_y_indices] = 0 #z_state an array of 0, 1
theta_state = theta_0

# Add to stored samples
Theta.append(copy.deepcopy(theta_state))
Z.append(copy.deepcopy(z_state))

n, bins, patches = plt.hist(Y, 20, density=True, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
plt.grid(True)
plt.show()


"""
slices = np.arange(5, 55, 5)
percentages = np.arange(0.1, 1, 0.1)
times_tot = []

#for slice in slices:
for perc in percentages:
    start = timer()
    
    def parallel_indices(ind_non, ind_z, possible_z, loglikelihood_z):
        possible_z[ind_non] = ind_z + 1
        prob_z[ind_z] = loglikelihood_z(possible_z)
        return prob_z
    
    
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
                #nonzero_y = np.random.choice(nonzero_y_indices, size=slice)
                nonzero_y = np.random.choice(nonzero_y_indices, size=int(perc*len(nonzero_y_indices)))
                for ind_nonzero in nonzero_y:
                    prob_z = np.zeros(9)
                    prob_z = Parallel(n_jobs=4)(delayed(parallel_indices)(ind_nonzero, ind_z, possible_z, loglikelihood_z)\
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
        print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],Y, X)))
    
    end = timer()
    print()
    print("Time elapsed:", end - start)
    print()
    times_tot.append(end-start)
    
# plt.figure(figsize=(10, 8))
# plt.plot(slices, times_tot, marker='o', color = 'b')
# plt.plot(slices, slices, linestyle='--', color = 'black')
# plt.xlabel("Number of points used")
# plt.ylabel("Time (sec)")
# plt.show()

plt.figure(figsize=(10, 8))
plt.plot(percentages, times_tot, marker='o', color = 'b')
plt.plot(percentages, percentages, linestyle='--', color = 'black')
plt.xlabel("Percentage")
plt.ylabel("Time (sec)")
plt.show()
"""
