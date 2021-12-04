import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal
import pylab as plt
from Sampler import EllipticalSliceSampling
import sys

#X = np.random.normal(size=(100, 5), loc=1, scale=1) # Returns a 100x2 matrix of random (normal) elements

X = np.load('../Data/Data/model_fields_Cardiff.npy')
# Calculating transpose such that each row corresponds for a day
X = np.transpose(X)
# Calculating windspeed and consider that as avraible
X = np.concatenate((X[:,[0,3,4,5]],np.sqrt(pow(X[:,1],2)+pow(X[:,2],2)).reshape(-1,1)), axis=1)
# Standardize data (making each column having 0 mean and stdev 1)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# 0 - Does X represent the model fields? YES
# 1 - Why add an 1 in the model fields? For the first step we don't use the ARMA model? We assume β=1? -- corresponds to k
# 2 - Do we fix r to be 5 in eq. (35) here? YES, r & s here set to be 5, but also fix 5=T at line 74
# 3 - Is line 94 correct? Shouldn't we return log Poisson?



""" Q - Why the , at the end? In self.beta etc """

class cptimeseries():
    def __init__(self, theta, k=6, p=5): #Improved verion of inputting prior
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
            llhd = 0
            # Only linear regression term
            omega_t = np.exp(np.dot(XX, self.beta_omega))
            lambda_t = np.exp(np.dot(XX, self.beta_lambda))
            mu_t = np.exp(np.dot(XX, self.beta_mu))
            # Add ARMA term
            for ind_t in range(1, T):
                # nonzero zs
                nonzero_z = z[ind_t - (min(ind_t, 5)):ind_t] != 0

                num = z[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]
                deno = np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t])
                MA_comp = np.sum(self.gamma_lambda[-min(ind_t, 5):][nonzero_z] * (num[nonzero_z] / deno[nonzero_z]) )
                # update lambda_t
                lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + np.sum(self.phi_lambda[-min(ind_t, 5):] *\
                                  (np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_lambda[0]))
                                         + MA_comp)
                #print('lambda_t :' +str(lambda_t[ind_t]))
                # Update mu_t
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
                if y[ind_t] > 0 and z[ind_t]>0:
                    llhd += gamma.logpdf(y[ind_t], a = z[ind_t] / omega_t[ind_t], scale = 1 / (omega_t[ind_t] * mu_t[ind_t]))
                elif y[ind_t] == 0 and z[ind_t] == 0:
                    llhd += - lambda_t[ind_t] # Is this correct? I thought a Poisson should be returned? Is this because of the log
            if np.isnan(llhd):
                # Return -ve inf for the theta and z, the timeseries diverges
                final = -np.inf
            else:
                final = llhd
        else:
            # Return -ve inf when y and z are not both zero
            final = -np.inf
        return final

##### Defining the priors from Sherman's paper .... without prior on sigmas, so just taking mean for them

""" Q - Would it make much difference, which one to select? """


## No realistic priors - just mean values ##
### 29 values for prior θ
### (29x29) matrix for Sigma_0

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
""" Not a 100% sure about this part - need to recheck """

# number of steps Gibbs we want to use
n_step_Gibbs = 100

### Lists to store the samples
Theta, Z = [], []

# Extract zero/non-zero indices of y
zero_y_indices = [i for i, e in enumerate(y) if e == 0] #"Maybe faster with numpy boolean, but maybe this way useful for later"
nonzero_y_indices = [i for i, e in enumerate(y) if e != 0]

## Lets first initialize theta and z for a Markov chain ##
z_state = np.ones(shape=y.shape)
z_state[zero_y_indices] = 0 #z_state an array of 0, 1
theta_state = theta_0

for ind_Gibbs in range(n_step_Gibbs):
    print(ind_Gibbs)

    #### First sample theta using Elliptic Slice Sampler ###
    # define conditional likelihood for theta
    loglikelihood_theta = lambda theta: cptimeseries(theta).loglikelihood(z_state, y, X)

    # Sample/Update theta
    ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
    ## f_0 defines the present state of the Markov chain
    Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n = 1, Mean=theta_0, Sigma=Sigma_0, f_0=theta_state)
    theta_state = Samples[-1]
    Theta.append(copy.deepcopy(theta_state))

    # define conditional likelihood for z
    loglikelihood_z = lambda z: cptimeseries(theta_state).loglikelihood(z, y, X)
    # Sample/Update z
    possible_z = z_state
    for ind_nonzero in nonzero_y_indices:
        prob_z = np.zeros(9)
        for ind_z in range(9): ### Why 9 here? Times it rains/day
            possible_z[ind_nonzero] = ind_z + 1
            prob_z[ind_z] = loglikelihood_z(possible_z)
        #print(prob_z)
        finite_indices = np.isfinite(prob_z)
        prob_z = np.exp(prob_z[finite_indices] - np.min(prob_z[finite_indices]))
        possible_z[ind_nonzero] = np.random.choice(a=np.arange(1, 10)[finite_indices], p=prob_z/np.sum(prob_z))
    z_state = possible_z
    Z.append(copy.deepcopy(z_state))
    print(str(ind_Gibbs)+'-th sample LogLikeliHood: '+str(cptimeseries(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],y, X)))






