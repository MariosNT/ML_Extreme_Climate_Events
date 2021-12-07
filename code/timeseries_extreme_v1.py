import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal, genpareto
from scipy import optimize
import pylab as plt
from Sampler import EllipticalSliceSampling

#X = np.random.normal(size=(100, 2), loc=1, scale=1)
# Model fields
X = np.load('../Data/Data/model_fields_Cardiff.npy')
# Calculating transpose such that each row corresponds for a day
X = np.transpose(X)
# Calculating windspeed and consider that as avraible
X = np.concatenate((X[:,[0,3,4,5]],np.sqrt(pow(X[:,1],2)+pow(X[:,2],2)).reshape(-1,1)), axis=1)
# Standardize data (making each column having 0 mean and stdev 1)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
# Rain fall
Y = np.load('../Data/Data/rainfall_Cardiff_1979.npy')
print(Y.shape)


class cptimeseries_extreme():
    def __init__(self, theta, k=6, p=5):
        self.beta_lambda = theta[:k,]
        self.beta_mu = theta[k:2*k,]
        self.beta_omega = theta[2*k:3*k,]
        self.beta_u = theta[3*k:4*k, ]
        self.beta_sigma = theta[4*k:5*k, ]
        self.phi_lambda = theta[5*k:5*k+p,]
        self.phi_mu = theta[5*k+p:5*k+2*p,]
        self.gamma_lambda = theta[5*k+2*p:5*k+3*p,]
        self.gamma_mu = theta[5*k+3*p:,]

    def simulate(self, X):

        num_model_field, T = X.shape[1], X.shape[0]
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1)
        # Only linear regression term
        omega_t = np.exp(np.dot(XX, self.beta_omega))
        lambda_t = np.exp(np.dot(XX, self.beta_lambda))
        mu_t = np.exp(np.dot(XX, self.beta_mu))
        u_t = np.exp(np.dot(XX, self.beta_u))
        sigma_t = np.exp(np.dot(XX, self.beta_sigma))
        # Add ARMA term
        z_t, y_t, C_t, eta_t = np.zeros(shape=(T, )), np.zeros(shape=(T, )), np.zeros(shape=(T, )), np.zeros(shape=(T, ))
        IQR, F_inv_2nd_quantile = np.ones(shape=(T, )), np.zeros(shape=(T, ))
        for ind_t in range(T):
            #print(ind_t)
            if ind_t == 0:
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
            else:
                # calculate lambda_t
                #### nonzero zs
                nonzero_z = z_t[ind_t - (min(ind_t, 5)):ind_t] != 0
                #### Update lambda_t
                num = z_t[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]
                deno = np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t])
                MA_comp = np.sum(self.gamma_lambda[-min(ind_t, 5):][nonzero_z] * (num[nonzero_z] / deno[nonzero_z]))
                lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + \
                                         np.sum(self.phi_lambda[-min(ind_t, 5):] * (
                                                     np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t])
                                                      - self.beta_lambda[0])) \
                                        + MA_comp
                                         )
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
                # Calculate mu_t
                ### Update mu_t
                num = y_t[ind_t - (min(ind_t, 5)):ind_t] - F_inv_2nd_quantile[ind_t - (min(ind_t, 5)):ind_t]
                deno = IQR[ind_t - (min(ind_t, 5)):ind_t]
                MA_comp = np.sum(self.gamma_mu[-min(ind_t, 5):][nonzero_z] * (num[nonzero_z] / deno[nonzero_z]))
                mu_t[ind_t] = np.exp(np.log(mu_t[ind_t]) + \
                                     np.sum(self.phi_mu[-min(ind_t, 5):] * (np.log(mu_t[ind_t -
                                     (min(ind_t, 5)):ind_t]) - self.beta_mu[0]))
                                        + MA_comp
                                     )
            # Simulate y_t
            if z_t[ind_t] == 0:
                # Calculate C_t
                C_t[ind_t] = 1
                # Calculate eta_t
                eta_t[ind_t] = 0
                # Calculate F_inv quantiles
                IQR[ind_t] = 1
                F_inv_2nd_quantile[ind_t] = 0
                y_t[ind_t] = 0
            else:
                # Calculate C_t
                C_t[ind_t] = gamma.cdf(u_t[ind_t], a=z_t[ind_t] / omega_t[ind_t],
                                       scale=1 / (omega_t[ind_t] * mu_t[ind_t]))
                # Calculate eta_t
                if (z_t[ind_t] * mu_t[ind_t] - u_t[ind_t]) / sigma_t[ind_t] <= np.log(2):
                    eta_t[ind_t] = 0
                else:
                    root_for_eta = lambda x: ((pow(2, x) - 1) / x) - ((z_t[ind_t] * mu_t[ind_t] - u_t[ind_t]) / sigma_t[
                        ind_t])
                    eta_t[ind_t] = optimize.root(root_for_eta, [0.1]).x
                # Compute Inter Quartile Range
                IQR[ind_t] = self.inv_CDF_mixture(0.75, C_t[ind_t], z_t[ind_t], omega_t[ind_t],
                                                          mu_t[ind_t], u_t[ind_t], eta_t[ind_t], sigma_t[ind_t])\
                             - self.inv_CDF_mixture(0.25, C_t[ind_t], z_t[ind_t], omega_t[ind_t],
                                                          mu_t[ind_t], u_t[ind_t], eta_t[ind_t], sigma_t[ind_t])
                F_inv_2nd_quantile[ind_t] = self.inv_CDF_mixture(0.5, C_t[ind_t], z_t[ind_t], omega_t[ind_t],
                                                          mu_t[ind_t], u_t[ind_t], eta_t[ind_t], sigma_t[ind_t])
                y_t[ind_t] = np.random.gamma(shape= z_t[ind_t] / omega_t[ind_t], scale = 1 / (omega_t[ind_t] * mu_t[ind_t]))
                if y_t[ind_t] >= u_t[ind_t]:
                    print(u_t[ind_t], sigma_t[ind_t])
                    y_t[ind_t] = genpareto.rvs(c=eta_t[ind_t], loc=u_t[ind_t], scale=sigma_t[ind_t])
                    print('extreme value :'+str(y_t[ind_t]))
        return z_t, y_t

    def inv_CDF_mixture(self, p, C_t, z_t, omega_t, mu_t, u_t, eta_t, sigma_t):
        if p > C_t:
            return genpareto.ppf((p-C_t)/(1-C_t), c=eta_t, loc=u_t, scale=sigma_t)
        else:
            return gamma.ppf(p, a=z_t/omega_t, scale=1 / (omega_t * mu_t))

    def loglikelihood(self, z, y, X):
        num_model_field, T = X.shape[1], X.shape[0]
        ## We check whether there are some days when z and y are not both 0, if so then llhd is -np.inf, ow we calculate llhd
        check = True
        for ind in range(T):
            if y[ind] == 0 and z[ind] != 0 or y[ind] != 0 and z[ind] == 0:
                check = False
        if check == True:
            XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1)
            # Only linear regression term
            omega_t = np.exp(np.dot(XX, self.beta_omega))
            lambda_t = np.exp(np.dot(XX, self.beta_lambda))
            mu_t = np.exp(np.dot(XX, self.beta_mu))
            u_t = np.exp(np.dot(XX, self.beta_u))
            sigma_t = np.exp(np.dot(XX, self.beta_sigma))
            # Add ARMA term
            z_t, y_t, C_t, eta_t = z, y, np.zeros(shape=(T,)), np.zeros(shape=(T,))
            IQR, F_inv_2nd_quantile = np.ones(shape=(T,)), np.zeros(shape=(T,))
            llhd = 0
            for ind_t in range(T):
                ####### Loop over data for every timepoint ########
                #print(ind_t)
                if ind_t > 0:
                    ########### Update lambda_t and mu_t if ind_t > 0 -- Add ARMA term #############
                    #### nonzero zs
                    nonzero_z = z_t[ind_t - (min(ind_t, 5)):ind_t] != 0
                    #### Update lambda_t
                    num = z_t[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]
                    deno = np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t])
                    MA_comp = np.sum(self.gamma_lambda[-min(ind_t, 5):][nonzero_z] * (num[nonzero_z] / deno[nonzero_z]) )
                    # update lambda_t
                    lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + \
                                             np.sum(self.phi_lambda[-min(ind_t, 5):] * (
                                                         np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t])
                                                          - self.beta_lambda[0])) \
                                             + MA_comp
                                             )
                    ### Update mu_t
                    num = y_t[ind_t - (min(ind_t, 5)):ind_t] - F_inv_2nd_quantile[ind_t - (min(ind_t, 5)):ind_t]
                    deno = IQR[ind_t - (min(ind_t, 5)):ind_t]
                    MA_comp = np.sum(self.gamma_mu[-min(ind_t, 5):][nonzero_z] * (num[nonzero_z] / deno[nonzero_z]))
                    mu_t[ind_t] = np.exp(np.log(mu_t[ind_t]) + \
                                         np.sum(self.phi_mu[-min(ind_t, 5):] * (np.log(mu_t[ind_t -
                                         (min(ind_t, 5)):ind_t]) - self.beta_mu[0]))
                                            + MA_comp
                                         )
                # Calculate C_t, eta_t, IGR, F_inv_2nd_quantile
                if z_t[ind_t] == 0:
                    # Calculate C_t
                    C_t[ind_t] = 1
                    # Calculate eta_t
                    eta_t[ind_t] = 0
                    # Calculate F_inv quantiles
                    IQR[ind_t] = 1
                    F_inv_2nd_quantile[ind_t] = 0
                else:
                    # Calculate C_t
                    C_t[ind_t] = gamma.cdf(u_t[ind_t], a=z_t[ind_t] / omega_t[ind_t],
                                           scale=1 / (omega_t[ind_t] * mu_t[ind_t]))
                    # Calculate eta_t
                    if (z_t[ind_t] * mu_t[ind_t] - u_t[ind_t]) / sigma_t[ind_t] <= np.log(2):
                        eta_t[ind_t] = 0
                    else:
                        root_for_eta = lambda x: ((pow(2, x) - 1) / x) - ((z_t[ind_t] * mu_t[ind_t] - u_t[ind_t]) / sigma_t[
                            ind_t])
                        eta_t[ind_t] = optimize.root(root_for_eta, [0.1]).x
                    # Compute Inter Quartile Range
                    IQR[ind_t] = self.inv_CDF_mixture(0.75, C_t[ind_t], z_t[ind_t], omega_t[ind_t],
                                                              mu_t[ind_t], u_t[ind_t], eta_t[ind_t], sigma_t[ind_t])\
                                 - self.inv_CDF_mixture(0.25, C_t[ind_t], z_t[ind_t], omega_t[ind_t],
                                                              mu_t[ind_t], u_t[ind_t], eta_t[ind_t], sigma_t[ind_t])
                    F_inv_2nd_quantile[ind_t] = self.inv_CDF_mixture(0.5, C_t[ind_t], z_t[ind_t], omega_t[ind_t],
                                                              mu_t[ind_t], u_t[ind_t], eta_t[ind_t], sigma_t[ind_t])

                if y[ind_t] > 0 and z[ind_t] > 0:
                    if y[ind_t] <= u_t[ind_t]:
                        llhd += gamma.logpdf(y[ind_t], a=z_t[ind_t]/omega_t[ind_t], scale= 1/(omega_t[ind_t] * mu_t[ind_t]))
                    else:
                        llhd += np.log(1-C_t[ind_t]) + genpareto.logpdf(y[ind_t],
                                                        c=eta_t[ind_t], loc=u_t[ind_t], scale=sigma_t[ind_t])
                elif y[ind_t] == 0 and z[ind_t] == 0:
                    llhd += - lambda_t[ind_t]
            if np.isnan(llhd):
                # Return -ve inf for the theta and z, the timeseries diverges
                final = -np.inf
            else:
                final = llhd
        else:
            final = -np.inf
        return final

# ## Realistic priors ##
# # Sampling from prior to define a true_theta
# beta_lambda, beta_mu, beta_omega = np.random.normal(size=(6,), loc=[-0.46, 0, 0, 0, 0, 0], scale=1/6), \
#                                    np.random.normal(size=(6,), loc=[1.44, 0, 0, 0, 0, 0], scale=1/6), \
#                                    np.random.normal(size=(6,), loc=[-0.45, 0, 0, 0, 0, 0], scale=1/6)
# beta_u, beta_sigma = np.random.normal(size=(6,), loc=[0, 0, 0, 0, 0, 0], scale=1/6), \
#                                    np.random.normal(size=(6,), loc=[0, 0, 0, 0, 0, 0], scale=1/100)
# phi_lambda, phi_mu, gamma_lambda, gamma_mu = np.random.normal(size=(5,), scale=1/(1.3*65)),\
#                                              np.random.normal(size=(5,), scale=1/(1.3*65)),\
#                                              np.random.normal(size=(5,), scale=1/(1.3*65)),\
#                                              np.random.normal(size=(5,), scale=1/(1.3*65))
# true_theta = np.array([])
# for array in [beta_lambda, beta_mu, beta_omega, beta_u, beta_sigma,
#               phi_lambda, phi_mu, gamma_lambda, gamma_mu, beta_u, beta_sigma]:
#     true_theta = np.concatenate([true_theta, array])
# #### Simulated data
# print('Simulating data')
# z, y = cptimeseries_extreme(true_theta).simulate(X)
# print('Computing likelihood')
# llhd = cptimeseries_extreme(true_theta).loglikelihood(z, y, X)
# print('LHD :'+ str(llhd))

#### Now we want to implment a Gibbs sample where we update theta and z one after another
theta_0 = np.concatenate(([-0.46, 0, 0, 0, 0, 0, 1.44, 0, 0, 0, 0, 0, -0.45, 0, 0, 0, 0, 0], np.zeros(shape=(32,))))
Sigma_0 = np.diag(np.concatenate(((1/6)*np.ones(shape=(30,)), (1/(1.3*65))*np.ones(shape=(20,)))))

#### Now we want to implment a Gibbs sample where we update theta and z one after another

# number of steps Gibbs we want to use
n_step_Gibbs = 100

### Lists to store the samples
Theta, Z = [], []

# Extract zero/non-zero indices of y
zero_y_indices = [i for i, e in enumerate(Y) if e == 0] #"Maybe faster with numpy boolean, but maybe this way useful for later"
nonzero_y_indices = [i for i, e in enumerate(Y) if e != 0]

## Lets first initialize theta and z for a Markov chain ##
z_state = np.ones(shape=Y.shape)
z_state[zero_y_indices] = 0 #z_state an array of 0, 1
theta_state = theta_0
# Add to stored samples
Theta.append(copy.deepcopy(theta_state))
Z.append(copy.deepcopy(z_state))


for ind_Gibbs in range(n_step_Gibbs):
    #print(ind_Gibbs)
    theta_state = copy.deepcopy(Theta[-1])
    z_state = copy.deepcopy(Z[-1])
    while True:
        try:
            #### First sample theta using Elliptic Slice Sampler ###
            # define conditional likelihood for theta
            loglikelihood_theta = lambda theta: cptimeseries_extreme(theta).loglikelihood(z_state, Y, X)
            # Sample/Update theta
            ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
            ## f_0 defines the present state of the Markov chain
            Samples = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=theta_0, Sigma=Sigma_0,
                                              f_0=theta_state)
            theta_state = Samples[-1]
            # define conditional likelihood for z
            loglikelihood_z = lambda z: cptimeseries_extreme(theta_state).loglikelihood(z, Y, X)
            # Sample/Update z
            possible_z = z_state
            for ind_nonzero in nonzero_y_indices:
                prob_z = np.zeros(9)
                for ind_z in range(9):  ### Why 9 here? Times it rains/day
                    possible_z[ind_nonzero] = ind_z + 1
                    prob_z[ind_z] = loglikelihood_z(possible_z)
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
    print(str(ind_Gibbs)+'-st/th sample LogLikeliHood: '+str(cptimeseries_extreme(Theta[ind_Gibbs]).loglikelihood(Z[ind_Gibbs],Y, X)))
