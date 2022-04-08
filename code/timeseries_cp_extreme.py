import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal, genpareto
from scipy import optimize
import pylab as plt
from Sampler import EllipticalSliceSampling
from joblib import Parallel, delayed


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
        n_X = X.shape[0]
        Y = np.zeros(shape=(X.shape[0], X.shape[1]))
        Z = np.zeros(shape=(X.shape[0], X.shape[1]))
        Lambda = np.zeros(shape=(X.shape[0], X.shape[1]))
        Omega = np.zeros(shape=(X.shape[0], X.shape[1]))
        Mu = np.zeros(shape=(X.shape[0], X.shape[1]))
        for ind in range(n_X):
            z_t, y_t, lambda_t, omega_t, mu_t = self._simulate_one(np.squeeze(X[ind,:,:]))
            Y[ind, :], Z[ind,:], Lambda[ind,:], Omega[ind, :], Mu[ind, :] = y_t, z_t, lambda_t, omega_t, mu_t
        return Z, Y, Lambda, Omega, Mu

    def loglikelihood(self, Z, Y, X):
        n_X = X.shape[0]
        lld = 0
        for ind in range(n_X):
            lld = lld + self._loglikelihood_one(Z[ind,:], Y[ind, :], np.squeeze(X[ind, :, :]))
        return lld

    def _simulate_one(self, X):
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
                    #print(u_t[ind_t], sigma_t[ind_t])
                    y_t[ind_t] = genpareto.rvs(c=eta_t[ind_t], loc=u_t[ind_t], scale=sigma_t[ind_t])
                    #print('extreme value :'+str(y_t[ind_t]))
        return z_t, y_t, lambda_t, omega_t, mu_t

    def inv_CDF_mixture(self, p, C_t, z_t, omega_t, mu_t, u_t, eta_t, sigma_t):
        if p > C_t:
            return genpareto.ppf((p-C_t)/(1-C_t), c=eta_t, loc=u_t, scale=sigma_t)
        else:
            return gamma.ppf(p, a=z_t/omega_t, scale=1 / (omega_t * mu_t))

    def _loglikelihood_one(self, z, y, X):
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

    def _compute_lambda_one(self, z, y, X):
        num_model_field, T = X.shape[1], X.shape[0]

        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1)
        # Only linear regression term
        lambda_t = np.exp(np.dot(XX, self.beta_lambda))
        # Add ARMA term
        z_t, y_t = z, y
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
        return lambda_t