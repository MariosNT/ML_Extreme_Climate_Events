# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:32:56 2022

@author: klera
"""

import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal, poisson
import pylab as plt
from Sampler import EllipticalSliceSampling
import sys
#import joblib
from joblib import Parallel, delayed


class cptimeseries():
    def __init__(self, theta, k=8, p=5):
        # Loading the priors
        self.beta_lambda = theta[:k,]
        self.beta_mu = theta[k:2*k,]
        self.beta_omega = theta[2*k:3*k,]
        self.phi_lambda = theta[3*k:3*k+p,]
        self.phi_mu = theta[3*k+p:3*k+2*p,]
        self.gamma_lambda = theta[3*k+2*p:3*k+3*p,]
        self.gamma_mu = theta[3*k+3*p:,]

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
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1) #Adds a column of 1s at the beginning of X
        # Only linear regression term
        # For k values, see paper p.5 & Gibbs_sampling.py
        # Shapes of ω, λ, μ = len(days)
        omega_t = np.exp(np.dot(XX, self.beta_omega)) #Eq. 60 - k = the constant term multiplied by 1
        lambda_t = np.exp(np.dot(XX, self.beta_lambda)) #Eq. 58 - Φ, Γ = 0
        mu_t = np.exp(np.dot(XX, self.beta_mu)) #Eq. 59 - Φ, Γ = 0
        # Add ARMA term
        z_t, y_t = np.zeros(shape=(T, )), np.zeros(shape=(T, )) #Rows of T zeros
        for ind_t in range(T): # We loop for all days
        # The ARMA terms are needed for day>1
            if ind_t == 0:
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
                # Simulate y_t - Eq. 4, 31, 32
                # Zero can only come if λ is zero
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
                # if lambda_t[ind_t] > 10**9:
                #     lambda_t[ind_t] = 10**9
                # elif np.isnan(lambda_t[ind_t]):
                #     lambda_t[ind_t] = 0
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
                # Calculate mu_t
                num = np.nan_to_num(y_t[ind_t - (min(ind_t, 5)):ind_t] - (z_t[ind_t - (min(ind_t, 5)):ind_t] *
                       mu_t[ind_t - (min(ind_t, 5)):ind_t]))
                deno = np.nan_to_num(mu_t[ind_t - (min(ind_t,5)):ind_t] *
                       np.sqrt(z_t[ind_t - (min(ind_t, 5)):ind_t] * omega_t[ind_t - (min(ind_t, 5)):ind_t]))
                MA_comp = np.nan_to_num(np.sum(self.gamma_mu[-min(ind_t, 5):][z_t[ind_t - (min(ind_t, 5)):ind_t]!=0]
                                 * (num[z_t[ind_t - (min(ind_t, 5)):ind_t]!=0] / deno[z_t[ind_t - (min(ind_t, 5)):ind_t]!=0])))
                mu_t[ind_t] = np.nan_to_num(np.exp(np.log(mu_t[ind_t]) + np.sum(self.phi_mu[-min(ind_t, 5):] *\
                                    (np.log(mu_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_mu[0])) + MA_comp))
                # Simulate y_t
                if z_t[ind_t] == 0:
                    y_t[ind_t] = 0
                else:
                    y_t[ind_t] = np.nan_to_num(np.random.gamma(shape=z_t[ind_t] / omega_t[ind_t], scale = 1 / (omega_t[ind_t] * mu_t[ind_t])))

        return z_t, y_t, lambda_t, omega_t, mu_t

    def _loglikelihood_one(self, z, y, X):

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
                if y[ind_t] > 0 and z[ind_t]>0 and lambda_t[ind_t]>0:
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

    def _compute_lambda_one(self, z, y, X):

        num_model_field, T = X.shape[1], X.shape[0]
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1)
        # Only linear regression term
        lambda_t = np.exp(np.dot(XX, self.beta_lambda))

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
                lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + np.sum(self.phi_lambda[-min(ind_t, 5):] * \
                                                                          (np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t]) - self.beta_lambda[0]))
                                         + MA_comp)
        return lambda_t