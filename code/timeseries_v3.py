# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:32:56 2022

@author: klera
"""

import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal
import pylab as plt
from Sampler import EllipticalSliceSampling
import sys
#import joblib
from joblib import Parallel, delayed


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

