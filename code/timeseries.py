import numpy as np
from scipy.stats import gamma

""" Questions """

# 1 - Why add an 1 in the model fields? For the first step we don't use the ARMA model? We assume β=1?
# 2 - Do we fix r to be 5 in eq. (35) here?
# 3 - Is line 93 correct? Shouldn't we return log Poisson?

X = np.random.normal(size=(100, 2), loc=1, scale=1) # Returns a 100x2 matrix of random (normal) elements

class cptimeseries():
    def __init__(self):
        self.T = 100

    def simulate(self, X, beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu):

        num_model_field, T = X.shape[1], X.shape[0]
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1) #Adds a column of 1s at the beginning of X
        # Only linear regression term
        omega_t = np.exp(np.dot(XX, beta_omega)) #Eq. 60 - k=0
        lambda_t = np.exp(np.dot(XX, beta_lambda)) #Eq. 58 - Φ, Γ, k=0
        mu_t = np.exp(np.dot(XX, beta_mu))  #Eq. 59 - Φ, Γ, k=0
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
                lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + np.sum(phi_lambda[-min(ind_t, 5):] *\
                                        (np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t]) - beta_lambda[0])) \
                                         + np.sum(gamma_lambda[-min(ind_t, 5):] *\
                                        ((z_t[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]) /\
                                         np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t]))))
                # Simulate z_t
                z_t[ind_t] = np.random.poisson(lambda_t[ind_t])
                # Calculate mu_t
                mu_t[ind_t] = np.exp(np.log(mu_t[ind_t]) + np.sum(phi_mu[-min(ind_t, 5):] *\
                                    (np.log(mu_t[ind_t - (min(ind_t, 5)):ind_t]) - beta_mu[0])) \
                                     + np.sum(gamma_mu[-min(ind_t, 5):] * \
                                     (np.nan_to_num((y_t[ind_t - (min(ind_t, 5)):ind_t] - z_t[ind_t - (min(ind_t, 5)):ind_t] *\
                                     mu_t[ind_t - (min(ind_t, 5)):ind_t]) / (mu_t[ind_t - (min(ind_t, 5)):ind_t] *\
                                     np.sqrt(z_t[ind_t - (min(ind_t, 5)):ind_t] * omega_t[ind_t - (min(ind_t, 5)):ind_t]))))))
                # Simulate y_t
                if z_t[ind_t] == 0:
                    y_t[ind_t] = 0
                else:
                    y_t[ind_t] = np.random.gamma(shape= z_t[ind_t] / omega_t[ind_t], scale = 1 / (omega_t[ind_t] * mu_t[ind_t]))

        return z_t, y_t

    def loglikelihood(self, beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu, z, y, X):

        num_model_field, T = X.shape[1], X.shape[0]
        XX = np.concatenate((np.ones(shape=(T, 1)), X), axis=1)
        ## We check whether there are some days when z and y are not both 0, if so then llhd is -np.inf, ow we calculate llhd
        check = True
        for ind in range(T):
            if y[ind] == 0 and z[ind] !=0 or y[ind] != 0 and z[ind] ==0:
                check = False
        if check == True:
            # Only linear regression term
            omega_t = np.exp(np.dot(XX, beta_omega))
            lambda_t = np.exp(np.dot(XX, beta_lambda))
            mu_t = np.exp(np.dot(XX, beta_mu))
            # Add ARMA term
            for ind_t in range(1, 5):
                lambda_t[ind_t] = np.exp(np.log(lambda_t[ind_t]) + np.sum(phi_lambda[-min(ind_t, 5):] *\
                                  (np.log(lambda_t[ind_t - (min(ind_t, 5)):ind_t]) - beta_lambda[0]))\
                                  + np.sum(gamma_lambda[-min(ind_t, 5):] *\
                                  ((z[ind_t - (min(ind_t, 5)):ind_t] - lambda_t[ind_t - (min(ind_t, 5)):ind_t]) /\
                                   np.sqrt(lambda_t[ind_t - (min(ind_t, 5)):ind_t]))))
                
                mu_t[ind_t] = np.exp(np.log(mu_t[ind_t]) + np.sum(phi_mu[-min(ind_t, 5):] *\
                                    (np.log(mu_t[ind_t - (min(ind_t, 5)):ind_t]) - beta_mu[0])) \
                                     + np.sum(gamma_mu[-min(ind_t, 5):] * \
                                    (np.nan_to_num((y[ind_t - (min(ind_t, 5)):ind_t] - z[ind_t - (min(ind_t, 5)):ind_t]*\
                                     mu_t[ind_t - (min(ind_t,5)):ind_t]) / (mu_t[ind_t - (min(ind_t,5)):ind_t] *\
                                     np.sqrt(z[ind_t - (min(ind_t, 5)):ind_t] * omega_t[ind_t - (min(ind_t, 5)):ind_t] ))))))

            alpha_t, beta_t = z / omega_t, 1 / (omega_t * mu_t)
            llhd = 0
            for ind_t in range(T):
                if y[ind_t] > 0 and z[ind_t]>0:
                    llhd += gamma.logpdf(y[ind_t], a = alpha_t[ind_t], scale = beta_t[ind_t])
                elif y[ind_t] == 0 and z[ind_t] == 0:
                    llhd += - lambda_t[ind_t] # Is this correct? I thought a Poisson should be returned?
            return llhd
        else:
            return -np.inf


beta_lambda, beta_mu, beta_omega = np.random.normal(size=(3,), loc=[-0.46, 0, 0], scale=1/6), \
                                   np.random.normal(size=(3,), loc=[1.44, 0, 0], scale=1/6), \
                                   np.random.normal(size=(3,), loc=[-0.45, 0, 0], scale=1/6)
phi_lambda, phi_mu, gamma_lambda, gamma_mu = np.random.normal(size=(5,1), scale=1/(1.3*65)), np.random.normal(size=(5,1), scale=1/(1.3*65)),\
                                             np.random.normal(size=(5,1), scale=1/(1.3*65)), np.random.normal(size=(5,1), scale=1/(1.3*65))

#print(beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu)
# beta_lambda, beta_mu, beta_omega = [-0.46, 0, 0], [1.44, 0, 0], [-0.45, 0, 0]
# phi_lambda, phi_mu, gamma_lambda, gamma_mu =  np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5)

cp = cptimeseries()
z, y = cp.simulate(X, beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu)
print(cp.loglikelihood(beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu, z, y, X))

