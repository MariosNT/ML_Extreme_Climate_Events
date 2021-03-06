"""
Importing packages
"""
import copy, os, time
import numpy as np
from Sampler import EllipticalSliceSampling
from joblib import Parallel, delayed
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parallel_indices(ind_non, ind_z, possible_z, loglikelihood_z, prob_z):
    possible_z[ind_non] = ind_z + 1
    prob_z[ind_z] = loglikelihood_z(possible_z)
    return prob_z

def sampling_function(location, X, Y, n_step_Gibbs = 2, perc = 0.1, z_range=9,\
                      year_training_start = "1998", year_training_end = "1999",\
                      Parallel_case = False, extreme_case = False):
        
    print()
    print("Gibbs sampling for location: "+str(location))
    print("Number of Gibbs Steps: ", n_step_Gibbs)
    print("Sampling percentage: ", 100*perc, "%")
    print("Max number of rainfalls per day: ", z_range)
    print("Start year is: ", year_training_start,"- End year is: ", str(int(year_training_end)-1),\
          "- Number of days: {}".format(X.shape[1]))
    print()

    
    if extreme_case:
        from timeseries_cp_extreme import cptimeseries_extreme as model
        print("We are running the extreme timeseries model")
        print()
    else:
        from timeseries_cp import cptimeseries as model
        print("We are running the standard timeseries model")
        print()

    if extreme_case:
        filename = 'Posteriors/PostSample_' + year_training_start + '_' + str(int(year_training_end)-1) +\
                   '_cp_extreme_' + str(location) + '_sr' + str(int(perc*100)) +\
                   '_maxZ' + str(z_range) + '_gs'
    else:
        filename = 'Posteriors/PostSample_' + year_training_start + '_' + str(int(year_training_end)-1) +\
                   '_cp_' + str(location) + '_sr' + str(int(perc*100)) +\
                   '_maxZ' + str(z_range) + '_gs'

    x_shape = X.shape
    y_shape = Y.shape

    X = X[location, :, :].reshape(1,x_shape[1],x_shape[2])
    Y = Y[location, :].reshape(1,y_shape[1])

    x_size = X.shape[-1] + 1
    diff = x_size - 6
    
    print("Shape of MFs: ", X.shape[-1])
    print("Size of X is: ", X.shape)
    print("Size of Y is: ", Y.shape)
    print()


    ##### Defining the priors from Sherman's paper .... without prior on sigmas, so just taking mean for them
    # Define prior hyperparmeters
    if extreme_case:
        true_theta_hp = np.concatenate(([-0.46, 0, 0, 0, 0, 0], np.zeros(diff), [1.44, 0, 0, 0, 0, 0], np.zeros(diff), \
                                        [-0.45, 0, 0, 0, 0, 0], np.zeros(diff), np.zeros(shape=(32 + diff * 3,))))
        Sigma_0_hp = np.diag(np.concatenate(
            ((1 / 6) * np.ones(shape=(30 + diff * 3,)), (1 / (1.3 * 65)) * np.ones(shape=(20 + diff * 3,)))))
    else:
        ## Realistic priors ##
        # Sampling from prior to define a true_theta
        loc1 = np.concatenate(([-0.46, 0, 0, 0, 0, 0, ], np.zeros(diff)))
        loc2 = np.concatenate(([1.44, 0, 0, 0, 0, 0, ], np.zeros(diff)))
        loc3 = np.concatenate(([-0.45, 0, 0, 0, 0, 0, ], np.zeros(diff)))

        beta_lambda, beta_mu, beta_omega = np.random.normal(size=(x_size,), loc=loc1, scale=1 / 6), \
                                           np.random.normal(size=(x_size,), loc=loc2, scale=1 / 6), \
                                           np.random.normal(size=(x_size,), loc=loc3, scale=1 / 6)
        phi_lambda, phi_mu, gamma_lambda, gamma_mu = np.random.normal(size=(5,), scale=1 / (1.3 * 65)), \
                                                     np.random.normal(size=(5,), scale=1 / (1.3 * 65)), \
                                                     np.random.normal(size=(5,), scale=1 / (1.3 * 65)), \
                                                     np.random.normal(size=(5,), scale=1 / (1.3 * 65))
        true_theta_hp = np.array([])
        for array in [beta_lambda, beta_mu, beta_omega, phi_lambda, phi_mu, gamma_lambda, gamma_mu]:
            true_theta_hp = np.concatenate([true_theta_hp, array])

        Sigma_0_hp = np.diag(
            np.concatenate(((1 / 6) * np.ones(shape=(18 + diff * 3,)), (1 / (1.3 * 65)) * np.ones(shape=(20,)))))

    #### Now we want to implment a Gibbs sample where we update theta and z one after another
    n_sample_z = 5
    
    ######################################## Initialization ########################################
    if os.path.isfile(filename):
        Theta = list(np.load(filename)['Theta'])
        Z_list = list(np.load(filename)['Z'])
        lhd_list = list(np.load(filename)['lhd_list'])
        Initial_steps = len(Theta)
        y_tmp = Y[0, :]
        # Extract zero/non-zero indices of y
        en = np.arange(len(y_tmp))
        bool_y_zero = (y_tmp == 0)
        zero_y_indices = en[bool_y_zero]
        nonzero_y_indices = en[np.invert(bool_y_zero)]
        # store the nonzero indces
        n_sample_z = max(n_sample_z, len(nonzero_y_indices))
    else:
        Initial_steps = 0
        ### Lists to store the samples
        Theta, Z_list, lhd_list = [], [], []
        ### Initial value of theta
        theta_state = true_theta_hp
        ### Initial value of Z
        y_tmp = Y[0, :]
        # Extract zero/non-zero indices of y
        en = np.arange(len(y_tmp))
        bool_y_zero = (y_tmp == 0)
        zero_y_indices = en[bool_y_zero]
        nonzero_y_indices = en[np.invert(bool_y_zero)]
        n_sample_z = max(n_sample_z, len(nonzero_y_indices))
        
        ## Lets first initialize theta and z for a Markov chain ##
        #### For non-zero y, get distribution of rainfalls and calculate quantiles
        #### Then use this to initialise z (1, 2, 3, 4), based on the quantiles
        y_non_zero = y_tmp[y_tmp > 0]
        edge1 = np.quantile(y_non_zero, 0.25)
        edge2 = np.quantile(y_non_zero, 0.5)
        edge3 = np.quantile(y_non_zero, 0.75)
        edge4 = np.max(y_tmp)

        bin_2 = (edge1 <= y_tmp) & (y_tmp <= edge2)
        bin_3 = (edge2 < y_tmp) & (y_tmp <= edge3)
        bin_4 = (edge3 < y_tmp) & (y_tmp <= edge4)

        z_state = np.ones(shape=y_tmp.shape)
        z_state[bin_2] = 2
        z_state[bin_3] = 3
        z_state[bin_4] = 4
        z_state[zero_y_indices] = 0  # z_state an array of 0, 1
        z_state = z_state.reshape(1,-1)
        
        # Add to stored samples
        Theta.append(copy.deepcopy(theta_state))
        Z_list.append(copy.deepcopy(z_state))

    # How many nonzero z s would be sampled at each stage by using perc
    n_sample_z = int(n_sample_z * perc)

    ################################################################################
    #### Now we want to implment a Gibbs sample where we update theta and z one after another
    
    start = timer()
    
    ## Start sampling ##
    for ind_Gibbs in range(n_step_Gibbs):
        start_time = time.time()
        theta_state = copy.deepcopy(Theta[-1])
        z_state = copy.deepcopy(Z_list[-1])

        #### First sample theta using Elliptic Slice Sampler ###
        
        # define conditional likelihood for theta
        loglikelihood_theta = lambda theta: model(theta, k=x_size).loglikelihood(z_state, Y, X)
        # Sample/Update theta
        ## Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
        ## f_0 defines the present state of the Markov chain
        Samples, lhd_f_prime = EllipticalSliceSampling(LHD=loglikelihood_theta, n=1, Mean=true_theta_hp, Sigma=Sigma_0_hp,
                                          f_0=theta_state)
        theta_state = Samples[-1]
        #print('Updated theta')
        
        # define conditional likelihood for z
        loglikelihood_z = lambda z: model(theta_state, k=x_size).loglikelihood(z, Y, X)

        # Sample/Update z
        possible_z = z_state
        originial_z = copy.deepcopy(z_state)

        try:
            nonzero_y = np.random.choice(nonzero_y_indices, size=n_sample_z)
            for ind_nonzero in nonzero_y:
                prob_z = np.zeros(z_range)
                if Parallel_case:
                    prob_z = Parallel(n_jobs=6, prefer="threads")(delayed(parallel_indices)(ind_nonzero, ind_z, possible_z, loglikelihood_z, prob_z)\
                           for ind_z in range(z_range))
                    prob_z = np.sum(prob_z, axis=0)
                else:
                    for ind_z in range(z_range):
                        possible_z[0, ind_nonzero] = ind_z + 1
                        prob_z[ind_z] = loglikelihood_z(possible_z)
                finite_indices = np.isfinite(prob_z)
                prob_z = np.exp(prob_z[finite_indices] - np.min(prob_z[finite_indices]))
                possible_z[0, ind_nonzero] = np.random.choice(a=np.arange(1, z_range+1)[finite_indices],
                                                           p=prob_z / np.sum(prob_z))
            z_state = possible_z
            #print('Updated z')
        except (RuntimeError, ValueError, TypeError, NameError, ZeroDivisionError, OSError):
            print('error so not updating')
            z_state = originial_z
            pass

        # Add to stored samples
        Theta.append(copy.deepcopy(theta_state))
        Z_list.append(copy.deepcopy(z_state))
        lhd_list.append(lhd_f_prime)
        if (ind_Gibbs+1)%1==0:
            print(str(ind_Gibbs+Initial_steps)+'-st/th iteration successfully finished')
            print(str(ind_Gibbs+Initial_steps)+'-st/th iteration took: ' + str(time.time()-start_time) + ' sec')
            print(str(ind_Gibbs + Initial_steps) + '-st/th sample LogLikeliHood: ' + str(lhd_f_prime))
            print()

        if np.mod(ind_Gibbs+1, 20) == 0:
            ### Save the posterior samples
            print("Another 20 steps")
            print(filename+".npz")
            np.savez(filename+".npz", Z=Z_list, Theta=Theta, lhd_list=lhd_list)

    #np.savez(filename+str(ind_Gibbs+1)+".npz", Z=Z_list, Theta=Theta, lhd_list=lhd_list)
    end = timer()
    print("Total time (sec):", end-start, " and total time (hours):", (end-start)/3600)

    return 1
