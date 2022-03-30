import numpy as np
import copy
import os.path
import pylab as plt
from Sampler import EllipticalSliceSampling
from scipy.stats import gamma, multivariate_normal, poisson
# For purely serial computing
#from parallel.backends import BackendDummy as Backend
# For MPI parallelized computing
from parallel.backends import BackendMPI as Backend
backend = Backend()
# number of steps Gibbs we want to use
n_step_Gibbs = 100
extreme_case = True

if extreme_case:
    from timeseries_cp_extreme import cptimeseries_extreme as model
else:
    from timeseries_cp import cptimeseries as model

year = '1999'
if extreme_case:
    filename = 'PostSample_'+year + '_cp_extreme.npz'
else:
    filename = 'PostSample_'+year + '_cp.npz'

# Read Model fields
X = np.load('../Data/Data/model_fields_multiple_'+year+'.npy')[:3,:,:]
x_size = X.shape[-1]+1
diff = x_size-6
# Read Rain fall
Y = np.load('../Data/Data/Rainfalls_'+year+'.npy')[:3,:]
# Broadcast Model Fields and Rainfall to workers
X_bds = backend.broadcast(X)
y_bds = backend.broadcast(Y)

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
######################################## Initialization ########################################
if os.path.isfile(filename):
    Theta = list(np.load(filename)['Theta'])
    Z_list = list(np.load(filename)['Z'])
    lhd_list = list(np.load(filename)['lhd_list'])
    Initial_steps = len(Theta)
    Non_Zero_indices = []
    for ind in range(Y.shape[0]):
        y_tmp = Y[ind,:]
        # Extract zero/non-zero indices of y
        en = np.arange(len(y_tmp))
        bool_y_zero = (y_tmp==0)
        zero_y_indices = en[bool_y_zero]
        nonzero_y_indices = en[np.invert(bool_y_zero)]
        # store the nonzero indces
        Non_Zero_indices.append(nonzero_y_indices)
else:
    Initial_steps = 0
    ### Lists to store the samples
    Theta, Z_list, lhd_list = [], [], []
    ### Initial value of theta
    theta_state = true_theta_hp
    ### Initial value of Z
    Z_state = np.zeros(shape=Y.shape)
    Non_Zero_indices = []
    for ind in range(Z_state.shape[0]):
        y_tmp = Y[ind,:]
        # Extract zero/non-zero indices of y
        en = np.arange(len(y_tmp))
        bool_y_zero = (y_tmp==0)
        zero_y_indices = en[bool_y_zero]
        nonzero_y_indices = en[np.invert(bool_y_zero)]
        # store the nonzero indces
        Non_Zero_indices.append(nonzero_y_indices)
        ## Lets first initialize theta and z for a Markov chain ##
        #### For non-zero y, get distribution of rainfalls and calculate quantiles
        #### Then use this to initialise z (1, 2, 3, 4), based on the quantiles
        y_non_zero = y_tmp[y_tmp>0]
        edge1 = np.quantile(y_non_zero, 0.25)
        edge2 = np.quantile(y_non_zero, 0.5)
        edge3 = np.quantile(y_non_zero, 0.75)
        edge4 = np.max(y_tmp)

        bin_2 = (edge1<=y_tmp) & (y_tmp<=edge2)
        bin_3 = (edge2<y_tmp) & (y_tmp<=edge3)
        bin_4 = (edge3<y_tmp) & (y_tmp<=edge4)

        z_state = np.ones(shape=y_tmp.shape)
        z_state[bin_2] = 2
        z_state[bin_3] = 3
        z_state[bin_4] = 4
        z_state[zero_y_indices] = 0 #z_state an array of 0, 1
        # Store the zstate
        Z_state[ind,:] = z_state
    # Add to stored samples
    Theta.append(copy.deepcopy(theta_state))
    Z_list.append(copy.deepcopy(Z_state))

################################################################################
######################################## Sampling ########################################
for ind_Gibbs in range(n_step_Gibbs):
    theta_state = copy.deepcopy(Theta[-1])
    Z_state = copy.deepcopy(Z_list[-1])
    #broadcast Z_state
    z_bds = backend.broadcast(Z_state)

    #define parallelized conditional LHD of theta fixing z
    def LHD_parallelized_over_locations(theta, backend):
        def myfunc(ind):
            return model(theta, k=x_size)._loglikelihood_one(z_bds.value()[ind,:], y_bds.value()[ind, :], np.squeeze(X_bds.value()[ind, :, :]))
        seed_arr = [ind for ind in range(X.shape[0])]
        seed_pds = backend.parallelize(seed_arr)
        accepted_parameters_pds = backend.map(myfunc, seed_pds)
        accepted_parameters = backend.collect(accepted_parameters_pds)
        accepted_parameters = np.array(accepted_parameters)
        return np.sum(accepted_parameters)

    ### Elliptical slice sampling for n steps to update theta
    ### Here Mean and Sigma are the mean and var-cov matrix of Multivariate normal used as the prior.
    ### Store initial value
    f_prime = theta_state
    n_elliptcal_sample_step = 1
    for ind_elliptical_sample in range(n_elliptcal_sample_step):
        f = f_prime
        #sample from prior
        nu = true_theta_hp + multivariate_normal.rvs(cov=Sigma_0_hp, size=1)
        #Compute and Define loglikelihood threshold
        u = np.random.random(size=(1,))
        log_y = LHD_parallelized_over_locations(f, backend) + np.log(u)
        ###### Initial proposal #####
        #Draw an initial proposal, also defining a bracket:
        theta_ellipse = np.random.random(size=(1,)) * (2*np.pi)
        theta_ellipse_min = theta_ellipse - (2*np.pi)
        theta_ellipse_max = theta_ellipse
        # new proposal
        f_prime = f * np.cos(theta_ellipse) + nu * np.sin(theta_ellipse)
        lhd_f_prime = LHD_parallelized_over_locations(f_prime, backend)
        ##### While loop until gets accepted ####
        # Accept and Reject Step
        while lhd_f_prime <= log_y:
            if theta_ellipse < 0:
                theta_ellipse_min = theta_ellipse
            else:
                theta_ellipse_max = theta_ellipse
            theta_ellipse = theta_ellipse_min + (theta_ellipse_max - theta_ellipse_min) * np.random.random(size=(1,))
            f_prime = f * np.cos(theta_ellipse) + nu * np.sin(theta_ellipse)
            lhd_f_prime = LHD_parallelized_over_locations(f_prime, backend)
        ## Save the accepted parameter value as theta_state
    theta_state = f_prime
    ### Elliptical slice sampling is finished

    # Sample/Update of z
    # Define conditional likelihood for z
    # Define parallelized conditional LHD of z fixing theta
    ind_arr = np.array([ind for ind in range(Z_state.shape[0])])
    ind_pds = backend.parallelize(ind_arr)
    def sample_z(ind):
        loglikelihood_z = lambda z: model(theta_state, k=x_size)._loglikelihood_one(z, y_bds.value()[ind,:], np.squeeze(X_bds.value()[ind,:,:]))
        possible_z = z_bds.value()[ind,:]
        lambda_t = model(theta_state, k=x_size)._compute_lambda_one(possible_z, y_bds.value()[ind,:], np.squeeze(X_bds.value()[ind,:,:]))
        nonzero_y = np.random.choice(Non_Zero_indices[ind], size=1)
        lambda_t_nonzero_y = lambda_t[nonzero_y]
        prob_z = np.zeros(6)
        finite_indices = [True for ind in range(6)]
        for ind_opt in range(6):
            possible_z[nonzero_y] = ind_opt + 1
            prob_z[ind_opt] = loglikelihood_z(possible_z) + poisson.logpmf(ind_opt + 1, lambda_t_nonzero_y) # Add poisson hierarchical prior
            # The following threshold is arbitrarily chosen to asses whether it has diveregd or not
            if prob_z[ind_opt] < -1e+4:
                finite_indices[ind_opt] = False
        prob_z = np.exp(prob_z[finite_indices] - np.min(prob_z[finite_indices]))
        prob_z = prob_z / np.sum(prob_z)
        if sum(np.isnan(prob_z)) == 0 or np.size(prob_z)==0:
            possible_z[nonzero_y] = np.random.choice(a=np.arange(1, 7)[finite_indices], p=prob_z)
            return possible_z
        else:
            return z_bds.value()[ind,:]
    sample_z_pds = backend.map(sample_z, ind_pds)
    accepted_zs = np.array(backend.collect(sample_z_pds))
    ## Sampling of z is finished

    print(str(ind_Gibbs+Initial_steps)+'-st/th iteration successfully finished' )
    # Add to stored samples
    Theta.append(copy.deepcopy(theta_state))
    Z_list.append(copy.deepcopy(accepted_zs))
    lhd_list.append(lhd_f_prime)
    print(str(ind_Gibbs+Initial_steps)+'-st/th sample LogLikeliHood: '+str(lhd_f_prime))

    if np.mod(ind_Gibbs, 100) == 0:
        ## Save the posterior samples
        np.savez(filename, Z=Z_list, Theta=Theta, lhd_list=lhd_list)

plt.figure()
plt.plot(lhd_list)
plt.show()
plt.close()