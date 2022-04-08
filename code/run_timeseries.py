import numpy as np
import Gibbs_locations_sampling_function as GLSF

"""
Loading data
"""
years_training = "1999"
# Model fields
X = np.load('../Data/Data/model_fields_multiple_'+years_training+'.npy')
# Rainfall
Y = np.load('../Data/Data/Rainfalls_'+years_training+'.npy')


# GLSF.sampling_function(location=2, X=X, Y=Y, n_step_Gibbs=5, perc=0.05, z_range=9,\
#                       Parallel_case = False, extreme_case = True)

# For purely serial computing
#from parallel.backends import BackendDummy as Backend
# For MPI parallelized computing
from parallel.backends import BackendMPI as Backend
backend = Backend()

X_bds = backend.broadcast(X)
Y_bds = backend.broadcast(Y)

def myfunc(ind):
    return GLSF.sampling_function(location=ind, X=X_bds.value(), Y=Y_bds.value(), n_step_Gibbs=2, perc=0.01, z_range=9,\
                      Parallel_case = False, extreme_case = True)


seed_arr = [ind for ind in range(X.shape[0])]
seed_pds = backend.parallelize(seed_arr)
accepted_parameters_pds = backend.map(myfunc, seed_pds)
accepted_parameters = backend.collect(accepted_parameters_pds)