import numpy as np
import Gibbs_locations_sampling_function as GLSF

"""
Loading data
"""
Year_training_start = "1998"
Year_training_end = "1999"
#loc = int(sys.argv[1])
#print(loc)
N_gibbs = 10000
EC = True


# Model fields
#X = np.load('../Data/Data/model_fields_multiple_{}_{}_small_allMF.npy'.format(Year_training_start, Year_training_end))[:,:,[0,1,2,3,6,7,8]]

X = np.load('../Data/Data/model_fields_multiple_{}_small_allMF.npy'.format(Year_training_end))[:,:,[0,1,2,3,6,7,8]]


# Rainfall
#Y = np.load('../Data/Data/Rainfalls_{}_{}_small.npy'.format(Year_training_start, Year_training_end))

Y = np.load('../Data/Data/Rainfalls_{}_small.npy'.format(Year_training_end))

# For purely serial computing
#from parallel.backends import BackendDummy as Backend
# For MPI parallelized computing
from parallel.backends import BackendMPI as Backend
backend = Backend()

X_bds = backend.broadcast(X)
Y_bds = backend.broadcast(Y)

def myfunc(ind):
    return GLSF.sampling_function(location=ind, X=X, Y=Y, n_step_Gibbs=N_gibbs,\
                       year_training_start = Year_training_start, year_training_end = Year_training_end,\
                       extreme_case=EC)

seed_arr = [ind for ind in range(X.shape[0])]
seed_pds = backend.parallelize(seed_arr)
accepted_parameters_pds = backend.map(myfunc, seed_pds)
accepted_parameters = backend.collect(accepted_parameters_pds)




