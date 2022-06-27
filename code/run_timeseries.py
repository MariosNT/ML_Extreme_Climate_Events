import numpy as np
import Gibbs_locations_sampling_function as GLSF


"""
Loading data
"""
Year_training_start = "1998"
Year_training_end = "1999"
loc = 4
N_gibbs = 3
EC = True


# Model fields
X = np.load('../Data/Data/model_fields_multiple_{}_{}_small_allMF.npy'.format(Year_training_start, Year_training_end))

# Rainfall
Y = np.load('../Data/Data/Rainfalls_{}_{}_small.npy'.format(Year_training_start, Year_training_end))



"""
Run code
"""

GLSF.sampling_function(location=loc, X=X, Y=Y, n_step_Gibbs=N_gibbs,\
                       year_training_start = Year_training_start, year_training_end = Year_training_end,\
                       extreme_case=EC)
