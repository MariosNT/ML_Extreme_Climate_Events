# import sys
# import numpy as np
# import pandas as pd
# import Gibbs_locations_sampling_function as GLSF
import Prediction as pred


###################
### Run Options ###
###################

extreme_case = True
zknown = True
postconvcheck = False
savefig = False


#####################
### Set Variables ###
#####################

location = 8
N_burn = 1000

perc=0.1
z_range=9
grid = 'small'
N_gibbs = 5803

Year_training_start = "1998"
Year_training_end = "2000"

Year_prediction_start = "2000"
Year_prediction_end = "2001"

model_fields = "Sherman"


"""
Run code
"""

X, Y, Theta, Z_list, Lhd_list, x_size, n_days, n_param, imlocation, filename_raw =\
pred.load_data(Year_training_start, Year_training_end, Year_prediction_start, Year_prediction_end,\
               location, perc, z_range, grid)
    
#pred.postconvcheck(n_param, 1000, Theta, Lhd_list, Year_training_start, Year_training_end, imlocation)
    
Y_samples = pred.model_prediction(Theta, Z_list, X, x_size, N_burn, imlocation, filename_raw)


pred.predictions_plot(Y_samples, Year_training_start, Year_training_end, n_days, X, Y, imlocation, filename_raw)