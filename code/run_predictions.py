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

location = 4
N_burn = 1000

perc = 0.1
z_range = 9
grid = 'small'

Year_training_start = "1998"
Year_training_end = "1999"

Year_prediction_start = "1999"
Year_prediction_end = "2000"

model_fields = "Sherman"


"""
Run code
"""

X, Y, Theta, Z_list, Lhd_list, x_size, n_days, n_param, imlocation, filename_raw =\
pred.load_data(Year_training_start, Year_training_end, Year_prediction_start, Year_prediction_end,\
               location, perc, z_range, grid)
    
#pred.postconvcheck(n_param, N_burn, Theta, Lhd_list, Year_training_start, Year_training_end, imlocation)
    
Y_samples = pred.model_prediction(Theta, Z_list, X, x_size, N_burn, imlocation, filename_raw)


pred.predictions_plot(Y_samples, Year_prediction_start, Year_prediction_end, n_days, X, Y, imlocation, filename_raw)
pred.rain_probability(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation)
#pred.scatter_plot_fit(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation)
pred.ROC_plottting(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation)
pred.calibration_error(Y_samples, Y)