import numpy as np
import Prediction as pred


###################
### Run Options ###
###################

extreme_case = True
z_known = True
postconvcheck = False
savefig = False


#####################
### Set Variables ###
#####################

N_burn = 5000

perc = 0.1
z_range = 9
grid = 'small'

Year_training_start = "1999"
Year_training_end = "1999"

Year_prediction_start = "1999"
Year_prediction_end = "2000"

model_fields = "Sherman"

N_locations = 8
Loc_list = np.arange(N_locations)


Pred_samples = len(Theta) - N_burn

N_days = 2

Prediction_array = np.zeros((N_locations, N_days, Pred_samples))

for loc in Loc_list:
    X, Y, Theta, Z_list, Lhd_list, x_size, n_days, n_param, imlocation, filename_raw =\
    pred.load_data(Year_training_start, Year_training_end, Year_prediction_start, Year_prediction_end,\
                   loc, perc, z_range, grid)
        
        
    Y_samples = pred.model_prediction(Theta, Z_list, X, x_size, N_burn, imlocation, filename_raw, zknown=z_known)
