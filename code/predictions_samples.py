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

#N_burn = 5200

perc = 0.1
z_range = 9
grid = 'small'

Year_training_start = "1999"
Year_training_end = "1999"

Year_prediction_start = "1999"
Year_prediction_end = "2000"

model_fields = "Sherman"

N_locations = 9
Loc_list = np.arange(N_locations)


Prediction_array = []

for loc in Loc_list:
    X, Y, Theta, Z_list, Lhd_list, x_size, n_days, n_param, imlocation, filename_raw =\
    pred.load_data(Year_training_start, Year_training_end, Year_prediction_start, Year_prediction_end,\
                   loc, perc, z_range, grid)
        
    N_burn = (len(Theta)) - 100
        
        
    Y_samples = pred.model_prediction(Theta, Z_list, X, x_size, N_burn, imlocation, filename_raw, zknown=z_known)
    Y_samples = Y_samples[:,0,:].T
    
    Prediction_array.append(Y_samples)
    Prediction_array = np.array(Prediction_array)