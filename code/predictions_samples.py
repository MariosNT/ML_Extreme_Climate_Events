import numpy as np
import Prediction as pred


###################
### Run Options ###
###################

extreme_case = True
z_known = False
postconvcheck = False
savefig = False
save_file = False


#####################
### Set Variables ###
#####################

N_samples = 10

perc = 0.1
z_range = 9
grid = 'small'

Year_training_start = "1999"
Year_training_end = "1999"

Year_prediction_start = "1999"
Year_prediction_end = "2001"

model_fields = "Sherman"

filename_samples = 'PostSamples_' + Year_prediction_start + '_' + Year_prediction_end +\
                   '_cp_locations_sr' + str(int(perc*100)) +\
                   '_maxZ' + str(z_range) + '_grid_' + grid 



N_locations = 3
Loc_list = np.arange(N_locations)


Predictions_array = []
Observations_array = []

for loc in Loc_list:
    X, Y, Theta, Z_list, Lhd_list, x_size, n_days, n_param, imlocation, filename_raw =\
    pred.load_data(Year_training_start, Year_training_end, Year_prediction_start, Year_prediction_end,\
                   loc, perc, z_range, grid)
        
    N_burn = (len(Theta)) - N_samples
        
        
    Y_samples = pred.model_prediction(Theta, Z_list, X, x_size, N_burn, imlocation, filename_raw, zknown=z_known)
    Y_samples = Y_samples[:,0,:].T
    
    Predictions_array.append(Y_samples)
    Observations_array.append(Y[0,:])

Predictions_array = np.array(Predictions_array)
print()
print("Predictions array shape", Predictions_array.shape)
print()
Observations_array = np.array(Observations_array)
    
if save_file:
    np.savez("Posterior_samples/"+filename_samples+".npz", Y_samples=Predictions_array, Y=Observations_array)