import numpy as np
import Prediction as pred


###################
### Run Options ###
###################

extreme_case = True
z_known = False
postconvcheck = False
savefig = False


#####################
### Set Variables ###
#####################

location = 4

perc = 0.1
z_range = 9
grid = 'small'

Year_prediction_start = "1999"
Year_prediction_end = "2000"


#######################################################
# READ the Y_samples (Predictions) & Y (Observations) #
#######################################################


filename_samples = 'PostSamples_' + Year_prediction_start + '_' + Year_prediction_end +\
                   '_cp_locations_sr' + str(int(perc*100)) +\
                   '_maxZ' + str(z_range) + '_grid_' + grid


Y_file = "Posterior_samples/"+filename_samples+".npz"

Y_samples = np.load(Y_file)['Y_samples']
Y = np.load(Y_file)['Y']
n_days = len(Y.T)
                   
#########################
# Make prediction plots #
#########################

pred.predictions_plot(Y_samples[location], Year_prediction_start, Year_prediction_end, n_days,\
                      Y[location], "test", "test")
pred.rain_probability(Y_samples[location], Y[location], Year_prediction_start, Year_prediction_end, "test")