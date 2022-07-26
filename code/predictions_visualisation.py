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
N_burn = 4000

perc = 0.1
z_range = 9
grid = 'small'

Year_training_start = "1999"
Year_training_end = "1999"

Year_prediction_start = "1999"
Year_prediction_end = "2003"

model_fields = "Sherman"


"""
READ the Y_samples from the other side and create plots
"""

pred.predictions_plot(Y_samples, Year_prediction_start, Year_prediction_end, n_days, X, Y, imlocation, filename_raw)
pred.rain_probability(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation)