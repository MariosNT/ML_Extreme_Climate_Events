# import sys
# import numpy as np
# import pandas as pd
# import Gibbs_locations_sampling_function as GLSF
import predictions_samples as psamples
import predictions_visualisation as pvis


###################
### Run Options ###
###################

# extreme_case = True
# z_known = False
# postconvcheck = False
# savefig = False


creating_samples = False
visualising_samples = True

#####################
### Set Variables ###
#####################

Year_prediction_start = 1999
Year_prediction_end = 2000



"""
Run code
"""
if creating_samples:
    psamples.predicting_samples(1999, 1999, Year_prediction_start, Year_prediction_end, N_samples=1000, save_file=True,\
                                z_known=False)


if visualising_samples:
    pvis.prediction_visualisation(Year_prediction_start, Year_prediction_end, N_locations=4)

