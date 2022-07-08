import sys
import numpy as np
import pandas as pd
import Gibbs_locations_sampling_function as GLSF


"""
Loading data
"""

automatic_locations = False

if automatic_locations:
    loc = int(sys.argv[1])
    print(loc)
else:
    loc=4



Year_training_start = "1998"
Year_training_end = "2000"
sampling_rate = 0.1
grid = 'small'
N_gibbs = 30
EC = True

model_fields = "Sherman"


# Model fields selection
X = np.load('../Data/Data/model_fields_multiple_1995_2018_{}_allMF.npy'.format(grid))

if model_fields == "Sherman":
    X = X[:,:,[0,1,2,3,6,7,8]]
elif model_fields == "Advection":
    X = X[:,:,[0,1,2,3,4,5,8]]
else:
    X=X
    
# Rainfall
Y = np.load('../Data/Data/Rainfalls_1995_2018_{}.npy'.format(grid))

# Time range selection
year_start = 1995
year_end = 2018
time = pd.date_range('{}-01-01'.format(year_start), '{}-12-31'.format(year_end), periods=X.shape[1])

boolean_time = (time>=Year_training_start) & (time<Year_training_end)

#print(time[boolean_time])

X = X[:, boolean_time, :]
Y = Y[:, boolean_time]

"""
Run code
"""

GLSF.sampling_function(location=loc, X=X, Y=Y, n_step_Gibbs=N_gibbs, perc=sampling_rate,\
                        year_training_start = Year_training_start, year_training_end = Year_training_end,\
                        extreme_case=EC)
