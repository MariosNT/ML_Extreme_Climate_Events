"""
Code to analyse the timeseries predictions
"""

### Importing useful packages

import numpy as np
import matplotlib.pyplot as plt
from timeseries_v3 import cptimeseries


### Importing observed data & model fields

year = 2000 #For now, we're focusing on a single year
location = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\code\\images\\year_'+str(year)+"\\"

Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\Rainfall_Cardiff_{}.npy'.format(year))
X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}.npy'.format(year))


### Importing timeseries of Z and Theta, after sampling
# Z = number of times rain/day
# Theta = parameters of the model (Eq. 10)

data_set = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_Cardiff_{}.npz".format(year))
Z = data_set["Z"]
Theta = data_set["Theta"]
n_param = len(Theta.T)
n_days = len(Z.T)

### Checking convergence
# Gibbs_steps = number of times the sampler was run
sampling_steps = len(Z)
Gibbs_steps = np.arange(sampling_steps)
N_burn = 100

def plot_Gibbs_samples(theta, n_burn=0):
    for i in range(n_param):
        plt.figure(figsize=(10, 8))
        plt.plot(Gibbs_steps[n_burn:], theta[n_burn:,i], linestyle = '-', color = 'b')
        plt.title("Parameter {} for year {}".format(i+1, year))
        plt.show()
        #plt.savefig(location+"parameteres_{}_Gsteps_{}.png".format(i+1, year))
        plt.close()

        
#plot_Gibbs_samples(Theta, N_burn)


### Making predictions
# y_pred = amount of rainfall (mm)
Z = Z[N_burn:,:]
Theta = Theta[N_burn:,:]
sampling_steps = len(Z)
 
# We name new parameter arrays after the burn in
# This will save computation time for predictions
z_pred = np.ones(Z.shape)
y_pred = np.ones(Z.shape)

# For the given model fields and sampled parameters, 
# we make predictions of the amount of rainfall per day.
# We discard nan values & re-predict for these
for i in range(sampling_steps):
    _, y_pred[i] = cptimeseries(Theta[i]).simulate(X)
    while sum(np.isnan(y_pred[i])) != 0.0:
        _, y_pred[i] = cptimeseries(Theta[i]).simulate(X)
        
    
### Summary statistics of predictions

# Gaussian measures
y_mean = np.mean(y_pred, axis=0)
y_std = np.std(y_pred, axis=0)

# Arbitrary (distribution) measures
y_median = np.quantile(y_pred, 0.5, axis=0)
y_68 = np.quantile(y_pred, 0.68, axis=0)
y_95 = np.quantile(y_pred, 0.95, axis=0)
y_max = np.max(y_pred, axis=0)

# Our distribution is not a gaussian, so we 
# use quantiles for our predictions and their errors

# time = number of days that we predict rainfall for
time = np.arange(n_days)

# Plot that uses the mean & std, as predicted value and error
plt.figure(figsize=(10, 8))
plt.plot(time, y_mean, linestyle = '-', color = 'b')
plt.plot(time, Y, marker='+', linestyle='', color = 'black')
plt.fill_between(time, y_mean-y_std, y_mean+y_std)
plt.ylim(0, np.max(Y)+1)
plt.title("Y mean - Year {}".format(year))
plt.xlabel("Days")
plt.ylabel("precipitation (mm)")
plt.show()
#plt.savefig(location+"precipitation_mean_{}.png".format(year))
plt.close()    


# Plot that uses the median & different quantiles, as predicted value and errors
plt.figure(figsize=(10, 8))
plt.plot(time, y_median, linestyle = '-', color = 'b')
plt.plot(time, Y, marker='+', linestyle='', color = 'black')
plt.fill_between(time, y_median, y_68, color='red')
plt.fill_between(time, y_median, y_95, color='red', alpha=0.5)
plt.fill_between(time, y_median, y_max, color='red', alpha=0.2)
plt.ylim(0, np.max(Y)+1)
plt.title("Y median - Year {}".format(year))
plt.xlabel("Days")
plt.ylabel("precipitation (mm)")
plt.show()
#plt.savefig(location+"precipitation_median_{}.png".format(year))
plt.close()    



##### Calculate RMS error & RMS spread

def rms_error_spread(rain_obs, rain_pred, rain_samples):
    rms_error = np.sqrt((rain_obs-rain_pred)**2)

    forecast_subs = np.sum((rain_samples-rain_pred)**2, axis=0)
    rms_spread = np.sqrt(forecast_subs/len(rain_samples))
    
    return rms_error, rms_spread

rms_error, rms_spread = rms_error_spread(Y, y_median, y_pred)
#sorted_rms_error, sorted_rms_spread = np.sort(rms_error), np.sort(rms_spread)

plt.figure(figsize=(10, 8))
plt.plot(rms_spread, rms_error, 'o-', color = 'black', alpha = 0.5)
plt.plot(rms_spread, rms_spread, linestyle = '--', color = 'black')
plt.xlabel("RMS spread")
plt.ylabel("RMS error")
plt.show()
#plt.savefig(location+"spread-skill_{}.png".format(year))
plt.close()    

"With bins"
#So far binning the final results - would it be better to been earlier?

n_bins = 16
rms_spread_binned = np.array(np.array_split(rms_spread, n_bins), dtype=object)
rms_error_binned = np.array(np.array_split(rms_error, n_bins), dtype=object)

v_mean = np.vectorize(np.mean)

rms_spread_mean = v_mean(rms_spread_binned)
rms_error_mean = v_mean(rms_error_binned)

plt.figure(figsize=(10, 8))
plt.plot(rms_spread_mean, rms_error_mean, 'o-', color = 'black')
plt.plot(np.sort(rms_spread_mean), np.sort(rms_spread_mean), linestyle = '--', color = 'grey')
plt.xlabel("RMS spread binned")
plt.ylabel("RMS error binned")
plt.show()

###### Calculate RMSB error & MAB error
# Here we only have 1 location, so S=1
# We have values for 1 year, so T=365

# T = len(Y)

# time_subs = np.sum((Y-y_median)**2)
# rmsb_error = np.sqrt(time_subs/T)

# abs_subs = np.sum(np.abs(Y-y_median))
# mab_error = abs_subs/T

# print("RMSB error is: ", rmsb_error)
# print("MAB error is: ", mab_error)


"""
### Calculate ROC curve
# Questions:
# 1 - Which values to use as predictions? 

def true_false_positives(rain_thres, rain_obs, rain_pred):    
    bool_obs = rain_obs > rain_thres
    bool_pred = rain_pred > rain_thres
    
    TP = np.sum(bool_obs & bool_pred)
    FP = np.sum(np.invert(bool_obs) & bool_pred)
    TN = np.sum(np.invert(bool_obs) & np.invert(bool_pred))
    FN = np.sum(bool_obs & np.invert(bool_pred))
    
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    
    return TPR, FPR

def ROC_plot(rain_thres, rain_obs, rain_pred):
    true_positives = []
    false_positives = []

    for i in rain_thres:
        tpr, fpr = true_false_positives(i, rain_obs, rain_pred)
        true_positives.append(tpr)
        false_positives.append(fpr)

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    AUC = np.round(np.abs(np.trapz(true_positives, false_positives)), 3)    

    plt.figure(figsize=(10, 8))
    plt.plot(false_positives, true_positives, linestyle = '--', color = 'black')
    plt.plot(false_positives, false_positives, linestyle = '-', color = 'black')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("AUC = {}".format(AUC))
    plt.show()
    #plt.savefig(location+"ROC_{}.png".format(year))
    plt.close()    
    
    return true_positives, false_positives
    
thresholds = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20])

Tpr, Fpr = ROC_plot(thresholds, Y, y_95)
"""


### Calculate probability of precipitation
# How many days we predict rainfall above a threshold
# Compare with observed values

# Question:
# Which value we use for predictions?

def precipitation_above_x(rain_thres, rainvalues):
    if np.shape(rainvalues)[0]==len(Y):
        all_days = len(rainvalues)
        length_above_x = np.sum(rainvalues>rain_thres)
    
        return length_above_x/all_days
    
    else: 
        rainfall_all = rainvalues.flatten()
        all_days = len(rainfall_all)
        length_above_x = np.sum(rainfall_all>rain_thres)
        
        return length_above_x/all_days

rain_thresholds = np.arange(0, 30, 1)
rain_probability_obs = []
rain_probability_pred = []
rain_probability_pred_95 = []
rain_probability_samples = []

for rain in rain_thresholds:
    rain_probability_obs.append(precipitation_above_x(rain, Y))
    rain_probability_pred.append(precipitation_above_x(rain, y_median))
    rain_probability_pred_95.append(precipitation_above_x(rain, y_95))
    rain_probability_samples.append(precipitation_above_x(rain, y_pred))

    
plt.figure(figsize=(10, 8))
plt.plot(rain_thresholds, rain_probability_obs, linestyle = '--', color = 'black', label = "Obs.")
plt.plot(rain_thresholds, rain_probability_pred, linestyle = '-', color = 'black', label = "Pred.")
plt.plot(rain_thresholds, rain_probability_pred_95, linestyle = '-.', color = 'black', label = "Pred. 95")
plt.plot(rain_thresholds, rain_probability_samples, linestyle = ':', color = 'black', label = "Pred. Samples")
plt.xlabel("Rain thresholds [x (mm)]")
plt.ylabel("Probability [rain>x]")
plt.legend()
plt.show()
#plt.savefig(location+"precipitation_prob_{}.png".format(year))
plt.close()    

### Plotting residuals
# plt.figure(figsize=(10, 8))
# plt.plot(time, y_mean-Y, linestyle = '-', color = 'b')
# plt.show()
