"""

"""

import numpy as np
import matplotlib.pyplot as plt
from timeseries_v3 import cptimeseries

#Observed data & model fields
Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\rainfall_Cardiff_1979.npy')

X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff.npy')
# Calculating transpose such that each row corresponds for a day
X = np.transpose(X)
# Calculating windspeed and consider that as avraible
X = np.concatenate((X[:,[0,3,4,5]],np.sqrt(pow(X[:,1],2)+pow(X[:,2],2)).reshape(-1,1)), axis=1)
# Standardize data (making each column having 0 mean and stdev 1)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)


#After sampling

data_set = np.load("timeseries_samples.npz")
Z = data_set["Z"]
Theta = data_set["Theta"]

### Checking convergence

Gibbs_steps = np.arange(len(Theta))

# for i in range(len(Theta.T)):
#     plt.figure(figsize=(10, 8))
#     plt.plot(Gibbs_steps, Theta[:,i], linestyle = '-', color = 'b')
#     plt.title("Parameter {}".format(i))
#     plt.show()


### Making predictions

z_pred = np.ones(Z.shape)
y_pred = np.ones(Z.shape)


# Discard nan values - re-predict for these
for i in range(len(Z)):
    _, y_pred[i] = cptimeseries(Theta[i]).simulate(X)
    while sum(np.isnan(y_pred[i])) != 0.0:
        _, y_pred[i] = cptimeseries(Theta[i]).simulate(X)
        
    
# Remember Z = number of times it rains & y = amount of rainfall(mm)
## Need to throw away some steps for burn-in

### Name a new y_pred after the burn in (instead of cutting all the time)

N_burn = 100

# Gaussian measures
y_mean = np.mean(y_pred[N_burn:], axis=0)
y_std = np.std(y_pred[N_burn:], axis=0)

# Arbitrary (distribution) measures
y_median = np.quantile(y_pred[N_burn:], 0.5, axis=0)
y_68 = np.quantile(y_pred[N_burn:], 0.68, axis=0)
y_95 = np.quantile(y_pred[N_burn:], 0.95, axis=0)
y_max = np.max(y_pred[N_burn:], axis=0)

### Update to plot quantiles instead of std (not gaussian dist.)
time = np.arange(len(y_mean))

plt.figure(figsize=(10, 8))
plt.plot(time, y_mean, linestyle = '-', color = 'b')
plt.plot(time, Y, marker='+', linestyle='', color = 'black')
plt.fill_between(time, y_mean-y_std, y_mean+y_std)
plt.ylim(0, np.max(Y)+1)
plt.title("Y mean")
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(time, y_median, linestyle = '-', color = 'b')
plt.plot(time, Y, marker='+', linestyle='', color = 'black')
plt.fill_between(time, y_median-y_68, y_median+y_68, color='red')
plt.fill_between(time, y_median-y_95, y_median+y_95, color='red', alpha=0.5)
plt.fill_between(time, y_median-y_max, y_median+y_max, color='red', alpha=0.2)
plt.ylim(0, np.max(Y)+1)
plt.title("Y median")
plt.show()



##### Calculate RMS error & RMS spread

rms_error = np.sqrt((Y-y_median)**2)

forecast_subs = np.sum((y_pred[N_burn:]-y_median)**2, axis=0)
rms_spread = np.sqrt(forecast_subs/len(y_pred[N_burn:]))

plt.figure(figsize=(10, 8))
plt.scatter(rms_spread, rms_error, linestyle = '-', color = 'black')
plt.xlabel("RMS spread")
plt.ylabel("RMS error")
plt.show()

"With bins"
#So far binning the final results - would it be better to been earlier?

## Break the 365 days to 73 bins of 5 days
# Y_binned = Y.reshape((73,5))
# Y_binned_mean = np.mean(Y_binned, axis=1)

# y_median_binned = y_median.reshape((73,5))
# y_median_binned_mean = np.mean(y_median_binned, axis=1)

# rms_error_binned = np.sqrt((Y_binned_mean-y_median_binned_mean)**2)

rms_error_binned = rms_error.reshape((73,5))
rms_error_mean = np.mean(rms_error_binned, axis=1)

rms_spread_binned = rms_spread.reshape((73,5))
rms_spread_mean = np.mean(rms_spread_binned, axis=1)

plt.figure(figsize=(10, 8))
plt.scatter(rms_spread_mean, rms_error_mean, color = 'black')
plt.plot(rms_spread_mean, rms_spread_mean, linestyle = '--', color = 'grey')
plt.xlabel("RMS spread binned")
plt.ylabel("RMS error binned")
plt.show()

###### Calculate RMSB error & MAB error
# Here we only have 1 location, so S=1
# We have values for 1 year, so T=365

T = 365

time_subs = np.sum((Y-y_median)**2)
rmsb_error = np.sqrt(time_subs/T)

abs_subs = np.sum(np.abs(Y-y_median))
mab_error = abs_subs/T

print("RMSB error is: ", rmsb_error)
print("MAB error is: ", mab_error)



##### Calculate ROC curve

def true_false_positives(rain_thres, rain_obs, rain_pred):
    all_days = len(rain_obs) 
    
    bool_obs = rain_obs > rain_thres
    P = sum(bool_obs)
    N = all_days-P
    
    bool_pred = rain_pred > rain_thres
    
    TP = sum(bool_obs & bool_pred)/P
    FP = sum(np.invert(bool_obs) & bool_pred)/N  
    
    return TP, FP

true_positives = []
false_positives = []
threshold = 5

for i in range(len(y_pred[N_burn:])):
    tp, fp = true_false_positives(threshold, Y, y_pred[N_burn:][i])
    true_positives.append(tp)
    false_positives.append(fp)

plt.figure(figsize=(10, 8))
plt.scatter(false_positives, true_positives, linestyle = '--', color = 'black')
plt.xlabel("FP")
plt.ylabel("TP")
plt.show()



##### Calculate probability of precipitation

def precipitation_above_x(rain_thres, rainvalues):
    all_days = len(rainvalues)
    length_above_x = len(rainvalues[rainvalues>rain_thres])
    
    return length_above_x/all_days

rain_thresholds = np.arange(0, 30, 5)
rain_probability_obs = []
rain_probability_pred = []

for rain in rain_thresholds:
    rain_probability_obs.append(precipitation_above_x(rain, Y))
    rain_probability_pred.append(precipitation_above_x(rain, y_median))
    
plt.figure(figsize=(10, 8))
plt.plot(rain_thresholds, rain_probability_obs, linestyle = '--', color = 'black')
plt.plot(rain_thresholds, rain_probability_pred, linestyle = '-', color = 'black')
plt.xlabel("Rain thresholds [x (mm)]")
plt.ylabel("Probability [rain>x]")
plt.show()
    

### Plotting residuals
# plt.figure(figsize=(10, 8))
# plt.plot(time, y_mean-Y, linestyle = '-', color = 'b')
# plt.show()