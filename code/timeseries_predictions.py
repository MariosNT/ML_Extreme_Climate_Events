"""
Code to analyse the timeseries predictions
"""

### Importing useful packages

import numpy as np
import matplotlib.pyplot as plt
from timeseries_v3 import cptimeseries
from timeseries_extreme import cptimeseries_extreme


### Importing observed data & model fields

year = 2000 #For now, we're focusing on a single year
extreme_case = False

location = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\code\\images\\year_'+str(year)+"\\"

Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\Rainfall_Cardiff_{}.npy'.format(year))
X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}.npy'.format(year))


if extreme_case:
    imlocation = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Images\\Cardiff_extreme_'+str(year)+"_gs5000"+"\\"   
    data_set = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_extreme_Cardiff_{}_gs5000.npz".format(year))

else:
    imlocation = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Images\\Cardiff_'+str(year)+"_gs10000"+"\\"
    data_set = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_Cardiff_{}_gs10000.npz".format(year))


### Importing timeseries of Z and Theta, after sampling
# Z = number of times rain/day
# Theta = parameters of the model (Eq. 10)
Z = data_set["Z"]
Theta = data_set["Theta"]
n_param = len(Theta.T)
n_days = len(Z.T)

### Checking convergence
# Gibbs_steps = number of times the sampler was run
sampling_steps = len(Z)
Gibbs_steps = np.arange(sampling_steps)
N_burn = 8000

def plot_Gibbs_samples(theta, n_burn=0):
    for i in range(n_param):
        plt.figure(figsize=(10, 8))
        plt.plot(Gibbs_steps[:n_burn], theta[:n_burn,i], linestyle = '-', color = 'b')
        plt.plot(Gibbs_steps[n_burn:], theta[n_burn:,i], linestyle = '-', color = 'r')
        plt.title("Parameter {} for year {}".format(i+1, year))
        plt.show()
        # plt.savefig(imlocation+"parameteres_{}_Gsteps_{}.png".format(i+1, year))
        # plt.close()

        
plot_Gibbs_samples(Theta, N_burn)


### Making predictions
# y_pred = amount of rainfall (mm)
Z = Z[N_burn:,:]
Theta = Theta[N_burn:,:]
sampling_steps = len(Z)
 
# We name new parameter arrays after the burn in
# This will save computation time for predictions
z_pred = np.ones(Z.shape)
y_pred = np.ones(Z.shape)
l_t = np.ones(Z.shape)
w_t = np.ones(Z.shape)
mu_t = np.ones(Z.shape)

# For the given model fields and sampled parameters, 
# we make predictions of the amount of rainfall per day.
# We discard nan values & re-predict for these
if extreme_case:
    for i in range(sampling_steps):
        _, y_pred[i], l_t[i], w_t[i], mu_t[i] = cptimeseries_extreme(Theta[i]).simulate(X)
        while sum(np.isnan(y_pred[i])) != 0.0:
            _, y_pred[i], l_t[i], w_t[i], mu_t[i] = cptimeseries_extreme(Theta[i]).simulate(X)    
else:
    for i in range(sampling_steps):
        _, y_pred[i], l_t[i], w_t[i], mu_t[i] = cptimeseries(Theta[i]).simulate(X)
        while sum(np.isnan(y_pred[i])) != 0.0:
            _, y_pred[i], l_t[i], w_t[i], mu_t[i] = cptimeseries(Theta[i]).simulate(X)
        
    
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
# plt.figure(figsize=(10, 8))
# plt.plot(time, y_mean, linestyle = '-', color = 'b')
# plt.plot(time, Y, marker='+', linestyle='', color = 'black')
# plt.fill_between(time, y_mean-y_std, y_mean+y_std)
# plt.ylim(0, np.max(Y)+1)
# plt.title("Y mean - Year {}".format(year))
# plt.xlabel("Days")
# plt.ylabel("precipitation (mm)")
# plt.show()
# plt.savefig(location+"precipitation_mean_{}.png".format(year))
# plt.close()    


# n, bins, patches = plt.hist(np.log10(np.ravel(l_t)), 15, density=True, facecolor='g', alpha=0.75, label='l_t (log)')
# plt.legend()
# plt.show()
# plt.savefig(imlocation+"lt_{}_Gsteps_{}.png".format(i+1, year))
# plt.close()

# n, bins, patches = plt.hist(np.log10(np.ravel(w_t)), 15, density=True, facecolor='g', alpha=0.75, label='w_t (log)')
# plt.legend()
# plt.show()
# plt.savefig(imlocation+"wt_{}_Gsteps_{}.png".format(i+1, year))
# plt.close()

# n, bins, patches = plt.hist(np.log10(np.ravel(np.nan_to_num(mu_t))), 15, density=True, facecolor='g', alpha=0.75, label='mu_t (log)')
# plt.legend()
# #plt.show()
# plt.savefig(imlocation+"mut_{}_Gsteps_{}.png".format(i+1, year))
# plt.close()

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
# plt.savefig(imlocation+"precipitation_median_{}.png".format(year))
# plt.close()    

lt_median = np.quantile(l_t, 0.5, axis=0)
lt_68 = np.quantile(l_t, 0.68, axis=0)
lt_95 = np.quantile(l_t, 0.95, axis=0)
wt_median = np.quantile(w_t, 0.5, axis=0)
mut_median = np.quantile(np.nan_to_num(mu_t), 0.5, axis=0)

plt.figure(figsize=(10, 8))
plt.plot(time, lt_median, linestyle = '-', color = 'b')
plt.fill_between(time, lt_median, lt_68, color='red')
plt.fill_between(time, lt_median, lt_95, color='red', alpha=0.5)
plt.xlabel("Days")
plt.ylabel("l_t")
# plt.show()
# plt.savefig(imlocation+"lt_median_{}.png".format(year))
# plt.close() 

plt.figure(figsize=(10, 8))
plt.plot(time, wt_median, linestyle = '-', color = 'b')
plt.xlabel("Days")
plt.ylabel("w_t")
plt.show()
# plt.savefig(imlocation+"wt_median_{}.png".format(year))
# plt.close() 

plt.figure(figsize=(10, 8))
plt.plot(time, mut_median, linestyle = '-', color = 'b')
plt.xlabel("Days")
plt.ylabel("mu_t")
plt.show()
# plt.savefig(imlocation+"mut_median_{}.png".format(year))
# plt.close() 



##### Calculate RMS error & RMS spread

def rms_error_spread(rain_obs, rain_pred, rain_samples):
    rms_error = np.sqrt((rain_obs-rain_pred)**2)

    forecast_subs = np.sum((rain_samples-rain_pred)**2, axis=0)
    rms_spread = np.sqrt(forecast_subs/len(rain_samples))
    
    return rms_error, rms_spread

rms_error, rms_spread = rms_error_spread(Y, y_median, y_pred)
# #sorted_rms_error, sorted_rms_spread = np.sort(rms_error), np.sort(rms_spread)

# plt.figure(figsize=(10, 8))
# plt.plot(rms_spread, rms_error, 'o-', color = 'black', alpha = 0.5)
# plt.plot(rms_spread, rms_spread, linestyle = '--', color = 'black')
# plt.xlabel("RMS spread")
# plt.ylabel("RMS error")
# plt.show()
# #plt.savefig(location+"spread-skill_{}.png".format(year))
# plt.close()    

"With bins"
#So far binning the final results - would it be better to been earlier?

rms_spread = rms_spread.reshape(len(rms_spread),1)
rms_error = rms_error.reshape(len(rms_error),1)

joint_rms = np.concatenate((rms_spread, rms_error), axis=1)
joint_rms_sorted = np.sort(joint_rms, axis=0)

n_bins = 16
rms_spread_binned = np.array(np.array_split(joint_rms_sorted[:,0], n_bins), dtype=object)
rms_error_binned = np.array(np.array_split(joint_rms_sorted[:,1], n_bins), dtype=object)

v_mean = np.vectorize(np.mean)

rms_spread_mean = v_mean(rms_spread_binned)
rms_error_mean = v_mean(rms_error_binned)

plt.figure(figsize=(10, 8))
plt.plot(rms_spread_mean, rms_error_mean, 'o-', color = 'black')
plt.plot(np.sort(rms_spread_mean), np.sort(rms_spread_mean), linestyle = '--', color = 'grey')
plt.xlabel("RMS spread binned")
plt.ylabel("RMS error binned")
plt.show()
# plt.savefig(imlocation+"spread-skill_{}.png".format(year))
# plt.close()

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



### Calculate ROC curve

def true_false_positives(rain_thres, rain_obs, rain_pred, roc_thres):    
    bool_obs = rain_obs > rain_thres
    bool_pred = rain_pred > rain_thres
    bool_new = np.sum(bool_pred, axis=0) >= roc_thres
    
    TP = np.sum(bool_obs & bool_new)
    FP = np.sum(np.invert(bool_obs) & bool_new)
    TN = np.sum(np.invert(bool_obs) & np.invert(bool_new))
    FN = np.sum(bool_obs & np.invert(bool_new))
    
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    
    return TPR, FPR

def ROC_curves(rain_thres, rain_obs, rain_pred, roc_thres):
    true_positives = []
    false_positives = []

    for i in roc_thres:
        tpr, fpr = true_false_positives(rain_thres, rain_obs, rain_pred, i)
        true_positives.append(tpr)
        false_positives.append(fpr)

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    AUC = np.round(np.abs(np.trapz(true_positives, false_positives)), 3)      
    
    return true_positives, false_positives, AUC

def ROC_plot(tpr_array, fpr_array, auc_array, rains): 
    plt.figure(figsize=(10, 8))
    linestyles = ['-', '--', '-.', ':']
    colors = ['blue', 'magenta', 'red', 'green']
    
    # plot diagonal line
    x_values = np.linspace(0, 1, 10)
    plt.plot(x_values, x_values, linestyle = '-', color = 'black')
    
    for i in range(len(tpr_array)):
        plt.plot(fpr_array[i], tpr_array[i], linestyle = linestyles[i],\
                 marker = 'x', color = colors[i], alpha=0.3,\
                 label= "AUC = {}".format(auc_array[i]) + " RT = {} mm".format(rains[i]))
            
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()
    # plt.savefig(imlocation+"ROC_{}.png".format(year))
    # plt.close()  
    
thresholds = np.union1d(np.arange(0, 10, 1), np.arange(20, sampling_steps//2, 30))

Tpr0, Fpr0, auc0 = ROC_curves(0, Y, y_pred, thresholds)
Tpr5, Fpr5, auc5 = ROC_curves(5, Y, y_pred, thresholds)
Tpr15, Fpr15, auc15 = ROC_curves(15, Y, y_pred, thresholds)
Tpr25, Fpr25, auc25 = ROC_curves(25, Y, y_pred, thresholds)


rains = [0, 5, 15, 25]
ROC_plot([Tpr0, Tpr5, Tpr15, Tpr25], [Fpr0, Fpr5, Fpr15, Fpr25], [auc0, auc5, auc15, auc25], rains)


### Calculate probability of precipitation
# How many days we predict rainfall above a threshold
# Compare with observed values

def precipitation_above_x(rain_thres, rainvalues, all_days=True):
    if np.shape(rainvalues)[0]==len(Y):
        all_days = len(rainvalues)
        length_above_x = np.sum(rainvalues>rain_thres)
    
        return length_above_x/all_days
    
    elif all_days: 
        rainfall_all = rainvalues.flatten()
        all_days = len(rainfall_all)
        length_above_x = np.sum(rainfall_all>rain_thres)
        
        return length_above_x/all_days
    
    else:
        samples_per_day = np.shape(rainvalues)[0]
        length_above_x = np.sum(rainvalues>rain_thres, axis=0)
        prob_per_day = length_above_x/samples_per_day
        
        return np.mean(prob_per_day)

rain_thresholds = np.arange(0, 30, 1)
rain_probability_obs = []
rain_probability_pred = []
rain_probability_pred_95 = []
rain_probability_samples = []
rain_probability_samples_day = []

for rain in rain_thresholds:
    rain_probability_obs.append(precipitation_above_x(rain, Y))
    rain_probability_pred.append(precipitation_above_x(rain, y_median))
    rain_probability_pred_95.append(precipitation_above_x(rain, y_95))
    rain_probability_samples.append(precipitation_above_x(rain, y_pred))
    rain_probability_samples_day.append(precipitation_above_x(rain, y_pred, False))

    
plt.figure(figsize=(10, 8))
plt.plot(rain_thresholds, rain_probability_obs, linestyle = '--', color = 'black', label = "Obs.")
plt.plot(rain_thresholds, rain_probability_pred, linestyle = '-', color = 'black', label = "Pred.")
plt.plot(rain_thresholds, rain_probability_pred_95, linestyle = '-.', color = 'black', label = "Pred. 95")
plt.plot(rain_thresholds, rain_probability_samples, linestyle = ':', color = 'black', label = "Pred. Samples")
plt.plot(rain_thresholds, rain_probability_samples_day, marker = 'x', linestyle = ' ', color = 'black', label = "Pred. Samples/Day")
plt.xlabel("Rain thresholds [x (mm)]")
plt.ylabel("Probability [rain>x]")
plt.legend()
# plt.show()
# plt.savefig(imlocation+"precipitation_prob_{}.png".format(year))
# plt.close()    

### Plotting residuals
# plt.figure(figsize=(10, 8))
# plt.plot(time, y_mean-Y, linestyle = '-', color = 'b')
# plt.show()
