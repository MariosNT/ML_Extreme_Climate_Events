"""
Code to analyse the timeseries predictions
"""

### Importing useful packages

import numpy as np
import matplotlib.pyplot as plt
from timeseries_v3 import cptimeseries
from timeseries_extreme import cptimeseries_extreme


### Importing observed data & model fields

year = 1 #For now, we're focusing on a single year
year_predict = 1

gs = 30000
N_burn = 29000

savefig = False


#location = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\code\\images\\year_'+str(year)+"\\"

Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\Rainfall_Cardiff_{}.npy'.format(year_predict))
X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}_wv.npy'.format(year_predict))


data_set_ext = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_extreme_Cardiff_{}_gs{}_Z1_wv.npz".format(year, gs))
data_set = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_Cardiff_{}_gs{}_Z1_wv.npz".format(year, gs))

imlocation = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Images\\Comparison\\Cardiff_comparison_'+str(year)+"_pred{}_gs{}_wv".format(year_predict, gs)+"\\"   

### Importing timeseries of Z and Theta, after sampling
# Z = number of times rain/day
# Theta = parameters of the model (Eq. 10)
Z = data_set["Z"]
Theta = data_set["Theta"]

Z_ext = data_set_ext["Z"]
Theta_ext = data_set_ext["Theta"]

n_param = len(Theta.T)
n_days = len(X)

### Checking convergence
# Gibbs_steps = number of times the sampler was run
sampling_steps = len(Z)
Gibbs_steps = np.arange(sampling_steps)

def plot_Gibbs_samples(theta, theta_ext, n_burn=0):
    for i in range(n_param):
        plt.figure(figsize=(10, 8))
        plt.plot(Gibbs_steps[:n_burn], theta[:n_burn,i], linestyle = '-', color = 'b', alpha=0.8, label="time")
        plt.plot(Gibbs_steps[n_burn:], theta[n_burn:,i], linestyle = '-', color = 'r', alpha=0.8)
        plt.plot(Gibbs_steps[:n_burn], theta_ext[:n_burn,i], linestyle = '--', color = 'black', alpha=0.8, label="time_ext")
        plt.plot(Gibbs_steps[n_burn:], theta_ext[n_burn:,i], linestyle = '--', color = 'orange', alpha=0.8)
        plt.legend()
        plt.title("Parameter {} for year {}".format(i+1, year))
        if savefig:
            plt.savefig(imlocation+"parameteres_{}_Gsteps_{}.png".format(i+1, year))
            plt.close()
        else:
            plt.show()

        
plot_Gibbs_samples(Theta, Theta_ext, N_burn)


### Making predictions
# y_pred = amount of rainfall (mm)
Z = Z[N_burn:,:]
Theta = Theta[N_burn:,:]

Z_ext = Z_ext[N_burn:,:]
Theta_ext = Theta_ext[N_burn:,:]

sampling_steps = len(Z)
 
# We name new parameter arrays after the burn in
# This will save computation time for predictions

y_pred = np.ones((sampling_steps, len(X)))
y_pred_ext = np.ones((sampling_steps, len(X)))

l_t = np.ones((sampling_steps, len(X)))
w_t = np.ones((sampling_steps, len(X)))
mu_t = np.ones((sampling_steps, len(X)))

l_t_ext = np.ones((sampling_steps, len(X)))
w_t_ext = np.ones((sampling_steps, len(X)))
mu_t_ext = np.ones((sampling_steps, len(X)))

# For the given model fields and sampled parameters, 
# we make predictions of the amount of rainfall per day.
# We discard nan values & re-predict for these
for i in range(sampling_steps):
    _, y_pred_ext[i], l_t_ext[i], w_t_ext[i], mu_t_ext[i] = cptimeseries_extreme(Theta_ext[i]).simulate(X)
    while sum(np.isnan(y_pred_ext[i])) != 0.0:
        _, y_pred_ext[i], l_t_ext[i], w_t_ext[i], mu_t_ext[i] = cptimeseries_extreme(Theta_ext[i]).simulate(X)    
    _, y_pred[i], l_t[i], w_t[i], mu_t[i] = cptimeseries(Theta[i]).simulate(X)
    while sum(np.isnan(y_pred[i])) != 0.0:
        _, y_pred[i], l_t[i], w_t[i], mu_t[i] = cptimeseries(Theta[i]).simulate(X)
        


### Summary statistics of predictions

# Arbitrary (distribution) measures
y_median = np.quantile(y_pred, 0.5, axis=0)
y_68 = np.quantile(y_pred, 0.68, axis=0)
y_95 = np.quantile(y_pred, 0.95, axis=0)
y_max = np.max(y_pred, axis=0)

y_median_ext = np.quantile(y_pred_ext, 0.5, axis=0)
y_68_ext = np.quantile(y_pred_ext, 0.68, axis=0)
y_95_ext = np.quantile(y_pred_ext, 0.95, axis=0)
y_max_ext = np.max(y_pred_ext, axis=0)

# Our distribution is not a gaussian, so we 
# use quantiles for our predictions and their errors

# time = number of days that we predict rainfall for
time = np.arange(n_days)

# Plot that uses the median & different quantiles, as predicted value and errors
# Comparison plot for 95% quantile
plt.figure(figsize=(10, 8))
plt.plot(time, y_95_ext, linestyle = '-', color = 'b', label="time_ext95", alpha=0.6)
plt.plot(time, y_95, linestyle = '-', color = 'r', label="time95", alpha=0.6)
plt.plot(time, Y, marker='+', linestyle='', color = 'black')
plt.ylim(0, np.max(Y)+1)
plt.title("Y median Comparison - Year {}".format(year))
plt.xlabel("Days")
plt.ylabel("precipitation (mm)")
plt.legend()
if savefig:
    plt.savefig(imlocation+"precipitation_median_{}.png".format(year))
    plt.close()
else:
    plt.show()    

l_t = np.exp(-l_t)
l_t_ext = np.exp(-l_t_ext)

lt_median = np.quantile(l_t, 0.5, axis=0)
lt_68 = np.quantile(l_t, 0.68, axis=0)
lt_95 = np.quantile(l_t, 0.95, axis=0)

lt_median_ext = np.quantile(l_t_ext, 0.5, axis=0)
lt_68_ext = np.quantile(l_t_ext, 0.68, axis=0)
lt_95_ext = np.quantile(l_t_ext, 0.95, axis=0)


plt.figure(figsize=(10, 8))
plt.plot(time, lt_median_ext, linestyle = '-', color = 'b', label="exp[-lt_ext50]", alpha=0.8)
plt.plot(time, lt_median, linestyle = '--', color = 'r', label="exp[-lt_50]", alpha=0.8)
plt.xlabel("Days")
plt.legend()
if savefig:
    plt.savefig(imlocation+"lt_median_comparison_{}.png".format(year))
    plt.close()  
else:
    plt.show()   
    

##### Calculate RMS error & RMS spread

def rms_error_spread(rain_obs, rain_pred, rain_samples):
    rms_error = np.sqrt((rain_obs-rain_pred)**2)

    forecast_subs = np.sum((rain_samples-rain_pred)**2, axis=0)
    rms_spread = np.sqrt(forecast_subs/len(rain_samples))
    
    return rms_error, rms_spread

rms_error, rms_spread = rms_error_spread(Y, y_median, y_pred)
rms_error_ext, rms_spread_ext = rms_error_spread(Y, y_median_ext, y_pred_ext)
    
"With bins"
#So far binning the final results - would it be better to been earlier?

# Sorting
rms_spread = rms_spread.reshape(len(rms_spread),1)
rms_error = rms_error.reshape(len(rms_error),1)

joint_rms = np.concatenate((rms_spread, rms_error), axis=1)
joint_rms_sorted = joint_rms[joint_rms[:,0].argsort()]

rms_spread_ext = rms_spread_ext.reshape(len(rms_spread_ext),1)
rms_error_ext = rms_error_ext.reshape(len(rms_error_ext),1)

joint_rms_ext = np.concatenate((rms_spread_ext, rms_error_ext), axis=1)
joint_rms_sorted_ext = joint_rms_ext[joint_rms_ext[:,0].argsort()]

# Binning
n_bins = 16
rms_spread_binned = np.array(np.array_split(joint_rms_sorted[:,0], n_bins), dtype=object)
rms_error_binned = np.array(np.array_split(joint_rms_sorted[:,1], n_bins), dtype=object)

v_mean = np.vectorize(np.mean)

rms_spread_mean = v_mean(rms_spread_binned)
rms_error_mean = v_mean(rms_error_binned)

rms_error_mean = rms_error_mean[np.invert(np.isinf(rms_spread_mean))]
rms_spread_mean = rms_spread_mean[np.invert(np.isinf(rms_spread_mean))]


rms_spread_binned_ext = np.array(np.array_split(joint_rms_sorted_ext[:,0], n_bins), dtype=object)
rms_error_binned_ext = np.array(np.array_split(joint_rms_sorted_ext[:,1], n_bins), dtype=object)

rms_spread_mean_ext = v_mean(rms_spread_binned_ext)
rms_error_mean_ext = v_mean(rms_error_binned_ext)

rms_error_mean_ext = rms_error_mean_ext[np.invert(np.isinf(rms_spread_mean_ext))]
rms_spread_mean_ext = rms_spread_mean_ext[np.invert(np.isinf(rms_spread_mean_ext))]

# Plotting
plt.figure(figsize=(10, 8))

x_values = np.linspace(0, np.max(np.union1d(rms_spread_mean, rms_spread_mean_ext)), 10)
plt.plot(x_values, x_values, linestyle = '--', color = 'black')

plt.plot(rms_spread_mean, rms_error_mean, 'o-', color = 'black', label="Time")
plt.plot(rms_spread_mean_ext, rms_error_mean_ext, 'x-', color = 'blue', label="Ext")
plt.plot(np.sort(rms_spread_mean), np.sort(rms_spread_mean), linestyle = '--', color = 'grey')
plt.xlabel("RMS spread binned")
plt.ylabel("RMS error binned")
plt.legend()
if savefig:
    plt.savefig(imlocation+"spread-skill_comparison_{}.png".format(year))
    plt.close()  
else:
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

def ROC_plot(tpr_array, fpr_array, auc_array, rains, model): 
    plt.figure(figsize=(10, 8))
    linestyles = ['-', '--', '-.', ':']
    colors = ['blue', 'magenta', 'red', 'green']
    
    # plot diagonal line
    x_values = np.linspace(0, 1, 10)
    plt.plot(x_values, x_values, linestyle = '-', color = 'black')
    
    for i in range(len(tpr_array)):
        plt.plot(fpr_array[i], tpr_array[i], linestyle = linestyles[i],\
                 marker = 'x', color = colors[i], alpha=0.3,\
                 label= "AUC = {}".format(auc_array[i]) + ", RT = {} mm".format(rains[i])\
                     +", model = {}".format(model[i]))
            
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    if savefig:
        plt.savefig(imlocation+"ROC_{}.png".format(year))
        plt.close()  
    else:
        plt.show() 
 
    
thresholds = np.union1d(np.arange(0, 550, 1), np.arange(550, 850, 5))
thresholds = np.union1d(thresholds, np.arange(850, sampling_steps, 100))

# Tpr0, Fpr0, auc0 = ROC_curves(0, Y, y_pred, thresholds)
# Tpr5, Fpr5, auc5 = ROC_curves(5, Y, y_pred, thresholds)
Tpr15, Fpr15, auc15 = ROC_curves(15, Y, y_pred, thresholds)
Tpr25, Fpr25, auc25 = ROC_curves(25, Y, y_pred, thresholds)

Tpr15_ext, Fpr15_ext, auc15_ext = ROC_curves(15, Y, y_pred_ext, thresholds)
Tpr25_ext, Fpr25_ext, auc25_ext = ROC_curves(25, Y, y_pred_ext, thresholds)


rains = [15, 25, 15, 25]
models = ["std", "std", "extr", "extr"]
ROC_plot([Tpr15, Tpr25, Tpr15_ext, Tpr25_ext],\
         [Fpr15, Fpr25, Fpr15_ext, Fpr25_ext],\
         [auc15, auc25, auc15_ext, auc25_ext], rains, models)


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
rain_probability_samples = []
rain_probability_samples_ext = []


for rain in rain_thresholds:
    rain_probability_obs.append(precipitation_above_x(rain, Y))
    rain_probability_samples.append(precipitation_above_x(rain, y_pred))
    rain_probability_samples_ext.append(precipitation_above_x(rain, y_pred_ext))


    
plt.figure(figsize=(10, 8))
plt.plot(rain_thresholds, rain_probability_obs, linestyle = '--', color = 'black', label = "Obs.")
plt.plot(rain_thresholds, rain_probability_samples, linestyle = ':', color = 'red', label = "Pred.")
plt.plot(rain_thresholds, rain_probability_samples_ext, linestyle = ':', color = 'blue', label = "Pred. Ext")
plt.xlabel("Rain thresholds [x (mm)]")
plt.ylabel("Probability [rain>x]")
plt.legend()
if savefig:
    plt.savefig(imlocation+"precipitation_prob_comparison_{}.png".format(year))
    plt.close()
else:
    plt.show()       

