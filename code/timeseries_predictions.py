"""
Code to analyse the timeseries predictions

Equations correspond to ArXiv:2012.09821
"""

### Importing useful packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeseries_cp import cptimeseries
from timeseries_cp_extreme import cptimeseries_extreme

### Plots styling (to include for the final plots)

from matplotlib import rcParams
font = 'serif'
font_size = 15
rcParams['font.family'] = font
rcParams['font.size']= font_size
plt.rc('text', usetex=True)


### Importing observed data & model fields

year_training = 1 
year_predict = "1"
year_start = 1999
year_end = 1999
gs = 50000  # Gibbs_steps = number of times the sampler was run
N_burn = 49000  # Number of steps we ignore to achieve convergence


extreme_case = True  # True, if we want to predict with the "extreme model"
savefig = False  # True, if we want to save the figures

# Y loads observed rainfall values
# X loads extracted model fields
Y = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\Rainfall_Cardiff_{}.npy'.format(year_predict))
X = np.load('C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\model_fields_Cardiff_{}.npy'.format(year_predict))
X = X[:,:5]

x_size = X.shape[1]+1

print()
print("We are training the model for:", year_training, "and making predictions for:", year_predict)
print()
print("Number of Gibbs Steps used are:", gs, "with", N_burn, "discarded as burn-in phase.")
print()
print("Number of model fields used:", x_size-1)
if extreme_case:
    print()
    print("Predictions, using the extreme model")
print()

# Defining the location to save the data
# Uploading the data set with the timeseries of the model parameters from training
if extreme_case:
    imlocation = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Images\\Extreme\\Cardiff_extreme_'+str(year_training)+"_pred{}_gs{}_Z1".format(year_predict,gs)+"\\"   
    data_set = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_extreme_Cardiff_{}_gs{}_Z1.npz".format(year_training, gs))

else:
    imlocation = 'C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Images\\Standard\\Cardiff_'+str(year_training)+"_pred{}_gs{}".format(year_predict,gs)+"\\"   
    data_set = np.load("C:\\Users\\klera\\Documents\\GitHub\\ML_Extreme_Climate_Events\\Data\\Data\\timeseries_Cardiff_{}_gs{}.npz".format(year_training, gs))


### Importing timeseries of Z and Theta, after sampling
# Z = number of times rain/day
# Theta = parameters of the model (Eq. 10)

Z = data_set["Z"]
Theta = data_set["Theta"]
n_param = len(Theta.T)
n_days = len(X)


### Checking convergence

sampling_steps = len(Z)
Gibbs_steps = np.arange(sampling_steps)

def plot_Gibbs_samples(theta, n_burn=0):
    """ Function that plots the parameters timeseries """
    """ with separates burn-in and final samples """
    for i in range(n_param):
        plt.figure(figsize=(10, 8))
        plt.plot(Gibbs_steps[:n_burn], theta[:n_burn,i], linestyle = '-', color = 'b')
        plt.plot(Gibbs_steps[n_burn:], theta[n_burn:,i], linestyle = '-', color = 'r')
        plt.title("Parameter {} for year {}".format(i+1, year_training))
        if savefig:
            plt.savefig(imlocation+"parameteres_{}_Gsteps_{}.png".format(i+1, year_training))
            plt.close()
        else:
            plt.show()


plot_Gibbs_samples(Theta, N_burn)


### Making predictions
# y_pred = amount of rainfall (mm)

Z = Z[N_burn:,:]
Theta = Theta[N_burn:,:]
sampling_steps = len(Z)
 
# We name new parameter arrays after the burn in.
# This will save computation time for predictions
z_pred = np.ones((sampling_steps, len(X)))
y_pred = np.ones((sampling_steps, len(X)))

# Variables for checks only (uncomment for testing purposes)

# l_t = np.ones((sampling_steps, len(X)))
# w_t = np.ones((sampling_steps, len(X)))
# mu_t = np.ones((sampling_steps, len(X)))


# For the given model fields and sampled parameters, 
# we make predictions of the amount of rainfall per day.
# We discard nan values & re-predict for these

if extreme_case:
    for i in range(sampling_steps):
        _, y_pred[i], _, _, _ = cptimeseries_extreme(Theta[i], k=x_size).simulate(X)
        while (sum(np.isnan(y_pred[i])) != 0.0): # or np.sum((y_pred[i]>100000))>0:
            print()
            print("High rainfall prediction in step:", i)
            _, y_pred[i], _, _, _ = cptimeseries_extreme(Theta[i], k=x_size).simulate(X)    
else:
    for i in range(sampling_steps):
        _, y_pred[i], _, _, _ = cptimeseries(Theta[i], k=x_size).simulate(X)
        while sum(np.isnan(y_pred[i])) != 0.0 or np.sum((y_pred[i]>1000))>0:
            print()
            print("High rainfall prediction in step:", i)
            _, y_pred[i], _, _, _ = cptimeseries(Theta[i], k=x_size).simulate(X)
        

    
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
time = pd.date_range('{}-01-01'.format(year_start), '{}-12-31'.format(year_end), periods=n_days)


# Plot that uses the mean & std, as predicted value and error

# plt.figure(figsize=(10, 8))
# plt.plot(time, y_mean, linestyle = '-', color = 'b')
# plt.plot(time, Y, marker='+', linestyle='', color = 'black')
# plt.fill_between(time, y_mean-y_std, y_mean+y_std)
# plt.ylim(0, np.max(Y)+1)
# plt.title("Y mean - Year {}".format(year))
# plt.xlabel("Days")
# plt.ylabel("precipitation (mm)")
# if savefig:
#     plt.savefig(imlocation+"precipitation_mean_{}.png".format(year_training))
#     plt.close()
# else:
#     plt.show()
  


### 1- Plot that uses the median & different quantiles, as predicted value and errors

plt.figure(figsize=(10, 8))
plt.plot(time, Y, marker='+', linestyle='', color = 'black', label = 'Obs.')
plt.plot(time, y_median, linestyle = '-', color = 'b', label='50 \%')
plt.fill_between(time, y_median, y_68, color='red', label='68 \%')
plt.fill_between(time, y_median, y_95, color='red', alpha=0.5, label='95 \%')
plt.ylim(0, np.max(Y)+1)
plt.title("Y median - Year {}".format(year_training))
plt.xlabel("Time")
plt.ylabel("precipitation (mm)")
plt.legend()
if savefig:
    plt.savefig(imlocation+"precipitation_median_{}.png".format(year_training))
    plt.close()
else:
    plt.show()
    

### 2- Scatter plot between observables and predictions (median & 95%)

plt.figure(figsize=(10, 8))

# Plot diagonal line
x_values = np.linspace(0, np.max(np.log10(Y+1)), 10)
plt.plot(x_values, x_values, linestyle = '--', color = 'black')

# Simple linear fit
z_median = np.polyfit(Y, y_median, 1)
p_50 = np.poly1d(z_median)

z_95 = np.polyfit(Y, y_95, 1)
p_95 = np.poly1d(z_95)

# We transform the values to Y+1, before taking the log
plt.scatter(np.log10(Y+1), np.log10(y_median+1), alpha=0.8, marker='x', c='r', label = '50 \%')
plt.scatter(np.log10(Y+1), np.log10(y_95+1), alpha=0.8, marker='x', c='b', label = '95 \%')
plt.plot(np.log10(np.sort(Y)+1), np.log10(p_50(np.sort(Y))+1), linestyle = '-.', alpha=0.6, color='r')
plt.plot(np.log10(np.sort(Y)+1), np.log10(p_95(np.sort(Y))+1), linestyle = '--', alpha=0.6, color='b')
plt.ylim(-0.05, np.max(np.log10(Y+1)))
plt.title("Scatter Log[Y+1] Plot - Year {}".format(year_training))
plt.ylabel("Predictions")
plt.xlabel("Observations")
plt.legend()
if savefig:
    plt.savefig(imlocation+"scatter_plot_{}.png".format(year_training))
    plt.close()
else:
    plt.show()   


### ONLY FOR TESTING

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

# l_t = np.exp(-l_t)
# lt_median = np.quantile(l_t, 0.5, axis=0)
# lt_68 = np.quantile(l_t, 0.68, axis=0)
# lt_95 = np.quantile(l_t, 0.95, axis=0)
# wt_median = np.quantile(w_t, 0.5, axis=0)
# mut_median = np.quantile(np.nan_to_num(mu_t), 0.5, axis=0)

# plt.figure(figsize=(10, 8))
# plt.plot(time, lt_median, linestyle = '-', color = 'b')
# plt.fill_between(time, lt_median, lt_68, color='red')
# plt.fill_between(time, lt_median, lt_95, color='red', alpha=0.5)
# plt.xlabel("Days")
# plt.ylabel("Exp[-l_t]")
# if savefig:
#     plt.savefig(imlocation+"lt_median_comparison_{}.png".format(year))
#     plt.close()  
# else:
#     plt.show()

# plt.figure(figsize=(10, 8))
# plt.plot(time, wt_median, linestyle = '-', color = 'b')
# plt.xlabel("Days")
# plt.ylabel("w_t")
# plt.show()
# plt.savefig(imlocation+"wt_median_{}.png".format(year))
# plt.close() 

# plt.figure(figsize=(10, 8))
# plt.plot(time, mut_median, linestyle = '-', color = 'b')
# plt.xlabel("Days")
# plt.ylabel("mu_t")
# plt.show()
# plt.savefig(imlocation+"mut_median_{}.png".format(year))
# plt.close() 



### 3- Calculate RMS error & RMS spread - Eqs. (18) & MAB p.12

def rms_error_spread(rain_obs, rain_pred, rain_samples):
    """ RMS error - b """
    """ RMS spread - σ """
    rms_error = np.sqrt((rain_obs-rain_pred)**2)
    mab_error = np.abs(rain_obs-rain_pred)

    forecast_subs = np.sum((rain_samples-rain_pred)**2, axis=0)
    rms_spread = np.sqrt(forecast_subs/len(rain_samples))
    
    y_75 = np.quantile(rain_samples, 0.75, axis=0)
    y_25 = np.quantile(rain_samples, 0.25, axis=0)
    
    return rms_error, rms_spread, mab_error, y_75-y_25

rms_error, rms_spread, mab_error, quant_reg = rms_error_spread(Y, y_median, y_pred) 

"With bins"
n_bins = 16

rms_spread = rms_spread.reshape(len(rms_spread),1)
rms_error = rms_error.reshape(len(rms_error),1)
mab_error = mab_error.reshape(len(mab_error),1)
quant_reg = quant_reg.reshape(len(quant_reg),1)

# Spread-Skill curve
joint_rms = np.concatenate((rms_spread, rms_error), axis=1)
joint_rms_sorted = joint_rms[joint_rms[:,0].argsort()]

rms_spread_binned = np.array(np.array_split(joint_rms_sorted[:,0], n_bins), dtype=object)
rms_error_binned = np.array(np.array_split(joint_rms_sorted[:,1], n_bins), dtype=object)

# Spread-Quantile curve
joint_rms_quant = np.concatenate((quant_reg, rms_error), axis=1)
joint_rms_sorted_quant = joint_rms_quant[joint_rms_quant[:,0].argsort()]

quant_reg_binned = np.array(np.array_split(joint_rms_sorted_quant[:,0], n_bins), dtype=object)
rms_error_binned_quant = np.array(np.array_split(joint_rms_sorted_quant[:,1], n_bins), dtype=object)

# MAB error binning
mab_error_binned = np.array(np.array_split(mab_error, n_bins), dtype=object)

# Vectorise the "mean" function, to calculate mean per bin
v_mean = np.vectorize(np.mean)

rms_spread_mean = v_mean(rms_spread_binned)
rms_error_mean = v_mean(rms_error_binned)

rms_error_mean_quant = v_mean(rms_error_binned_quant)
quant_reg_mean = v_mean(quant_reg_binned)

mab_error_mean = v_mean(mab_error_binned)

### Spread-Skill plot
plt.figure(figsize=(10, 8))

# plot diagonal line
x_values = np.linspace(0, np.max(rms_spread_mean), 10)
plt.plot(x_values, x_values, linestyle = '--', color = 'black')

plt.plot(rms_spread_mean, rms_error_mean, 'o-', color = 'black')
plt.xlabel("RMS spread binned")
plt.ylabel("RMS error binned")
if savefig:
    plt.savefig(imlocation+"spread-skill_{}.png".format(year_training))
    plt.close()  
else:
    plt.show()   

### Spread-Quantiles plot
plt.figure(figsize=(10, 8))

# plot diagonal line
x_values = np.linspace(0, np.max(quant_reg_mean), 10)
plt.plot(x_values, x_values, linestyle = '--', color = 'black')

plt.plot(quant_reg_mean, rms_error_mean_quant, 'o-', color = 'black')
plt.xlabel("Interquantiles binned")
plt.ylabel("RMS error binned")
if savefig:
    plt.savefig(imlocation+"spread-quantiles_{}.png".format(year_training))
    plt.close()  
else:
    plt.show()   

"""
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

plt.figure(figsize=(10, 8))

# plot diagonal line
t_values = np.linspace(0, 16, 16)
plt.plot(t_values, mab_error_mean, linestyle = '--', color = 'black')

plt.xlabel("t values")
plt.ylabel("MAB error binned")
if savefig:
    plt.savefig(imlocation+"MAB_error_{}.png".format(year))
    plt.close()  
else:
    plt.show()   


plt.figure(figsize=(10, 8))

# plot diagonal line
t_values = np.linspace(0, 16, 16)
plt.plot(t_values, rms_error_mean, linestyle = '--', color = 'black')

plt.xlabel("t values")
plt.ylabel("RMS error binned")
if savefig:
    plt.savefig(imlocation+"RMS_error_{}.png".format(year))
    plt.close()  
else:
    plt.show()  
"""


### 4- Calculate ROC curve

def true_false_positives(rain_thres, rain_obs, rain_pred, roc_thres):
    """ Function that calculates the TP and FP rates """
    """ TP: positive prediction, when positive observation """    
    """ FP: positive prediction, when negative observation """
    """ Positive observation = above a rain threshold """    
    
    bool_obs = rain_obs > rain_thres
    bool_pred = rain_pred > rain_thres
    
    # To consider the prediction correct, we want a certain number of 
    # samples to be correct. This is checked by the ROC threshold
    # If roc_thres small, almost all predictions will be positive
    # If roc_thres high, almost all predictions will be negative
    
    bool_new = np.sum(bool_pred, axis=0) >= roc_thres
    
    TP = np.sum(bool_obs & bool_new)
    FP = np.sum(np.invert(bool_obs) & bool_new)
    TN = np.sum(np.invert(bool_obs) & np.invert(bool_new))
    FN = np.sum(bool_obs & np.invert(bool_new))
    
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    
    return TPR, FPR, TP, FP, TN, FN


def ROC_curves(rain_thres, rain_obs, rain_pred, roc_thres):
    """ Function that calculates the ROC curves """
    true_positives = []
    false_positives = []

    for i in roc_thres:
        # For different "roc_thres", the TPR, FPR change
        tpr, fpr, _, _, _, _ = true_false_positives(rain_thres, rain_obs, rain_pred, i)
        true_positives.append(tpr)
        false_positives.append(fpr)

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    
    # AUC: area under the curve
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
                 label= "AUC = {}".format(auc_array[i]) + ", RT = {} mm".format(rains[i]))
            
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    if savefig:
        plt.savefig(imlocation+"ROC_{}.png".format(year_training))
        plt.close()  
    else:
        plt.show()   

# thresholds = np.union1d(np.arange(0, 550, 1), np.arange(550, 650, 5))
# thresholds = np.union1d(thresholds, np.arange(650, sampling_steps, 100))
thresholds = np.arange(0,sampling_steps,1)

Tpr0, Fpr0, auc0 = ROC_curves(0, Y, y_pred, thresholds)
Tpr5, Fpr5, auc5 = ROC_curves(5, Y, y_pred, thresholds)
Tpr15, Fpr15, auc15 = ROC_curves(15, Y, y_pred, thresholds)
Tpr25, Fpr25, auc25 = ROC_curves(25, Y, y_pred, thresholds)


rains = [0, 5, 15, 25]
ROC_plot([Tpr0, Tpr5, Tpr15, Tpr25], [Fpr0, Fpr5, Fpr15, Fpr25], [auc0, auc5, auc15, auc25], rains)



def positive_negative_timeseries(rain_thres, rain_obs, rain_pred, thres):
    tp = np.zeros(sampling_steps)
    fp = np.zeros(sampling_steps)
    tn = np.zeros(sampling_steps)
    fn = np.zeros(sampling_steps)
    tpr = np.zeros(sampling_steps)
    fpr = np.zeros(sampling_steps)
    
    for i in range(sampling_steps):
        tpr[i], fpr[i], tp[i], fp[i], tn[i], fn[i] = true_false_positives(rain_thres, rain_obs, rain_pred, thres[i])
        
    return tpr, fpr, tp, fp, tn, fn

def positive_negative_plot(tpr_array, fpr_array, tp_array, fp_array, tn_array, fn_array, thres): 
    plt.figure(figsize=(10, 8))  
    
    plt.plot(thres, tpr_array, linestyle = '-',\
             color = 'b', alpha=0.5,\
             label= "TPR")
    plt.plot(thres, fpr_array, linestyle = '-',\
             color = 'r', alpha=0.5,\
             label= "FPR")
    
    plt.plot(thres[::10], tp_array[::10]/np.max(tp_array), linestyle = 'None',\
             color = 'b', alpha=0.4, marker ='x',\
             label= "TP")
    plt.plot(thres, fn_array/np.max(fn_array), linestyle = '-.',\
             color = 'b', alpha=0.3,\
             label= "FN")
                
    plt.plot(thres[::10], fp_array[::10]/np.max(fp_array), linestyle = 'None',\
             color = 'r', alpha=0.4, marker ='x',\
             label= "FP")
    plt.plot(thres, tn_array/np.max(tn_array), linestyle = '-.',\
             color = 'r', alpha=0.3,\
             label= "TN")
        
    plt.xlabel("Sampling Steps")
    plt.legend()
    if savefig:
        plt.savefig(imlocation+"PN_timeseries_{}.png".format(year_training))
        plt.close()  
    else:
        plt.show()   

tpr, fpr, tp, fp, tn, fn = positive_negative_timeseries(2, Y, y_pred, thresholds)
positive_negative_plot(tpr, fpr, tp, fp, tn, fn, thresholds)


### 5- Calculate probability of precipitation
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
#     rain_probability_pred.append(precipitation_above_x(rain, y_median))
#     rain_probability_pred_95.append(precipitation_above_x(rain, y_95))
    rain_probability_samples.append(precipitation_above_x(rain, y_pred))
    # rain_probability_samples_day.append(precipitation_above_x(rain, y_pred, False))

    
plt.figure(figsize=(10, 8))
plt.plot(rain_thresholds, rain_probability_obs, linestyle = '--', color = 'black', label = "Obs.")
# plt.plot(rain_thresholds, rain_probability_pred, linestyle = '-', color = 'black', label = "Pred.")
# plt.plot(rain_thresholds, rain_probability_pred_95, linestyle = '-.', color = 'black', label = "Pred. 95")
plt.plot(rain_thresholds, rain_probability_samples, linestyle = ':', color = 'black', label = "Pred. Samples")
# plt.plot(rain_thresholds, rain_probability_samples_day, marker = 'x', linestyle = ' ', color = 'black', label = "Pred. Samples/Day")
plt.xlabel("Rain thresholds [x (mm)]")
plt.ylabel("Probability [rain>x]")
plt.legend()
if savefig:
    plt.savefig(imlocation+"precipitation_prob_comparison_{}.png".format(year_training))
    plt.close()
else:
    plt.show()   
