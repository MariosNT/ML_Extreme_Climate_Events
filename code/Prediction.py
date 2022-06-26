########################
### Loading Packages ###
########################

import os.path
import numpy as np
#import copy
import pandas as pd
import pylab as plt


#from Sampler import EllipticalSliceSampling
#from scipy.stats import gamma, multivariate_normal, poisson
# For purely serial computing
#from parallel.backends import BackendDummy as Backend
# For MPI parallelized computing
#from parallel.backends import BackendMPI as Backend
#backend = Backend()
# number of steps Gibbs we want to use


###################
### Run Options ###
###################

extreme_case = True
ef = True
zknown = True
postconvcheck = True
savefig = False

#####################
### Set Variables ###
#####################

years_training = '1999'
location = 4
N_burn = 1000


###############################
### Loading and saving data ###
###############################

if extreme_case:
    from timeseries_cp_extreme import cptimeseries_extreme as model
else:
    from timeseries_cp import cptimeseries as model

if extreme_case:
    imlocation = 'Figure/Extreme/'
else:
    imlocation = 'Figure/Standard/'



# Read Model fields
if ef:
    X = np.load('../Data/Data/model_fields_multiple_{}_small_ef.npy'.format(years_training))
else:
    X = np.load('../Data/Data/model_fields_multiple_{}_small.npy'.format(years_training))

# Read Rainfall data
Y = np.load('../Data/Data/Rainfalls_{}_small.npy'.format(years_training))
x_shape = X.shape
y_shape = Y.shape
X = X[location, :, :].reshape(1,x_shape[1],x_shape[2])
Y = Y[location, :].reshape(1,y_shape[1])
x_size = X.shape[-1] + 1
diff = x_size - 6

# Define raw filename
if extreme_case:
    if ef:
        filename_raw = 'PostSample_'+ years_training + '_cp_extreme_'+str(location)+'_ef'
    else:
        filename_raw = 'PostSample_' + years_training + '_cp_extreme_'+str(location)
else:
    if ef:
        filename_raw = 'PostSample_'+ years_training + '_cp_'+str(location)+'_ef'
    else:
        filename_raw = 'PostSample_' + years_training + '_cp_'+str(location)

# Read saved posteriors
filename = 'Posteriors/'+filename_raw + '.npz'
Theta = list(np.load(filename)['Theta'])
Z_list = list(np.load(filename)['Z'])
lhd_list = list(np.load(filename)['lhd_list'])
Theta_numpy = np.array(Theta)

n_param = Theta_numpy.shape[1]
n_days = X.shape[1]

if postconvcheck:
    ### Checking convergence of posteriors
    sampling_steps = Theta_numpy.shape[0]
    Gibbs_steps = np.arange(sampling_steps)
    print('No of posterior sampling steps: '+str(sampling_steps))

    def plot_Gibbs_samples(theta, n_burn=N_burn):
        """ Function that plots the parameters timeseries """
        """ with separates burn-in and final samples """
        for i in range(n_param):
            plt.figure(figsize=(10, 8))
            plt.plot(Gibbs_steps[10000:-n_burn], theta[10000:-n_burn,i], linestyle = '-', color = 'b')
            plt.plot(Gibbs_steps[-n_burn:], theta[-n_burn:,i], linestyle = '-', color = 'r')
            plt.title("Parameter {} for year {}".format(i+1, years_training))
            if savefig:
                plt.savefig(imlocation+"parameteres_{}_Gsteps_{}.png".format(i+1, years_training))
            else:
                plt.show()
            plt.close()
    
    #plot_Gibbs_samples(Theta_numpy, n_burn=N_burn)
    
    # PLot LHD
    plt.figure()
    plt.plot(lhd_list[-2000:])
    #plt.savefig(imlocation+'LHD.png')
    plt.show()
    plt.close()




# Prediction filename
if zknown:
    predict_filename = imlocation+filename_raw+'_predict'+'_zknown.npz'
else:
    predict_filename = imlocation+filename_raw+'_predict.npz'

# Prediction
if os.path.isfile(predict_filename):
    Y_samples = np.load(predict_filename)['Y_samples']
else:
    Y_samples = []
    for ind in range(1000):
        print(ind)
        if zknown:
            z, y, lambda_t, _, _ = model(Theta[-ind], k=x_size).simulate_known_Z(X, Z_list[-ind])
        else:
            z, y, lambda_t, _, _ = model(Theta[-ind], k=x_size).simulate_known_Z_5(X, Z_list[-ind])
        print(np.mean(y))
        while (sum(sum(np.isnan(y))) != 0) or (np.isinf(np.mean(y)) != 0):
            if zknown:
                z, y, lambda_t, _, _ = model(Theta[-ind], k=x_size).simulate_known_Z(X, Z_list[-ind])
            else:
                z, y, lambda_t, _, _ = model(Theta[-ind], k=x_size).simulate_known_Z_5(X, Z_list[-ind])
        print(np.mean(y))
        Y_samples.append(y)
#np.savez(predict_filename, Y_samples=Y_samples)


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
        plt.savefig(imlocation+"ROC_{}.png".format(years_training))
        plt.close()  
    else:
        plt.show()   


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



#Plot prediction at a Specified location
Y_samples = np.array(Y_samples)
#print(Y_samples.shape)
year_start, year_end = years_training, years_training
time = pd.date_range('{}-01-01'.format(year_start), '{}-12-31'.format(year_end), periods=n_days)
for location in range(X.shape[0]):
    Y_mean = np.mean(Y_samples[:,location,:],axis=0)
    y_median = np.median(Y_samples[:,location,:],axis=0)
    y_32 = np.quantile(Y_samples[:,location,:],axis=0, q=0.32)
    y_68 = np.quantile(Y_samples[:,location,:],axis=0, q=0.68)
    y_80 = np.quantile(Y_samples[:,location,:],axis=0, q=0.8)
    y_95 = np.quantile(Y_samples[:,location,:],axis=0, q=0.95)
    y_99 = np.quantile(Y_samples[:,location,:],axis=0, q=0.99)
    ### 1- Plot that uses the median & different quantiles, as predicted value and errors
    # time = number of days that we predict rainfall for
    plt.figure(figsize=(10, 8))
    plt.plot(time, Y[location,:], marker='+', linestyle='', color = 'black', label = 'Obs.')
    plt.plot(time, y_median, linestyle = '-', color = 'b', label='50 \%')
    plt.fill_between(time, y_32, y_68, color='red', label='32-68 \%')
    plt.fill_between(time, y_median, y_95, color='red', alpha=0.5, label='95 \%')
    plt.plot(time, y_99, linestyle = '--', color = 'r', label='99 \%')
    plt.ylim(0, np.max(Y)+1)
    plt.title("Y median - Year {}".format(years_training))
    plt.xlabel("Time")
    plt.ylabel("precipitation (mm)")
    plt.legend()
    if zknown:
        if savefig:
            plt.savefig(imlocation+filename_raw+'_precipitation_median_'+str(years_training)+'_zknown.png')
        else:
            plt.show()
    else:
        if savefig:
            plt.savefig(imlocation+filename_raw+'_precipitation_median_'+str(years_training)+'.png')
        else:
            plt.show()
    plt.close()
#
#     ### 2- Scatter plot between observables and predictions (median & 95%)
#
    plt.figure(figsize=(10, 8))
    # Plot diagonal line
    x_values = np.linspace(0, np.max(np.log10(Y[location,:]+1)), 10)
    plt.plot(x_values, x_values, linestyle = '--', color = 'black')

    # Simple linear fit
    z_median = np.polyfit(Y[location,:], y_median, 1)
    p_50 = np.poly1d(z_median)

    z_95 = np.polyfit(Y[location,:], y_95, 1)
    p_95 = np.poly1d(z_95)

    # We transform the values to Y+1, before taking the log
    plt.scatter(np.log10(Y[location,:]+1), np.log10(y_median+1), alpha=0.8, marker='x', c='r', label = '50 \%')
    plt.scatter(np.log10(Y[location,:]+1), np.log10(y_95+1), alpha=0.8, marker='x', c='b', label = '95 \%')
    plt.plot(np.log10(np.sort(Y[location,:])+1), np.log10(p_50(np.sort(Y[location,:]))+1), linestyle = '-.', alpha=0.6, color='r')
    plt.plot(np.log10(np.sort(Y[location,:])+1), np.log10(p_95(np.sort(Y[location,:]))+1), linestyle = '--', alpha=0.6, color='b')
    plt.ylim(-0.05, np.max(np.log10(Y[location,:]+1)))
    plt.title("Scatter Log[Y+1] Plot - Year {}".format(years_training))
    plt.ylabel("Predictions")
    plt.xlabel("Observations")
    plt.legend()
    if savefig:
        plt.savefig(imlocation+str(location)+"_scatter_plot_{}.png".format(years_training))
    else:
        plt.show()
    plt.close()

    ### 3- Calculate RMS error & RMS spread - Eqs. (18) & MAB p.12


    rms_error, rms_spread, mab_error, quant_reg = rms_error_spread(Y[location,:], y_median, Y_samples[:,location,:]) 

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
    
    """
    TO CHECK - Why extreme values of RMS spread?
    """
    
    # plot diagonal line
    x_values = np.linspace(0, np.max(rms_spread_mean[:-2]), 10)
    plt.plot(x_values, x_values, linestyle = '--', color = 'black')
    
    plt.plot(rms_spread_mean[:-2], rms_error_mean[:-2], 'o-', color = 'black')
    plt.xlabel("RMS spread binned")
    plt.ylabel("RMS error binned")
    if savefig:
        plt.savefig(imlocation+"spread-skill_{}.png".format(years_training))
        plt.close()  
    else:
        plt.show()
        
        
    rain_thresholds = np.arange(0, 30, 1)
    rain_probability_obs = []
    rain_probability_pred = []
    rain_probability_pred_95 = []
    rain_probability_samples = []
    rain_probability_samples_day = []
    
    for rain in rain_thresholds:
        rain_probability_obs.append(precipitation_above_x(rain, Y[location,:]))
    #     rain_probability_pred.append(precipitation_above_x(rain, y_median))
    #     rain_probability_pred_95.append(precipitation_above_x(rain, y_95))
        rain_probability_samples.append(precipitation_above_x(rain, Y_samples[:,location,:]))
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
        plt.savefig(imlocation+"precipitation_prob_comparison_{}.png".format(years_training))
    else:
        plt.show()
    plt.close()
    
    
    thresholds = np.arange(0,len(Y_samples),1)

    Tpr0, Fpr0, auc0 = ROC_curves(0, Y[location,:], Y_samples[:,location,:], thresholds)
    Tpr5, Fpr5, auc5 = ROC_curves(5, Y[location,:], Y_samples[:,location,:], thresholds)
    Tpr15, Fpr15, auc15 = ROC_curves(15, Y[location,:], Y_samples[:,location,:], thresholds)
    Tpr25, Fpr25, auc25 = ROC_curves(25, Y[location,:], Y_samples[:,location,:], thresholds)
    
    
    rains = [0, 5, 15, 25]
    ROC_plot([Tpr0, Tpr5, Tpr15, Tpr25], [Fpr0, Fpr5, Fpr15, Fpr25], [auc0, auc5, auc15, auc25], rains)
    
# ### Spread-Quantiles plot
# plt.figure(figsize=(10, 8))

# # plot diagonal line
# x_values = np.linspace(0, np.max(quant_reg_mean), 10)
# plt.plot(x_values, x_values, linestyle = '--', color = 'black')

# plt.plot(quant_reg_mean, rms_error_mean_quant, 'o-', color = 'black')
# plt.xlabel("Interquantiles binned")
# plt.ylabel("RMS error binned")
# if savefig:
#     plt.savefig(imlocation+"spread-quantiles_{}.png".format(year_training))
#     plt.close()  
# else:
#     plt.show()   