########################
### Loading Packages ###
########################

import os.path
import numpy as np
import pandas as pd
import pylab as plt


###############################
### Loading and saving data ###
###############################

def load_data(Year_training_start, Year_training_end, Year_prediction_start, Year_prediction_end,\
              location, perc, z_range, grid,\
              extreme_case=True, model_fields="Sherman"):
    
    """
    Add location in "imlocation"?
    """
    
    # "Extreme" or "Standard" model - Define posteriors' file names
    if extreme_case:   
        imlocation = 'Figure/Extreme/'
    
        filename_raw = 'PostSample_' + Year_training_start + '_' + str(int(Year_training_end)-1) +\
               '_cp_extreme_' + str(location) + '_sr' + str(int(perc*100)) +\
               '_maxZ' + str(z_range) + '_gs'
    
    else:   
        imlocation = 'Figure/Standard/'
    
        filename_raw = 'PostSample_' + Year_training_start + '_' + str(int(Year_training_end)-1) +\
                   '_cp_' + str(location) + '_sr' + str(int(perc*100)) +\
                   '_maxZ' + str(z_range) + '_gs'
    
    
    
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
    
    # Time range selection for prediction
    year_start = 1995
    year_end = 2018
    time = pd.date_range('{}-01-01'.format(year_start), '{}-12-31'.format(year_end), periods=X.shape[1])
    

    # We first load information for all times and then cut to the period of interest
    # We want to load the MF & Rainfalss for the period we want to predict
    boolean_time = (time>=Year_prediction_start) & (time<Year_prediction_end)
    
    X = X[:, boolean_time, :]
    Y = Y[:, boolean_time]
    
    
    # Re-configure fields shapes, to specify location
    # From (#Loc, #Days, #Model_Fields) -> (Loc, #Days, #Model_Fields)
    x_shape = X.shape
    y_shape = Y.shape
    X = X[location, :, :].reshape(1,x_shape[1],x_shape[2])
    Y = Y[location, :].reshape(1,y_shape[1])

    
    # Definitions to help with prediction model conventions
    x_size = X.shape[-1] + 1
    
    
    # Read saved posteriors (from the training data set)
    
    # Theta - parameters of the model - (#Gibbs_Steps, #Parameters)
    # Z_list - rain predictions - (#Gibbs_Steps, Loc, #Days_training)
    # lhd_list - evolution of log likelihood - (#Gibbs_Steps-1)
    
    filename = 'Posteriors/'+filename_raw + '.npz'
    Theta = np.load(filename)['Theta']
    Z_list = np.load(filename)['Z']
    lhd_list = np.load(filename)['lhd_list']

    # Save the number of parameters in the model and the number of days training
    n_param = Theta.shape[1]
    n_days = X.shape[1]
    print(n_days)

    return X, Y, Theta, Z_list, lhd_list, x_size, n_days, n_param, imlocation, filename_raw

  

def postconvcheck(n_param, n_burn, theta, lhd_list, Year_training_start, Year_training_end,\
                  imlocation, savefig=False):
    """ Checking convergence of posteriors """
    sampling_steps = theta.shape[0]
    Gibbs_steps = np.arange(sampling_steps)
    print('No of posterior sampling steps: '+str(sampling_steps))

    # Plot parameters timeseries
    # separating burn-in & final samples
    for i in range(n_param):

        plt.figure(figsize=(10, 8))
        plt.plot(Gibbs_steps[:n_burn], theta[:n_burn,i], linestyle = '-', color = 'b')
        plt.plot(Gibbs_steps[n_burn:], theta[n_burn:,i], linestyle = '-', color = 'g')
        plt.axvline(x=n_burn, color='r', linestyle = '--')
        plt.xlabel('Sampling steps')
        plt.title("Parameter {} for training years {}-{}".format(i+1, Year_training_start, Year_training_end))
        if savefig:
            plt.savefig(imlocation+"parameteres_{}_Gsteps_{}-{}.png".format(i+1, Year_training_start,\
                                                                            Year_training_end))
        else:
            plt.show()
        plt.close()
   
    # Plot log LHD
    plt.figure()
    plt.plot(lhd_list[:])
    plt.axvline(x=n_burn, color='r', linestyle = '--')
    plt.xlabel('Sampling steps')
    plt.ylabel('Log LHD')
    if savefig:
        plt.savefig(imlocation+'LHD.png')
    else:
        plt.show()
        plt.close()





# Prediction

def model_prediction(Theta, Z_list, X, x_size, n_burn, imlocation, filename_raw, zknown=True, extreme_case=True):
    # Prediction filename
    if zknown:
        predict_filename = imlocation+filename_raw+'_predict'+'_zknown.npz'
    else:
        predict_filename = imlocation+filename_raw+'_predict.npz'

    if extreme_case:
        from timeseries_cp_extreme import cptimeseries_extreme as model
    else:
        from timeseries_cp import cptimeseries as model
    
    ####
    #### CHECK WITH BURN-IN HERE 
    ####
    if os.path.isfile(predict_filename):
        Y_samples = np.load(predict_filename)['Y_samples']
    else:
        Y_samples = []
        N_samples = len(Theta) - n_burn
        for ind in range(N_samples):
            print("Sample: ", ind)
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
            print()
            Y_samples.append(y)
    #np.savez(predict_filename, Y_samples=Y_samples)
    
    return np.array(Y_samples)


####
### RMS Spread Error
###


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


def ROC_plot(tpr_array, fpr_array, auc_array, rains,\
             Year_prediction_start, Year_prediction_end, imlocation,\
             savefig=False): 
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
        plt.savefig(imlocation+"ROC_{}-{}.png".format(Year_prediction_start, Year_prediction_end))
        plt.close()  
    else:
        plt.show()   


###
### Check this function
###

def precipitation_above_x(Y, rain_thres, rainvalues, all_days=True):
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



def predictions_plot(Y_samples, Year_prediction_start, Year_prediction_end, n_days, X, Y,\
                     imlocation, filename_raw,\
                     zknown=True, savefig=False):
    #Plot prediction at a Specified location

    time_plot = pd.date_range('{}-01-01'.format(Year_prediction_start), '{}-12-31'.format(Year_prediction_end), periods=n_days)
    
    #Y_mean = np.mean(Y_samples[:,0,:],axis=0)
    y_median = np.median(Y_samples[:,0,:],axis=0)
    y_32 = np.quantile(Y_samples[:,0,:],axis=0, q=0.32)
    y_68 = np.quantile(Y_samples[:,0,:],axis=0, q=0.68)
    #y_80 = np.quantile(Y_samples[:,0,:],axis=0, q=0.8)
    y_95 = np.quantile(Y_samples[:,0,:],axis=0, q=0.95)
    y_99 = np.quantile(Y_samples[:,0,:],axis=0, q=0.99)
    ### 1- Plot that uses the median & different quantiles, as predicted value and errors
    # time = number of days that we predict rainfall for
    plt.figure(figsize=(10, 8))
    plt.plot(time_plot, Y[0,:], marker='+', linestyle='', color = 'black', label = 'Obs.')
    plt.plot(time_plot, y_median, linestyle = '-', color = 'b', label='50 \%')
    plt.fill_between(time_plot, y_32, y_68, color='red', label='32-68 \%')
    plt.fill_between(time_plot, y_median, y_95, color='red', alpha=0.5, label='50-95 \%')
    plt.plot(time_plot, y_99, linestyle = '--', color = 'r', label='99 \%')
    plt.ylim(0, np.max(Y)+1)
    plt.title("Y median - Years {}-{}".format(Year_prediction_start, Year_prediction_end))
    plt.xlabel("Time")
    plt.ylabel("precipitation (mm)")
    plt.legend()
    if zknown:
        if savefig:
            plt.savefig(imlocation+filename_raw+'_precipitation_median_'+Year_prediction_start+\
                        '_'+Year_prediction_end+'_zknown.png')
        else:
            plt.show()
    else:
        if savefig:
            plt.savefig(imlocation+filename_raw+'_precipitation_median_'+Year_prediction_start+\
                        '_'+Year_prediction_end+'.png')
        else:
            plt.show()
    plt.close()



### 2- Scatter plot between observables and predictions (median & 95%)

 
def scatter_plot_fit(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation, savefig=False):
    plt.figure(figsize=(10, 8))
    # Plot diagonal line
    x_values = np.linspace(0, np.max(np.log10(Y[0,:]+1)), 10)
    plt.plot(x_values, x_values, linestyle = '--', color = 'black')

    # Simple linear fit
    y_median = np.median(Y_samples[:,0,:],axis=0)
    z_median = np.polyfit(Y[0,:], y_median, 1)
    p_50 = np.poly1d(z_median)

    y_95 = np.quantile(Y_samples[:,0,:],axis=0, q=0.95)
    z_95 = np.polyfit(Y[0,:], y_95, 1)
    p_95 = np.poly1d(z_95)

    # We transform the values to Y+1, before taking the log
    plt.scatter(np.log10(Y[0,:]+1), np.log10(y_median+1), alpha=0.8, marker='x', c='r', label = '50 \%')
    plt.scatter(np.log10(Y[0,:]+1), np.log10(y_95+1), alpha=0.8, marker='x', c='b', label = '95 \%')
    plt.plot(np.log10(np.sort(Y[0,:])+1), np.log10(p_50(np.sort(Y[0,:]))+1), linestyle = '-.', alpha=0.6, color='r')
    plt.plot(np.log10(np.sort(Y[0,:])+1), np.log10(p_95(np.sort(Y[0,:]))+1), linestyle = '--', alpha=0.6, color='b')
    plt.ylim(-0.05, np.max(np.log10(Y[0,:]+1)))
    plt.title("Scatter Log[Y+1] Plot - Years {}-{}".format(Year_prediction_start, Year_prediction_end))
    plt.ylabel("Predictions")
    plt.xlabel("Observations")
    plt.legend()
    if savefig:
        plt.savefig(imlocation+"_scatter_plot_{}-{}.png".format(Year_prediction_start, Year_prediction_end))
    else:
        plt.show()
    plt.close()

#     ### 3- Calculate RMS error & RMS spread - Eqs. (18) & MAB p.12

# def spread_skill():

#     rms_error, rms_spread, mab_error, quant_reg = rms_error_spread(Y[location,:], y_median, Y_samples[:,location,:]) 

#     "With bins"
#     n_bins = 16

#     rms_spread = rms_spread.reshape(len(rms_spread),1)
#     rms_error = rms_error.reshape(len(rms_error),1)
#     mab_error = mab_error.reshape(len(mab_error),1)
#     quant_reg = quant_reg.reshape(len(quant_reg),1)
    
#     # Spread-Skill curve
#     joint_rms = np.concatenate((rms_spread, rms_error), axis=1)
#     joint_rms_sorted = joint_rms[joint_rms[:,0].argsort()]
    
#     rms_spread_binned = np.array(np.array_split(joint_rms_sorted[:,0], n_bins), dtype=object)
#     rms_error_binned = np.array(np.array_split(joint_rms_sorted[:,1], n_bins), dtype=object)
    
#     # Spread-Quantile curve
#     joint_rms_quant = np.concatenate((quant_reg, rms_error), axis=1)
#     joint_rms_sorted_quant = joint_rms_quant[joint_rms_quant[:,0].argsort()]
    
#     quant_reg_binned = np.array(np.array_split(joint_rms_sorted_quant[:,0], n_bins), dtype=object)
#     rms_error_binned_quant = np.array(np.array_split(joint_rms_sorted_quant[:,1], n_bins), dtype=object)
    
#     # MAB error binning
#     mab_error_binned = np.array(np.array_split(mab_error, n_bins), dtype=object)
    
#     # Vectorise the "mean" function, to calculate mean per bin
#     v_mean = np.vectorize(np.mean)
    
#     rms_spread_mean = v_mean(rms_spread_binned)
#     rms_error_mean = v_mean(rms_error_binned)
    
#     rms_error_mean_quant = v_mean(rms_error_binned_quant)
#     quant_reg_mean = v_mean(quant_reg_binned)
    
#     mab_error_mean = v_mean(mab_error_binned)
    
#     ### Spread-Skill plot
#     plt.figure(figsize=(10, 8))
    
#     """
#     TO CHECK - Why extreme values of RMS spread?
#     """
    
#     # plot diagonal line
#     x_values = np.linspace(0, np.max(rms_spread_mean[:-2]), 10)
#     plt.plot(x_values, x_values, linestyle = '--', color = 'black')
    
#     plt.plot(rms_spread_mean[:-2], rms_error_mean[:-2], 'o-', color = 'black')
#     plt.xlabel("RMS spread binned")
#     plt.ylabel("RMS error binned")
#     if savefig:
#         plt.savefig(imlocation+"spread-skill_{}-{}.png".format(Year_training_start, Year_training_end))
#         plt.close()  
#     else:
#         plt.show()

        

def rain_probability(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation, savefig=False):        
    rain_thresholds = np.arange(0, 30, 1)
    rain_probability_obs = []
    rain_probability_pred = []
    rain_probability_pred_95 = []
    rain_probability_samples = []
    rain_probability_samples_day = []
    
    for rain in rain_thresholds:
        rain_probability_obs.append(precipitation_above_x(Y, rain, Y[0,:]))
    #     rain_probability_pred.append(precipitation_above_x(rain, y_median))
    #     rain_probability_pred_95.append(precipitation_above_x(rain, y_95))
        rain_probability_samples.append(precipitation_above_x(Y, rain, Y_samples[:,0,:]))
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
        plt.savefig(imlocation+"precipitation_prob_comparison_{}-{}.png".format(Year_prediction_start, Year_prediction_end))
    else:
        plt.show()
    plt.close()
    
    
def ROC_plottting(Y_samples, Y, Year_prediction_start, Year_prediction_end, imlocation):
    thresholds = np.arange(0,len(Y_samples),1)

    Tpr0, Fpr0, auc0 = ROC_curves(0, Y[0,:], Y_samples[:,0,:], thresholds)
    Tpr5, Fpr5, auc5 = ROC_curves(5, Y[0,:], Y_samples[:,0,:], thresholds)
    Tpr15, Fpr15, auc15 = ROC_curves(15, Y[0,:], Y_samples[:,0,:], thresholds)
    Tpr25, Fpr25, auc25 = ROC_curves(25, Y[0,:], Y_samples[:,0,:], thresholds)
    
    
    rains = [0, 5, 15, 25]
    ROC_plot([Tpr0, Tpr5, Tpr15, Tpr25], [Fpr0, Fpr5, Fpr15, Fpr25], [auc0, auc5, auc15, auc25], rains,\
             Year_prediction_start, Year_prediction_end, imlocation)
    
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
    plt.show()   