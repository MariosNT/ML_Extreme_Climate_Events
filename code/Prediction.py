import numpy as np
import copy
import os.path
import pandas as pd
import pylab as plt
from Sampler import EllipticalSliceSampling
from scipy.stats import gamma, multivariate_normal, poisson
# For purely serial computing
from parallel.backends import BackendDummy as Backend
# For MPI parallelized computing
#from parallel.backends import BackendMPI as Backend
#backend = Backend()
# number of steps Gibbs we want to use
extreme_case = True
ef = True
zknown = True
postconvcheck = True

if extreme_case:
    from timeseries_cp_extreme import cptimeseries_extreme as model
else:
    from timeseries_cp import cptimeseries as model

if extreme_case:
    imlocation = 'Figure/Extreme/'
else:
    imlocation = 'Figure/Standard/'

year_training = '1999'
location = 4

# Read Model fields
if ef:
    X = np.load('../Data/Data/model_fields_multiple_1999_small_ef.npy')
else:
    X = np.load('../Data/Data/model_fields_multiple_1999_small.npy')
Y = np.load('../Data/Data/Rainfalls_1999_small.npy')
x_shape = X.shape
y_shape = Y.shape
X = X[location, :, :].reshape(1,x_shape[1],x_shape[2])
Y = Y[location, :].reshape(1,y_shape[1])
x_size = X.shape[-1] + 1
diff = x_size - 6

# Define raw filename
if extreme_case:
    if ef:
        filename_raw = 'PostSample_'+year_training + '_cp_extreme_'+str(location)+'_ef'
    else:
        filename_raw = 'PostSample_' + year_training + '_cp_extreme_'+str(location)
else:
    if ef:
        filename_raw = 'PostSample_'+year_training + '_cp_'+str(location)+'_ef'
    else:
        filename_raw = 'PostSample_' + year_training + '_cp_'+str(location)

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

    def plot_Gibbs_samples(theta, n_burn=5000):
        """ Function that plots the parameters timeseries """
        """ with separates burn-in and final samples """
        for i in range(n_param):
            plt.figure(figsize=(10, 8))
            plt.plot(Gibbs_steps[10000:-n_burn], theta[10000:-n_burn,i], linestyle = '-', color = 'b')
            plt.plot(Gibbs_steps[-n_burn:], theta[-n_burn:,i], linestyle = '-', color = 'r')
            plt.title("Parameter {} for year {}".format(i+1, year_training))
            plt.savefig(imlocation+"parameteres_{}_Gsteps_{}.png".format(i+1, year_training))
            plt.close()
    plot_Gibbs_samples(Theta_numpy, n_burn=1000)
    # PLot LHD
    plt.figure()
    plt.plot(lhd_list[-2000:])
    plt.savefig(imlocation+'LHD.png')
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
np.savez(predict_filename, Y_samples=Y_samples)

#Plot prediction at a Specified location
Y_samples = np.array(Y_samples)
print(Y_samples.shape)
year_start, year_end = year_training, year_training
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
    plt.title("Y median - Year {}".format(year_training))
    plt.xlabel("Time")
    plt.ylabel("precipitation (mm)")
    plt.legend()
    if zknown:
        plt.savefig(imlocation+filename_raw+'_precipitation_median_'+str(year_training)+'_zknown.png')
    else:
        plt.savefig(imlocation+filename_raw+'_precipitation_median_'+str(year_training)+'.png')
    plt.close()
#
#     ### 2- Scatter plot between observables and predictions (median & 95%)
#
#     plt.figure(figsize=(10, 8))
#     # Plot diagonal line
#     x_values = np.linspace(0, np.max(np.log10(Y[location,:]+1)), 10)
#     plt.plot(x_values, x_values, linestyle = '--', color = 'black')
#
#     # Simple linear fit
#     z_median = np.polyfit(Y[location,:], y_median, 1)
#     p_50 = np.poly1d(z_median)
#
#     z_95 = np.polyfit(Y[location,:], y_95, 1)
#     p_95 = np.poly1d(z_95)
#
#     # We transform the values to Y+1, before taking the log
#     plt.scatter(np.log10(Y[location,:]+1), np.log10(y_median+1), alpha=0.8, marker='x', c='r', label = '50 \%')
#     plt.scatter(np.log10(Y[location,:]+1), np.log10(y_95+1), alpha=0.8, marker='x', c='b', label = '95 \%')
#     plt.plot(np.log10(np.sort(Y[location,:])+1), np.log10(p_50(np.sort(Y[location,:]))+1), linestyle = '-.', alpha=0.6, color='r')
#     plt.plot(np.log10(np.sort(Y[location,:])+1), np.log10(p_95(np.sort(Y[location,:]))+1), linestyle = '--', alpha=0.6, color='b')
#     plt.ylim(-0.05, np.max(np.log10(Y[location,:]+1)))
#     plt.title("Scatter Log[Y+1] Plot - Year {}".format(year_training))
#     plt.ylabel("Predictions")
#     plt.xlabel("Observations")
#     plt.legend()
#     plt.savefig(imlocation+str(location)+"_scatter_plot_{}.png".format(year_training))
#     plt.close()
#
# ### 2- Scatter plot between observables and predictions (median & 95%)
#
# Y_all = Y.flatten()
# print(Y_all.shape)
# plt.figure(figsize=(10, 8))
# # Plot diagonal line
# x_values = np.linspace(0, np.max(np.log10(Y_all+1)), 10)
# plt.plot(x_values, x_values, linestyle = '--', color = 'black')
#
# # Simple linear fit
# y_median_all = np.median(Y_samples[:,:,:],axis=0).flatten()
# z_median = np.polyfit(Y_all, y_median_all, 1)
# p_50 = np.poly1d(z_median)
#
# y_95_all = np.quantile(Y_samples[:,:,:],axis=0, q=0.95).flatten()
# z_95 = np.polyfit(Y_all, y_95_all, 1)
# p_95 = np.poly1d(z_95)
#
# # We transform the values to Y+1, before taking the log
# plt.scatter(np.log10(Y_all+1), np.log10(y_median_all+1), alpha=0.8, marker='x', c='r', label = '50 \%')
# plt.scatter(np.log10(Y_all+1), np.log10(y_95_all+1), alpha=0.8, marker='x', c='b', label = '95 \%')
# plt.plot(np.log10(np.sort(Y_all)+1), np.log10(p_50(np.sort(Y_all))+1), linestyle = '-.', alpha=0.6, color='r')
# plt.plot(np.log10(np.sort(Y_all)+1), np.log10(p_95(np.sort(Y_all))+1), linestyle = '--', alpha=0.6, color='b')
# plt.ylim(-0.05, np.max(np.log10(Y_all+1)))
# plt.title("Scatter Log[Y+1] Plot - Year {}".format(year_training))
# plt.ylabel("Predictions")
# plt.xlabel("Observations")
# plt.legend()
# plt.savefig(imlocation+"All_location_scatter_plot_{}.png".format(year_training))
# plt.close()