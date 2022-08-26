import numpy as np
import Prediction as pred


###################
### Run Options ###
###################

# extreme_case = True
# z_known = False
# postconvcheck = False
# savefig = False

"""
TO ADD - run only for specific location
"""



def prediction_visualisation(Years_training, Year_prediction_start, Year_prediction_end,\
                             N_samples = 1000, N_locations=1, perc=0.1, z_range=9,\
                             grid="small", zknown=False, savefig=False):
    
    
    #####################
    ### Set Variables ###
    #####################
    
    Years_training = str(Years_training)
    Year_prediction_start = str(Year_prediction_start)
    Year_prediction_end = str(Year_prediction_end)
    
    Loc_list = np.arange(N_locations)
    
    #######################################################
    # READ the Y_samples (Predictions) & Y (Observations) #
    #######################################################
    
    if zknown:
        filename_samples = 'PostSamples_N' + str(N_samples) + '_' + Year_prediction_start + '_' + Year_prediction_end +\
                               '_cp_locations_sr' + str(int(perc*100)) +\
                               '_maxZ' + str(z_range) + '_grid_' + grid +\
                               '_yt' + Years_training + '_zknown_' 
        
        
    else:
        filename_samples = 'PostSamples_N' + str(N_samples) + '_' + Year_prediction_start + '_' + Year_prediction_end +\
                           '_cp_locations_sr' + str(int(perc*100)) +\
                           '_maxZ' + str(z_range) + '_grid_' + grid +\
                           '_yt' + Years_training
    
    
    Y_file = "Posterior_samples/"+filename_samples+".npz"
    
    Y_samples = np.load(Y_file)['Y_samples']
    Y = np.load(Y_file)['Y']
    n_days = len(Y.T)
                       
    #########################
    # Make prediction plots #
    #########################
    
    for loc in Loc_list:
        print()
        print("We're now plotting location {}".format(loc))
        
        imlocation = 'Vis_samples/' + grid + '_new/loc' + str(loc) +"/"

        pred.predictions_plot(Y_samples[loc], Year_prediction_start, Year_prediction_end, n_days,\
                              Y[loc], loc, imlocation, zknown=zknown, savefig=savefig)
        pred.rain_probability(Y_samples[loc], Y[loc], Year_prediction_start, Year_prediction_end, loc,\
                              imlocation, savefig)
        pred.scatter_plot_fit(Y_samples[loc], Y[loc], Year_prediction_start, Year_prediction_end, loc,\
                              imlocation, savefig)
        pred.ROC_plottting(Y_samples[loc], Y[loc], Year_prediction_start, Year_prediction_end, loc,\
                            imlocation, savefig)
        pred.calibration_error(Y_samples[loc], Y[loc], Year_prediction_start, Year_prediction_end, loc,\
                            imlocation, savefig)
            
    return Y_samples, Y

# Y_samples, Y = prediction_visualisation(1999, 2000, N_locations=1, savefig=False)


# from scipy import signal
# import matplotlib.pyplot as plt

# predicted = np.mean(np.median(Y_samples, axis=2), axis=1).reshape((3,3))
# observed = np.mean(Y, axis=1).reshape((3,3))

# corr = signal.correlate2d(observed, predicted, boundary='symm', mode='same')

# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

# pos = ax1.imshow(observed, cmap='Blues', interpolation='none')
# ax1.title.set_text('Observed')
# fig.colorbar(pos, ax=ax1)

# neg = ax2.imshow(predicted, cmap='Blues', interpolation='none')
# ax2.title.set_text('Predicted')
# fig.colorbar(neg, ax=ax2)

# pos_neg_clipped = ax3.imshow(corr, cmap='Blues', interpolation='none')
# ax3.title.set_text('Correlation')

# cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
# cbar.minorticks_on()
# plt.show()