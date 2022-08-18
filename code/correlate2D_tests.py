"""
Simple script to test measures of quantifying correlation between two images
"""

#######################
### Upload Packages ###
#######################

import numpy as np
from numpy.random import default_rng
rng = default_rng()

from scipy import signal
import matplotlib.pyplot as plt


########################
### Create test data ###
########################

archive_tests = True

Image_size = 24
picture_1 = rng.integers(1, 10, size = (Image_size, Image_size))
picture_2 = np.copy(picture_1)

picture_3 = np.copy(picture_1)
picture_3[Image_size//2:, Image_size//2:] =\
          rng.integers(1, 10, size = (Image_size//2, Image_size//2))


#############
### Tests ###
#############



# Test 1 #
# We find the difference pixel by pixel and plot the percentage difference

def image_comparison(pic1, pic2):
    comparison = (pic1 - pic2)/pic1*100

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
    
    pos = ax1.imshow(pic1, cmap='Blues', interpolation='none')
    ax1.title.set_text('Observed')
    fig.colorbar(pos, ax=ax1)
    
    neg = ax2.imshow(pic2, cmap='Blues', interpolation='none')
    ax2.title.set_text('Predicted')
    fig.colorbar(neg, ax=ax2)
    
    pos_neg_clipped = ax3.imshow(comparison, cmap='Blues', interpolation='none')
    ax3.title.set_text('Difference ratio (%)')
    
    cbar = fig.colorbar(pos_neg_clipped, ax=ax3)
    cbar.minorticks_on()
    plt.show()

image_comparison(picture_1, picture_2)
image_comparison(picture_1, picture_3)


# Test 2 #
# We calculate the Pearson correlation coefficient between the images pixel by pixel

def image_pearson(pic1, pic2):
    picture1_normed = (pic1-pic1.mean())/np.sqrt(np.sum((pic1-pic1.mean())**2))
    picture2_normed = (pic2-pic2.mean())/np.sqrt(np.sum((pic2-pic2.mean())**2))
    Rp = np.sum(picture2_normed*picture1_normed)
    
    print("Pearson coefficient is:", Rp)

image_pearson(picture_1, picture_2)
image_pearson(picture_1, picture_3)



if archive_tests:
    
    # Test 1 #
    # Translates the full image and compares (normalised)
   
    corr = signal.correlate2d(picture_1-picture_1.mean(),\
                              picture_2-picture_2.mean(),\
                              boundary='symm', mode='same')
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
    
    pos = ax1.imshow(picture_1, cmap='Blues', interpolation='none')
    ax1.title.set_text('Observed')
    fig.colorbar(pos, ax=ax1)
    
    neg = ax2.imshow(picture_2, cmap='Blues', interpolation='none')
    ax2.title.set_text('Predicted')
    fig.colorbar(neg, ax=ax2)
    
    pos_neg_clipped = ax3.imshow(corr, cmap='Blues', interpolation='none')
    ax3.title.set_text('Correlation 2D')
    
    cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
    cbar.minorticks_on()
    plt.show()
    
    
    # Test 2 #
    # Translates the full image and compares (un-normalised)
    
    corr = signal.correlate(picture_1, picture_2)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
    
    pos = ax1.imshow(picture_1, cmap='Blues', interpolation='none')
    ax1.title.set_text('Observed')
    fig.colorbar(pos, ax=ax1)
    
    neg = ax2.imshow(picture_2, cmap='Blues', interpolation='none')
    ax2.title.set_text('Predicted')
    fig.colorbar(neg, ax=ax2)
    
    pos_neg_clipped = ax3.imshow(corr, cmap='Blues', interpolation='none')
    ax3.title.set_text('Correlation')
    
    cbar = fig.colorbar(pos_neg_clipped, ax=ax3, extend='both')
    cbar.minorticks_on()
    plt.show()
    
    
    # Test 3 #
    # Simple example of how correlate2d works:
    # moves 'temp' around, till it finds a matching pattern in img
    
    from imageio import imread

    img = imread('https://i.stack.imgur.com/JL2LW.png', pilmode='L')
    temp = imread('https://i.stack.imgur.com/UIUzJ.png', pilmode='L')
    
    corr = signal.correlate2d(img - img.mean(), 
                          temp - temp.mean(),
                          boundary='symm',
                          mode='full')
    
    # coordinates where there is a maximum correlation
    max_coords = np.where(corr == np.max(corr))
    
    plt.plot(max_coords[1], max_coords[0],'c*', markersize=5)
    plt.imshow(corr, cmap='hot')
    