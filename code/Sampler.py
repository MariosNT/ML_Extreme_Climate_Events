import copy

import numpy as np
from scipy.stats import gamma, multivariate_normal

def EllipticalSliceSampling(LHD, n=1000, Mean=np.zeros(shape=(29,)), Sigma=np.identity(29), f_0=np.zeros(shape=(29,))):
##### Implemented using the Murray et. al. Elliptical Slice Sampling (http://proceedings.mlr.press/v9/murray10a/murray10a.pdf)
# Figure 2 ####
    Samples = []

    # Store initial value
    f_prime = f_0
    Samples.append(f_prime)
    # print("n", n)
    # print("Mean", Mean)
    # print("Sigma", Sigma)
    # print("f_0", f_0)

    for ind in range(n):
        f = f_prime
        original_f = copy.deepcopy(f_prime)
        #sample from prior
        nu = Mean + multivariate_normal.rvs(cov=Sigma, size=1)

        #Compute and Define loglikelihood threshold
        u = np.random.random(size=(1,))
        log_y = LHD(f) + np.log(u)

        ###### Initial proposal #####
        #Draw an initial proposal, also defining a bracket:
        theta_ellipse = np.random.random(size=(1,)) * (2*np.pi)
        theta_ellipse_min = theta_ellipse - (2*np.pi)
        theta_ellipse_max = theta_ellipse
        # new proposal
        f_prime = f * np.cos(theta_ellipse) + nu * np.sin(theta_ellipse)
        lhd_f_prime = LHD(f_prime)
        ##### While loop until gets accepted ####
        # Accept and Reject Step
        count = 0
        while lhd_f_prime <= log_y and count < 50:
            count = count + 1
            if theta_ellipse < 0:
                theta_ellipse_min = theta_ellipse
            else:
                theta_ellipse_max = theta_ellipse
            theta_ellipse = theta_ellipse_min + (theta_ellipse_max - theta_ellipse_min) * np.random.random(size=(1,))
            f_prime = f * np.cos(theta_ellipse) + nu * np.sin(theta_ellipse)
            lhd_f_prime = LHD(f_prime)
        print('final count: '+str(count))
        if count == 50:
            f_prime = original_f
        # Store the updated sample
        Samples.append(f_prime)
    return Samples, lhd_f_prime