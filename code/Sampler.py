import numpy as np
from scipy.stats import gamma, multivariate_normal

##### Lets first sample theta using Elliptical Slice Sampling ####

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
        #sample from prior
        nu = Mean + multivariate_normal.rvs(cov=Sigma, size=1)

        #Compute and Define loglikelihood threshold
        u = np.random.random(size=(1,))
        log_y = LHD(f)[0] + np.log(u)
        #log_y = LHD(f) + np.log(u)

        ###### Initial proposal #####
        #Draw an initial proposal, also defining a bracket:
        theta_ellipse = np.random.random(size=(1,)) * (2*np.pi)
        theta_ellipse_min = theta_ellipse - (2*np.pi)
        theta_ellipse_max = theta_ellipse
        # new proposal
        f_prime = f * np.cos(theta_ellipse) + nu * np.sin(theta_ellipse)

        ##### While loop until gets accepted ####
        # Accept and Reject Step
        while LHD(f_prime)[0] <= log_y:
        #while LHD(f_prime) <= log_y:
            if theta_ellipse < 0:
                theta_ellipse_min = theta_ellipse
            else:
                theta_ellipse_max = theta_ellipse
            theta_ellipse = theta_ellipse_min + (theta_ellipse_max - theta_ellipse_min) * np.random.random(size=(1,))
            f_prime = f * np.cos(theta_ellipse) + nu * np.sin(theta_ellipse)
        # Store the updated sample
        Samples.append(f_prime)
    return Samples