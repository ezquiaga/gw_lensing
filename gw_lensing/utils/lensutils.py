import numpy as np
from scipy.special import gamma
from astropy.cosmology import Planck18 as cosmo

from ..utils.constants import *

def t_distance(zL,zS):
    """Time delay distance in seconds

    Parameters
    ----------
    zL : float
        redshift of the lens
    zS : float
        redshift of the source
    """
    DL = cosmo.angular_diameter_distance(zL).value #Mpc
    DS = cosmo.angular_diameter_distance(zS).value #Mpc
    DLS = cosmo.angular_diameter_distance_z1z2(zL,zS).value #Mpc
    return (1.+zL)*(DS*DL/DLS) * (MPC / Clight) #s

"""Velocity dispersion distribution"""
def dpdsigma(sigma,sigmaS,alpha,beta):
    return np.power(sigma/sigmaS,alpha)*np.exp(-np.power(sigma/sigmaS,beta))*beta/gamma(alpha/beta)/sigma

"""Axis ratio distribution"""
def variance_q(sigma):
    # sigma in km/s
    # Fit by Eq (4) of https://arxiv.org/pdf/1507.02657.pdf
    return 0.38 + 5.72e-4* sigma

# Rayleight distribution for axis ratio q
def g(x,sigma):
    # x = 1 - q
    # sigma in km/s
    # Eq (4) of https://arxiv.org/pdf/1507.02657.pdf, also used in (A.10) of https://arxiv.org/pdf/1807.07062.pdf
    s = variance_q(sigma)
    return (x / np.power(s,2))*np.exp(-0.5*np.power(x/s,2))

# Rayleight distribution normalization
def g_norm(qmin,sigma):
    s = variance_q(sigma)
    return 1 - np.exp(-0.5 * np.power((1 - qmin)/s,2))

#Rayleight distribution is truncated at q_min = 0.2
def g_truncated(x,qmin,sigma):
    return g(x,sigma) / g_norm(qmin,sigma)