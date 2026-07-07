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

"""Generic image-plane helpers (lens-model independent)"""
def classify_image_from_hessian(psi_xx, psi_yy, psi_xy):
    """
    Generic (lens-model-independent) Morse-index classification of a lensed
    image from the second derivatives of the LENS potential at the image
    position. Builds the Hessian of the Fermat potential, H = I - hessian,
    and applies

        det(H) < 0                -> 'saddle'
        det(H) > 0 and tr(H) > 0  -> 'min'
        det(H) > 0 and tr(H) < 0  -> 'max'

    Parameters
    ----------
    psi_xx, psi_yy, psi_xy : float or array-like
        Second derivatives of the lens potential at the image position(s).

    Returns
    -------
    image_type : str or ndarray of str
        'min', 'saddle' or 'max' ('degenerate' exactly on the critical
        curve, det(H) == 0, a measure-zero case).
    """
    det = (1. - psi_xx) * (1. - psi_yy) - psi_xy**2
    tr = 2. - psi_xx - psi_yy
    det = np.asarray(det)
    tr = np.asarray(tr)
    out = np.where(det < 0., 'saddle',
                   np.where(det > 0., np.where(tr > 0., 'min', 'max'),
                            'degenerate'))
    return out if out.ndim else str(out)

def shoelace_area(x, y):
    """
    Area enclosed by the closed
    polygon with vertices (x, y) via the shoelace formula (the last vertex
    is implicitly joined back to the first; orientation does not matter,
    the absolute value is taken). Useful e.g. for caustic/cut areas of any
    lens model from a parametric contour.

    Parameters
    ----------
    x, y : array-like
        Vertex coordinates of the closed polygon.

    Returns
    -------
    area : float
    """
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))