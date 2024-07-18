import numpy as np
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