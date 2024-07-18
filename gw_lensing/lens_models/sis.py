import numpy as np
from astropy.cosmology import Planck18 as cosmo

from ..utils.constants import *
from ..utils import lensutils

"""Image properties"""
#Magnification
def mu_plus(y):
    """Magnification for the positive image
    
    Parameters
    ----------
    y : float
        modulus of dimensionless source position
    """

    return 1 + 1./y
def mu_minus(y):
    """Magnification for the negative image

    Parameters
    ----------
    y : float
        modulus of dimensionless source position
    """
    return 1 - 1./y

#Time delay
def T_plus(y):
    """Time delay for the positive image

    Parameters
    ----------
    y : float
        modulus of dimensionless source position
    """
    return -1./2 - y
def T_minus(y):
    """Time delay for the negative image

    Parameters
    ----------
    y : float
        modulus of dimensionless source position
    """
    return -1./2 + y
def DeltaT(y):
    """Time delay difference between the two images

    Parameters
    ----------
    y : float
        modulus of dimensionless source position
    """
    return 2*y

"""Velocity dispersion"""
def sigma_v(M,z): #m/s
    #M200 in Msun
    rho_c_g_cm3 = cosmo.critical_density(z).value# g/cm^3
    rho_c = rho_c_g_cm3*1e-3*1e6 # kg/m^3
    rho_200 = 200*rho_c
    return np.power(np.sqrt(np.pi*np.power(Gnewton,3)*rho_200/6)*M*MSUN,1./3)

"""Einstein radius"""
def theta_E(sigma,z_L,z_S):
    #sigma in m/s
    #Clight in m/s
    DS = cosmo.angular_diameter_distance(z_S).value #Mpc
    DLS = cosmo.angular_diameter_distance_z1z2(z_L,z_S).value #Mpc
    return 4*np.pi* np.power(sigma / Clight,2.) * DLS / DS #rad

"""Cross section"""
#Multiple image cross-section: area where there are 2 images determined theta < theta_E
def sigma_two(M,z_L,z_S): #cross section
    sigma = sigma_v(M,z_L) #velocity dispersion
    return np.pi * theta_E(sigma,z_L,z_S)**2


#Magnification cross section: mu > mu0
def sigma_mu(M,z_L,z_S,mu0): #cross section
    sigma = sigma_v(M,z_L) #velocity dispersion
    return 2. * np.pi * np.power(theta_E(sigma,z_L,z_S),2.) * (mu0**2 + 1) / np.power((mu0**2 - 1),2.)

"""Time delay"""
def t_delay(y,ML,zL,zS): 
    """Time delay in years
    
    Parameters
    ----------
    y : float
        modulus of dimensionless source position
    ML : float
        lens mass in solar masses
    zL : float
        redshift of the lens
    zS : float
        redshift of the source
    """
    sigma = sigma_v(ML,zL)
    return lensutils.t_distance(zL,zS)*DeltaT(y)*theta_E(sigma,zL,zS)**2 / YEAR