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

"""Virial radius"""
def R_vir(M,z): #meters
    """Virial (R200c) radius of a halo of mass M at redshift z.

    Uses the same rho_200 = 200*rho_c(z) definition sigma_v() relies on
    internally, so M, R_vir and sigma_v are mutually consistent (an SIS
    whose enclosed mass at R_vir equals M, sourced by rho_200). Deliberately
    not using colossus.halo.mass_so.M_to_R, which works in Mpc/h, Msun/h
    internal units that risk a silent mismatch against this module's plain
    Msun/Mpc astropy convention.

    Parameters
    ----------
    M : float
        Halo mass in solar masses (M200c).
    z : float
        Halo redshift.
    """
    rho_c_g_cm3 = cosmo.critical_density(z).value# g/cm^3
    rho_c = rho_c_g_cm3*1e-3*1e6 # kg/m^3
    rho_200 = 200*rho_c
    return np.power(3*M*MSUN/(4*np.pi*rho_200),1./3)

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

def sigma_two_velocity(sigma,z_L,z_S): #cross section
    return np.pi * theta_E(sigma,z_L,z_S)**2


#Magnification cross section: mu > mu0
def sigma_mu(M,z_L,z_S,mu0): #cross section
    sigma = sigma_v(M,z_L) #velocity dispersion
    return 2. * np.pi * np.power(theta_E(sigma,z_L,z_S),2.) * (mu0**2 + 1) / np.power((mu0**2 - 1),2.)

"""Time delay"""
def t_ref_sigma(sigma,zL,zS): 
    """Reference time delay in years
    
    Parameters
    ----------
    sigma : float
        velocity dispersion in m/s
    zL : float
        redshift of the lens
    zS : float
        redshift of the source
    """
    return lensutils.t_distance(zL,zS)*theta_E(sigma,zL,zS)**2 / YEAR


def t_ref(ML,zL,zS): 
    """Reference time delay in years
    
    Parameters
    ----------
    ML : float
        lens mass in solar masses
    zL : float
        redshift of the lens
    zS : float
        redshift of the source
    """
    sigma = sigma_v(ML,zL)
    return t_ref_sigma(sigma,zL,zS)

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
    return DeltaT(y) * t_ref(ML,zL,zS)

def mu_plus_Dt(Dt,ML,zL,zS):
    """Magnification for the positive image given a time delay
    
    Parameters
    ----------
    Dt : float
        time delay in years
    ML : float
        lens mass in solar masses
    zL : float
        redshift of the lens
    zS : float
        redshift of the source
    """
    return 1 + 2. * t_ref(ML,zL,zS) / Dt

def mu_minus_Dt(Dt,ML,zL,zS):
    """Magnification for the negative image given a time delay
    
    Parameters
    ----------
    Dt : float
        time delay in years
    ML : float
        lens mass in solar masses
    zL : float
        redshift of the lens
    zS : float
        redshift of the source
    """
    return 1 - 2. * t_ref(ML,zL,zS) / Dt