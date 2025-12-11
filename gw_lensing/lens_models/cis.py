import numpy as np
from scipy.optimize import newton
from astropy.cosmology import Planck18 as cosmo

from ..utils.constants import *
from ..utils import lensutils

def magnification_isothermal_sphere(theta,theta_c,theta_0):
    """
    Calculate the magnification for an isothermal sphere lens.
    Parameters
    ----------
    theta : float or array-like
        Angular separation from the lens center in radians.
    theta_c : float
        Core radius of the lens in radians.
    theta_0 : float
        Einstein radius of the lens in radians.
    Returns
    -------
    mu : float or array-like
        Magnification at the given angular separation.
    """
    # Ensure theta is an array for vectorized operations
    theta = np.asarray(theta)
    # Calculate the magnification using the isothermal sphere formula
    sqrt_term = np.sqrt(theta**2 + theta_c**2)
    mu_inv = (1 - theta_0 /2 / sqrt_term)**2 - theta_0**2 * (2*theta_c*sqrt_term - 2*theta_c**2- theta**2)**2 / (4 * theta**4 * sqrt_term**2)
    return 1 / mu_inv


def theta_E_isothermal_sphere(sigma,z_L,z_S):
    """
    Calculate the Einstein radius for an isothermal sphere lens.
    
    Parameters
    ----------
    sigma : float
        Velocity dispersion of the lens in km/s.
    z_L : float
        Redshift of the lens.
    z_S : float
        Redshift of the source.
    
    Returns
    -------
    theta_E : float
        Einstein radius in radians.
    """
    #D_L = cosmo.angular_diameter_distance(z_L).value  # Mpc
    D_S = cosmo.angular_diameter_distance(z_S).value  # Mpc
    D_LS = cosmo.angular_diameter_distance_z1z2(z_L, z_S).value  # Mpc

    sigma_ms = sigma * 1000  # Convert km/s to m/s
    
    theta_E = (4 * np.pi * sigma_ms**2 / Clight**2) * (D_LS / (D_S))  # in radians
    return theta_E 

#Solve lens equation for a given source position in the cored isothermal sphere lens model.
def lens_equation_isothermal_sphere(theta_S, theta_c, theta_0):
    """
    Solve the lens equation for a cored isothermal sphere lens model.
    
    Parameters
    ----------
    theta_S : float
        Angular position of the source in radians.
    theta_c : float
        Core radius of the lens in radians.
    theta_0 : float
        Einstein radius of the lens in radians.
    
    Returns
    -------
    theta_lens : float
        Angular position of the image in radians.
    """
    # Define the function to find the root
    def f(theta_lens):
        return magnification_isothermal_sphere(theta_lens, theta_c, theta_0) - (theta_S / theta_lens)
    
    # Use a numerical method to find the root (e.g., Newton's method)
    theta_lens = newton(f, x0=theta_S)  # Initial guess is the source position
    return theta_lens

#Magnification central image
def Sigma_crit(z_L, z_S):
    """
    Calculate the critical surface mass density for lensing.
    
    Parameters
    ----------
    z_L : float
        Redshift of the lens.
    z_S : float
        Redshift of the source.
    
    Returns
    -------
    Sigma_crit : float
        Critical surface mass density in Msun/pc^2.
    """
    D_L = cosmo.angular_diameter_distance(z_L).value * 1e6  # Mpc to pc
    D_S = cosmo.angular_diameter_distance(z_S).value * 1e6  # Mpc to pc
    D_LS = cosmo.angular_diameter_distance_z1z2(z_L, z_S).value * 1e6  # Mpc to pc
    # Convert constants to appropriate units
    Sigma_crit = Clight**2 / (4 * np.pi * Gnewton) * (D_S / (D_L * D_LS))*PC/MSUN  # Msun/pc^2
    return Sigma_crit



def mu_central_image(Sigma_c,z_L,z_S,gamma=0):
    """
    Calculate the magnification of the central image in a lensing system.
    
    Parameters
    ----------
    Sigma_c : float
        Surface mass density at the center of the lens in Msun/pc^2.
    z_L : float
        Redshift of the lens.
    z_S : float
        Redshift of the source.
    
    Returns
    -------
    mu_central : float
        Magnification of the central image.
    """
    Sigma_crit_value = Sigma_crit(z_L, z_S)
    kappa_c = Sigma_c / Sigma_crit_value  # Dimensionless magnification
    return 1. / ((1 - kappa_c)**2 - abs(gamma))  # Magnification of the central image