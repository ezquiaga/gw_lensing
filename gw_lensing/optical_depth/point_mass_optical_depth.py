# lensing_optical_depth.py

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import c

def d_tau_dz(
    z_l,
    z_s,
    f_DM,
    y0,
    Omega_DM=0.265,  # Planck18 value
    H0=67.66,        # Planck18 value in km/s/Mpc
):
    """
    Compute dτ/dz_ell for gravitational wave microlensing (Eq. B2).


    Parameters
    ----------
    z_l : float or np.ndarray
        Lens redshift(s)
    z_s : float
        Source redshift (must be > z_l)
    f_DM : float
        Fraction of dark matter in MACHOs
    y0 : float
        Maximum impact parameter (dimensionless)
    Omega_DM : float
        Dark matter density parameter (default: Planck18)
    H0 : float
        Hubble constant in km/s/Mpc (default: Planck18)

    Returns
    -------
    d_tau_dz : float or np.ndarray
        Differential optical depth at z_l
    """
    # Speed of light in km/s
    c_km_s = c.to('km/s').value

    # Hubble parameter at z_l in km/s/Mpc
    Hz = cosmo.H(z_l).value


    # Angular diameter distances in Mpc
    D_l = cosmo.angular_diameter_distance(z_l).value
    D_s = cosmo.angular_diameter_distance(z_s).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value

    # Avoid division by zero or negative distances
    if np.any(z_l >= z_s):
        raise ValueError("Lens redshift z_l must be less than source redshift z_s.")

    prefactor = f_DM * 1.5 * y0**2 * Omega_DM * (H0**2) / c_km_s
    numerator = (1 + z_l)**2 * D_ls * D_l
    denominator = Hz * D_s

    return prefactor * numerator / denominator



def tau(
    z_s,
    f_DM,
    y0,
    Omega_DM=0.265,  # Planck18 value
    H0=67.66,        # Planck18 value in km/s/Mpc
):
    """
    Compute the optical depth τ for gravitational wave microlensing (Eq. B1).

    Parameters
    ----------
    z_l : float or np.ndarray
        Lens redshift(s)
    z_s : float
        Source redshift (must be > z_l)
    f_DM : float
        Fraction of dark matter in MACHOs
    y0 : float
        Maximum impact parameter (dimensionless)
    Omega_DM : float
        Dark matter density parameter (default: Planck18)
    H0 : float
        Hubble constant in km/s/Mpc (default: Planck18)

    Returns
    -------
    tau : float or np.ndarray
        Optical depth at z_l
    """
    dz = 1e-3  # Small step in redshift for integration

    # Integrate dτ/dz over the lens redshift range
    z_range = np.arange(0, z_s, dz)
    d_tau_dz_values = d_tau_dz(z_range, z_s, f_DM, y0, Omega_DM, H0)

    return np.trapz(d_tau_dz_values, z_range)
tau = np.vectorize(tau)