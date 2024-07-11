import numpy as np
from astropy.cosmology import Planck18 as cosmo
h0 = cosmo.H(0).value/100
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')

from scipy.integrate import trapezoid

from ..utils.constants import *
from ..lens_models.sis import *


"""Multiple image optical depth"""
def d2taudzdlnM(M,z_L,z_S):
    #hmf in Mpc^-3
    #dVc/dzdOmega in Mpc^3/sr
    dVcdzdOm = cosmo.differential_comoving_volume(z_L).value
    dndlnM = (h0**3)*mass_function.massFunction(M, z_L, mdef = '200c', model = 'tinker08',q_out = 'dndlnM')
    return dVcdzdOm * dndlnM * sigma_two(M,z_L,z_S)

def dtaudz(z_L,z_S,log10Mmin,log10Mmax,nMs):
    Ms = np.logspace(log10Mmin,log10Mmax,nMs)
    return trapezoid(d2taudzdlnM(Ms,z_L,z_S),np.log(Ms))
dtaudz = np.vectorize(dtaudz)

def tau(z_S,log10Mmin,log10Mmax,nMs,nzs):
    #zs = np.linspace(0.,z_S,nzs)
    zs = np.logspace(-3,np.log10(z_S),nzs)
    return trapezoid(dtaudz(zs,z_S,log10Mmin,log10Mmax,nMs),zs)
tau = np.vectorize(tau)

"""High magnification optical depth"""
def d2taudzdlnM_mu(M,z_L,z_S,mu0):
    #Input dndlnM in (Mpc/h)^-3
    #dVc/dzdOmega in Mpc^3/sr
    dVcdzdOm = cosmo.differential_comoving_volume(z_L).value
    dndlnM_tinker = (h0**3)*mass_function.massFunction(M, z_L, mdef = '200c', model = 'tinker08',q_out = 'dndlnM')
    return dVcdzdOm * dndlnM_tinker * sigma_mu(M,z_L,z_S,mu0)

def dtaudz_mu(z_L,z_S,mu0,log10Mmin,log10Mmax,nMs):
    Ms = np.logspace(log10Mmin,log10Mmax,nMs)
    return trapezoid(d2taudzdlnM_mu(Ms,z_L,z_S,mu0),np.log(Ms))
dtaudz_mu = np.vectorize(dtaudz_mu)

def tau_mu(z_S,mu0,log10Mmin,log10Mmax,nMs,nzs):
    #zs = np.linspace(0.,z_S,nzs)
    zLs = np.logspace(-3,np.log10(z_S),nzs)
    return trapezoid(dtaudz_mu(zLs,z_S,mu0,log10Mmin,log10Mmax,nMs),zLs)
tau_mu = np.vectorize(tau_mu)