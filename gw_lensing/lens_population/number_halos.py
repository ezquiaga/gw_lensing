import numpy as np
from astropy.cosmology import Planck18 as cosmo
h0 = cosmo.H(0).value/100
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')

from scipy.integrate import trapezoid

from ..utils.constants import *

def d2n_halodzdlnM(M,z_L,z_S):
    #hmf in Mpc^-3
    #dVc/dz in Mpc^3
    dVcdzdOm = 4*np.pi*cosmo.differential_comoving_volume(z_L).value
    dndlnM = (h0**3)*mass_function.massFunction(M, z_L, mdef = '200c', model = 'tinker08',q_out = 'dndlnM')
    #dndlnM = (h0**3)*hmf.dndlnM_int(M,z_L)
    return dVcdzdOm * dndlnM 

def dn_halodz(z_L,z_S,log10Mmin,log10Mmax,nMs):
    Ms = np.logspace(log10Mmin,log10Mmax,nMs)
    return trapezoid(d2n_halodzdlnM(Ms,z_L,z_S),np.log(Ms))
dn_halodz = np.vectorize(dn_halodz)

def n_halo(z_S,log10Mmin,log10Mmax,nMs,nzs):
    #zs = np.linspace(0.,z_S,nzs)
    zs = np.logspace(-3,np.log10(z_S),nzs)
    return trapezoid(dn_halodz(zs,z_S,log10Mmin,log10Mmax,nMs),zs)
n_halo = np.vectorize(n_halo)