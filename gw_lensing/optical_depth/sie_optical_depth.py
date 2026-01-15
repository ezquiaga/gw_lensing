import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from astropy.cosmology import Planck18 as cosmo
h0 = cosmo.H(0).value/100
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')


from ..utils.constants import *
from ..utils import lensutils
from ..lens_models import sie

"""Optical depth for SIE lenses"""
def dddtaudzdsdq_Schechter(zS,zL,n,q,sigma,sigmaS,qmin,alpha,beta,cross_section):
    #sigmaS km/s
    #n in #/Mpc^3
    c_km = Clight/1000
    Hz = cosmo.H(zL).value #Km/Mpc /s
    DL = cosmo.angular_diameter_distance(zL).value #Mpc
    DS = cosmo.angular_diameter_distance(zS).value #Mpc
    DLS = cosmo.angular_diameter_distance_z1z2(zL,zS).value #Mpc
    
    factor = 16 * np.power(np.pi,3) 
    sis = factor*np.power(1+zL,2)*(c_km*n/Hz)*np.power(DL*DLS/DS,2)*np.power(sigma/c_km,4)*lensutils.dpdsigma(sigma,sigmaS,alpha,beta)
    return sis * (cross_section(q)/np.pi)*lensutils.g_truncated(1 - q,qmin,sigma)
#note that we have normalized the dimensionless cut region to 1 at f->1

def ddtaudzds_Schechter(zS,zL,n,sigma,sigmaS,qmin,alpha,beta,nqs,cross_section):
    q_s = np.linspace(0.2,0.99,nqs)
    return trapezoid(dddtaudzdsdq_Schechter(zS,zL,n,q_s,sigma,sigmaS,qmin,alpha,beta,cross_section),q_s)
vddtaudzds_Schechter = np.vectorize(ddtaudzds_Schechter)

def dtaudz_Schechter(zS,zL,n,sigmaS,qmin,alpha,beta,ns,nqs,cross_section):
    sigma_s = np.logspace(np.log10(sigmaS/100),np.log10(sigmaS*100),ns)
    return trapezoid(vddtaudzds_Schechter(zS,zL,n,sigma_s,sigmaS,qmin,alpha,beta,nqs,cross_section),sigma_s)
vdtaudz_Schechter = np.vectorize(dtaudz_Schechter)

def tau_Schechter(zS,n,sigmaS,qmin,alpha,beta,nz,ns,nqs,cross_section):
    zLs = np.logspace(min(-3,np.log10(zS/100)),np.log10(zS),nz)
    return trapezoid(vdtaudz_Schechter(zS,zLs,n,sigmaS,qmin,alpha,beta,ns,nqs,cross_section),zLs)
tau_Schechter = np.vectorize(tau_Schechter)