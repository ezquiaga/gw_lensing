import numpy as np
from scipy.special import gamma

from astropy.cosmology import Planck18 as cosmo
h0 = cosmo.H(0).value/100
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')

from scipy.integrate import trapezoid

from ..utils.constants import *
from ..lens_models.sis import *


"""Multiple image optical depth"""
def d2tau_dzdlnM(M,z_L,z_S):
    #hmf in Mpc^-3
    #dVc/dzdOmega in Mpc^3/sr
    dVcdzdOm = cosmo.differential_comoving_volume(z_L).value
    dndlnM = (h0**3)*mass_function.massFunction(M, z_L, mdef = '200c', model = 'tinker08',q_out = 'dndlnM')
    return dVcdzdOm * dndlnM * sigma_two(M,z_L,z_S)
vd2tau_dzdlnM = np.vectorize(d2tau_dzdlnM)

def dtau_dz(z_L,z_S,log10Mmin,log10Mmax,nMs):
    Ms = np.logspace(log10Mmin,log10Mmax,nMs)
    return trapezoid(d2tau_dzdlnM(Ms,z_L,z_S),np.log(Ms))
dtau_dz = np.vectorize(dtau_dz)

def dtau_dlnM(ML,z_S,nzs):
    zLs = np.logspace(-3,np.log10(z_S),nzs)
    zLs[-1] = z_S
    return trapezoid(vd2tau_dzdlnM(ML,zLs,z_S),zLs)
dtau_dlnM = np.vectorize(dtau_dlnM)

def tau(z_S,log10Mmin,log10Mmax,nMs,nzs):
    #zs = np.linspace(0.,z_S,nzs)
    zs = np.logspace(-3,np.log10(z_S),nzs)
    zs[-1] = z_S
    return trapezoid(dtau_dz(zs,z_S,log10Mmin,log10Mmax,nMs),zs)
tau = np.vectorize(tau)

#Optical depth for single lens
"""Multiple image optical depth"""
def dtau_singlelens_dz(sigma,z_L,z_S):
    #sigma in m/s
    dOm = 4*np.pi
    return sigma_two_velocity(sigma,z_L,z_S)/dOm

def tau_singlelens(z_S,sigma,z_L):
    #zs = np.linspace(zL,z_S,nzs)
    return dtau_singlelens_dz(sigma,z_L,z_S)#trapezoid(dtau_dz(sigma,zs,z_S),zs)
tau_singlelens = np.vectorize(tau_singlelens)

#SIS with Schechter Mass Function
def dtau_Schechterdz(zS,zL,n,sigmaS,alpha,beta):
    #sigmaS km/s
    #n in #/Mpc^3
    c_km = Clight/1000
    Hz = cosmo.H(zL).value #Km/Mpc /s
    DL = cosmo.angular_diameter_distance(zL).value #Mpc
    DS = cosmo.angular_diameter_distance(zS).value #Mpc
    DLS = cosmo.angular_diameter_distance_z1z2(zL,zS).value #Mpc
    
    factor = 16 * np.power(np.pi,3)
    
    return factor*np.power(1+zL,2)*(c_km*n/Hz)*np.power(DL*DLS/DS,2)*np.power(sigmaS/c_km,4)*gamma((4+alpha)/beta)/gamma(alpha/beta)
    
def tau_Schechter(zS,n,sigmaS,alpha,beta):
    zLs = np.logspace(min(-3,np.log10(zS/100)),np.log10(zS),101)
    return trapezoid(dtau_Schechterdz(zS,zLs[:-1],n,sigmaS,alpha,beta),zLs[:-1])
tau_Schechter = np.vectorize(tau_Schechter)

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

"""Optical depth with timde delay selection function"""

def d2tau_Dt_dzdlnM(ML,z_L,z_S,t,Tobs,y):
    #hmf in Mpc^-3
    #dVc/dzdOmega in Mpc^3/sr
    dVcdzdOm = cosmo.differential_comoving_volume(z_L).value
    dndlnM = (h0**3)*mass_function.massFunction(ML, z_L, mdef = '200c', model = 'tinker08',q_out = 'dndlnM')

    #time delay selection
    dt = t_delay(y,ML,z_L,z_S)
    time_available = Tobs - t
    pdet_dt = np.heaviside(time_available - dt,1)

    return dVcdzdOm * dndlnM * sigma_two(ML,z_L,z_S) * pdet_dt
vd2tau_Dt_dzdlnM = np.vectorize(d2tau_Dt_dzdlnM)

def dtau_Dt_dz(z_L,z_S,t,Tobs,y,log10Mmin,log10Mmax,nMs):
    Ms = np.logspace(log10Mmin,log10Mmax,nMs)
    return trapezoid(d2tau_Dt_dzdlnM(Ms,z_L,z_S,t,Tobs,y),np.log(Ms))
dtau_Dt_dz = np.vectorize(dtau_Dt_dz)

def tau_Dt(z_S,t,Tobs,y,log10Mmin,log10Mmax,nMs,nzLs):
    #zs = np.linspace(0.,z_S,nzs)
    #zLs_plus_one = np.logspace(-3,np.log10(z_S),nzLs+1)
    #zLs = zLs_plus_one[:-1]
    zLs = np.logspace(-3,np.log10(z_S-1.0e-10),nzLs)
    return trapezoid(dtau_Dt_dz(zLs,z_S,t,Tobs,y,log10Mmin,log10Mmax,nMs),zLs)
tau_Dt = np.vectorize(tau_Dt)

"""Time delay optical depth"""
def d3tau_dDtdzdlnM(Dt,M,z_L,z_S):
    #hmf in Mpc^-3
    #dVc/dzdOmega in Mpc^3/sr
    dVcdzdOm = cosmo.differential_comoving_volume(z_L).value
    dndlnM = (h0**3)*mass_function.massFunction(M, z_L, mdef = '200c', model = 'tinker08',q_out = 'dndlnM')
    return dVcdzdOm * dndlnM * sigma_two(M,z_L,z_S) * np.heaviside(t_ref(M,z_L,z_S)-Dt,1.)/t_ref(M,z_L,z_S)
vd3tau_dDtdzdlnM = np.vectorize(d3tau_dDtdzdlnM)

def d2tau_dDtdz(Dt,z_L,z_S,log10Mmin,log10Mmax,nMs):
    Ms = np.logspace(log10Mmin,log10Mmax,nMs)
    return trapezoid(d3tau_dDtdzdlnM(Dt,Ms,z_L,z_S),np.log(Ms))
d2tau_dDtdz = np.vectorize(d2tau_dDtdz)

def dtau_dDt(Dt,z_S,log10Mmin,log10Mmax,nMs,nzs):
    #zs = np.linspace(0.,z_S,nzs)
    zs = np.logspace(-3,np.log10(z_S-1e-10),nzs)
    return trapezoid(d2tau_dDtdz(Dt,zs,z_S,log10Mmin,log10Mmax,nMs),zs)
dtau_dDt = np.vectorize(dtau_dDt)