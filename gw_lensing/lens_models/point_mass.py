"Python module with lensing function for a point mass"
import numpy as np
import mpmath as mp
from astropy.cosmology import Planck18 as cosmo
from scipy.special import gamma, hyp1f1
from scipy.interpolate import interp1d
from ..utils.constants import *

"Time delays"
#Wave-optics
def t_delay_wave(w): #Assumes w <<1 !!!
     return 0.5*(1.+np.euler_gamma+np.log(w/2.))
def t_delay_wave_phase(w): #Assumes w <<1 !!!
     return 0.5*(np.euler_gamma+np.log(w/2.))
#Geometric optics    
def t_delay_geom_plus(y):
    return (y**2. + 2. - y*np.sqrt(y**2 +4.))/4.-np.log(np.abs(y+np.sqrt(y**2+4.))/2.)
def t_delay_geom_minus(y):
    return (y**2 + 2. + y*np.sqrt(y**2 +4.))/4.-np.log(np.abs(y-np.sqrt(y**2+4.))/2.)
def DeltaT(y):
    return t_delay_geom_minus(y)-t_delay_geom_plus(y)

def t_ref(M_L,z_L):
    """Reference time delay for a point mass lens
    
    Parameters
    ----------
    M_L : float
        Lens mass in solar masses
    z_L : float
        Lens redshift
    """
    return 4 * (1+z_L)* Gnewton * M_L*MSUN / (Clight**3) /YEAR

"Angular position"
def theta_plus(b):
    return 0.5*b*(1.+np.sqrt(1.+4./b**2.))
def theta_minus(b):
    return 0.5*b*(1.-np.sqrt(1.+4./b**2.))

"Magnification"
def mu_plus(y):
    return 0.5 + (y**2. + 2.)/(2.*y*np.sqrt(y**2 + 4.))
def mu_minus(y):
    return 0.5 - (y**2. + 2.)/(2.*y*np.sqrt(y**2 + 4.))

"""Einstein radius"""
def theta_E(M_L,z_L,z_S):
    #M in solar masses
    #Clight in m/s
    DL = cosmo.angular_diameter_distance(z_L).value #Mpc
    DS = cosmo.angular_diameter_distance(z_S).value #Mpc
    DLS = cosmo.angular_diameter_distance_z1z2(z_L,z_S).value #Mpc
    
    thetaE2 = 4 * Gnewton * (1 + z_L)* M_L * MSUN / (Clight**2) * DLS / (DL * DS * MPC) #rad^2
    return np.sqrt(thetaE2) #rad

"Diffraction integral"
#Amplification function
def laguerre(n,a,x):
    nplaguerre = np.frompyfunc(mp.laguerre,3,1)
    return nplaguerre(n,a,x)
vlaguerre = np.vectorize(laguerre)
def F_wave(fs,b):
    lague = np.array(vlaguerre(-0.5j*fs, 0, 0.5j*fs*b**2.),dtype='complex')
    ampl = np.power(-0.5j,1.+0.5j*fs)*np.power(fs,1.+0.5j*fs)*gamma(-0.5j*fs)*lague
    ampl[fs==0. + 0.j] = 1. #gamma of 0 is inf but F(0,b)=1.
    return ampl
#Geometric optics - multiple images
def F_geom_opt(ws,y):
    Fplus = np.sqrt(np.abs(mu_plus(y)))*np.exp(1.0j*ws*t_delay_geom_plus(y))
    Fminus = np.sqrt(np.abs(mu_minus(y)))*np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fplus + Fminus

def F_geom_opt_I(ws,y):
    Fplus = np.sqrt(np.abs(mu_plus(y)))*np.exp(1.0j*ws*t_delay_geom_plus(y))
    #Fminus = np.sqrt(np.abs(mu_minus(y)))*np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fplus 

def F_geom_opt_II(ws,y):
    #Fplus = np.sqrt(np.abs(mu_plus(y)))*np.exp(1.0j*ws*t_delay_geom_plus(y))
    Fminus = np.sqrt(np.abs(mu_minus(y)))*np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fminus
#Geometric optics - phases
def F_geom_opt_phase(ws,y):
    Fplus = np.exp(1.0j*ws*t_delay_geom_plus(y))
    Fminus = np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fplus + Fminus

def F_geom_opt_I_phase(ws,y):
    Fplus = np.exp(1.0j*ws*t_delay_geom_plus(y))
    #Fminus = np.sqrt(np.abs(mu_minus(y)))*np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fplus 

def F_geom_opt_II_phase(ws,y):
    #Fplus = np.sqrt(np.abs(mu_plus(y)))*np.exp(1.0j*ws*t_delay_geom_plus(y))
    Fminus = np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fminus

def F_geom_opt_III_phase(ws,y):
    #Fplus = np.sqrt(np.abs(mu_plus(y)))*np.exp(1.0j*ws*t_delay_geom_plus(y))
    Fminus = np.exp(-1.0j*(np.pi)*np.sign(ws))
    return Fminus

#all together
def F(w,b,arg1):
    if arg1=='wave':
        res  = F_wave(w,b)
    elif arg1=='geom':
        res  = F_geom_opt(w,b)
    elif arg1 == 'I':
        res  = F_geom_opt_I(w,b)
    elif arg1 == 'II':
        res  = F_geom_opt_II(w,b)
    elif arg1=='geom_phase':
        res  = F_geom_opt_phase(w,b)
    elif arg1 == 'I_phase':
        res  = F_geom_opt_I_phase(w,b)
    elif arg1 == 'II_phase':
        res  = F_geom_opt_II_phase(w,b)
    elif arg1 == 'III_phase':
        res  = F_geom_opt_III_phase(w,b)
    return res

def F_shifted(w,b,t0,arg1):
    if arg1=='wave':
        res  = F_wave(w,b)
    elif arg1=='geom':
        res  = F_geom_opt(w,b)
    elif arg1 == 'I':
        res  = F_geom_opt_I(w,b)
    elif arg1 == 'II':
        res  = F_geom_opt_II(w,b)
    elif arg1=='geom_phase':
        res  = F_geom_opt_phase(w,b)
    elif arg1 == 'I_phase':
        res  = F_geom_opt_I_phase(w,b)
    elif arg1 == 'II_phase':
        res  = F_geom_opt_II_phase(w,b)
    elif arg1 == 'III_phase':
        res  = F_geom_opt_III_phase(w,b)
    return res*np.exp(-1.0j*w*t0)

"Phase wave-optics"
def w_Tgw(w,b):
    return -1.0j*np.log(F(w,b)/np.abs(F(w,b)))

def Tphase(w,b):
    ws = np.linspace(w/1.5,w*1.5,1000)
    Tp_int = interp1d(ws,w_Tgw(ws,b)/ws)
    return Tp_int(w)

def Tgroup(w,b):
    ws = np.linspace(w/1.5,w*1.5,1000)
    Tg_int = interp1d(ws,np.gradient(w_Tgw(ws,b),ws))
    return Tg_int(w)