import numpy as np
from scipy.integrate import trapezoid
from astropy.cosmology import Planck18 as cosmo

from ..utils.constants import *
from ..utils import lensutils

"""Cross section"""
#Area cut for SIE lenses, where 2 images are formed
def area_cut(f,nDs):
    ff = np.sqrt(1.-f**2)
    D = np.linspace(f,1.,nDs)
    return (4.*f/np.power(ff,2))*trapezoid(np.arccos(D[1:])/np.sqrt(D[1:]**2 - f**2),D[1:])
area_cut = np.vectorize(area_cut)

#Area caustic for SIE lenses, where 4 images are formed
def area_caustic(f,nDs):
    ff = np.sqrt(1.-f**2)
    D = np.linspace(f,1.,nDs)
    integrand = (np.sqrt(D**2 - f**2)/np.power(D,2))*(np.sqrt(1-D**2)/D - np.arccos(D))
    return (4.*f/np.power(ff,2))*trapezoid(integrand,D)
area_caustic = np.vectorize(area_caustic)