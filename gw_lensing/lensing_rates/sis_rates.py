import numpy as np
from scipy.integrate import trapezoid as trapz

from ..cosmology import gwcosmo
from ..utils import gwutils
from ..detectors import sensitivity_curves as sc
from ..gw_rates.rates import dNcbc_dz

from ..optical_depth import sis_optical_depth as sistau
from ..lens_models import sis

"""Strong lensing rates
(rates that a GW is lensed, not necessarily detected)
"""

def dNcbc_lens_dz(z,pz,R0,H0,Om0,Tobs,log10Mmin,log10Mmax,nMs,nzLs):
    """Differential rate of CBCs lensed by SISs at redshift z
    
    Parameters
    ----------
    z : float
        Redshift
    pz : function
        Redshift distribution of CBCs
    R0 : float
        Rate of CBCs at z=0
    H0 : float
        Hubble constant in km/s/Mpc
    Om0 : float
        Matter density at z=0
    Tobs : float
        Observation time in years
    log10Mmin : float
        Minimum log10 mass of the SIS in Msun
    log10Mmax : float
        Maximum log10 mass of the SIS in Msun
    nMs : int
        Number of lens mass bins
    nzLs : int
        Number of lens redshift bins
        """
    return dNcbc_dz(z,pz,R0,H0,Om0,Tobs)*sistau.tau(z,log10Mmin,log10Mmax,nMs,nzLs)

def Ncbc_lens(pz,R0,H0,Om0,Tobs,log10Mmin,log10Mmax,nMs,nzLs,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn = dNcbc_lens_dz(zs,pz,R0,H0,Om0,Tobs,log10Mmin,log10Mmax,nMs,nzLs)
    return trapz(dn,zs)

"""Strongly lensed detected GW rates
(rates that a GW is lensed and detected WITHOUT magnification bias)
"""

def d3Ndet_lens_dzdm1dm2(z,mass_1,mass_2,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs):
    #rate in yr^-1 Gpc^-3
    #f in *detector* frame
    #M in *source* frame
    #Tobs in *detector* frame
    Mc = gwutils.mchirp(mass_1,mass_2) #source frame masses
    dL = gwcosmo.dL_approx(z,H0,Om0)
    fmin_gw = fmin_detect # gwutils.f_ini(Tobs,Mc*(1.+z))
    mass_1z = mass_1*(1.+z)
    mass_2z = mass_2*(1.+z)
    snr_opt = gwutils.vsnr_from_psd(mass_1z,mass_2z,dL,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based)
    pw = sc.pw_hl(snr_th/snr_opt)

    pm = norm_m1 * pm1(mass_1) * pm2(mass_2,mass_1)
    integrand = dNcbc_dz(z,pz,R0,H0,Om0,Tobs) * pm * pw *sistau.tau(z,log10Mmin,log10Mmax,nMs,nzLs)
    return integrand
vd3Ndet_lens_dzdm1dm2 = np.vectorize(d3Ndet_lens_dzdm1dm2)

def d2Ndet_lens_dzdm1(z,mass_1,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mmin,n_m2):
    # Mass in Msun at *source* frame
    mass_2s = np.linspace(mmin,mass_1,n_m2)
    dn_detec = d3Ndet_lens_dzdm1dm2(z,mass_1,mass_2s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs)
    return trapz(dn_detec,mass_2s)
vd2Ndet_lens_dzdm1 = np.vectorize(d2Ndet_lens_dzdm1)

def dNdet_lens_dz(z,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mmin,mmax,n_m1,n_m2):
    # Mass in Msun at *source* frame
    mass_1s = np.linspace(mmin,mmax,n_m1)
    dn_detec = vd2Ndet_lens_dzdm1(z,mass_1s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mmin,n_m2)
    return trapz(dn_detec,mass_1s)
vdNdet_lens_dz = np.vectorize(dNdet_lens_dz)

def Ndet_lens(pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn_detec = vdNdet_lens_dz(zs,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mmin,mmax,n_m1,n_m2)
    return trapz(dn_detec,zs)

"""Strongly lensed GW image detection rates
(rates that a GW is lensed and detected WITH magnification bias)
"""

def d4Ndet_lens_mu_dzdm1dm2dy(z,mass_1,mass_2,y,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image):
    #rate in yr^-1 Gpc^-3
    #f in *detector* frame
    #M in *source* frame
    #Tobs in *detector* frame
    Mc = gwutils.mchirp(mass_1,mass_2) #source frame masses
    dL = gwcosmo.dL_approx(z,H0,Om0)
    fmin_gw = fmin_detect # gwutils.f_ini(Tobs,Mc*(1.+z))
    mass_1z = mass_1*(1.+z)
    mass_2z = mass_2*(1.+z)
    snr_opt = gwutils.vsnr_from_psd(mass_1z,mass_2z,dL,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based)
    snr_opt_mu = snr_opt * np.sqrt(abs(mu_image(y)))
    pw = sc.pw_hl(snr_th/snr_opt_mu)

    pm = norm_m1 * pm1(mass_1) * pm2(mass_2,mass_1)
    integrand = dNcbc_dz(z,pz,R0,H0,Om0,Tobs) * pm * pw *sistau.tau(z,log10Mmin,log10Mmax,nMs,nzLs)
    return integrand
vd4Ndet_lens_dzdm1dm2dy = np.vectorize(d4Ndet_lens_mu_dzdm1dm2dy)

def d3Ndet_lens_mu_dzdm1dm2(z,mass_1,mass_2,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys):
    # Mass in Msun at *source* frame
    eps = 1e-10
    ys = np.linspace(0+eps,1.-eps,nys)
    dn_detec = d4Ndet_lens_mu_dzdm1dm2dy(z,mass_1,mass_2,ys,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image)
    return trapz(dn_detec,ys)
vd3Ndet_lens_mu_dzdm1dm2 = np.vectorize(d3Ndet_lens_mu_dzdm1dm2)

def d2Ndet_lens_mu_dzdm1(z,mass_1,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,n_m2):
    # Mass in Msun at *source* frame
    mass_2s = np.linspace(mmin,mass_1,n_m2)
    dn_detec = vd3Ndet_lens_mu_dzdm1dm2(z,mass_1,mass_2s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys)
    return trapz(dn_detec,mass_2s)
vd2Ndet_lens_mu_dzdm1 = np.vectorize(d2Ndet_lens_mu_dzdm1)

def dNdet_lens_mu_dz(z,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2):
    # Mass in Msun at *source* frame
    mass_1s = np.linspace(mmin,mmax,n_m1)
    dn_detec = vd2Ndet_lens_mu_dzdm1(z,mass_1s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,n_m2)
    return trapz(dn_detec,mass_1s)
vdNdet_lens_mu_dz = np.vectorize(dNdet_lens_mu_dz)

def Ndet_lens_mu(pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn_detec = vdNdet_lens_mu_dz(zs,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2)
    return trapz(dn_detec,zs)

"""Strongly lensed GW image detection rates
(rates that a GW is lensed and detected WITH magnification bias AND time delay selection)
"""

def d5Ndet_lens_mu_Dt_dtdzdm1dm2dy(t,z,mass_1,mass_2,y,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image):
    #rate in yr^-1 Gpc^-3
    #f in *detector* frame
    #M in *source* frame
    #Tobs in *detector* frame
    Mc = gwutils.mchirp(mass_1,mass_2) #source frame masses
    dL = gwcosmo.dL_approx(z,H0,Om0)
    fmin_gw = fmin_detect # gwutils.f_ini(Tobs,Mc*(1.+z))
    mass_1z = mass_1*(1.+z)
    mass_2z = mass_2*(1.+z)
    snr_opt = gwutils.vsnr_from_psd(mass_1z,mass_2z,dL,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based)
    snr_opt_mu = snr_opt * np.sqrt(abs(mu_image(y)))
    pw = sc.pw_hl(snr_th/snr_opt_mu)

    pm = norm_m1 * pm1(mass_1) * pm2(mass_2,mass_1)
    integrand = dNcbc_dz(z,pz,R0,H0,Om0,Tobs) * pm * pw *sistau.tau_Dt(z,t,Tobs,y,log10Mmin,log10Mmax,nMs,nzLs)
    return integrand
vd5Ndet_lens_mu_Dt_dtdzdm1dm2dy = np.vectorize(d5Ndet_lens_mu_Dt_dtdzdm1dm2dy)

def d4Ndet_lens_mu_Dt_dtdzdm1dm2(t,z,mass_1,mass_2,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys):
    # Mass in Msun at *source* frame
    eps = 1e-10
    ys = np.linspace(0+eps,1.-eps,nys)
    dn_detec = d5Ndet_lens_mu_Dt_dtdzdm1dm2dy(t,z,mass_1,mass_2,ys,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image)
    return trapz(dn_detec,ys)
vd4Ndet_lens_mu_Dt_dtdzdm1dm2 = np.vectorize(d4Ndet_lens_mu_Dt_dtdzdm1dm2)

def d3Ndet_lens_mu_Dt_dtdzdm1(t,z,mass_1,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,n_m2):
    # Mass in Msun at *source* frame
    mass_2s = np.linspace(mmin,mass_1,n_m2)
    dn_detec = vd4Ndet_lens_mu_Dt_dtdzdm1dm2(t,z,mass_1,mass_2s,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys)
    return trapz(dn_detec,mass_2s)
vd3Ndet_lens_mu_Dt_dtdzdm1 = np.vectorize(d3Ndet_lens_mu_Dt_dtdzdm1)

def d2Ndet_lens_mu_Dt_dtdz(t,z,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2):
    # Mass in Msun at *source* frame
    mass_1s = np.linspace(mmin,mmax,n_m1)
    dn_detec = vd3Ndet_lens_mu_Dt_dtdzdm1(t,z,mass_1s,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,n_m2)
    return trapz(dn_detec,mass_1s)
vd2Ndet_lens_mu_Dt_dtdz = np.vectorize(d2Ndet_lens_mu_Dt_dtdz)

def dNdet_lens_mu_Dt_dt(t,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn_detec = vd2Ndet_lens_mu_Dt_dtdz(t,zs,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2)
    return trapz(dn_detec,zs)
vdNdet_lens_mu_Dt_dt = np.vectorize(dNdet_lens_mu_Dt_dt)

def Ndet_lens_mu_Dt(pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z,n_t):
    # Mass in Msun at *source* frame
    ts = np.linspace(0.,Tobs,n_t)
    dn_detec = vdNdet_lens_mu_Dt_dt(ts,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,log10Mmin,log10Mmax,nMs,nzLs,mu_image,nys,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z)
    return trapz(dn_detec,ts)

