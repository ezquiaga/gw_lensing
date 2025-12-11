import numpy as np
from scipy.integrate import trapz
from scipy.integrate import cumtrapz

from ..cosmology import gwcosmo
#from spectral_sirens.utils import gwutils
from ..detectors import sensitivity_curves as sc

from ..utils import gwutils
from ..utils import utils

def dNcbc_dz(z,pz,R0,H0,Om0,Tobs):
    #rate in yr^-1 Gpc^-3
    #Tobs in *detector* frame

    Rz = R0 * pz(z)
    Vol = gwcosmo.diff_comoving_volume_approx(z,H0,Om0)

    integrand = Tobs * Rz * Vol / (1.+z) 
    return integrand
vdNcbc_dz = np.vectorize(dNcbc_dz)

def d3Ndet_dzdm1dm2(z,mass_1,mass_2,pz,pm1,pm2,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based):
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
    integrand = dNcbc_dz(z,pz,R0,H0,Om0,Tobs) * pm * pw 
    return integrand
vd3Ndet_dzdm1dm2 = np.vectorize(d3Ndet_dzdm1dm2)

def d2Ndet_dzdm1(z,mass_1,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,n_m2):
    # Mass in Msun at *source* frame
    mass_2s = np.linspace(mmin,mass_1,n_m2)
    dn_detec = d3Ndet_dzdm1dm2(z,mass_1,mass_2s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based)
    return trapz(dn_detec,mass_2s)
vd2Ndet_dzdm1 = np.vectorize(d2Ndet_dzdm1)

def dNdet_dz(z,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,mmax,n_m1,n_m2):
    # Mass in Msun at *source* frame
    mass_1s = np.linspace(mmin,mmax,n_m1)
    dn_detec = vd2Ndet_dzdm1(z,mass_1s,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,n_m2)
    return trapz(dn_detec,mass_1s)
vdNdet_dz = np.vectorize(dNdet_dz)

def Ndet(pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,mmax,n_m1,n_m2,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn_detec = vdNdet_dz(zs,pz,pm1,pq,R0,norm_m1,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,mmin,mmax,n_m1,n_m2)
    return trapz(dn_detec,zs)

def Ncbc(pz,R0,H0,Om0,Tobs,zmin,zmax,n_z):
    # Mass in Msun at *source* frame
    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn = dNcbc_dz(zs,pz,R0,H0,Om0,Tobs)
    return trapz(dn,zs)


""" Sampling populations """
def compute_cdf(pz,pm1,pq,R0,H0,Om0,Tobs,zmin,zmax,mmin,mmax):
    zs = np.linspace(zmin,zmax,1000)
    cdf_z = cumtrapz(dNcbc_dz(zs,pz,R0,H0,Om0,Tobs),zs,initial=0)
    norm_z = cdf_z[-1] 
    cdf_z /= norm_z #make it a pdf

    masses = np.linspace(mmin,mmax,1000)
    cdf_m1 = cumtrapz(pm1(masses),masses,initial=0)
    norm_m1 = cdf_m1[-1]
    cdf_m1 /= norm_m1 #make it a pdf

    qs = np.linspace(0,1,1000)
    cdf_q = cumtrapz(pq(qs),qs,initial=0)
    norm_q = cdf_q[-1]
    cdf_q /= norm_q #make it a pdf

    return cdf_z, cdf_m1, cdf_q, norm_z, norm_m1, norm_q, zs, masses, qs

def mock_source_parameters(n_sources,cdf_z,cdf_m1,cdf_q,zs,masses,qs):

    z_mock = utils.inverse_transf_sampling(cdf_z,zs,n_sources)
    m1_mock = utils.inverse_transf_sampling(cdf_m1,masses,n_sources)
    q_mock = utils.inverse_transf_sampling(cdf_q,qs,n_sources)
    m2_mock = m1_mock * q_mock

    y_mock = np.random.uniform(0,1,n_sources) #dimensionless source position

    return m1_mock, m2_mock, z_mock, y_mock

""" Monte Carlo integration for rates """

def dNdet_MC_dz(z,N_mc,pz,pm1,pq,R0,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,zmin,zmax,mmin,mmax):
    #rate in yr^-1 Gpc^-3
    #f in *detector* frame
    #M in *source* frame
    #Tobs in *detector* frame
    
    cdf_z, cdf_m1, cdf_q, norm_z, norm_m1, norm_q, zs_cdf, masses_cdf, qs_cdf = compute_cdf(pz,pm1,pq,R0,H0,Om0,Tobs,zmin,zmax,mmin,mmax)

    #Note: the samples in z are actually not needed in this implementation
    # samples in y are for the magnification weighting
    m1_mock, m2_mock, z_mock, y_mock = mock_source_parameters(N_mc,cdf_z,cdf_m1,cdf_q,zs_cdf,masses_cdf,qs_cdf)

    dL = gwcosmo.dL_approx(z,H0,Om0)
    mass_1z = m1_mock*(1.+z)
    mass_2z = m2_mock*(1.+z)

    fmin_gw = fmin_detect # gwutils.f_ini(Tobs,Mc*(1.+z))
    snr_opt = gwutils.vsnr_from_psd(mass_1z,mass_2z,dL,fmin_gw,Tobs,detectorSn,fmin_detect,fmax_detect,based)
    pw = sc.pw_hl(snr_th/snr_opt)

    #We make a Monte Carlo integral for all variable but the redshift

    integrand = dNcbc_dz(z,pz,R0,H0,Om0,Tobs) * pw 
    mean_integrand = np.sum(integrand)/N_mc

    mean_integrand_square = np.sum(integrand**2)/N_mc
    error_std = np.sqrt((mean_integrand_square-mean_integrand**2)/N_mc)

    return mean_integrand, error_std
vdNdet_MC_dz = np.vectorize(dNdet_MC_dz)

def Ndet_MC(N_mc,pz,pm1,pq,R0,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,zmin,zmax,mmin,mmax,n_z):

    zs = np.logspace(np.log10(zmin),np.log10(zmax),n_z)
    dn_detec, dn_error = vdNdet_MC_dz(zs,N_mc,pz,pm1,pq,R0,H0,Om0,Tobs,snr_th,detectorSn,fmin_detect,fmax_detect,based,zmin,zmax,mmin,mmax)
    return trapz(dn_detec,zs), trapz(dn_error,zs)