"""Fiducial universe based on "GWTC-5.0: Population Properties of Merging Compact Binaries",
LVK Collaboration, arXiv:2605.27226 (2026).
"""
#Fiducial universe
#-----------------

#Cosmology
#---------
# Note: arXiv:2605.27226 (Sec. IV.4.4) assumes a Planck 2015 LCDM cosmology (Ade et al. 2016).
H0_fid = 67.66 # cosmo.H(0).value #cosmology is Planck'18
Om0_fid = 0.30966 # cosmo.Om(0) #cosmology is Planck'18

#Merger rate
#-----------
# Note: GWTC-5.0's Default (BBH) redshift model is a pure power law in the comoving merger
r0_fid = 30.4 # R(z=0.2) = median[rate*(1.2)^lamb], [24.95, 36.85] (raw 'rate'=R(z=0)=19.12)
alpha_z_fid = 2.54 # 'lamb' (redshift index kappa_z), [1.84, 3.23]
zp_fid = 1.9 # Fixed to the Madau & Dickinson (2014) SFR peak (arXiv:1403.0007); repo convention
beta_fid = 5.6 - alpha_z_fid # Fixed to the Madau & Dickinson (2014) SFR high-z slope (arXiv:1403.0007); repo convention

#Mass distribution -- GWTC-5.0 Default BBH: broken power law + two Gaussian peaks
#-----------------
# Model:  p(m1) = lam_0 * broken_powerlaw(mMin,mMax,m_break,alpha_1,alpha_2)
#               + lam_1 * N(mu_1,sig_1) + (1-lam_0-lam_1) * N(mu_2,sig_2),
#         tapered by the low-/high-mass filters (gwpop.broken_powerlaw_two_peaks_smooth).
#
# SIGN CONVENTION: broken_powerlaw uses p ~ m^alpha
mmin_bpl_fid = 4.49  # 'mlow_1' (primary low-mass edge m_1,low), [3.15, 6.46]
mmax_bpl_fid = 300.0 # 'mmax' fixed at 300 Msun (delta prior, Table 5); no fitted high-mass cutoff
m_break_fid  = 37.45 # 'break_mass', [29.6, 48.3]
alpha_1_fid  = -1.48 # 'alpha_1' = 1.48, [0.11, 2.52]; SIGN FLIPPED
alpha_2_fid  = -5.42 # 'alpha_2' = 5.42, [3.87, 6.84]; SIGN FLIPPED
mu_1_fid  = 9.91 # 'mpp_1' (low-mass peak mean), [9.26, 10.29]
sig_1_fid = 0.78 # 'sigpp_1' (low-mass peak width), [0.497, 1.35]
mu_2_fid  = 32.33 # 'mpp_2' (high-mass peak mean), [26.5, 36.8]
sig_2_fid = 5.73 # 'sigpp_2' (high-mass peak width), [1.88, 9.54]
# Mixing fractions (Dirichlet(1,1,1) prior, Table 5); implicit high-peak fraction
# lam_2 = 1 - lam_0 - lam_1 = 0.05.
lam_0_fid = 0.40 # 'lam_0' (broken-power-law fraction), [0.189, 0.624]
lam_1_fid = 0.55 # 'lam_1' (low-mass-peak fraction), [0.327, 0.760]
# Low-mass taper: gwpop.broken_powerlaw_two_peaks_smooth applies the Planck-taper window
# (gwpop.Sfilter) over [mlow_1, mlow_1+delta_m_1], matching the release's primary-mass
# smoothing S(m1 | mlow_1, delta_m_1). The release's secondary-mass smoothing
# (mlow_2=3.46, delta_m_2=4.81) is not used by this single-primary-mass model.
mMin_filter_fid  = 4.49 # 'mlow_1' (primary low-mass taper edge), [3.15, 6.46]
dmMin_filter_fid = 3.53 # 'delta_m_1' (primary low-mass taper width), [0.299, 8.79]
# GWTC-5.0 fits NO high-mass cutoff (mmax pinned to 300); high filter set to be inert.
mMax_filter_fid  = 300.0 # 'mmax' fixed at 300 Msun (delta prior)
dmMax_filter_fid = 10.0  # inert (no fitted high-mass taper in the release)

#Mass ratio
bq_fid = 1.04 # 'beta' for p(q) ~ q^beta, [0.367, 1.82]

Tobs_fid = 1. #yr
