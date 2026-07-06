"""
Singular isothermal sphere with external shear (SIS+XS) lens model.

This module was built verifying the analytic results of Finch, Carlivati, Winn & Schechter (2002),
"Analytic expressions for mean magnification by a quadrupole gravitational lens",
https://arxiv.org/abs/astro-ph/0205489

The lens potential is (Eq. 6 of the paper)

    psi(r, theta) = b*r + (gamma/2)*r^2*cos(2*theta)

where (r, theta) are image-plane polar coordinates, b is the Einstein radius
(mass scale) and gamma is the external shear. All functions are dimensionless
in (b, gamma); magnifications are scale invariant so b=1 is a fine default.
"""

import numpy as np
from scipy.integrate import quad

"""Inverse magnification"""

def mu_inv(x, y, b=1., gamma=0.):
    """
    Inverse magnification of the SIS + external shear lens (Eq. 7).

    mu^-1(r,theta) = 1 - gamma^2 - (b/r)*(1 - gamma*cos(2*theta))

    Parameters
    ----------
    x, y : float or array-like
        Image-plane Cartesian coordinates (units of the Einstein radius b).
    b : float
        Einstein radius (mass scale) of the SIS.
    gamma : float
        External shear.

    Returns
    -------
    mu_inv : float or array-like
        Inverse (signed) magnification at the image position.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    cos2theta = (x**2 - y**2) / r2
    return 1. - gamma**2 - (b / r) * (1. - gamma * cos2theta)


"""Analytic curves (Eqs. 8-18)"""

def radial_caustic(theta, b=1.):
    """
    Radial (circular) caustic in the source plane (Eq. 8).

    Parameters
    ----------
    theta : float or array-like
        Parametrizing angle in radians.
    b : float
        Einstein radius.

    Returns
    -------
    (s_x, s_y) : tuple of arrays
        Source-plane Cartesian coordinates of the caustic.
    """
    theta = np.asarray(theta, dtype=float)
    return -b * np.cos(theta), -b * np.sin(theta)


def area_radial_caustic(b=1.):
    """Area enclosed by the radial caustic, A_r = pi*b^2 (Eq. 8)."""
    return np.pi * b**2


def transition_locus_12(theta_t, b=1., gamma=0.):
    """
    Image-plane locus separating 1-image from 2-image regions (Eq. 10),

    r_t(theta_t) = 2b*(1 - gamma*cos(2*theta_t)) / (1 - 2*gamma*cos(2*theta_t) + gamma^2)

    Parameters
    ----------
    theta_t : float or array-like
        Image-plane polar angle in radians.
    b : float
        Einstein radius.
    gamma : float
        External shear.

    Returns
    -------
    (x, y) : tuple of arrays
        Image-plane Cartesian coordinates of the locus.
    """
    theta_t = np.asarray(theta_t, dtype=float)
    c2 = np.cos(2. * theta_t)
    r_t = 2. * b * (1. - gamma * c2) / (1. - 2. * gamma * c2 + gamma**2)
    return r_t * np.cos(theta_t), r_t * np.sin(theta_t)


def area_transition_12(b=1., gamma=0.):
    """Area enclosed by the 1-2 transition locus (Eq. 11), A_t = 4*pi*b^2*(1-gamma^2/2)/(1-gamma^2)."""
    return 4. * np.pi * b**2 * (1. - gamma**2 / 2.) / (1. - gamma**2)


def critical_curve(theta_c, b=1., gamma=0.):
    """
    Critical curve in the image plane (Eq. 12),

    r_c(theta_c) = b*(1 - gamma*cos(2*theta_c)) / (1 - gamma^2)

    Parameters
    ----------
    theta_c : float or array-like
        Image-plane polar angle in radians.
    b : float
        Einstein radius.
    gamma : float
        External shear.

    Returns
    -------
    (x, y) : tuple of arrays
        Image-plane Cartesian coordinates of the critical curve.
    """
    theta_c = np.asarray(theta_c, dtype=float)
    r_c = b * (1. - gamma * np.cos(2. * theta_c)) / (1. - gamma**2)
    return r_c * np.cos(theta_c), r_c * np.sin(theta_c)


def astroid_caustic(theta_c, b=1., gamma=0.):
    """
    Astroid (tangential) caustic in the source plane (Eq. 13),

    x_a = -(2*b*gamma/(1+gamma))*cos^3(theta_c),  y_a = (2*b*gamma/(1-gamma))*sin^3(theta_c)

    The astroid lies entirely inside the radial caustic only for gamma < 1/3;
    for gamma > 1/3 its cusps poke outside (naked cusps).

    Parameters
    ----------
    theta_c : float or array-like
        Parametrizing angle (polar angle on the critical curve) in radians.
    b : float
        Einstein radius.
    gamma : float
        External shear.

    Returns
    -------
    (s_x, s_y) : tuple of arrays
        Source-plane Cartesian coordinates of the caustic.
    """
    theta_c = np.asarray(theta_c, dtype=float)
    x_a = -(2. * b * gamma / (1. + gamma)) * np.cos(theta_c)**3
    y_a = (2. * b * gamma / (1. - gamma)) * np.sin(theta_c)**3
    return x_a, y_a


def area_astroid(b=1., gamma=0.):
    """Area enclosed by the astroid caustic (Eq. 14), A_a = (3*pi/2)*b^2*gamma^2/(1-gamma^2)."""
    return (3. * np.pi / 2.) * b**2 * gamma**2 / (1. - gamma**2)


def transition_locus_24(t, b=1., gamma=0.):
    """
    Image-plane loci separating 2-image from 4-image regions (Eqs. 17-18),
    parametrized by t = cos(theta_c) in [-1, 1].

    The polar angle of the locus satisfies (Eq. 17)

    cos(theta_4) = t*(t^2 - 1 +/- sqrt(t^4 - t^2 + 1))

    ('+' root: inner locus, '-' root: outer locus) and the radius is (Eq. 18)

    r_4(t) = (b/(1-gamma)) * [1 - 2*gamma*t^3 / ((1+gamma)*cos(theta_4)(t))]

    Each branch covers the upper half plane (y >= 0) as t runs over [-1, 1];
    reflect y -> -y for the lower half. For gamma > 1/3 the inner branch
    develops r_4 < 0 (naked cusps) and the parametrization breaks down;
    points with r_4 < 0 are returned as NaN.

    Parameters
    ----------
    t : float or array-like
        Parameter t = cos(theta_c) in [-1, 1].
    b : float
        Einstein radius.
    gamma : float
        External shear.

    Returns
    -------
    (x_in, y_in, x_out, y_out) : tuple of arrays
        Image-plane Cartesian coordinates of the inner and outer loci.
    """
    t = np.asarray(t, dtype=float)
    sqS = np.sqrt(t**4 - t**2 + 1.)
    # inner ('+') branch; t^3/(sqS + 1 - t^2) == t*(t^2 - 1 + sqS), stable near t=0
    cos_in = t**3 / (sqS + 1. - t**2)
    ratio_in = sqS + 1. - t**2  # = t^3/cos(theta_4)
    # outer ('-') branch
    cos_out = t * (t**2 - 1. - sqS)
    ratio_out = t**2 / (t**2 - 1. - sqS)
    sin_in = np.sqrt(np.clip(1. - cos_in**2, 0., None))
    sin_out = np.sqrt(np.clip(1. - cos_out**2, 0., None))
    r_in = (b / (1. - gamma)) * (1. - 2. * gamma * ratio_in / (1. + gamma))
    r_out = (b / (1. - gamma)) * (1. - 2. * gamma * ratio_out / (1. + gamma))
    r_in = np.where(r_in >= 0., r_in, np.nan)
    r_out = np.where(r_out >= 0., r_out, np.nan)
    return r_in * cos_in, r_in * sin_in, r_out * cos_out, r_out * sin_out


"""I1 integral and Table 1 (valid only for gamma < 1/3)"""

def _area_locus_24(b, gamma, branch):
    """
    Area enclosed by one 2-4 transition locus, A = int_0^pi r_4(theta)^2 dtheta
    (upper half covers half the curve; the factor of 2 from the y -> -y symmetry
    cancels the 1/2 of the polar area formula). Integrated over the parameter
    t = sin(u) so the endpoint singularities of dtheta/dt are removed.
    """
    def integrand(u):
        t = np.sin(u)
        S = t**4 - t**2 + 1.
        sqS = np.sqrt(S)
        if branch == 'inner':
            c = t**3 / (sqS + 1. - t**2)
            cp = 3. * t**2 - 1. + sqS + (2. * t**4 - t**2) / sqS
            ratio = sqS + 1. - t**2
        else:
            c = t * (t**2 - 1. - sqS)
            cp = 3. * t**2 - 1. - sqS - (2. * t**4 - t**2) / sqS
            ratio = t**2 / (t**2 - 1. - sqS)
        r = (b / (1. - gamma)) * (1. - 2. * gamma * ratio / (1. + gamma))
        sin_theta = np.sqrt(max(1. - c**2, 1e-300))
        # |dtheta/dt| * dt/du, with dt/du = cos(u) cancelling the endpoint singularity
        return r**2 * abs(cp) / sin_theta * np.cos(u)
    A, _ = quad(integrand, -np.pi / 2., np.pi / 2., limit=200)
    return A


_I1_cache = {}

def I1_constant(gamma_ref=0.2):
    """
    Numerical evaluation of the I1 integral (Eqs. 19-21), I1 ~= 1.35111.

    The difference of the areas enclosed by the outer and inner 2-4 transition
    loci is A_o - A_i = 4*pi*b^2*gamma*(I1 + I2*gamma)/(1-gamma^2)^2 with I2 = 0
    identically (antisymmetric integrand), so I1 can be extracted exactly at any
    reference shear gamma_ref < 1/3 by integrating the r_4(t) parametrization.

    Parameters
    ----------
    gamma_ref : float
        Reference shear (< 1/3) at which the area difference is evaluated.
        The result is independent of this choice.

    Returns
    -------
    I1 : float
        The constant I1 ~= 1.35111.
    """
    if gamma_ref not in _I1_cache:
        A_i = _area_locus_24(1., gamma_ref, 'inner')
        A_o = _area_locus_24(1., gamma_ref, 'outer')
        _I1_cache[gamma_ref] = (A_o - A_i) * (1. - gamma_ref**2)**2 / (4. * np.pi * gamma_ref)
    return _I1_cache[gamma_ref]


def sigma2_analytic(b=1., gamma=0.):
    """
    Source-plane cross section for 2-image systems (Table 1),
    sigma_2 = pi*b^2*(1 - (5/2)*gamma^2)/(1 - gamma^2). 
    Valid only for gamma < 1/3.
    """
    return np.pi * b**2 * (1. - 2.5 * gamma**2) / (1. - gamma**2)


def sigma4_analytic(b=1., gamma=0.):
    """
    Source-plane cross section for 4-image systems (Table 1),
    sigma_4 = (3*pi/2)*b^2*gamma^2/(1 - gamma^2) = A_a. 
    Valid only for gamma < 1/3.
    """
    return area_astroid(b, gamma)


def mu2_mean_analytic(gamma):
    """
    Mean total magnification of 2-image systems (Table 1),
    <mu_2> = 4*[(1 - gamma^2/2)*(1 - gamma^2) - I1*gamma] / [(1 - (5/2)*gamma^2)*(1 - gamma^2)].
    Valid only for gamma < 1/3.
    """
    gamma = np.asarray(gamma, dtype=float)
    I1 = I1_constant()
    return 4. * ((1. - gamma**2 / 2.) * (1. - gamma**2) - I1 * gamma) \
        / ((1. - 2.5 * gamma**2) * (1. - gamma**2))


def mu4_mean_analytic(gamma):
    """
    Mean total magnification of 4-image systems (Table 1),
    <mu_4> = 8*I1/(3*gamma*(1 - gamma^2)) ~= 3.6/(gamma*(1 - gamma^2)).
    Valid only for gamma < 1/3.
    """
    gamma = np.asarray(gamma, dtype=float)
    I1 = I1_constant()
    return 8. * I1 / (3. * gamma * (1. - gamma**2))


def mu2_min(gamma):
    """
    Minimum total magnification of 2-image systems (Eq. 23),
    mu_2,min = 2/[(1 + 3*gamma)*(1 - gamma)]. 
    Valid only for gamma < 1/3.
    """
    gamma = np.asarray(gamma, dtype=float)
    return 2. / ((1. + 3. * gamma) * (1. - gamma))


def mu4_min(gamma):
    """
    Minimum total magnification of 4-image systems (Eq. 24),
    mu_4,min = 2/[gamma*(1 - gamma^2)]. 
    Valid only for gamma < 1/3.
    """
    gamma = np.asarray(gamma, dtype=float)
    return 2. / (gamma * (1. - gamma**2))


"""Image finding: quartic in r from the lens equation"""

def quartic_coefficients(sx, sy, b=1., gamma=0.):
    """
    Coefficients [c4, c3, c2, c1, c0] of the quartic in the image radius r,

    [(1-gamma)*r - b]^2 * [(1+gamma)*r - b]^2
        - sx^2*[(1+gamma)*r - b]^2 - sy^2*[(1-gamma)*r - b]^2 = 0,

    obtained by solving each component of the lens equation for x(r), y(r)
    and substituting into r^2 = x^2 + y^2. Every real positive root r (with
    nonvanishing denominators) corresponds to exactly one image.

    Parameters
    ----------
    sx, sy : float or array-like
        Source-plane Cartesian coordinates.
    b : float
        Einstein radius.
    gamma : float
        External shear.

    Returns
    -------
    [c4, c3, c2, c1, c0] : list
        Quartic coefficients, highest degree first. c4 and c3 are scalars,
        the rest have the shape of sx, sy.
    """
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    A = 1. - gamma
    B = 1. + gamma
    C = 1. - gamma**2
    c4 = C**2
    c3 = -4. * b * C
    c2 = 2. * b**2 * (2. + C) - (sx**2 * B**2 + sy**2 * A**2)
    c1 = -4. * b**3 + 2. * b * (sx**2 * B + sy**2 * A)
    c0 = b**4 - (sx**2 + sy**2) * b**2
    return [c4, c3, c2, c1, c0]


def find_images(sx, sy, b=1., gamma=0., tol=1e-6, resid_tol=1e-2):
    """
    Find all images of a single source by solving the quartic in r.

    A root is kept if it is real (imaginary part below `tol`), positive,
    and self-consistent: substituting it back into x=sx*r/den_x,
    y=sy*r/den_y must satisfy r^2=x^2+y^2 (true by construction for any
    exact root of the quartic) to within `resid_tol` (relative). 
    
    The consistency tolerance is deliberately looser than `tol`: near the
    coordinate axes (sx=0 or sy=0) the relevant root is a perturbed
    double root (den_x or den_y -> 0 together with sx or sy), which is
    ill-conditioned in double precision -- root errors of order
    sqrt(machine epsilon) are expected there even though the root is
    genuine, so a strict residual tolerance would wrongly discard real
    images. `resid_tol=1e-2` still keeps images accurate to ~1% or
    better; only sources within roughly 1e-6*b of an axis (a
    negligible fraction of source-plane area) remain unresolved, an
    inherent floating-point limit rather than a logic error.

    Parameters
    ----------
    sx, sy : float
        Source-plane Cartesian coordinates.
    b : float
        Einstein radius.
    gamma : float
        External shear.
    tol : float
        Tolerance (in units of b) for accepting a root as real and positive.
    resid_tol : float
        Relative tolerance on the r^2=x^2+y^2 self-consistency check.

    Returns
    -------
    images : ndarray, shape (n_images, 3)
        One row (x, y, mu) per image: image-plane position and signed
        magnification (Eq. 7).
    """
    A = 1. - gamma
    B = 1. + gamma
    roots = np.roots(quartic_coefficients(sx, sy, b, gamma))
    images = []
    for root in roots:
        if abs(root.imag) > tol * b:
            continue
        r = root.real
        if r <= 1e-12 * b:
            continue
        den_x = A * r - b
        den_y = B * r - b
        with np.errstate(divide='ignore', invalid='ignore'):
            x = sx * r / den_x
            y = sy * r / den_y
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if abs(x**2 + y**2 - r**2) > resid_tol * r**2:
            continue
        mu = 1. / mu_inv(x, y, b, gamma)
        images.append([x, y, float(mu)])
    return np.array(images).reshape(-1, 3)


def find_images_batch(sx, sy, b=1., gamma=0., tol=1e-6, resid_tol=1e-2, return_images=False):
    """
    Vectorized image finder for many sources at once.

    Builds the companion matrix of the (monic) quartic in r for every source
    and diagonalizes them all with a single broadcast call to
    numpy.linalg.eigvals, avoiding a Python loop over numpy.roots.

    A root is kept if it is real (imaginary part below `tol`), positive,
    and self-consistent: r^2=x^2+y^2 must hold (true by construction for
    any exact root) to within `resid_tol` (relative). `resid_tol` is
    deliberately looser than `tol`: near the coordinate axes (sx=0 or
    sy=0) the relevant root is a perturbed double root, ill-conditioned
    in double precision, so a strict residual would wrongly discard real
    images there -- see `find_images` for the full explanation. Only
    sources within roughly 1e-6*b of an axis remain unresolved, an
    inherent floating-point limit rather than a logic error.

    Parameters
    ----------
    sx, sy : array-like, shape (N,)
        Source-plane Cartesian coordinates.
    b : float
        Einstein radius.
    gamma : float
        External shear.
    tol : float
        Tolerance (in units of b) for accepting a root as real and positive.
    resid_tol : float
        Relative tolerance on the r^2=x^2+y^2 self-consistency check.
    return_images : bool
        If True, also return the image positions, magnifications and
        validity mask.

    Returns
    -------
    n_images : ndarray, shape (N,)
        Number of images (1, 2, 3 or 4) of each source.
    mu_total : ndarray, shape (N,)
        Total unsigned magnification sum_i |mu_i| of each source.
    x, y, mu, valid : ndarrays, shape (N, 4), optional
        If return_images is True: image positions, signed magnifications and
        the mask selecting the physical roots.
    """
    sx = np.atleast_1d(np.asarray(sx, dtype=float))
    sy = np.atleast_1d(np.asarray(sy, dtype=float))
    N = sx.size
    A = 1. - gamma
    B = 1. + gamma
    c4, c3, c2, c1, c0 = quartic_coefficients(sx, sy, b, gamma)
    # monic coefficients r^4 + a3 r^3 + a2 r^2 + a1 r + a0
    a3 = c3 / c4
    a2 = c2 / c4
    a1 = c1 / c4
    a0 = c0 / c4
    # stacked companion matrices, shape (N, 4, 4)
    comp = np.zeros((N, 4, 4))
    comp[:, 1, 0] = 1.
    comp[:, 2, 1] = 1.
    comp[:, 3, 2] = 1.
    comp[:, 0, 3] = -a0
    comp[:, 1, 3] = -a1
    comp[:, 2, 3] = -a2
    comp[:, 3, 3] = -a3
    roots = np.linalg.eigvals(comp)  # (N, 4), complex
    r = roots.real
    den_x = A * r - b
    den_y = B * r - b
    with np.errstate(divide='ignore', invalid='ignore'):
        x = sx[:, None] * r / den_x
        y = sy[:, None] * r / den_y
        mu = 1. / mu_inv(x, y, b, gamma)
    # Reject only genuine numerical garbage, not the removable singularity
    # where a small denominator is paired with a proportionally small
    # numerator (e.g. sources near sx=0 or sy=0): any exact root of the
    # quartic satisfies r^2=x^2+y^2 by construction, so a large mismatch
    # flags a spurious root.
    consistent = np.abs(x**2 + y**2 - r**2) <= resid_tol * r**2
    valid = (np.abs(roots.imag) < tol * b) & (r > 1e-12 * b) \
        & np.isfinite(x) & np.isfinite(y) & consistent
    n_images = valid.sum(axis=1)
    mu_total = np.where(valid, np.abs(mu), 0.).sum(axis=1)
    if return_images:
        return n_images, mu_total, x, y, mu, valid
    return n_images, mu_total
