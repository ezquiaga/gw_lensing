import jax
import jax.numpy as jnp

xp = jnp

def logdiffexp(x, y): 
        return x + xp.log1p(xp.exp(y-x))

""" Useful functions  """
def powerlaw(m,mMin,mMax,alpha):
    norm = (1. + alpha)/(mMax**(alpha+1.) - mMin**(alpha+1.))
    prob = xp.power(m,alpha)
    prob *= norm
    return prob

def gaussian(x,mu,sig):
    return xp.exp(-(x-mu)**2/(2.*sig**2))/xp.sqrt(2.*xp.pi*sig**2)

def broken_powerlaw(m,mMin,mMax,m_break,alpha_1,alpha_2):
    r"""
    Broken (two-slope) power-law probability density, continuous at the break

    .. math::
        p(m) = \frac{1}{N}
        \begin{cases}
            (m/m_\mathrm{break})^{\alpha_1} & m_\mathrm{min} \leq m < m_\mathrm{break} \\
            (m/m_\mathrm{break})^{\alpha_2} & m_\mathrm{break} \leq m \leq m_\mathrm{max}
        \end{cases}

    with :math:`N` such that the density integrates to 1 over
    [mMin, mMax]. It is zero outside [mMin, mMax].

    Parameters
    ----------
    m : array_like
        Component mass
    mMin : float
        Minimum mass of the power-law support
    mMax : float
        Maximum mass of the power-law support
    m_break : float
        Break mass where the slope changes (mMin < m_break < mMax)
    alpha_1 : float
        Power-law index below the break (p ~ m^alpha_1)
    alpha_2 : float
        Power-law index above the break (p ~ m^alpha_2)

    Returns
    -------
    array_like
        Normalized broken power-law density evaluated at ``m``
    """
    # Piecewise power law, continuous at m_break by construction
    prob = xp.where(m < m_break,
                    xp.power(m/m_break,alpha_1),
                    xp.power(m/m_break,alpha_2))

    # Piecewise analytic integrals over [mMin,m_break] and [m_break,mMax]
    # (xp.array cast keeps the alpha=-1 limit from raising on python scalars)
    one_p_a1 = xp.array(1. + alpha_1)
    one_p_a2 = xp.array(1. + alpha_2)
    int_low = m_break*xp.where(one_p_a1 == 0.,
                               xp.log(m_break/mMin),
                               (1. - xp.power(mMin/m_break,one_p_a1))/one_p_a1)
    int_high = m_break*xp.where(one_p_a2 == 0.,
                                xp.log(mMax/m_break),
                                (xp.power(mMax/m_break,one_p_a2) - 1.)/one_p_a2)
    prob = prob/(int_low + int_high)
    prob = prob*((m <= mMax) & (m >= mMin))
    return prob

""" Filter functions """
def sigmoid(x,edge,width):
    #1./(1.+xp.exp(-(x-edge)/width))
    exponent = (x-edge)/width
    return jax.nn.sigmoid(exponent)

def lowfilter(m,mMin,dmMin):
    low_filter = xp.exp(-(m-mMin)**2/(2.*dmMin**2))
    low_filter = xp.where(m<mMin,low_filter,1.)
    return low_filter

def highfilter(m,mMax,dmMax):
    high_filter = xp.exp(-(m-mMax)**2/(2.*dmMax**2))
    high_filter = xp.where(m>mMax,high_filter,1.)
    return high_filter

def Sfilter(m,mMin,deltaM):
    """
    Smoothed filter function

    See Eq. B5 in https://arxiv.org/pdf/2111.03634.pdf
    """
    def f(mm,deltaMM):
        return xp.exp(deltaMM/mm + deltaMM/(mm-deltaMM))
    
    S_filter = 1./(f(m-mMin,deltaM) + 1.)
    S_filter = xp.where(m<mMin+deltaM,S_filter,1.)
    S_filter = xp.where(m>mMin,S_filter,0.)
    return S_filter

""" Log filter functions """

def loglowfilter(m,mMin,dmMin):
    low_filter = -(m-mMin)**2/(2.*dmMin**2)
    low_filter = xp.where(m<mMin,low_filter,0.)
    return low_filter

def loghighfilter(m,mMax,dmMax):
    high_filter = -(m-mMax)**2/(2.*dmMax**2)
    high_filter = xp.where(m>mMax,high_filter,0.)
    return high_filter