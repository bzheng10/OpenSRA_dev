import numpy as np
from numba import njit, float64, int64


@njit(
    float64[:](float64[:]),
    fastmath=True,
    cache=True,
)
def erf2(x):
    """modified from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf"""
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # Save the sign of x
    signs = np.sign(x)
    x = np.abs(x)
    # A & S 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return signs*y
    
    
@njit(
    float64[:,:](float64[:,:]),
    fastmath=True,
    cache=True,
)
def erf2_2d(x):
    """modified from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf"""
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # Save the sign of x
    signs = np.sign(x)
    x = np.abs(x)
    # A & S 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return signs*y


@njit(
    float64[:](int64),
    fastmath=True,
    cache=True,
)
def erfinv_coeff(order=20):
    # initialize
    c = np.empty(order+1)
    # starting value
    c[0] = 1
    for i in range(1,order+1):
        c[i] = sum([c[j]*c[i-1-j]/(j+1)/(2*j+1) for j in range(i)])
    # return
    return c


@njit(
    float64[:](float64[:],int64),
    fastmath=True,
    cache=True,
)
def erfinv(x, order=20):
    """returns inverse erf(x)"""
    # get coeffcients
    c = erfinv_coeff(order)
    # initialize
    root_pi_over_2 = np.sqrt(np.pi)/2
    y = np.zeros(x.shape)
    for i in range(order):
        y += c[i]/(2*i+1)*(root_pi_over_2*x)**(2*i+1)
    # return
    return y


@njit(
    float64[:,:](float64[:,:],int64),
    fastmath=True,
    cache=True,
)
def erfinv_2d(x, order=20):
    """returns inverse erf(x)"""
    # get coeffcients
    c = erfinv_coeff(order)
    # initialize
    root_pi_over_2 = np.sqrt(np.pi)/2
    y = np.zeros(x.shape)
    for i in range(order):
        y += c[i]/(2*i+1)*(root_pi_over_2*x)**(2*i+1)
    # return
    return y


@njit(
    float64[:](float64[:], float64, float64),
    fastmath=True,
    cache=True
)
def norm2_ppf(p, loc, scale):
    """
    modified implementation of norm.ppf function from numba_stats, using self-implemented erfinv function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = np.sqrt(2) * erfinv(2*p-1,order=20)
    return scale * inter + loc

@njit(
    float64[:,:](float64[:,:], float64, float64),
    fastmath=True,
    cache=True
)
def norm2_ppf_2d(p, loc, scale):
    """
    modified implementation of norm.ppf function from numba_stats, using self-implemented erfinv function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = np.sqrt(2) * erfinv_2d(2*p-1,order=20)
    return scale * inter + loc


@njit(
    float64[:](float64[:], float64, float64),
    fastmath=True,
    cache=True
)
def norm2_cdf(x, loc, scale):
    """
    modified implementation of norm.cdf function from numba_stats, using self-implemented erf function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = (x - loc)/scale
    return 0.5 * (1 + erf2(inter * np.sqrt(0.5)))


@njit(
    float64[:,:](float64[:,:], float64, float64),
    fastmath=True,
    cache=True
)
def norm2_cdf_2d(x, loc, scale):
    """
    modified implementation of norm.cdf function from numba_stats, using self-implemented erf function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = (x - loc)/scale
    return 0.5 * (1 + erf2_2d(inter * np.sqrt(0.5)))


@njit(
    float64[:,:](float64[:,:], float64[:], float64[:], float64[:], float64[:]),
    fastmath=True,
    cache=True
)
def truncnorm2_ppf_2d(p, loc, scale, xmin, xmax):
    """
    modified implementation of norm.cdf function from numba_stats, using self-implemented erf function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/truncnorm.py
    """
    # dims
    shape = p.shape
    dim1 = shape[0]
    dim2 = shape[1]
    # expand dim to 2d
    loc_2d = loc.repeat(dim2).reshape((-1, dim2))
    scale_2d = scale.repeat(dim2).reshape((-1, dim2))
    # calcs
    inv_scale = 1 / scale
    zmin = (xmin - loc) * inv_scale
    zmax = (xmax - loc) * inv_scale
    pmin = 0.5 * (1 + erf2(zmin * np.sqrt(0.5)))
    pmax = 0.5 * (1 + erf2(zmax * np.sqrt(0.5)))
    # expand dim to 2D
    pmin_2d = pmin.repeat(dim2).reshape((-1, dim2))
    pmax_2d = pmax.repeat(dim2).reshape((-1, dim2))
    # more calc
    r = p * (pmax_2d - pmin_2d) + pmin_2d
    r = np.sqrt(2) * erfinv_2d(2*r-1,order=20)
    return scale_2d * r + loc_2d