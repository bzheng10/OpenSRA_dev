# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Common functions
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import os
import sys
import numpy as np
import rasterio as rio
from numba import jit, njit
from scipy import sparse, stats
from scipy.interpolate import interp2d


# -----------------------------------------------------------
def expand_dims(val):
    """expand dimensions to 1D of number of dimensions is 0"""
    if np.ndim(val) == 0:
        return np.expand_dims(val,axis=0)
    else:
        return val
    
    
# -----------------------------------------------------------
def to_float(val):
    """apply float to data array"""
    return np.asarray(val, dtype=float)


# -----------------------------------------------------------
def from_dict_to_array(data_dict, loc_dict, shape, fill_zeros=0, decimals=1):
    """converts a dictionary to array, where keys of dictionary are rows and entries are columns"""
    data = np.hstack([data_dict[key] for key in data_dict.keys()])
    row = np.hstack([[key]*len(loc_dict[key]) for key in loc_dict.keys()])
    col = np.hstack([loc_dict[key] for key in loc_dict.keys()])
    array = sparse.coo_matrix((data-1e5, (row, col)), shape=shape).toarray() # -1e5 in case some distances in data are already 0
    array[array==0] = fill_zeros # set zero entries to fill value
    array = array + 1e5 # undo 1e5
    array[array==fill_zeros+1e5] = fill_zeros
    array[array<fill_zeros] = np.round(array[array<fill_zeros],decimals=decimals)
    return array


# -----------------------------------------------------------
def smart_round(number, decimal=None):
    """Determine number decimals to round to base on input and round the number"""
    if decimal is None:
        n_decimal = int(abs(np.floor(np.log10(number))))
        n_decimal_total = decimal_count(number)
        round_low = np.round(number,n_decimal)
        if n_decimal == n_decimal_total:
            return round_low
        while n_decimal < n_decimal_total:
            n_decimal += 1
            round_curr = np.round(number,n_decimal)
            if np.equal(round_low,round_curr):
                return round_low
            round_low = round_curr
        return round_curr
    else:
        return np.round(number,decimal)


# -----------------------------------------------------------
def decimals(number):
    """determine number of decimals after rounding"""
    return int(abs(np.round(np.log(number)/np.log(10))))


# -----------------------------------------------------------
def decimal_count(number):
    """Count number of decimal places"""
    return len(str(number)[str(number).find('.')+1:])


# -----------------------------------------------------------
def get_basename_without_extension(path):
    """returns basename of path without extention"""
    return os.path.splitext(os.path.basename(path))[0]


# -----------------------------------------------------------
@jit
def make_grid(x_min, y_min, x_max, y_max, dx, dy):
    """given min/max for x and y, and dx and dy, make grid"""
    grid = [
        ([
            [x0,    y0      ],
            [x0,    y0+dy   ],
            [x0+dx, y0+dy   ],
            [x0+dx, y0      ]
        ]) \
        for x0 in np.arange(x_min, x_max+dx, dx) \
        for y0 in np.arange(y_min, y_max+dy, dy)]
    return grid


# -----------------------------------------------------------
# @jit
def wgs84_to_utm(lon, lat, include_north_south=False, force_zone_num=10):
    """
    using eqs from Snyder (1987) Map Projections - A Working Manual, USGS Professional Paper
    eq numbers 8-9 through 8-15
    """
    
    # for California, zone_num = 10

    # basic params
    a = 6378137 # m, semi major
    b = 6356752 # m, semi minor
    k0 = 0.9996 # correction
    pi = np.pi
    
    # convert coords to radian
    lat_rad = lat * pi / 180
    lon_rad = lon * pi / 180
    
    # get zone number
    if force_zone_num == -99:
        if isinstance(lon,float):
            zone = min(int((lon+180)/6+1), 60)
        else:
            zone = np.asarray([min(int((val+180)/6+1), 60) for val in lon])
    else:
        if isinstance(lon,float) or isinstance(lon,int):
            zone = force_zone_num
        else:
            zone = np.ones(lon.shape)*force_zone_num
    
    # get coords at central meridian of corresponding zone
    central_lat_rad = 0 # lat at central meridian
    central_lon = 6 * zone - 180 - 3 # lon at central meridian
    central_lon_rad = central_lon * pi / 180
    
    # precompute parameters for cleaner codes
    e2 = (a**2 - b**2) / a**2 # eccentricity
    e4 = e2**2
    e6 = e2*e4
    ep2 = e2 / (1 - e2)
    N_val = a / np.sqrt(1 - e2*np.sin(lat_rad)**2)
    T_val = np.tan(lat_rad)**2
    C_val = ep2 * np.cos(lat_rad)**2
    A_val = (lon_rad - central_lon_rad) * np.cos(lat_rad)
    
    # compute true distance from equator
    M_term1 = (1 - e2/4 - 3*e4/64 - 5*e6/256) * lat_rad
    M_term2 = (3*e2/8 + 3*e4/32 + 45*e6/1024) * np.sin(2*lat_rad)
    M_term3 = (15*e4/256 + 45*e6/1024) * np.sin(4*lat_rad)
    M_term4 = (35*e6/3072) * np.sin(6*lat_rad)
    M_val = a * (M_term1 - M_term2 + M_term3 - M_term4) # true distance from equator
    M_val_central = 0 # true distance at central median = 0 because central_lat = 0

    # compute easting
    x_term1 = A_val
    x_term2 = (1 - T_val + C_val) * A_val**3 / 6
    x_term3 = (5 - 18*T_val + T_val**2 + 72*C_val - 58*ep2) * A_val**5 / 120
    x = k0 * N_val * (x_term1 + x_term2 + x_term3)
    x = x + 500000 # easting correction
    
    # compute northing
    y_term1 = A_val**2 / 2
    y_term2 = (5 - T_val + 9*C_val + 4*C_val**2) * A_val**4 / 24
    y_term3 = (61 - 58*T_val + T_val**2 + 600*C_val - 330*ep2) * A_val**6 / 720
    y = k0 * (M_val - M_val_central + N_val * np.tan(lat_rad) * (y_term1 + y_term2 + y_term3))

    # further refine northing into "Northern" or "Southern" hemisphere
    if include_north_south is True and force_zone_num == -99:
        if lat < 0:
            hemi = "South"
            y = y + 10000000
        else:
            hemi = "North"
    else:
        hemi = None

    # output
    return x, y, zone, hemi


# -----------------------------------------------------------
# @jit
def utm_to_wgs84(x, y, zone=10):
    """
    using eqs from Snyder (1987) Map Projections - A Working Manual, USGS Professional Paper
    eq numbers 8-17 through 8-25
    """
    
    # for California, zone_num = 10

    # basic params
    a = 6378137 # m, semi major
    b = 6356752 # m, semi minor
    k0 = 0.9996 # correction
    pi = np.pi
    
    # get coords at central meridian of corresponding zone
    central_lat_rad = 0 # lat at central meridian
    central_lon = 6 * zone - 180 - 3 # lon at central meridian
    central_lon_rad = central_lon * pi / 180
    
    # precompute parameters for cleaner codes
    e2 = (a**2 - b**2) / a**2 # eccentricity
    ecc = np.sqrt(e2)
    e4 = e2**2
    e6 = e2*e4
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    e1_2 = e1**2
    e1_3 = e1_2 * e1
    e1_4 = e1_3 * e1
    M_val_central = 0 # true distance at central median = 0 because central_lat = 0
    M_val = M_val_central + y / k0 # true distance from equator
    mu = M_val / (a * (1 - e2/4 - 3*e4/64 - 5*e6/256))
    
    # intermediate calc
    lat1_rad_term1 = mu
    lat1_rad_term2 = (3*e1/2 - 27*e1_3/32) * np.sin(2*mu)
    lat1_rad_term3 = (21*e1_2/16 - 55*e1_4/32) * np.sin(4*mu)
    lat1_rad_term4 = (151*e1_3/96) * np.sin(6*mu)
    lat1_rad_term5 = (1097*e1_4/512) * np.sin(8*mu)
    lat1_rad = lat1_rad_term1 + lat1_rad_term2 + lat1_rad_term3 + lat1_rad_term4 + lat1_rad_term5
    
    # more intermediate clac
    ep2 = e2 / (1 - e2)
    C1_val = ep2 * np.cos(lat1_rad)**2
    T1_val = np.tan(lat1_rad)**2
    N1_val = a / np.sqrt(1 - e2*np.sin(lat1_rad)**2)
    R1_val = a * (1 - e2) / (1 - e2*np.sin(lat1_rad)**2) ** (3/2)
    D_val = (x - 500000) / (N1_val*k0)

    # compute lat
    lat_rad_term1 = D_val**2/2
    lat_rad_term2 = (5 + 3*T1_val + 10*C1_val - 4*C1_val**2 - 9*ep2) * D_val**4/24
    lat_rad_term3 = (61 + 90*T1_val + 298*C1_val + 45*T1_val**2 - 252*ep2 - 3*C1_val**2) * D_val**6/720
    lat_rad = lat1_rad - (N1_val * np.tan(lat1_rad) / R1_val) * (lat_rad_term1 - lat_rad_term2 + lat_rad_term3)
    
    # compute lon
    lon_rad_term1 = D_val
    lon_rad_term2 = (1 + 2*T1_val + C1_val) * D_val**3/6
    lon_rad_term3 = (5 - 2*C1_val + 28*T1_val - 3*C1_val**2 + 8*ep2 + 24*T1_val**2) * D_val**5/120
    lon_rad = central_lon_rad + (lon_rad_term1 - lon_rad_term2 + lon_rad_term3) / np.cos(lat1_rad)
    
    # convert to degrees
    lat = lat_rad * 180 / pi
    lon = lon_rad * 180 / pi
    
    # output
    return lon, lat


# -----------------------------------------------------------
# @jit
def get_midpoint(lon1, lat1, lon2, lat2):
    """
    returns midpoint given two coordinates
    
    """
    
    # the longitudes or latitutes are equal, then just return halfway point
    # if lon1 == lon2:
        # lon_mid = lon1
        # lat_mid = (lat1+lat2)/2
    # elif lat1 == lat2:
        # lon_mid = (lon1+lon2)/2
        # lat_mid = lat1
    # if longitudes and latitudes are not equal, apply trig
    # else:
    # convert long lat from degrees to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    # get delta in longitude
    d_lon = lon2-lon1
    # intermediate calculations
    Bx = np.cos(lat2) * np.cos(d_lon)
    By = np.cos(lat2) * np.sin(d_lon)
    # lat and lon for mid points
    lon_mid = lon1 + np.arctan2(By, np.cos(lat1) + Bx)
    lat_mid = np.arctan2(np.sin(lat1)+np.sin(lat2), np.sqrt((np.cos(lat1)+Bx)*(np.cos(lat1)+Bx) + By*By))
    # convert back to degrees
    lon_mid = np.degrees(lon_mid)
    lat_mid = np.degrees(lat_mid)
    #
    return lon_mid, lat_mid


# -----------------------------------------------------------
# @jit
def get_quick_dist(lon1, lat1, lon2, lat2):
    """
    Get quick (crude) distance between two points
    
    """
    
    return ((lon1-lon2)**2 + (lat1-lat2)**2) ** (0.5)


# -----------------------------------------------------------
# def lhs(n_var, n_samp, dist=['normal'], low=None, high=None, return_prob=False):
def lhs(n_site, n_var, n_samp, dist='normal', low=None, high=None, return_prob=False):
    """
    Performs Latin-Hypercube Sampling and returns both the cdfs and the residuals for the user-specified distribution.
    
    Allowed distribution types: **normal**(default), **truncated_normal**, **uniform**
    
    """
    
    # permutation of bins
    # boxes = np.transpose([np.random.permutation(n_samp) for i in range(n_var)])
    boxes = np.asarray([np.transpose([np.random.permutation(n_samp) for i in range(n_var)]) for j in range(n_site)])
    # draw uniform samples from 0 to 1, add to bin permutations, and normalize by sample size to get cdfs
    norm_uniform_samples = np.random.uniform(size=(n_site,n_samp,n_var))
    cdfs = (boxes+norm_uniform_samples)/n_samp
    # residuals
    if 'norm' in dist.lower() and not 'trunc' in dist.lower():
        res = stats.norm.ppf(cdfs)
        # if return_prob:
            # probs = stats.norm.pdf(res)
    elif 'trunc' in dist.lower():
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        res = stats.truncnorm.ppf(cdfs,low,high)
        # if return_prob:
            # probs = stats.truncnorm.pdf(res,low,high)
    elif 'uniform' in dist.lower():
        if low is None:
            low = 0
        if high is None:
            high = 1
        res = cdfs*(high-low)+low
        # if return_prob:
            # probs = np.ones(res.shape)/n_samp
    #
    # if return_prob:
        # return res, probs
    # else:
    return res


# -----------------------------------------------------------
def interp_from_raster(raster_path, x, y,
    interp_scheme='linear', out_of_bound_value=0, invalid_value=9999):
    """
    Imports a raster and performs 2D interpolation at (x,y) pairs. Accepted interp_scheme = 'nearest', 'linear', 'cubic', and 'quintic'
    
    """
    if os.path.exists(raster_path):
        # read raster
        ras = rio.open(raster_path)
        # get gridded data and make x and y grid vectors
        data = ras.read(1)
        # if user wants 'nearest':
        if interp_scheme == 'nearest':
            samples = np.array([x[0] for x in ras.sample(np.vstack(x,y).T)])
        else:
            x_vect = np.linspace(ras.bounds.left, ras.bounds.right, ras.width, endpoint=False)
            y_vect = np.linspace(ras.bounds.bottom, ras.bounds.top, ras.height, endpoint=False)
            # create interp2d function
            interp_function = interp2d(
                x_vect, y_vect, np.flipud(data),
                kind=interp_scheme, fill_value=out_of_bound_value)
            # get samples
            samples = np.transpose([interp_function(x[i],y[i]) for i in range(len(x))])[0]
        ras.close()
        # clean up invalid values (returned as 1e38 by NumPy)
        # samples[abs(samples)>1e10] = np.nan
        samples[abs(samples)>1e10] = -9999
        # return
        # logging.info(f'\t\t{os.path.basename(raster_path)}')
        return samples
    else:
        logging.info(f'\t\t{os.path.basename(raster_path)} does not exist')
        return None
        
        
# -----------------------------------------------------------
def make_dir(target_dir):
    """
    Check if folder/directory exists; if not, then create it
    
    """
    if os.path.isdir(target_dir) is False: # check if directory exists
        os.mkdir(target_dir) # create folder if it doesn't exist
        
        
# -----------------------------------------------------------
def set_logging(level, file=None, msg_format='simple'):
    """
    This method sets the logging level and formatting of the logs
    """
    
    # setting logging level
    logging.shutdown()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if level.lower() == 'debug' else logging.INFO)

    # setting log format for print
    handlerStream = logging.StreamHandler(sys.stdout)
    if msg_format == 'full':
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    elif msg_format == 'simple':
        formatter = logging.Formatter('%(message)s')
    handlerStream.setFormatter(formatter)
    logger.addHandler(handlerStream)
    
    # setting log format for save
    if file is not None:
        handlerFile = logging.FileHandler(file, mode='w')
        handlerFile.setFormatter(formatter)
        logger.addHandler(handlerFile)


# -----------------------------------------------------------
def check_common_member(a, b):
    """
    Check for common members between two lists
    
    """
    
    a_set = set(a) 
    b_set = set(b) 
    
    if (a_set & b_set): 
        return True
    else: 
        return False
    

# -----------------------------------------------------------
def get_closest_pt(loc, line):
    """
    returns the closest location and shortest distance between a location and a line segment

    Parameters
    ----------
    loc : [float, float, float (optional)]
        [degree, degree, km] coordinate (lon, lat) of point; depth is optional (default = 0)
    line : [[float, float, float], [float, float, float]]
        [degree, degree, km] two pairs of coordinates [pt1, pt2] that define the line; depth is optional (default = 0)
    
    Returns
    -------
    closest_loc : [float, float, float]
        [degree, degree, km] closest point on line
    shortest_dist : [float, float]
        [km] shorest distance
    
    """
    
    # convert to NumPy arrays
    loc = np.asarray(loc)
    line = np.asarray(line)
    
    # check if site location contains depth, if not, assign 0 as depth
    if len(loc) < 3:
        loc = np.hstack([loc,0])
    
    # see if line segment given contains depth, if not, assign 0 as depth
    if len(line[0]) < 3:
        line = np.vstack([line.T,[0,0]]).T

    # calculations
    line_vect = line[1] - line[0] # vector from pt1 to pt2
    loc_vect = loc - line[0] # get vector from pt1 of line to site
    len_line = np.dot(line_vect,line_vect)**0.5 # length of line
    
    # check if length of line is 0
    if len_line == 0:
        closest_loc = line[0] # if length == 0, then line segment is a point, which is also the closest point
    else:
        proj_dist = np.dot(loc_vect,line_vect/len_line) # compute projected length of pt on line
        
        # check if point is within or outside of line:
        if proj_dist <= 0:
            closest_loc = line[0]
        elif proj_dist >= len_line:
            closest_loc = line[1]
        else:
            closest_loc = np.dot(loc_vect,line_vect)/np.dot(line_vect,line_vect) * line_vect + line[0]
    
    # approximate shortest distance by first calculating the Haversine distance, then adding to depth
    shortest_dist = get_haversine_dist(loc[0],loc[1],closest_loc[0],closest_loc[1])
    shortest_dist = (shortest_dist**2 + closest_loc[2]**2)**0.5
    
    #
    return closest_loc, shortest_dist


# -----------------------------------------------------------
@jit
def get_haversine_dist(lon1, lat1, lon2, lat2, unit='km'):
    """
    calculates the Haversine distance between two sets of coordinates
    
    Parameters
    ----------
    lon1 : float, array
        [degree] longitude of site 1
    lat1 : float, array
        [degree] latitude of site 1
    lon2 : float, array
        [degree] longitude of site 2
    lat2 : float, array
        [degree] latitude of site 2
    unit : str
        unit for output: '**km**' or '**miles**'
    
    Returns
    -------
    d : float, array
        distance
    
    """
    
    # determine unit to reference for Earth's radius
    if 'k' in unit.lower():
        r = 6371 # km
    elif 'mi' in unit.lower():
        r = 3958.8 # miles
    
    # convert long lat from degrees to radians
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    
    # Haversine function for epicentral distance
    d = 2*r*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
    
    #
    return d


# -----------------------------------------------------------
def get_integration(y, x=None):
    """
    computes integration of a vector; currently performed with trapezoidal integration
    
    Parameters
    ----------
    y : float, array
        values to perform integration on
    x : float, array
        time array to compute time step from
    
    Returns
    -------
    
    inte_y : float, array
        integration of y
    
    """
    
    # check if x values are given (non-uniform time step)
    if x is None:
        x = np.array(range(len(y))) # time-step = 1
    
    # integrate by loop
    # initialize array
    # inte_y = np.zeros(len(y))
    # for i in range(1,len(y)):
        # inte_y[i] = inte_y[i-1] + (y[i-1] + y[i])/2 * (x[i]-x[i-1])
        # inte_y[i] = _integrate_interim(x[i-1], x[i], y[i-1], y[i], inte_y[i-1])
    
    # numpy array operation
    inte_y = (y[1:] + y[:-1])/2 * (x[1:]-x[:-1]) # compute rate per increment
    inte_y = np.cumsum(inte_y) # cumulative summation over array
    inte_y = np.hstack([0,inte_y]) # pad 0 to initial value
    
    # in case integration goes beyond 1 (need better way of checking for total sum)
    # if inte_max is not None:
        # inte_y = np.clip(inte_y,None,inte_max)
    
    #
    return inte_y


# -----------------------------------------------------------
# @jit
def _integrate_interim(x1, x2, y1, y2, inte_y1):
    """ Placeholder """
    return inte_y1 + (y1 + y2)/2 * (x2-x1)


# -----------------------------------------------------------
def count_in_range(list_to_count, a=-np.inf, b=np.inf, flag_include_a=True, flag_include_b=True):
    """
    counts the number of instances in a list between lower and upper limits, inclusive or exclusive
    
    Parameters
    ----------
    list_to_count : float, array
        list of values
    a : float, optional
        lower limit; default to **-inf**
    b : float, optional
        upper limit; default to **inf**
    flag_include_a : boolean, optional
        include **a** in count; default = True
    flag_include_b : boolean, optional
        include **b** in count; default = True
    
    Returns
    -------
    
    count : float
        number of instances
    
    """
    
    # upper limit
    if flag_include_a is True:
        count_a = sum([1 if i >= a else 0 for i in list_to_count])
    else:
        count_a = sum([1 if i > a else 0 for i in list_to_count])
        
    # lower limit
    if flag_include_b is True:
        count_b = sum([1 if i > b else 0 for i in list_to_count])
    else:
        count_b = sum([1 if i >= b else 0 for i in list_to_count])
        
    #
    return count_a-count_b


# -----------------------------------------------------------
def sort_list(list_to_sort, col):
    """
    Sorts a list of **m** entries and each entry with **n** values and sort by the column **col** relative to **m**
    
    Parameters
    ----------
    list_to_sort : float, array
        list of values
    col : int
        column to sort
    
    Returns
    -------
    
    list_sort : float, array
        sorted list
    
    """
    
    #
    return(sorted(list_to_sort, key = lambda x: x[col]))


# -----------------------------------------------------------
def generate_randon_combinations(nBin,nVar):
    """
    get random combinations, for LHS
    
    """
    # initialize array
    comb = np.zeros((nBin, nVar))
    
    # find permunation given number of bins
    comb[0:,0] = np.random.permutation(np.arange(nBin))
    
    # loop and determine corresponding pairs, remove permutation when already selected
    for i in range(1,nVar):
        for j in range(nBin):
            choice = np.arange(nBin)
            usedVal = np.unique(np.append(np.transpose(comb[j,0:i]),comb[0:j,i]))
            ind2del = [next(ii for ii, jj in enumerate(choice) if jj == kk) for kk in usedVal]
            choice2 = np.delete(choice,ind2del)
            comb[j,i] = np.random.permutation(choice2)[0]
            
    #
    return comb


# -----------------------------------------------------------
def convert_triu_to_sym_mat(x, n, matrix_type='sparse'):
    """
    Converts a 1D array of TriU elements to a symmetric matrix, e.g., 3 elements -> 2x2 matrix, 6 elements -> 3x3 matrix
    
    Parameters
    ----------
    x : float, array
        1D array of n! elements, where elements should be ordered left-right, top-down, e.g., for a 3x3 matrix, 1D array consists of components (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    n : int
        dimension of the symmetric matrix
    matrix_type : str, optional
        format to store matrix; **sparse** (default) or **ndarray**
        
    Returns
    -------
    mat : float, matrix
        symmetric matrix
    
    """
    
    if matrix_type == 'ndarray':
        # initialize mat
        mat = np.zeros((n,n))
        # set upper triangle to values in x
        mat[np.triu_indices(n)] = x
        # add transpose of matrix and subtract diagonal to get symmetric matrix
        mat = mat + mat.T - np.diag(np.diag(mat))
    else:
        # get indices for triu order
        ind1,ind2 = fast_triu_indices(n)
        # set upper triangle to values in x
        mat = sparse.coo_matrix((x, (ind1,ind2)))
        # add transpose of matrix and subtract diagonal to get symmetric matrix
        mat = mat + mat.transpose() - sparse.diags(mat.diagonal())
    
    #
    return mat
    
    
    # -----------------------------------------------------------
def get_rate_of_exceedance(x, y, rate=None):
    """
    Count the number of instances in **y** where **y** > **i** for every **i** in **x**. Probability = count/total. 
    
    Parameters
    ----------
    x : float, array
        an array of values to get probability of exceendance for
    y : float, array
        sample or population as a function of **x**
    
    Returns
    -------
    prob : float
        [%] probability of exceedance
    count : int
        number of instances of exceedance
    
    """
    
    #
    if len(y) > 0:
        # count number of instances where y > i for every i in x
        count = [sum(map(lambda val: val>i, y)) for i in x]
        
        # calculate probability = count/total
        prob = np.divide(count,len(y)) * 100
        
    else:
        count = np.zeros(1,dtype='int32')
        prob = np.zeros(1)
    
    #
    return prob, count


# -----------------------------------------------------------
def get_weighted_average(x, weights):
    """
    Calculate the weighted average for **x** values and their corresponding **weights**
    
    Parameters
    ----------
    x : float, array
        array of unweighted values
    weights : float, array
        list of weights corresponding to x
    
    Returns
    -------
    sum_weighted : float
        sum of weighted x-values
    avg_weighted : float
        weighted average of x-values
    
    """
    
    # make sure the shape of x and weights are the same (horizontal or vertical)
    if len(x[0]) != len(weights):
        x = np.transpose(x)
        flag_transpose = True
    
    # calcualte weighted x values
    x_weighted = np.multiply(x, weights)
    
    # if trannsposed previously, undo transpose for consistency with input
    if flag_transpose is True:
        x_weighted = np.transpose(x_weighted)
    
    # sum up weighted x values and dividy by total of weights
    sum_weighted = sum(x_weighted)
    avg_weighted = sum_weighted/sum(weights)
    
    #
    return sum_weighted, avg_weighted


# -----------------------------------------------------------
def get_conditional_rate_of_exceedance(xbins, x, y, ycriteria, rate=None):
    """
    Count the number of instances in **y_red** where **y** > **i** for every **i** in **x**. Take the number of instances and sort them into **xbinx**. Probability = count/total. 
    
    Parameters
    ----------
    xbins : float, array
        an array of bins to group x values (e.g., M = 5, 6, 6.5 for bins 5-6 and 6-6.5)
    x : float, array
        an array of values to get probability of exceendance for
    y : float, array
        sample or population as a function of **x**
    
    ycriteria : float
        the criteria in prob(y > ycriteria)
    
    Returns
    -------
    prob : float
        [%] conditional probability of exceedance
    
    """
    
    # find indices where y exceeds criteria and reduce the x and y arrays
    ind_red = np.where(y > ycriteria)
    y_red = y[ind_red]
    x_red = x[ind_red]
    
    #
    if len(y_red) > 0:
    
        # get counts for bins
        count = []
        for i in range(len(xbins)-1):
            ind2search = np.where((x_red > xbins[i]) & (x_red < xbins[i+1]))
            y2search = y_red[ind2search]
            _,j = get_prob_exceed([ycriteria], y2search)
            count.append(j[0])
    
        # sum up number of counts to get probability of exceedance
        count = [sum(count[0:i]) for i in range(len(count))]
        
        # calculate probability = count/total
        prob = np.divide(count,len(y_red))
        
    else:
        count = np.zeros(1,dtype='int32')
        prob = np.zeros(len(xbins)-1)*100
    
    #
    return prob


# -----------------------------------------------------------
def fast_triu_indices(dim, k=0):
    """
    faster algorithm to get triu_indices (over numpy.triu_indices)
    
    Parameters
    ----------
    
    Returns
    -------

    """
    
    tmp_range = np.arange(dim-k)
    rows = np.repeat(tmp_range,(tmp_range+1)[::-1])
    cols = np.ones(rows.shape[0],dtype=np.int)
    inds = np.cumsum(tmp_range[1:][::-1]+1)
    np.put(cols,inds,np.arange(dim*-1+2+k,1))
    cols[0] = k
    np.cumsum(cols,out=cols)
    #
    return rows, cols
