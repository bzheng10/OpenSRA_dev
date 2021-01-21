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
import numpy as np
import rasterio as rio
from scipy import sparse, stats
from scipy.interpolate import interp2d


# -----------------------------------------------------------
def lhs(n_var, n_samp, dist='normal', low=None, high=None, return_prob=False):
    """
    Performs Latin-Hypercube Sampling and returns both the cdfs and the residuals for the user-specified distribution.
    
    Allowed distribution types: **normal**(default), **truncated_normal**, **uniform**
    
    """
    
    # permutation of bins
    boxes = np.transpose([np.random.permutation(n_samp) for i in range(n_var)])
    # draw uniform samples from 0 to 1, add to bin permutations, and normalize by sample size to get cdfs
    norm_uniform_samples = np.random.uniform(size=(n_samp,n_var))
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
        logging.info(f'\t\t{os.path.basename(raster_path)}')
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
def set_logging(level, file=None):
    """
    This method sets the logging level and formatting of the logs
    """
    
    # setting logging level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if level.lower() == 'debug' else logging.INFO)

    # setting log format for print
    handlerStream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
		x = range(len(y)) # time-step = 1
	
	# initialize array
	inte_y = np.zeros(len(y))
	
	# integrate
	for i in range(1,len(y)):
		inte_y[i] = inte_y[i-1] + (y[i-1] + y[i])/2 * (x[i]-x[i-1])
	
	# in case integration goes beyond 1 (need better way of checking for total sum)
	# if inte_max is not None:
		# inte_y = np.clip(inte_y,None,inte_max)
	
	#
	return inte_y


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
def get_prob_exceed(x, y):
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
def get_cond_prob_exceed(xbins, x, y, ycriteria):
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