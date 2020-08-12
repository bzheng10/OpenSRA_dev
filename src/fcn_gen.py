#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### General functions
#####
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### required packages
import logging
import numpy as np
#####################################################################################################################


#####################################################################################################################
##### logging function
#####################################################################################################################
def setLogging(level,file=None):
    """
    This method set the logging level and the format of the logs
    """
    
    ## setting logging level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if level.lower() == 'debug' else logging.INFO)

    ## setting log format for print
    handlerStream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handlerStream.setFormatter(formatter)
    logger.addHandler(handlerStream)
    
    ## setting log format for save
    if file is not None:
        handlerFile = logging.FileHandler(file, mode='w')
        handlerFile.setFormatter(formatter)
        logger.addHandler(handlerFile)


#####################################################################################################################
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
  
    if (a_set & b_set): 
        return True
    else: 
        return False
    

#####################################################################################################################
##### get closest projection and spherical distance for a location relative to a line segment
#####################################################################################################################
def get_closest_pt(loc,line):
    """
    returns the closest location and shortest distance between a location and a line segment

    Parameters
    ----------
    loc : [float, float]
        [degree] coordinate of point
    line : [[float, float], [float, float]]
        [degree] two pairs of coordinates [pt1, pt2] that define the line
    
    Returns
    -------
    closest_loc : [float, float]
        [degree] closest point on line
    shortest_dist : [float, float]
        [degree] shorest distance
    
    """
    ## calculations
    line_vect = [line[1][0]-line[0][0],line[1][1]-line[0][1]] # get vector for line from pt 1 to pt 2
    len_line = np.dot(line_vect,line_vect)**0.5 # length of line
    if line_vect[0] == 0:
        line_theta = np.radians(90)
    else:
        line_theta = np.arctan(line_vect[1]/line_vect[0]) # rad, angle for line vector relative to horizontal
    loc_vect = [loc[0]-line[0][0],loc[1]-line[0][1]] # get vector from pt 1 of line to pt to project
    proj_dist = np.dot(loc_vect,line_vect/len_line) # compute projected length of pt on line
    
    ## check if point is within or outside of line:
    if proj_dist <= 0:
        closest_loc = line[0]
    elif proj_dist >= len_line:
        closest_loc = line[1]
    else:
        closest_loc = [line[0][0]+proj_dist*np.cos(line_theta),
                        line[0][1]+proj_dist*np.sin(line_theta)] # compute projection
        closest_loc = np.round(closest_loc,decimals=6)
    
    ##
    shortest_dist = get_haversine_dist(loc[0],loc[1],closest_loc[0],closest_loc[1])
    
    ##
    return closest_loc,shortest_dist


#####################################################################################################################
##### Haversine equation for distances
#####################################################################################################################
def get_haversine_dist(lon1,lat1,lon2,lat2,unit='km'):
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
	
	## determine unit to reference for Earth's radius
	if 'k' in unit.lower():
		r = 6371 # km
	elif 'mi' in unit.lower():
		r = 3958.8 # miles
	
	## convert long lat from degrees to radians
	lon1 = np.radians(lon1)
	lat1 = np.radians(lat1)
	lon2 = np.radians(lon2)
	lat2 = np.radians(lat2)
	
	## Haversine function for epicentral distance
	d = 2*r*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
	
	##
	return d


#####################################################################################################################
##### trapezoidal integration
#####################################################################################################################
def inte_trap(y,x=None):
	"""
	computes the trapezoidal integration of a vector
	
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
	
	## check if x values are given (non-uniform time step)
	if x is None:
		x = range(len(y)) ## time-step = 1
	
	## initialize array
	inte_y = np.zeros(len(y))
	
	## integrate
	for i in range(1,len(y)):
		inte_y[i] = inte_y[i-1] + (y[i-1] + y[i])/2 * (x[i]-x[i-1])
	
	## in case integration goes beyond 1 (need better way of checking for total sum)
	# if inte_max is not None:
		# inte_y = np.clip(inte_y,None,inte_max)
	
	##
	return inte_y


#####################################################################################################################
##### count number of instances in a given range
#####################################################################################################################
def count_in_range(list, a=-np.inf, b=np.inf, flag_include_a=True, flag_include_b=True):
	"""
	counts the number of instances in a list between lower and upper limits, inclusive or exclusive
	
	Parameters
	----------
	list : float, array
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
	
	## upper limit
	if flag_include_a is True:
		count_a = sum([1 if i >= a else 0 for i in list])
	else:
		count_a = sum([1 if i > a else 0 for i in list])
		
	## lower limit
	if flag_include_b is True:
		count_b = sum([1 if i > b else 0 for i in list])
	else:
		count_b = sum([1 if i >= b else 0 for i in list])
		
	##
	return count_a-count_b


#####################################################################################################################
##### sort list where each index has multiple elements, sort by column index
#####################################################################################################################
def sort(list,col):
	"""
	Sorts a list of **m** entries and each entry with **n** values and sort by the column **col** relative to **m**
	
	Parameters
	----------
	list : float, array
		list of values
	col : int
		column to sort
	
	Returns
	-------
	
	list_sort : float, array
		sorted list
	
	"""
	
	##
	return(sorted(list, key = lambda x: x[col]))


#####################################################################################################################
##### get random combination, for LHS
#####################################################################################################################
def gen_rand_com(nBin,nVar):
	"""
	
	text
	
	"""
	## initialize array
	comb = np.zeros((nBin, nVar))
	
	## find permunation given number of bins
	comb[0:,0] = np.random.permutation(np.arange(nBin))
	
	## loop and determine corresponding pairs, remove permutation when already selected
	for i in range(1,nVar):
		for j in range(nBin):
			choice = np.arange(nBin)
			usedVal = np.unique(np.append(np.transpose(comb[j,0:i]),comb[0:j,i]))
			ind2del = [next(ii for ii, jj in enumerate(choice) if jj == kk) for kk in usedVal]
			choice2 = np.delete(choice,ind2del)
			comb[j,i] = np.random.permutation(choice2)[0]
			
	##
	return comb


#####################################################################################################################
##### convert a 1D array to a symmetric matrix
#####################################################################################################################
def convert_array_to_sym_mat(x,n):
	"""
	Converts a 1D array to a symmetric matric, e.g., 3 elements -> 2x2 matrix, 6 elements -> 3x3 matrix
	
	Parameters
	----------
	x : float, array
		1D array of n! elements, where elements should be ordered left-right, top-down, e.g., for a 3x3 matrix, 1D array consists of components (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
		
	n : int
		dimension of the symmetric matrix
		
	Returns
	-------
	mat : float, matrix
		symmetric matrix
	
	"""
	
	## initialize mat
	mat = np.zeros((n,n))
	## set upper triangle to values in x
	mat[np.triu_indices(n)] = x
	## add transpose of matrix and subtract diagonal to get symmetric matrix
	mat = mat + mat.T - np.diag(np.diag(mat))
	
	##
	return mat


#####################################################################################################################
##### get probability of exceedance by count
#####################################################################################################################
def get_prob_exceed_by_count(x, y):
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
	
	##
	if len(y) > 0:
		## count number of instances where y > i for every i in x
		count = [sum(map(lambda val: val>i, y)) for i in x]
		
		## calculate probability = count/total
		prob = np.divide(count,len(y)) * 100
		
	else:
		count = np.zeros(1,dtype='int32')
		prob = np.zeros(1)
	
	##
	return prob, count


#####################################################################################################################
##### calculated weighted average
#####################################################################################################################
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
	
	## make sure the shape of x and weights are the same (horizontal or vertical)
	if len(x[0]) != len(weights):
		x = np.transpose(x)
		flag_transpose = True
	
	## calcualte weighted x values
	x_weighted = np.multiply(x, weights)
	
	## if trannsposed previously, undo transpose for consistency with input
	if flag_transpose is True:
		x_weighted = np.transpose(x_weighted)
	
	## sum up weighted x values and dividy by total of weights
	sum_weighted = sum(x_weighted)
	avg_weighted = sum_weighted/sum(weights)
	
	##
	return sum_weighted, avg_weighted


#####################################################################################################################
##### get probability of exceedance given criteria
#####################################################################################################################
def get_cond_prob_exceed_by_count(xbins, x, y, ycriteria):
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
	
	## find indices where y exceeds criteria and reduce the x and y arrays
	ind_red = np.where(y > ycriteria)
	y_red = y[ind_red]
	x_red = x[ind_red]
	
	##
	if len(y_red) > 0:
	
		# get counts for bins
		count = []
		for i in range(len(xbins)-1):
			ind2search = np.where((x_red > xbins[i]) & (x_red < xbins[i+1]))
			y2search = y_red[ind2search]
			_,j = get_prob_by_count([ycriteria], y2search)
			count.append(j[0])
	
		## sum up number of counts to get probability of exceedance
		count = [sum(count[0:i]) for i in range(len(count))]
		
		## calculate probability = count/total
		prob = np.divide(count,len(y_red))
		
	else:
		count = np.zeros(1,dtype='int32')
		prob = np.zeros(len(xbins)-1)*100
	
	##
	return prob


#####################################################################################################################
##### find elements in an array given condition and set to new value
#####################################################################################################################
def find_set_nan(arr,find_val,set_val):
    """
    Finds all the elements in an array **arr** that equals the value **find_val** and sets these elements to the value **set_val**.
    
    Parameters
    ----------
    arr: float, array
        an array of values
    find_val: float
        value to search for
    set_val: float
        value to set to
        
    Returns
    -------
    arr: float, array
        modified array
    
    """
	
    ##
    filters = np.where(arr==find_val)
    if len(filters[0]) > 0:
        arr[filters] = set_val
    
    ##
    return arr


#####################################################################################################################
##### faster algorithm to get triu_indices
##### ref = https://mail.python.org/pipermail/numpy-discussion/2013-September/067534.html
#####################################################################################################################
def fast_triu_indices(dim,k=0):
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
	##
	return np.vstack([rows,cols])