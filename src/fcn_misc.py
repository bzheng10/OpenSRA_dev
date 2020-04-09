#####################################################################################################################
##### Miscellaneous functions for Task 4B - liquefaction and landslide
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### calculate the epicentral distances using the Haversine equation
#####################################################################################################################
def get_haversine_dist(lon1,lat1,lon2,lat2,z=0,unit='km',unit='degree'):
	
	## determine unit to reference for Earth's radius
	if unit == 'km':
		r = 6371 # km
	elif unit == 'miles':
		r = 3,958.8
		
	## check if in radians or degrees, assume to be in degrees by default
    if 'deg' in unit.lower():
	
		## convert long lat from degrees to radians
		lon1 = np.log(lon1)
		lat1 = np.log(lat1)
		lon2 = np.log(lon2)
		lat2 = np.log(lat2)
	
    ## Haversine function for epicentral distance
    d = 2*r*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((long2-long1)/2)**2))
		
	##
	return d
	

#####################################################################################################################
##### trapezoidal integration function
#####################################################################################################################
def inte_trap(y,x=None,inte_max=None):
	
	## check if x values are given (non-uniform time step)
    if x is None:
        x = range(len(y)) ## time-step = 1
	
	## initialize array
    inte_y = np.zeros(len(y))
	
	## integrate 
    for i in range(1,len(y)):
        inte_y[i] = inte_y[i-1] + (y[i-1] + y[i])/2 * (x[i]-x[i-1])
	
	## in case integration goes beyond 1 (need better way of checking for total sum)
    if inte_max is not None:
        inte_y = np.clip(inte_y,None,inte_max)
	
	##
    return inte_y
	

#####################################################################################################################
##### count number of instances in a given range
#####################################################################################################################
def count_in_range(vect, a=-np.inf, b=np.inf, a_flag=True, b_flag=True):

	## upper limit
    if a_flag is True:
        count_a = sum([1 if i >= a else 0 for i in vect])
    else:
        count_a = sum([1 if i > a else 0 for i in vect])
		
	## lower limit
    if b_flag is True:
        count_b = sum([1 if i > b else 0 for i in vect])
    else:
        count_b = sum([1 if i >= b else 0 for i in vect])
		
	##
    return count_a-count_b
	

#####################################################################################################################
##### sort list where each index has multiple elements, sort by column index
#####################################################################################################################
def sort(arr,col):

	##
    return(sorted(arr, key = lambda x: x[col]))
	
	
#####################################################################################################################
##### get random combination, for LHS
#####################################################################################################################
def genRandComb(nBin,nVar):

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
	x: 1D array of n! elements
		elements should be ordered left-right, top-down
		e.g., for a 3x3 matrix, 1D array consists of components (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
		
	n: the dimension of the matrix

	"""
    mat = np.zeros((n,n))
    mat[np.triu_indices(n)] = x
    mat = mat + mat.T - np.diag(np.diag(mat))
    
	##
    return mat
	
	
#####################################################################################################################
##### get probability of exceedance by count
#####################################################################################################################
def count_prob_exceed(x, y):
    """
	Count the number of instances in y where y > i for every i in x.
	Probability = count/total

	Parameters
	----------
	x: list of values to get probability of exceendance for
	
	y: sample/population
	"""
    ##
    if len(y) > 0:
		## count number of instances where y > i for every i in x
        count = [sum(map(lambda val: val>i, y)) for i in x]
		
		## calculate probability = count/total
        prob = np.divide(count,len(y))
		
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
	Calculate the weighted average

	Parameters
	----------
	x: values
	
	weights = list of weights corresponding to x
	
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
def get_cond_prob_exceed(xbins, x, y, ycriteria):
    """
	Count the number of instances in y where y > i for every i in x.
	Probability = count/total

	Parameters
	----------
	xbins: list of bins to group x values (e.g., M = 5, 6, 6.5 for bins 5-6 and 6-6.5)
	
	x: list of values to get probability of exceendance for
	
	y: sample/population
	
	ycriteria: prob(y > ycriteria)
	
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
        prob = np.zeros(len(xbins)-1)
    
	##
    return prob