#####################################################################################################################
##### Functions for surface fault rupture
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### HAZUS approach FEMA (2004), modified from Wells & Coppersmith (1994)
#####################################################################################################################
def hazus_2004(M):
	"""
	Compute surface fault rupture displacement using Wells & Coppersmith (1994) with slightly modified parameters
	
    Parameters
    ----------
	M: moment magnitude

	"""
	return 10**(-5.26 + 0.79*M)
	
	
#####################################################################################################################
##### Wells & Coppersmith (1994)
#####################################################################################################################
def wells_coppersmith_1994(M, d_type='max', fault_type='all'):
	"""
	Compute surface fault rupture displacement
	
    Parameters
    ----------
	M: moment magnitude
	
	d_type: maximum (max) or average (avg) displacement
	
	fault_type: ss (strike-slip), r (reverse), n (normal), or all

	"""
	## coefficients
	if d_type == 'max':
		if fault_type == 'ss':
			a = -7.03
			b = 1.03
		elif fault_type == 'r':
			a = -1.84
			b = 0.29
		elif fault_type == 'n':
			a = -5.90
			b = 0.89
		elif fault_type == 'all':
			a = -5.46
			b = 0.82
		else:
			a = np.nan
			b = np.nan
			
	elif d_type == 'avg':
		if fault_type == 'ss':
			a = -6.32
			b = 0.90
		elif fault_type == 'r':
			a = -0.74
			b = 0.08
		elif fault_type == 'n':
			a = -4.45
			b = 0.63
		elif fault_type == 'all':
			a = -4.80
			b = 0.69
		else:
			a = np.nan
			b = np.nan
		
    ## Displacement for surface fault rupture
	return a + b*M