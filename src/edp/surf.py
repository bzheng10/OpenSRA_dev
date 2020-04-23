#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for surface fault rupture displacements
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### HAZUS approach FEMA (2004), modified from Wells & Coppersmith (1994)
#####################################################################################################################
def hazus_2014_surf(M):
	"""
	Compute surface fault rupture displacement using Wells & Coppersmith (1994) with modified parameters
	
	Parameters
	----------
	M : float
		moment magnitude
	
	Returns
	-------
	d : float
		[m] surface fault rupture displacement
		
	References
	----------
	.. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
	.. [2] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
	
	"""
	
	return 10**(-5.26 + 0.79*M)
	
	
#####################################################################################################################
##### Wells & Coppersmith (1994)
#####################################################################################################################
def wells_coppersmith_1994(M, d_type='max', fault_type='all'):
	"""
	Compute surface fault rupture displacement using Wells & Coppersmith (1994).
	
	Parameters
	----------
	M : float
		moment magnitude
	
	d_type : str
		type of displacement type: either **maximum** or **average**
	
	fault_type : str
		type of fault, enter bolded keywords: **ss** (strike-slip), **r** (reverse), **n** (normal), or **all**
	
	Returns
	-------
	d : float
		[m] surface fault rupture displacement
		
	References
	----------
	.. [1] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
	
	"""
	
	## coefficients
	if d_type.lower() == 'maximum':
		if fault_type.lower() == 'ss':
			a = -7.03
			b = 1.03
		elif fault_type.lower() == 'r':
			a = -1.84
			b = 0.29
		elif fault_type.lower() == 'n':
			a = -5.90
			b = 0.89
		elif fault_type.lower() == 'all':
			a = -5.46
			b = 0.82
		else:
			a = np.nan
			b = np.nan
			
	elif d_type.lower() == 'average':
		if fault_type.lower() == 'ss':
			a = -6.32
			b = 0.90
		elif fault_type.lower() == 'r':
			a = -0.74
			b = 0.08
		elif fault_type.lower() == 'n':
			a = -4.45
			b = 0.63
		elif fault_type.lower() == 'all':
			a = -4.80
			b = 0.69
		else:
			a = np.nan
			b = np.nan
		
	## Displacement for surface fault rupture
	return a + b*M