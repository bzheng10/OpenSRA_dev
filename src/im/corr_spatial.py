#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for spatial correlations
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### get spatial correlations (between sites)
##### Jayaram & Baker (2009) Correlation model for spatially distributed ground-motion intensities
#####################################################################################################################
def jayaram_baker_2009(h, T, geo_cond=2):
	"""
	Compute correlations between spatial ordinates using Jayaram & Baker (2009).
	
	Parameters
	----------
	h : float
		[km] distance
	T : float
		[sec] period
	geo_cond : int, optional
		geologic condition: **1** for variability within soil, **2** for homogenous conditions; default = 2
		
	Returns
	-------
	corr : float
		correlation for two sites at a distance of **d**
		
	References
	----------
	.. [1] Jayaram, N., and Baker, J.W., 2009, Correlation Model for Spatially Distributed Ground‐Motion Intensities, Earthquake Engineering and Structural Dynamics, vol. 38, no. 15, pp. 1687-1708.
	
	"""
	
	## calculations
	if T < 1:
		if geo_cond == 1:
			b = 8.5 + 17.2*T
		if geo_cond == 2:
			b = 40.7 - 15.0*T
	elif T >= 1:
		b = 22.0 + 3.7*T

	##
	return np.exp(-3*h/b)

