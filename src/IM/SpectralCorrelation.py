# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for CPT penetration correction
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python Modules
import numpy as np


# -----------------------------------------------------------
def BakerJayaram2008(T1, T2):
	"""
	Compute correlations between spectral ordinates using Baker & Jayaram (2008).
	
	Parameters
	----------
	T1 : float
		[sec] first period
	T2 : float
		[sec] second period
		
	Returns
	-------
	corr : float
		correlation between **T1** and **T2**
		
	References
	----------
	.. [1] Baker, J.W., and Jayaram, N., 2008, Correlation of Spectral Acceleration Values from NGA Ground Motion Models, Earthquake Spectra, vol. 24, no. 1, pp. 299-317.
	
	"""
	
	# determine which period is larger
	Tmax = max(T1,T2)
	Tmin = min(T1,T2)
	
	# calculate correlations
	C1 = 1 - np.cos(np.pi/2 - 0.366*np.log(Tmax/max(Tmin,0.109)))
	if Tmax < 0.2:
		C2 = 1 - 0.105*(1 - 1/(1+np.exp(100*Tmax-5)))*((Tmax-Tmin)/(Tmax-0.0099))
	else:
		C2 = 0
	if Tmax < 0.109:
		C3 = C2
	else:
		C3 = C1
	C4 = C1 + 0.5*(np.sqrt(C3) - C3)*(1 + np.cos(np.pi*Tmin/0.109))
	
	# return the right correlation based on the period values
	if Tmax < 0.109:
		corr = C2
	elif Tmin > 0.109:
		corr = C1
	elif Tmax < 0.2:
		corr = min(C2,C4)
	else:
		corr = C4
	
	#
	return corr