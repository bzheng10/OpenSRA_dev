# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for transient ground strain
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging


# -----------------------------------------------------------
def Newmark1967(pgv, **kwargs):
	"""
	Computes the transient ground strain using the Newmark (1967) approach, as described in O'Rourke and Liu (2004). Two analytical solutions are provided, for shear wave and Rayleigh wave propogation, where:
	
	1. **eps_g** by shear wave = f(**pgv**, **vs**, **gamma**)
	2. **eps_g** by Rayleigh wave = f(**pgv**, **cr**)
	
	Parameters
	----------
	pgv : float
		[cm/s] peak ground velocity
		
	Parameters for shear wave solution
	vs : float
		[m/s] shear wave velocity
	gamma : float
		[degree] angle of incidence
	
	Parameters for shear wave solution
	cr : float
		[m/s] Rayleigh wave velocity
		
	Returns
	-------
	eps_g : float
		[%] transient ground strain parallel to pipe axis
	
	References
	----------
	.. [1] Newmark, N.M., 1967, Problems in Wave Propagation in Soil and Rocks, Proceedings of the International Symposium on Wave Propagation and Dynamic Properties of Earth Materials, University of New Mexico Press, pp. 7-26.
	.. [2] O’Rourke, M.J., and Liu, J., 2004, The Seismic Design of Buried and Offshore Pipelines, MCEER Monograph No. 4, 2012, MCEER, University at Buffalo, Buffalo, NY.
	
	"""
	
	# case 1: shear wave
	vs = kwargs.get('Vs30',None) # shear wave velocity
	gamma_s = kwargs.get('gamma_s',None) # deg, propagating angle relative to vertical axis
	
	# case 2: Rayleigh wave
	cr = kwargs.get('cr',None) # propagation or phase velocity of Rayleigh wave
	
	# ground strain parallel to pipe axis
	if vs is not None:
		# shear wave propagation
		eps_g = pgv/vs*np.sin(np.radians(gamma_s))*np.cos(np.radians(gamma_s))*100 # eq. 10.1
	elif cr is not None:
		# Rayleigh wave traveling parallel to pipe axis
		eps_g = pgv/cr*100 # eq. 10.2
		
	#
	return eps_g