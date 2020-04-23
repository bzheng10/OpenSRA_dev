#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for transient strain
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
### Newmark (1967)
#####################################################################################################################
def newmark_1967(pgv, **kwargs):
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
	
	## case 1: shear wave
	vs = kwargs.get('vs',None) # shear wave velocity
	gamma_s = kwargs.get('gamma_s',None) # deg, propagating angle relative to vertical axis
	
	## case 2: Rayleigh wave
	cr = kwargs.get('cr',None) # propagation or phase velocity of Rayleigh wave
	
	## ground strain parallel to pipe axis
	if vs is not None:
		# shear wave propagation
		eps_g = pgv/vs*np.sin(np.radians(gamma_s))*np.cos(np.radians(gamma_s))*100 # eq. 10.1
	elif cr is not None:
		# Rayleigh wave traveling parallel to pipe axis
		eps_g = pgv/cr*100 # eq. 10.2
		
	##
	return eps_g
	
	
#####################################################################################################################
### Shinozuka and Koike (1979)
#####################################################################################################################
def shinozuka_koike_1979(eps_g, D, t, E, G, lam, **kwargs):
	"""
	Computes the transient pipe strain given transient ground strain using the Shinozuka & Koike (1979) approach, as described in O'Rourke and Liu (2004).
	
	Parameters
	----------
	eps_g : float
		[%] transient ground strain
	D : float
		[m] outer diameter of pipe
	t : float
		[m] wall thickness of pipe
	E : float
		[kPa] modulus of elasticity of pipe
	G : float
		[kPa] shear modulus
	lam : float
		[m] wavelength
	gamma_cr : float, optional
		[%] critical shear strain before slippage occurs; default = 0.1%
	q : float, optional
		degree of slippage at the pipe-soil interface, ranges from 1 to :math:`\pi`/2; default = :math:`\pi`/2 (slippage over the whole pipe length)
	
	Returns
	-------
	eps_p : float
		[%] transient pipe (structural) strain
	
	References
	----------
	.. [1] Shinozuka, M., and Koike, T., 1979, Estimation of Structural Strain in Underground Lifeline Pipes, Lifeline Earthquake Engineering-Buried Pipelines, Seismic Risk and instrumentation, ASME, New York, pp. 31-48.
	.. [2] O’Rourke, M.J., and Liu, J., 2004, The Seismic Design of Buried and Offshore Pipelines, MCEER Monograph No. 4, 2012, MCEER, University at Buffalo, Buffalo, NY.
	
	"""
	
	## ground to pipe conversion factor
	A = np.pi*(D**2 - (D-2*t)**2) # cross-sectional area of pipe
	Kg = 2*np.pi*G # equivalent spring constant to reflect soil-structural interaction
	beta_0 = 1/(1 + (2*np.pi/lam)**2 * A*E/Kg) # eq. 10.5
	
	## shear strain at soil-pipe interface
	gamma_0 = 2*np.pi/lam*E*t/G*eps_g*beta_0 # eq. 10.6
		
	## critical shear strain, default = 0.1%
	## if gamma_0 <= gamma_cr, no slippage
	## if gamma_0 > gamma_cr, slips
	gamma_cr = kwargs.get('gamma_cr',0.1)
	
	## ground to pipe conversion factor, for large ground movement, i.e., gamma_0 > gamma_cr
	q = kwargs.get('q',np.pi/2)
	beta_c = gamma_cr/gamma_0*q*beta_0 # eq. 10.8
	
	## pipe axial strain
	eps_p = beta_c*eps_g # eq. 10.9
		
	##
	return eps_p
	
	
#####################################################################################################################
### O'Rourke and El Hmadi (1988)
#####################################################################################################################
def orourke_elhmadi_1988(**kwargs):
	"""
	Computes the transient pipe strain given transient ground strain using the O'Rourke & El Hmadi (1988) approach, as described in O'Rourke and Liu (2004).
	
	Parameters
	----------
	eps_g : float
		[%] transient ground strain
	D : float
		[m] outer diameter of pipe
	t : float
		[m] wall thickness of pipe
	E : float
		[kPa] modulus of elasticity of pipe
	tu : float
		[kPa] maximum frictional resistance at the shear interface
	lam : float
		[m] wavelength
	
	Returns
	-------
	eps_p : float
		[%] transient pipe (structural) strain
	
	References
	----------
	.. [1] O’Rourke, M.J., and El Hmadi, K.E., 1988, Analysis of Continuous Buried Pipelines for Seismic Wave Effects, Earthquake Engineering and Structural Dynamics, v. 16, pp. 917-929.
	.. [2] O’Rourke, M.J., and Liu, J., 2004, The Seismic Design of Buried and Offshore Pipelines, MCEER Monograph No. 4, 2012, MCEER, University at Buffalo, Buffalo, NY.
	
	"""
	
	# cross-sectional area of pipe
	A = np.pi*(D**2 - (D-2*t)**2)
	
	## strain due to friction forces acting over 1/4 of wavelength
	## controls when ground strain becomes large
	eps_f = tu*(lam/4)/(A*E) # eq. 10.15
	
	## pipe axial strain
	eps_p = min(eps_g, eps_f) # eq. 10.16
		
	##
	return eps_p