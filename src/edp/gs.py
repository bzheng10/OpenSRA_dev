#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for ground settlement
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Cetin et al. (2009) Probabilistic model for assessment of cyclically induced reconsolidation settlement
#####################################################################################################################
def cetin_etal_2009(**kwargs):
	"""
	Compute volumetric strain following the Cetin et al. (2009) probabilistic method.
	
	Parameters
	----------
	N1_60_cs : float
			   [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
	amax : float
		   [g] peak ground acceleration
	sv0_tot : float
	          [kPa] initial vertical total stress
	sv0_eff : float
			  [kPa] initial vertical effective stress
	rd : float
		 stress reduction factor with depth
	Dr : float
		 [%] relative density
	patm : float, optional
		   [kPa] atmospheric pressure; **default = 101.3 kPa**
	
	Returns
	-------
	eps_v : float
			[%] volumetric strain
			
	References
	----------
	.. [1] Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 3, pp. 387-398.
	
	"""
	
	############ Inputs ############
	N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
	amax = kwargs.get('amax',None) # g, peak surface acceleration
	sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
	sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
	rd = kwargs.get('rd',None) # stress reduction factor with depth
	Dr = kwargs.get('Dr',None) # %, relative density
	patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
	
	## Multidirectional shaking effects
	if Dr is None:
		K_md = 1.0
	else:
		K_md = 0.361*np.log(Dr) - 0.579 # eq. 3
	
	## Magnitude scaling factor
	if M is None:
		K_M = 1.0
	else:
		K_M = 87.1/M**2.217 # eq. 4
	
	## Overburden correction factor
	if Dr is None:
		K_sigma = 1.0 # eq. 31
	else:
		f = 1 - 0.005*Dr # factor for K_sigma, eq. 5
		K_sigma = (sv0_eff/patm)**(f-1) # eq. 5
	
	if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
		csr = None
		print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
	else:
		## Cyclic stress ratio (demand)
		csr_field = 0.65 * amax * sv0_tot/sv0_eff * rd # Seed and Idriss (1971)
		csr = csr_m_sigv/K_md/K_M/K_sigma # eq. 2, CSR corrected for unidirectionality in lab, magnitude, and overburden
		
	##
	if N1_60_cs < 5 or N1_50_cs > 40:
		print('N1_60_cs is outside the range of 5 to 40 given by Cetin et al. (2009)')
	if csr < 5 or csr > 40:
		print('CSR_SS_20_1D_1_atm is outside the range of 0.05 to 0.60 given by Cetin et al. (2009)')
		
	##
	ln_eps_v = np.log(1.879*np.log((780.416*np.log(csr) - N1_60_cs + 2442.465)/(636.613*N1_60_cs + 306.732)) + 5.583)
	eps_v = np.exp(ln_eps_v) * 100 # %
	sigma = 0.689
	
	## maximum volumetric strain, after Huang (2008)
	eps_v_max = 9.765 - 2.427*np.log(N1_60_cs) # %
	
	## volumetric strain as the minimum of the correlated and maximum values
	eps_v = min(eps_v, eps_v_max) # %
	
	##
	eps_v = eps_v * 1.15 # correction factor suggested in Peterson (2016)
	
	##
	return eps_v
	

#####################################################################################################################
##### FEMA (2004) HAZUS
#####################################################################################################################
def hazus_2014_gs(liq_susc):
	"""
	Compute volumetric strain at a given location using a simplified deterministic approach (after Tokimatsu and Seed, 1987).
	
	Parameters
	----------
	liq_susc : str
			   susceptibility category to liquefaction (none, very low, low, moderate, high, very high)
	
	Returns
	-------
	s : float
		[cm] settlement
			
	References
	----------
	.. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Buildin Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
	.. [2] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

	"""
	
	## Estimates based on liquefaction susceptibility
	if liq_susc.lower() == 'very high':
		s = 12 # inches
	elif liq_susc.lower() == 'high':
		s = 6 # inches
	elif liq_susc.lower() == 'moderate':
		s = 2 # inches
	elif liq_susc.lower() == 'low':
		s = 1 # inches
	elif liq_susc.lower() == 'very low':
		s = 0 # inches
	elif liq_susc.lower() == 'none':
		s = 0 # inches
	else:
		s = np.nan
	
	##
	return s
	

#####################################################################################################################
##### Ishihara & Yoshimine (1992) Settlemenets in sand deposits following liquefaction
#####################################################################################################################
def ishihara_yoshimine_1992(N1_60_cs, Dr, FS_liq):
	"""
	Compute volumetric strain following the Ishihara and Yoshimine (1992) deterministic method.
	
	Parameters
	----------
	N1_60_cs : float
			   [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
	Dr : float
		 [%] relative density
	FS_liq : float
			 factor of safety against liquefaction triggering
	
	Returns
	-------
	eps_v : float
			[%] volumetric strain
			
	References
	----------
	.. [1] Ishihara, K., and Yoshimine, M., 1992, Evaluation of Settlements in Sand Deposits Following Liquefaction During Earthquakes, Soils and Foundations, vol. 32, np. 1, pp. 173-188.
			
	"""
	
	##
	if Dr > 1:
		Dr = Dr/100 # convert to decimal if Dr is obviously in percent
	
	F_alpha = 0.032 + 4.7*Dr - 6.0*Dr**2
	
	##
	gamma_lim = max(1.859*(1.1 - (N1_60_cs/46)**(0.5))**3,0)
	
	##
	if FS_liq >= 2:
		gamma_max = 0
	elif FS_liq <= F_alpha:
		gamma_max = gamma_lim
	else:
		gamma_max = min(gamma_lim, 0.035*(2-FS_liq)*(1-F_alpha)/(FS_liq-F_alpha))
	
	##
	eps_v = 1.5*np.exp(-0.369 * N1_60_cs**(0.5)) * min(0.08,gamma_max) * 100 # %
	
	##
	eps_v = eps_v * 0.9 ## Correction suggested by Cetin et al. (2009)

	##
	return eps_v
	
		
#####################################################################################################################
##### Tokimatsu & Seed (1987) Settlements in sand due to earthquake shaking
#####################################################################################################################
def tokimatsu_seed_1987(**kwargs):
	"""
	Compute volumetric strain following the Tokimatsu and Seed (1987) deterministic method.
	
	Parameters
	----------
	TBD : float
		  TBD
	
	Returns
	-------
	eps_v : float
			[%] volumetric strain
			
	References
	----------
	.. [1] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.
	
	"""
	
	print('Placeholder - under development')
