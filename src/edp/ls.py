#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for lateral spreading
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Grant et al. (2016) Multimodal method for coseismic landslide hazard assessment
#####################################################################################################################
def grant_etal_2016_ls(liq_susc, pga, M, z, dr):
	"""
	Compute rock and soil landslide displacement using following the Grant et al. (2016) deterministic procedure.
	Of the 40 coseismic landslide datasets described in Keefer (1984), Grant et al. (2016) selected the four
	most fundamental modes of failures (see Table 1 in Grant et al., 2016, for detailed descriptions):
	
	1. Rock-slope failures: wedge geometry, slope range = 35-90 degrees, typically tens of meters to kilometers
	2. Disrupted soil slides: infinite slope, slope range = 15-50 degrees, typically meters to a hundred meters
	3. Coherent rotational slides: circular-rotation, slope range = 20-35 degrees, typically under 2 meters
	4. Lateral spreads: empirirically-developed geometry, slope range = 0-6 degrees, typically under 2 meters
	
	Note that:\n
	- The regression models for **failure modes 1-3** are coded separately; see :func: `edp.ls.grant_etal_2016_ls`.
	
	Parameters
	----------	
	pga : float
		[g] peak ground acceleration
	liq_susc : str
			susceptibility category to liquefaction (none, very low, low, moderate, high, very high)
	M : float
		moment magnitude
	z : float
		[m] elevation; site is susceptible to lateral spreads if z < 25 m
	dr : float
		[m] distance to river; site is susceptible to lateral spreads if dr < 25 m
	
	Returns
	-------
	d : float
		[cm] permanent ground deformation (see *return_param* under "Parameters")
	
	References
	----------
	[1] Grant, A., Wartman, J., and Abou-Jaoude, G., 2016, Multimodal Method for Coseismic Landslide Hazard 
	Assessment, Engineering Geology, vol. 212, pp. 146-160.
	[2] Keefer, D.K., 1984., Landslides Caused by Earthquakes, Geological Society of America Bulletin, vol. 95, 
	no. 4, pp. 406-421.
	[3] Wills, C.J., Perez, F.G., and Gutierrez, C.I., 2011, Susceptibility to Deep-Seated 
	Landslides in California. California Geological Survey Map Sheet no. 58.
	
	"""
	
	###############################
	## 4) Lateral spreads:
	###############################
	
	## calculation
	## get threshold pga against liquefaction
	if liq_susc.lower() == 'very high':
		pga_t = 0.09 # g
	elif liq_susc.lower() == 'high':
		pga_t = 0.12 # g
	elif liq_susc.lower() == 'moderate':
		pga_t = 0.15 # g
	elif liq_susc.lower() == 'low':
		pga_t = 0.21 # g
	elif liq_susc.lower() == 'very low':
		pga_t = 0.26 # g
	elif liq_susc.lower() == 'none':
		pga_t = 999. # g
	else:
		pga_t = np.nan
		
	## magnitude correction
	Kdelta = 0.0086*M**3 - 0.0914*M**2 + 0.4698*M - 0.9835
	
	## normalized stress, opportunity for liquefaction
	r = pga/pga_t
	
	## get normalized displacement, a, for M=7
	if r <= 1:
		a = 0
	elif r > 1 and r <= 2:
		a = 12*r - 12
	elif r > 2 and r <= 3:
		a = 18*r - 24
	elif r > 3 and r <= 4:
		a = 70*r - 180
	else:
		a = 100
	
	## susceptibility to lateral spreading only for low-lying soils (z < 25 m) and deposits found near river (dr < 25 m)
	if z < 25 or dr < 25:
		## correct for magnitude
		d = Kdelta*r
	else:
		d = 0
	
	##
	return d
	
	
#####################################################################################################################
##### Youd et al. (2002) Revised multilinear regression equations for prediction of lateral spread displacement
#####################################################################################################################
def youd_etal_2002(M, R, W, S, T_15, F_15, D50_15):
	"""
	Text
	
	"""
	##### Regression coefficients for empirical lateral spread model
	## Model inputs:
	## - M = moment magnitude
	## - R = closest distance from site to source (km)
	## - W = free-face ratio (height and/or horizontal distance from site to toe) (%)
	## - S = ground slope
	## - T_15 = cumulative thickness (in upper 20 m) of all saturated soil layers susceptible to liquefaction initiation with N1_60
	## - F_15 = average fines content of the soil comprising T_15 (%)
	## - D50_15 = average mean grain size of the soil comprising T_15 (mm)
	##
	## Model output:
	## - Dh = median computed permanent lateral spread displacement (m)
	##
	## flag_model == 1: ground slope (infinite slope)
	## flag_model == 2: free face
	
	## Ground slope
	if flag_model == 1:
		b0 = -16.213
		b4 = 0
		b5 = 0.338
	elif flag_model == 2:
		b0 = -16.713
		b4 = 0.592
		b5 = 0
		
	## model params
	b1 = 1.532
	b2 = -1.406
	b3 = -0.012
	b6 = 0.540
	b7 = 3.413
	b8 = -0.795
	
	## adjusted distance
	R_star = R + 10**(0.89*M-5.64)
	
	## standard deviation
	sigma_log_Dh = 0.197
	sigma_ln_Dh = sigma_log_Dh*np.log(10)
	
	## calcualte ln(D)
	log_Dh = b0 + b1*M + b2*np.log10(R_star) + b3*R + b4*np.log10(W) + \
			b4*np.log10(S) + b6*np.log(T_15) + b7*np.log10(100-F_15) + b8*np.log10(D50_15+0.1)
	
	## calculate D
	Dh = 10**log_Dh
	
	##
	return Dh, sigma_ln_Dh
	
	
#####################################################################################################################
##### FEMA (2014) HAZUS
#####################################################################################################################
def hazus_2014_ls(*args):
	"""
	Compute lateral spreading, which is the the same procedure used by Grant et al. (2016). See and use the function :func:`edp.ls.grant_etal_2016_ls`.
	
	References
	----------
	.. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
	
	"""
	
	return None