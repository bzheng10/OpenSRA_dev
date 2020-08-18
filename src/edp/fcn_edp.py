#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Miscellaneous functions for liquefaction and landslide
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Required packages
import numpy as np
#####################################################################################################################


#####################################################################################################################
##### calculate stress profile - for cpt where properties at specific depths are given
#####################################################################################################################
def get_stress(**kwargs):
	"""
	Computes the in-situ total and effective stress and pore pressure profiles an interpolate pressures at user-specified depths. The function handles two types of inputs:
	
	1. properties by layer (typical for SPT): properties for various layers are provided, along with layer thicknesses
	2. properties by depth (typical for CPT): properties are given at specfic depths
	
	Parameters
	----------
	z : float, array
		[m] depths to interpolate for stresses
	dtw : float
		[m] depth to water table
	gamma_w = float, optional
		[kN/m^3] unit weight of water, default = 9.81 kN/m^3
	
	Type 1 inputs
	gamma_layer : float, array
		[kN/m^3] unit weight of soil at each layer
	dh_layer : float, array
		[m] layer thicknesses
	
	Type 2 inputs
	gamma_z : float, array
		[kN/m^3] unit weights at specific depths
		
	Returns
	-------
	sig_tot_z = float, array
		[kPa] in-situ vertical total stress
	u_z = float, array
		[kPa] hydrostatic pore pressure (ignoring capillary suction above **gwt**)
	sig_eff_z = float, array
		[kPa] in-situ vertical effective stress
	
	"""
	
	## get inputs
	z = kwargs.get('z',None) ## m, depth to get stresses
	dtw = kwargs.get('dtw',0) ## m, depth to water-table, default 0 m
	gamma_w = kwargs.get('gamma_w',9.81) # kN/m^3 unit weight of water, default to metric
	
	## method-dependent parameters
	gamma_layer = kwargs.get('gamma_layer',None) ## unit weight of layers
	dH_layer = kwargs.get('dH_layer',None) ## layer thicknesses, must have the same length as gamma_layer
	gamma_z = kwargs.get('gamma_z',None) ## kN/m^3, unit weight of layers
	
	## convert to numpy array for easier computation
	if type(z) is not np.array:
		z = np.asarray(z)
	
	## check method to use for calculating total stress
	## 1) by layer: gamma_layer & dH_layer - for spt where properties for a number of layers are given
	## 2) by depth: gamma_z - for cpt where properties at specific depths are given
	if gamma_layer is not None and dH_layer is not None:
		## convert to numpy array for easier computation
		if type(gamma_layer) is not np.array:
			gamma_layer = np.asarray(gamma_layer)
		if type(dH_layer) is not np.array:
			dH_layer = np.asarray(dH_layer)
			
		## determine stress and depth arrays from input layer properties
		nLayers = len(gamma_layer) # number of layers
		dsig_tot = np.multiply(gamma_layer,dH_layer) # calculate total stress increase per layer
		sig_sum = [sum(dsig_tot[0:i+1]) for i in range(nLayers)] # get cumulative total stress at the base of each layer
		z_full = np.hstack([0,[sum(dH_layer[0:i+1]) for i in range(nLayers)]]) # get full depth array, pad 0 at the start for interpolation
		sig_tot_full = np.hstack([0,sig_sum]) # pad 0 kPa to start of total stress array (for interpolation)
		
		## interpolate to calculate total stress at target depths
		sig_tot_z = np.interp(z,z_full,sig_tot_full) # vertical total stress
	
	elif gamma_z is not None:
		## convert to numpy array for easier computation
		if type(gamma_z) is not np.array:
			gamma_z = np.asarray(gamma_z)
	
		ndepth = len(z) # number of depth indices
		sig_tot_z = np.zeros(ndepth) # initialize vertial total stress array
		## loop to calculate total stress
		for i in range(ndepth):
			if i == 0:
				sig_tot_z[i] = z[i]*gamma_z[i] # total stress for first depth
			else:
				sig_tot_z[i] = sig_tot_z[i-1] + (z[i]-z[i-1])*gamma_z[i] # total stress for subsequent depths
	
	## calculate pore pressure and effective stress
	u_z = np.asarray(gamma_w*(z-dtw)) # hydrostatic pore pressure
	u_z = np.maximum(u_z,np.zeros(u_z.shape)) # set minimum of pore pressure to 0 (no suction)
	sig_eff_z = sig_tot_z - u_z # vertical effective stress
	
	##
	return sig_tot_z, u_z, sig_eff_z
	
	
#####################################################################################################################
##### calculate Vs over depth zmax (average slowness)
#####################################################################################################################
def get_Vs_zmax(**kwargs):
	"""
	Computes the avery slowness over a depth of **zmax** (e.g., vs30) by zmax dividing by the total travel time to depth of **zmax** (total travel time = sum(dh_i/vs_i), where dh_i = layer thickness, and vs_i = layer's shear wave velocity)
	
	Parameters
	----------
	vs_arr : float, array
		[m/sec] shear wave velocity profile
	z_bot_arr : float, optional
		[m] depth to bottom of each transition of shear wave velocity
	zmax : float
		[m] maximum depth to compute mean shear wave velocity over, default to 30 m
		
	Returns
	-------
	vs_zmax = float
		[m/sec] average slowness down to a depth o **zmax**
	
	"""
	
	## get inputs
	zmax = kwargs.get('zmax',30) ## target depth to compute Vs over, default to 30 m
	vs_arr = kwargs.get('vs_arr',18) ## shear wave velocity profile
	z_bot_arr = kwargs.get('z_bot_arr',0) ## depths to bottom of traisition of vs
	
	## calculate Vs over depth zmax (average slowness)
	nLayers = len(vs_arr) # number of layers
	z_bot_arr = np.hstack([0,z_bot_arr]) # pad 0 to depth array for interpolation
	dz_layer = [z_bot_arr[i+1]-z_bot_arr[i] for i in range(nLayers)] # get layer thicknesses
	t_tot = sum(np.divide(dz_layer,vs_arr)) # calculate layer travel time and sum up total time to travel to target depth
	vs_zmax = zmax/t_tot # calculate Vs (average slowness) over target depth given total travel time
	
	##
	return vs_zmax
	
	
#####################################################################################################################
##### calculate depth-reduction factor
#####################################################################################################################
def get_rd(**kwargs):
	"""
	docstring
	
	"""
	
	## Current methods coded
	## method 1 = Youd et al. (2001) NCEER
	## method 2 = Cetin (2000) Dissertation - Reliability-based assessment of seismic soil liquefaction initiation hazard (used in Moss et al. CPT liq)
	## method 3 = Cetin et al. (2004) - SPT-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential
	## method 4 = Idriss (1999), used in Idriss and Boulanger (2008, 2012) and Boulanger and Idriss (2014, 2016)
	method = kwargs.get('method','idriss') # meters, depth value or array of values, default to Idriss (1999)
	
	## get inputs
	z = kwargs.get('z',None) # meters, depth value or array of values
	## convert to numpy array for easier computation
	if type(z) is not np.array:
		z = np.asarray(z)
	
	##
	if method == 'youd_etal_2001': # Youd et al. (2001)
	
		## calculate rd(z)
		rd = np.asarray((1.000 - 0.4113*z**0.5 + 0.04052*z + 0.001753*z**1.5) /\
						(1.000 - 0.4177*z**0.5 + 0.05729*z - 0.006205*z**1.5 + 0.001210*z**2))
						
		## check to make sure nothing is over 1
		rd = np.minimum(rd,np.ones(rd.shape))
		
	##
	elif method == 'cetin_2000': ## Cetin (2000), no Vs term compared to Cetin et a. (2004)
	
		## get additional inputs
		M = kwargs.get('M',None) # moment magnitude
		amax = kwargs.get('amax',None) # g, peak surface acceleration
		
		## initialize arrays
		rd = []
		sigma_rd = []
		
		## loop through depths
		for d in z:
		
			## calculate sigma(z)
			if d < 12.2: ## eq. 10 in Moss et al. (2006)
				temp_sigma_rd = (d*3.28)**0.864 * 0.00814
			elif d >= 12.2: ## eq. 11 in Moss et al. (2006)
				temp_sigma_rd = 40**0.864 * 0.00814
			
			## calculate rd(z)
			sigma_rd.append(temp_sigma_rd)
			if d < 20: ## eq. 8 in Moss et al. (2006)
				rd.append((1 + (-9.147 - 4.173*amax + 0.652*M) / \
							(10.567 + 0.089*np.exp(0.089*(-d*3.28 - 7.760*amax + 78.576)))) / \
						(1 + (-9.147 - 4.173*amax + 0.652*M) / \
							(10.567 + 0.089*np.exp(0.089*(-7.760*amax + 78.576)))))
			elif d >= 20: ## eq. 9 in Moss et al. (2006)
				rd.append((1 + (-9.147 - 4.173*amax + 0.652*M) / \
							(10.567 + 0.089*np.exp(0.089*(-d*3.28 - 7.760*amax + 78.576)))) / \
						(1 + (-9.147 - 4.173*amax + 0.652*M) / \
							(10.567 + 0.089*np.exp(0.089*(-7.760*amax + 78.576)))) - \
						0.0014*(d*3.28 - 65))
		
		## convert to numpy arrays
		rd = np.asarray(rd)
		sigma_rd = np.asarray(sigma_rd)
	
	##
	elif method == 'cetin_etal_2004': # Cetin et al. (2004)
	
		# get additional inputs
		M = kwargs.get('M',None) # moment magnitude
		amax = kwargs.get('amax',None) # g, peak surface acceleration
		Vs12 = kwargs.get('Vs12',None) # m/s, Vs in the upper 12 m (40 ft)
		
		## initialize arrays
		rd = []
		sigma_rd = []
		
		## loop through depths
		for d in z:
			## calculate sigma(z)
			if d >= 12: ## eq. 8 in Cetin et al. (2004)
				temp_sigma_rd = 12**0.8500 * 0.0198
			elif d < 12: ## eq. 8 in Cetin et al. (2004)
				temp_sigma_rd = d**0.8500 * 0.0198
			
			## calculate rd(z)
			sigma_rd.append(temp_sigma_rd)
			if d < 20: ## eq. 8 in Cetin et al. (2004)
				rd.append((1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
							(16.258 + 0.201*np.exp(0.341*(-d + 0.0785*Vs12 + 7.586)))) / \
						(1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
							(16.258 + 0.201*np.exp(0.341*(0.0785*Vs12 + 7.586)))))
			elif d >= 20: ## eq. 8 in Cetin et al. (2004)
				rd.append((1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
							(16.258 + 0.201*np.exp(0.341*(-20 + 0.0785*Vs12 + 7.586)))) / \
						(1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
							(16.258 + 0.201*np.exp(0.341*(0.0785*Vs12 + 7.586)))) - \
						0.0046*(d-20))
		
		## convert to numpy arrays
		rd = np.asarray(rd)
		sigma_rd = np.asarray(sigma_rd)
	
	##
	elif method == 'idriss_1999': # Idriss (1999)
	
		# get additional inputs
		M = kwargs.get('M',None) # moment magnitude
		
		## check if M is given
		if M is None:
			print('Idriss (1999) requires M as input; return without calculating rd')
		
		else:
			## calculate rd
			alpha = -1.012 - 1.126*np.sin(z/11.73 + 5.133) # eq. 3b in Boulanger and Idriss (2016)
			beta = 0.106 + 0.118*np.sin(z/11.28 + 5.142) # eq. 3c in Boulanger and Idriss (2016)
			rd = np.exp(alpha + beta*M) # eq. 3a in Boulanger and Idriss (2016)
			rd = np.minimum(rd,np.ones(rd.shape)) # check to make sure nothing is over 1
			
			## check if i is over 20 meters
			for i in z:
				if i > 20:
					print('Z exceeds 20 m, the maximum recommended depth for this correlation (Idriss and Boulanger, 2008)')
					print('--> Consider site response analysis for stress reduction factor')
					break
	##
	else: # requests for other methods
		rd = None
	
	##
	return rd
	
	
#####################################################################################################################	
##### get arias intensity
#####################################################################################################################
def get_Ia(t, acc, gval=9.81):
	"""
	Computes the arias intensity, Ia, for a given acceleration time history.
	
	Parameters
	----------
	t : float, array
		[sec] time
	acc : float, array
		[g] acceleration
	gval : float, optional
		[g] gravitational acceleration, default = 9.81 m/s^2
		
	Returns
	-------
	Tm = float
		[sec] mean period
	
	"""
	
	## Determine time step of array 
	dt = [t[i+1]-t[i] for i in range(len(t)-1)] # sec
	
	## Pad 1 to beginning of dt array for index multiplication of vectors
	dt = np.asarray(np.hstack([1,dt])) # sec
	
	## Multiply indices of dt and acc array
	Ia = np.asarray([abs(acc[i])**2 * dt[i] for i in range(len(acc))]) # m/s^2 * m/s^2 * sec = m^2/s^3
	
	## Sum up all the indices to get Ia
	Ia = np.asarray([sum(Ia[0:i]) for i in range(len(Ia))]) * np.pi/2/gval # m^2/s^3 / m/s^2 = m/s
	
	##
	return max(Ia)
	
	
#####################################################################################################################
#### get Tm, mean period, a measure of frequency content of ground motion, eq. 3.15 in Saygili (2008) dissertation
#####################################################################################################################
def get_Tm(t,y):
	"""
	Computes the mean period, **Tm**, a measure of the mean frequency content in the ground motion record; used in Saygili (2008). A Fourier Transform is applied a time history. The frequencies between 0.25 and 25 Hz are weighted by the square of the respective FFT amplitudes and then summed. The sum is divided by the sum of the weights (i.e., square of FFT amplitudes) to obtain the mean period, **Tm**.
	
	Parameters
	----------
	t : float, array
		[sec] time
	y : float, array
		[varies] y values
		
	Returns
	-------
	Tm = float
		[sec] mean period
	
	References
	----------
	.. [1] Saygili, G., 2008, A Probabilistic Approach for Evaluating Earthquake-Induced Landslides, PhD Thesis, Universtiy of Texas at Austin.
	
	"""
	
	## get FFT on time history
	n = len(t) # length of time history
	dt = t[1]-t[0] # # sec, time step
	f = fft.fftfreq(n,d=dt) # Hz, frequency array
	y_fft = fft.fft(y) # Fourier transform
	
	## Determine number of points to Nyquist (odd versus even number for length of record)
	if np.mod(n,2) == 0:
		mid_pt = int(n/2)
	else:
		mid_pt = int((n-1)/2+1)
		
	## Amplitude of FFT
	y_fft_amp = np.abs(y_fft)
	
	## Calculate Tm discretely by evaluating the numerator and the denominator,
	numer = sum([y_fft_amp[i]**2/f[i] for i in range(mid_pt) if f[i] >= 0.25 and f[i] <= 20]) # 1/Hz = sec
	denom = sum([y_fft_amp[i]**2 for i in range(mid_pt) if f[i] >= 0.25 and f[i] <= 20])
	
	## get Tm
	Tm = numer/denom # sec
	
	##
	return Tm
	
	
#####################################################################################################################
##### summing volumetric strain over depth, with depth-weighted factor given by Cetin et al. (2009)
##### Cetin et al. (2009) Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements
#####################################################################################################################
def get_total_settlement(dh,eps_v,flag_DF=True,z_cr=18):
	"""
	Computes total settlement for a 1D soil column given the volumetrain strain in each layer.
	
	Parameters
	----------
	dh : float, array
		[m] thickness of each layer
	eps_v : float, array
		[%] volumetric strain at each layer
	flag_DF : boolean, optional
		flag for calculation of depth-weight factor, to increase settlement contribution from depths less than **z_cr** and reduce contribution from depths greater than **z_cr** (cetin et al., 2009), default = True
	z_cr : float, optional
		[m] critical depth for where depth-weighted factor is equal to 1 (cetin et al., 2009), default = 18 m
		
	Returns
	-------
	s_sum = float
		[m] cumulative ground settlement
	
	
	References
	----------
	.. [1] Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 3, pp. 387-398.
	
	"""
	
	## calculate depth to bottom and middle of each layer
	z_bot = np.asarray([sum(dh[0:i]) for i in range(len(dh))])
	z_mid = np.asarray([z_bot[i]/2 if i == 0 else (z_bot[i-1]+z_bot[i])/2 for i in range(z_bot)])
	
	## maximum depth
	h_sum = z_bot[len(z_bot)]
	
	## calculate depth-weighted factor (Cetin et al., 2009)
	if flag_DF is True:
		Df = 1-z_mid/z_cr
	else:
		DF = np.ones(z_bot.shape)
	
	## calculate total volumetric strain and settlement
	numer = sum([eps_v[i]*dh[i]*DF[i] for i in range(len(dh))])
	denom = sum([dh[i]*DF[i] for i in range(len(dh))])
	eps_v_sum = numer/denom
	s_sum = eps_v_sum/100*h_sum
	
	return eps_v_sum, s_sum
	
	
#####################################################################################################################
##### estimate yield accelerations for landslide
#####################################################################################################################
def get_ky(slope_type,**kwargs):
	"""
	Various methods to calculate the yield acceleration:
	
	1. Newmark sliding **block**
	2. **Rock**-slope failures
	3. Disrupted soil slides (**infinite slope**)
	4. Coherent **rotational** slides
	5. Simplified method (Bray and Travasarou, 2009)
	
	Parameters
	----------
	slope_type : str
				type of slope to assess; enter any of the bolded words shown in the above list of failure modes (i.e., **rock**, **infinite slope**, or **rotational**)
	phi : float
		[degree] friction angle
	c : float
		[kPa] cohesion
	beta : float
		[degree] slope angle
	gamma : float
			[kN/m^3] unit weight
	
	Additional parameters specific to sliding block:
	Incomplete
		
	Additional parameters specific to rock-slope failures:
	H : float
		[m] height of slope (local relief)
	
	Additional parameters specific to disrupted soil slides:
	cr : float
		[kPa] root cohesion
	t : float
		[m] thickness of failure mass
	m : float
		[%] percent of failure thickness that is saturated
	gamma_w : float
			[kN/m^3] unit weight of water, default = 9.81 kN/m^3
	method : str
		slight difference in form between **Bray** (2007), **Grant** et al. (2016), and **Rathje** & Antonakos (2011), specify author name
	
	Parameters for Coherent Rotational Slides (failure_mode == 3):
	H : float
		[m] height of slope (local relief)
	y : float
		[m] width of cylindrical pixel
	
	Returns
	-------
	ky : float
		[g] yield acceleration (see *return_param* under "Parameters")
	FS : float
		factor of safety (if calculated explicitly)
	
	References
	----------
	.. [1] Grant, A., Wartman, J., and Abou-Jaoude, G., 2016, Multimodal Method for Coseismic Landslide Hazard Assessment, Engineering Geology, vol. 212, pp. 146-160.
	.. [2] Keefer, D.K., 1984., Landslides Caused by Earthquakes, Geological Society of America Bulletin, vol. 95, no. 4, pp. 406-421.
	.. [3] Newmark, N.M., 1965, Effects of Earthquakes on Dams and Embankments, Geotechnique, vol. 15, pp. 139–159.
	.. [4] Rathje, E.M., and Saygili, G., 2009, Probabilistic Assessment of Earthquake-Induced Sliding Displacements of Natural Slopes, Bulletin of the New Zealand Society for Earthquake Engineering, vol. 42, no. 1, pp. 18-27.
	.. [5] Bray, J.D., and Travasarou, T., 2009, Technical Notes: Pseudostatic Coefficient for Use in Simplified Seismic Slope Stability Evaluation, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 9, pp. 1336-1340.
	.. [6] Bray, J.D., 2007, Simplified Seismic Slope Displacement Procedures, Earthquake Geotechnical Engineering, Springer, Dordrecht, pp. 327-353.
	
	"""
	
	###############################
	## 1) Sliding block:
	###############################
	if 'block' in slope_type.lower():
		## assign passed values to parameters
		beta = kwargs.get('alpha',None) # deg, slope angle
		FS = kwargs.get('FS',None) # factor of safety, either provided or computed below
		
		## calculate yield acceleration
		ky = (FS-1)*np.sin(np.radians(beta)) ## eq. 1 in Jibson (2007)
	
	###############################
	## 2) Rock-slope or sliding block failures:
	###############################
	elif 'rock' in slope_type.lower():
		## assign passed values to parameters
		phi = kwargs.get('phi',None) # deg, friction angle
		c = kwargs.get('c',None) # kPa, cohesion
		beta = kwargs.get('beta',None) # deg, slope angle
		gamma = kwargs.get('gamma',None) # kN/m3, unit weight
		H = kwargs.get('H',None) # m, height of slope
		
		## intermediate calculations
		alpha = (beta + phi)/2 # rad, slope's critical angle
		h = 0.25*H # m, vertical height of failure mass
		
		## calculate factor of safety
		FS = 2*c*np.sin(np.radians(beta))/(gamma*h*np.sin(np.radians(beta-alpha))*np.sin(np.radians(alpha))) +\
			np.tan(np.radians(phi))/np.tan(np.radians(alpha))
		
		## calculate yield acceleration
		ky = (FS-1)*np.sin(np.radians(alpha)) # g
	
	###############################
	## 3) Disrupted soil slides (infinite slope):
	###############################
	elif 'infinite' in slope_type.lower() or 'slope' in slope_type.lower():
		## assign passed values to parameters
		phi = kwargs.get('phi',None) # deg, friction angle
		c = kwargs.get('c',None) # kPa, cohesion
		beta = kwargs.get('beta',None) # deg, slope angle
		gamma = kwargs.get('gamma',None) # kN/m3, unit weight
		t = kwargs.get('t',None) # m, thickness of failure mass
		method = kwargs.get('method',None) # Grant, Rathje, or Bray form for calculating ky
				
		## additional factors that are used in Grant et al. (2016) and Rathje and Saygili (2009) for factor of safety
		cr = kwargs.get('cr',0) # kPa, root cohesion, see Grant et al. (2016)
		m = kwargs.get('m',0) # %, percent of failure thickness that is saturated, see Rathje & Saygili (2009)
		gamma_w = kwargs.get('gamma_w',9.81) # kN/m3, unit weight of water
		
		## avoid tan(0 deg)
		beta[beta==0] = 0.001
		
		if 'bray' in method.lower():
			ky = np.tan(np.radians(phi-beta)) + c/(gamma*t*(np.cos(np.radians(beta))**2*(1+np.tan(np.radians(phi))*np.tan(np.radians(beta)))))
		
		else:
			## calculate factor of safety
			FS = (c+cr)/(gamma*t*np.sin(np.radians(beta))) + \
				np.tan(np.radians(phi))/np.tan(np.radians(beta)) - \
				gamma_w * m/100 * np.tan(np.radians(phi)) / (gamma * np.tan(np.radians(beta))) # factor of safety
		
			## calculate yield acceleration, form depends if phi is used (depends on method)
			if 'grant' in method.lower():
				## simplest form for ky without internal friction (used by Grant et al., 2016)
				ky = (FS-1)*np.sin(np.radians(beta)) # g
			
			## 
			elif 'rathje' in method.lower():
				## eq. 1 in Rathje and Saygili (2009)
				ky = (FS-1) / (np.cos(np.radians(beta)) * np.tan(np.radians(phi)) + 1/np.tan(np.radians(beta))) # g
		
	###############################
	## 4) Coherent rotational slides (deep):
	###############################
	elif 'rotational' in slope_type.lower():
		## assign passed values to parameters
		phi = kwargs.get('phi',None) # deg, friction angle
		c = kwargs.get('c',None) # deg, slope angle
		beta = kwargs.get('beta',None) # deg, slope angle
		gamma = kwargs.get('gamma',None) # kN/m3, unit weight
		H = kwargs.get('H',None) # m, local hillside relief/height of slope
		y = kwargs.get('y',None) # m, cylindrical width of pixel
		method = kwargs.get('method',None) # Grant or Bray form for calculating ky
	
		if 'grant' in method.lower():
			## intermediate calculations
			R = 1.5*H # m, radius of circular failure plane acting through a dry homogeneous hillslope
			delta = np.arcsin(1/(3*np.sin(np.radians(beta)))) # rad, for beta > 20 degrees
			L = 2*delta*R # m, failure plane length
			a = (4*R*(np.sin(delta))**3 / (3*(2*delta-np.sin(2*delta))) - R) * np.cos(delta) # m, landslide body's centroid
			W = 1/2*gamma*y*R**2*(2*delta - np.sin(2*delta)) # kN
			
			## calculate yield acceleration
			ky = (c*L*y + W*(np.cos(np.radians(beta))*np.tan(np.radians(phi)) - np.sin(np.radians(beta)))) / \
				(W*(a/R + np.sin(np.radians(beta))*np.tan(np.radians(phi)))) # g
		
		elif 'bray' in method.lower():
			print('Bray (2007) for deep slides - not coded')
	
	## 
	return ky
	

#####################################################################################################################
#### get Ts, site period, Bray and Travasarou (2007)
#####################################################################################################################
def get_Ts(H,vs,nDim):
	"""
	Calculates the site period, **Ts**. For 1D, Ts = 4*H/vs; for 2D, Ts = 2.6*H/vs
	
	Parameters
	----------
	H : float
		[m] slope/structure height
	vs : float
		[m/s] shear wave velocity
	nDim : int
		**1** (trapezoidal) or **2** (triangular) dimension for calculating **Ts**; default = 2
		
	Returns
	-------
	Ts : float
		[sec] site period
		
	References
	----------
	.. [1] Bray, J.D., and Travasarou, T., 2007, Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 133, no. 4, pp. 381-392.
	
	"""
	
	if nDim == 1:
		Ts = 4*H/vs
	elif nDim == 2:
		Ts = 2.6*H/vs
		
	##
	return Tm