#####################################################################################################################
##### Miscellaneous functions
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Calculate the epicentral distances using the Haversine equation
#####################################################################################################################
def get_haversine_dist(lat1,long1,lat2,long2,z=0,unit='km'):
	
	## determine unit to reference for Earth's radius
	if unit == 'km':
		r = 6371 # km
	elif unit == 'miles':
		r = 3,958.8
	
	## convert long lat from degrees to radians
    lat1 = np.log(lat1)
    long1 = np.log(long1)
    lat2 = np.log(lat2)
    long2 = np.log(long2)
	
    ## Haversine function for epicentral distance
    d = 2*r*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((long2-long1)/2)**2))
		
	##
	return d


#####################################################################################################################
##### Calculate stress profile - for cpt where properties at specific depths are given
#####################################################################################################################
def get_stress(**kwargs):

	## get inputs
    gamma_w = kwargs.get('gamma_w',9.81) # kN/m^3 unit weight of water, default to metric
    z = kwargs.get('z',0) ## m, depth to get stresses
    dtw = kwargs.get('dtw',0) ## m, depth to water-table, default 0 m
	
	## method-dependent parameters
    gamma_layer = kwargs.get('gamma_layer',18) ## unit weight of layers, default to 18 kN/m^3
    dH_layer = kwargs.get('dH_layer',0) ## layer thicknesses, must have the same length as gamma_layer
    gamma_z = kwargs.get('gamma_z',18) ## kN/m^3, unit weight of layers, default to 18 kN/m^3
    
	## check method to use for calculating total stress
	## 1) by layer: gamma_layer & dH_layer - for spt where properties for a number of layers are given
	## 1) by depth: gamma_z - for cpt where properties at specific depths are given
	if gamma_layer is not None and dH_layer is not None:
	
		## determine stress and depth arrays from input layer properties
		nLayers = len(gamma_layer) # number of layers
		dsig_tot = np.multiply(gamma_layer,dH_layer) # calculate total stress increase per layer
		sig_sum = [sum(dsig_tot[0:i+1]) for i in range(nLayers)] # get cumulative total stress at the base of each layer
		z_full = np.hstack([0,[sum(dH_layer[0:i+1]) for i in range(nLayers)]]) # get full depth array, pad 0 at the start for interpolation
		sig_tot_full = np.hstack([0,sig_sum]) # pad 0 kPa to start of total stress array (for interpolation)
		
		## interpolate to calculate total stress at target depths
		sig_tot_z = np.interp(z,z_full,sig_tot_full) # vertical total stress
	
	elif gamma_z is not None:
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
##### Calculate Vs over depth zmax (average slowness)
#####################################################################################################################
def get_Vs_zmax(**kwargs):

	## get inputs
    zmax = kwargs.get('zmax',0) ## target depth to compute Vs over
    Vs_layer = kwargs.get('Vs_layer',18) ## unit weight of layers, default to 18 kN/m^3
    z_bot_layer = kwargs.get('z_bot_layer',0) ## depths to bottom of layers
    
	## calculate Vs over depth zmax (average slowness)
    nLayers = len(Vs_layer) # number of layers
    z_bot_layer = np.hstack([0,z_bot_layer]) # pad 0 to depth array for interpolation
    dz_layer = [z_bot_layer[i+1]-z_bot_layer[i] for i in range(nLayers)] # get layer thicknesses
    t_tot = sum(np.divide(dz_layer,Vs_layer)) # calculate layer travel time and sum up total time to travel to target depth
    Vs_over_zmax = zmax/t_tot # calculate Vs (average slowness) over target depth given total travel time
    
	##
    return Vs_over_zmax


#####################################################################################################################
##### Calculate depth-reduction factor
#####################################################################################################################
def get_rd(**kwargs):
	## Current methods coded
    ## method 1 = Youd et al. (2001) NCEER
	## method 2 = Cetin (2000) Dissertation - Reliability-based assessment of seismic soil liquefaction initiation hazard (used in Moss et al. CPT liq)
    ## method 3 = Cetin et al. (2004) - SPT-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential
    ## method 4 = Idriss (1999), used in Idriss and Boulanger (2008, 2012) and Boulanger and Idriss (2014, 2016)
    method = kwargs.get('method','idriss') # meters, depth value or array of values, default to Idriss (1999)

    ## get inputs
    z = kwargs.get('z',None) # meters, depth value or array of values
    
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
## Get arias intensity
#####################################################################################################################
def get_Ia(t, acc, gval=9.81):
    
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
## Get Tm, mean period, a measure of frequency content of ground motion, eq. 3.15 in Saygili (2008) dissertation
#####################################################################################################################
def get_Tm(t,y):
    
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
### Summing volumetric strain over depth, with depth-weighted factor given by Cetin et al. (2009)
### Cetin et al. (2009) Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements
#####################################################################################################################
def get_total_settlement(dh,eps_v,z_cr=18,flag_DF=True):
    
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
    s_sum = eps_v_sum*h_sum
    
    return eps_v_sum, s_sum