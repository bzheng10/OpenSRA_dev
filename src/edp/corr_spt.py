#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for SPT blow count correction
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Youd et al. (2001) Liquefaction Resistance of Soils (NCEER)
#####################################################################################################################
def youd_etal_2001_corr(**kwargs):
	"""
	SPT blow count correction to **N1_60_cs** using the NCEER method (Youd et al., 2001). Users can specify any type of blow count values (**Nm**, **N60**, **N1**, **N1_60**) and the function will apply the remaining corrections. Note that this function is used to produce N1_60, which is the primary input for the Cetin et al. (2004) probabilistic liquefaction triggering procedure (see :func:`edp.liq.cetin_etal_2004_liq`).
	
	Parameters
	----------
	N : float
		[blows/ft] N-value
	N_type : str
		describe the N-value specified, pick from **Nm**, **N60**, **N1**, **N1_60**; default = **Nm**
		
	Parameters for energy correction
	ERm : float
		[%] measured energy ratio, default = 75%
	BH_diam : float
		[mm]: borehold diameter
	L_rod : float
		[m] rod stick-up length, default = 5 m
	flag_liner_room : boolean
		flag for whether sampler has room for liner, default = False
		
	Parameters for overburden correction
	CN_method : str
		method for estimating the overburden correction factor, **CN**, methods available includes '**lw86**' for Liao and Whitman (1986) and '**k92**' for Kayen et al. (1992); default = '**lw86**'
	sv0_eff : float
		[kPa] initial vertical effective stress
	patm : float, optional
		[kPa] atmospheric pressure, default = 101.3 kPa
		
	Parameters for fines content correction
	fc : float
		[%] fines content; default = 0
		
	Returns
	-------
	N1_60 : float
		[blows/ft] N-value corrected for energy and overburden
	N1_60_cs : float
		[blows/ft] N-value corrected for energy, overburden, and fines content
		
	References
	----------
	.. [1] Youd, T.L., et al., 2001, Liquefaction Resistance of Soils; Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops on Evaluation of Liquefaction Resistance of Soils, Journal of Geotechnical and Geoenvironmental Engineering, vol. 127, no. 10, pp. 817–833.
	.. [2] Liao, S.S., and Whitman, R.V., 1986, Overburden Correction Factors for SPT in Sand, Journal of Geotechnical Engineering, vol. 112, no. 3, pp. 373-377.
	.. [3] Kayen, R.E., Mitchell, J.K., Seed, R.B., Lodge, A., Nishio, S., and Coutinho, R., 1992, Evaluation of SPT-, CPT-, and Shear Wave-Based Methods for Liquefaction Potential Assessment Using Loma Prieta Data, Proceeding of the 4th Japan-U.S. Workshop on Earthquake-Resistant Design of Lifeline Facilities and Countermeasures for Soil Liquefaction, vol. 1, pp. 177–204.
	
	"""
	
	## inputs for correction
	N = kwargs.get('N',None)
	sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
	patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
	ERm = kwargs.get('ERm',75) # %, measured energy ratio, default to 75%
	BH_diam = kwargs.get('BH_diam',None) # mm, borehole diameter
	L_rod = kwargs.get('L_rod',5) # m, rod stick-up length, default to 5 m
	flag_liner_room = kwargs.get('flag_liner_room',False) # flag for room for liner in sampler
	CN_method = kwargs.get('CN_method','LW86') # give option for Liao and Whitman (1986) and Kayen et al. (1992)
	fc = kwargs.get('fc',0) # percent, fines content
	
	############ Type of N-value for input ############
	## N_type:
	## - Nm - measured (raw)
	## - N60 - corrected for energy
	## - N1 - corrected for overburden
	## - N1_60 - corrected for overburden and energy
	N_type = kwargs.get('N_type','Nm')
	
	## check if overburden and energy corrections are needed
	if N_type == 'Nm':
		flag_energy = True
		flag_stress = True
	if N_type == 'N60':
		flag_energy = False
		flag_stress = True
	if N_type == 'N1':
		flag_energy = True
		flag_stress = False
	if N_type == 'N1_60':
		flag_energy = False
		flag_stress = False
	
	## always run FC correction, if FC is not provided, assume 0.
	flag_fc = True
	
	## energy correction
	if flag_energy is True:
	
		## correction for hammer energy
		CE = ERm/60
	
		## correction for borehole diameter
		if BH_diam is None:
			CB = 1.0
		else:
			if BH_diam >= 65 and BH_diam <= 115:
				CB = 1.0
			elif BH_diam == 150:
				CB = 1.05
			elif BH_diam == 200:
				CB = 1.15
	
		## correction for rod stick-up length
		if L_rod <= 3:
			CR = 0.75
		elif L_rod > 3 and L_rod <= 4:
			CR = 0.80
		elif L_rod > 4 and L_rod <= 6:
			CR = 0.85
		elif L_rod > 6 and L_rod <= 10:
			CR = 0.95
		elif L_rod > 10 and L_rod <= 30:
			CR = 1.0
	
		## correction for liner in sampler
		if flag_liner_room is False:
			CS = 1.0
		else:
			CS = 1.2 # varies from 1.1 to 1.3
	
		N = N * CE * CB * CR * CS
	
	## set N60 to N
	N60 = N
	
	## overburden correction
	if flag_stress is True:
		if sv0_eff > 300:
			print('sv0_eff too high for currently available correlations')
		else:
			if CN_method == 'LW86':
				CN = min((patm/sv0_eff)**0.5,1.7) # capped at 1.7
			elif CN_method == 'Ketal92':
				CN = 2.2/(1.2+sv0_eff/patm)
		N = N * CN
	
	## set N1_60 to N
	N1_60 = N
		
	## fines content correction
	if flag_fc is True:
	
		## Coefficients for effect of fines content (eq. 6a through 7c)
		if fc <= 5:
			alpha = 0.0
			beta = 1.0
		elif fc > 5 and fc < 35:
			alpha = np.exp(1.76 - 190/fc**2)
			beta = 0.99 + fc**1.5/1000
		elif fc >= 35:
			alpha = 5.0
			beta = 1.2
		N = alpha + beta*N # eq. 5
	
	## set N1_60_cs to N
	N1_60_cs = N
	
	##
	return N1_60, N1_60_cs
	
	
#####################################################################################################################
##### Cetin et al. (2004) SPT-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential
#####################################################################################################################
def cetin_etal_2004_corr(**kwargs):
	"""
	SPT blow count correction to **N1_60** and **N1_60_cs** using the Cetin et al. (2004) method. Users can specify any type of blow count values (**Nm**, **N60**, **N1**, **N1_60**) and the function will apply the remaining corrections. Note that this function is used to produce **N1_60**, which is the primary input for the Youd et al. (2001) deterministic liquefaction triggering procedure (see :func:`edp.liq.youd_etal_2001_liq`).
	
	Parameters
	----------
	N : float
		[blows/ft] N-value
	N_type : str
		describe the N-value specified, pick from **Nm**, **N60**, **N1**, **N1_60**; default = **Nm**
		
	Parameters for energy correction
	ERm : float
		[%] measured energy ratio, default = 75%
	BH_diam : float
		[mm]: borehold diameter
	L_rod : float
		[m] rod stick-up length, default = 5 m
	flag_liner_room : boolean
		flag for whether sampler has room for liner, default = False
	flag_liner_absent : boolean
		flag for the absence of the liner, given that there is room, default = True
		
	Parameters for overburden correction
	CN_method : str
		method for estimating the overburden correction factor, **CN**, methods available includes '**lw86**' for Liao and Whitman (1986) and '**k92**' for Kayen et al. (1992); default = '**lw86**'
	sv0_eff : float
		[kPa] initial vertical effective stress
	patm : float, optional
		[kPa] atmospheric pressure, default = 101.3 kPa
		
	Parameters for fines content correction
	fc : float
		[%] fines content; default = 0
		
	Returns
	-------
	N1_60 : float
		[blows/ft] N-value corrected for energy and overburden
	N1_60_cs : float
		[blows/ft] N-value corrected for energy, overburden, and fines content
		
	References
	----------
	.. [1] Cetin, K.O., Seed, R.B., Der Kiureghian, A., Tokimatsu, K., Harder Jr, L.F., Kayen, R.E., and Moss, R.E., 2004, Standard Penetration Test-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential, Journal of Geotechnical and Geoenvironmental Engineering, vol. 130, no. 12, pp. 1314-1340.
	.. [2] Liao, S.S., and Whitman, R.V., 1986, Overburden Correction Factors for SPT in Sand, Journal of Geotechnical Engineering, vol. 112, no. 3, pp. 373-377.
	.. [3] Kayen, R.E., Mitchell, J.K., Seed, R.B., Lodge, A., Nishio, S., and Coutinho, R., 1992, Evaluation of SPT-, CPT-, and Shear Wave-Based Methods for Liquefaction Potential Assessment Using Loma Prieta Data, Proceeding of the 4th Japan-U.S. Workshop on Earthquake-Resistant Design of Lifeline Facilities and Countermeasures for Soil Liquefaction, vol. 1, pp. 177–204.
	
	"""
	
	## inputs for correction
	N = kwargs.get('N',None)
	sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
	patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
	ERm = kwargs.get('ERm',75) # %, measured energy ratio, default to 75%
	BH_diam = kwargs.get('BH_diam',None) # mm, borehold diameter
	L_rod = kwargs.get('L_rod',5) # m, rod stick-up length, default to 5 m
	flag_liner_room = kwargs.get('flag_liner_room',False) # flag for liner in sampler
	flag_liner_absent = kwargs.get('flag_liner_presence',True) # flag for liner in sampler
	CN_method = kwargs.get('CN_method','LW86') # give option for Liao and Whitman (1986) and Kayen et al. (1992)
	fc = kwargs.get('fc',0) # percent, fines content
	
	############ Type of N-value for input ############
	## N_type:
	## - Nm - measured (raw)
	## - N60 - corrected for energy
	## - N1 - corrected for overburden
	## - N1_60 - corrected for overburden and energy
	N_type = kwargs.get('N_type','Nm')
	
	## check if overburden and energy corrections are needed
	if N_type == 'Nm':
		flag_energy = True
		flag_stress = True
	if N_type == 'N60':
		flag_energy = False
		flag_stress = True
	if N_type == 'N1':
		flag_energy = True
		flag_stress = False
	if N_type == 'N1_60':
		flag_energy = False
		flag_stress = False
	
	## always run FC correction, if FC is not provided, assume 0
	flag_fc = True
	
	## interation params
	N_old = 999 # first iteration
	tol_N = 11 # %, tolerance
	iterNum = 0
	
	## loop for N1_60
	while abs(N-N_old)/N*100 > tol_N and flag_stress is True and iterNum < 10:
		
		## set N_old to N from previous iteration 
		N_old = N
		iterNum += 1
		
		## energy correction
		if flag_energy is True:
	
			## correction for hammer energy
			CE = ERm/60
	
			## correction for borehole diameter
			if BH_diam is None:
				CB = 1.0
			else:
				if BH_diam >= 65 and BH_diam <= 115:
					CB = 1.0
				elif BH_diam == 150:
					CB = 1.05
				elif BH_diam == 200:
					CB = 1.15
	
			## correction for rod stick-up length
			## data from Figure 7
			CR_arr = np.loadtxt(os.getcwd()+'/cr_liao_whitman_1986.txt')
			CR = np.interp(L_Rod,CR_arr[:,[1]],CR_arr[:,[0]])
	
			## correction for liner in sampler
			if flag_liner_room is False:
				CS = 1.0
			else:
				if flag_liner_absent == True:
					CS = max(1.1,min(1.3,1 + N/100)) # varies from 1.1 to 1.3
			
			## apply corrections
			N = N * CE * CB * CR * CS
		
		## set N60 to N
		N60 = N
	
		## overburden correction
		if flag_stress is True:
			## Lian and Whitman (1986)
			CN = min((patm/sv0_eff)**0.5,1.6) # capped at 1.6
			N = N * CN
	
	## set N1_60 to N
	N1_60 = N
	
	## fines content correction
	if flag_fc is True:
	
		## Coefficients for effect of fines content (eq. 6a through 7c)
		if fc < 5:
			fc = 0
		elif fc > 35:
			fc = 35
		CFines = (1+0.004*fc)+0.05*fc/N # eq. 15        
		N = N*CFines # eq. 14
		
	## set N1_60_cs to N
	N1_60_cs = N
	
	##
	return N1_60, N1_60_cs
	
	
#####################################################################################################################
##### Idriss and Boulanger (2008) Soil Liquefaction During Earthquakes (Monograph)
##### Boulanger and Idriss (2014) CPT and SPT Based Liquefaction Triggering Procedures
#####################################################################################################################
def boulanger_idriss_2014_corr_spt(**kwargs):
	"""
	SPT blow count correction to **N1_60_cs** using works by Idriss and Boulanger (2008; 2010) and Boulanger and Idriss (2014). Users can specify any type of blow count values (**Nm**, **N60**, **N1**, **N1_60**) and the function will apply the remaining corrections to obtain **N1_60_cs**. Note that this function is used to produce **N1_60_cs**, which is the primary input for the Boulanger & Idriss (2014) deterministic and probabilistic liquefaction triggering procedures (see :func:`edp.liq.boulanger_idriss_2014_liq`).
	
	Parameters
	----------
	N : float
		[blows/ft] N-value
	N_type : str
		describe the N-value specified, pick from **Nm**, **N60**, **N1**, **N1_60**; default = **Nm**
		
	Parameters for energy correction
	ERm : float
		[%] measured energy ratio, default = 75%
	BH_diam : float
		[mm]: borehold diameter
	L_rod : float
		[m] rod stick-up length, default = 5 m
	flag_liner_room : boolean
		flag for whether sampler has room for liner, default = False
	flag_liner_absent : boolean
		flag for the absence of the liner, given that there is room, default = True
		
	Parameters for overburden correction
	sv0_eff : float
		[kPa] initial vertical effective stress
	patm : float, optional
		[kPa] atmospheric pressure, default = 101.3 kPa
		
	Parameters for fines content correction
	fc : float
		[%] fines content; default = 0
		
	Returns
	-------
	N1_60 : float
		[blows/ft] N-value corrected for energy and overburden
	N1_60_cs : float
		[blows/ft] N-value corrected for energy, overburden, and fines content
		
	References
	----------
	.. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.
	.. [2] Idriss, I.M., and Boulanger, R.W., 2010, SPT-Based Liquefaction Triggering Procedures, Report UCD/CGM-10/02, Department of Civil and Environmental Engineering, University of California, Davis, CA, 259 pp.
	.. [3] Boulanger, R.W., and Idriss, I.M., 2014, CPT and SPT Based Liquefaction Triggering procedures, Report UCD/CGM-14/01, Department of Civil and Environmental Engineering, University of California, Davis, CA, 134 pp.
	
	"""
	
	## inputs for correction
	N = kwargs.get('N',None)
	sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
	patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
	ERm = kwargs.get('ERm',75) # %, measured energy ratio, default to 75%
	BH_diam = kwargs.get('BH_diam',None) # mm, borehold diameter
	L_rod = kwargs.get('L_rod',5) # m, rod stick-up length, default to 5 m
	flag_liner_room = kwargs.get('flag_liner_room',False) # flag for liner in sampler
	flag_liner_absent = kwargs.get('flag_liner_presence',True) # flag for liner in sampler
	fc = kwargs.get('fc',0) # percent, fines content
	
	############ Type of N-value for input ############
	## N_type:
	## - Nm - measured (raw)
	## - N60 - corrected for energy
	## - N1 - corrected for overburden
	## - N1_60 - corrected for overburden and energy
	N_type = kwargs.get('N_type','Nm')
	
	## check if overburden and energy corrections are needed
	if N_type == 'Nm':
		flag_energy = True
		flag_stress = True
	if N_type == 'N60':
		flag_energy = False
		flag_stress = True
	if N_type == 'N1':
		flag_energy = True
		flag_stress = False
	if N_type == 'N1_60':
		flag_energy = False
		flag_stress = False
		
	## always run FC correction, if FC is not provided, assume 0.
	flag_fc = True
	
	## iteration params
	N_old = 999 # first iteration
	tol_N = 1 # %, tolerance
	iterNum = 0
	
	## loop for N1_60_cs
	while abs(N-N_old)/N*100 > tol_N and iterNum < 10:
		
		## set N_old to N from previous iteration 
		N_old = N
		iterNum += 1
	
		## energy correction
		if flag_energy is True:
	
			## correction for hammer energy
			CE = ERm/60
	
			## correction for borehole diameter
			if BH_diam is None:
				CB = 1.0
			else:
				if BH_diam >= 65 and BH_diam <= 115:
					CB = 1.0
				elif BH_diam == 150:
					CB = 1.05
				elif BH_diam == 200:
					CB = 1.15
	
			## correction for rod stick-up length
			if L_rod <= 3:
				CR = 0.75
			elif L_rod > 3 and L_rod <= 4:
				CR = 0.80
			elif L_rod > 4 and L_rod <= 6:
				CR = 0.85
			elif L_rod > 6 and L_rod <= 10:
				CR = 0.95
			elif L_rod > 10 and L_rod <= 30:
				CR = 1.0
	
			## correction for liner in sampler
			if flag_liner_room is False:
				CS = 1.0
			else:
				if flag_liner_absent is True:
					CS = max(1.1,min(1.3,1+N/100)) # between 1.1 and 1.3 depending on N1_60, to be iterated
	
			## apply corrections
			N = N * CE * CB * CR * CS
	
		## set N60 to N
		N60 = N
		
		### Overburden correction
		if flag_stress is True:
			
			m = 0.784 - 0.0768*N**0.5 ## eq. 2.15c, limited to N under 46
			if N > 46:
				print('N1_60_cs > 46, over the limit described in Boulanger and Idriss (2014)')
	
			CN = min(1.7,(patm/sv0_eff)**m) # capped at 1.7, eq. 2.15a
			N = N * CN
	
		## set N1_60 to N
		N1_60 = N
			
		### Fines content correction
		if flag_fc is True:
	
			# Increase in N-value due to fines
			d_N = np.exp(1.63 + 9.7/(fc+0.01) - (15.7/(fc+0.01))**2) # eq. 2.23
			N = N + d_N # eq. 2.11
			
			if flag_stress is False:
				## break while loop
				break
	
	## set N1_60_cs to N
	N1_60_cs = N
	
	##
	return N1_60, N1_60_cs
	
	