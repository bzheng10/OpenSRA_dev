#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Methods for CPT penetration correction
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Required packages
import numpy as np
#####################################################################################################################


#####################################################################################################################
##### Moss et al. (2006a) CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential
##### Moss et al. (2006b) Normalizing the CPT for Overburden Stress
#####################################################################################################################
def moss_etal_2006_corr(**kwargs):
	"""
	CPT tip resistance correction to **qc_1** using the Moss et al. (2006a) method. Note that Moss et al. (2006a) uses **qc_1** instead as one of the model parameters for estimating the probability of liquefaction (see :func:`edp.liq.moss_etal_2006_liq`). Users can need to specify **qc**, and the function will apply the overburden correction to obtain qc_1.
	
	Parameters
	----------
	qc : float
		[MPa] tip resistance
		
	Parameters for overburden correction
	sv0_tot : float
		[kPa] initial vertical total stress
	sv0_eff : float
		[kPa] initial vertical effective stress
	patm : float
		[kPa] atmospheric pressure, default = 101.3 kPa
	c : float, optional
		normalization exponent for overburden correction, users can specify this instead of calculating it interally
	fs : float, optional
		[MPa] sleeve friction, required to calculate **Rf** if **Rf** is not given
	Rf : float, optional
		[%] friction ratio, required if **fs** is not provided to calculate **Rf**
		
	Returns
	-------
	qc_1 : float
		normalized tip resistance corrected for overburden, used by Moss et al. (2006a) for liquefaction triggering
	Rf : float
		[%] friction ratio, used by Moss et al. (2006a) for liquefaction triggering
	c : float
		overburden normalization exponent, used by Moss et al. (2006a) for liquefaction triggering
		
	References
	----------
	.. [1] Moss, R.E.S., Seed, R.B., Kayen, R.E., Stewart, J.P., Der Kiureghian, A., and Cetin, K.O., 2006a, CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential, Journal of Geotechnical and Geoenvironmental Engineering, vol. 132, no. 8, pp. 1032-1051.
	.. [2] Moss, R.E.S., Seed, R.B., and Olsen, R.S., 2006b, Normalizing the CPT for Overburden Stress, Journal of Geotechnical and Geoenvironmental Engineering, vol. 132, no. 3, pp. 378-387.
	
	"""
	
	## get inputs
	qc = kwargs.get('qc',None) ## input q value
	patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of qc, default to 100 kPa
	sv0_tot = kwargs.get('sv0_tot',None) # initial vertical total stress
	sv0_eff = kwargs.get('sv0_eff',None) # initial vertical effective stress
	Rf = kwargs.get('Rf',None) # % friction ratio, optional if fs is not given
	if Rf is None:
		try: 
			fs = kwargs.get('fs',None) # sleeve friction, optional if Rf is not given (to calculate Rf)
		except:
			print('Must provide either sleeve friction (fs) or friction ratio (Rf); cannot proceed with procedure')
	c = kwargs.get('c',None) # normalization exponent for overburden correction: Cq = (patm/sv0_eff)**c
	
	## if exponent is not given, loop to calculate it
	if c is None:
	
		# fitting / model params
		x1 = 0.78
		x2 = -0.33
		y1 = -0.32
		y2 = -0.35
		y3 = 0.49
		z1 = 1.21
		
		## iteration params
		c_old = 999 # old guess
		tol_c = 0.1 # %, tolerance
		iterNum = 0 # start count
		maxIter = 50 # max number of iterations
		
		## first guess
		if Rf is None:
			Rf = fs/qc*100 # friction ratio
		f1 = x1*qc**x2
		f2 = -(y1*qc**y2+y3)
		f3 = abs(np.log10(10+qc))**z1
		c = f1*(Rf/f3)**f2 ## eq. 5
		
		## iterative loop
		while abs((c-c_old)/c)*100 > tol_c and iterNum < maxIter:
		
			## set c_old to c from previous iteration 
			c_old = c
			iterNum += 1
			print('iterNum = ',str(iterNum), ', c = ', str(c))
			
			## calculate c
			Cq = (patm/sv0_eff)**c ## eq. 4
			qc = Cq*qc
			if Rf is None:
				Rf = fs/qc*100 # friction ratio
			f1 = x1*qc**x2
			f2 = -(y1*qc**y2+y3)
			f3 = abs(np.log10(10+qc))**z1
			c = f1*(Rf/f3)**f2 ## eq. 5
	
	## calculate overbuden correction factor and apply correction
	Cq = (patm/sv0_eff)**c # eq. 4
	qc_1 = Cq*qc # corrected tip resistance
	if Rf is None:
		Rf = fs/qc*100 # friction ratio
	
	##
	return qc_1, Rf, c
	
	
#####################################################################################################################
##### Idriss and Boulanger (2008) Soil Liquefaction During Earthquakes (Monograph)
##### Boulanger and Idriss (2014) CPT and SPT Based Liquefaction Triggering Procedures
#####################################################################################################################
def boulanger_idriss_2014_corr_cpt(**kwargs):
	"""
	CPT tip resistance correction to **qc_1N_cs** using works by Idriss and Boulanger (2008) and Boulanger and Idriss (2014). Users can specify any type of tip resistance measures (**qc**, **qt**, **qc_1N**) and the function will apply the remaining corrections. Note that this function is used to produce **qc_1N_cs**, which is the primary input for the Boulanger & Idriss (2014) deterministic and probabilistic liquefaction triggering procedures (see :func:`edp.liq.boulanger_idriss_2014_liq`).
	
	Parameters
	----------
	qc : float
		[kPa] tip resistance
	qc_type : str
		describe the qc value specified, pick from **qc**, **qt**, **qc_1N**; default = **qc**
		
	Parameters for cone tip area correction
	u2 : float
		[kPa] pore pressure measured above the cone tip
	ar : float
		net area ratio for cone tip, varies between 0.65 to 0.85 (Boulanger & Idriss, 2014); default = 0.8
		
	Parameters for overburden correction
	fs : float
		[kPa] sleeve friction
	sv0_tot : float
		[kPa] initial vertical total stress
	sv0_eff : float
		[kPa] initial vertical effective stress
	patm : float
		[kPa] atmospheric pressure, default = 101.3 kPa
		
	Parameters for fines content correction
	fc : float, optional
		[%] fines content, can be estimated from the SBT index **Ic**; default = 0
	Cfc : float, optional
		fitting parameter for quantifying the uncertainty in estimating **fc** using **Ic**, :math:`\pm` 0.29 = 1 standard deviation; default to 0
		
	Returns
	-------
	qc_1N_cs : float
		[blows/ft] qc value corrected for area, overburden, and fines content
		
	References
	----------
	.. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.
	.. [2] Boulanger, R.W., and Idriss, I.M., 2014, CPT and SPT Based Liquefaction Triggering procedures, Report UCD/CGM-14/01, Department of Civil and Environmental Engineering, University of California, Davis, CA, 134 pp.
	
	"""
	
	########### Type of q-value for input ############
	qc_type = kwargs.get('qc_type','qc')
	## q_type:
	## - qc - measured (raw)
	## - qt - area corrected, primarily for clays. for sands, qt and qc are interchangeable (u2 ~ u0)
	## - qc_1N - cone tip resistance with overburden correction
	
	## decide if area and overburden corrections are needed given input
	if qc_type == 'qc':
		flag_area = True
		flag_stress = True
	elif qc_type == 'qt':
		flag_area = False
		flag_stress = True
	elif qc_type == 'qc_1N':
		flag_area = False
		flag_stress = False
		
	## always run FC correction, if FC is not provided, assume 0.
	flag_fc = True
	
	## get inputs
	qc = kwargs.get('q',None) ## input q value
	u2 = kwargs.get('u2',0) ## pore pressure behind cone tip
	ar = kwargs.get('ar',0.8) ## net area ratio for cone tip, default to 0.8 (0.65 to 0.85 per BI14)
	patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of qc, default to 101.3 kPa
	sv0_tot = kwargs.get('sv0_tot',None) # initial vertical total stress
	sv0_eff = kwargs.get('sv0_eff',None) # initial vertical effective stress
	fs = kwargs.get('fs',None) # sleeve friction
	fc = kwargs.get('fc',None) # percent, fines content
	Cfc = kwargs.get('Cfc',0) # fitting parameter, default = 0, +/- 0.29 = 1 standard deviation on fc
	
	### cone tip area correction (necessary for clays, for sand qt ~ qc)
	if flag_area is True:
		qc = qc + (1-ar)*u2
		
	### calculating sleeve resistance for Ic
	F = fs/(qc-sv0_tot)*100 # eq. 14 in Boulanger and Iriss (2016)
	
	## iteration params
	Ic_old = 999 # old guess
	Ic = 2.4 # first guess
	tol_Ic = 0.1 # %, tolerance
	iterNum = 0 # start count
	maxIter = 20 # max number of iterations
	
	## check to see if sleeve friction is given, if not cannot proceed to estimate Ic
	if fs > 0:
	
		## iterate for n and Ic 
		while abs(Ic-Ic_old)/Ic*100 > tol_Ic and iterNum < maxIter:
			### Set q_old to q from previous iteration 
			Ic_old = Ic
			iterNum += 1
	
			## calculate Ic
			n = max(0.5,min(1,0.381*Ic + 0.05*(sv0_eff/patm) - 0.15)) # eq. 7 in Robertson (2009), between 0.5 (sand) and 1 (clay)
			Q = (qc-sv0_tot)/patm * (patm/sv0_eff)**n # eq. 13 in Boulanger and Iriss (2016)
			Ic = ((3.47 - np.log10(Q))**2 + (1.22 + np.log10(F))**2)**0.5 # eq. 12 in Boulanger and Iriss (2016)
	
		## estimate fines content using SBT
		if fc is None:
			fc = max(0,min(100,80*(Ic + Cfc) - 137)) # eq. 15 in Boulanger and Idriss (2016), between 0 and 100%
	
	##
	else:
	
		## cannot determin Ic
		Q = -999
		F = -999
		IC = -999
		n = -999
		fc = -999
	
	## iteration parameters for qc_1N_cs
	qc_1N_cs_old = 999 # first iteration
	tol_q = 0.1 # %, tolerance
	iterNum = 0 # start count
	maxIter = 20 # max number of iterations
	qc_N = qc/patm # qc_N
	qc_1N_cs = qc_N # first guess
	
	## if qc is a true value (greater than 0)
	if qc_N > 0:
	
		## loop
		while abs(qc_1N_cs-qc_1N_cs_old)/qc_1N_cs*100 > tol_q and iterNum < maxIter:
	
			## set q_old to q from previous iteration 
			qc_1N_cs_old = qc_1N_cs
			iterNum += 1
	
			## overburden correction factor
			if flag_stress is True:
				m = max(0.264,min(0.782,1.338 - 0.249*qc_1N_cs**0.264)) ## eq. 2.15b, between 0.265 and 0.782, for qc_1N_cs between 21 and 254
				if (qc_1N_cs < 21 or qc_1N_cs > 254) and iterNum == 1:
					print('Note: qc_1N_cs is outside the range of 21 and 254 described in Boulanger & Idriss (2014)')
				CN = min(1.7,(patm/sv0_eff)**m) # capped at 1.7, eq. 2.15a
			else:
				CN = 1
			
			## correct for overburden
			qc_1N = CN * qc_N # eq. 2.4
	
			## fines content correction factor (increase in tip resistance)
			if flag_fc is True:
				# Increase in qc_1n due to fines
				d_qc_1N = (11.9 + qc_1N/14.6) * np.exp(1.63 - 9.7/(fc+2) - (15.7/(fc+2))**2) ## eq. 2.22
			else:
				d_qc_1N = 0
				
			## correction for fines
			qc_1N_cs = qc_1N + d_qc_1N # eq. 2.10
	
	##
	else:
	
		## cannot correct qc is value of qc is invalid
		m = -999
		CN = -999
		qc_1N = -999
		qc_1N_cs = -999
	
	##
	return qc_1N_cs
	
	