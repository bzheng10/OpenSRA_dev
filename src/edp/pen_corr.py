#####################################################################################################################
##### Correct SPT blow count to N1_60_cs or CPT qc to qc_1N_cs
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Youd et al. (2001) Liquefaction Resistance of Soils (NCEER)
#####################################################################################################################
def youd_etal_2001(**kwargs):
    
    ## inputs for correction
    N = kwargs.get('N',None)
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    ERm = kwargs.get('ERm',75) # %, measured energy ratio, default to 75%
    BH_diam = kwargs.get('BH_diam',None) # mm, give option for Liao and Whitman (1986) and Kayen et al. (1992)
    L_rod = kwargs.get('L_rod',5) # m, rod stick-up length, default to 5 m
    flag_liner_room = kwargs.get('flag_liner_room',False) # flag for liner in sampler
    flag_liner_absent = kwargs.get('flag_liner_presence',True) # flag for liner in sampler
    CN_method = kwargs.get('CN_method','LW86') # give option for Liao and Whitman (1986) and Kayen et al. (1992)
    fc = kwargs.get('fc',0.0) # percent, fines content
    
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
def cetin_etal_2004(**kwargs):
    
    ## inputs for correction
    N = kwargs.get('N',None)
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    ERm = kwargs.get('ERm',75) # %, measured energy ratio, default to 75%
    BH_diam = kwargs.get('BH_diam',None) # mm, give option for Liao and Whitman (1986) and Kayen et al. (1992)
    L_rod = kwargs.get('L_rod',5) # m, rod stick-up length, default to 5 m
    flag_liner_room = kwargs.get('flag_liner_room',False) # flag for liner in sampler
    flag_liner_absent = kwargs.get('flag_liner_presence',True) # flag for liner in sampler
    CN_method = kwargs.get('CN_method','LW86') # give option for Liao and Whitman (1986) and Kayen et al. (1992)
    fc = kwargs.get('fc',0.0) # percent, fines content
    
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
def boulanger_idriss_2014(input_type, **kwargs):
    
	## input_type:
	## - spt
	## - cpt
	input_type = kwargs.get('input_type',None)
	
	#############################################################################################################
	## SPT-Based
	if 's' in input_type.lower():
		## inputs for correction
		N = kwargs.get('N',None)
		sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
		patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
		ERm = kwargs.get('ERm',75) # %, measured energy ratio, default to 75%
		BH_diam = kwargs.get('BH_diam',None) # mm, give option for Liao and Whitman (1986) and Kayen et al. (1992)
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
	
	#############################################################################################################
	## CPT-Based
	elif 'c' in input_type.lower():
	   ############ Type of q-value for input ############
		q_type = kwargs.get('q_type','qc')
		## q_type:
		## - qc - measured (raw)
		## - qt - area corrected, primarily for clays. for sands, qt and qc are interchangeable (u2 ~ u0)
		## - qc_1N - cone tip resistance with overburden correction
		
		## decide if area and overburden corrections are needed given input
		if q_type == 'qc':
			flag_area = True
			flag_stress = True
		elif q_type == 'qt':
			flag_area = False
			flag_stress = True
		elif q_type == 'qc_1N':
			flag_area = False
			flag_stress = False
			
		## always run FC correction, if FC is not provided, assume 0.
		flag_fc = True
		
		## get inputs
		qc = kwargs.get('q',None) ## input q value
		u2 = kwargs.get('u2',0) ## pore pressure behind cone tip
		ar = kwargs.get('ar',0.8) ## net area ratio for cone tip, default to 0.8 (0.65 to 0.85 per BI14)
		patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of qc, default to 100 kPa
		fc = kwargs.get('fc',None) # percent, fines content
		sv0_tot = kwargs.get('sv0_tot',None) # initial vertical total stress
		sv0_eff = kwargs.get('sv0_eff',None) # initial vertical effective stress
		fs = kwargs.get('fs',None) # sleeve friction
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
		return qc_1N, qc_1N_cs, Q, F, Ic, m, CN, fc
		
		
	############################################################################################################# 
	else:
		print('Must enter either "spt" or "cpt" as "input_type"; cannot proceed')
		return None
		
	
#####################################################################################################################
##### Moss et al. (2006a) CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential
##### Moss et al. (2006b) Normalizing the CPT for Overburden Stress
#####################################################################################################################
def moss_etal_2006(**kwargs):
    
    ############ Type of q-value for input ############
    q_type = kwargs.get('q_type','qc')
    ## q_type:
    ## - qc - measured (raw)
    ## - qc_1N - cone tip resistance with overburden correction
    
	## decide if overburden correction is needed given input
    if q_type == 'qc':
        flag_stress = True
    elif 'qc_1' in q_type:
        flag_stress = False
    
    ## get inputs
    qc = kwargs.get('q',None) ## input q value
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of qc, default to 100 kPa
    sv0_tot = kwargs.get('sv0_tot',None) # initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # initial vertical effective stress
    fs = kwargs.get('fs',None) # sleeve friction
    c = kwargs.get('c',None) # normalization exponent for overburden correction: Cq = (patm/sv0_eff)**c
    
    ### Overburden correction
    if flag_stress is True:
	
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
            c = 0.5 # first guess
            tol_c = 0.1 # %, tolerance
            iterNum = 0 # start count
            maxIter = 20 # max number of iterations
            
			## iterative loop
            while abs(c-c_old)/c*100 > tol_c and iterNum < maxIter:
			
                ## set c_old to c from previous iteration 
                c_old = c
                iterNum += 1
                
                ## calculate c
                Cq = (patm/sv0_eff)**c ## eq. 4
                qc = Cq*qc
                Rf = fs/qc*100 # friction ratio
                f1 = x1*qc**x2
                f2 = -(y1*qc**y2+y3)
                f3 = abs(np.log10(10+qc))**z1
                c = f1*(Rf/f3)**f2 ## eq. 5

		## calculate overbuden correction factor and apply correction
        Cq = (patm/sv0_eff)**c # eq. 4
        qc_1 = Cq*qc # corrected tip resistance
        Rf = fs/qc*100 # friction ratio
        
    else:
	
		## no correction / correction already performed
        qc_1 = qc # corrected tip resistance
        Rf = fs/qc*100 # friction ratio

    ##
    return qc_1, Rf, c