#####################################################################################################################
##### Methods for liquefaction triggering: CRR, CSR, FS, and/or pLiq
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Youd et al. (2001) SPT-Based Liquefaction Resistance of Soils (NCEER)
#####################################################################################################################
def youd_etal_2001(**kwargs):
    
    ############ Inputs ############
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    Dr = kwargs.get('Dr',None) # %, relative density
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    tau_stat = kwargs.get('tau_stat',None) # kPa, static shear stress
    
    ## magnitude scaling factor
    if M is None:
        msf = 1.0 # eq. 31
    else:
        msf = 10**2.24/M**2.56 # Idriss (1999) factors, recommended by Youd et al. (2001), eq. 24
    
    ## overburden correction factor
    if Dr is None:
        K_sigma = 1.0 # eq. 31
    else:
        f = min(0.8,max(0.6,(0.8-0.6)/(80-40)*(Dr-40)+0.6)) # factor for K_sigma, Figure 15
        K_sigma = (sv0_eff/patm)**(f-1) # eq. 31
        
    ## sloping ground correction factor
    if tau_stat is None:
        K_alpha = 1.0 # eq. 31
    else:
        K_alpha = tau_stat/sv0_eff # eq. 32
    
    if N1_60_cs is None:
        crr = None
    else:
        ## SPT base CRR curve (cyclic resistance ratio)
        crr = 1/(34-min(N1_60_cs,30)) + min(N1_60_cs,30)/135 + 50/(10*min(N1_60_cs,30) + 45)**2 - 1/200 # eq. 4
    
	## see if enough inputs are given for csr
    if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
        csr = None
        print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
    else:
        ## Cyclic stress ratio (demand)
        csr_m_sigv = 0.65 * amax * sv0_tot/sv0_eff * rd # Seed and Idriss (1971)
        csr = csr_m_sigv/msf/K_sigma/K_alpha # CSR for M=7.5 and corrected to 1 atm and static bias
    
	## determine if FS can be calculated
    if crr is None or csr is None:
        fs = None
    else:
        ## Factor of safety
        fs = crr/csr
    
    ##
    return crr, csr, fs
	

#####################################################################################################################
##### Cetin et al. (2004) SPT-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential
#####################################################################################################################
def cetin_etal_2004(**kwargs):
    
    ############ Inputs ############
    N1_60 = kwargs.get('N1_60',None) # corrected SPT blow count
    fc = kwargs.get('fc',0.0) # percent, fines content
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    p_liq = kwargs.get('p_liq',None) # decimals, probability of liquefaction, if want to determine equivalent CRR
	
    # flag for cases 1 and 2: measurement/estimation errors included/removed, default to remove (Cetin et al., 2004)
    flag_include_error = kwargs.get('flag_include_error',False)
    
    ## model parameters with measurement/estimation errors included/removed (Kramer and Mayfield, 2007)
    if flag_include_error is True:
        t1 = 0.004
        t2 = 13.79
        t3 = 29.06
        t4 = 3.82
        t5 = 0.06
        t6 = 15.25
        sigma = 4.21
    else:
        t1 = 0.004
        t2 = 13.32
        t3 = 29.53
        t4 = 3.70
        t5 = 0.05
        t6 = 16.85
        sigma = 2.70

    ## overburden correction factor, K_sigma
    # data from Figure 12, form provided in Idriss and Boulanger (2012, Spectra)
    K_sigma = (sv0_eff/patm)**-0.278kPa

    ## duration weighting factor (or magnitude scaling factor)
    # data from Figure 10, form provided in Idriss and Boulanger (2012, Spectra)
    dwf = (M/7.5)**-2.217
    
    ## check what inputs are given
    if p_liq is None:
        
        ## cyclic stress ratio
        csr_eq = 0.65*amax*sv0_tot/sv0_eff*rd # eq. 10, uncorrected for magnitude and duration
        csr = csr_eq/dwf/K_sigma # correction for duration and to 1 atm
        
        ## probability of liquefaction (Cetin et al., 2004)
        p_liq = norm.cdf(-(N1_60*(1+t1*fc) - t2*np.log(csr_eq) - t3*np.log(M) - t4*np.log(sv0_eff/patm) + t5*fc + t6)/sigma) # eq. 19
        
        ## cyclic resistance ratio
        crr = np.nan # not used

    else:
        
        ## cyclic stress ratio
        csr = None # not used
        
        ## inverse analysis of CRR given pLiq, (Cetin et al., 2004)
        crr = np.exp((N1_60*(1+t1*fc) - t3*np.log(M) - t4*np.log(sv0_eff/patm) + t5*fc + t6 + sigma*norm.ppf(p_liq))/t2) # eq. 20
    
    ##
    return crr, csr, p_liq
	

#####################################################################################################################
##### Idriss and Boulanger (2008) Soil Liquefaction During Earthquakes (Monograph)
##### Boulanger and Idriss (2012) Probabilistic Standard Penetration Test–Based Liquefaction–Triggering Procedure
##### Boulanger and Idriss (2014) CPT and SPT Based Liquefaction Triggering Procedures
##### Boulanger and Idriss (2016) CPT Based Liquefaction Triggering Procedure (journal)
#####################################################################################################################
def boulanger_idriss_2014(**kwargs):
    
	## input_type:
	## - spt (resistance = N1_60_cs)
	## - cpt (resistance = qc_1N_cs)
	input_type = kwargs.get('input_type',None)
	resistance = kwargs.get('resistance',None)
    # N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    # qc_1N_cs = kwargs.get('qc_1N_cs',None) # corrected CPT tip measurement
	
    ############ Inputs ############
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    tau_stat = kwargs.get('tau_stat',None) # kPa, static shear stress
    Q = kwargs.get('Q',10) # fitting parameter for critical state line, for relative state parameter
    K0 = kwargs.get('K0',0.5) # coefficient of lateral earth pressure
    flag_det_prob = kwargs.get('flag_det_prob','deterministic') # flag for deterministic or probabilistic approach (FS vs prob)
    p_liq = kwargs.get('p_liq',None) # decimals, probability of liquefaction, if want to determine equivalent CRR=
    
    ## magnitude scaling factor
    if M is None:
        msf = 1.0 # eq. 31
    else:
    #     msf = min(1.8,6.9*np.exp(-M/4)-0.058) # Eq. 2.17, Idriss and Boulanger (2008)
		if 's' in input_type.lower():
			msf_c1 = 31.5
			msf_c2 = 2
			
		elif 'c' in input_type.lower():
			msf_c1 = 180
			msf_c2 = 3
			
        msf_max = min(2.2,1.09+(resistance/msf_c1)**msf_c2) # eq. 2.20 and 2.21
        msf = 1 + (msf_max-1) * (8.64*np.exp(-M/4)-1.325) # eq. 2.19, Boulanger and Idriss (2014)
    
    ## overburden correction factor
	if 's' in input_type.lower():
		C_sigma_c1 = 18.9
		C_sigma_c2 = 2.55
		C_sigma_c3 = 0.5
		if resistane > 37:
			print('N1_60_cs > 37, over the limit described in Boulanger and Idriss (2014)')
			
	elif 'c' in input_type.lower():
		C_sigma_c1 = 37.3
		C_sigma_c2 = 8.27
		C_sigma_c3 = 0.264
		if resistane > 211:
			print('qc_1N_cs > 211, over the limit described in Boulanger and Idriss (2014)')
			
    C_sigma = min(0.3,1/(C_sigma_c1-C_sigma_c2*resistance**C_sigma_c3)) # eq. 2.16b and 2.16c
    K_sigma = min(1.1,1 - C_sigma*np.log(sv0_eff/patm)) # eq. 2.16a
        
    ## static shear stress correction factor
    if tau_stat is None:
        K_alpha = 1.0
    else:
		if 's' in input_type.lower():
			xi_r_c1 = (resistance/46)**0.5 # eq. 64
			
		elif 'c' in input_type.lower():
			xi_r_c1 = 0.478*resistance**0.264 - 1.063 # eq. 65
			if resistance < 21:
				print('qc_1N_cs < 21, below the limit recommended in Idriss and Boulanger (2008)')
		
        xi_r = 1/(Q-np.log(100*(1+2*K0)/3*sv0_eff/patm)) - xi_r_c1 # eq. 64 and 65
        xi_r = min(0.1,max(-0.6,xi_r)) # eq. 67
        alpha = min(0.35, tau_stat/sv0_eff) # eq. 63, 66
        a = 1267 + 636*alpha**2 - 634*np.exp(alpha) - 632*np.exp(-alpha) # eq. 60
        b = np.exp(-1.11 + 12.3*alpha**2 + 1.31*np.log(alpha + 0.0001)) # eq. 61
        c = 0.138 + 0.126*alpha + 2.52*alpha**3 # eq. 62
        K_alpha = a + b*np.exp(-xi_r/c) # eq. 59
        
	## see if enough inputs are given for csr
    if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
        csr = None
        print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
    else:
        ## cyclic stress ratio (demand)
        csr_m_sigv = 0.65 * amax * sv0_tot/sv0_eff *rd # Seed and Idriss (1971), uncorrected
        csr = csr_m_sigv/msf/K_sigma/K_alpha # CSR for M=7.5 and corrected to 1 atm and static bias
        
    ## see if enough inputs are given for crr or p_liq
    if resistance is None:
        print('N1_60_cs or qc_1N_cs is not provided as an input, exiting procedure')
        return None
    
    else:
		## overburden correction factor
		if 's' in input_type.lower():
			crr_c1 = 14.1
			crr_c2 = 126
			crr_c3 = 23.6
			crr_c4 = 25.4
				
		elif 'c' in input_type.lower():
			crr_c1 = 113
			crr_c2 = 1000
			crr_c3 = 140
			crr_c4 = 137
		
		## if deterministic, calculate factor of safety
        if flag_det_prob[0] == 'd':
            Co = 2.8 # value for deterministic analysis, pp. 1197 in BI12
			
            ## SPT base CRR curve (cyclic resistance ratio)
            crr = np.exp(resistance/crr_c1 + (resistance/crr_c2)**2 - \
						 (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - Co) # eq. 2.24 and 2.25 in BI14

            if crr is None or csr is None:
                fs = None
            else:
                ## Factor of safety
                fs = crr/csr

            ##
            return crr, csr, fs
        
		## if probabilistic, calculate p_liq or equivalent crr
        if flag_det_prob[0] == 'p':
            
            ## check what inputs are given
            if p_liq is None:
                Co = 2.67 # recommended on pp. 1192 in BI12
                sigma_R = 0.13 # recommended on pp. 1192 in BI12

                ## Probability of liquefaction
                p_liq = norm.cdf(-(resistance/crr_c1 + (resistance/crr_c2)**2 - \
								 (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - \
								 Co - np.log(csr))/sigma_R) # eq. 31 in BI12 and 36 in BI16

                ## Cyclic resistance ratio
                crr = np.nan # not used
                
            else:
                ## Inverse analysis of CRR given pLiq, (Cetin et al., 2004)
                crr = np.exp(resistance/crr_c1 + (resistance/crr_c2)**2 - \
							 (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - \
							 Co + sigma*norm.ppf(p_liq)) # eq. 30 in BI12 and 34 in BI16

            ##
            return crr, csr, p_liq


#####################################################################################################################
##### Moss et al. (2006a) CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential
#####################################################################################################################
def moss_etal_2006(**kwargs):
    
    ############ Inputs ############
    qc_1 = kwargs.get('qc_1',None) # qc corrected for overburden
    Rf = kwargs.get('Rf',None) # %, friction ratio = fs/qc
    c = kwargs.get('Rf',None) # %, overburden normalization exponent
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    p_liq = kwargs.get('p_liq',None) # decimals, probability of liquefaction, if want to determine equivalent CRR
	
    # flag for cases 1 and 2: measurement/estimation errors included/removed, default to remove (Cetin et al., 2004)
#     flag_include_error = kwargs.get('flag_include_error',False)

	## check if enough base inputs are given
    if qc_1 is None or Rf is None or c is None or M is None or sv0_eff is None:
        
        ##
        csr = None
        crr = None
		p_liq = None
        print('Not enough inputs to proceed with procedure')
    
	##
    else:
        
        ## Duration weighting factor (or magnitude scaling factor)
        dwf = 17.84*M**-1.43
        if M < 5.5 or M > 8.5:
            print('Magnitude is outside the valid range of 5.5-8.5')

        ## Cyclic stress ratio
        csr_eq = 0.65*amax*sv0_tot/sv0_eff*rd # eq. 1, uncorrected for magnitude and duration
        csr = csr_eq/dwf # correction for duration, eq. 6

        ## Increase in qc_1 and friction ratio due to frictional effects of apparent "fines" and character
        if Rf <= 0.5:
            d_qc = 0
		##
        else:
            x1 = 0.38*(min(Rf,5)) - 0.19
            x2 = 1.46*(min(Rf,5)) - 0.73
            d_qc = x1*np.log(csr) + x2

		## apply correction
        qc_1_mod = qc_1 + d_qc

        ## Model parameters
        t1 = 1.045
        t2 = 0.110
        t3 = 0.001
        t4 = 0.850
        t5 = 7.177
        t6 = 0.848
        t7 = 0.002
        t8 = 20.923
        sigma = 1.632

        ## check what inputs are given
        if p_liq is None:

            ## Probability of liquefaction
            p_liq = norm.cdf(-(qc_1_mod**t1 + qc_1_mod*(t2*Rf) + t3*Rf + \
                               c*(1+t4*Rf) - t5*np.log(csr) - t6*np.log(M) - \
                               t7*np.log(sv0_eff) - t8)/sigma) # eq. 20

            ## Cyclic resistance ratio
            crr = np.nan # not used

		##
        else:

            ## Inverse analysis of CRR given pLiq
            crr = np.exp((qc_1_mod**t1 + qc_1_mod*(t2*Rf)  + t3*Rf + \
                          c*(1+t4*Rf) - t6*np.log(M) - t6*np.log(M) - \
                          t7 + sigma*norm.ppf(p_liq))/t5) # eq. 21
    
    ##
    return crr, csr, p_liq
	
	
#####################################################################################################################
##### Zhu et al. (2017) An Updated Geospatial Liquefaction Model for Global Application
#####################################################################################################################
def zhu_etal_2017(pgv, vs30, precip, dc, dr, dw, wtd, M):
    
    ## Input parameters
    ## - pgv = peak ground velocity (cm/s)
    ## - vs30 = shear-wave velocity over the first 30 m (slope derived) (m/s)
    ## - precip = mean annual precipitation (mm)
    ## - dc = distance to nearest coast (km)
    ## - dr = distance to nearest river (km)
    ## - wtd = global water table depth (m)
    
    ## two models by Zhu et al. (2017):
    ## -- Model 1: better globally
    ## -- Model 2: better for noncoastal (coastal cutoff at 20 km?)

    sf = 1/(1+np.exp(-2*(M-6))) # ad-hoc magnitude scaling factor added by USGS, to be applied to pgv
                                # https://earthquake.usgs.gov/data/ground-failure/background.php
    
	## cutoff distance to coast at 20 km
    if dc < 20:
        ## model coefficients
        b_0 = 12.435 # intercept
        b_lnpgv = 0.301 # cm/s, peak ground velocity
        b_vs30 = -2.615 # m/s, shear-wave velocity over the first 30 m (slope derived)
        b_precip = 5.556e-4 # mm, mean annual precipitation
        b_sqrt_dc = -0.0287 # km, distance to nearest coast
        b_dr = 0.0666 # km, distance to nearest river
        b_sqrt_dc_dr = -0.0369 #

        ## probability of liquefaction - sum of model variables
        X_p_liq = b_0 + b_lnpgv*np.log(pgv*sf) + b_vs30*np.log(vs30) + b_precip*precip + \
                     b_sqrt_dc*np.sqrt(dc) + b_dr*dr + b_sqrt_dc_dr*np.sqrt(dc)*dr    

        ## coeffcients for areal liquefaction
        a = 42.08
        b = 62.59
        c = 11.43

	##
    else:
        ## model coefficients
        b_0 = 8.801 # intercept
        b_lnpgv = 0.334 # cm/s, peak ground velocity
        b_vs30 = -1.918 # m/s, shear-wave velocity over the first 30 m (slope derived)
        b_precip = 5.408e-4 # mm, mean annual precipitation
        b_dw = -0.2054 # km, distance to nearest water body
        b_wtd = -0.0333 # m, global water table depth

        ## probability of liquefaction - sum of model variables
        X_p_liq = b_0 + b_lnpgv*np.log(pgv*sf) + b_vs30*np.log(vs30) + b_precip*precip + b_dw*dw + b_wtd*wtd

        ## coeffcients for areal liquefaction
        a = 49.15
        b = 42.40
        c = 9.165
    
    ## probability of liquefacion
    if pgv < 3 or vs30 > 620:
        p_liq = 0
    else:
        p_liq = 1 / (1 + np.exp(-X_p_liq))

    ## areal liquefaction percent
    p_liq_areal = a / (1 + b * np.exp(-c * p_liq))**2

    ## liquefaction susceptibility 
    susc_liq = X_p_liq - b_lnpgv*np.log(pgv*sf)

	## if on water (distance to coast < 0)
    if dc < 0:
	
		## set outputs to be invalid
        p_liq = -999
        p_liq_areal = -999
        susc_liq = -999

	##
    return p_liq, p_liq_areal, susc_liq
	

#####################################################################################################################
##### FEMA (2004) HAZUS - after Liao et al. (1988)
#####################################################################################################################
def hazus_2004(pga, M, d_w, susc_liq):
    """

    """
    
    ## Correlations based on liquefaction susceptibility
    if susc_liq.lower() == 'very high':
        p_liq_pga = np.maximum(np.minimum(9.09*pga-0.82,1),0)
        p_ml = 0.25
    elif susc_liq.lower() == 'high':
        p_liq_pga = np.maximum(np.minimum(7.67*pga-0.92,1),0)
        p_ml = 0.20
    elif susc_liq.lower() == 'moderate':
        p_liq_pga = np.maximum(np.minimum(6.67*pga-1.00,1),0)
        p_ml = 0.10
    elif susc_liq.lower() == 'low':
        p_liq_pga = np.maximum(np.minimum(5.57*pga-1.18,1),0)
        p_ml = 0.05
    elif susc_liq.lower() == 'very low':
        p_liq_pga = np.maximum(np.minimum(4.16*pga-1.08,1),0)
        p_ml = 0.02
    elif susc_liq.lower() == 'none':
        p_liq_pga = np.ones(len(pga))*0.00
        p_ml = 0.00
    else:
        p_liq_pga = np.ones(len(pga))*np.nan
        p_ml = np.nan

    ## Liquefaction likelihood, p_liq    
    k_m = 0.0027 * M**3 - 0.0267 * M**2 - 0.2055 * M + 2.9188
    k_w = 0.022 * d_w + 0.93
    p_liq = p_liq_pga / k_m / k_w * p_ml