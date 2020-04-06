#####################################################################################################################
##### Get liquefaction triggering parameters CRR, CSR, FS, and or pLiq
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Moss et al. (2006a) CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential
#####################################################################################################################
def moss_etal_2006(**kwargs):
    
    ############ Inputs ############
    qc_1 = kwargs.get('qc_1',None) # qc corrected for overburden
    Rf = kwargs.get('Rf',None) # %, friction ratio = fs/qc
    c = kwargs.get('Rf',None) # %, friction ratio = fs/qc
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
##### Idriss and Boulanger (2008) Soil Liquefaction During Earthquakes (Monograph)
##### Boulanger and Idriss (2014) CPT and SPT Based Liquefaction Triggering Procedures (report)
##### Boulanger and Idriss (2016) CPT Based Liquefaction Triggering Procedure (journal)
#####################################################################################################################
def boulanger_idriss_2014(**kwargs):
    
    ############ Inputs ############
    qc_1N_cs = kwargs.get('qc_1N_cs',None) # corrected CPT tip measurement
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
        msf_max = min(2.2,1.09+(qc_1N_cs/180)**3) # eq. 2.20
        msf = 1 + (msf_max-1) * (8.64*np.exp(-M/4)-1.325) # eq. 2.19, Boulanger and Idriss (2014)
    
    ## overburden correction factor
    C_sigma = min(0.3,1/(37.3-8.27*qc_1N_cs**0.264)) # eq. 2.16b
    if qc_1N_cs > 211:
        print('qc_1N_cs > 211, over the limit described in Boulanger and Idriss (2014)')
    K_sigma = min(1.1,1 - C_sigma*np.log(sv0_eff/patm)) # eq. 2.16a
        
    ## static shear stress correction factor
    if tau_stat is None:
        K_alpha = 1.0 # eq. 31
    else:
        xi_r = 1/(Q-np.log(100*(1+2*K0)/3*sv0_eff/patm)) - (N1_60_cs/46)**0.5 # eq. 64
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
    if qc_1N_cs is None:
        print('Not enough inputs to proceed with procedure')
		return None
		
    else:
		## if deterministic, calculate factor of safety
        if flag_det_prob[0] == 'd':
            
            Co = 2.8 # value for deterministic analysis, pp. 1197 in BI12
            
            ## SPT base CRR curve (cyclic resistance ratio)
            crr = np.exp(qc_1N_cs/113 + (qc_1N_cs/1000)**2 - (qc_1N_cs/140)**3 + (qc_1N_cs/137)**4 - Co) # eq. 2.24 in BI14

            if crr is None or csr is None:
                fs = None
            else:
                ## Factor of safety
                fs = crr/csr

            ##
            return crr, csr, fs
        
		## if probabilistic, calculate p_liq or equivalent crr
        elif flag_det_prob[0] == 'p':
            
            ## check what inputs are given
            if p_liq is None:
                
                Co = 2.60 # recommended on pp. 9 in BI16
                sigma_R = 0.20 # recommended on pp. 1192 in BI16

                ## Probability of liquefaction (Cetin et al., 2004)
                p_liq = norm.cdf(-(qc_1N_cs/113 + (qc_1N_cs/1000)**2 - (qc_1N_cs/140)**3 + \
                                   (qc_1N_cs/137)**4 - Co - np.log(csr))/sigma_R) # eq. 31 in BI12

                ## Cyclic resistance ratio
                crr = np.nan # not used
				
            ##
            else:
			
                ## Inverse analysis of CRR given pLiq, (Cetin et al., 2004)
                crr = np.exp(qc_1N_cs/113 + (qc_1N_cs/1000)**2 - (qc_1N_cs/140)**3 + \
                            (qc_1N_cs/137)**4 - Co + sigma*norm.ppf(p_liq)) # eq. 30 in BI12


			##
			return crr, csr, p_liq