#####################################################################################################################
##### Get liquefaction triggering parameters CRR, CSR, FS, and or pLiq
#####################################################################################################################


#####################################################################################################################
##### Youd et al. (2001) Liquefaction Resistance of Soils (NCEER)
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
#####################################################################################################################
def boulanger_idriss_2014(**kwargs):
    
    ############ Inputs ############
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
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
        msf_max = min(2.2,1.09+(N1_60_cs/31.5)**2) # eq. 2.21
        msf = 1 + (msf_max-1) * (8.64*np.exp(-M/4)-1.325) # eq. 2.19, Boulanger and Idriss (2014)
    
    ## overburden correction factor
    C_sigma = min(0.3,1/(18.9-2.55*N1_60_cs**0.5)) # eq. 2.16c
    if N1_60_cs > 37:
        print('N1_60_cs > 37, over the limit described in Boulanger and Idriss (2014)')
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
    if N1_60_cs is None:
        print('N1_60_cs is not provided as an input, exiting procedure')
        return None
    
    else:
		## if deterministic, calculate factor of safety
        if flag_det_prob[0] == 'd':
            
            Co = 2.8 # value for deterministic analysis, pp. 1197 in BI12
            
            ## SPT base CRR curve (cyclic resistance ratio)
            crr = np.exp(N1_60_cs/14.1 + (N1_60_cs/126)**2 - (N1_60_cs/23.6)**3 + (N1_60_cs/25.4)**4 - Co) # eq. 2.25 in BI14

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

                ## Probability of liquefaction (Cetin et al., 2004)
                p_liq = norm.cdf(-(N1_60_cs/14.1 + (N1_60_cs/126)**2 - (N1_60_cs/23.6)**3 + \
                                   (N1_60_cs/25.4)**4 - Co - np.log(csr))/sigma_R) # eq. 31 in BI12

                ## Cyclic resistance ratio
                crr = np.nan # not used
                
            else:

                ## Inverse analysis of CRR given pLiq, (Cetin et al., 2004)
                crr = np.exp(N1_60_cs/14.1 + (N1_60_cs/126)**2 - (N1_60_cs/23.6)**3 + \
                            (N1_60_cs/25.4)**4 - Co + sigma*norm.ppf(p_liq)) # eq. 30 in BI12

            ##
            return crr, csr, p_liq