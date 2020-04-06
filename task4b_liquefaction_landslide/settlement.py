#####################################################################################################################
##### Methods for ground settlement
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
### Cetin et al. (2009) Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements
### Peterson (2016) Dissertation - Development of a Performance-Based Procedure for Assessment of Liquefaction-Induced Free-Field
###                 Settlements
#####################################################################################################################
def cetin_etal_2009(**kwargs):
    
    ############ Inputs ############
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    Dr = kwargs.get('Dr',None) # %, relative density
    patm = kwargs.get('patm',101.325) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    
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
    eps_v = np.exp(ln_eps_v)
    sigma = 0.689
    
    ## maximum volumetric strain, after Huang (2008)
    eps_v_max = 9.765 - 2.427*np.log(N1_60_cs) # %
    eps_v = min(eps_v, eps_v_max)
    
    ##
    eps_v = eps_v * 1.15 # correction factor suggested in Peterson (2016)
    
    ##
    return eps_v, sigma
	

#####################################################################################################################
### Ishihara and Yoshimine (1992) Evaluation of Settlements in Sand Deposits Following Liquefaction During Earthquakes
#####################################################################################################################
def ishihara_yoshimine_1992(N1_60_cs, Dr, FS_liq):
    
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
    eps_v = 1.5*np.exp(-0.369 * N1_60_cs**(0.5)) * min(0.08,gamma_max)
    
    ##
    eps_v = eps_v * 0.9 ## Correction suggested by Cetin et al. (2009)

    ##
    return eps_v