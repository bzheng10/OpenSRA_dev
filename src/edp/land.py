#####################################################################################################################
##### Methods for landslide deformation
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Grant et al. (2016) Multimodal method for coseismic landslide hazard assessment
#####################################################################################################################
def grant_etal_2016(flag_failure, **kwargs):
    
    ## Input variable dictionary:
    ## - pga = peak ground acceleration (g)
    ## - phi = friction angle
    ## - c = cohesion
    ## - beta = slope angle
    ## - gamma = unit weight
    ## - H = local relief/height of slope
    ## - y = cell size of assessed pixel
    ## - M = moment magnitude
    ## - z = elevation
    ## - dr = distance to nearest river
    ## - susc_liq = liquefaction susceptibility
    
    
    ## Four types of failures quantified:
    ## 1) Rock-slope failures:
    ##    - inpus = phi, c, beta, gamma, H, pga
    ## 2) Disrupted soil slides:
    ##    - inputs = phi, c, cr, beta, gamma, t, pga
    ## 3) Coherent rotational slides:
    ##    - inputs = phi, c, beta, gamma, H, y, pga
    ## 4) Lateral spreading:
    ##    - inputs = susc_liq, pga, M, z, dr
    ##    - a = vertical distance between center of slip circle and landslide body's centroid
    ##
    
	###############################
    ## 1) Rock-slope failures:
	###############################
    if flag_failure == 1:
        ## assign passed values to parameters
        phi = kwargs['phi']
        c = kwargs['c']
        beta = kwargs['beta']
        gamma = kwargs['gamma']
        H = kwargs['H']
        pga = kwargs['pga']
        ## calculation
        alpha = (beta + phi)/2 # rad, slope's critical angle
        theta = alpha # rad, rock-slope failure angle, theta varies between material and failure scenario
        h = 0.25*H # m, vertical height of failure mass
        FS = 2*c*np.sin(np.radians(beta))/(gamma*h*np.sin(np.radians(beta-alpha))*np.sin(np.radians(alpha))) +\
                                           np.tan(np.radians(phi))/np.tan(np.radians(alpha)) # factor of safety
        ky = (FS-1)*np.sin(np.radians(theta)) # g, yield acceleration		
        if pga > ky:
            D = np.exp(0.215 + np.log((1-ky/pga)**2.341 * (ky/pga)**-1.438)) # cm, coseismic displacement, pga and ky in units of g
        else:
            D = 0
    
	###############################
    elif flag_failure == 2:
	###############################
        ## assign passed values to parameters
        phi = kwargs['phi']
        c = kwargs['c']
        cr = kwargs['cr']
        beta = kwargs['beta']
        gamma = kwargs['gamma']
        t = kwargs['t']
        pga = kwargs['pga']
        ## calculation
        theta = beta # rad, infinite slope condition, theta varies between material and failure scenario
        FS = (c+cr)/(gamma*t*np.sin(np.radians(beta))) + np.tan(np.radians(phi))/np.tan(np.radians(beta)) # factor of safety
        ky = (FS-1)*np.sin(np.radians(theta)) # g, yield acceleration
        if pga > ky:
            D = np.exp(0.215 + np.log((1-ky/pga)**2.341 * (ky/pga)**-1.438)) # cm, coseismic displacement, pga and ky in units of g
        else:
            D = 0
    
	###############################
    elif flag_failure == 3:
	###############################
        ## assign passed values to parameters
        phi = kwargs['phi']
        c = kwargs['c']
        beta = kwargs['beta']
        gamma = kwargs['gamma']
        H = kwargs['H']
        y = kwargs['y']
        pga = kwargs['pga']
        ## calculation
        R = 1.5*H # units of H, radius of circular failure plane acting through a dry homogeneous hillslope
        delta = np.arcsin(1/(3*np.sin(np.radians(beta)))) # rad, for beta > 20 degrees
        L = 2*delta*R # units of R, failure plane length
        a = (4*R*(np.sin(delta))**3 / (3*(2*delta-np.sin(2*delta))) - R) * np.cos(delta) # m, landslide body's centroid
        W = 1/2*gamma*y*R**2*(2*delta - np.sin(2*delta)) # kN
        ky = (c*L*y + W*(np.cos(np.radians(beta))*np.tan(np.radians(phi)) - np.sin(np.radians(beta)))) / \
              (W*(a/R + np.sin(np.radians(beta))*np.tan(np.radians(phi)))) # g, yield acceleration
        if pga > ky:
            D = np.exp(0.215 + np.log((1-ky/(0.5*pga))**2.341 * (ky/(0.5*pga))**-1.438)) # cm, coseismic displacement, 
                                                                                         # pga and ky in units of g, factor of 50% applied to pga
        else:
            D = 0

	###############################
    elif flag_failure == 4:
	###############################
        ## assign passed values to parameters
        susc_liq = kwargs['susc_liq']
        pga = kwargs['pga']
        M = kwargs['M']
        z = kwargs['z']
        dr = kwargs['dr']
        ## calculation
        ## get threshold pga against liquefaction
        if susc_liq.lower() == 'very high':
            pga_t = 0.09 # g
        elif susc_liq.lower() == 'high':
            pga_t = 0.12 # g
        elif susc_liq.lower() == 'moderate':
            pga_t = 0.15 # g
        elif susc_liq.lower() == 'low':
            pga_t = 0.21 # g
        elif susc_liq.lower() == 'very low':
            pga_t = 0.26 # g
        elif susc_liq.lower() == 'none':
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
            D = Kdelta*r
        else:
            D = 0

	##
    return D, ky
	

#####################################################################################################################
##### Jibson (2007) Regression models for estimating coseismic landslide displacement
##### Saygili (2008) Dissertation - A probabilistic approach for evaluating earthquake-induced landslides
#####################################################################################################################
def jibson_2007(**kwargs):
    
    ## Get inputs
    pga = kwargs.get('pga',None) # g, peak ground acceleration = amax in paper
    M = kwargs.get('M',None) # moment magnitude
    Ia = kwargs.get('Ia',None) # m/s, arias intensity
    ky = kwargs.get('ky',None) # g, yield acceleration = ac in paper, either provided or computed below
    
    ## Check if ky is provided, if not then compute it
    if ky is None:
        
        ## Get inputs to calculate ky
        alpha = kwargs.get('alpha',None) # deg, slope angle
        FS = kwargs.get('FS',None) # factor of safety, either provided or computed below
    
        ## Try to calculate ky
        try:
            ky = (FS-1)*np.sin(np.radians(alpha)) ## eq. 1 in Jibson (2007)
            
        except:
            print('Not enough inputs to calculate ky - need factor of safety and slope angle')
            ky = None
        
    ###########################################################
    ## Pre-define output variables
    D_pga = None
    sig_pga = None

    D_pga_M = None
    sig_pga_M = None

    D_Ia = None
    sig_Ia = None

    D_pga_Ia = None
    sig_pga_Ia = None
        
    ###########################################################
    if pga is None or ky is None:
        print('Requires at the minimum pga and ky as inputs; cannot proceed with procedure')
    
    else:
        ###########################################################
        ## Model as a function of pga and ky only
        
        ## displacement, cm
        D_pga = 10**(0.215 + np.log10((1 - ky/pga)**2.341 * (ky/pga)**-1.438)) ## eq. 6 in Jibson (2007)
        
        ## sigma for ln(D)
        sig_pga = 0.510*np.log(10) ## eq. 6 in Jibson (2007)
    
        ###########################################################
        ## Model as a function of pga, ky, and M
        if M is not None:

            ## displacement, cm
            D_pga_M = 10**(-2.710 + np.log10((1 - ky/pga)**2.335 * (ky/pga)**-1.478) + 0.424*M) ## eq. 7 in Jibson (2007)
            
            ## sigma for ln(D)
            sig_pga_M = 0.454*np.log(10) ## eq. 7 in Jibson (2007)
            
        ###########################################################
        if Ia is not None:

            ###########################################################
            ## Model as a function of ky and Ia (no pga)
            ## displacement, cm
            D_Ia = 10**(2.401*np.log10(Ia) - 3.481*np.log10(ky) - 3.230) ## eq. 9 in Jibson (2007)
            
            ## sigma for ln(D)
            sig_Ia = 0.656*np.log(10) ## eq. 9 in Jibson (2007)
            
            ###########################################################
            ## Model as a function of ky, Ia, and pga
            ## displacement, cm
            D_pga_Ia = 10**(0.561*np.log10(Ia) - 3.833*np.log10(ky/pga) - 1.474) ## eq. 10 in Jibson (2007)
            
            ## sigma for ln(D)
            sig_pga_Ia = 0.616*np.log(10) ## eq. 10 in Jibson (2007)
    
    
    ###
    outMat = np.array([[D_pga,        sig_pga],
                       [D_pga_M,      sig_pga_M],
                       [D_Ia,    	  sig_Ia],
                       [D_pga_Ia,     sig_pga_Ia]])
    
    ##
    return outMat
	
	
#####################################################################################################################
##### Saygili (2008) Dissertation - A probabilistic approach for evaluating earthquake-induced landslides
##### Rathje & Saygili (2008) Empirical predictive models for earthquake-induced sliding displacements of slopes
##### Rathje & Saygili (2009) Probabilistic assessment of earthquakeinduced sliding displacements of natural slopes
#####################################################################################################################
def saygili_2008(**kwargs):
    
    ## Get inputs
    pga = kwargs.get('pga',None) # g, peak ground acceleration
    pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
    M = kwargs.get('M',None) # moment magnitude
    Ia = kwargs.get('Ia',None) # m/s, arias intensity
    Tm = kwargs.get('Tm',None) # sec, mean period
    ky = kwargs.get('ky',None) # g, yield acceleration, either provided or computed below
    
    ## Check if ky is provided, if not then compute it
    if ky is None:
        ## Calculation of ky for infinite slope
        coh = kwargs.get('coh',None) # kPa, effective cohesion
        phi = kwargs.get('phi',None) # deg, effective friction angle
        t = kwargs.get('t',None) # m, slope normal to thickness of failure surface
        m = kwargs.get('m',0) # %, percent of failure thickness that is saturated
        gamma = kwargs.get('gamma',None) # kN/m3, unit weight of soil
        gamma_w = kwargs.get('gamma_w',9.81) # kN/m3, unit weight of water
        alpha = kwargs.get('alpha',None) # deg, slope angle
        
        try:
            FS = kwargs.get('FS',None) # factor of safety from strength, either provided or computed below
            
            if FS is None:
                try:
                    ## eq. 2 in Rathje and Saygili (2011)
                    FS = coh/gamma * t * np.sin(np.radians(alpha)) + \
                         np.tan(np.radians(phi)) / np.tan(np.radians(alpha)) + \
                         gamma_w * m/100 * np.tan(np.radians(phi)) / (gamma * np.tan(np.radians(alpha)))
                except:
                    print('Not enough inputs to calculate FS - see Rathje & Saygili (2011) for all required inputs')
                
            ## eq. 1 in Rathje and Saygili (2011)
            ky = (FS-1) / (np.cos(np.radians(alpha)) * np.tan(np.radians(phi)) + 1/np.tan(np.radians(alpha)))
            
        except:
            print('Not enough inputs to calculate ky - see Rathje & Saygili (2011) for all required inputs')
            ky = None
        
    ###########################################################
    ## Pre-define output variables
    D_pga = None
    sig_pga = None

    D_pga_M = None
    sig_pga_M = None

    D_pga_pgv = None
    sig_pga_pgv = None

    D_pga_pgv_Ia = None
    sig_pga_pgv_Ia = None

    D_pga_Tm = None
    sig_pga_Tm = None

    D_pga_Tm_Ia = None
    sig_pga_Tm_Ia = None

    D_pga_Ia = None
    sig_pga_Ia = None
        
    ###########################################################
    if pga is None or ky is None:
        print('Requires at the minimum PGA and ky as inputs; cannot proceed with procedure')
    
    else:
        ###########################################################
        ## scalar model: f(pga)
        ## Table 4.2 in Saygili (2008) dissertation
        a1 = 5.52
        a2 = -4.43
        a3 = -20.39
        a4 = 42.61
        a5 = -28.74
        a6 = 0.72
        a7 = 0
        
        ## displacement, cm
        D_pga = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
                       a6*np.log(pga)) ## eq. 4.5 in Saygili (2008) dissertation
        
        ## sigma for ln(D)
        sig_pga = 1.13 ## Table 4.2 in Saygili (2008) dissertation
    
        ###########################################################
        ## modified scalar model: f(pga,M)
        if M is not None:
            ## Table 4.2 in Saygili (2008) dissertation
            a1 = 4.89
            a2 = -4.85
            a3 = -19.64
            a4 = 42.49
            a5 = -29.06
            a6 = 0.72
            a7 = 0.89

            ## displacement, cm
            D_pga_M = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
                             a6*np.log(pga) + a7*(M-6)) ## eq. 4.6 in Saygili (2008) dissertation
            
            ## sigma for ln(D)
            sig_pga_M = 0.73 + 0.789*(ky/pga) - 0.539*(ky/pga)**2 ## eq. 4.7 in Saygili (2008) dissertation
                                                                  ## and eq. 9 in Rathje and Saygili (2009)
            
        ###########################################################
        ## two-parameter vector model: f(pga,pgv)
        if pgv is not None:
            ## Table 4.3 in Saygili (2008) dissertation
            a1 = -1.56
            a2 = -4.58
            a3 = -20.84
            a4 = 44.75
            a5 = -30.50
            a6 = -0.64
            a7 = 1.55

            ## displacement, cm
            D_pga_pgv = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 +\
                               a6*np.log(pga) + a7*np.log(pgv)) ## eq. 4.8 in Saygili (2008) dissertation
            
            ## sigma for ln(D)
            sig_pga_pgv = 0.405 + 0.524*(ky/pga) ## Table 4.4 in Saygili (2008) dissertation
                                                 ## and eq. 11 in Rathje and Saygili (2009)
            
            ###########################################################
            ## three-parameter vector model: f(pga,pgv,Ia)
            if Ia is not None:
                ## Table 4.3 in Saygili (2008) dissertation
                a1 = -0.74
                a2 = -4.93
                a3 = -19.91
                a4 = 43.75
                a5 = -30.12
                a6 = -1.30
                a7 = 1.04
                a8 = 0.67

                ## displacement, cm
                D_pga_pgv_Ia = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
                                      a6*np.log(pga) + a7*np.log(pgv) + a8*np.log(Ia)) ## eq. 4.8 in Saygili (2008) dissertation

                ## sigma for ln(D)
                sig_pga_pgv_Ia = 0.20 + 0.79*(ky/pga) ## Table 4.4 in Saygili (2008) dissertation
        
        ###########################################################
        ## two-parameter vector model: f(pga,Tm)
        if Tm is not None:
            ## Table 4.3 in Saygili (2008) dissertation
            a1 = 6.62
            a2 = -3.93
            a3 = -23.71
            a4 = 49.37
            a5 = -32.94
            a6 = -0.93
            a7 = 1.79

            ## displacement, cm
            D_pga_Tm = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 +\
                              a6*np.log(pga) + a7*np.log(Tm)) ## eq. 4.8 in Saygili (2008) dissertation
            
            ## sigma for ln(D)
            sig_pga_Tm = 0.60 + 0.26*(ky/pga) ## Table 4.4 in Saygili (2008) dissertation

            ###########################################################
            ## three-parameter vector model: f(pga,Tm,Ia)
            if Ia is not None:
                ## Table 4.3 in Saygili (2008) dissertation
                a1 = 4.27
                a2 = -4.62
                a3 = -21.49
                a4 = 46.53
                a5 = -31.66
                a6 = -0.57
                a7 = 1.14
                a8 = 0.86

                ## displacement, cm
                D_pga_Tm_Ia = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
                                     a6*np.log(pga) + a7*np.log(Tm) + a8*np.log(Ia)) ## eq. 4.8 in Saygili (2008) dissertation

                ## sigma for ln(D)
                sig_pga_Tm_Ia = 0.19 + 0.75*(ky/pga) ## Table 4.4 in Saygili (2008) dissertation
                
        ###########################################################
        ## two-parameter vector model: f(pga,Ia)
        if Ia is not None:
            ## Table 4.3 in Saygili (2008) dissertation
            a1 = 2.39
            a2 = -5.24
            a3 = -18.78
            a4 = 42.01
            a5 = -29.15
            a6 = -1.56
            a7 = 1.38

            ## displacement, cm
            D_pga_Ia = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 +\
                              a6*np.log(pga) + a7*np.log(Ia)) ## eq. 4.8 in Saygili (2008) dissertation
            
            ## sigma for ln(D)
            sig_pga_Ia = 0.46 + 0.56*(ky/pga) ## Table 4.4 in Saygili (2008) dissertation
    
    
    ###
    outMat = np.array([[D_pga,        sig_pga],
                       [D_pga_M,      sig_pga_M],
                       [D_pga_pgv,    sig_pga_pgv],
                       [D_pga_pgv_Ia, sig_pga_pgv_Ia],
                       [D_pga_Tm,     sig_pga_Tm],
                       [D_pga_Tm_Ia,  sig_pga_Tm_Ia],
                       [D_pga_Ia,     sig_pga_Ia]])
    
    ##
    return outMat


#####################################################################################################################
##### Rathje & Antonakos (2011) A unified model for predicting earthquake-induced sliding displacements of rigid and flexible slopes
#####################################################################################################################
def rathje_antonakos_2011(**kwargs):

    ## Get inputs
    pga = kwargs.get('pga',None) # g, peak ground acceleration
    pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
    M = kwargs.get('M',None) # moment magnitude
    Tm = kwargs.get('Tm',None) # sec, mean period
    Ts = kwargs.get('Tm',None) # sec, site period
    ky = kwargs.get('ky',None) # g, yield acceleration, either provided or computed below
    
    ## Check if ky is provided, if not then compute it
    if ky is None:
        ## Calculation of ky for infinite slope
        coh = kwargs.get('coh',None) # kPa, effective cohesion
        phi = kwargs.get('phi',None) # deg, effective friction angle
        t = kwargs.get('t',None) # m, slope normal to thickness of failure surface
        m = kwargs.get('m',0) # %, percent of failure thickness that is saturated
        gamma = kwargs.get('gamma',None) # kN/m3, unit weight of soil
        gamma_w = kwargs.get('gamma_w',9.81) # kN/m3, unit weight of water
        alpha = kwargs.get('alpha',None) # deg, slope angle
        
        try:
            FS = kwargs.get('FS',None) # factor of safety from strength, either provided or computed below
            
            if FS is None:
                try:
                    ## eq. 2 in Rathje and Saygili (2011)
                    FS = coh/gamma * t * np.sin(np.radians(alpha)) + \
                         np.tan(np.radians(phi)) / np.tan(np.radians(alpha)) + \
                         gamma_w * m/100 * np.tan(np.radians(phi)) / (gamma * np.tan(np.radians(alpha)))
                except:
                    print('Not enough inputs to calculate FS - see Rathje & Saygili (2011) for all required inputs')
                
            ## eq. 1 in Rathje and Saygili (2011)
            ky = (FS-1) / (np.cos(np.radians(alpha)) * np.tan(np.radians(phi)) + 1/np.tan(np.radians(alpha)))
            
        except:
            print('Not enough inputs to calculate ky - see Rathje & Saygili (2011) for all required inputs')
            ky = None
        
    ###########################################################
    ## Pre-define output variables
    D_pga_M_flex = None
    sig_pga_M_flex = None

    D_pga_pgv_flex = None
    sig_pga_pgv_flex = None

    ###########################################################
    if pga is None or ky is None:
        print('Not enough inputs: cannot proceed with evaluation')

    else:
    
        ## compute Kmax to be used in place of pga
        try:
            if Ts/Tm >= 0.1:
                Kmax = pga * np.exp((0.459 - 0.702*pga) * np.log((Ts/Tm)/0.1) + \
                                    (-0.228 + 0.076*pga) * np.log((Ts/Tm)/0.1)**2)
            else:
                Kmax = pga * np.exp(0)

        except:
            print('Not enough inputs for Kmax; setting Kmax = pga')
            Kmax = pga

        print(Kmax)
        ###########################################################
        ## modified scalar model: f(pga,M)
        if M is not None:
            
            ## Table 4.2 in Saygili (2008) dissertation
            a1 = 4.89
            a2 = -4.85
            a3 = -19.64
            a4 = 42.49
            a5 = -29.06
            a6 = 0.72
            a7 = 0.89

            ## cm, displacement of rigid sliding mass
            D_pga_M = np.exp(a1 + a2*(ky/Kmax) + a3*(ky/Kmax)**2 + a4*(ky/Kmax)**3 + a5*(ky/Kmax)**4 + \
                             a6*np.log(Kmax) + a7*(M-6)) ## eq. 4.6 in Saygili (2008) dissertation
            
            ## cm, correct displacement for rigid mass for site flexibility, eq. 3 in Rathje and Antonakos (2011)
            if Ts <= 1.5:
                D_pga_M_flex = np.exp(np.log(D_pga_M) + 3.69*Ts - 1.22*Ts**2)
            else:
                D_pga_M_flex = np.exp(np.log(D_pga_M) + 2.78)

            ## sigma for ln(D_flex)
            sig_pga_M_flex = 0.694 + 0.32*(ky/Kmax) ## eq. 5 in Rathje and Antonakos (2011)
            
        ###########################################################
        ## two-parameter vector model: f(pga,pgv)
        if pgv is not None:
            
            ## compute K_velmax to be used in place of pgv
            try:
                if Ts/Tm >= 0.2:
                    K_velmax = pgv * np.exp(0.240 * np.log((Ts/Tm)/0.2) + \
                                            (-0.091 - 0.171*pga) * np.log((Ts/Tm)/0.2)**2)
                else:
                    K_velmax = pgv * np.exp(0)
                    
            except:
                print('Not enough inputs for K_velmax; setting K_velmax = pgv')
                K_velmax = pgv
                
            ## Table 4.3 in Saygili (2008) dissertation
            a1 = -1.56
            a2 = -4.58
            a3 = -20.84
            a4 = 44.75
            a5 = -30.50
            a6 = -0.64
            a7 = 1.55

            ## cm, displacement of rigid sliding mass
            D_pga_pgv = np.exp(a1 + a2*(ky/Kmax) + a3*(ky/Kmax)**2 + a4*(ky/Kmax)**3 + a5*(ky/Kmax)**4 +\
                               a6*np.log(Kmax) + a7*np.log(K_velmax)) ## eq. 4.8 in Saygili (2008) dissertation
            
            ## cm, correct displacement for rigid mass for site flexibility, eq. 3 in Rathje and Antonakos (2011)
            if Ts <= 0.5:
                D_pga_pgv_flex = np.exp(np.log(D_pga_pgv) + 1.42*Ts)
            else:
                D_pga_pgv_flex = np.exp(np.log(D_pga_pgv) + 0.71)
    
            ## sigma for ln(D_flex)
            sig_pga_pgv_flex = 0.400 + 0.284*(ky/Kmax) ## eq. 5 in Rathje and Antonakos (2011)
            
    ###
    outMat = np.array([[D_pga_M_flex,      sig_pga_M_flex],
                       [D_pga_pgv_flex,    sig_pga_pgv_flex]])
    
    ##
    return outMat
	

#####################################################################################################################
##### Bray and Travasarou (2007) Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements
#####################################################################################################################
def bray_travasarou_2007(**kwargs):
    
    ## Demand
    M = kwargs.get('M',None) ## moment magnitude
    T_arr = kwargs.get('T_arr',None) ## sec, periods for the input spectral accelerations, used to interpolate Sa(Ts_deg)
    Sa_arr = kwargs.get('Sa_arr',None) ## g, spectral accelerations used to interpolate Sa(Ts_deg)
    slope = kwargs.get('slope',None) ## deg, shallow = <30, moderately steep = 30 to 60, steep = > 60, Ashford & Sitar (2002)
    eps = kwargs.get('eps',None) ## number of standard deviation
    pga = kwargs.get('pga',None) ## g, peak ground acceleration
    flag_topo = kwargs.get('flag_topo','local') ## localized versus long potential sliding mass
            
            
    ## Correct pga for steepness of slope and localization of failure
    if pga is not None:
        if 'long' in flag_topo:
            f_pga = 0.65 ## long potential sliding mass
        elif 'local' in flag_topo: ## localized potential sliding mass
            ## factor for increase in pga given steepness of slope
            if slope is None:
                f_pga = 1 ## default
            else:
                if slope > 30 and slope <= 60:
                    f_pga = 1.3 ## Rathje & Bray (2001)
                elif slope > 60:
                    f_pga = 1.5 ## Ashford & Sitar (2002)
                else:
                    f_pga = 1 ## default
        pga_corr = pga*f_pga ## slope-corrected pga
    else:
        f_pga = None
        pga_corr = None

        
    ## Fundamental period of structure:
    Ts = kwargs.get('Ts',None) ## sec, allow user-defined Ts = 0
    Sa_Ts_deg = kwargs.get('Sa_Ts_deg',None)
    
    if Ts == 0:
        Ts_deg = None
        Sa_Ts_deg = None
    else:
        if Sa_Ts_deg is None:
            if Ts is None:
                n_Ts = kwargs.get('n_Ts',1) ## 1D versus 2D case for fundamental period
                if n_Ts == 1:
                    f_Ts = 4 ## 1D = trapezoidal-shaped sliding mass
                elif n_Ts == 2:
                    f_Ts = 2.6 ## 2D = triangular-shaped sliding mass
                h = kwargs.get('h',None) ## m, thickness/height of structure
                vs = kwargs.get('vs',None) ## m/sec, shear wave velocity
                Ts = f_Ts*h/vs ## sec, fundamental period of site/structure
            Ts_deg = 1.3*Ts ## sec, degraded period
            Sa_Ts_deg = np.exp(np.interp(Ts_deg, T_arr, np.log(Sa_arr))) ## g, spectral acceleration at degraded period
    
    
    ## Strength
    ky = kwargs.get('ky',None) ## yield coefficient, from pseudostatic analysis or Grant et al. (2016) correlations
    
    
    ## standard deviation
    sigma = 0.66

    ## period-dependent coefficients and parameters
    if Ts < 0.05:
        a1 = -0.22
        a2 = 0.0
        Sa = pga_corr
    else:
        a1 = -1.10
        a2 = 1.50
        Sa = Sa_Ts_deg

    ## seismic displacement, cm (eq. 5)
    lnD = a1 - 2.83*np.log(ky) - 0.333*(np.log(ky))**2 + 0.566*np.log(ky)*np.log(Sa) +\
          3.04*np.log(Sa) - 0.244*(np.log(Sa))**2 + a2*Ts + 0.278*(M-7)

    ## probably of displacement = 0 (threshold = 1 cm), eq. 3
    p_D_eq_0 = 1-norm.cdf(-1.76 - 3.22*np.log(ky) - 0.484*Ts*np.log(ky) + 3.52*np.log(Sa))


    ## other percentiles
    lnD_84 = lnD - sigma
    lnD_16 = lnD + sigma
    
    ## convert from ln to d
    D = np.exp([lnD_84, lnD, lnD_16]) # cm
    
    ## calculate p(D > d) for given d array
    d_arr = kwargs.get('d_arr',None)
    if d_arr is not None:
        for i in d_arr:
            p_d_gt_d = (1-p_D_eq_0) * (1-norm.sf(np.log(i), loc=lnD, scale=sigma)) # eq. 10 and 11
    else:
        p_d_gt_d = None
        
    ##
    return D, p_D_eq_0, p_d_gt_d
	
	
#####################################################################################################################
##### Bray and Macedo (2019) Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes
#####################################################################################################################
def bray_macedo_2019(**kwargs):
    
    ## Demand
    gm_type = kwargs.get('gm_type','gen') ## takes ord = ordinary GMs, nf = near-fault GMs, gen/full = general or full (ordinary + near-fault) GMs
    M = kwargs.get('M',None) ## moment magnitude
    T_arr = kwargs.get('T_arr',None) ## sec, periods for the input spectral accelerations, used to interpolate Sa(Ts_deg)
    Sa_arr = kwargs.get('Sa_arr',None) ## g, spectral accelerations used to interpolate Sa(Ts_deg)
    slope = kwargs.get('slope',None) ## deg, shallow = <30, moderately steep = 30 to 60, steep = > 60, Ashford & Sitar (2002)
    eps = kwargs.get('eps',None) ## number of standard deviation
    pga = kwargs.get('pga',None) ## g, peak ground acceleration
    pgv = kwargs.get('pgv',None) ## cm/s, peak ground velocity
    flag_topo = kwargs.get('flag_topo','local') ## localized versus long potential sliding mass
    slope_loc = kwargs.get('slope_loc',90) ## deg, slope location with respect to fault-normal for near-fault GMs
                                           ## if <= 45, output D100, else, output D50
                                           ## default to 90 deg (fault parallel)
            
            
    ## Correct pga for steepness of slope and localization of failure
    if pga is not None:
        if 'long' in flag_topo:
            f_pga = 0.65 ## long potential sliding mass
        elif 'local' in flag_topo: ## localized potential sliding mass
            ## factor for increase in pga given steepness of slope
            if slope is None:
                f_pga = 1 ## default
            else:
                if slope > 30 and slope <= 60:
                    f_pga = 1.3 ## Rathje & Bray (2001)
                elif slope > 60:
                    f_pga = 1.5 ## Ashford & Sitar (2002)
                else:
                    f_pga = 1 ## default
        pga_corr = pga*f_pga ## slope-corrected pga
    else:
        f_pga = None
        pga_corr = None

        
    ## Fundamental period of structure:
    Ts = kwargs.get('Ts',None) ## sec, allow user-defined Ts = 0
    Sa_Ts_deg = kwargs.get('Sa_Ts_deg',None)
    
    if Ts == 0:
        Ts_deg = None
        Sa_Ts_deg = None
    else:
        if Sa_Ts_deg is None:
            if Ts is None:
                n_Ts = kwargs.get('n_Ts',1) ## 1D versus 2D case for fundamental period
                if n_Ts == 1:
                    f_Ts = 4 ## 1D = trapezoidal-shaped sliding mass
                elif n_Ts == 2:
                    f_Ts = 2.6 ## 2D = triangular-shaped sliding mass
                h = kwargs.get('h',None) ## m, thickness/height of structure
                vs = kwargs.get('vs',None) ## m/sec, shear wave velocity
                Ts = f_Ts*h/vs ## sec, fundamental period of site/structure
            Ts_deg = 1.3*Ts ## sec, degraded period
            Sa_Ts_deg = np.exp(np.interp(Ts_deg, T_arr, np.log(Sa_arr))) ## g, spectral acceleration at degraded period
    
    
    ## Strength
    ky = kwargs.get('ky',None) ## yield coefficient, from pseudostatic analysis or Grant et al. (2016) correlations
    
    
    ## Calculations:
    if 'ord' in gm_type:
        
        ## standard deviation
        sigma = 0.72
        
        ## period-dependent coefficients and parameters
        if Ts == 0:
            a1 = -4.684
            a2 = 0.0  # -9.471 in spreadsheet, but Ts = 0, so mathematically consistent
            a3 = 0.0
            Sa = pga_corr
        elif Ts > 0 and Ts < 0.1:
            a1 = -4.684
            a2 = -9.471
            a3 = 0.0
            Sa = Sa_Ts_deg
        elif Ts >= 0.1:
            a1 = -5.981
            a2 = 3.223
            a3 = -0.945
            Sa = Sa_Ts_deg

        ## seismic displacement, cm (eq. 3)
        lnD = a1 - 2.482*np.log(ky) - 0.244*(np.log(ky))**2 + 0.344*np.log(ky)*np.log(Sa) +\
              2.649*np.log(Sa) - 0.090*(np.log(Sa))**2 + a2*Ts + a3*(Ts)**2 + 0.603*M

        ## probably of displacement = 0 (threshold = 0.5 cm)
        if Ts <= 0.7:
            ## eq. 2(a)
            p_D_eq_0 = 1-norm.cdf(-2.48 - 2.97*np.log(ky) - 0.12*(np.log(ky))**2 - 0.72*Ts*np.log(ky) +\
                                  1.70*Ts + 2.78*np.log(Sa))
        elif Ts > 0.7:
            ## eq. 2(b)
            p_D_eq_0 = 1-norm.cdf(-3.42 - 4.93*np.log(ky) - 0.30*(np.log(ky))**2 - 0.35*Ts*np.log(ky) -\
                                  0.62*Ts + 2.86*np.log(Sa))

    elif 'nf' in gm_type:
        
        ## decision for pga versus Sa(Ts_deg)
        if Ts == 0.0:
            Sa = pga_corr
        else:
            Sa = Sa_Ts_deg
        
        ## check if slope is oriented within 45 degree of fault-normal direction
        if slope_loc <= 45:
            
            ## standard deviation
            sigma = 0.56
            
            ## pgv- and period-dependent coefficients and parameters for D100
            if pgv <= 150:
                if Ts < 0.1:
                    c1 = -6.235
                    c2 = -2.744
                    c3 = 0.0
                    c4 = 1.547
                elif Ts >= 0.1:
                    c1 = -6.462
                    c2 = 1.069
                    c3 = -0.498
                    c4 = 1.547
            if pgv > 150:
                if Ts < 0.1:
                    c1 = 2.480
                    c2 = -2.744
                    c3 = 0.0
                    c4 = -0.097
                elif Ts >= 0.1:
                    c1 = 2.253
                    c2 = 1.069
                    c3 = -0.498
                    c4 = -0.097
            
            ## maximum seismic displacement, D100 (eq. 5)
            lnD = c1 - 2.632*np.log(ky) - 0.278*(np.log(ky))**2 + 0.527*np.log(ky)*np.log(Sa) +\
                  1.978*np.log(Sa) - 0.233*(np.log(Sa))**2 + c2*Ts + c3*Ts**2 + 0.06*M +\
                  c4*np.log(pgv)
            
            ## probability of maximum seismic displacement = 0
            if Ts <= 0.7:
                ## eq. 4(a)
                p_D_eq_0 = 1/(1 + np.exp(-10.787 - 8.717*np.log(ky) + 1.660*np.log(pgv) + 3.150*Ts +
                                         7.560*np.log(Sa)))
            elif Ts > 0.7:
                ## eq. 4(b)
                p_D_eq_0 = 1/(1 + np.exp(-12.771 - 9.979*np.log(ky) + 2.286*np.log(pgv) - 4.965*Ts +
                                         4.817*np.log(Sa)))
            
        elif slope_loc > 45:
            
            ## standard deviation
            sigma = 0.54
            
            ## pgv- and period-dependent coefficients and parameters for D50
            if pgv <= 150:
                if Ts < 0.1:
                    c1 = -7.497
                    c2 = -2.731
                    c3 = 0.0
                    c4 = 1.458
                elif Ts >= 0.1:
                    c1 = -7.718
                    c2 = 1.031
                    c3 = -0.480
                    c4 = 1.458
            if pgv > 150:
                if Ts < 0.1:
                    c1 = 2.480   # -0.148
                    c2 = -2.731
                    c3 = 0.0
                    c4 = 0.025
                elif Ts >= 0.1:
                    c1 = -0.369
                    c2 = 1.031
                    c3 = -0.480
                    c4 = 0.025
                    
            ## median seismic displacement, D50 (eq. 7)
            lnD = c1 - 2.931*np.log(ky) - 0.319*(np.log(ky))**2 + 0.584*np.log(ky)*np.log(Sa) +\
                  2.261*np.log(Sa) - 0.241*(np.log(Sa))**2 + c2*Ts + c3*Ts**2 + 0.05*M +\
                  c4*np.log(pgv)
           
            ## probability of maximum seismic displacement = 0
            if Ts <= 0.7:
                ## eq. 5(a)
                p_D_eq_0 = 1/(1 + np.exp(-14.930 - 10.383*np.log(ky) + 1.971*np.log(pgv) + 3.763*Ts +
                                         8.812*np.log(Sa)))
            elif Ts > 0.7:
                ## eq. 6(b)
                p_D_eq_0 = 1/(1 + np.exp(-14.671 - 10.489*np.log(ky) + 2.222*np.log(pgv) - 4.759*Ts +
                                         5.549*np.log(Sa)))
    
    elif 'full' in gm_type or 'gen' in gm_type:
        
        ## standard deviation
        sigma = 0.74 # 0.736
        
        ## decision for pga versus Sa(Ts_deg)
        if Ts == 0.0:
            Sa = pga_corr
        else:
            Sa = Sa_Ts_deg
        
        ## pgv- and period-dependent coefficients and parameters for D100
        if pgv <= 115:
            if Ts < 0.1:
                a1 = -4.551
                a2 = -9.690 # -9.688
                a3 = 0.0
                a4 = 0.0
                a5 = 0.0
            elif Ts >= 0.1:
                a1 = -5.894
                a2 = 3.152
                a3 = -0.910
                a4 = 0.0
                a5 = 0.0
        if pgv > 115:
            if Ts < 0.1:
                a1 = -4.551
                a2 = -9.690 # -9.688
                a3 = 0.0
                a4 = 1.0
                a5 = -4.75
            elif Ts >= 0.1:
                a1 = -5.894
                a2 = 3.152
                a3 = -0.910
                a4 = 1.0
                a5 = -4.75

        ## seismic displacement, cm (eq. 9)
        lnD = a1 - 2.491*np.log(ky) - 0.245*(np.log(ky))**2 + 0.344*np.log(ky)*np.log(Sa) +\
              2.703*np.log(Sa) - 0.089*(np.log(Sa))**2 + a2*Ts + a3*(Ts)**2 + 0.607*M +\
              a4*np.log(pgv) + a5

        ## probably of displacement = 0 (threshold = 0.5 cm)
        if Ts <= 0.7:
            ## eq. 8(a)
            p_D_eq_0 = 1-norm.cdf(-2.46 - 2.98*np.log(ky) - 0.12*(np.log(ky))**2 - 0.71*Ts*np.log(ky) +\
                                  1.69*Ts + 2.76*np.log(Sa))

        elif Ts > 0.7:
            ## eq. 8(b)
            p_D_eq_0 = 1-norm.cdf(-3.40 - 4.95*np.log(ky) - 0.30*(np.log(ky))**2 - 0.33*Ts*np.log(ky) -\
                                  0.62*Ts + 2.85*np.log(Sa))

    ## other percentiles
    lnD_84 = lnD - sigma
    lnD_16 = lnD + sigma
    
    ## convert from ln to d
    D = np.exp([lnD_84, lnD, lnD_16]) # cm
    
    ## calculate p(D > d) for given d array
    d_arr = kwargs.get('d_arr',None)
    if d_arr is not None:
        for i in d_arr:
            p_d_gt_d = (1-p_D_eq_0) * (1-norm.sf(np.log(i), loc=lnD, scale=sigma)) # eq. 10 and 11
    else:
        p_d_gt_d = None
        
    ##
    return D, p_D_eq_0, p_d_gt_d
	
	
#####################################################################################################################
##### FEMA (2004) HAZUS
#####################################################################################################################
def hazus_2004_land(data_makdisi_seed, pga, M, susc_land):
    """


    """
    ## Load Makdisi & Seed digitized data
    makdisi_seed = pd.read_csv(dir_makdisi_seed, sep='\t')
    makdisi_seed_keys = makdisi_seed.keys()

    ## Critical PGA based on landslide susceptibility
    pga_c = [0.60 if susc_land == 1 else
             0.50 if susc_land == 2 else
             0.40 if susc_land == 3 else
             0.35 if susc_land == 4 else
             0.30 if susc_land == 5 else
             0.25 if susc_land == 6 else
             0.20 if susc_land == 7 else
             0.15 if susc_land == 8 else
             0.10 if susc_land == 9 else
             0.05 if susc_land == 10 else 999]
    pga_c = pga_c[0]
    
    ## PGD for landslide
    n_cyc = 0.3419 * M**3 - 5.5214 * M**2 + 33.6154 * M - 70.7692
    pga_is = pga # default - pga_is = pga
    pgd_pgais_n_upper = np.interp(pga_c/pga_is,makdisi_seed[makdisi_seed_keys[0]],makdisi_seed[makdisi_seed_keys[1]]) 
    pgd_pgais_n_lower = np.interp(pga_c/pga_is,makdisi_seed[makdisi_seed_keys[2]],makdisi_seed[makdisi_seed_keys[3]])
    pgd_pgais_n = (pgd_pgais_n_upper + pgd_pgais_n_lower)/2
    pgd_land = pgd_pgais_n * pga_is * n_cyc * globals()['cm']/globals()['inch']