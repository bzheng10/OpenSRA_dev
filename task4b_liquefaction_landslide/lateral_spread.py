#####################################################################################################################
##### Methods for lateral spreading
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
    
    
    ## 1) Rock-slope failures:
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
    
    elif flag_failure == 2:
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
        
    elif flag_failure == 3:
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

    elif flag_failure == 4:
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

    return D, ky
	

#####################################################################################################################
##### Youd et al. (2002) Revised multilinear regression equations for prediction of lateral spread displacement
#####################################################################################################################
def youd_etal_2002(**kwargs):

    ##### Regression coefficients for empirical lateral spread model
    ## Model inputs:
    ## - M = moment magnitude
    ## - R = closest distance from site to source (km)
    ## - W = free-face ratio (height and/or horizontal distance from site to toe) (%)
    ## - S = ground slope
    ## - T_15 = cumulative thickness (in upper 20 m) of all saturated soil layers susceptible to liquefaction initiation with N1_60
    ## - F_15 = average fines content of the soil comprising T_15 (%)
    ## - D50_15 = average mean grain size of the soil comprising T_15 (mm)
    ##
    ## Model output:
    ## - Dh = median computed permanent lateral spread displacement (m)
    ##
    ## flag_model == 1: ground slope (infinite slope)
    ## flag_model == 2: free face
    ##
    ## Ground slope
    if flag_model == 1:
        b0 = -16.213
        b4 = 0
        b5 = 0.338
    elif flag_model == 2:
        b0 = -16.713
        b4 = 0.592
        b5 = 0
        
	## model params
    b1 = 1.532
    b2 = -1.406
    b3 = -0.012
    b6 = 0.540
    b7 = 3.413
    b8 = -0.795
    
	## adjusted distance
    R_star = R + 10**(0.89*M-5.64)
    
	## standard deviation
    sigma_log_Dh = 0.197
	sigma_ln_Dh = sigma_log_Dh*np.log(10)
    
	## calcualte ln(D)
    log_Dh = b0 + b1*M + b2*np.log10(R_star) + b3*R + b4*np.log10(W) + \
             b4*np.log10(S) + b6*np.log(T_15) + b7*np.log10(100-F_15) + b8*np.log10(D50_15+0.1)
    
	## calculate D
    Dh = 10**log_Dh
    
    ##
    return Dh, sigma_ln_Dh