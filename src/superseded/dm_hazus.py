#####################################################################################################################
##### Buried pipeline damage measure assessment by Hazus
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### FEMA (2004) HAZUS
#####################################################################################################################
def assess_pipe_Hazus(pga, pgv, M, d_w, susc_liq, susc_land, pipe_type='brittle', l_seg=1):
    """
    Following the Hazus (FEMA, 2003) methodology

    Parameters
    ----------
    Peak ground acceleration, pga
    Moment magnitude, M
    Depth to ground water, d_w
    Susceptibility category for LIQUEFACTION, susc_liq = none, very low, low, moderate, high, very high
    Susceptibility category for LANDSLIDE, susc_land = 0 to 10
    Pipe type, pipe_type = brittle or ductile

    Returns
    -------
    Inferred PGV, pgv_in
    Liquefaction likelihood, p_liq
    Peak ground displacement for lateral spread, pgd_ls
    Ground settlement, pgd_gs
    Peak ground displacement for landslide, pgd_land
    Maximum displacement for surface fault rupture, pgd_sfr
    Repair_rate by PGV, rr_pgv
    Repair rate by PGD, rr_pgd

    """
    ## Initialize dictionary
    dm_out = {}
    
    ## Load Makdisi & Seed digitized data
    file_makdisi_seed = 'makdisi_seed.txt'
    makdisi_seed = pd.read_csv(file_makdisi_seed, sep='\t')
    makdisi_seed_keys = makdisi_seed.keys()

    ## conjugate distribution for repair rates
    ln_rr_mean = np.log(0.1)
    ln_rr_beta = 0.85
    
    ## Correlations based on liquefaction susceptibility
    if susc_liq.lower() == 'very high':
        p_liq_pga = np.maximum(np.minimum(9.09*pga-0.82,1),0)
        p_ml = 0.25
        pga_t = 0.09 # g
        pgd_gs = 12 # inches
    elif susc_liq.lower() == 'high':
        p_liq_pga = np.maximum(np.minimum(7.67*pga-0.92,1),0)
        p_ml = 0.20
        pga_t = 0.12 # g
        pgd_gs = 6 # inches
    elif susc_liq.lower() == 'moderate':
        p_liq_pga = np.maximum(np.minimum(6.67*pga-1.00,1),0)
        p_ml = 0.10
        pga_t = 0.15 # g
        pgd_gs = 2 # inches
    elif susc_liq.lower() == 'low':
        p_liq_pga = np.maximum(np.minimum(5.57*pga-1.18,1),0)
        p_ml = 0.05
        pga_t = 0.21 # g
        pgd_gs = 1 # inches
    elif susc_liq.lower() == 'very low':
        p_liq_pga = np.maximum(np.minimum(4.16*pga-1.08,1),0)
        p_ml = 0.02
        pga_t = 0.26 # g
        pgd_gs = 0 # inches
    elif susc_liq.lower() == 'none':
        p_liq_pga = np.ones(len(pga))*0.00
        p_ml = 0.00
        pga_t = 999. # g
        pgd_gs = 0 # inches
    else:
        p_liq_pga = np.ones(len(pga))*np.nan
        p_ml = np.nan
        pga_t = np.nan
        pgd_gs = np.nan

    ## Liquefaction likelihood, p_liq    
    k_m = 0.0027 * M**3 - 0.0267 * M**2 - 0.2055 * M + 2.9188
    k_w = 0.022 * d_w + 0.93
    p_liq = p_liq_pga / k_m / k_w * p_ml
    
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

    ## Peak ground deformations (pgd):    
    ## PGD for lateral spread
    k_delta = 0.0086 * M**3 - 0.0914 * M**2 + 0.4698 * M - 0.9835
    pgd_pga_pgat = np.asarray([12*max(j/pga_t,1)-12 if j/pga_t<=2
                              else 18*j/pga_t-24 if j/pga_t>2 and j/pga_t<=3
                              else 70*min(j/pga_t,4)-180
                              for j in pga])  # inches
    pgd_ls = k_delta*pgd_pga_pgat # inches
    
    ## PGD for ground settlement (Tokimatsu and Seed, 1987)
    pgd_gs = np.ones(len(pga))*pgd_gs # see decision tree above
    
    ## Average PGD due to liquefaction-induced deformation
    pgd_liq = np.sqrt(pgd_ls*pgd_gs)
    
    ## PGD for landslide
    n_cyc = 0.3419 * M**3 - 5.5214 * M**2 + 33.6154 * M - 70.7692
    pga_is = pga # default - pga_is = pga
    pgd_pgais_n_upper = np.interp(pga_c/pga_is,makdisi_seed[makdisi_seed_keys[0]],makdisi_seed[makdisi_seed_keys[1]]) 
    pgd_pgais_n_lower = np.interp(pga_c/pga_is,makdisi_seed[makdisi_seed_keys[2]],makdisi_seed[makdisi_seed_keys[3]])
    pgd_pgais_n = (pgd_pgais_n_upper + pgd_pgais_n_lower)/2
    pgd_land = pgd_pgais_n * pga_is * n_cyc * globals()['cm']/globals()['inch'] # 

    ## PGD for surface fault rupture
    pgd_sfr = np.ones(len(pga))* 10**(-5.26 + 0.79*M) * globals()['M']/globals()['inch']
    
    ## correction factor for ductility of pipes
    rr_multi = [0.3 if pipe_type == 'ductile' else 1]

    ## Repair rates by PGV and PGD
    rr_pgv = 0.0001 * pgv**2.25 * rr_multi # PGV in cm/s, repair rate in repairs/km
    rr_pgd_liq = p_liq * pgd_liq**0.56 * rr_multi # PGD in inches, repair rate in repairs/km, using only liquefaction-induced deformation for now
    rr_pgd = rr_pgd_liq # for now set repair rate by PGD to rr_pgd_liq, since the empirical formula for rr_pgd is a function of the
                        # probability of liquefaction and shouldn't be associated with landslide and surface fault rupture
    
	## break rr_pgv and rr_pgd into rr_leak and rr_break and combine
    rr_leak = rr_pgv*0.8 + rr_pgd*0.2
    rr_break = rr_pgv*0.2 + rr_pgd*0.8
    
	## number of breaks for segment
    n_pgv = rr_pgv*l_seg # by pgv
    n_pgd = rr_pgd*l_seg # by pgd
    n_leak = rr_leak*l_seg # for leaks
    n_break = rr_break*l_seg # for breaks
    
	##
    return rr_pgv, rr_pgd, rr_leak, rr_break, n_pgv, n_pgd, n_leak, n_break