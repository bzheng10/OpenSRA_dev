#####################################################################################################################
##### Functions for damage measures given intensity measures and engineering demand parameters
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### sort list where each index has multiple elements, sort by column index
##### FEMA (2004) HAZUS
#####################################################################################################################
def hazus_2004(pgv, pgd_ls, pgd_gs, pipe_type='brittle', l_seg=1):
	
    ## Average PGD due to liquefaction-induced deformation
    pgd_liq = np.sqrt(pgd_ls*pgd_gs)
	
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
    # n_pgv = rr_pgv*l_seg # by pgv
    # n_pgd = rr_pgd*l_seg # by pgd
    # n_leak = rr_leak*l_seg # for leaks
    # n_break = rr_break*l_seg # for breaks
    
	##
    return rr_pgv, rr_pgd, rr_leak, rr_break