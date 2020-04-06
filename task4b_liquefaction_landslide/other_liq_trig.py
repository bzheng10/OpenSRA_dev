#####################################################################################################################
##### Higher level methods for liquefaction triggering
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Zhu et al. (2017) An Updated Geospatial Liquefaction Model for Global Application
#####################################################################################################################
def zhu_etal_2017(pgv, vs30, precip, dc, dr, dw, wtd, m):
    
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

    sf = 1/(1+np.exp(-2*(m-6))) # ad-hoc magnitude scaling factor added by USGS, to be applied to pgv
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
        X_prob_liq = b_0 + b_lnpgv*np.log(pgv*sf) + b_vs30*np.log(vs30) + b_precip*precip + \
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
        X_prob_liq = b_0 + b_lnpgv*np.log(pgv*sf) + b_vs30*np.log(vs30) + b_precip*precip + b_dw*dw + b_wtd*wtd

        ## coeffcients for areal liquefaction
        a = 49.15
        b = 42.40
        c = 9.165
    
    ## probability of liquefacion
    if pgv < 3 or vs30 > 620:
        prob_liq = 0
    else:
        prob_liq = 1 / (1 + np.exp(-X_prob_liq))

    ## areal liquefaction percent
    prob_liq_areal = a / (1 + b * np.exp(-c * prob_liq))**2

    ## liquefaction susceptibility 
    liq_susc = X_prob_liq - b_lnpgv*np.log(pgv*sf)

	## if on water (distance to coast < 0)
    if dc < 0:
	
		## set outputs to be invalid
        prob_liq = -999
        prob_liq_areal = -999
        liq_susc = -999

	##
    return prob_liq, prob_liq_areal, liq_susc