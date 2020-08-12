#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### Methods for lateral spreading
#####
##### Created: April 13, 2020
##### Updated: July 14, 2020
##### --- improved documentation
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Required packages
import numpy as np
from scipy import sparse
#####################################################################################################################


#####################################################################################################################
##### Grant et al. (2016) Multimodal method for coseismic landslide hazard assessment
#####################################################################################################################
def grant_etal_2016_ls(**kwargs):
    """
    Compute rock and soil landslide displacement using following the Grant et al. (2016) deterministic procedure. Of the 40 coseismic landslide datasets described in Keefer (1984), Grant et al. (2016) selected the four most fundamental modes of failures (see Table 1 in Grant et al., 2016, for detailed descriptions):

    1. Rock-slope failures: wedge geometry, slope range = 35-90 degrees, typically tens of meters to kilometers
    2. Disrupted soil slides: infinite slope, slope range = 15-50 degrees, typically meters to a hundred meters
    3. Coherent rotational slides: circular-rotation, slope range = 20-35 degrees, typically under 2 meters
    4. Lateral spreads: empirirically-developed geometry, slope range = 0-6 degrees, typically under 2 meters

    Note that:\n
    - The regression models for **failure modes 1-3** are coded separately; see :func: `edp.ls.grant_etal_2016_ls`.

    Parameters
    ----------
    pga : float
        [g] peak ground acceleration
    liq_susc : str
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)
    M : float
        moment magnitude
    flag_extrap_Epgd : boolean, optional
        True to extrapolate the expected PGD beyond upper limit of Figure 4.9 in HAZUS (FEMA, 2014); default = False
    z : float, optional
        [m] elevation; site is susceptible to lateral spreads if z < z_cutoff; default = 1 m (i.e., susceptible everywhere)
    dw : float, optional
        [m] distance to nearest water body (coast+river); site is susceptible to lateral spreads if 0 < dw < dw_cutoff; default = 1 m (i.e., susceptible everywhere)
    z_cutoff : float, optional
        [m] cutoff elevation for which a site is susceptible to lateral spreads; default = 25 m (used in Grant et al., 2016)
    dw_cutoff : float, optional
        [m] cutoff distance to nearest water body for which a site is susceptible to lateral spreads; default = 25 m (25 m for rivers as used in Grant et al., 2016)

    Returns
    -------
    pgd_ls : float
        [cm] permanent ground deformation, lateral spreading

    References
    ----------
    .. [1] Grant, A., Wartman, J., and Abou-Jaoude, G., 2016, Multimodal Method for Coseismic Landslide Hazard Assessment, Engineering Geology, vol. 212, pp. 146-160.
    .. [2] Keefer, D.K., 1984., Landslides Caused by Earthquakes, Geological Society of America Bulletin, vol. 95, no. 4, pp. 406-421.
    .. [3] Wills, C.J., Perez, F.G., and Gutierrez, C.I., 2011, Susceptibility to Deep-Seated Landslides in California. California Geological Survey Map Sheet no. 58.

    """

    ###############################
    ## 4) Lateral spreads:
    ###############################

    ## get inputs
    pga = kwargs.get('pga',None) # g, peak ground acceleration
    liq_susc = kwargs.get('liq_susc',None) # liquefaction susceptibility category
    M = kwargs.get('M',None) # moment magnitude
    z = kwargs.get('z',1) # m, elevation, default to 0 such that pgd_ls > 0
    dw = kwargs.get('dw',1) # m, distance to water body, default to 0 such that pgd_ls > 0
    z_cutoff = kwargs.get('z_cutoff',25) # m, default to 25 m as used in Grant et al. (2016)
    dw_cutoff = kwargs.get('dw_cutoff',25) # m, default to 25 m as used in Grant et al. (2016)
    n_samp_im = kwargs.get('n_samp_im',1) # number of samples, default to 1
    flag_extrap_Epgd = kwargs.get('flag_extrap_Epgd',False) # True to extrapolate Epgd in Figure 4.9 of HAZUS

    ## number of sites
    n_site = pga[0].shape[1]
    n_rup = pga[0].shape[0]

    ## magnitude correction
    Kdelta = 0.0086*M**3 - 0.0914*M**2 + 0.4698*M - 0.9835
    # Kdelta = np.repeat(Kdelta[:,np.newaxis],n_site,axis=1)

    ## get threshold pga against liquefaction
    # pga_t = np.ones(liq_susc.shape)*np.nan
    # pga_t[liq_susc=='very high'] = 0.09 # g
    # pga_t[liq_susc=='high'] = 0.12 # g
    # pga_t[liq_susc=='moderate'] = 0.15 # g
    # pga_t[liq_susc=='low'] = 0.21 # g
    # pga_t[liq_susc=='very low'] = 0.26 # g
    # pga_t[liq_susc=='none'] = 999. # g

    ## preset list
    pgd_ls = {}

    ## loop through all realizations
    for k in range(n_samp_im):

        ## get non-zero IM of all sites for current sample
        # pga_k = pga[k].toarray()
        pga_k = pga[k].data
        row_k = pga[k].row
        col_k = pga[k].col


        ## get threshold pga against liquefaction for current sample
        liq_susc_k = liq_susc[col_k]
        pga_t_k = np.ones(liq_susc_k.shape)*np.nan
        pga_t_k[liq_susc_k=='very high'] = 0.09 # g
        pga_t_k[liq_susc_k=='high'] = 0.12 # g
        pga_t_k[liq_susc_k=='moderate'] = 0.15 # g
        pga_t_k[liq_susc_k=='low'] = 0.21 # g
        pga_t_k[liq_susc_k=='very low'] = 0.26 # g
        pga_t_k[liq_susc_k=='none'] = 999. # g

        ## repeat for all ruptures
        # pga_t = np.transpose(np.repeat(pga_t[:,np.newaxis],n_rup,axis=1))
        # z = np.transpose(np.repeat(z[:,np.newaxis],n_rup,axis=1))
        # dw = np.transpose(np.repeat(dw[:,np.newaxis],n_rup,axis=1))

        ## generate array for current sample
        Kdelta_k = Kdelta[row_k]
        z_k = z[col_k]
        dw_k = dw[col_k]

        ## normalized stress, opportunity for liquefaction
        # r = np.divide(pga_k,pga_t)
        r_k = np.divide(pga_k,pga_t_k)

        ## get normalized displacement, a, for M=7
        # a = np.ones(pga_k.shape)*np.nan
        # a[r<=1] = 0
        # a[np.logical_and(r>1,r<=2)] = 12*r[np.logical_and(r>1,r<=2)] - 12
        # a[np.logical_and(r>2,r<=3)] = 18*r[np.logical_and(r>2,r<=3)] - 24
        # if flag_extrap_Epgd is True:
            # a[r>3] = 70*r[r>3] - 180
        # else:
            # a[np.logical_and(r>3,r<=4)] = 70*r[np.logical_and(r>3,r<=4)] - 180
            # a[r>4] = 100
        a_k = np.ones(pga_k.shape)*np.nan
        a_k[r_k<=1] = 0
        a_k[np.logical_and(r_k>1,r_k<=2)] = 12*r_k[np.logical_and(r_k>1,r_k<=2)] - 12
        a_k[np.logical_and(r_k>2,r_k<=3)] = 18*r_k[np.logical_and(r_k>2,r_k<=3)] - 24
        if flag_extrap_Epgd is True:
            a_k[r_k>3] = 70*r_k[r_k>3] - 180
        else:
            a_k[np.logical_and(r_k>3,r_k<=4)] = 70*r_k[np.logical_and(r_k>3,r_k<=4)] - 180
            a_k[r_k>4] = 100

        ## susceptibility to lateral spreading only for low-lying soils (z < z_cutoff) or deposits found near river (dw < dw_cutoff)
        # pgd_ls_k = np.multiply(Kdelta,a)
        # pgd_ls_k[z>=z_cutoff] = 0
        # pgd_ls_k[np.logical_or(dw<=0,dw>=dw_cutoff)] = 0
        pgd_ls_k = np.multiply(Kdelta_k,a_k)
        pgd_ls_k[z_k>=z_cutoff] = 0
        pgd_ls_k[np.logical_or(dw_k<=0,dw_k>=dw_cutoff)] = 0

        ## append sims
        ind_red = np.where(pgd_ls_k>0)[0]
        pgd_ls_k_red = pgd_ls_k[ind_red]
        row_k_red = row_k[ind_red]
        col_k_red = col_k[ind_red]
        pgd_ls.update({k:sparse.coo_matrix((pgd_ls_k_red,(row_k_red,col_k_red)),shape=(n_rup,n_site))})

    ## set to numpy arrays
    # pgd_ls = np.asarray(pgd_ls)

    ## probability distribution
    prob_dist_type = 'uniform'
    factor_aleatory = 3
    factor_epistemic = 4

    ## store outputs
    output = {}
    output.update({'pgd_ls': pgd_ls})
    output.update({'prob_dist': {'type': prob_dist_type,
                                'factor_aleatory': factor_aleatory,
                                'factor_epistemic': factor_epistemic}})

    ##
    return output


#####################################################################################################################
##### Youd et al. (2002) Revised multilinear regression equations for prediction of lateral spread displacement
#####################################################################################################################
def youd_etal_2002(M, R, W, S, T_15, F_15, D50_15, flag_model=1):
    """
    Text

    """
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


#####################################################################################################################
##### FEMA (2014) HAZUS
#####################################################################################################################
def hazus_2014_ls(**kwargs):
    """
    Compute lateral spreading, which is the the same procedure used by Grant et al. (2016). See and use the function :func:`edp.ls.grant_etal_2016_ls`.

    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.

    """

    output = grant_etal_2016_ls(**kwargs)

    return output