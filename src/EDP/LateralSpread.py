# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for lateral spread
#
# Created: April 13, 2020
# Updated: July 14, 2020
# --- improved documentation
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import numpy as np
from scipy import sparse


# -----------------------------------------------------------
def IdrissBoulanger2008(FS_liq, N1_60_cs=None, qc_1N_cs=None):
    """
    Compute maximum shear strain following Idriss & Boulanger (2008); must specify either **N1_60_cs** or **qc_1N_cs**.

    Parameters
    ----------
    FS_liq : float
        factor of safety against liquefaction triggering
    N1_60_cs : float, optional
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
    qc_1N_cs : float, optional
        CPT tip resistance corrected for overburden and fines content

    Returns
    -------
    gamma_max : float
        [%] maximum shear strain
    Dr : float, optional
        [%] relative density

    References
    ----------
    .. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.

    """
    
    if N1_60_cs is None and qc_1N_cs is None:
        #
        logging.info(f"Must enter either N1_60_cs or qc_1N_cs to proceed with method")
        
        #
        return None
        
    else:
        # Check if param is an array and convert if not
        if not isinstance(FS_liq,np.ndarray):
            FS_liq = np.asarray([FS_liq])
            
        # based on type of penetration resistance
        if N1_60_cs is not None:
            # Check if param is an array and convert if not
            if not isinstance(N1_60_cs,np.ndarray):
                N1_60_cs = np.asarray([N1_60_cs])
                
            # Estimate relative density, Dr
            Dr = (min(N1_60_cs,46)/46)**(0.5) # limit N1_60_cs under 46, Idriss & Boulanger (2008)
            # F_alpha = 0.032 + 0.69*max(N1_60_cs,7)**0.5 - 0.13*max(N1_60_cs,7) # limit Dr a minimum of 0.4 when calculating F_alpha, Idriss & Boulanger (2008)
        
        #
        elif qc_1N_cs is not None:
            # Check if param is an array and convert if not
            if not isinstance(qc_1N_cs,np.ndarray):
                qc_1N_cs = np.asarray([qc_1N_cs])
                
            # Estimate relative density, Dr
            Dr = 0.478*(max(min(qc_1N_cs,254),21))**0.264 - 1.063 # limit qc_1N_cs to 21 to 254, Idriss & Boulanger (2008)
            # F_alpha = 0.032 + 4.7*max(Dr,0.4) - 6.0*max(Dr,0.4)**2 # limit Dr a minimum of 0.4 when calculating F_alpha, Idriss & Boulanger (2008)

        # Estimate limiting factor of safety, F_alpha
        F_alpha = 0.032 + 4.7*max(Dr,0.4) - 6.0*max(Dr,0.4)**2 # limit Dr a minimum of 0.4 when calculating F_alpha, Idriss & Boulanger (2008)

        # Limiting shear strain, gamma_max = gamma_lim when FS_liq <= F_alpha
        gamma_lim = max(1.859*(1.1 - Dr)**3,0)

        # Calculate maximum shear strain, interpolate between FS=F_alpha and FS=2, and shear strains between 0 and gamma_lim
        if FS_liq >= 2:
            gamma_max = 0
        elif FS_liq <= F_alpha:
            gamma_max = gamma_lim
        else:
            gamma_max = min(gamma_lim, 0.035*(2-FS_liq)*(1-F_alpha)/(FS_liq-F_alpha))

        # Convert to %
        gamma_max = gamma_max * 100
        Dr = Dr * 100
        
        #
        return gamma_max, Dr


# -----------------------------------------------------------
def ZhangEtal2004(FS_liq, N1_60_cs=None, qc_1N_cs=None):
    """
    Compute volumetric strain following Zhang et al. (2004); must specify either **N1_60_cs** or **qc_1N_cs**.

    Parameters
    ----------
    FS_liq : float
        factor of safety against liquefaction triggering
    N1_60_cs : float, optional
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
    qc_1N_cs : float, optional
        CPT tip resistance corrected for overburden and fines content

    Returns
    -------
    gamma_max : float
        [%] maximum shear strain

    References
    ----------
    .. [1] Zhang, G., Robertson, P.K., and Brachman, R.W.I., 2004, Estimating Liquefaction-Induced Lateral Displacements Using the Standard Penetration Test or Cone Penetration Test, Journal of Geotechnical or Geoenvironmental Engineering, vol. 130, no. 8, pp. 861-871.

    """

    #
    if N1_60_cs is None and qc_1N_cs is None:
        #
        logging.info(f"Must enter either N1_60_cs or qc_1N_cs to proceed with method")
        
        #
        return None
        
    else:
        # Check if param is an array and convert if not
        if not isinstance(FS_liq,np.ndarray):
            FS_liq = np.asarray([FS_liq])
            
        # based on type of penetration resistance
        if N1_60_cs is not None:
            # Check if param is an array and convert if not
            if not isinstance(N1_60_cs,np.ndarray):
                N1_60_cs = np.asarray([N1_60_cs])
                
            # Estimate relative density, Dr, 
            Dr = 14*(np.minimum(N1_60_cs,42))**(0.5) # %, Eq. 1 in Zhang et al. (2004), note that for N1_60_cs=42, Dr=90.73% (SPT upper limit)
        
        #
        elif qc_1N_cs is not None:
            # Check if param is an array and convert if not
            if not isinstance(qc_1N_cs,np.ndarray):
                qc_1N_cs = np.asarray([qc_1N_cs])
                
            # Estimate relative density, Dr
            Dr = -85 + 76*np.log10(np.minimum(qc_1N_cs,200)) # %, Eq. 2 in Zhang et al. (2004), note that qc_1N_cs=200, Dr=89.88% (CPT upper limit)
        
        # Since relationships in Zhang et al. (2004) for discrete Dr, first compute strain at bounds, then interpolate
        # 6 discrete Dr values where relationships are given, 7 bins total (e.g., bin 1 = Dr <= 40%, bin 7 = Dr >= 90%):
        Dr_bin_range = np.array([40, 50, 60, 70, 80, 90, 100])
        
        # Bound applicable range for FS_liq and Dr
        FS_liq = np.minimum(FS_liq,2)
        Dr = np.round(np.maximum(np.minimum(Dr,100),40),decimals=2)
        
        # Determine FS_liq upper and lower bin values
        bin_arr = np.digitize(Dr,Dr_bin_range)
        
        bin_arr_high = bin_arr
        bin_arr_high[bin_arr_high>len(Dr_bin_range)-1] = len(Dr_bin_range)-1 # upper limit = number of bin range
        bin_arr_low = bin_arr-1
        bin_arr_low[bin_arr_low<0] = 0 # lower limit = 0
        
        Dr_high = Dr_bin_range[bin_arr_high]
        Dr_low = Dr_bin_range[bin_arr_low]
        
        # Zhang et al. (2004) provides relationships for maximum shear strain generally in the form of:
        # --- gamma_max = min(gamma_lim, c1 * (FS) ^ c2), where gamma_lim is the limiting shear strain, and gamma_lim, c1, and c2 are dependent on FS and Dr
        # There is a unique case where gamma_max = c3*(1-FS) + c4; this case is addressed separately later
        # Determine coeffcients c1 and c2 at the Dr upper and lower bounds
        # Start with Dr == 100, no volumetric strain
        # --- note that for:
        ##------------------- 1. CPT: qc_1N_cs<=200, Dr<=89.88%
        ##------------------- 2. SPT: N1_60_cs<=42, Dr<=90.73%
        c1_high = np.zeros(Dr.shape)
        c2_high = np.zeros(Dr.shape)
        c1_low = np.zeros(Dr.shape)
        c2_low = np.zeros(Dr.shape)
        gamma_lim_high = np.zeros(Dr.shape)
        gamma_lim_low = np.zeros(Dr.shape)
        
        # Dr >= 90 and < 100
        c1_low[Dr < 100] = 3.26
        c2_low[Dr < 100] = -1.80
        gamma_lim_low[Dr < 100] = 6.2
        
        # Dr >= 80 and < 90
        c1_high[Dr < 90] = 3.26
        c2_high[Dr < 90] = -1.80
        c1_low[Dr < 90] = 3.22
        c2_low[Dr < 90] = -2.08
        gamma_lim_high[Dr < 90] = 6.2
        gamma_lim_low[Dr < 90] = 10
        
        # Dr >= 70 and < 80
        c1_high[Dr < 80] = 3.22
        c2_high[Dr < 80] = -2.08
        c1_low[Dr < 80] = 3.20
        c2_low[Dr < 80] = -2.89
        gamma_lim_high[Dr < 80] = 10
        gamma_lim_low[Dr < 80] = 14.5
        
        # Dr >= 60 and < 70
        c1_high[Dr < 70] = 3.20
        c2_high[Dr < 70] = -2.89
        c1_low[Dr < 70] = 3.58
        c2_low[Dr < 70] = -4.42
        gamma_lim_high[Dr < 70] = 14.5
        gamma_lim_low[Dr < 70] = 22.7
        
        # Dr >= 50 and < 60
        c1_high[Dr < 60] = 3.58
        c2_high[Dr < 60] = -4.42
        c1_low[Dr < 60] = 4.22
        c2_low[Dr < 60] = -6.39
        gamma_lim_high[Dr < 60] = 22.7
        gamma_lim_low[Dr < 60] = 34.1

        # Dr > 40 and < 50
        c1_high[Dr < 50] = 4.22
        c2_high[Dr < 50] = -6.39
        c1_low[Dr < 50] = 3.31
        c2_low[Dr < 50] = -7.97
        gamma_lim_high[Dr < 50] = 34.1
        gamma_lim_low[Dr < 50] = 51.2
        
        # Dr == 40
        c1_high[Dr == 40] = 3.31
        c2_high[Dr == 40] = -7.97
        gamma_lim_high[Dr == 40] = 51.2
        
        # Compute maximum shear strain strain at bounds and apply limiting strain
        gamma_max_high = np.minimum(c1_high*(FS_liq**c2_high),gamma_lim_high)
        gamma_max_low = np.minimum(c1_low*(FS_liq**c2_low),gamma_lim_low)
        
        # Apply unique case when Dr = 40% and 0.81 <= FS <= 1.0
        logic_1 = np.logical_and(Dr==40,FS_liq<=1)
        gamma_max_high[logic_1] = np.minimum(250*(1-FS_liq[logic_1]) + 3.5, 51.2)
        logic_2 = np.logical_and(Dr<50,FS_liq<=1)
        gamma_max_low[logic_2] = np.minimum(250*(1-FS_liq[logic_2]) + 3.5, 51.2)
        
        # Interpolate for volumetric strain at FS_liq
        gamma_max = gamma_max_low + (gamma_max_high - gamma_max_low)*(Dr - Dr_low)/(Dr_high - Dr_low)
        
        #
        return gamma_max


# -----------------------------------------------------------
def GrantEtal2016(**kwargs):
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
    
    # 4) Lateral spreads:

    # get inputs
    pga = kwargs.get('pga',None) # g, peak ground acceleration
    n_site = kwargs.get('n_site',pga[0].shape[1]) # number of sites
    n_event = kwargs.get('n_event',pga[0].shape[0]) # number of events
    liq_susc = kwargs.get('liq_susc',None) # liquefaction susceptibility category
    M = kwargs.get('M',None) # moment magnitude
    z = kwargs.get('z',np.ones(n_site)) # m, elevation, default to 0 such that pgd_ls > 0
    dw = kwargs.get('DistanceToWaterBody',np.ones(n_site)/1000)*1000 # m, distance to water body, default to 0 such that pgd_ls > 0
    z_cutoff = kwargs.get('z_cutoff',25) # m, default to 25 m as used in Grant et al. (2016)
    dw_cutoff = kwargs.get('DistanceToWaterCutoff',25) # m, default to 25 m as used in Grant et al. (2016)
    n_sample = kwargs.get('n_sample',1) # number of samples, default to 1
    flag_extrap_Epgd = kwargs.get('flag_extrap_Epgd',False) # True to extrapolate Epgd in Figure 4.9 of HAZUS

    # magnitude correction
    Kdelta = 0.0086*M**3 - 0.0914*M**2 + 0.4698*M - 0.9835
    # Kdelta = np.repeat(Kdelta[:,np.newaxis],n_site,axis=1)

    # get threshold pga against liquefaction
    # pga_t = np.ones(liq_susc.shape)*np.nan
    # pga_t[liq_susc=='very high'] = 0.09 # g
    # pga_t[liq_susc=='high'] = 0.12 # g
    # pga_t[liq_susc=='moderate'] = 0.15 # g
    # pga_t[liq_susc=='low'] = 0.21 # g
    # pga_t[liq_susc=='very low'] = 0.26 # g
    # pga_t[liq_susc=='none'] = 999. # g

    # preset list
    pgd_ls = {}

    # loop through all realizations
    for k in range(n_sample):

        # get non-zero IM of all sites for current sample
        # pga_k = pga[k].toarray()
        pga_k = pga[k].data
        row_k = pga[k].row
        col_k = pga[k].col

        # get threshold pga against liquefaction for current sample
        liq_susc_k = liq_susc[col_k]
        pga_t_k = np.ones(liq_susc_k.shape)*np.nan
        pga_t_k[liq_susc_k=='very high'] = 0.09 # g
        pga_t_k[liq_susc_k=='high'] = 0.12 # g
        pga_t_k[liq_susc_k=='moderate'] = 0.15 # g
        pga_t_k[liq_susc_k=='low'] = 0.21 # g
        pga_t_k[liq_susc_k=='very low'] = 0.26 # g
        pga_t_k[liq_susc_k=='none'] = 999. # g

        # repeat for all ruptures
        # pga_t = np.transpose(np.repeat(pga_t[:,np.newaxis],n_event,axis=1))
        # z = np.transpose(np.repeat(z[:,np.newaxis],n_event,axis=1))
        # dw = np.transpose(np.repeat(dw[:,np.newaxis],n_event,axis=1))

        # generate array for current sample
        Kdelta_k = Kdelta[row_k]
        z_k = z[col_k]
        dw_k = dw[col_k]

        # normalized stress, opportunity for liquefaction
        # r = np.divide(pga_k,pga_t)
        r_k = np.divide(pga_k,pga_t_k)

        # get normalized displacement, a, for M=7
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

        # susceptibility to lateral spreading only for low-lying soils (z < z_cutoff) or deposits found near river (dw < dw_cutoff)
        # pgd_ls_k = np.multiply(Kdelta,a)
        # pgd_ls_k[z>=z_cutoff] = 0
        # pgd_ls_k[np.logical_or(dw<=0,dw>=dw_cutoff)] = 0
        pgd_ls_k = np.multiply(Kdelta_k,a_k)
        pgd_ls_k[z_k>=z_cutoff] = 0
        pgd_ls_k[np.logical_or(dw_k<=0,dw_k>=dw_cutoff)] = 0

        # append sims
        ind_red = np.where(pgd_ls_k>0)[0]
        pgd_ls_k_red = pgd_ls_k[ind_red]
        row_k_red = row_k[ind_red]
        col_k_red = col_k[ind_red]
        pgd_ls.update({k:sparse.coo_matrix((pgd_ls_k_red,(row_k_red,col_k_red)),shape=(n_event,n_site))})

    # set to numpy arrays
    # pgd_ls = np.asarray(pgd_ls)

    # probability distribution
    prob_dist_type = 'Uniform'
    factor_aleatory = 3
    factor_epistemic = 4

    # store outputs
    output = {}
    output.update({'pgd_ls': pgd_ls})
    output.update({'prob_dist': {'type': prob_dist_type,
                                'factor_aleatory': factor_aleatory,
                                'factor_epistemic': factor_epistemic}})

    #
    return output


# -----------------------------------------------------------
def YoudEtal2002(M, R, W, S, T_15, F_15, D50_15, flag_model=1):
    """
    Revised multilinear regression equations for prediction of lateral spread displacement (Youd et al., 2002)

    """
    # Regression coefficients for empirical lateral spread model
    # Model inputs:
    # - M = moment magnitude
    # - R = closest distance from site to source (km)
    # - W = free-face ratio (height and/or horizontal distance from site to toe) (%)
    # - S = ground slope
    # - T_15 = cumulative thickness (in upper 20 m) of all saturated soil layers susceptible to liquefaction initiation with N1_60
    # - F_15 = average fines content of the soil comprising T_15 (%)
    # - D50_15 = average mean grain size of the soil comprising T_15 (mm)
    #
    # Model output:
    # - Dh = median computed permanent lateral spread displacement (m)
    #
    # flag_model == 1: ground slope (infinite slope)
    # flag_model == 2: free face

    # Ground slope
    if flag_model == 1:
        b0 = -16.213
        b4 = 0
        b5 = 0.338
    elif flag_model == 2:
        b0 = -16.713
        b4 = 0.592
        b5 = 0

    # model params
    b1 = 1.532
    b2 = -1.406
    b3 = -0.012
    b6 = 0.540
    b7 = 3.413
    b8 = -0.795

    # adjusted distance
    R_star = R + 10**(0.89*M-5.64)

    # standard deviation
    sigma_log_Dh = 0.197
    sigma_ln_Dh = sigma_log_Dh*np.log(10)

    # calcualte ln(D)
    log_Dh = b0 + b1*M + b2*np.log10(R_star) + b3*R + b4*np.log10(W) + \
            b4*np.log10(S) + b6*np.log(T_15) + b7*np.log10(100-F_15) + b8*np.log10(D50_15+0.1)

    # calculate D
    Dh = 10**log_Dh

    #
    return Dh, sigma_ln_Dh


# -----------------------------------------------------------
# FEMA (2014) HAZUS
# -----------------------------------------------------------
def Hazus2014(**kwargs):
    """
    Compute lateral spreading, which is the the same procedure used by Grant et al. (2016). See and use the function :func:`edp.ls.grant_etal_2016_ls`.

    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.

    """

    output = GrantEtal2016(**kwargs)

    return output