# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for ground settlement
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import numpy as np
from scipy import sparse
import LateralSpread


# -----------------------------------------------------------
def IdrissBoulanger2008(FS_liq, N1_60_cs=None, qc_1N_cs=None):
    """
    Compute volumetric strain following Idriss & Boulanger (2008); must specify either **N1_60_cs** or **qc_1N_cs**.

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
    e_vol : float
        [%] volumetric strain

    References
    ----------
    .. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.

    """
    
    # Use ls.idriss_boulanger_2008_ls to compute shear strain and relative density
    gamma_max, Dr = ls.idriss_boulanger_2008_ls(FS_liq, N1_60_cs, qc_1N_cs)
    
    # Estimate maximum volumetric strain from maximum shear strain
    # eps_v = 1.5*np.exp(-0.369 * N1_60_cs**(0.5)) * min(0.08,gamma_max) * 100 # %, using N1_60_cs
    eps_v = 1.5*np.exp(-2.5 * Dr/100) * min(0.08,gamma_max/100) * 100 # %, using Dr, eq. 95 in Idriss & Boulanger (2008)

    # Correction suggested by Cetin et al. (2009)
    # eps_v = eps_v * 0.9

    #
    return eps_v
        

# -----------------------------------------------------------
def ZhangEtal2002(FS_liq, qc_1N_cs):
    """
    Compute volumetric strain from CPT following the Zhang et al. (2002) method.

    Parameters
    ----------
    FS_liq : float, array
        factor of safety against liquefaction triggering
    qc_1N_cs : float, array
        CPT tip resistance corrected for overburden and fines content

    Returns
    -------
    eps_v : float
        [%] volumetric strain

    References
    ----------
    .. [1] Zhang, G., Robertson, P.K., and Brachman, R.W.I., 2002, Estimating liquefaction-induced ground settlements from CPT for level ground, Canadian Geotechnical Journal, vol. 39, no. 5, pp. 1168-1180.

    """

    #
    # if qc_1N_cs is None or FS_liq is None:
        #
        # logging.info(f"Must specify qc_1N_cs and FS_liq to proceed with method")
        
        #
        # return None
        
    # else:
    # Check if params are arrays and convert if not
    if not isinstance(FS_liq,np.ndarray):
        FS_liq = np.asarray([FS_liq])
        
    if not isinstance(qc_1N_cs,np.ndarray):
        qc_1N_cs = np.asarray([qc_1N_cs])
        
    # Since relationships in Zhang et al. (2002) for discrete Dr, first compute strain at bounds, then interpolate
    # 10 discrete FS values where relationships are given, 11 bins total (e.g., bin 1 = FS <= 0.5, bin 11 = FS >= 2):
    FS_bin_range = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2])
    
    # Bound applicable range for FS_liq and qc_1N_cs
    FS_liq = np.maximum(np.minimum(FS_liq,2),0.5)
    qc_1N_cs = np.maximum(np.minimum(qc_1N_cs,200),33)
    
    # Determine FS_liq upper and lower bin values
    bin_arr = np.digitize(FS_liq,FS_bin_range)
    
    bin_arr_high = bin_arr
    bin_arr_high[bin_arr_high>len(FS_bin_range)-1] = len(FS_bin_range)-1 # upper limit = number of bin range
    bin_arr_low = bin_arr-1
    bin_arr_low[bin_arr_low<0] = 0 # lower limit = 0
    
    FS_liq_high = FS_bin_range[bin_arr_high]
    FS_liq_low = FS_bin_range[bin_arr_low]
    
    # Appendix A in Zhang et al. (2002) provides relationships for volumetric strain in the form of:
    # --- eps_v = c1 * (qc_1N_cs) ^ c2, where c1 and c2 are dependent on FS_liq and qc_1N_cs
    # Determine coeffcients c1 and c2 at the FS_liq upper and lower bounds
    # Start with FS >= 2, no volumetric strain
    c1_high = np.zeros(FS_liq.shape)
    c2_high = np.zeros(FS_liq.shape)
    c1_low = np.zeros(FS_liq.shape)
    c2_low = np.zeros(FS_liq.shape)
    
    # FS_liq >= 1.3 and < 2
    c1_low[FS_liq < 2] = 7.6
    c2_low[FS_liq < 2] = -0.71
    
    # FS_liq >= 1.2 and < 1.3
    c1_high[FS_liq < 1.3] = 7.6
    c2_high[FS_liq < 1.3] = -0.71
    c1_low[FS_liq < 1.3] = 9.7
    c2_low[FS_liq < 1.3] = -0.69
    
    # FS_liq >= 1.1 and < 1.2
    c1_high[FS_liq < 1.2] = 9.7
    c2_high[FS_liq < 1.2] = -0.69
    c1_low[FS_liq < 1.2] = 11
    c2_low[FS_liq < 1.2] = -0.65
    
    # FS_liq >= 1.0 and < 1.1
    c1_high[FS_liq < 1.1] = 11
    c2_high[FS_liq < 1.1] = -0.65
    c1_low[FS_liq < 1.1] = 64
    c2_low[FS_liq < 1.1] = -0.93
    
    # FS_liq >= 0.9 and < 1.0
    c1_high[FS_liq < 1.0] = 64
    c2_high[FS_liq < 1.0] = -0.93
    c1_low[FS_liq < 1.0] = 1430
    c2_low[FS_liq < 1.0] = -1.48
    
    # FS_liq >= 0.8 and < 0.9
    c1_high[FS_liq < 0.9] = 1430
    c2_high[FS_liq < 0.9] = -1.48
    c1_low[FS_liq < 0.9] = 1690
    c2_low[FS_liq < 0.9] = -1.46
    
    # FS_liq >= 0.7 and < 0.8
    c1_high[FS_liq < 0.8] = 1690
    c2_high[FS_liq < 0.8] = -1.46
    c1_low[FS_liq < 0.8] = 1701
    c2_low[FS_liq < 0.8] = -1.42
    
    # FS_liq >= 0.6 and < 0.7
    c1_high[FS_liq < 0.7] = 1701
    c2_high[FS_liq < 0.7] = -1.42
    c1_low[FS_liq < 0.7] = 2411
    c2_low[FS_liq < 0.7] = -1.45
    
    # FS_liq > 0.5 and < 0.6
    c1_high[FS_liq < 0.6] = 2411
    c2_high[FS_liq < 0.6] = -1.45
    c1_low[FS_liq < 0.6] = 102
    c2_low[FS_liq < 0.6] = -0.82
    
    # FS_liq <= 0.5
    c1_high[FS_liq <= 0.5] = 102
    c2_high[FS_liq <= 0.5] = -0.82
    
    # Compute volumetric strain at bounds
    eps_v_high = c1_high*qc_1N_cs**c2_high
    eps_v_low = c1_low*qc_1N_cs**c2_low
    
    # Apply limiting strain
    eps_v_high = np.minimum(eps_v_high,102*qc_1N_cs**(-0.82))
    eps_v_low = np.minimum(eps_v_low,102*qc_1N_cs,33**(-0.82))
    
    # Interpolate for volumetric strain at FS_liq
    eps_v = eps_v_low + (eps_v_high - eps_v_low)*(FS_liq - FS_liq_low)/(FS_liq_high - FS_liq_low)
    
    #
    return eps_v
        

# -----------------------------------------------------------
def CetinEtal2009(**kwargs):
    """
    Compute volumetric strain following the Cetin et al. (2009) probabilistic method.

    Parameters
    ----------
    N1_60_cs : float
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [kPa] initial vertical total stress
    sv0_eff : float
        [kPa] initial vertical effective stress
    rd : float
        stress reduction factor with depth
    Dr : float
        [%] relative density
    M : float
        Moment magnitude
    patm : float, optional
        [kPa] atmospheric pressure; **default = 101.3 kPa**

    Returns
    -------
    eps_v : float
        [%] volumetric strain

    References
    ----------
    .. [1] Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 3, pp. 387-398.

    """

    ############ Inputs ############
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    Dr = kwargs.get('Dr',None) # %, relative density
    patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    M = kwargs.get('M',None) # moment magnitude

    # Multidirectional shaking effects
    if Dr is None:
        K_md = 1.0
    else:
        K_md = 0.361*np.log(Dr) - 0.579 # eq. 3

    # Magnitude scaling factor
    if M is None:
        K_M = 1.0
    else:
        K_M = 87.1/M**2.217 # eq. 4

    # Overburden correction factor
    if Dr is None:
        K_sigma = 1.0 # eq. 31
    else:
        f = 1 - 0.005*Dr # factor for K_sigma, eq. 5
        K_sigma = (sv0_eff/patm)**(f-1) # eq. 5

    if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
        csr = None
        print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
    else:
        # Cyclic stress ratio (demand)
        csr_field = 0.65 * amax * sv0_tot/sv0_eff * rd # Seed and Idriss (1971)
        csr = csr_field/K_md/K_M/K_sigma # eq. 2, CSR corrected for unidirectionality in lab, magnitude, and overburden

    #
    if N1_60_cs < 5 or N1_60_cs > 40:
        print('N1_60_cs is outside the range of 5 to 40 given by Cetin et al. (2009)')
    if csr < 5 or csr > 40:
        print('CSR_SS_20_1D_1_atm is outside the range of 0.05 to 0.60 given by Cetin et al. (2009)')

    #
    ln_eps_v = np.log(1.879*np.log((780.416*np.log(csr) - N1_60_cs + 2442.465)/(636.613*N1_60_cs + 306.732)) + 5.583)
    eps_v = np.exp(ln_eps_v) * 100 # %
    sigma = 0.689

    # maximum volumetric strain, after Huang (2008)
    eps_v_max = 9.765 - 2.427*np.log(N1_60_cs) # %

    # volumetric strain as the minimum of the correlated and maximum values
    eps_v = min(eps_v, eps_v_max) # %

    #
    eps_v = eps_v * 1.15 # correction factor suggested in Peterson (2016)

    #
    return eps_v


# -----------------------------------------------------------
def Hazus2014(**kwargs):
    """
    Compute volumetric settlement at a given location using a simplified deterministic approach (after Tokimatsu and Seed, 1987).

    Parameters
    ----------
    liq_susc : str
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    pgd_gs : float
        [cm] permanent ground deformation (settlement)

    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Buildin Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
    .. [2] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

    """

    # get inputs
    liq_susc = kwargs.get('liq_susc',None) # liquefaction susceptibility category

    # get threshold pga against liquefaction
    pgd_gs = np.ones(liq_susc.shape)*np.nan
    pgd_gs[liq_susc=='very high'] = 12 # inches
    pgd_gs[liq_susc=='high'] = 6 # inches
    pgd_gs[liq_susc=='moderate'] = 2 # inches
    pgd_gs[liq_susc=='low'] = 1 # inches
    pgd_gs[liq_susc=='very low'] = 0 # inches
    pgd_gs[liq_susc=='none'] = 0 # inches

    # convert to cm
    pgd_gs = sparse.coo_matrix(pgd_gs*2.54)

    # probability distribution
    prob_dist_type = 'uniform'
    factor_aleatory = 3
    factor_epistemic = 4

    # store outputs
    output = {}
    output.update({'pgd_gs': pgd_gs})
    output.update({'prob_dist': {'type': prob_dist_type,
                                'factor_aleatory': factor_aleatory,
                                'factor_epistemic': factor_epistemic}})

    print(output)

    #
    return output


# -----------------------------------------------------------
def IshiharaYoshimine1992(FS_liq, N1_60_cs=None, qc_1N_cs=None):
    """
    Compute volumetric strain following Ishihara and Yoshimine (1992); must specify either **N1_60_cs** or **qc_1N_cs**.

    Parameters
    ----------
    FS_liq : float
        factor of safety against liquefaction triggering
    N1_60_cs : float, optional
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content; used to estimate :math:`Dr = \sqrt{(N_{1-60})_{cs}/46}`
    qc_1N_cs : float, optional
        CPT tip resistance corrected for overburden and fines content; used to estimate :math:`Dr = \sqrt{(N_{1-60})_{cs}/46}`

    Returns
    -------
    eps_v : float
        [%] volumetric strain

    References
    ----------
    .. [1] Ishihara, K., and Yoshimine, M., 1992, Evaluation of Settlements in Sand Deposits Following Liquefaction During Earthquakes, Soils and Foundations, vol. 32, no. 1, pp. 173-188.
    .. [2] Franke, K.W., Ekstrom, L.T., Ulmer, K.J., Astorga, L., and Error, B., 2016, Simplified Standard Penetration Test Performance-Based Assessment of Liquefaction and Effects, Brigham Young University, Report No. UT-16.16, Provo, UT, USA.
    .. [3] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.

    """

    #
    eps_v = idriss_boulanger_2008_gs(FS_liq, N1_60_cs, qc_1N_cs)
    
    #
    return eps_v


# -----------------------------------------------------------
def TokimatsuSeed1987(**kwargs):
    """
    Compute volumetric strain following the Tokimatsu and Seed (1987) deterministic method.

    Parameters
    ----------
    TBD : float
        TBD

    Returns
    -------
    eps_v : float
        [%] volumetric strain

    References
    ----------
    .. [1] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

    """

    print('Placeholder - under development')

    return None