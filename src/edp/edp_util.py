# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Additional functions for engineering demand parameters
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python Modules
import logging
import sys
import numpy as np
from numba import njit, float64
from numba.types import Tuple


@njit(
    float64[:,:,:](float64[:,:,:],float64[:,:]),
    fastmath=True,
    cache=True
)
def get_shear_strain_ZhangEtal2004(FSLiq, DR):
    """
    Maximum shear strain based on Zhang et al. (2004)
    Parameters:
        FSLiq: Factor of Safety against Liquefaction triggering
        DR: Relative Density
    Returns:
        gamma_max: Maximum shear strain (%)
    """
    # get dimensions
    n_domain = FSLiq.shape[0]
    n_depth = FSLiq.shape[1]
    n_sample = FSLiq.shape[2]
    # initialize gamma_max array
    gamma_max = np.zeros(FSLiq.shape)
    # loop through domain vector
    for i in range(n_domain):
        for j in range(n_depth):
            FSLiq_i_j = FSLiq[i,j]
            DR_j = DR[j]
            #
            cond = np.logical_and(
                np.logical_and(DR_j>=85.0,DR_j<95.0),
                FSLiq_i_j<=0.7
            )
            gamma_max[i,j][cond] = 6.2
            cond = np.logical_and(
                np.logical_and(DR_j>=85.0,DR_j<95.0),
                np.logical_and(FSLiq_i_j>0.7,FSLiq_i_j<=2.0)
            )
            gamma_max[i,j][cond] = 3.26 * (FSLiq_i_j[cond] ** -1.8)
            #
            cond = np.logical_and(
                np.logical_and(DR_j>=75.0,DR_j<85.0),
                FSLiq_i_j<=0.56
            )
            gamma_max[i,j][cond] = 10.0
            cond = np.logical_and(
                np.logical_and(DR_j>=75.0,DR_j<85.0),
                np.logical_and(FSLiq_i_j>0.56,FSLiq_i_j<=2.0)
            )
            gamma_max[i,j][cond] = 3.22 * (FSLiq_i_j[cond] ** -2.08)
            #
            cond = np.logical_and(
                np.logical_and(DR_j>=65.0,DR_j<75.0),
                FSLiq_i_j<=0.59
            )
            gamma_max[i,j][cond] = 14.5
            cond = np.logical_and(
                np.logical_and(DR_j>=65.0,DR_j<75.0),
                np.logical_and(FSLiq_i_j>0.59,FSLiq_i_j<=2.0)
            )
            gamma_max[i,j][cond] = 3.20 * (FSLiq_i_j[cond] ** -2.89)
            #
            cond = np.logical_and(
                np.logical_and(DR_j>=55.0,DR_j<65.0),
                FSLiq_i_j<=0.66
            )
            gamma_max[i,j][cond] = 22.7
            cond = np.logical_and(
                np.logical_and(DR_j>=55.0,DR_j<65.0),
                np.logical_and(FSLiq_i_j>0.66,FSLiq_i_j<=2.0)
            )
            gamma_max[i,j][cond] = 3.58 * (FSLiq_i_j[cond] ** -4.42)
            #
            cond = np.logical_and(
                np.logical_and(DR_j>=45.0,DR_j<55.0),
                FSLiq_i_j<=0.72
            )
            gamma_max[i,j][cond] = 34.1
            cond = np.logical_and(
                np.logical_and(DR_j>=45.0,DR_j<55.0),
                np.logical_and(FSLiq_i_j>0.72,FSLiq_i_j<=2.0)
            )
            gamma_max[i,j][cond] = 4.22 * (FSLiq_i_j[cond] ** -6.39)
            #
            cond = np.logical_and(DR_j<45.0,FSLiq_i_j<=0.81)
            gamma_max[i,j][cond] = 51.2
            cond = np.logical_and(
                DR_j<45.0,
                np.logical_and(FSLiq_i_j>0.81,FSLiq_i_j<=1.0)
            )
            gamma_max[i,j][cond] = 250 * (1.0 - FSLiq_i_j[cond]) + 3.5
            cond = np.logical_and(
                DR_j<45.0,
                np.logical_and(FSLiq_i_j>1.0,FSLiq_i_j<=2.0)
            )
            gamma_max[i,j][cond] = 3.31 * (FSLiq_i_j[cond] ** -7.97)
    # limit to 1e-5% strain
    # gamma_max = np.maximum(gamma_max,1e-5)
    # return
    return gamma_max


@njit(
    float64[:,:,:](float64[:,:,:],float64[:,:]),
    fastmath=True,
    cache=True
)
def get_shear_strain_IdrissBoulanger2008(FSLiq, DR):
    """
    Deviatoric strain prediction based on Idriss & Boulanger (2008)
    Parameters:
        qc1ncs: Weighted qc1ncs value
    Returns:
        gamma_max_IB: Maximum shear strain (%)
    """
    # get dimensions
    n_domain = FSLiq.shape[0]
    n_depth = FSLiq.shape[1]
    n_sample = FSLiq.shape[2]
    # # intermediate params
    # gamma_lim = np.maximum(0, np.minimum(50, (1.859 * (1.1 - (DR / 100)) ** 3) * 100))
    #     gamma_lim[DR>=95.0] = 0
    #     gamma_lim[FSLiq>=95.0] = 0
    F_alpha = np.zeros((n_depth,n_sample))
    for j in range(n_depth):
        DR_j = DR[j]
        F_alpha[j] = 0.032 + 4.7 * (DR_j / 100) - 6 * (DR_j / 100) ** 2
        F_alpha[j][DR_j<40] = 0.032 + 4.7 * 0.4 - 6 * 0.4 ** 2
    # # initialize gamma_max array
    gamma_max = np.zeros(FSLiq.shape)
    # # loop through domain vector
    for i in range(n_domain):
        for j in range(n_depth):
            DR_j = DR[j]
            FSLiq_i_j = FSLiq[i,j]
            F_alpha_j = F_alpha[j]
            #
            gamma_lim_j = np.maximum(0, np.minimum(50, (1.859 * (1.1 - (DR_j / 100)) ** 3) * 100))
            gamma_lim_j[DR_j>=95.0] = 0
            gamma_lim_j[FSLiq_i_j>=2.0] = 0
            #
            cond = np.logical_and(FSLiq_i_j<2,FSLiq_i_j>F_alpha_j)
            gamma_max[i,j][cond] = np.minimum(
                gamma_lim_j[cond],
                np.maximum(0, 0.035 * (2 - FSLiq_i_j[cond]) * ((1 - F_alpha_j[cond]) / (FSLiq_i_j[cond] - F_alpha_j[cond])) * 100)
            )
            #
            gamma_max[i,j][FSLiq_i_j <= F_alpha_j] = gamma_lim_j[FSLiq_i_j <= F_alpha_j]
    # limit to 1e-5% strain
    # gamma_max = np.maximum(gamma_max,1e-5)
    #
    return gamma_max


@njit(
    float64[:,:](
        float64[:],float64[:,:],float64[:,:],float64
    ),
    fastmath=True,
    cache=True
)
def get_cpt_dr(qc, tot_sig_v0, eff_sig_v0, pa=101.3):
    """
    Return the relative density based on CPT correlations proposed by:
    Idriss & Boulanger (2008) (0.25 weight)
    Jamiolkowski et al (2001) (0.25 weight)
    Kulhawy & Maine (1990) (0.25 weight)
    Tatsuoka et al. (1990) (0.25 weight)
    Parameters:
        qc: Measured tip resistance (MPa)
        eff_sig_vo: Effective stress at specific depth (kPa)
        k0: Coefficient of lateral pressure at rest
    Return:
        DR: Relative density in percent
    """
    # get dimensions
    n_depth = eff_sig_v0.shape[0]
    n_sample = eff_sig_v0.shape[1]
    # initialize
    dr_calc = np.zeros(eff_sig_v0.shape)
    ####    CALCULATE RELATIVE DENSITY USING IDRISS & BOULANGER (2008)    ####
    # loop through layers
    for i in range(n_depth):
        qc_i = qc[i]
        # loop through samples
        for j in range(n_sample):
            # get current params
            tot_sig_v0_i_j = tot_sig_v0[i,j]
            eff_sig_v0_i_j = eff_sig_v0[i,j]
            # starting guess
            dr_iter_i_j, dr_calc_i_j = 0.5, 0.5
            counter = 0
            while True:
                dr_iter_i_j = dr_calc_i_j
                m_i_j = 0.784 - 0.521 * dr_iter_i_j
                cn_i_j = np.minimum(1.7, (pa / eff_sig_v0_i_j) ** (m_i_j))
                Qtn_i_j = np.maximum(0.2, ((qc_i * 1000 - tot_sig_v0_i_j) / pa) * cn_i_j)
                dr_calc_i_j = 0.478 * ((Qtn_i_j) ** 0.264) - 1.063
                counter += 1
                if counter > 25:
                    break
            # store after iteration
            dr_calc[i,j] = dr_calc_i_j
    # repeat to be in dimensions of n_depth x n_samples
    qc_repeat = qc.repeat(n_sample).reshape((-1, n_sample))
    # get dr for the various methods
    dr_IB = np.maximum(5, np.minimum(98, dr_calc * 100))
    ####    CALCULATE RELATIVE DENSITY USING JAMIOLKOWSKI ET AL. (2001)    ####
    dr_Jam = np.maximum(5, np.minimum(98, 100 * (1 / 3.1) * np.log((qc_repeat * 9.81) / (17.68 * (eff_sig_v0 / 100) ** 0.5))))
    ####    CALCULATE RELATIVE DENSITY USING KULHAWY & MAYNE (1990)    ####
    dr_KM = np.maximum(5, np.minimum(98, 100 * ((1 / 305) * ((qc_repeat * 1000 / pa) / ((eff_sig_v0 / pa) ** 0.5))) ** 0.5))
    ####    CALCULATE WEIGHTED RELATIVE DENSITY    ####
    w_IB, w_Jam, w_KM = 0.4, 0.3, 0.3
    DR = (w_IB * dr_IB + w_Jam * dr_Jam + w_KM * dr_KM)
    return DR


@njit(
    Tuple((float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:]))(
        float64[:],float64[:],float64[:,:],float64[:,:],float64
    ),
    fastmath=True,
    cache=True
)
def get_sbt_index(qc, fs, tot_sig_v0, eff_sig_v0, pa=101.3):
    """
    Return the normalized tip resistance and sleeve friction
    Return the SBT index (Ic) from Robertson (2009)
    Iteration to obtain n value is performed
    Parameters:
        qc: Measured tip resistance (MPa)
        fs: Measured sleeve resistance (MPa)
        tot_sig_vo: Total stress at specific depth (kPa)
        eff_sig_vo: Effective stress at specific depth (kPa)
        pa: Reference pressure 1 atm = 101.3 kPa
    Return:
        qn_it: Normalized, dimensionless tip resistance (1 atm & overburden)
        fn_it: Normalized friction ratio (%)
        ic_it: SBT Index, Ic Robertson (2009)
        n_calc: Stress exponent
    """
    # get dimensions
    n_depth = eff_sig_v0.shape[0]
    n_sample = eff_sig_v0.shape[1]
    # initialize
    cn_it = np.zeros(eff_sig_v0.shape)
    qn_it = np.zeros(eff_sig_v0.shape)
    fn_it = np.zeros(eff_sig_v0.shape)
    ic_it = np.zeros(eff_sig_v0.shape)
    n_calc = np.zeros(eff_sig_v0.shape)
    # loop through layers
    for i in range(n_depth):
        qc_i = qc[i]
        fs_i = fs[i]
        # loop through samples
        for j in range(n_sample):
            # get current params
            tot_sig_v0_i_j = tot_sig_v0[i,j]
            eff_sig_v0_i_j = eff_sig_v0[i,j]
            # starting guess
            n_iter_i_j, n_calc_i_j = 0.5, 1.0
            # iteration
            while True:
                n_iter_i_j = n_calc_i_j
                cn_it_i_j = min((pa / eff_sig_v0_i_j) ** n_iter_i_j, 2)
                qn_it_i_j = max(0.2, (((qc_i * 1000) - tot_sig_v0_i_j) / pa) * cn_it_i_j)
                fn_it_i_j = max(0.1, ((fs_i * 1000) / ((qc_i * 1000) - tot_sig_v0_i_j)) * 100)
                ic_it_i_j = ((3.47 - np.log10(qn_it_i_j)) ** 2 + (np.log10(fn_it_i_j) + 1.22) ** 2) ** 0.5
                n_calc_i_j = min((0.381 * ic_it_i_j + 0.05 * (eff_sig_v0_i_j / pa) - 0.15), 1)
                ####    ERROR TOLERANCE, ROBERTSON (2009), CGJ, V.46,pg 1337-1355    ####
                err = np.fabs(n_iter_i_j - n_calc_i_j)
                if err < 0.0001:
                    break
            # store after iteration
            cn_it[i,j] = cn_it_i_j
            qn_it[i,j] = qn_it_i_j
            fn_it[i,j] = fn_it_i_j
            ic_it[i,j] = ic_it_i_j
            n_calc[i,j] = n_calc_i_j
    # return
    return cn_it, qn_it, fn_it, ic_it, n_calc


@njit(
    Tuple((float64[:,:,:],float64[:,:]))(
        float64[:],float64[:],float64[:],float64[:,:],float64[:,:],
        float64[:,:],float64[:,:],float64[:,:],
        float64[:],float64,
        float64
    ),
    fastmath=True,
    cache=True
)
def get_FSLiq_Rob09(
    qc, fs, z, tot_sig_v0, eff_sig_v0,
    Qtn, fn, ic,
    pga, mag,
    pa=101.3
):
    """
    Return the Factor of Safety against Liquefaction triggering based on Robertson (2009)
    Implements probabilistic Robertson & Wride procedure from Ku et al. (2012)
    Parameters:
        qc: Measured tip resistance (MPa)
        fs: Measured sleeve resistance (MPa)
        tot_sig_vo: Total stress at specific depth (kPa)
        eff_sig_vo: Effective stress at specific depth (kPa)
        z: Depth (m)
        pga: Peak ground acceleration at the ground surface
        mag: Earthquake moment magnitude
    Return:
        FSLiq_Rob09: Factor of Safety against Liquefaction triggering based on Robertson (2009)
    """
    # def f(CRR):
    #     return ((0.102 + np.log(CRR_step / CRR)) / 0.276) - z_score
    
    # get dimensions
    n_depth = eff_sig_v0.shape[0]
    n_sample = eff_sig_v0.shape[1]
    n_domain = len(pga)
    
    # initialize matrix for fs_liq
    FSLiq = np.zeros((n_domain,n_depth,n_sample))
    
    # first get resistance, not dependent on IM
    # get SBT index
    # _, Qtn, fn, ic, _ = get_sbt_index(qc, fs, tot_sig_v0, eff_sig_v0, pa=pa)
    # loop through number of samples since numba does not support 2D boolean operations
    kc = np.ones(eff_sig_v0.shape)
    for j in range(n_sample):
        # get values for current sample
        ic_j = ic[:,j]
        # if ic <= 1.64:
            # kc = 1
        # elif ic > 1.64 and ic < 2.36 and fn < 0.5:
            # kc = 1
        # elif ic > 1.64 and ic <= 2.5:
        kc[np.logical_and(ic_j>1.64,ic_j<=2.5),j] = 5.581*(ic_j[np.logical_and(ic_j>1.64,ic_j<=2.5)])**3 - \
            0.403*(ic_j[np.logical_and(ic_j>1.64,ic_j<=2.5)])**4 - \
            21.63*(ic_j[np.logical_and(ic_j>1.64,ic_j<=2.5)])**2 + \
            33.75*ic_j[np.logical_and(ic_j>1.64,ic_j<=2.5)] - 17.88
        # elif ic > 2.5 and ic < 2.7:
        kc[np.logical_and(ic_j>2.5,ic_j<=2.7),j] = 6*(10)**-7 * (ic_j[np.logical_and(ic_j>2.5,ic_j<=2.7)])**16.76
        # else:
        kc[ic_j>2.7,j] = 5.581*(ic_j[ic_j>2.7])**3 - 0.403*(ic_j[ic_j>2.7])**4 - 21.63*(ic_j[ic_j>2.7])**2 + 33.75*ic_j[ic_j>2.7] - 17.88
        
    # get qc1ncs
    qc1ncs = kc * Qtn
    
    # loop through number of samples since numba does not support 2D boolean operations
    CRR = np.ones(eff_sig_v0.shape)*4
    for j in range(n_sample):
        # get values for current sample
        qc1ncs_j = qc1ncs[:,j]
        ic_j = ic[:,j]
        ####    CYCLIC RESISTANCE RATIO (CRR) CALCULATION    ####
        # if qc1ncs > 160 or ic > 2.4:
        #     CRR = 4
        # elif qc1ncs < 50 and ic <= 2.4:
        CRR_step_j = 0.833 * (qc1ncs_j[np.logical_and(qc1ncs_j<50,ic_j<=2.4)] / 1000) + 0.05
        CRR[np.logical_and(qc1ncs_j<50,ic_j<=2.4),j] = CRR_step_j*np.exp(0.102) # instead of using fsolve, find theoretical CRR where f=0
            # CRR = fsolve(f, CRR_step)
        # elif qc1ncs >= 50 and qc1ncs <= 160 and ic <= 2.4:
        CRR_step_j = 93 * (qc1ncs_j[np.logical_and(np.logical_and(qc1ncs_j>=50,qc1ncs_j<=160),ic_j<=2.4)] / 1000) ** 3 + 0.08
        CRR[np.logical_and(np.logical_and(qc1ncs_j>=50,qc1ncs_j<=160),ic_j<=2.4),j] = CRR_step_j*np.exp(0.102) # instead of using fsolve, find theoretical CRR where f=0
            # CRR = fsolve(f, CRR_step)
        # else:
        #     CRR = 4
        
    # next get demand
    ####    DEPTH REDUCTION FACTOR BASED ON IDRISS (1999)    ####
    # reshape before calculations
    alpha = -1.012 - 1.126 * np.sin((z / 11.73) + 5.133)
    beta = 0.106 + 0.118 * np.sin((z / 11.28) + 5.142)
    rd = np.exp(alpha + beta * mag)
    rd = rd.repeat(n_sample).reshape((-1, n_sample)) # repeat to be in dimensions of n_depth x n_samples
    ####    MAGNITUDE SCALING FACTOR CALCULATION    ####
    MSF = (10 ** 2.24) / (mag ** 2.56)
    # loop through number of points for domain vector
    for i in range(n_domain):
        ####    CYCLIC STRESS RATIO (CSR) CALCULATION    ####
        CSR_i = (0.65 * tot_sig_v0 * pga[i] * rd) / (eff_sig_v0 * MSF)
        ####    FACTOR OF SAFETY AGAINST LIQUEFACTION TRIGGERING CALCULATION    ####
        FSLiq_i = CRR/CSR_i
        # if qc1ncs > 160 or ic > 2.4:
            # FSLiq = 4
        for j in range(n_sample):
            # get values for current sample
            qc1ncs_j = qc1ncs[:,j]
            ic_j = ic[:,j]
            FSLiq_i[np.logical_or(qc1ncs_j>160,ic_j>2.4),j] = 4
        # else:
        #     FSLiq = CRR / CSR
        # store to output
        # FSLiq[i] = np.minimum(FSLiq_i,4)
        FSLiq[i] = FSLiq_i
    # return
    return FSLiq, qc1ncs


@njit(
    Tuple((float64[:,:,:],float64[:,:]))(
        float64[:],float64[:],float64[:],float64[:,:],float64[:,:],
        float64[:,:],
        float64[:],float64,
        float64
    ),
    fastmath=True,
    cache=True
)
def get_FSLiq_BI16(
    qc, fs, z, tot_sig_v0, eff_sig_v0,
    ic, 
    pga, mag,
    pa=101.3
):
    """
    Return the Factor of Safety against Liquefaction triggering based on Boulanger & Idriss (2016)
    Parameters:
        qc: Measured tip resistance (MPa)
        fs: Measured sleeve resistance (MPa)
        tot_sig_vo: Total stress at specific depth (kPa)
        eff_sig_vo: Effective stress at specific depth (kPa)
        z: Depth (m)
        pa: Reference pressure 1 atm = 101.3 kPa
        mag: Earthquake moment magnitude
        pga: Peak ground acceleration at the ground surface
    Return:
        FSLiq_BI16: Factor of Safety against Liquefaction triggering based on Boulanger & Idriss (2016)
    """
    # get dimensions
    n_depth = eff_sig_v0.shape[0]
    n_sample = eff_sig_v0.shape[1]
    n_domain = len(pga)
    
    # initialize matrix for fs_liq
    FSLiq = np.zeros((n_domain,n_depth,n_sample))
    qc1ncs = np.zeros((n_depth,n_sample))
    
    # first get resistance, not dependent on IM
    # get SBT index
    # _, _, _, ic, _ = get_sbt_index(qc, fs, tot_sig_v0, eff_sig_v0, pa=pa)
    ####    GENERIC FINES CONTENT CORRELATION FROM BOULANGER & IDRISS (2014), COMMENT OUR FOR SITES IN NEW ZEALAND    ####
    cfc = 0
    fc = np.minimum(100, np.maximum(0, 80 * (ic + cfc) - 137))
    ####    FINES CONTENT RELATIONSHIP FOR NEW ZEALAND FROM MAURER ET AL. (2019), COMMENT OUT FOR SITES OUTSIDE OF NEW ZEALAND    ####
    # fc = np.minimum(100, np.maximum(0, 80.645 * ic - 128.5967))
    qcn = (qc * 1000) / pa
    # qcn = qcn.repeat(n_sample).reshape((-1, n_sample)) # repeat to be in dimensions of n_depth x n_samples
    # loop through layers
    for i in range(n_depth):
        qcn_i = qcn[i]
    
        # loop through samples
        for j in range(n_sample):
            eff_sig_v0_i_j = eff_sig_v0[i,j]
            fc_i_j = fc[i,j]
    
            # starting guess
            qc1ncs_iter_i_j, qc1ncs_calc_i_j = 100.0, 50.0
            while True:
                qc1ncs_iter_i_j = qc1ncs_calc_i_j
                ####    OVERBURDEN CORRECTION FACTOR BASED ON BOULANGER & IDRISS (2016)    ####
                cn_i_j = np.minimum(1.7, (pa / eff_sig_v0_i_j) ** (1.338 - 0.249 * np.maximum(21, np.minimum(qc1ncs_iter_i_j, 254)) ** 0.264))
                qc1n_i_j = qcn_i * cn_i_j
                ####    del_qc1n = FC CORRECTION BASED ON BOULANGER & IDRISS (2016)    ####
                del_qc1n_i_j = (11.9 + qc1n_i_j / 14.6) * np.exp(1.63 - 9.7 / (fc_i_j + 2) - ((15.7 / (fc_i_j + 2)) ** 2))
                qc1ncs_calc_i_j = qc1n_i_j + del_qc1n_i_j
                err = np.fabs(qc1ncs_iter_i_j - qc1ncs_calc_i_j)
                if err < 0.001:
                    break
            # store after iteration
            qc1ncs[i,j] = qc1ncs_calc_i_j
    ####    CYCLIC RESISTANCE RATIO (CRR) CALCULATION    ####
    CRR = np.exp(
        (np.minimum(211, qc1ncs) / 113) + \
        (np.minimum(211, qc1ncs) / 1000) ** 2 - \
        (np.minimum(211, qc1ncs) / 140) ** 3 + \
        (np.minimum(211, qc1ncs) / 137) ** 4 - 2.6
    )
            
    # next get demand
    ####    DEPTH REDUCTION FACTOR BASED ON IDRISS (1999)    ####
    alpha = -1.012 - 1.126 * np.sin((z / 11.73) + 5.133)
    beta = 0.106 + 0.118 * np.sin((z / 11.28) + 5.142)
    rd = np.exp(alpha + beta * mag)
    rd = rd.repeat(n_sample).reshape((-1, n_sample)) # repeat to be in dimensions of n_depth x n_samples
    ####    MAGNITUDE SCALING FACTOR CALCULATION    ####
    MSF_max = np.minimum(2.2, 1.09 + (qc1ncs / 180) ** 3)
    MSF = 1 + (MSF_max - 1) * (8.64 * np.exp(-mag / 4) - 1.325)
    ####    K_SIGMA CALCULATION    ####
    c_sigma = np.minimum(0.3, 1 / (37.3 - 8.27 * np.minimum(qc1ncs, 211) ** 0.264))
    k_sigma = np.minimum(1.1, 1 - c_sigma * np.log(eff_sig_v0 / pa))
    # loop through number of points for domain vector
    for i in range(n_domain):
        ####    CYCLIC STRESS RATIO (CSR) CALCULATION    ####
        CSR_i = (0.65 * tot_sig_v0 * pga[i] * rd) / (eff_sig_v0 * MSF * k_sigma)
        ####    FACTOR OF SAFETY AGAINST LIQUEFACTION TRIGGERING CALCULATION    ####
        FSLiq_i = CRR / CSR_i
        for j in range(n_sample):
            ic_j = ic[:,j]
            FSLiq_i[ic_j>2.4,j] = 4
        FSLiq[i] = FSLiq_i
        # if ic > 2.4:
        #     FSLiq_i = 4
        # else:
        #     FSLiq_i = CRR / CSR
    return FSLiq, qc1ncs
    # return np.ones((2,2,2)), np.ones((2,2))


@njit(float64[:](float64[:]),fastmath=True,cache=True)
def numbadiff(x):
    return x[1:] - x[:-1]


@njit(float64[:](float64[:],float64[:],float64),fastmath=True,cache=True)
def get_cpt_density(qc, fs, pa=101.3):
    """
    Return the ratio of unit weight of soil to unit weight of water (9.81 kN/m3).
    Parameters:
        qc = measured tip CPT resistance (MPa)
        fs = measured sleeve CPT resistance (MPa)
    Ref: Robertson & Cabal (2010), 2nd CPT Int. Symposium
    """
    densratio = 0.27*np.log10(fs/qc*100) + 0.36*np.log10(qc*1000/pa) + 1.236
    return densratio


@njit(
    Tuple((float64[:,:],float64[:,:],float64[:,:]))(
        float64[:],float64[:],float64[:],float64[:],float64
    ),
    fastmath=True,
    cache=True
)
def get_cpt_stress(qc, fs, z, gw_depth, gamma_water=9.81):
    """
    estimate in-situ stresses using CPT resistances
    """
    # dimensions
    n_depth = len(z)
    n_sample = len(gw_depth)
    # get density kN/m^3
    # density = get_cpt_density(qc, fs, pa=101.3) * gamma_water
    density = get_cpt_density(qc, fs, pa=101.3)*gamma_water
    # unit depth
    dz = np.zeros(z.shape)
    dz[0] = z[0]
    dz[1:] = numbadiff(z) # m
    # dz = np.tile(dz,(len(gw_depth),1)).T # repeat to be in dimensions of n_depth x n_samples
    # total stress
    tot_sig_v0 = np.maximum(np.cumsum(density * dz), 0.0002) # kPa, avoid 0 kPa
    tot_sig_v0 = tot_sig_v0.repeat(n_sample).reshape((-1, n_sample)) # repeat to be in dimensions of n_depth x n_samples
    # pore pressure
    z = z.repeat(n_sample).reshape((-1, n_sample)) # repeat to be in dimensions of n_depth x n_samples
    gw_depth = gw_depth.repeat(n_depth).reshape((-1, n_depth)).T # repeat to be in dimensions of n_depth x n_samples
    pore_press = np.maximum(z-gw_depth, 0) * gamma_water # kPa
    # effective stress
    eff_sig_v0 = tot_sig_v0 - pore_press # kPa
    # # return
    # return density, tot_sig_v0, eff_sig_v0
    return tot_sig_v0, pore_press, eff_sig_v0


# -----------------------------------------------------------
# def read_cpt_data(fn):
#     '''
#     Read the CPT data with csv format only.
#         Column 0: Depth
#         Column 1: Measured qc (MPa)
#         Column 2: Measured fs (MPa)
#         Column 3: Measured Pore Pressure
#     Note: Every row must contain valid data
#     '''
#     path = os.path.join(os.getcwd(), fn)
#     with open(path) as csvfile:
#         readCSV = csv.reader(csvfile, delimiter=',')
#         next(readCSV, None)
#         depth, qc, fs, u = [], [], [], []
#         for row in readCSV:
#             depth.append(float(row[0]))
#             qc.append(float(row[1]))
#             fs.append(np.maximum(0.001, float(row[2])))
#             u.append(float(row[3]))
#     return depth, qc, fs, u


# -----------------------------------------------------------
def get_stress(**kwargs):
    """
    Computes the in-situ total and effective stress and pore pressure profiles an interpolate pressures at user-specified depths. The function handles two types of inputs:
    
    1. properties by layer (typical for SPT): properties for various layers are provided, along with layer thicknesses
    2. properties by depth (typical for CPT): properties are given at specfic depths
    
    Parameters
    ----------
    z : float, array
        [m] depths to interpolate for stresses
    dtw : float
        [m] depth to water table
    gamma_w = float, optional
        [kN/m^3] unit weight of water, default = 9.81 kN/m^3
    
    Type 1 inputs
    gamma_layer : float, array
        [kN/m^3] unit weight of soil at each layer
    dh_layer : float, array
        [m] layer thicknesses
    
    Type 2 inputs
    gamma_z : float, array
        [kN/m^3] unit weights at specific depths
        
    Returns
    -------
    sig_tot_z = float, array
        [kPa] in-situ vertical total stress
    u_z = float, array
        [kPa] hydrostatic pore pressure (ignoring capillary suction above **gwt**)
    sig_eff_z = float, array
        [kPa] in-situ vertical effective stress
    
    """
    
    # get inputs
    z = kwargs.get('z',None) # m, depth to get stresses
    dtw = kwargs.get('dtw',0) # m, depth to water-table, default 0 m
    gamma_w = kwargs.get('gamma_w',9.81) # kN/m^3 unit weight of water, default to metric
    
    # method-dependent parameters
    gamma_layer = kwargs.get('gamma_layer',None) # unit weight of layers
    dH_layer = kwargs.get('dH_layer',None) # layer thicknesses, must have the same length as gamma_layer
    gamma_z = kwargs.get('gamma_z',None) # kN/m^3, unit weight of layers
    
    # convert to numpy array for easier computation
    if type(z) is not np.array:
        z = np.asarray(z)
    
    # check method to use for calculating total stress
    # 1) by layer: gamma_layer & dH_layer - for spt where properties for a number of layers are given
    # 2) by depth: gamma_z - for cpt where properties at specific depths are given
    if gamma_layer is not None and dH_layer is not None:
        # convert to numpy array for easier computation
        if type(gamma_layer) is not np.array:
            gamma_layer = np.asarray(gamma_layer)
        if type(dH_layer) is not np.array:
            dH_layer = np.asarray(dH_layer)
            
        # determine stress and depth arrays from input layer properties
        nLayers = len(gamma_layer) # number of layers
        dsig_tot = np.multiply(gamma_layer,dH_layer) # calculate total stress increase per layer
        sig_sum = [sum(dsig_tot[0:i+1]) for i in range(nLayers)] # get cumulative total stress at the base of each layer
        z_full = np.hstack([0,[sum(dH_layer[0:i+1]) for i in range(nLayers)]]) # get full depth array, pad 0 at the start for interpolation
        sig_tot_full = np.hstack([0,sig_sum]) # pad 0 kPa to start of total stress array (for interpolation)
        
        # interpolate to calculate total stress at target depths
        sig_tot_z = np.interp(z,z_full,sig_tot_full) # vertical total stress
    
    elif gamma_z is not None:
        # convert to numpy array for easier computation
        if type(gamma_z) is not np.array:
            gamma_z = np.asarray(gamma_z)
    
        ndepth = len(z) # number of depth indices
        sig_tot_z = np.zeros(ndepth) # initialize vertial total stress array
        # loop to calculate total stress
        for i in range(ndepth):
            if i == 0:
                sig_tot_z[i] = z[i]*gamma_z[i] # total stress for first depth
            else:
                sig_tot_z[i] = sig_tot_z[i-1] + (z[i]-z[i-1])*gamma_z[i] # total stress for subsequent depths
    
    # calculate pore pressure and effective stress
    u_z = np.asarray(gamma_w*(z-dtw)) # hydrostatic pore pressure
    u_z = np.maximum(u_z,np.zeros(u_z.shape)) # set minimum of pore pressure to 0 (no suction)
    sig_eff_z = sig_tot_z - u_z # vertical effective stress
    
    #
    return sig_tot_z, u_z, sig_eff_z
    
    
# -----------------------------------------------------------
def get_Vs_zmax(**kwargs):
    """
    Computes the average Vs over a depth of **zmax** (e.g., vs30) by zmax dividing by the total travel time to depth of **zmax** (total travel time = sum(dh_i/vs_i), where dh_i = layer thickness, and vs_i = layer's shear wave velocity)
    
    Parameters
    ----------
    vs_arr : float, array
        [m/sec] shear wave velocity profile
    z_bot_arr : float, optional
        [m] depth to bottom of each transition of shear wave velocity
    zmax : float
        [m] maximum depth to compute mean shear wave velocity over, default to 30 m
        
    Returns
    -------
    vs_zmax = float
        [m/sec] average slowness down to a depth o **zmax**
    
    """
    
    # get inputs
    zmax = kwargs.get('zmax',30) # target depth to compute Vs over, default to 30 m
    vs_arr = kwargs.get('vs_arr',18) # shear wave velocity profile
    z_bot_arr = kwargs.get('z_bot_arr',0) # depths to bottom of traisition of vs
    
    # calculate Vs over depth zmax (average slowness)
    nLayers = len(vs_arr) # number of layers
    z_bot_arr = np.hstack([0,z_bot_arr]) # pad 0 to depth array for interpolation
    dz_layer = [z_bot_arr[i+1]-z_bot_arr[i] for i in range(nLayers)] # get layer thicknesses
    t_tot = sum(np.divide(dz_layer,vs_arr)) # calculate layer travel time and sum up total time to travel to target depth
    vs_zmax = zmax/t_tot # calculate Vs (average slowness) over target depth given total travel time
    
    #
    return vs_zmax
    
    
# -----------------------------------------------------------
def get_rd(**kwargs):
    """
    Calculates the depth-reduction factor **rd**
    
    """
    
    # Current methods coded
    # method 1 = Youd et al. (2001) NCEER
    # method 2 = Cetin (2000) Dissertation - Reliability-based assessment of seismic soil liquefaction initiation hazard (used in Moss et al. CPT liq)
    # method 3 = Cetin et al. (2004) - SPT-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential
    # method 4 = Idriss (1999), used in Idriss and Boulanger (2008, 2012) and Boulanger and Idriss (2014, 2016)
    method = kwargs.get('method','idriss') # meters, depth value or array of values, default to Idriss (1999)
    
    # get inputs
    z = kwargs.get('z',None) # meters, depth value or array of values
    # convert to numpy array for easier computation
    if type(z) is not np.array:
        z = np.asarray(z)
    
    #
    if method == 'youd_etal_2001': # Youd et al. (2001)
    
        # calculate rd(z)
        rd = np.asarray((1.000 - 0.4113*z**0.5 + 0.04052*z + 0.001753*z**1.5) /\
                        (1.000 - 0.4177*z**0.5 + 0.05729*z - 0.006205*z**1.5 + 0.001210*z**2))
                        
        # check to make sure nothing is over 1
        rd = np.minimum(rd,np.ones(rd.shape))
        
    #
    elif method == 'cetin_2000': # Cetin (2000), no Vs term compared to Cetin et a. (2004)
    
        # get additional inputs
        M = kwargs.get('M',None) # moment magnitude
        amax = kwargs.get('amax',None) # g, peak surface acceleration
        
        # initialize arrays
        rd = []
        sigma_rd = []
        
        # loop through depths
        for d in z:
        
            # calculate sigma(z)
            if d < 12.2: # eq. 10 in Moss et al. (2006)
                temp_sigma_rd = (d*3.28)**0.864 * 0.00814
            elif d >= 12.2: # eq. 11 in Moss et al. (2006)
                temp_sigma_rd = 40**0.864 * 0.00814
            
            # calculate rd(z)
            sigma_rd.append(temp_sigma_rd)
            if d < 20: # eq. 8 in Moss et al. (2006)
                rd.append((1 + (-9.147 - 4.173*amax + 0.652*M) / \
                            (10.567 + 0.089*np.exp(0.089*(-d*3.28 - 7.760*amax + 78.576)))) / \
                        (1 + (-9.147 - 4.173*amax + 0.652*M) / \
                            (10.567 + 0.089*np.exp(0.089*(-7.760*amax + 78.576)))))
            elif d >= 20: # eq. 9 in Moss et al. (2006)
                rd.append((1 + (-9.147 - 4.173*amax + 0.652*M) / \
                            (10.567 + 0.089*np.exp(0.089*(-d*3.28 - 7.760*amax + 78.576)))) / \
                        (1 + (-9.147 - 4.173*amax + 0.652*M) / \
                            (10.567 + 0.089*np.exp(0.089*(-7.760*amax + 78.576)))) - \
                        0.0014*(d*3.28 - 65))
        
        # convert to numpy arrays
        rd = np.asarray(rd)
        sigma_rd = np.asarray(sigma_rd)
    
    #
    elif method == 'cetin_etal_2004': # Cetin et al. (2004)
    
        # get additional inputs
        M = kwargs.get('M',None) # moment magnitude
        amax = kwargs.get('amax',None) # g, peak surface acceleration
        Vs12 = kwargs.get('Vs12',None) # m/s, Vs in the upper 12 m (40 ft)
        
        # initialize arrays
        rd = []
        sigma_rd = []
        
        # loop through depths
        for d in z:
            # calculate sigma(z)
            if d >= 12: # eq. 8 in Cetin et al. (2004)
                temp_sigma_rd = 12**0.8500 * 0.0198
            elif d < 12: # eq. 8 in Cetin et al. (2004)
                temp_sigma_rd = d**0.8500 * 0.0198
            
            # calculate rd(z)
            sigma_rd.append(temp_sigma_rd)
            if d < 20: # eq. 8 in Cetin et al. (2004)
                rd.append((1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
                            (16.258 + 0.201*np.exp(0.341*(-d + 0.0785*Vs12 + 7.586)))) / \
                        (1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
                            (16.258 + 0.201*np.exp(0.341*(0.0785*Vs12 + 7.586)))))
            elif d >= 20: # eq. 8 in Cetin et al. (2004)
                rd.append((1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
                            (16.258 + 0.201*np.exp(0.341*(-20 + 0.0785*Vs12 + 7.586)))) / \
                        (1 + (-23.013 - 2.949*amax + 0.999*M + 0.0525*Vs12) / \
                            (16.258 + 0.201*np.exp(0.341*(0.0785*Vs12 + 7.586)))) - \
                        0.0046*(d-20))
        
        # convert to numpy arrays
        rd = np.asarray(rd)
        sigma_rd = np.asarray(sigma_rd)
    
    #
    elif method == 'idriss_1999': # Idriss (1999)
    
        # get additional inputs
        M = kwargs.get('M',None) # moment magnitude
        
        # check if M is given
        if M is None:
            print('Idriss (1999) requires M as input; return without calculating rd')
        
        else:
            # calculate rd
            alpha = -1.012 - 1.126*np.sin(z/11.73 + 5.133) # eq. 3b in Boulanger and Idriss (2016)
            beta = 0.106 + 0.118*np.sin(z/11.28 + 5.142) # eq. 3c in Boulanger and Idriss (2016)
            rd = np.exp(alpha + beta*M) # eq. 3a in Boulanger and Idriss (2016)
            rd = np.minimum(rd,np.ones(rd.shape)) # check to make sure nothing is over 1
            
            # check if i is over 20 meters
            for i in z:
                if i > 20:
                    print('Z exceeds 20 m, the maximum recommended depth for this correlation (Idriss and Boulanger, 2008)')
                    print('--> Consider site response analysis for stress reduction factor')
                    break
    #
    else: # requests for other methods
        rd = None
    
    #
    return rd
    
    
# -----------------------------------------------------------
def get_Ia(t, acc, gval=9.81):
    """
    Computes the arias intensity, Ia, for a given acceleration time history.
    
    Parameters
    ----------
    t : float, array
        [sec] time
    acc : float, array
        [g] acceleration
    gval : float, optional
        [g] gravitational acceleration, default = 9.81 m/s^2
        
    Returns
    -------
    Tm = float
        [sec] mean period
    
    """
    
    # Determine time step of array 
    dt = [t[i+1]-t[i] for i in range(len(t)-1)] # sec
    
    # Pad 1 to beginning of dt array for index multiplication of vectors
    dt = np.asarray(np.hstack([1,dt])) # sec
    
    # Multiply indices of dt and acc array
    Ia = np.asarray([abs(acc[i])**2 * dt[i] for i in range(len(acc))]) # m/s^2 * m/s^2 * sec = m^2/s^3
    
    # Sum up all the indices to get Ia
    Ia = np.asarray([sum(Ia[0:i]) for i in range(len(Ia))]) * np.pi/2/gval # m^2/s^3 / m/s^2 = m/s
    
    #
    return max(Ia)
    
    
# -----------------------------------------------------------
def get_Tm(t,y):
    """
    Computes the mean period, **Tm**, a measure of the mean frequency content in the ground motion record; used in Saygili (2008). A Fourier Transform is applied a time history. The frequencies between 0.25 and 25 Hz are weighted by the square of the respective FFT amplitudes and then summed. The sum is divided by the sum of the weights (i.e., square of FFT amplitudes) to obtain the mean period, **Tm**.
    
    Parameters
    ----------
    t : float, array
        [sec] time
    y : float, array
        [varies] y values
        
    Returns
    -------
    Tm = float
        [sec] mean period
    
    References
    ----------
    .. [1] Saygili, G., 2008, A Probabilistic Approach for Evaluating Earthquake-Induced Landslides, PhD Thesis, Universtiy of Texas at Austin.
    
    """
    
    # get FFT on time history
    n = len(t) # length of time history
    dt = t[1]-t[0] # # sec, time step
    f = fft.fftfreq(n,d=dt) # Hz, frequency array
    y_fft = fft.fft(y) # Fourier transform
    
    # Determine number of points to Nyquist (odd versus even number for length of record)
    if np.mod(n,2) == 0:
        mid_pt = int(n/2)
    else:
        mid_pt = int((n-1)/2+1)
        
    # Amplitude of FFT
    y_fft_amp = np.abs(y_fft)
    
    # Calculate Tm discretely by evaluating the numerator and the denominator,
    numer = sum([y_fft_amp[i]**2/f[i] for i in range(mid_pt) if f[i] >= 0.25 and f[i] <= 20]) # 1/Hz = sec
    denom = sum([y_fft_amp[i]**2 for i in range(mid_pt) if f[i] >= 0.25 and f[i] <= 20])
    
    # get Tm
    Tm = numer/denom # sec
    
    #
    return Tm
    
    
# -----------------------------------------------------------
def get_total_settlement(dh,eps_v,flag_DF=True,z_cr=18):
    """
    Computes total settlement for a 1D soil column given the volumetrain strain in each layer.
    
    Parameters
    ----------
    dh : float, array
        [m] thickness of each layer
    eps_v : float, array
        [%] volumetric strain at each layer
    flag_DF : boolean, optional
        flag for calculation of depth-weight factor, to increase settlement contribution from depths less than **z_cr** and reduce contribution from depths greater than **z_cr** (cetin et al., 2009), default = True
    z_cr : float, optional
        [m] critical depth for where depth-weighted factor is equal to 1 (cetin et al., 2009), default = 18 m
        
    Returns
    -------
    s_sum = float
        [m] cumulative ground settlement
    
    
    References
    ----------
    .. [1] Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 3, pp. 387-398.
    
    """
    
    # calculate depth to bottom and middle of each layer
    z_bot = np.asarray([sum(dh[0:i]) for i in range(len(dh))])
    z_mid = np.asarray([z_bot[i]/2 if i == 0 else (z_bot[i-1]+z_bot[i])/2 for i in range(z_bot)])
    
    # maximum depth
    h_sum = z_bot[len(z_bot)]
    
    # calculate depth-weighted factor (Cetin et al., 2009)
    if flag_DF is True:
        Df = 1-z_mid/z_cr
    else:
        DF = np.ones(z_bot.shape)
    
    # calculate total volumetric strain and settlement
    numer = sum([eps_v[i]*dh[i]*DF[i] for i in range(len(dh))])
    denom = sum([dh[i]*DF[i] for i in range(len(dh))])
    eps_v_sum = numer/denom
    s_sum = eps_v_sum/100*h_sum
    
    return eps_v_sum, s_sum
    
    
# -----------------------------------------------------------
# def get_ky(slope_type,**kwargs):
def get_ky(**kwargs):
    """
    Various methods to calculate the yield acceleration:
    
    1. Newmark sliding **block**
    2. **Rock**-slope failures
    3. Disrupted soil slides (**infinite slope**)
    4. Coherent **rotational** slides
    5. Simplified method (Bray and Travasarou, 2009)
    
    Parameters
    ----------
    slope_type : str
                type of slope to assess; enter any of the bolded words shown in the above list of failure modes (i.e., **rock**, **infinite slope**, or **rotational**)
    phi : float
        [degree] friction angle
    c : float
        [kPa] cohesion
    beta : float
        [degree] slope angle
    gamma : float
            [kN/m^3] unit weight
    
    Additional parameters specific to sliding block:
    Incomplete
        
    Additional parameters specific to rock-slope failures:
    H : float
        [m] height of slope (local relief)
    
    Additional parameters specific to disrupted soil slides:
    cr : float
        [kPa] root cohesion
    t : float
        [m] thickness of failure mass
    m : float
        [%] percent of failure thickness that is saturated
    gamma_w : float
            [kN/m^3] unit weight of water, default = 9.81 kN/m^3
    method : str
        slight difference in form between **Bray** (2007), **Grant** et al. (2016), and **Rathje** & Antonakos (2011), specify author name
    
    Parameters for Coherent Rotational Slides (failure_mode == 3):
    H : float
        [m] height of slope (local relief)
    y : float
        [m] width of cylindrical pixel
    
    Returns
    -------
    ky : float
        [g] yield acceleration (see *return_param* under "Parameters")
    FS : float
        factor of safety (if calculated explicitly)
    
    References
    ----------
    .. [1] Grant, A., Wartman, J., and Abou-Jaoude, G., 2016, Multimodal Method for Coseismic Landslide Hazard Assessment, Engineering Geology, vol. 212, pp. 146-160.
    .. [2] Keefer, D.K., 1984., Landslides Caused by Earthquakes, Geological Society of America Bulletin, vol. 95, no. 4, pp. 406-421.
    .. [3] Newmark, N.M., 1965, Effects of Earthquakes on Dams and Embankments, Geotechnique, vol. 15, pp. 139–159.
    .. [4] Rathje, E.M., and Saygili, G., 2009, Probabilistic Assessment of Earthquake-Induced Sliding Displacements of Natural Slopes, Bulletin of the New Zealand Society for Earthquake Engineering, vol. 42, no. 1, pp. 18-27.
    .. [5] Bray, J.D., and Travasarou, T., 2009, Technical Notes: Pseudostatic Coefficient for Use in Simplified Seismic Slope Stability Evaluation, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 9, pp. 1336-1340.
    .. [6] Bray, J.D., 2007, Simplified Seismic Slope Displacement Procedures, Earthquake Geotechnical Engineering, Springer, Dordrecht, pp. 327-353.
    
    """
    
    # get params
    n_site = kwargs.get('n_site') # number of sites
    gamma_w = kwargs.get('gamma_w',9.81) # kN/m3, unit weight of water
    phi = kwargs.get('FrictionAngle',np.zeros(n_site)) # deg, friction angle
    c = kwargs.get('Cohesion',np.zeros(n_site)) # kPa, cohesion
    gamma = kwargs.get('UnitWeight',np.ones(n_site)*20) # kN/m3, unit weight
    H = kwargs.get('SlopeThickness',np.ones(n_site)) # m, local hillside relief or height (thickness) of slope
    method = kwargs.get('MethodForKy','Bray') # Grant or Bray form for calculating ky
    #
    slope_angle = kwargs.get('SlopeAngle',np.zeros(n_site)) # deg, slope angle
    if isinstance(slope_angle,str):
        # if slope_angle == 'Preferred':
        slope_angle = np.zeros(n_site)
    
    # Slope type specific parameters
    # 1) sliding block
    FS_block = kwargs.get('FactorOfSafety',None) # factor of safety, either provided or computed below
    # 2) Rock-slope or sliding block failures:
    # 3) Disrupted soil slides (infinite slope):
    # --- Grant et al. (2016)
    cr = kwargs.get('RootCohesion',np.zeros(n_site)) # kPa, root cohesion, see Grant et al. (2016)
    if isinstance(cr,str):
        # if cr == 'Preferred':
        cr = np.zeros(n_site)
    # --- Rathje & Saygili (2009)
    m = kwargs.get('SlopeSaturation',np.zeros(n_site)) # %, percent of failure thickness that is saturated, see Rathje & Saygili (2009)
    if isinstance(m,str):
        # if m == 'Preferred':
        m = np.zeros(n_site)
    # 4) Coherent rotational slides (deep):
    # --- Grant et al. (2016)
    y = kwargs.get('PixelWidth',np.zeros(n_site)) # m, cylindrical width of pixel
    # --- Bray (2007)
    S1 = kwargs.get('UpslopeNormHorizLength',np.ones(n_site)) # upslope horizontal length, normalized to H (S1 = 1 for 1H:1V)
    S2 = kwargs.get('DownslopeNormHorizLength',np.ones(n_site)) # downslope horizontal length, normalized to H (S2 = 1 for 1H:1V)
    L = kwargs.get('SlideMassLength',np.ones(n_site)) # length of sliding mass
    
    # avoid tan(0 deg)
    slope_angle[slope_angle==0] = 0.1
    phi[phi==0] = 0.1
    # slope_angle[slope_angle==9999] = 0.1
    
    # convert angles to radians
    phi_rad = np.radians(phi) # friction angle in radians
    slope_angle_rad = np.radians(slope_angle) # slope angle in radians
    phi_slope_diff_rad = np.radians(phi-slope_angle) # difference between phi and slope angle, in radians
    
    # extract input params
    slope_type = kwargs.get('SlopeType','infinite')
    # if isinstance(slope_type,str) and 'Preferred' in slope_type:
    if 'Preferred' in slope_type:
        slope_type = np.array(['infinite']*n_site) # preferred slope type
    # elif isinstance(slope_type,np.ndarray):
    else:
        slope_type = np.array([item.lower() for item in slope_type])

    # initialize ky
    ky = np.zeros(n_site)
    
    # -----------------------------------------------------------
    # 1) Sliding block:
    # if 'block' in slope_type.lower():
    i_block = [i for i,val in enumerate(slope_type) if 'block' in val]
    if len(i_block) > 0:
        # calculate yield acceleration
        ky_block = (FS_block-1)*np.sin(slope_angle_rad) # eq. 1 in Jibson (2007)
        ky[i_block] = ky_block[i_block]
    
    # -----------------------------------------------------------
    # 2) Rock-slope or sliding block failures:
    # elif 'rock' in slope_type.lower():
    i_rock = [i for i,val in enumerate(slope_type) if 'rock' in val]
    if len(i_rock) > 0:
        # intermediate calculations
        alpha = (slope_angle + phi)/2 # rad, slope's critical angle
        h = 0.25*H # m, vertical height of failure mass
        alpha_rad = np.radians(alpha) # alpha in radians
        slope_alpha_diff_rad = np.radians(slope_angle-alpha) # difference slope angle and alpha, in radians
        # calculate factor of safety
        FS = 2*c*np.sin(slope_angle_rad)/(gamma*h*np.sin(slope_alpha_diff_rad)*np.sin(alpha_rad)) +\
            np.tan(phi_rad)/np.tan(alpha_rad)
            
        # calculate yield acceleration
        ky_rock = (FS-1)*np.sin(alpha_rad) # g
        ky[i_rock] = ky_rock[i_rock]
    
    # -----------------------------------------------------------
    # 3) Disrupted soil slides (infinite slope):
    # elif 'infinite' in slope_type.lower() or 'slope' in slope_type.lower():
    i_inf = [i for i,val in enumerate(slope_type) if 'inf' in val]
    if len(i_inf) > 0:
        # check method for ky
        if 'bray' in method.lower():
            ky_inf = np.tan(phi_slope_diff_rad) + \
                c / (gamma*H*(np.cos(slope_angle_rad)**2 * \
                (1+np.tan(phi_rad)*np.tan(slope_angle_rad))))
        else:
            # print(c.shape)
            # print(cr.shape)
            # print(gamma.shape)
            # print(H.shape)
            # print(slope_angle_rad.shape)
            # calculate factor of safety
            FS = (c+cr)/(gamma*H*np.sin(slope_angle_rad)) + \
                np.tan(phi_rad)/np.tan(slope_angle_rad) - \
                gamma_w*m/100*np.tan(phi_rad) / \
                (gamma*np.tan(slope_angle_rad)) # factor of safety
            # calculate yield acceleration, form depends if phi is used (depends on method)
            if 'grant' in method.lower():
                # simplest form for ky without internal friction (used by Grant et al., 2016)
                ky_inf = (FS-1)*np.sin(slope_angle_rad) # g
            # using Rathje and Saygili (2009)
            elif 'rathje' in method.lower():
                # eq. 1
                ky_inf = (FS-1) / (np.cos(slope_angle_rad)*np.tan(phi_rad)+1/np.tan(slope_angle_rad)) # g
        ky[i_inf] = ky_inf[i_inf]
    
    # -----------------------------------------------------------
    # 4) Coherent rotational slides (deep):
    # elif 'rotational' in slope_type.lower():
    i_rot = [i for i,val in enumerate(slope_type) if 'rot' in val]
    if len(i_rot) > 0:
        # check method for ky
        if 'grant' in method.lower():
            # intermediate calculations
            R = 1.5*H # m, radius of circular failure plane acting through a dry homogeneous hillslope
            delta = np.arcsin(1/(3*np.sin(slope_angle_rad))) # rad, for slope_angle > 20 degrees
            L = 2*delta*R # m, failure plane length
            a = (4*R*(np.sin(delta))**3 / (3*(2*delta-np.sin(2*delta))) - R) * np.cos(delta) # m, landslide body's centroid
            W = 1/2*gamma*y*R**2*(2*delta - np.sin(2*delta)) # kN
            # calculate yield acceleration
            ky_rot = (c*L*y + W*(np.cos(slope_angle_rad)*np.tan(phi_rad) - np.sin(slope_angle_rad))) / \
                (W*(a/R + np.sin(slope_angle_rad)*np.tan(phi_rad))) # g
        elif 'bray' in method.lower():
            # intermediate calculations
            theta_1_rad = np.arctan(1/S1)
            FS = np.tan(phi_rad)*(S1*H/2*(np.cos(slope_angle_rad))**2+L+S2*H/2) / (np.cos(theta_1_rad)*np.sin(theta_1_rad)*S1*H/2)
            ky_rot = (FS-1)*(np.cos(theta_1_rad)*np.sin(theta_1_rad)*S1*H/2) / (H*(S1+S2)/2+L)
        ky[i_rot] = ky_inf[i_rot]
    
    # 
    return ky
    

# -----------------------------------------------------------
def get_Ts(H,vs,nDim):
    """
    Calculates the site period, **Ts**. For 1D, Ts = 4*H/vs; for 2D, Ts = 2.6*H/vs
    
    Parameters
    ----------
    H : float
        [m] slope/structure height
    vs : float
        [m/s] shear wave velocity
    nDim : int
        **1** (trapezoidal) or **2** (triangular) dimension for calculating **Ts**; default = 2
        
    Returns
    -------
    Ts : float
        [sec] site period
        
    References
    ----------
    .. [1] Bray, J.D., and Travasarou, T., 2007, Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 133, no. 4, pp. 381-392.
    
    """
    
    if nDim == 1:
        Ts = 4*H/vs
    elif nDim == 2:
        Ts = 2.6*H/vs
        
    #
    return Tm