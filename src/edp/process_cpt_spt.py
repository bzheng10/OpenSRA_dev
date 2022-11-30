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
# Python base modules
import logging
import os
import sys

# scientific processing modules
import numpy as np
from pandas import read_csv

# geospatial processing modules
from pyproj import Transformer
from geopandas import GeoDataFrame, points_from_xy

# precompiling
from numba import njit, float64
from numba.types import Tuple, UniTuple

# OpenSRA modules and functions
from src.site.site_util import make_grid_nodes


# -----------------------------------------------------------
def make_def_poly_from_cpt_spt(
    
):
    """uses CPT/SPT data to make deformation polygon"""
    pass


def read_cpt_data(cpt_base_dir):
    '''
    Read the CPT data with csv format only.
        Column 0: Depth (m)
        Column 1: Measured qc (MPa)
        Column 2: Measured fs (MPa)
        Column 3: Measured Pore Pressure
    Note: Every row must contain valid data
    '''
    # check if paths are valid
    if not os.path.exists(cpt_base_dir):
        raise ValueError("Path to CPT root folder is not valid")
    cpt_data_dir = os.path.join(cpt_base_dir,'CPTs')
    if not os.path.exists(cpt_data_dir):
        raise ValueError(r'Path to CPT data folder (root/CPTs/) is not valid')
    # get list of CPT files
    cpt_data_files = os.listdir(cpt_data_dir)
    # create transformers for transforming coordinates
    epsg_wgs84 = 4326 # degrees
    epsg_utm_zone10 = 32610 # meters
    transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
    transformer_utmzone10_to_wgs84 = Transformer.from_crs(epsg_utm_zone10, epsg_wgs84)
    # read CPT metadata with locations
    cpt_meta_fpath = os.path.join(cpt_base_dir,'Summary.csv')
    cpt_meta = read_csv(cpt_meta_fpath)
    cpt_meta['utm_x'], cpt_meta['utm_y'] = \
        transformer_wgs84_to_utmzone10.transform(
            cpt_meta['Latitude'].values,
            cpt_meta['Longitude'].values,
        )
    # number of CPTs
    n_cpt = cpt_meta.shape[0]
    # create GeoDataFrames for geospatial processing
    cpt_meta_utm = GeoDataFrame(
        cpt_meta.copy(),
        geometry=points_from_xy(cpt_meta.utm_x.values,cpt_meta.utm_y.values,),
        crs=epsg_utm_zone10
    )
    cpt_meta_wgs84 = GeoDataFrame(
        cpt_meta.copy(),
        geometry=points_from_xy(cpt_meta.Longitude.values,cpt_meta.Latitude.values,),
        crs=epsg_wgs84
    )
    # read CPT files in a folder
    cpt_data = [read_csv(os.path.join(cpt_data_dir,each)) for each in cpt_data_files]
    cpt_file_headers = list(cpt_data[0].columns)
    if 'qt' in cpt_file_headers:
        cpt_file_headers = list(map(lambda x: x.replace('qt','qc'), cpt_file_headers))
    if 'u2' in cpt_file_headers:
        cpt_file_headers = list(map(lambda x: x.replace('u2','u'), cpt_file_headers))
    # clean-up and preprocessing
    for each in cpt_data:
        # change header from qt -> qc
        each.columns = cpt_file_headers
        # set qc >= 0.001
        each.qc = each.qc.apply(lambda x: max(x, 0.001))
        # set fs >= 0.001
        each.fs = each.fs.apply(lambda x: max(x, 0.001))
    # return
    return cpt_meta_wgs84, cpt_meta_utm, cpt_data, n_cpt


def preprocess_for_hull(pgdef_val, grid_node_gdf, col_with_aspect, pgdef_cutoff=0.05):
    """preprocess for getting deformation polygon"""
    ind_cond = pgdef_val>pgdef_cutoff
    geom = grid_node_gdf.geometry.copy()
    geom_cond = geom[ind_cond]
    gdf_for_hull = GeoDataFrame(
        np.vstack([
            pgdef_val[ind_cond],
            grid_node_gdf[col_with_aspect][ind_cond].values
        ]).T,
        columns=['pgdef_m','aspect'],
        crs=4326,
        geometry=geom_cond.values
    )
    return gdf_for_hull


@njit(
    UniTuple(float64,2)(float64),
    fastmath=True,
    cache=True
)
def get_azimuth_comp(angle):
    """get x and y component given azimuth angle"""
    angle_rad = np.radians(angle)
    x_comp = np.sin(angle_rad)
    y_comp = np.cos(angle_rad)
    return x_comp, y_comp


@njit(
    float64[:,:](float64[:],float64,float64),
    fastmath=True,
    cache=True
)
def make_dir_vect(start_loc, angle, length=1):
    """make direction vector given start location, angle, and length"""
    angle_comp = get_azimuth_comp(angle)
    dir_vect = np.zeros((2,2))
    dir_vect[0,:] = start_loc
    dir_vect[1,0] = start_loc[0]+length*angle_comp[0]
    dir_vect[1,1] = start_loc[1]+length*angle_comp[1]
    return dir_vect


@njit(
    float64[:,:](float64[:,:],float64[:,:]),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_shear_strain_ZhangEtal2004(fs_liq, rel_dens):
    """
    Maximum shear strain based on Zhang et al. (2004)
    Parameters:
        fs_liq: factor of safety against liquefaction triggering
        rel_dens: relative density
    Returns:
        gamma_max: Maximum shear strain (%)
    """
    # get dimensions
    n_depth = fs_liq.shape[0]
    n_sample = fs_liq.shape[1]
    # initialize gamma_max array
    gamma_max = np.zeros(fs_liq.shape)
    # loop through domain vector
    for i in range(n_depth):
        fs_liq_i = fs_liq[i]
        rel_dens_i = rel_dens[i]
        #
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=85.0,rel_dens_i<95.0),
            fs_liq_i<=0.7
        )
        gamma_max[i][cond] = 6.2
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=85.0,rel_dens_i<95.0),
            np.logical_and(fs_liq_i>0.7,fs_liq_i<=2.0)
        )
        gamma_max[i][cond] = 3.26 * (fs_liq_i[cond] ** -1.8)
        #
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=75.0,rel_dens_i<85.0),
            fs_liq_i<=0.56
        )
        gamma_max[i][cond] = 10.0
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=75.0,rel_dens_i<85.0),
            np.logical_and(fs_liq_i>0.56,fs_liq_i<=2.0)
        )
        gamma_max[i][cond] = 3.22 * (fs_liq_i[cond] ** -2.08)
        #
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=65.0,rel_dens_i<75.0),
            fs_liq_i<=0.59
        )
        gamma_max[i][cond] = 14.5
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=65.0,rel_dens_i<75.0),
            np.logical_and(fs_liq_i>0.59,fs_liq_i<=2.0)
        )
        gamma_max[i][cond] = 3.20 * (fs_liq_i[cond] ** -2.89)
        #
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=55.0,rel_dens_i<65.0),
            fs_liq_i<=0.66
        )
        gamma_max[i][cond] = 22.7
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=55.0,rel_dens_i<65.0),
            np.logical_and(fs_liq_i>0.66,fs_liq_i<=2.0)
        )
        gamma_max[i][cond] = 3.58 * (fs_liq_i[cond] ** -4.42)
        #
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=45.0,rel_dens_i<55.0),
            fs_liq_i<=0.72
        )
        gamma_max[i][cond] = 34.1
        cond = np.logical_and(
            np.logical_and(rel_dens_i>=45.0,rel_dens_i<55.0),
            np.logical_and(fs_liq_i>0.72,fs_liq_i<=2.0)
        )
        gamma_max[i][cond] = 4.22 * (fs_liq_i[cond] ** -6.39)
        #
        cond = np.logical_and(rel_dens_i<45.0,fs_liq_i<=0.81)
        gamma_max[i][cond] = 51.2
        cond = np.logical_and(
            rel_dens_i<45.0,
            np.logical_and(fs_liq_i>0.81,fs_liq_i<=1.0)
        )
        gamma_max[i][cond] = 250 * (1.0 - fs_liq_i[cond]) + 3.5
        cond = np.logical_and(
            rel_dens_i<45.0,
            np.logical_and(fs_liq_i>1.0,fs_liq_i<=2.0)
        )
        gamma_max[i][cond] = 3.31 * (fs_liq_i[cond] ** -7.97)
    # limit to 1e-5% strain
    # gamma_max = np.maximum(gamma_max,1e-5)
    # return
    return gamma_max


@njit(
    float64[:,:](float64[:,:],float64[:,:]),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_shear_strain_IdrissBoulanger2008(fs_liq, rel_dens):
    """
    Maximum shear strain prediction based on Idriss & Boulanger (2008)
    Parameters:
        fs_liq: factor of safety against liquefaction triggering
        rel_dens: relative density
    Returns:
        gamma_max_IB: Maximum shear strain (%)
    """
    # get dimensions
    n_depth = fs_liq.shape[0]
    n_sample = fs_liq.shape[1]
    # # intermediate params
    # gamma_lim = np.maximum(0, np.minimum(50, (1.859 * (1.1 - (rel_dens / 100)) ** 3) * 100))
    #     gamma_lim[rel_dens>=95.0] = 0
    #     gamma_lim[fs_liq>=95.0] = 0
    # initialize
    f_alpha = np.zeros(fs_liq.shape)
    gamma_max = np.zeros(fs_liq.shape)
    for i in range(n_depth):
        rel_dens_i = rel_dens[i]
        fs_liq_i = fs_liq[i]
        # intermediate calcs
        f_alpha[i] = 0.032 + 4.7 * (rel_dens_i / 100) - 6 * (rel_dens_i / 100) ** 2
        f_alpha[i][rel_dens_i<40] = 0.032 + 4.7 * 0.4 - 6 * 0.4 ** 2
        #
        gamma_lim_i = np.maximum(0, np.minimum(50, (1.859 * (1.1 - (rel_dens_i / 100)) ** 3) * 100))
        gamma_lim_i[rel_dens_i>=95.0] = 0
        gamma_lim_i[fs_liq_i>=2.0] = 0
        #
        cond = np.logical_and(fs_liq_i<2,fs_liq_i>f_alpha[i])
        gamma_max[i][cond] = np.minimum(
            gamma_lim_i[cond],
            np.maximum(0, 0.035 * (2 - fs_liq_i[cond]) * ((1 - f_alpha[i][cond]) / (fs_liq_i[cond] - f_alpha[i][cond])) * 100)
        )
        #
        gamma_max[i][fs_liq_i <= f_alpha[i]] = gamma_lim_i[fs_liq_i <= f_alpha[i]]
    # limit to 1e-5% strain
    # gamma_max = np.maximum(gamma_max,1e-5)
    #
    return gamma_max


@njit(
    float64[:,:](float64[:,:],float64[:,:],float64),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_cpt_based_shear_strain(fs_liq, rel_dens, weight_z04=0.5):
    """
    Max shear strain weighted between Zhang et al. (2004) and Idriss & Boulanger (2008)
    Parameters:
        fs_liq: factor of safety against liquefaction triggering
        rel_dens: relative density
        weight_z04: weight for ZhangEtal04; weight for IB08 = 1 - weight for ZhangEtal04
    Returns:
        gamma_max: Maximum shear strain (%)
    """
    # get maximum shear strains (%) for each method
    weight_ib08 = 1 - weight_z04
    if weight_z04 > 0:
        gamma_max_ZhangEtal2004 = get_shear_strain_ZhangEtal2004(
            fs_liq, rel_dens
        )
    else:
        gamma_max_ZhangEtal2004 = np.zeros(fs_liq.shape)
    if weight_ib08 > 0:
        gamma_max_IdrissBoulanger2008 = get_shear_strain_IdrissBoulanger2008(
            fs_liq, rel_dens
        )
    else:
        gamma_max_IdrissBoulanger2008 = np.zeros(fs_liq.shape)
    # get weighted gamma_max
    gamma_max = \
        gamma_max_ZhangEtal2004 * weight_z04 + \
        gamma_max_IdrissBoulanger2008 * weight_ib08 # %
    # return
    return gamma_max


@njit(
    float64[:,:](float64[:,:],float64[:,:],float64),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_cpt_based_vol_strain(fs_liq, rel_dens, weight_z04=0.5):
    """
    Volumetric strain prediction based on Zhang et al. (2002)
    Additional references: Idriss & Boulanger (2008), Ishihara & Yoshimine (1992) and Yoshimine et al. (2006)
    Parameters:
        fs_liq: factor of safety against liquefaction triggering
        rel_dens: relative density
        weight_z04: weight for ZhangEtal04; weight for IB08 = 1 - weight for ZhangEtal04
    Returns:
        eps_vol: Volumetric strain (%)
    """
    # get dimensions
    n_depth = fs_liq.shape[0]
    n_sample = fs_liq.shape[1]
    # get gamma_max from Zhang et al. (2004) and Idriss & Boulanger (2008)
    gamma_max = get_cpt_based_shear_strain(fs_liq, rel_dens, weight_z04)
    # apply upper limits of 8% on gamma_max
    for j in range(n_sample):
        gamma_max[:,j] = np.minimum(8,gamma_max[:,j]) # %, eq. 95 in IB08
    # get volumetric strain
    eps_vol = 1.5*np.exp(-2.5*rel_dens/100)*gamma_max # %, eq. 95 in IB08
    #
    return eps_vol


@njit(
    float64[:,:](
        float64[:],float64[:,:],float64[:,:],float64
    ),
    fastmath=True,
    cache=True,
    # parallel=True
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
        rel_dens: Relative density in percent
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
            counter = 1
            while True:
                dr_iter_i_j = dr_calc_i_j
                m_i_j = 0.784 - 0.521 * dr_iter_i_j
                cn_i_j = np.minimum(1.7, (pa / eff_sig_v0_i_j) ** (m_i_j))
                qtn_i_j = np.maximum(0.2, ((qc_i * 1000 - tot_sig_v0_i_j) / pa) * cn_i_j)
                dr_calc_i_j = 0.478 * ((qtn_i_j) ** 0.264) - 1.063
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
    rel_dens = (w_IB * dr_IB + w_Jam * dr_Jam + w_KM * dr_KM)
    return rel_dens


@njit(
    Tuple((float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:]))(
        float64[:],float64[:],float64[:,:],float64[:,:],float64
    ),
    fastmath=True,
    cache=True,
    # parallel=True
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
    Tuple((float64[:,:],float64[:,:]))(
        float64[:],float64[:],float64[:],float64[:,:],float64[:,:],
        float64[:,:],float64[:,:],float64[:,:],
        float64,float64,
        float64
    ),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_fs_liq_Robertson2009(
    qc, fs, z, tot_sig_v0, eff_sig_v0,
    qtn, fn, ic,
    pga, mag,
    pa=101.3
):
    """
    Return the factor of safety against liquefaction triggering based on Robertson (2009)
    Implements probabilistic Robertson & Wride procedure from Ku et al. (2012)
    Parameters:
        qc: measured tip resistance (MPa)
        fs: measured sleeve resistance (MPa)
        z: depth (m)
        tot_sig_vo: total stress at specific depth (kPa)
        eff_sig_vo: effective stress at specific depth (kPa)
        qtn: noormalized, dimensionless tip resistance (1 atm & overburden)
        fn: normalized friction ratio (%)
        ic: SBT Index, Ic Robertson (2009)
        mag: earthquake moment magnitude
        pga: peak ground acceleration at the ground surface
        pa: reference pressure 1 atm = 101.3 kPa
    Return:
        fs_liq: factor of safety against liquefaction triggering
        qc1ncs: normalized tip resistance
    """
    # def f(CRR):
    #     return ((0.102 + np.log(CRR_step / CRR)) / 0.276) - z_score
    
    # get dimensions
    n_depth = eff_sig_v0.shape[0]
    n_sample = eff_sig_v0.shape[1]
    
    # initialize matrix for fs_liq
    fs_liq = np.zeros((n_depth,n_sample))
    
    # first get resistance, not dependent on IM
    # get SBT index
    # _, qtn, fn, ic, _ = get_sbt_index(qc, fs, tot_sig_v0, eff_sig_v0, pa=pa)
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
    qc1ncs = kc * qtn
    
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
    ####    CYCLIC STRESS RATIO (CSR) CALCULATION    ####
    CSR = (0.65 * tot_sig_v0 * pga * rd) / (eff_sig_v0 * MSF)
    ####    FACTOR OF SAFETY AGAINST LIQUEFACTION TRIGGERING CALCULATION    ####
    fs_liq = CRR / CSR
    # if qc1ncs > 160 or ic > 2.4:
        # fs_liq = 4
    for j in range(n_sample):
        # get values for current sample
        qc1ncs_j = qc1ncs[:,j]
        ic_j = ic[:,j]
        fs_liq[np.logical_or(qc1ncs_j>160,ic_j>2.4),j] = 4
    # else:
    #     fs_liq = CRR / CSR
    # store to output
    # fs_liq[i] = np.minimum(fs_liq_i,4)
    # fs_liq[i] = fs_liq_i
    # return
    return fs_liq, qc1ncs


@njit(
    Tuple((float64[:,:],float64[:,:]))(
        float64[:],float64[:],float64[:],float64[:,:],float64[:,:],
        float64[:,:],
        float64,float64,
        float64
    ),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_fs_liq_BoulangerIdriss2016(
    qc, fs, z, tot_sig_v0, eff_sig_v0,
    ic, 
    pga, mag,
    pa=101.3
):
    """
    Return the factor of safety against liquefaction triggering based on Boulanger & Idriss (2016)
    Parameters:
        qc: measured tip resistance (MPa)
        fs: measured sleeve resistance (MPa)
        z: depth (m)
        tot_sig_vo: total stress at specific depth (kPa)
        eff_sig_vo: effective stress at specific depth (kPa)
        ic: SBT Index, Ic Robertson (2009)
        mag: earthquake moment magnitude
        pga: peak ground acceleration at the ground surface
        pa: reference pressure 1 atm = 101.3 kPa
    Return:
        fs_liq: factor of safety against liquefaction triggering
        qc1ncs: normalized tip resistance
    """
    # get dimensions
    n_depth = eff_sig_v0.shape[0]
    n_sample = eff_sig_v0.shape[1]
    
    # initialize matrix for fs_liq
    fs_liq = np.zeros((n_depth,n_sample))
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
    # for i in range(n_domain):
    ####    CYCLIC STRESS RATIO (CSR) CALCULATION    ####
    CSR = (0.65 * tot_sig_v0 * pga * rd) / (eff_sig_v0 * MSF * k_sigma)
    ####    FACTOR OF SAFETY AGAINST LIQUEFACTION TRIGGERING CALCULATION    ####
    fs_liq = CRR / CSR
    for j in range(n_sample):
        ic_j = ic[:,j]
        fs_liq[ic_j>2.4,j] = 4
    # fs_liq[i] = fs_liq_i
    # if ic > 2.4:
    #     fs_liq_i = 4
    # else:
    #     fs_liq_i = CRR / CSR
    return fs_liq, qc1ncs
    # return np.ones((2,2,2)), np.ones((2,2))


@njit(
    Tuple((float64[:,:],float64[:,:]))(
        float64[:],float64[:],float64[:],float64[:,:],float64[:,:],
        float64[:,:],float64[:,:],float64[:,:],
        float64,float64,
        float64,float64
    ),
    fastmath=True,
    cache=True,
    # parallel=True
)
def get_cpt_based_fs_liq(
    qc, fs, z, tot_sig_v0, eff_sig_v0,
    qtn, fn, ic,
    pga, mag,
    weight_r09=0.5, pa=101.3
):
    """
    Return the factor of safety against liquefaction using Robertson (2009) and Boulanger & Idriss (2016)
    Parameters:
        qc: measured tip resistance (MPa)
        fs: measured sleeve resistance (MPa)
        z: depth (m)
        tot_sig_vo: total stress at specific depth (kPa)
        eff_sig_vo: effective stress at specific depth (kPa)
        qtn: noormalized, dimensionless tip resistance (1 atm & overburden)
        fn: normalized friction ratio (%)
        ic: SBT Index, Ic Robertson (2009)
        mag: earthquake moment magnitude
        pga: peak ground acceleration at the ground surface
        weight_r09: weight for Robertson09; weight for BI16 = 1 - weight for Robertson09
        pa: reference pressure 1 atm = 101.3 kPa
    Return:
        fs_liq: weighted factor of safety against liquefaction triggering
        qc1ncs: normalized tip resistance
    """
    # get fs_liq amd qc1ncs for each method
    weight_bi16 = 1 - weight_r09
    if weight_r09 > 0:
        fs_liq_Robertson2009, qc1ncs_Robertson2009 = \
            get_fs_liq_Robertson2009(
                qc, fs, z, tot_sig_v0, eff_sig_v0,
                qtn, fn, ic,
                pga, mag,
                pa
            )
    else:
        fs_liq_Robertson2009 = np.zeros(ic.shape)
        qc1ncs_Robertson2009 = np.zeros(ic.shape)
    if weight_bi16 > 0:
        fs_liq_BoulangerIdriss2016, qc1ncs_BoulangerIdriss2016 = \
            get_fs_liq_BoulangerIdriss2016(
                qc, fs, z, tot_sig_v0, eff_sig_v0,
                ic, 
                pga, mag,
                pa
            )
    else:
        fs_liq_BoulangerIdriss2016 = np.zeros(ic.shape)
        qc1ncs_BoulangerIdriss2016 = np.zeros(ic.shape)
    # get weighted fs_liq and qc1ncs
    fs_liq = \
        fs_liq_Robertson2009 * weight_r09 + \
        fs_liq_BoulangerIdriss2016 * weight_bi16 # %
    qc1ncs = \
        qc1ncs_Robertson2009 * weight_r09 + \
        qc1ncs_BoulangerIdriss2016 * weight_bi16 # %
    # return
    return fs_liq, qc1ncs


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
    cache=True,
    # parallel=True
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