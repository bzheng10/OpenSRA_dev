# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Returns geologic units and associated properties given a list of coordinates
#
# Created: August 5, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import os
import logging
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon
from shapely.ops import linemerge, polygonize
from shapely.prepared import prep
from numba import njit, float64, boolean, int64, typeof
from numba.types import Tuple, List

# OpenSRA modules and functions
from src.edp import edp_util
from src.util import *
    

# -----------------------------------------------------------
@njit(
    float64[:,:](
        float64,float64,float64,float64,float64,float64
    ),
    cache=True
)
def make_grid_nodes(x_min, y_min, x_max, y_max, dx, dy):
    """makes grid of nodes given min and max in x and y directions, and dx and dy"""
    return np.asarray([
        [x0, y0] \
        for x0 in np.arange(x_min, x_max+dx, dx) \
        for y0 in np.arange(y_min, y_max+dy, dy)
    ])

# -----------------------------------------------------------
def make_list_of_linestrings(
    pt1_x, pt1_y, pt2_x, pt2_y
):
    '''fast way to make a list of LineStrings without looping and making one LineString at a time'''
    pts1 = gpd.points_from_xy(pt1_x, pt1_y)
    pts2 = gpd.points_from_xy(pt2_x, pt2_y)
    lines = pts1.union(pts2).convex_hull
    return lines

# -----------------------------------------------------------
# @njit
def get_regional_liq_susc(witter_geo_unit, bedrossian_geo_unit, gw_depth, get_mean=False, default='none'):
    """get liquefaction susceptibility category based on groundwater depth (m)"""
    
    # get dimensions
    if get_mean is False:
        n_sample = gw_depth.shape[1]
    
    # initialize output
    liq_susc = np.empty_like(gw_depth, dtype="<U10")
    
    # find where gw_depth is at the following ranges
    gw_depth_le_3m_all = gw_depth<=3
    gw_depth_btw_3m_9m_all = np.logical_and(gw_depth>3,gw_depth<=9)
    gw_depth_btw_9m_12m_all = np.logical_and(gw_depth>9,gw_depth<=12)
    gw_depth_gt_12m_all = gw_depth>12
    
    # first check Witter et al. (2006)
    ind_witter_geo_unit_avail = np.where(witter_geo_unit.notna())[0] # where Witter geo units exist
    if len(ind_witter_geo_unit_avail) > 0:
        # get subset of gw_depth conditions for where witter is available
        gw_depth_le_3m = gw_depth_le_3m_all[ind_witter_geo_unit_avail]
        gw_depth_btw_3m_9m = gw_depth_btw_3m_9m_all[ind_witter_geo_unit_avail]
        gw_depth_btw_9m_12m = gw_depth_btw_9m_12m_all[ind_witter_geo_unit_avail]
        gw_depth_gt_12m = gw_depth_gt_12m_all[ind_witter_geo_unit_avail]
        # pull these geo units
        witter_geo_unit_avail = witter_geo_unit[ind_witter_geo_unit_avail].values
        # some units in Witter may end with a number or question mark, if so, map it to the unit without the added character
        witter_geo_unit_avail = np.asarray([
            val[:-1] if val[-1] == "?" or val[-1].isdigit() \
            else val for val in witter_geo_unit_avail
        ])
        if get_mean:
            witter_geo_unit_avail_expanded = witter_geo_unit_avail.copy()
            # establish empty array for tracking liq_susc
            liq_susc_witter = np.empty((len(ind_witter_geo_unit_avail)), dtype="<U10")
        else:
            witter_geo_unit_avail_expanded = np.tile(witter_geo_unit_avail,(n_sample,1)).T
            # establish empty array for tracking liq_susc
            liq_susc_witter = np.empty((len(ind_witter_geo_unit_avail), n_sample), dtype="<U10")
        # start correlating to liq_susc
        # artificial fill
        curr_unit = 'af'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # artificial fill over estuarine mud
        curr_unit = 'afem'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # artificial fill, levee
        curr_unit = 'alf'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # artificial fill, channel
        curr_unit = 'acf'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # artificial fill, dam
        curr_unit = 'adf'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # gravel quarry
        curr_unit = 'gq'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # artificial stream channel
        curr_unit = 'ac'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'low'
        # modern stream channel deposits
        curr_unit = 'Qhc'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # latest holocene alluvial fan deposits
        curr_unit = 'Qhfy'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'low'
        # latest holocene alluvial fan levee deposits
        curr_unit = 'Qhly'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'low'
        # latest holocene stream terrace deposits
        curr_unit = 'Qhty'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'low'
        # latest holocene alluvial deposits undifferentiated
        curr_unit = 'Qhay'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'low'
        # holocene beach sand
        curr_unit = 'Qhbs'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene dune sand
        curr_unit = 'Qhds'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene SF Bay mud
        curr_unit = 'Qhbm'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene basin deposits
        curr_unit = 'Qhb'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene fine grained alluvial fan-estuarine complex deposits
        curr_unit = 'Qhfe'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene estuarine delta deposits
        curr_unit = 'Qhed'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene alluvial fan deposits
        curr_unit = 'Qhf'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene alluvial fan deposits fine facies
        curr_unit = 'Qhff'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene alluvial fan levee deposits
        curr_unit = 'Qhl'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene stream terrace deposits
        curr_unit = 'Qht'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene alluvial deposits, undifferentiated
        curr_unit = 'Qha'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene to holocene dune sand
        curr_unit = 'Qds'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene to holocene basin deposits
        curr_unit = 'Qb'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene to holocene alluvial fan deposits
        curr_unit = 'Qf'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene to holocene stream terrace deposits
        curr_unit = 'Qt'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene to holocene alluvial deposits, undifferentiated
        curr_unit = 'Qa'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene alluvial fan deposits
        curr_unit = 'Qpf'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene stream terrace deposits
        curr_unit = 'Qpt'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late pleistocene alluvial deposits, undifferentiated
        curr_unit = 'Qpa'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # pleistocene marine terrace deposits
        curr_unit = 'Qmt'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # pleistocene bay terrace deposits
        curr_unit = 'Qbt'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # early to late pleistocene pediment deposits
        curr_unit = 'Qop'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # early to middle pleistocene alluvial fan deposits
        curr_unit = 'Qof'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # early to middle pleistocene stream terrace deposits
        curr_unit = 'Qot'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # early to middle pleistocene alluvial deposits, undifferentiated
        curr_unit = 'Qoa'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_witter[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # early quaternary and older (>1.4 Ma) deposits and bedrock
        curr_unit = 'br'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[where_curr_unit] = 'none'
        # waterbodies
        curr_unit = 'H2O'
        where_curr_unit = witter_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_witter[where_curr_unit] = 'none'
        # add to liq_susc dataframe
        liq_susc[ind_witter_geo_unit_avail] = liq_susc_witter
    
    # next check Bedrossian et al. (2012)
    ind_bedrossian_geo_unit_avail = np.where(bedrossian_geo_unit.notna())[0] # where geo units exist
    if len(ind_bedrossian_geo_unit_avail) > 0:
        # get subset of gw_depth conditions for where witter is available
        gw_depth_le_3m = gw_depth_le_3m_all[ind_bedrossian_geo_unit_avail]
        gw_depth_btw_3m_9m = gw_depth_btw_3m_9m_all[ind_bedrossian_geo_unit_avail]
        gw_depth_btw_9m_12m = gw_depth_btw_9m_12m_all[ind_bedrossian_geo_unit_avail]
        gw_depth_gt_12m = gw_depth_gt_12m_all[ind_bedrossian_geo_unit_avail]
        # pull these geo units
        bedrossian_geo_unit_avail = bedrossian_geo_unit[ind_bedrossian_geo_unit_avail].values
        if get_mean:
            bedrossian_geo_unit_avail_expanded = bedrossian_geo_unit_avail.copy()
            # establish empty array for tracking liq_susc
            liq_susc_bedrossian = np.empty((len(ind_bedrossian_geo_unit_avail)), dtype="<U10")
        else:
            bedrossian_geo_unit_avail_expanded = np.tile(bedrossian_geo_unit_avail,(n_sample,1)).T
            # establish empty array for tracking liq_susc
            liq_susc_bedrossian = np.empty((len(ind_bedrossian_geo_unit_avail), n_sample), dtype="<U10")
        # start correlating to liq_susc
        # late holocene - artificial fill
        curr_unit = 'af'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - beach
        curr_unit = 'Qb'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - wash
        curr_unit = 'Qw'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'very high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - eolian and dune
        curr_unit = 'Qe'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - alluvial fan
        curr_unit = 'Qf'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - terrace
        curr_unit = 'Qt'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - lacustrine, playa, estuarine
        curr_unit = 'Ql'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - undifferentiated
        curr_unit = 'Qsu'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late holocene - alluvial valley
        curr_unit = 'Qa'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene to late pleistocene - wash
        curr_unit = 'Qyw'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene to late pleistocene - eolian dune
        curr_unit = 'Qye'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene to late pleistocene - alluvial fan
        curr_unit = 'Qyf'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene to late pleistocene - terrace
        curr_unit = 'Qyt'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene to late pleistocene - lacustrine, playa, estuarine
        curr_unit = 'Qyl'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # holocene to late pleistocene - alluvial valley
        curr_unit = 'Qya'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'high'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late to early pleistocene - wash
        curr_unit = 'Qow'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late to early pleistocene - eolian and dune
        curr_unit = 'Qoe'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'moderate'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late to early pleistocene - alluvial fan
        curr_unit = 'Qof'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late to early pleistocene - alluvial valley
        curr_unit = 'Qoa'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late to early pleistocene - terrace
        curr_unit = 'Qot'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # late to early pleistocene - lacustrine, playa, estuarine
        curr_unit = 'Qol'
        where_curr_unit = bedrossian_geo_unit_avail_expanded==curr_unit
        if True in where_curr_unit:
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_le_3m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_3m_9m)] = 'low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_btw_9m_12m)] = 'very low'
            liq_susc_bedrossian[np.logical_and(where_curr_unit,gw_depth_gt_12m)] = 'very low'
        # add to liq_susc dataframe
        liq_susc[ind_bedrossian_geo_unit_avail] = liq_susc_bedrossian
        
    # for all other conds, set to default
    liq_susc[liq_susc==''] = default
    
    # return
    return liq_susc


# -----------------------------------------------------------
@njit(
    Tuple((float64[:,:],boolean))(
        float64[:,:],float64
    ),
    fastmath=True,
    cache=True
)
def split_line_by_max_length(coord, l_max=0.1):
    """
    split a segment of a line if it exceeds maximum length
    
    Parameters
    ----------
    coord : [float, float], array
        [degree, degree] list of coordinates (lon, lat) for line
    l_max : float
        [km] maximum length
    
    Returns
    -------
    coord_split : [float, float], array
        [degree, degree] list of coordinates (lon, lat) for line after split
    flag_performed_split : boolean
        [] True if performed split on line
    
    """
    
    # calculate inter-segment lengths
    lengths = get_haversine_dist(
        lon1=coord[:-1,0], lat1=coord[:-1,1],
        lon2=coord[1:,0], lat2=coord[1:,1],
        unit='km'
    )

    # flag for splitting
    flag_performed_split = False
    # perform split based on lengths
    coord_split = np.expand_dims(coord[0,:],0) # initialize list and add in first point, expand_dims to enforce consistent array dimensions
    for i,length in enumerate(lengths):
        # check if segment length is greater than maximum length
        if length > l_max:
            n_split = np.ceil(length/l_max) # number of splits based ratio of distance/max length
            ind_split = np.arange(1,n_split+1)/n_split # get positions for interpolation, exclude first point
            coord_split = np.vstack((
                coord_split,
                np.vstack((
                    (coord[i+1,0]-coord[i,0])*ind_split + coord[i,0],   # interpolation
                    (coord[i+1,1]-coord[i,1])*ind_split + coord[i,1]    # interpolation
                )).T
            ))
            flag_performed_split = True
        else:
            coord_split = np.vstack((coord_split, np.expand_dims(coord[i+1,:],0))) # add end point if segment length is under max length

    # return
    return coord_split, flag_performed_split


# -----------------------------------------------------------
def get_segment_within_bound(bound, geom):
    """keep segments that are contained in boundary"""
    
    # get coordinates for geometry
    coord = np.asarray(geom.coords)

    # lists for tracking segments
    seg_within_set = [] # for current geometry
    seg_within_curr = [] # for current continuous segment
    # sum_segment = 0

    # loop through and check each line
    for j in range(len(coord)-1):
        current_line = LineString([coord[j,:],coord[j+1,:]]) # create LineString
        if bound.contains(current_line): # check if line is contained by boundary
            seg_within_curr.append(current_line) # append to list
            # sum_segment += 1 # increment counter
        else:
            # line becomes disjointed
            # first merge lines from previous set if segments exist
            if len(seg_within_curr) > 0:
                seg_within_set.append(linemerge(MultiLineString(seg_within_curr)))
                seg_within_curr = [] # reset list of lines for next set
    # merge lines and add to full set
    if len(seg_within_curr) > 0:
        seg_within_set.append(linemerge(MultiLineString(seg_within_curr)))
        
    # return
    # return seg_within_set, sum_segment
    return seg_within_set


# -----------------------------------------------------------
def polygonize_cells(cells):
    """"convert a list of cells into shapely polygons and combine into a geopandas GeoSeries"""
    return gpd.GeoSeries([Polygon(cell) for cell in cells])


# -----------------------------------------------------------
def get_site_data(param, dataset_info, dataset_main_dir, x, y):
    """
    
    """
    
    # remove residual decimals from x and y
    x = np.round(x,decimals=6)
    y = np.round(y,decimals=6)
    site_loc = np.vstack([x,y])
    
    # file type for dataset
    file_type = dataset_info['FileType']
    # full file path
    if dataset_info['Path'] is None:
        file_path = None
    else:
        file_path = os.path.join(dataset_main_dir, dataset_info['Path'])
    # factor to scale results to target unit
    conversion_factor = dataset_info['ConversionToOutputUnit']
    
    # try import, if fails, return None
    try:
        # import from raster files
        if file_type == 'Raster':
            param_values = Fcn_Common.interp_from_raster(
                raster_path = file_path, x = x, y = y)
            # scale to target unit
            param_values = param_values*conversion_factor
            # specific corrections
            if 'slope' in param.lower():
                param_values[param_values<=0] = 0.1 # prevent slope values from <= 0 for tangent asymptote
            elif 'vs30' in param.lower():
                param_values[param_values<=0] = 999 # prevent vs30 values from being negative
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # OpenSHA sets a check for minimum Vs30
                # if Vs30 sampled < 150, OpenSHA raises warning and ends
                # opensha-core/src/org/opensha/sha/imr/attenRelImpl/ngaw2/NGAW2_WrapperFullParam.java
                param_values[param_values<150] = 150
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                param_values[param_values<0] = 0 # prevent values from being negative
                
        # import from shapefiles
        if file_type == 'Shapefile':
            gdf = gpd.GeoDataFrame.from_file(file_path) # import shapefile with geologic units
            gdf = df_geo_shp.to_crs('epsg:4326') # convert spatial reference system
            param_values = []
            for site_loc_i in site_loc:
                try:
                    param_values.append(np.where(gdf.intersects(Point(site_loc_i)).values)[0][0])
                except:
                    param_values.append(-99)
        
        # import from CSV
        if file_type == 'CSV':
            return None
        
        #
        return param_values
    except:
        return None


# -----------------------------------------------------------
def get_geologic_param(site_data, dataset_main_dir, geo_unit_info,
    geo_param_info, geo_params_to_get, csv_param_name,
    dataset_meta_dict = {}, flag_to_update=False):
    """
    Returns geologic units and associated properties given a list of coordinates.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """

    # extract site locations and remove residual decimals
    site_loc = site_data.loc[:,['Mid Longitude','Mid Latitude']].values
    site_loc = np.round(site_loc,decimals=6)
    
    # set up key parameters
    appended_columns = []
    for param in geo_param_info:
        geo_prop_csv_file_path = os.path.join(dataset_main_dir, geo_param_info[param]['Datasets']['Set1']['Path'])
        break
    geo_unit_shp_file_path = os.path.join(dataset_main_dir, geo_unit_info['Datasets']['Set1']['Path'])
    geo_unit_column_name = geo_unit_info['ColumnNameToStoreAs']
    
    # Get ky param from user-specified source
    for param in geo_params_to_get:
        column_name = geo_param_info[param]['ColumnNameToStoreAs']
        # If column with header ending with "_Existing" exists, then already performed
        #   data search - don't perform search for the param again
        if column_name+'_Existing' in site_data:
            pass
        else:
            flag_to_update = True # updated site data
            break
    
    # only go perform actions if flag_to_update is True
    if flag_to_update:
        # import shapefile with geologic units
        gdf_geo_map = gpd.GeoDataFrame.from_file(geo_unit_shp_file_path)
        gdf_geo_map = gdf_geo_map.to_crs('epsg:4326') # convert spatial reference system

        # If column with header ending with "_Existing" exists, then already performed
        #   data search - don't perform search for the param again
        if geo_unit_column_name+'_Existing' in site_data:
            geo_unit_ind = site_data[geo_unit_column_name].values
        else:
            # import file with geologic unit properties
            df_geo_prop = pd.read_csv(geo_prop_csv_file_path)
            geo_unit_abbr = df_geo_prop['Unit Abbreviation'].values # extract unit abbreviations
            # import shapefile with geologic units
            gdf_geo_map = gpd.GeoDataFrame.from_file(geo_unit_shp_file_path)
            gdf_geo_map = gdf_geo_map.to_crs('epsg:4326') # convert spatial reference system
            # get geologic unit index for each site - index corresponds to order of unit in spreadsheet
            geo_unit_ind = []
            for site_loc_i in site_loc:
                try:
                    geo_unit_ind.append(np.where(gdf_geo_map.intersects(Point(site_loc_i)).values)[0][0])
                except:
                    geo_unit_ind.append(-99)
            # map unit index to unit abbreviation
            geo_unit = np.asarray([gdf_geo_map['PTYPE'][i] \
                if i >= 0 and gdf_geo_map['PTYPE'][i] in geo_unit_abbr else 'Others' \
                for i in geo_unit_ind])
            # logging.info(f"\tMapped coordinates to geologic units")
            site_data[geo_unit_column_name] = geo_unit
            dataset_meta_dict['GeologicUnit'] = geo_unit_info['Datasets']['Set1']
            dataset_meta_dict['GeologicUnit'].update({'ColumnNameStoredAs': geo_unit_column_name})
            logging.info(f"\t - {'GeologicUnit'}: {geo_unit_info['Datasets']['Set1']['Source']}")
            appended_columns.append(geo_unit_column_name)
            flag_to_update = True # updated site data
        
        # Get ky param from user-specified source
        for param in geo_params_to_get:
            column_name = geo_param_info[param]['ColumnNameToStoreAs']
            dataset_info = geo_param_info[param]['Datasets']['Set1']
            conversion_factor = dataset_info['ConversionToOutputUnit']
            # If column with header ending with "_Existing" exists, then already performed
            #   data search - don't perform search for the param again
            if column_name+'_Existing' in site_data:
                pass
            else:
                # If column with header already exists, save it in another column
                if column_name in site_data:
                    site_data[column_name+'_Existing'] = site_data[column_name].values
                # get id used in geologic param csv file and get data
                geologic_data_in_csv = df_geo_prop[csv_param_name[param]].values
                # Initialize array
                param_values = np.zeros(geo_unit.shape)
                # loop through units and map other properties to site based on geologic unit
                for i in range(len(geo_unit_abbr)):
                    ind = np.where(geo_unit==geo_unit_abbr[i])
                    param_values[ind] = geologic_data_in_csv[i]
                # scale to target unit
                param_values = param_values*conversion_factor
                # update site_date with geologic units and properties
                site_data[column_name] = param_values
                dataset_meta_dict[param] = dataset_info
                dataset_meta_dict[param].update({'ColumnNameStoredAs': column_name})
                logging.info(f"\t - {param}: {dataset_info['Source']}")
                appended_columns.append(column_name)
                flag_to_update = True # updated site data
    
    #
    return site_data, dataset_meta_dict, appended_columns, flag_to_update