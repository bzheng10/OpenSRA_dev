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
import os, time, logging, importlib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# OpenSRA modules and functions
from src.edp import Fcn_EDP


# -----------------------------------------------------------
def get_geologic_unit(site_data, geo_shp_file, geo_unit_param_file, ky_param_to_use, source_param_for_ky):
    """
    Returns geologic units and associated properties given a list of coordinates
    
    Parameters
    ----------
    
    Returns
    -------
    
    """

    # extract site locations and remove residual decimals
    site_loc = site_data.loc[:,['Longitude','Latitude']].values
    site_loc = np.round(site_loc,decimals=6)
    
    # extract slopes from input file
    slope = site_data.loc[:,'Slope (deg)'].values
    
    # number of sites
    nsite = len(site_loc)
    
    # import properties for geologic units
    df_geo_prop = pd.read_csv(geo_unit_param_file)
    unit_abbr = df_geo_prop['Unit Abbreviation'].values # extract unit abbreviations
    
    # import shapefile with geologic units
    df_geo_shp = gpd.GeoDataFrame.from_file(geo_shp_file)
    df_geo_shp = df_geo_shp.to_crs('epsg:4326') # convert spatial reference system

    # get geologic unit index for each site - index corresponds to order of unit in Micaela's spreadsheet
    unit_ind = []
    for site_loc_i in site_loc:
        try:
            unit_ind.append(np.where(df_geo_shp.intersects(Point(site_loc_i)).values)[0][0])
        except:
            unit_ind.append(-99)
    
    # map unit index to unit abbreviation
    unit = np.asarray([df_geo_shp['PTYPE'][i] if i >= 0 else 'Others' for i in unit_ind])
    logging.info(f"\tMapped coordinates to geologic units")
    
    # pull following columns into a dictionary for comparison
    items = ['Unit Weight (kN/m3)', 'Thickness (m, assume 6 m)', 'Friction Angle (degrees)', 'Cohesion (kPa)', 'Vs30 (m/s)']
    items_store = ['unit_weight','thickness','phi','c', 'vs30_geo']
    val_dict = {}
    for i,item in enumerate(items):
        val_dict[items_store[i]] = df_geo_prop[item].values
    
    # Initialize arrays
    unit_weight = np.zeros(unit.shape) 
    thickness = np.ones(unit.shape)*6 # fix to 6 m
    phi = np.zeros(unit.shape)
    c = np.zeros(unit.shape)
    vs30_geo = np.zeros(unit.shape)
    
    # loop through units and map other properties to site based on geologic unit
    for i in range(len(unit_abbr)):
        ind = np.where(unit==unit_abbr[i])
        unit_weight[ind] = val_dict['unit_weight'][i]
        phi[ind] = val_dict['phi'][i]
        c[ind] = val_dict['c'][i]
        vs30_geo[ind] = val_dict['vs30_geo'][i]
        
    # update site_date with geologic units and properties
    site_data['Geologic Unit'] = unit
    site_data['Unit Weight geo (kN/m^3)'] = unit_weight
    site_data['Thickness geo (m)'] = thickness
    site_data['Friction Angle geo (deg)'] = phi
    site_data['Cohesion geo (kPa)'] = c
    site_data['Vs30 geo (m/s)'] = vs30_geo
    logging.info(f"\tMapped coordinates to geologic properties and updated site data table")
    
    # Get params to calculate ky
    beta2use = site_data['Slope (deg)'].values
    
    # Other params
    if source_param_for_ky == 'Default':
        phi2use = site_data['Friction Angle geo (deg)'].values
        c2use = site_data['Cohesion geo (kPa)'].values
        t2use = site_data['Thickness geo (m)'].values
        gamma2use = site_data['Unit Weight geo (kN/m^3)'].values
        
    elif source_param_for_ky == 'UserDefined':
        try:
            phi2use = site_data[ky_param_to_use['friction_angle']].values
        except:
            phi2use = site_data['Friction Angle geo (deg)'].values
        try:
            c2use = site_data[ky_param_to_use['cohesion']].values
        except:
            c2use = site_data['Cohesion geo (kPa)'].values
        try:
            t2use = site_data[ky_param_to_use['thickness']].values
        except:
            t2use = site_data['Thickness geo (m)'].values
        try:
            gamma2use = site_data[ky_param_to_use['unit_weight']].values
        except:
            gamma2use = site_data['Unit Weight geo (kN/m^3)'].values
    
    # Calculate ky using different methods
    ky_bray = Fcn_EDP.get_ky(slope_type='infinite',method='bray',phi=phi2use,c=c2use,beta=beta2use,
                            t=t2use,gamma=gamma2use)
    ky_grant = Fcn_EDP.get_ky(slope_type='infinite',method='grant',phi=phi2use,c=c2use,beta=beta2use,
                            t=t2use,gamma=gamma2use)
    ky_rathje = Fcn_EDP.get_ky(slope_type='infinite',method='rathje',phi=phi2use,c=c2use,beta=beta2use,
                            t=t2use,gamma=gamma2use)
    ky_bray[ky_bray<0]=0
    ky_grant[ky_grant<0]=0
    ky_rathje[ky_rathje<0]=0
    
    # update site_date with yield accelerations
    site_data['ky_inf_bray'] = ky_bray
    site_data['ky_inf_grant'] = ky_grant
    site_data['ky_inf_rathje'] = ky_rathje
    logging.info(f"\tCalculated yield accelerations updated site data table")
    
    #
    return site_data
    
    # update input data spreadsheet with properties
    # site_data.to_csv(site_file)