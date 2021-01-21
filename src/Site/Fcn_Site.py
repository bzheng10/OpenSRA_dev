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
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# OpenSRA modules and functions
from src.EDP import Fcn_EDP


# -----------------------------------------------------------
def get_geologic_param(site_data, geo_shp_file, geo_unit_param_file,
    ky_param_to_use, source_param_for_ky, appended_columns=[]):
    """
    Returns geologic units and associated properties given a list of coordinates.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """

    # extract site locations and remove residual decimals
    site_loc = site_data.loc[:,['LONG_MIDDLE','LAT_MIDDLE']].values
    site_loc = np.round(site_loc,decimals=6)

    # get slope values
    slope = site_data.loc[:,'Slope (deg)'].values
    
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
    unit = np.asarray([df_geo_shp['PTYPE'][i] \
        if i >= 0 and df_geo_shp['PTYPE'][i] in unit_abbr else 'Others' \
        for i in unit_ind])
    # logging.info(f"\tMapped coordinates to geologic units")
    
    # pull following columns into a dictionary for comparison
    items = ['Unit Weight (kN/m3)', 'Thickness (m, assume 6 m)', 'Friction Angle (degrees)', 'Cohesion (kPa)']
    items_store = ['unit_weight_slate','thickness_slate','phi_slate','c_slate']
    val_dict = {}
    for ind,item in enumerate(items):
        val_dict[items_store[ind]] = df_geo_prop[item].values
    
    # Initialize arrays
    unit_weight_slate = np.zeros(unit.shape) 
    thickness_slate = np.ones(unit.shape)*6 # fix to 6 m
    phi_slate = np.zeros(unit.shape)
    c_slate = np.zeros(unit.shape)
    # vs30_slate = np.zeros(unit.shape)
    
    # loop through units and map other properties to site based on geologic unit
    for i in range(len(unit_abbr)):
        ind = np.where(unit==unit_abbr[i])
        unit_weight_slate[ind] = val_dict['unit_weight_slate'][i]
        phi_slate[ind] = val_dict['phi_slate'][i]
        c_slate[ind] = val_dict['c_slate'][i]
        # vs30_slate[ind] = val_dict['vs30_slate'][i]
        
    # update site_date with geologic units and properties
    site_data['Geologic Unit (CGS, 2010)'] = unit
    site_data['Unit Weight_Slate (kN/m3)'] = unit_weight_slate
    site_data['Thickness_Slate (m)'] = thickness_slate
    site_data['Friction Angle_Slate (deg)'] = phi_slate
    site_data['Cohesion_Slate (kPa)'] = c_slate
    appended_columns.extend([
        'Geologic Unit (CGS, 2010)', 'Unit Weight_Slate (kN/m3)', 'Thickness_Slate (m)',
        'Friction Angle_Slate (deg)', 'Cohesion_Slate (kPa)'])
    # site_data['Vs30_Slate (m/s)'] = vs30_slate
    # logging.info(f"\tMapped coordinates to geologic properties and updated site data table")
    logging.info(f'\t\t{os.path.basename(geo_shp_file)}')
    
    # Get params to calculate ky
    beta2use = site_data['Slope (deg)'].values
    
    # Other params
    if source_param_for_ky == "Preferred":
        phi2use = site_data['Friction Angle_Slate (deg)'].values
        c2use = site_data['Cohesion_Slate (kPa)'].values
        t2use = site_data['Thickness_Slate (m)'].values
        gamma2use = site_data['Unit Weight_Slate (kN/m3)'].values
        
    elif source_param_for_ky == 'User Defined':
        try:
            phi2use = site_data[ky_param_to_use['friction_angle']].values
        except:
            phi2use = site_data['Friction Angle_Slate (deg)'].values
        try:
            c2use = site_data[ky_param_to_use['cohesion']].values
        except:
            c2use = site_data['Cohesion_Slate (kPa)'].values
        try:
            t2use = site_data[ky_param_to_use['thickness']].values
        except:
            t2use = site_data['Thickness_Slate (m)'].values
        try:
            gamma2use = site_data[ky_param_to_use['unit_weight']].values
        except:
            gamma2use = site_data['Unit Weight_Slate (kN/m3)'].values
    
    # Calculate ky using different methods
    ky_bray = Fcn_EDP.get_ky(slope_type='infinite',method='bray',phi=phi2use,c=c2use,beta=beta2use,
                            t=t2use,gamma=gamma2use)
    ky_grant = Fcn_EDP.get_ky(slope_type='infinite',method='grant',phi=phi2use,c=c2use,beta=beta2use,
                            t=t2use,gamma=gamma2use)
    ky_rathje = Fcn_EDP.get_ky(slope_type='infinite',method='rathje',phi=phi2use,c=c2use,beta=beta2use,
                            t=t2use,gamma=gamma2use)
    # if less than 0, set to 99
    ky_bray[ky_bray<0]=99
    ky_grant[ky_grant<0]=99
    ky_rathje[ky_rathje<0]=99
    
    # update site_date with yield accelerations
    site_data['Ky for Infinite Slope_Bray'] = ky_bray
    site_data['Ky for Infinite Slope_Grant'] = ky_grant
    site_data['Ky for Infinite Slope_Rathje'] = ky_rathje
    appended_columns.extend([
        'Ky for Infinite Slope_Bray',
        'Ky for Infinite Slope_Grant',
        'Ky for Infinite Slope_Rathje'])
    logging.info(f"\tCalculated yield accelerations updated site data table")
    
    #
    return site_data, appended_columns