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
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon
from shapely.ops import linemerge, polygonize
from shapely.prepared import prep
from numba import jit, njit

# OpenSRA modules and functions
from src.edp import edp_util
from src.util import *


# -----------------------------------------------------------
@njit
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