# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:58:50 2022

@author: TomClifford

This version implements the anchor block length calculation
"""

import math
import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
# import pyproj
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import cascaded_union

# OpenSRA modules
from src.site.geodata import NetworkData


def get_pipe_crossing(
    path_to_def_shp,
    infra_site_data,
    export_dir=None,
    def_type='landslide'
    # export_path
):
    """function to determine crossing"""
    #rough estimate of landslide direction for now
    landslides = gpd.read_file(path_to_def_shp)
    
    # convert DataFrame to GeoDataFrame
    infra_site_data_gdf = gpd.GeoDataFrame(
        infra_site_data,
        crs=4326,
        geometry=[
            LineString([
                (infra_site_data['LON_BEGIN'][i], infra_site_data['LAT_BEGIN'][i]),
                (infra_site_data['LON_END'][i], infra_site_data['LAT_END'][i])
            ]) for i in range(infra_site_data.shape[0])
        ]
    )

    # use spatial index to get crossing
    intx = infra_site_data_gdf.sindex.query_bulk(landslides.geometry.boundary, predicate='intersects') #returns faults, then pipes
    #listed in fault order, resort to pipes so pipes can be analyzed together
    intx_df = pd.DataFrame({'landslide_index': intx[0], 'pipe_index':intx[1]})
    #sort  by pipes
    intx_df = intx_df.sort_values(by='pipe_index').reset_index(drop=True)

    #landslide 44 is cordelia junction case study
    # intx_df = intx_df[intx_df.landslide_index == 44]

    #536 polygons, 535 that intersect for me?
    dict_list = []
    for i, r in intx_df.iterrows():
        # get current pipe and fault
        pipe_id, fault_id = r.pipe_index, r.landslide_index
        pipe = infra_site_data_gdf.iloc[[pipe_id]] 
        fault = landslides.iloc[[fault_id]]
        
        #UTM Zone 11 3718, zone to 3740
        #choose CRS based on location    
        crs = getUTM_crs(pipe.iloc[0].geometry.centroid.x)
        pipe = pipe.to_crs(crs) #UTM ZONE 10N
        fault = fault.to_crs(crs)  
        
        ''' rake vector'''
        centroid = fault.geometry[fault_id].centroid

        dseg = pipe.iloc[0]
        ''' use outwardVectorfuction to get vector of pipe pointing out of the landslide zone'''
        intx_pipe = InwardVector(dseg, fault, fault_id, crs) 
        end_seg = intx_pipe.iloc[0]
        vector_plot_length = intx_pipe.length[0]*5
        
        '''determine landslide direction here'''
        landslide_direction = fault.iloc[0]['Azimuth']

        #second pipe vector longer for plotting
        pipe_vector_plotting = bearing(end_seg.geometry.centroid, azimuth(end_seg.geometry), vector_plot_length, side='one', crs=crs)
        landslide_vector = bearing(end_seg.geometry.centroid, landslide_direction, vector_plot_length, side='one', crs=crs)
        beta = azimuth_lines(azimuth(end_seg.geometry), landslide_direction) #small for tension, large compression
        
        ''' calculate anchor length La'''
        #calculate La
        La = 30 #default 30 meters
        #La 30, then check for bending > 40 within 30 meters and fault, then check if crossing length < 30, use half that
        # #find intersection point
        intx_point = dseg.geometry.intersection(fault.boundary.unary_union)
        intx_point = gpd.GeoSeries(intx_point)
        #pull pipe within 30 meters of intersection and within fault
        anchor_buff = intx_point.buffer(La)
        anchor_buff = gpd.GeoDataFrame(geometry=anchor_buff, crs=crs)
        #clip pipe intersectin with buffer
        pipe_buff = pipe.clip(anchor_buff).clip(fault) 
        #determine if multiple segments (more than two coords)
        num_coords = len(pipe_buff.geometry.iloc[0].coords)
        
        # if multiple crossings within same segment`
        if num_coords > 2:
            for c in range(num_coords):
                if c > 1:
                    #point c-1 is the common vertex between both segments = anchor point if big bend
                    az1 = azimuth(LineString([Point(pipe_buff.geometry.iloc[0].coords[c-2]),
                                             Point(pipe_buff.geometry.iloc[0].coords[c-1])]))
                    az2 = azimuth(LineString([Point(pipe_buff.geometry.iloc[0].coords[c-1]),
                                             Point(pipe_buff.geometry.iloc[0].coords[c])]))
                    bend_angle = np.abs(angleDiff(az1, az2))
                    if bend_angle > 40: #degrees - make 40
                        #distance from intersection to vertex
                        La_line = gpd.GeoSeries(LineString([intx_point.iloc[0], Point(pipe_buff.geometry.iloc[0].coords[c-1])]), crs=crs)
                        La = La_line.iloc[0].length #new length to hard point
        #find length of total pipe within 30 meters and the zone
        crossing_length = pipe_buff.iloc[0].geometry.length
        #if length of pipe within fault zone is less than default La (more than 1% because of measuring difference), use half that lenght
        if crossing_length/La < 0.98:
            La = crossing_length/2
        
        # make dictionary to be used to convert to DataFrame
        temp = {
            'beta_crossing': beta,
            'strike': np.nan,
            # 'psi_dip': np.nan,
            'psi_dip': 75,
            'theta_slip': landslide_direction,
            'l_anchor': La
            # 'landslide':fault.iloc[0]. 
            }
        temp = {**dseg, **temp} #combine
        dict_list.append(temp)
    
    # make DataFrame
    # landslide_geometries = pd.DataFrame.from_dict(dict_list)
    landslide_geometries_pd = pd.DataFrame.from_dict(dict_list)
    # landslide_geometries_pd.drop
    landslide_geometries = gpd.GeoDataFrame(
        landslide_geometries_pd,
        crs=crs,
        geometry=landslide_geometries_pd.geometry
    )
    landslide_geometries.to_crs(4326,inplace=True)
    
    # reorder to make obj_id first
    if 'obj_id' in landslide_geometries.columns:
        resort_col_name = 'obj_id'
    elif 'obj_id'.upper() in landslide_geometries.columns:
        resort_col_name = 'obj_id'.upper()
    landslide_geometries = landslide_geometries[
        [resort_col_name] + list(landslide_geometries.columns.drop(resort_col_name))
    ]
    
    # export
    if export_dir is not None:
        landslide_geometries.to_csv(
            os.path.join(
                export_dir,
                'site_data_PROCESSED_CROSSING_ONLY.csv'
            ), index=False
        )
        landslide_geometries.to_file(
            os.path.join(
                export_dir,
                'site_data_PROCESSED_CROSSING_ONLY.shp'
            ), index=False
        )
        
    # return
    return landslide_geometries


# functions for getting fault-crossing - logic by Tom Clifford
def dip2strike(dipdirection):
    strike = dipdirection - 90
    if strike < 0:
        strike += 360
    return strike


def getUTM_crs(longitude):
    #returns corresponding utm zone crs based on given longitude
    #only need UTM Zones 10N and 11N for this project
    if -126 < longitude <= -120:
        crs = 32610
    elif -120 < longitude <= -114:
        crs = 32611
    else:
        crs=None
    return crs


#this works
def azimuth(linestring):
    '''azimuth between 2 linestring geometies'''
    point1 = Point(linestring.xy[0][0], linestring.xy[1][0])
    point2 = Point(linestring.xy[0][1], linestring.xy[1][1])
    angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360
    

def azimuth_lines(ang1, ang2, angle_type=None):
    #angles clockwise from north, returns acute angle 
    theta = np.abs(ang1 - ang2)
    if theta > 180:
        theta -= 360 
    if theta < -180:
        theta += 360
    if angle_type == 'acute':
        if theta > 90:
            theta -= 90 #find acute
    theta = np.abs(theta)
    return theta
    
    
def angleDiff(a1, a2):
    a = a1 - a2
    a = (a + 180) % 360 - 180
    return a
    
    
# #new bearing using gpd.rotate for more acurate azimuth
def bearing(centroid, az, radius, side='both', crs='EPSG:4326'):
    rad = LineString([centroid, (centroid.x, centroid.y + radius)])
    bearing_line = gpd.GeoSeries(rad, crs=crs)
    bearing = bearing_line.rotate(-az, origin=centroid)
    return bearing


def InwardVector(seg, fault, fault_id, crs='EPSG:4326'):
        #returns segment as a vector going from outside to inside polygon
        # seg=dseg
        p1 = Point([seg.geometry.xy[0][0], seg.geometry.xy[1][0]])
        p2 = Point([seg.geometry.xy[0][1], seg.geometry.xy[1][1]])
        #first compare distance from fault polygon
        p1dist = p1.distance(fault.geometry[fault_id])
        p2dist = p2.distance(fault.geometry[fault_id])
        if min(p1dist, p2dist) != 0 or max(p1dist, p2dist) <= 0:
            #if both points inside or both outside polygon - use distance from centroid
            p1dist = p1.distance(fault.geometry[fault_id].centroid)
            p2dist = p2.distance(fault.geometry[fault_id].centroid)                    
        if p1dist > p2dist:
            p_in = p1
            p_out = p2
        if p2dist > p1dist:
            p_in = p2
            p_out = p1

        return gpd.GeoDataFrame(geometry=[LineString([p_in, p_out])], crs=crs)