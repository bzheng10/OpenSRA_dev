# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# This function:
# 1) Obtains 2D crossings between pipe trace and deformation polygons
# 2) Obtains anchorage length to be used in pipe strain models
#
# Created: September 1, 2022
# @author: Barry Zheng (Slate Geotechnical Consultants)
# 
# -----------------------------------------------------------

# Python base modules
import logging
import os
import sys
import warnings

# scientific processing modules
import pandas as pd
import numpy as np

# geospatial processing modules
import geopandas as gpd
from shapely.affinity import rotate
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString
from shapely.ops import split, unary_union
from pyproj import Transformer

# precompile
# import numba as nb
from numba import njit
# from numba.types import List, unicode_type
from numba.core.errors import NumbaPendingDeprecationWarning
# suppress warning that may come up
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# OpenSRA modules
# from src.site.geodata import NetworkData
from src.util import get_basename_without_extension
from src.site.geodata import LocationData


# ---
def dot_product(
    vect1_pt1_in_meters,
    vect1_pt2_in_meters,
    vect2_pt1_in_meters,
    vect2_pt2_in_meters,
    return_angle=True,
    return_comp_vect1_on_vect2=False
):
    """
    performs dot product between two vector defined by their points;
    returns the internal angle and or the component of vector 1 on vector 2
    """
    # check return param
    if return_angle is False and return_comp_vect1_on_vect2 is False:
        raise ValueError("either return_angle or return_comp_vect1_on_vect2 must be True.")
    # form vector 1 from pt1 to pt2
    dir_vect1 = vect1_pt2_in_meters - vect1_pt1_in_meters
    dir_vect1_mag = (dir_vect1[:,0]**2 + dir_vect1[:,1]**2)**0.5
    dir_vect1_norm = dir_vect1 / np.vstack([dir_vect1_mag,dir_vect1_mag]).T
    # form vector 2 from pt1 to pt2
    dir_vect2 = vect2_pt2_in_meters - vect2_pt1_in_meters
    dir_vect2_mag = (dir_vect2[:,0]**2 + dir_vect2[:,1]**2)**0.5
    dir_vect2_norm = dir_vect2 / np.vstack([dir_vect2_mag,dir_vect2_mag]).T
    # get internal angle
    if return_angle:
        cos_angle = dir_vect1_norm[:,0]*dir_vect2_norm[:,0] + dir_vect1_norm[:,1]*dir_vect2_norm[:,1]
        angle = np.degrees(np.arccos(cos_angle))
    # get component of vector 1 on vector 2
    elif return_comp_vect1_on_vect2:
        comp = dir_vect1[:,0]*dir_vect2_norm[:,0] + dir_vect1[:,1]*dir_vect2_norm[:,1]
    # return
    if return_angle and return_comp_vect1_on_vect2:
        return angle, comp
    else:
        if return_angle:
            return angle
        elif return_comp_vect1_on_vect2:
            return comp


# ---
def get_azimuth(
    pt1_in_meters, # start point of direction vector
    pt2_in_meters  # end point of direction vector
):
    """returns the azimuth (deg) for a direction vector defined by pt1 (start) and pt2 (end)"""
    # form vector from pt1 to pt2
    dir_vect = pt2_in_meters - pt1_in_meters
    dir_vect_mag = (dir_vect[:,0]**2 + dir_vect[:,1]**2)**0.5
    dir_vect_norm = dir_vect / np.vstack([dir_vect_mag,dir_vect_mag]).T
    # calculate slip_dir (direction relative to North)
    # where cos(azimuth) = dot(dir_vect_norm, (0,1))
    cos_azimuth = dir_vect_norm[:,0]*0 + dir_vect_norm[:,1]*1
    azimuth = np.degrees(np.arccos(cos_azimuth))
    # correction for internal angles when dx = negative
    azimuth[dir_vect_norm[:,0]<0] = 360 - azimuth[dir_vect_norm[:,0]<0]
    return azimuth


# ---
def get_dx_dy_for_vector_along_azimuth(
    azimuth, # azimuth in degrees, relative to North
    vect_length=1 # vector length; default = 1 meter
):
    """returns dx and dy of a vector given azimuth and target length"""
    # convert deg to rad
    azimuth_rad = azimuth*np.pi/180
    dx = vect_length * np.sin(azimuth_rad)
    dy = vect_length * np.cos(azimuth_rad)
    return dx, dy


# ---
def split_geom_into_scarp_body_toe(
    geom, # shapely geometry, in UTM (meters)
    slip_dir, # deg, relative to North (i.e., azimuth)
    high_point, # highest point from DEM
    low_point, # lowest point from DEM
    scarp_cutoff_ratio=0.15, # transition from scarp to body, ratio
    toe_cutoff_ratio=1-0.15, # transition from body to toe, ratio
    get_cutoff_lines=False, # return cutoff line geometries
    get_upper_and_lower_halves=True # split geometry into upper and lower halves and return geometries
):
    """
    splits a deformation polygon into scarp (head), body, and toe based on cutoff ratios,
    also splits polygon into upper and lower halves if requested
    """
    # get geometry centroid for rotation
    centroid = list(geom.centroid.coords)[0]
    
    # rotate such that slip direction is pointing up
    geom_rotate = rotate(geom=geom, angle=slip_dir, origin=centroid)
    
    # get extent of rotated geometry
    extent_rotate = geom_rotate.bounds
    xleft = extent_rotate[0]
    xright = extent_rotate[2]
    
    # rotate high and low points relative to centroid of geometry, then get dy
    high_point_rotate = rotate(geom=Point(high_point), angle=slip_dir, origin=centroid)
    low_point_rotate = rotate(geom=Point(low_point), angle=slip_dir, origin=centroid)
    y0 = high_point_rotate.y
    yend = low_point_rotate.y
    
    # find y-coord along vertical extent at cutoffs
    dy = yend - y0
    y_scarp_cutoff = scarp_cutoff_ratio*dy + y0
    y_toe_cutoff = toe_cutoff_ratio*dy + y0
    line_at_y_scarp_cutoff = LineString((
        (xleft,y_scarp_cutoff),
        (xright,y_scarp_cutoff),
    ))
    line_at_y_toe_cutoff = LineString((
        (xleft,y_toe_cutoff),
        (xright,y_toe_cutoff),
    ))

    # get head scarp polygon
    split_poly = split(geom_rotate,line_at_y_scarp_cutoff)
    scarp_rotate = []
    remain_rotate = []
    for each in split_poly.geoms:
        if each.bounds[3] <= y_scarp_cutoff:
            scarp_rotate.append(each)
        else:
            remain_rotate.append(each)
    scarp_rotate = unary_union(scarp_rotate)
    remain_rotate = unary_union(remain_rotate)
    # get toe polygon
    split_poly = split(remain_rotate,line_at_y_toe_cutoff)
    toe_rotate = []
    body_rotate = []
    for each in split_poly.geoms:
        if each.bounds[1] >= y_toe_cutoff:
            toe_rotate.append(each)
        else:
            body_rotate.append(each)
    toe_rotate = unary_union(toe_rotate)
    body_rotate = unary_union(body_rotate)\
    # revert rotation
    scarp = rotate(geom=scarp_rotate, angle=-slip_dir, origin=centroid)
    body = rotate(geom=body_rotate, angle=-slip_dir, origin=centroid)
    toe = rotate(geom=toe_rotate, angle=-slip_dir, origin=centroid)
    
    # if need to split into upper and lower halves
    if get_upper_and_lower_halves:
        y_mid_cutoff = 0.5*dy + y0
        line_at_y_mid_cutoff = LineString((
            (xleft,y_mid_cutoff),
            (xright,y_mid_cutoff),
        ))
        # get upper and half polygon
        split_poly = split(geom_rotate,line_at_y_mid_cutoff)
        upper_rotate = []
        lower_rotate = []
        for each in split_poly.geoms:
            if each.bounds[3] <= y_mid_cutoff:
                upper_rotate.append(each)
            else:
                lower_rotate.append(each)
        upper_rotate = unary_union(upper_rotate)
        lower_rotate = unary_union(lower_rotate)
        # revert rotation
        upper = rotate(geom=upper_rotate, angle=-slip_dir, origin=centroid)
        lower = rotate(geom=lower_rotate, angle=-slip_dir, origin=centroid)
    
    # return
    if get_cutoff_lines is False:
        # primary return
        if get_upper_and_lower_halves:
            return scarp, body, toe, upper, lower
        else:
            return scarp, body, toe
    else:
        # for specific cases
        line_at_y_scarp_cutoff_unrotated = \
            rotate(geom=line_at_y_scarp_cutoff, angle=-slip_dir, origin=centroid)
        line_at_y_toe_cutoff_unrotated = \
            rotate(geom=line_at_y_toe_cutoff, angle=-slip_dir, origin=centroid)
        return line_at_y_scarp_cutoff_unrotated, line_at_y_toe_cutoff_unrotated


@njit(
    fastmath=True,
    cache=True
)
def get_anchorage_length_full_system(
    pipe_id_full,
    sub_segment_index_list,
    segment_length_list,
    pipe_id_crossed_unique,
    hard_points_ind,
):
    """get anchorage lengths for full system, ignoring polygons"""
    # initialize
    n_segment = len(pipe_id_full)
    anchorage_length = np.zeros(n_segment)
    # for every segment crossed with deformation polygon, get the nearest hard points and compare to minimum anchorage length
    for i in range(len(pipe_id_full)):
        # current segment index
        curr_segment_index = sub_segment_index_list[i]
        # pipeline ID
        curr_pipe_id = pipe_id_full[i]
        # get hard points along current pipeline
        where_curr_pipe_id = np.where(pipe_id_crossed_unique==curr_pipe_id)[0][0]
        curr_hard_points_ind = hard_points_ind[where_curr_pipe_id]
        # get all row indices of segments for current pipe_id
        ind_for_curr_pipe_id = np.where(pipe_id_full==curr_pipe_id)[0]
        # segment index list for current pipe
        segment_index_list_curr_pipe_id = sub_segment_index_list[ind_for_curr_pipe_id]
        # find nearest hard point from start and end side
        nearest_hard_pt_ind_start = curr_hard_points_ind[np.where(curr_hard_points_ind<=curr_segment_index)[0][-1]]
        nearest_hard_pt_ind_end = curr_hard_points_ind[np.where(curr_hard_points_ind>curr_segment_index)[0][0]]
        
        # get segment just within hard points from from start and end side
        segment_index_within_hard_pt_start_side = nearest_hard_pt_ind_start
        segment_index_within_hard_pt_end_side = nearest_hard_pt_ind_end-1
        
        # start side
        length_start_side = 0
        # get list of segment index from nearest hard pt on start side to current segment
        segment_index_list_start_side = np.arange(
            segment_index_within_hard_pt_start_side,
            curr_segment_index
        )
        if len(segment_index_list_start_side) > 0:
            length_start_side = sum(segment_length_list[ind_for_curr_pipe_id[segment_index_list_start_side]])
        
        # end side
        length_end_side = 0
        # get list of segment index from nearest hard pt on start side to current segment
        segment_index_list_end_side = np.arange(
            curr_segment_index+1,
            segment_index_within_hard_pt_end_side+1,
        )
        if len(segment_index_list_end_side) > 0:
            length_end_side = sum(segment_length_list[ind_for_curr_pipe_id[segment_index_list_end_side]])
        
        # pick the side with shorter length
        anchorage_length_curr = min(length_start_side,length_end_side)
        # add in the half length of the current segment to get to mid-point
        anchorage_length_curr = anchorage_length_curr + segment_length_list[ind_for_curr_pipe_id[curr_segment_index]]/2
        # append to list
        anchorage_length[i] = anchorage_length_curr

    # return
    return anchorage_length


# ---
def get_pipe_crossing(
    path_to_def_shp,
    infra_site_data,
    infra_site_data_geom,
    opensra_dir,
    export_dir=None,
    def_type='landslide',
    dem_raster_fpath=None,
    flag_using_state_network=False,
    # export_path
):
    """
    function to determine crossing
    note: a [pipe]line is composed  a series of segments
    """
    
    # ---
    # create transformers for transforming coordinates
    epsg_wgs84 = 4326
    epsg_utm_zone10 = 32610
    transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
    transformer_utmzone10_to_wgs84 = Transformer.from_crs(epsg_utm_zone10, epsg_wgs84)
    
    # ---
    # load shapefile with deformation polygons
    if path_to_def_shp is None:
        poly_exist = False
    else:
        poly_exist = True
        # check if file path to deformation polygons exist
        if not os.path.exists(path_to_def_shp):
            raise ValueError("Path to deformation polygons for pipe crossing does not exist.")
        def_poly_gdf = gpd.read_file(path_to_def_shp,crs=epsg_wgs84)

    # convert site data DataFrame to GeoDataFrame
    if infra_site_data_geom is None:
        geoms=[
            LineString([
                (infra_site_data['LON_BEGIN'][i], infra_site_data['LAT_BEGIN'][i]),
                (infra_site_data['LON_END'][i], infra_site_data['LAT_END'][i])
            ]) for i in range(infra_site_data.shape[0])
        ]
        segment_gdf = gpd.GeoDataFrame(
            infra_site_data,
            crs=epsg_wgs84,
            geometry=geoms,
        )
    else:
        segment_gdf = gpd.GeoDataFrame(
            infra_site_data,
            crs=epsg_wgs84,
            geometry=infra_site_data_geom,
        )
    pipe_id_full = segment_gdf.OBJ_ID.values # complete list of pipe IDs
    segment_id_full = segment_gdf.ID.values # complete list of segment IDs
    segment_index_full = np.asarray(segment_gdf.index) # complete list of segment indices
    
    # ---
    # get crossings between segments and deformation polygons
    if poly_exist:
        crossed_poly_index, crossed_segment_index = segment_gdf.sindex.query_bulk(
            # geometry=list(def_poly_gdf.geometry.boundary),
            geometry=def_poly_gdf.geometry.boundary,
            predicate='intersects'
        )
        # print(crossed_poly_index)
        # print(crossed_segment_index)
        
        # sys.exit()
        # crossed_segment_id = crossed_segment_index + 1
        crossed_segment_id = segment_index_full[crossed_segment_index]
        # get unique deformation polygons that crossed with segments
        def_poly_crossed_unique_gdf = def_poly_gdf.loc[np.unique(crossed_poly_index)].reset_index(drop=True)
        # get subset of segments that crossed deformation polygon
        segment_crossed_gdf = segment_gdf.loc[crossed_segment_index].reset_index(drop=True)
        # get list of unique pipeline IDs from that crossed with deformation polygons
        pipe_id_crossed_unique = np.unique(segment_crossed_gdf.OBJ_ID.values)
    else:
        pipe_id_crossed_unique = np.unique(pipe_id_full)
    
    # ---
    # convert start and endpoints of segments to UTM zone 10 (meters)
    segment_full_begin_utm = np.transpose(
        transformer_wgs84_to_utmzone10.transform(
            segment_gdf.LAT_BEGIN.values,
            segment_gdf.LON_BEGIN.values,
        )
    )
    segment_full_end_utm = np.transpose(
        transformer_wgs84_to_utmzone10.transform(
            segment_gdf.LAT_END.values,
            segment_gdf.LON_END.values,
        )
    )
    if poly_exist:
        segment_crossed_begin_utm = segment_full_begin_utm[crossed_segment_index]
        segment_crossed_end_utm = segment_full_end_utm[crossed_segment_index]
    
    # ---
    # if using state network, hard points have been located in preprocessing
    if flag_using_state_network:
        # no action to perform
        pass
    else:
        # find hard points within unique pipelines with crossings
        hard_points_ind = []
        hard_points = []
        for pipe_id in pipe_id_crossed_unique:
            # get all row indices of segments for current pipe_id
            ind_for_curr_pipe_id = np.where(pipe_id_full==pipe_id)[0]
            # number of segments in current pipeline
            n_segment = len(ind_for_curr_pipe_id)
            # get start and end points of segments for current pipeline
            segment_start_for_curr_pipe_id = segment_full_begin_utm[ind_for_curr_pipe_id]
            segment_end_for_curr_pipe_id = segment_full_end_utm[ind_for_curr_pipe_id]
            # get angles
            angles = dot_product(
                vect1_pt1_in_meters=segment_start_for_curr_pipe_id[:-1],
                vect1_pt2_in_meters=segment_end_for_curr_pipe_id[:-1],
                vect2_pt1_in_meters=segment_start_for_curr_pipe_id[1:],
                vect2_pt2_in_meters=segment_end_for_curr_pipe_id[1:],
                return_angle=True
            )
            # find hard points
            hard_points_ind.append(np.where(angles>40)[0])
            hard_points.append(segment_end_for_curr_pipe_id[hard_points_ind[-1]])
            # add start of pipeline as a hard point
            hard_points_ind[-1] = np.hstack([[0], hard_points_ind[-1]+1])
            hard_points[-1] = np.vstack([segment_start_for_curr_pipe_id[0], hard_points[-1]])
            # add end of pipeline as a hard point
            hard_points_ind[-1] = np.hstack([hard_points_ind[-1], n_segment])
            hard_points[-1] = np.vstack([hard_points[-1], segment_end_for_curr_pipe_id[-1]])
    
    # ---
    # get DEM for unique deformatoin polygon boundaries to determine direction of slip
    if poly_exist:
        def_poly_crossed_unique_gdf_utm = def_poly_crossed_unique_gdf.to_crs(epsg_utm_zone10).copy()
        # get boundary coordinates for deformatoin polygons with crossings for DEM sampling
        n_bound_coord = []
        # wgs84
        bound_coord = []
        for i,each in enumerate(def_poly_crossed_unique_gdf.geometry.boundary):
            if isinstance(each,LineString):
                bound_coord.append(list(each.coords))
            elif isinstance(each,MultiLineString):
                bound_coord.append(list(each.geoms[0].coords))
            n_bound_coord.append(len(bound_coord[-1]))
        bound_coord_flat = np.asarray(sum(bound_coord, [])) # flatten into 1 array
        # utm
        bound_coord_utm = []
        for i,each in enumerate(def_poly_crossed_unique_gdf_utm.geometry.boundary):
            if isinstance(each,LineString):
                bound_coord_utm.append(list(each.coords))
            elif isinstance(each,MultiLineString):
                bound_coord_utm.append(list(each.geoms[0].coords))
        bound_coord_utm_flat = np.asarray(sum(bound_coord_utm, [])) # flatten into 1 array
        # sample elevations around deformation polygon boundaries from DEM raster
        bound_coord_flat_df = LocationData(
            lon=bound_coord_flat[:,0],
            lat=bound_coord_flat[:,1],
            # crs=epsg_utm_zone10,
            crs=epsg_wgs84,
        )
        if dem_raster_fpath is None:
            dem_raster_fpath = os.path.join(
                # opensra_dir,'lib','Datasets','CA_DEM_90m_WGS84_meter','CA_DEM_90m_WGS84_meter.tif'
                opensra_dir,'lib','Datasets','CA_DEM_30m_WGS84_meter','CA_DEM_30m_WGS84_meter.tif'
            )
        if not os.path.exists(dem_raster_fpath):
            raise ValueError("Path to DEM raster does not exist. Check pipe crossing function")
        bound_coord_flat_df.data = bound_coord_flat_df.sample_raster(
            table=bound_coord_flat_df.data,
            fpath=dem_raster_fpath
        )
        dem_basename = get_basename_without_extension(dem_raster_fpath)
        bound_coord_flat_df.data['x'] = bound_coord_utm_flat[:,0] # add utm x
        bound_coord_flat_df.data['y'] = bound_coord_utm_flat[:,1] # add utm y

    # ---
    # get slip direction (azimuth, relative to North) for deformation polygons
    if poly_exist:
        utm_coord_for_max_dem = []
        utm_coord_for_min_dem = []
        for i,each in enumerate(def_poly_crossed_unique_gdf_utm.geometry.boundary):
            n_coord_i = n_bound_coord[i]
            start_ind = sum(n_bound_coord[:i])
            end_ind = start_ind + n_coord_i
            dem_for_geom_i = bound_coord_flat_df.data[dem_basename].iloc[start_ind:end_ind].values.copy()
            utm_x_for_geom_i = bound_coord_flat_df.data.x[start_ind:end_ind].values.copy()
            utm_y_for_geom_i = bound_coord_flat_df.data.y[start_ind:end_ind].values.copy()
            # get location of max and min DEM
            max_dem_i = max(dem_for_geom_i)
            min_dem_i = min(dem_for_geom_i)
            ind_max_dem_i = np.where(dem_for_geom_i==max_dem_i)[0][0]
            ind_min_dem_i = np.where(dem_for_geom_i==min_dem_i)[0][0]
            utm_coord_for_max_dem.append([
                utm_x_for_geom_i[ind_max_dem_i],
                utm_y_for_geom_i[ind_max_dem_i],
            ])
            utm_coord_for_min_dem.append([
                utm_x_for_geom_i[ind_min_dem_i],
                utm_y_for_geom_i[ind_min_dem_i],
            ])
        # convert to numpy array
        utm_coord_for_max_dem = np.asarray(utm_coord_for_max_dem)
        utm_coord_for_min_dem = np.asarray(utm_coord_for_min_dem)
        # get slip direction relative to North
        slip_dir = np.round(get_azimuth(utm_coord_for_max_dem,utm_coord_for_min_dem),1)
        slip_dir[np.isnan(slip_dir)] = 0 # if max and min DEMs are on the same point
        # append to dataframe
        def_poly_crossed_unique_gdf_utm['slip_dir'] = slip_dir
        # get unit slip in dx and dy, to be used for crossing angles later
        slip_vect_dx = []
        slip_vect_dy = []
        for i,each in enumerate(def_poly_crossed_unique_gdf_utm.geometry.boundary):
            dx, dy = get_dx_dy_for_vector_along_azimuth(def_poly_crossed_unique_gdf_utm.slip_dir[i])
            slip_vect_dx.append(dx)
            slip_vect_dy.append(dy)
        def_poly_crossed_unique_gdf_utm['slip_vect_dx'] = slip_vect_dx
        def_poly_crossed_unique_gdf_utm['slip_vect_dy'] = slip_vect_dy

    # ---
    # split deformation polygons (geometries) into scarp (head), body, toe
    if poly_exist:
        scarp_list = []
        body_list = []
        toe_list = []
        upper_list = []
        lower_list = []
        for i,each in enumerate(def_poly_crossed_unique_gdf_utm.geometry.boundary):
            scarp_i, body_i, toe_i, upper_i, lower_i = split_geom_into_scarp_body_toe(
                geom=def_poly_crossed_unique_gdf_utm.geometry[i],
                slip_dir=def_poly_crossed_unique_gdf_utm.slip_dir[i],
                high_point=utm_coord_for_max_dem[i],
                low_point=utm_coord_for_min_dem[i],
                get_upper_and_lower_halves=True
            )
            scarp_list.append(scarp_i)
            body_list.append(body_i)
            toe_list.append(toe_i)
            upper_list.append(upper_i)
            lower_list.append(lower_i)

    # ---
    if poly_exist:
        # remap crossed polygons to dataframe
        crossed_def_poly_map = np.asarray([[
                i,
                def_poly_ind,
                np.where(def_poly_crossed_unique_gdf_utm.FID==def_poly_ind)[0][0]
            ] for i,def_poly_ind in enumerate(crossed_poly_index)])
        def_poly_crossed_gdf_utm = def_poly_crossed_unique_gdf_utm.loc[crossed_def_poly_map[:,2]].reset_index(drop=True)
        
    # ---
    # get all crossings 
    if poly_exist:
        crossings_list = []
        n_crossings_list = []
        segment_crossed_gdf_utm = segment_crossed_gdf.to_crs(epsg_utm_zone10)
        # loop through deformation polygons with crossings
        for i in range(len(crossed_poly_index)):
            # get current geometries
            segment_crossed_i = segment_crossed_gdf_utm.geometry[i]
            def_poly_geom_i = def_poly_crossed_gdf_utm.geometry[i]
            def_poly_bound_geom_i = def_poly_geom_i.boundary
            # first find crossing between segment and deformation geometry.boundary
            crossing_i = def_poly_bound_geom_i.intersection(segment_crossed_i)
            # get number of crossings
            if isinstance(crossing_i,MultiPoint):
                n_crossings_list.append(len(list(crossing_i.geoms)))
            elif isinstance(crossing_i,Point):
                n_crossings_list.append(1)
            crossings_list.append(crossing_i)
        # create geodataframe of crossings
        crossing_summary_gdf_utm = gpd.GeoDataFrame(
            pd.DataFrame(segment_crossed_gdf_utm.drop(columns='geometry')),
            crs=epsg_utm_zone10,
            geometry=crossings_list
        )
        # append to dataframe
        # print(n_crossings_list)
        # print(segment_crossed_gdf_utm.shape)
        crossing_summary_gdf_utm['n_crossings_with_segment'] = n_crossings_list
        crossing_summary_gdf_utm['def_poly_index_crossed'] = crossed_poly_index
        crossing_summary_gdf_utm['crossing_algo_index'] = list(crossing_summary_gdf_utm.index)
        # explode segments with multiple crossings such that each row represents one crossing
        crossing_summary_gdf_utm = crossing_summary_gdf_utm.explode(ignore_index=True)
    else:
        segment_full_mid_utm = np.transpose(
            transformer_wgs84_to_utmzone10.transform(
                segment_gdf.LAT_MID.values,
                segment_gdf.LON_MID.values,
            )
        )
        crossings_list = [Point(segment_full_mid_utm[i]) for i in range(segment_full_mid_utm.shape[0])]
        # create geodataframe of crossings
        crossing_summary_gdf_utm = gpd.GeoDataFrame(
            pd.DataFrame(segment_gdf.drop(columns='geometry')),
            crs=epsg_utm_zone10,
            geometry=crossings_list
        )
        
    # ---
    # go through each crossing and determine crossing angle and section of deformation geometry
    if poly_exist:
        section_crossed_list = []
        section_geom_list = []
        half_crossed_list = []
        half_geom_list = []
        # due to round-offs, apply small buffer on deformation geometry to find crossing
        buffer = 0.01
        # loop throuch each crossing
        for i in range(crossing_summary_gdf_utm.shape[0]):
            # segment ID
            # segment_id_i = crossing_summary_gdf_utm.ID[i]
            # deformation polygon index
            def_poly_index_i = crossing_summary_gdf_utm.def_poly_index_crossed[i]
            # crossing algo index for getting scarp, body, and toe geometries
            crossing_algo_index = crossing_summary_gdf_utm.crossing_algo_index[i]
            # get section and half geometries
            scarp_crossed_i = scarp_list[crossed_def_poly_map[crossing_algo_index,2]]
            body_crossed_i = body_list[crossed_def_poly_map[crossing_algo_index,2]]
            toe_crossed_i = toe_list[crossed_def_poly_map[crossing_algo_index,2]]
            upper_crossed_i = upper_list[crossed_def_poly_map[crossing_algo_index,2]]
            lower_crossed_i = lower_list[crossed_def_poly_map[crossing_algo_index,2]]
            # first find crossing between segment and deformation geometry.boundary
            crossing_i = crossing_summary_gdf_utm.geometry[i]
            # see which section the crossing lies on
            if scarp_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                section_crossed_list.append('scarp')
                section_geom_list.append(scarp_crossed_i)
            elif toe_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                section_crossed_list.append('toe')
                section_geom_list.append(toe_crossed_i)
            # elif body_crossed_i.boundary.buffer(buffer).contains(crossing_i):
            #     section_crossed_list.append('body')
            #     section_geom_list.append(body_crossed_i)
            else:
                # assign to body
                section_crossed_list.append('body')
                section_geom_list.append(body_crossed_i)
            # else:
            #     print(f"Cannot determine if crossing #{i} lies on scarp, body, or toe of deformation polygon #{def_poly_index_i}")
            # see which half the crossing lies on
            if upper_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                half_crossed_list.append('upper')
                half_geom_list.append(upper_crossed_i)
            elif lower_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                half_crossed_list.append('lower')
                half_geom_list.append(lower_crossed_i)
            else:
                print(f"Cannot determine if crossing #{i} lies on upper or lower half of deformation polygon #{def_poly_index_i}")
        # append to dataframe
        crossing_summary_gdf_utm['section_crossed'] = section_crossed_list
        crossing_summary_gdf_utm['which_half'] = half_crossed_list
    else:
        crossing_summary_gdf_utm['section_crossed'] = None
        crossing_summary_gdf_utm['which_half'] = None
        
    # ---
    if poly_exist:
        # convert start and endpoints of segment in UTM zone 10 meters
        crossing_summary_segment_begin_utm = np.transpose(
            transformer_wgs84_to_utmzone10.transform(
                crossing_summary_gdf_utm.LAT_BEGIN.values,
                crossing_summary_gdf_utm.LON_BEGIN.values,
            )
        )
        crossing_summary_segment_end_utm = np.transpose(
            transformer_wgs84_to_utmzone10.transform(
                crossing_summary_gdf_utm.LAT_END.values,
                crossing_summary_gdf_utm.LON_END.values,
            )
        )
        # get crossing coords
        crossing_coords = np.vstack([
            crossing_summary_gdf_utm.geometry.x.values,
            crossing_summary_gdf_utm.geometry.y.values,
        ]).T
        # get slip vector dx and dy
        slip_vect_dx_dy_norm = np.asarray([
            [
                def_poly_crossed_gdf_utm.slip_vect_dx[crossing_summary_gdf_utm.crossing_algo_index[i]],
                def_poly_crossed_gdf_utm.slip_vect_dy[crossing_summary_gdf_utm.crossing_algo_index[i]],
            ]
            for i in range(crossing_summary_gdf_utm.shape[0])
        ])
        # to direction for segment into the crossing:
        # get point = crossing + vector to segment entpoint * dL, with dL ~ 1cm
        direct = []
        d_length = 0.01 # m, 1 cm
        for i in range(len(crossing_coords)):
            crossing_coords_i = crossing_coords[i]
            # first try segment endpoint
            # if point falls in deformation polygon, then direction towards polygon is in the direction towards the endpoint
            node2 = crossing_summary_segment_end_utm[i]
            vect = node2 - crossing_coords_i
            vect_norm = vect/np.sqrt(np.dot(vect,vect))
            pt = Point(crossing_coords_i + vect_norm*d_length)
            if section_geom_list[i].contains(pt):
                direct.append(1)
            else:
                # sanity check with startpoint
                node2 = crossing_summary_segment_begin_utm[i]
                vect = node2 - crossing_coords_i
                vect_norm = vect/np.sqrt(np.dot(vect,vect))
                pt = Point(crossing_coords_i + vect_norm*d_length)
                if section_geom_list[i].contains(pt):
                    direct.append(-1)
                else:
                    raise ValueError("Directionality check for vector from crossing towards deformation polygon failed.")
        reverse_dir = np.asarray(direct)<0
        # get crossing angle as the angle between the slip vector and the vector from the crossing pointing into the deformation polygon
        pt2_vector_for_crossing = crossing_summary_segment_end_utm.copy()
        pt2_vector_for_crossing[reverse_dir,:] = crossing_summary_segment_begin_utm[reverse_dir,:]
        beta_crossing = dot_product(
            vect1_pt1_in_meters=crossing_coords,
            vect1_pt2_in_meters=pt2_vector_for_crossing,
            vect2_pt1_in_meters=crossing_coords,
            vect2_pt2_in_meters=crossing_coords+slip_vect_dx_dy_norm,
        )
        # round and append to dataframe
        beta_crossing = np.round(beta_crossing,decimals=1)
        
    # ---
    # get anchorage lengths
    # initialize
    anchorage_length = []
    length_tol = 1e-1 # m
    if poly_exist:
        # for every segment crossed with deformation polygon, get the nearest hard points and compare to minimum anchorage length
        for i in range(crossing_summary_gdf_utm.shape[0]):
            # current deformation geometry crossed
            curr_def_poly_index = crossing_summary_gdf_utm.def_poly_index_crossed[i]
            curr_def_poly_crossed = def_poly_crossed_gdf_utm.geometry[np.where(def_poly_crossed_gdf_utm.FID==curr_def_poly_index)[0][0]]
            
            # index of current pipeline from crossing algorithm
            crossing_algo_index = crossing_summary_gdf_utm.crossing_algo_index[i]
            # current crossing
            curr_crossing = list(crossing_summary_gdf_utm.geometry[i].coords)[0]
            # pts for current segment
            curr_segment_start_utm = segment_crossed_begin_utm[crossing_algo_index]
            curr_segment_end_utm = segment_crossed_end_utm[crossing_algo_index]
            # current segment index
            curr_segment_index = crossing_summary_gdf_utm.SUB_SEGMENT_ID[i]-1
            
            # pipeline ID
            curr_pipe_id = crossing_summary_gdf_utm.OBJ_ID[i]
            # get hard points along current pipeline
            where_curr_pipe_id = np.where(pipe_id_crossed_unique==curr_pipe_id)[0][0]
            curr_hard_points_ind = hard_points_ind[where_curr_pipe_id]
            curr_hard_points = hard_points[where_curr_pipe_id]
            # get all row indices of segments for current pipe_id
            ind_for_curr_pipe_id = np.where(pipe_id_full==curr_pipe_id)[0]
            # get start and end points of segments for current pipeline
            segment_start_for_curr_pipe_id = segment_full_begin_utm[ind_for_curr_pipe_id]
            segment_end_for_curr_pipe_id = segment_full_end_utm[ind_for_curr_pipe_id]
            
            # find nearest hard point from start and end side
            nearest_hard_pt_ind_start = curr_hard_points_ind[np.where(curr_hard_points_ind<=curr_segment_index)[0][-1]]
            nearest_hard_pt_ind_end = curr_hard_points_ind[np.where(curr_hard_points_ind>curr_segment_index)[0][0]]
            nearest_hard_pt_start = curr_hard_points[np.where(curr_hard_points_ind<=curr_segment_index)[0][-1]]
            nearest_hard_pt_end = curr_hard_points[np.where(curr_hard_points_ind>curr_segment_index)[0][0]]
            
            # get nodes for segments from crossing to the nearest hard point on start side
            start_nodes_from_crossing_to_hard_pt_on_start_side = np.vstack([
                segment_end_for_curr_pipe_id[nearest_hard_pt_ind_start:curr_segment_index,:],
                curr_crossing,
            ])
            end_nodes_from_crossing_to_hard_pt_on_start_side = \
                segment_start_for_curr_pipe_id[nearest_hard_pt_ind_start:curr_segment_index+1,:]
            start_nodes_from_crossing_to_hard_pt_on_start_side = np.flipud(start_nodes_from_crossing_to_hard_pt_on_start_side)
            end_nodes_from_crossing_to_hard_pt_on_start_side = np.flipud(end_nodes_from_crossing_to_hard_pt_on_start_side)
            # get nodes for segments from crossing to the nearest hard point on end side
            start_nodes_from_crossing_to_hard_pt_on_end_side = np.vstack([
                curr_crossing,
                segment_start_for_curr_pipe_id[curr_segment_index+1:nearest_hard_pt_ind_end,:]
            ])
            end_nodes_from_crossing_to_hard_pt_on_end_side = \
                segment_end_for_curr_pipe_id[curr_segment_index:nearest_hard_pt_ind_end,:]
            
            # make lines from crossing to nearest hard points on start and end side
            lines_for_start_side = MultiLineString([
                LineString([
                    start_nodes_from_crossing_to_hard_pt_on_start_side[i],
                    end_nodes_from_crossing_to_hard_pt_on_start_side[i]
                ])
                for i in range(len(start_nodes_from_crossing_to_hard_pt_on_start_side))
            ])
            lines_for_end_side = MultiLineString([
                LineString([
                    start_nodes_from_crossing_to_hard_pt_on_end_side[i],
                    end_nodes_from_crossing_to_hard_pt_on_end_side[i]
                ])
                for i in range(len(start_nodes_from_crossing_to_hard_pt_on_end_side))
            ])
            
            # get length of pipe segments from crossing to the nearest hard points on start and end side
            # if the lines of the segments cross the deformation geometry, then use half of pipeline length within the polygon as controlling length
            # start side
            length_to_hard_point_start_side = lines_for_start_side.length
            length_def_poly_overlap_start_side = lines_for_start_side.intersection(curr_def_poly_crossed).length
            if length_def_poly_overlap_start_side > length_tol: # allow some tolerance
                length_start_side = min(length_to_hard_point_start_side,length_def_poly_overlap_start_side/2)
            else:
                length_start_side = length_to_hard_point_start_side
            # end side
            length_to_hard_point_end_side = lines_for_end_side.length
            length_def_poly_overlap_end_side = lines_for_end_side.intersection(curr_def_poly_crossed).length
            if length_def_poly_overlap_end_side > length_tol:
                length_end_side = min(length_to_hard_point_end_side,length_def_poly_overlap_end_side/2)
            else:
                length_end_side = length_to_hard_point_end_side
            # determine anchorlage length as the shorter of the lengths on start and end side
            anchorage_length.append(min(length_start_side,length_end_side))
        
    else:
        # see if infra data contains the column 'L_ANCHOR_FULL_METER', if so, use this values
        if 'L_ANCHOR_FULL_METER' in crossing_summary_gdf_utm:
            anchorage_length = crossing_summary_gdf_utm.L_ANCHOR_FULL_METER.values
        else:
            # get inputs
            sub_segment_index_list = crossing_summary_gdf_utm.SUB_SEGMENT_ID.values - 1
            segment_length_list = crossing_summary_gdf_utm.LENGTH_KM.values * 1000 # convert to meters
            # run function to calculate anchorage length from hard points
            anchorage_length = get_anchorage_length_full_system(
                pipe_id_full=pipe_id_full,
                sub_segment_index_list=sub_segment_index_list,
                segment_length_list=segment_length_list,
                pipe_id_crossed_unique=pipe_id_crossed_unique,
                hard_points_ind=hard_points_ind,
            )
    
    # round and append to dataframe
    anchorage_length = np.round(anchorage_length,decimals=1)
    crossing_summary_gdf_utm['l_anchor'] = np.maximum(anchorage_length,1e-2) # limit to 1 cm
    
    # ---
    # convert crossing_summary_gdf_utm back to lat lon
    crossing_summary_gdf = crossing_summary_gdf_utm.to_crs(epsg_wgs84)
    crossing_summary_gdf['crossing_lon'] = crossing_summary_gdf.geometry.x.values
    crossing_summary_gdf['crossing_lat'] = crossing_summary_gdf.geometry.y.values
    
    # ---
    # probability of crossing
    if path_to_def_shp is not None:
        crossing_summary_gdf[f'prob_crossing'] = 1
    else:
        crossing_summary_gdf[f'prob_crossing'] = 0.25
    
    # ---
    # append angles and geometries
    # for landslide, if deformation polygons exist, psi and theta are dependent on sampled beta
    if def_type == 'landslide':
        if poly_exist:
            crossing_summary_gdf['beta_crossing'] = beta_crossing
            crossing_summary_gdf['psi_dip'] = "sampling_dependent"
            # crossing_summary_gdf['theta_rake'] = "sampling_dependent"
        else:
            crossing_summary_gdf['beta_crossing'] = 135 # for SSComp
            crossing_summary_gdf['psi_dip'] = 15 # not used
            # crossing_summary_gdf['theta_rake'] = 45 # not used
    # for lateral spread, if deformation polygons exist, psi and theta are dependent on sampled beta
    elif def_type == 'lateral_spread':
        crossing_summary_gdf['psi_dip'] = 45 # all cases, used or unused
        if poly_exist:
            crossing_summary_gdf['beta_crossing'] = beta_crossing
        else:
            crossing_summary_gdf['beta_crossing'] = 90 # not used
    # for settlement, psi and theta can be assigned now, since only "normal-slip" is assumed for slip mechanism
    elif def_type == 'settlement':
        crossing_summary_gdf['psi_dip'] = 75 # always, for settlement
        if poly_exist:
            crossing_summary_gdf['beta_crossing'] = beta_crossing
            # crossing_summary_gdf['theta_rake'] = "sampling_dependent"
        else:
            crossing_summary_gdf['beta_crossing'] = 90
            # crossing_summary_gdf['theta_rake'] = 45 # not used
    elif def_type == 'surface_fault_rupture':
        raise ValueError(f'The hazard "{def_type}" is not implemented at this point')
    else:
        raise ValueError(f'The hazard "{def_type}" is not a hazard under this study')
    
    #################
    # add a new row to induce repeating crossings
    # crossing_summary_gdf.loc[crossing_summary_gdf.shape[0]] = crossing_summary_gdf.loc[crossing_summary_gdf.shape[0]-1]
    # crossing_summary_gdf.loc[crossing_summary_gdf.shape[0]-1,'l_anchor'] = crossing_summary_gdf.loc[crossing_summary_gdf.shape[0]-1,'l_anchor'] * 2
    #################
    
    # sort by ID column
    crossing_summary_gdf.sort_values('ID',inplace=True)
    
    # ---
    # export crossing summary table
    crossing_summary_gdf.drop(columns='geometry').to_csv(
        # os.path.join(export_dir,f'site_data_{def_type.upper()}_CROSSINGS_ONLY.csv'),
        os.path.join(export_dir,f'site_data_PROCESSED_CROSSING_ONLY.csv'),
        index=False
    )
    
    # ---
    # return
    return crossing_summary_gdf