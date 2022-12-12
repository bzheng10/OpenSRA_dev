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
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from shapely.affinity import rotate
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString
from shapely.ops import split, unary_union, nearest_points
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
from src.site.geodata import LocationData
from src.site.site_util import make_list_of_linestrings
from src.util import get_basename_without_extension


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
        # avoid roundoffs such that cos_angle is outside of -1 and 1 (e.g., 1.0000000000000002)
        cos_angle = np.maximum(np.minimum(cos_angle,1),-1)
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
def azimuth_lines(ang1, ang2, angle_type=None):
    """
    Tom Clifford: angles clockwise from north, returns acute angle
    - modified for vectors/arrays
    """
    theta = np.abs(ang1 - ang2)
    theta[theta>180] = theta[theta>180] - 360
    theta[theta<-180] = theta[theta<-180] + 360
    if angle_type == 'acute':
        theta[theta>90] = theta[theta>90] -90
    theta = np.abs(theta)
    return theta


# ---
def get_rake_azimuth(strike,dip,rake):
    """
    Tom Clifford: returns rake-azimuth from basic fault angles
    - modified for vectors/arrays
    """
    map_scale = np.cos(np.radians(dip))
    # for rake >= 90 and <= 90
    rake_from_strike = rake * map_scale
    rake_azimuth = rake - rake_from_strike #substract angle between rake and strike
    # for rake > 90
    cond = rake>90
    if True in cond:
        rake_from_strike[cond] = (180-rake[cond]) * map_scale[cond]
        rake_azimuth[cond] = strike[cond] + 180 + rake_from_strike[cond]
    # for rake < -90
    cond = rake<-90
    if True in cond:
        rake_from_strike[cond] = (-180-rake[cond]) * map_scale[cond]
        rake_azimuth[cond] = strike[cond] + 180 + rake_from_strike[cond]
    # correction 0-360
    rake_azimuth[rake_azimuth>360] = rake_azimuth[rake_azimuth>360] - 360
    rake_azimuth[rake_azimuth<0] = rake_azimuth[rake_azimuth<0] + 360
    # return
    return rake_azimuth


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


# ---
def split_geom_into_halves_without_high_low_points(
    geom, # shapely geometry, in UTM (meters)
    slip_dir, # deg, relative to North (i.e., azimuth)
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
    ybottom = extent_rotate[1]
    xright = extent_rotate[2]
    ytop = extent_rotate[3]
    
    # y0 top of extent, dy = top - bottom of extent
    y0 = ytop
    yend = ybottom
    
    # find y-coord along vertical extent at cutoffs
    dy = yend - y0
    
    # split into upper and lower halves
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
    return upper, lower


@njit(
    fastmath=True,
    cache=True
)
def get_anchorage_length_full_system_landslide_or_liq(
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
def get_pipe_crossing_fault_rup(
    opensra_dir,
    processed_input_dir,
    im_dir,
    infra_site_data,
    avail_data_summary,
    reduced_ucerf_fpath,
    fault_disp_model='PetersenEtal2011',
    im_source='UCERF',
    infra_site_data_geom=None,
):
    """
    function to determine crossing for fault rupture
    """
    
    # if PetersenEtal2011 == 'Thompson2021' or PetersenEtal2011 == 'WellsCoppersmith1994':
        # raise ValueError('Missing implementation for pipe crossing with UCERF for "Thompson2021" and "WellsCoppersmith1994"')
    
    if not im_source == 'UCERF':
        raise ValueError('Surface fault rupture currently set up for QFault hazard zones from LCI, which interacts with UCERF3')
        
    else:
        # ---
        # create transformers for transforming coordinates
        epsg_wgs84 = 4326
        epsg_utm_zone10 = 32610
        transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
        transformer_utmzone10_to_wgs84 = Transformer.from_crs(epsg_utm_zone10, epsg_wgs84)

        # read Norm's reduced UCERF scenarios
        rupture_table = read_file(reduced_ucerf_fpath,crs=epsg_wgs84)
        rupture_table_utm = rupture_table.to_crs(epsg_utm_zone10) # convert to UTM (m)

        # ---
        # convert site data DataFrame to GeoDataFrame
        if infra_site_data_geom is None:
            segment_gdf = GeoDataFrame(
                infra_site_data,
                crs=epsg_wgs84,
                geometry=make_list_of_linestrings(
                    pt1_x=infra_site_data['LON_BEGIN'].values,
                    pt1_y=infra_site_data['LAT_BEGIN'].values,
                    pt2_x=infra_site_data['LON_END'].values,
                    pt2_y=infra_site_data['LAT_END'].values,
                ),
            )
        else:
            segment_gdf = GeoDataFrame(
                infra_site_data,
                crs=epsg_wgs84,
                geometry=infra_site_data_geom,
            )
        pipe_id_full = segment_gdf.OBJ_ID.values # complete list of pipe IDs
        segment_id_full = segment_gdf.ID.values # complete list of segment IDs
        segment_index_full = np.asarray(segment_gdf.index) # complete list of segment indices
        
        # ---
        # get segment start and end arrays
        segment_start = np.vstack([
            segment_gdf.LON_BEGIN.values,
            segment_gdf.LAT_BEGIN.values,
            ]).T
        segment_end = np.vstack([
            segment_gdf.LON_END.values,
            segment_gdf.LAT_END.values,
        ]).T
        # convert to UTM (m)
        segment_full_begin_utm = np.asarray(transformer_wgs84_to_utmzone10.transform(
            segment_start[:,1],segment_start[:,0]
        )).T
        segment_full_end_utm = np.asarray(transformer_wgs84_to_utmzone10.transform(
            segment_end[:,1],segment_end[:,0]
        )).T
        # make segment gdf in UTM
        segment_start_utm_geom = points_from_xy(segment_full_begin_utm[:,0],segment_full_begin_utm[:,1])
        segment_end_utm_geom = points_from_xy(segment_full_end_utm[:,0],segment_full_end_utm[:,1])
        gdf_segment_utm = GeoDataFrame(
            segment_gdf.drop('geometry',axis=1),
            crs=epsg_utm_zone10,
            geometry=segment_start_utm_geom.union(segment_end_utm_geom).convex_hull
        )
        # append x and y values
        gdf_segment_utm['x1'] = segment_full_begin_utm[:,0]
        gdf_segment_utm['y1'] = segment_full_begin_utm[:,1]
        gdf_segment_utm['x2'] = segment_full_end_utm[:,0]
        gdf_segment_utm['y2'] = segment_full_end_utm[:,1]
        
        # ---
        # storing indices of segment and qfault crossed
        segment_by_qfault = {}
        # segment_crossed_by_rupture = {}
        segment_id_crossed_by_rupture = {}
        prob_crossing_by_rupture = {}
        norm_dist_by_rupture = {}
        # export qfault
        gdf_qfault_crossed_export = {}

        # for each fault displacement hazard shapefile
        for each in['primary','secondary']:
            # ---
            # initialize for storage
            # segment_crossed_by_rupture[each] = {}
            segment_id_crossed_by_rupture[each] = {}
            prob_crossing_by_rupture[each] = {}
            norm_dist_by_rupture[each] = {}
            
            # ---
            # string for fault reference in available dataset json
            qfault_str = f'qfault_{each}'
            # get qfault file path
            qfault_fpath = os.path.join(opensra_dir,
                avail_data_summary['Parameters'][qfault_str]['Datasets']['Set1']['Path']
            )
            qfault_crs = avail_data_summary['Parameters'][qfault_str]['Datasets']['Set1']['CRS']
            # read qfault shapefile
            gdf_qfault = read_file(qfault_fpath,crs=qfault_crs)
            gdf_qfault_utm = gdf_qfault.to_crs(epsg_utm_zone10) # convert to utm
            # get crossing between segments and qfault
            qfault_crossed, segment_crossed = gdf_segment_utm.sindex.query_bulk(
                gdf_qfault_utm.geometry,predicate='intersects')
            
            # remove repeating segment cases - there are repeating sections in the qfault inventory
            qfault_crossed_list = list(qfault_crossed)
            segment_crossed_list = list(segment_crossed)
            qfault_crossed = []
            segment_crossed = []
            for i,seg in enumerate(segment_crossed_list):
                if not seg in segment_crossed:
                    segment_crossed.append(seg)
                    qfault_crossed.append(qfault_crossed_list[i])

            # ------------------------------------
            # this section is used to get beta crossings
            # get list intersections for getting beta crossing
            intersections = GeoSeries(gdf_qfault_utm.loc[qfault_crossed].geometry.values).intersection(
                GeoSeries(gdf_segment_utm.loc[segment_crossed].geometry.values)).values
            # loop through intersections to get 1 point on line that is representative of line
            crossing_coords = []
            for i in range(len(intersections)):
                if isinstance(intersections[i],MultiLineString):
                    geoms = list(intersections[i].geoms)[0]
                    geom_to_use = list(intersections[i].geoms)[0]
                    crossing_coords.append(list(geom_to_use.coords)[0])
                elif isinstance(intersections[i],LineString):
                    crossing_coords.append(list(intersections[i].coords)[0])
                elif isinstance(intersections[i],Point):
                    crossing_coords.append(qfault_crossed[i])
                else:
                    print(type(intersections[i]))
                    raise ValueError('Invalid type for intersections')
            crossing_coords = np.asarray(crossing_coords)
            
            # ---
            # get subset of qfaults and segments with crossings
            gdf_qfault_crossed = gdf_qfault_utm.loc[qfault_crossed].copy()
            segment_by_qfault[each] = gdf_segment_utm.loc[segment_crossed].copy()
            
            # get fault angles
            strike = gdf_qfault_crossed.DipDirec_1.values - 90
            dip = gdf_qfault_crossed.AveDip_1.values
            rake = gdf_qfault_crossed.AveRake_1.values
            # get azimuth (slip dir)
            azimuth = get_rake_azimuth(strike,dip,rake)
            # get unit slip in dx and dy, to be used for crossing angles later
            slip_vect_dx_dy_norm = np.transpose(get_dx_dy_for_vector_along_azimuth(azimuth))
            
            # ---
            # get start and end points for segments crossed
            crossing_start = segment_full_begin_utm[segment_crossed]
            crossing_end = segment_full_end_utm[segment_crossed]
            # to direction for segment into the crossing:
            # get point = crossing + vector to segment entpoint * dL, with dL ~ 1cm
            direct = []
            d_length = 0.01 # m, 1 cm
            for i in range(len(crossing_coords)):
                crossing_coords_i = crossing_coords[i]
                # first try segment endpoint
                # if point falls in deformation polygon, then direction towards polygon is in the direction towards the endpoint
                node2 = crossing_end[i]
                vect = node2 - crossing_coords_i
                vect_norm = vect/np.sqrt(np.dot(vect,vect))
                pt = Point(crossing_coords_i + vect_norm*d_length)
                geom_list_to_use = gdf_qfault_crossed.geometry.values
                if geom_list_to_use[i].contains(pt):
                    direct.append(1)
                else:
                    # sanity check with startpoint
                    node2 = crossing_start[i]
                    vect = node2 - crossing_coords_i
                    vect_norm = vect/np.sqrt(np.dot(vect,vect))
                    pt = Point(crossing_coords_i + vect_norm*d_length)
                    if geom_list_to_use[i].contains(pt):
                        direct.append(-1)
                    else:
                        raise ValueError("Directionality check for vector from crossing towards deformation polygon failed.")
            reverse_dir = np.asarray(direct)<0
            # get crossing angle as the angle between the slip vector and the vector from the crossing pointing into the deformation polygon
            pt2_vector_for_crossing = crossing_end.copy()
            pt2_vector_for_crossing[reverse_dir,:] = crossing_start[reverse_dir,:]
            beta_crossing = dot_product(
                vect1_pt1_in_meters=crossing_coords,
                vect1_pt2_in_meters=pt2_vector_for_crossing,
                vect2_pt1_in_meters=crossing_coords,
                vect2_pt2_in_meters=crossing_coords+slip_vect_dx_dy_norm,
            )
            # round and append to dataframe
            beta_crossing = np.round(beta_crossing,decimals=1)            
            
            # ------------------------------------
            # this section is used to get normalized distance for the Petersen method
            # initialize lists to store various items
            scenario_crossed_list = [] # scenario considered as crossed
            scenario_id_crossed_list = [] # scenario considered as crossed
            norm_dist_list = [] # tracking normalized distance from endpoint
            prob_crossing_list = [] # probability of crossing
            
            # -- used for Petersen et al. (2011) only
            if not fault_disp_model == 'PetersenEtal2011':
                # for other fault rupture models (e.g., Wells & Coppersmith), set distance metrics to null
                # every crossing pair of segment and fault
                for i in range(len(qfault_crossed)):
                    scenario_crossed_list.append([])
                    scenario_id_crossed_list.append([])
                    norm_dist_list.append([])
                    prob_crossing_list.append([])
                
            else:
                # get unique parent IDs with crossings
                unique_parent_id_crossed = np.unique(
                    gdf_qfault_crossed.parentSe_1.values)
                # get strike
                strike = gdf_qfault_crossed.DipDirec_1.values - 90
                # get segment start and end arrays
                segment_crossed_start = np.vstack([
                    segment_by_qfault[each].x1.values,
                    segment_by_qfault[each].y1.values,
                    ]).T
                segment_crossed_end = np.vstack([
                    segment_by_qfault[each].x2.values,
                    segment_by_qfault[each].y2.values,
                ]).T
                # get crossing angles
                segment_crossed_azimuth = get_azimuth(segment_crossed_start,segment_crossed_end)
                crossing_theta = azimuth_lines(segment_crossed_azimuth, strike)
                # probability of crossing
                norm_seg_len = segment_by_qfault[each].LENGTH_KM.values*1000/200
                prob_crossing = norm_seg_len*np.abs(np.sin(np.radians(crossing_theta)))
                prob_crossing = np.round(prob_crossing,decimals=4)
                
                # get fault scenarios from reduced UCERF for unique parent IDs crossed
                rows_with_parent_ids_in_norm_scenario = {
                    parent_id: np.where(rupture_table_utm.ParentID.values==parent_id)[0]
                    for parent_id in unique_parent_id_crossed
                }
                
                # every crossing pair of segment and fault
                for i in range(len(qfault_crossed)):
                    # get the geometry of the segment
                    segment_crossed_i = segment_by_qfault[each].iloc[i].copy().geometry
                    # segment ID
                    segment_id_crossed_i = segment_by_qfault[each].ID.iloc[i]
                    # parentID crossed
                    parent_id_crossed_i = gdf_qfault_crossed.iloc[i].parentSe_1
                    # get scenarios from Norm for parent ID crossed
                    rows_for_parent_id_crossed_i = rows_with_parent_ids_in_norm_scenario[parent_id_crossed_i]
                    # get subset of scenarios
                    gdf_norm_scenario_utm_crossed_i = rupture_table_utm.iloc[rows_for_parent_id_crossed_i].copy()
                    # initialize lists to track for current segment with UCERF scenarios
                    scenario_crossed_seg_i = []
                    scenario_id_crossed_seg_i = []
                    norm_dist_seg_i = []
                    prob_crossing_seg_i = []
                    # for each scenario
                    for j in range(gdf_norm_scenario_utm_crossed_i.shape[0]):
                        # fault index
                        fault_ind_j = gdf_norm_scenario_utm_crossed_i.index.values[j]
                        # get fault trace geometry
                        fault_trace_j = gdf_norm_scenario_utm_crossed_i.geometry.iloc[j]
                        # convert string of coordinates into list
                        fault_trace_j_arr = np.asarray(fault_trace_j.coords)
                        fault_trace_j_length = fault_trace_j.length # length in meters
                        nearest_pt_j = np.asarray(list(nearest_points(segment_crossed_i,fault_trace_j))[1].coords[0])
                        
                        # see if nearest point is the end points of fault
                        if (nearest_pt_j[0] == fault_trace_j_arr[0][0] and \
                            nearest_pt_j[1] == fault_trace_j_arr[0][1]) or \
                            (nearest_pt_j[0] == fault_trace_j_arr[-1][0] and \
                            nearest_pt_j[1] == fault_trace_j_arr[-1][1]):
                            pass
                        else:
                            # if False, consider scenario as crossed for current segment i
                            scenario_crossed_seg_i.append(gdf_norm_scenario_utm_crossed_i.index[j])
                            scenario_id_crossed_seg_i.append(gdf_norm_scenario_utm_crossed_i.EventID.iloc[j])
                            # to get distance from each end of the fault trace
                            # -- note that due to round off errors, nearest point does not actually fall on line
                            # -- and so cannot split fault trace for nearest point to get the individual sections.
                            # -- instead, create buffers from each end of fault and crop fault trace outside of buffer.
                            # -- then find length of fault trace segments remaining
                            # 1) get direct dist from endpoints on fault trace to nearest point
                            dir_dist_start = (
                                (fault_trace_j_arr[0][0]-nearest_pt_j[0])**2 + \
                                (fault_trace_j_arr[0][1]-nearest_pt_j[1])**2
                            )**0.5
                            dir_dist_end = (
                                (fault_trace_j_arr[-1][0]-nearest_pt_j[0])**2 + \
                                (fault_trace_j_arr[-1][1]-nearest_pt_j[1])**2
                            )**0.5
                            # 2) from each end of the fault trace, create a buffer
                            fault_start_buffer = Point(fault_trace_j_arr[0]).buffer(dir_dist_start)
                            fault_end_buffer = Point(fault_trace_j_arr[-1]).buffer(dir_dist_end)
                            # 3) crop fault trace by buffers at each end
                            fault_trace_j_from_start = fault_trace_j.intersection(fault_start_buffer)
                            fault_trace_j_from_end = fault_trace_j.intersection(fault_end_buffer)
                            # 4) get short of two trace sections and normalize by total length
                            norm_dist_seg_i.append(
                                np.round(
                                    min(fault_trace_j_from_start.length,fault_trace_j_from_end.length)/fault_trace_j_length,
                                    decimals=4
                                )
                            )
                            # all UCERF scenarios will have same prob crossing determined from qfault
                            prob_crossing_seg_i.append(prob_crossing[i])
                            # also append to dictionary that tracks segment crossed ordered by rupture
                            if not fault_ind_j in segment_id_crossed_by_rupture[each]:
                                # segment_crossed_by_rupture[each][fault_ind_j] = [segment_crossed[i]]
                                segment_id_crossed_by_rupture[each][fault_ind_j] = [segment_id_crossed_i]
                                prob_crossing_by_rupture[each][fault_ind_j] = [prob_crossing[i]]
                                norm_dist_by_rupture[each][fault_ind_j] = [norm_dist_seg_i[-1]]
                            else:
                                # segment_crossed_by_rupture[each][fault_ind_j].append(segment_crossed[i])
                                segment_id_crossed_by_rupture[each][fault_ind_j].append(segment_id_crossed_i)
                                prob_crossing_by_rupture[each][fault_ind_j].append(prob_crossing[i])
                                norm_dist_by_rupture[each][fault_ind_j].append(norm_dist_seg_i[-1])
                    
                    # append to overall list
                    scenario_crossed_list.append(scenario_crossed_seg_i)
                    scenario_id_crossed_list.append(scenario_id_crossed_seg_i)
                    norm_dist_list.append(norm_dist_seg_i)
                    prob_crossing_list.append(prob_crossing_seg_i)
            
            # convert crossing_coords to lat lon
            crossing_coords_lat, crossing_coords_lon = transformer_utmzone10_to_wgs84.transform(
                crossing_coords[:,0],crossing_coords[:,1]
            )
            
            # append to dataframe
            segment_by_qfault[each][f'qfault_crossed'] = qfault_crossed
            segment_by_qfault[each][f'psi_dip'] = np.round(dip,decimals=1)
            segment_by_qfault[each][f'theta_rake'] = np.round(rake,decimals=1)
            segment_by_qfault[each][f'rake_azimuth'] = np.round(azimuth,decimals=1)
            segment_by_qfault[each][f'crossing_lon'] = crossing_coords_lon
            segment_by_qfault[each][f'crossing_lat'] = crossing_coords_lat
            segment_by_qfault[each][f'beta_crossing'] = beta_crossing
            segment_by_qfault[each][f'prob_crossing'] = prob_crossing_list
            segment_by_qfault[each][f'event_ind_crossed'] = scenario_crossed_list
            segment_by_qfault[each][f'event_id_crossed'] = scenario_id_crossed_list
            segment_by_qfault[each][f'norm_dist'] = norm_dist_list
            
            # ---
            # to get anchorage length
            # see if infra data contains the column 'L_ANCHOR_FULL_METER', if so, use this values
            # this column contains anchorage length that has been precomputed
            if 'L_ANCHOR_FULL_METER' in segment_by_qfault[each]:
                anchorage_length = segment_by_qfault[each].L_ANCHOR_FULL_METER.values
            else:
                # ---
                # get unique pipe ID
                pipe_id_crossed_unique = np.unique(segment_by_qfault[each].OBJ_ID.values)
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
                
                # get anchorage lengths
                # initialize
                anchorage_length = []
                length_tol = 1e-1 # m
                segment_crossed_start_utm = segment_by_qfault[each][['x1','y1']].values
                segment_crossed_end_utm = segment_by_qfault[each][['x2','y2']].values
                
                # for every segment crossed with deformation polygon, get the nearest hard points and compare to minimum anchorage length
                for i in range(segment_by_qfault[each].shape[0]):
                    # current crossing
                    curr_crossing = list(segment_by_qfault[each][f'crossing_coord_{each}'])[i]
                    # pts for current segment
                    curr_segment_start_utm = segment_crossed_start_utm[i]
                    curr_segment_end_utm = segment_crossed_end_utm[i]
                    # current segment index
                    curr_segment_index = segment_by_qfault[each].SUB_SEGMENT_ID[i]-1
                    
                    # pipeline ID
                    curr_pipe_id = segment_by_qfault[each].OBJ_ID[i]
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
                    lines_for_start_side = MultiLineString(
                        make_list_of_linestrings(
                            pt1_x=start_nodes_from_crossing_to_hard_pt_on_start_side[:,0],
                            pt1_y=start_nodes_from_crossing_to_hard_pt_on_start_side[:,1],
                            pt2_x=end_nodes_from_crossing_to_hard_pt_on_start_side[:,0],
                            pt2_y=end_nodes_from_crossing_to_hard_pt_on_start_side[:,1],
                        )
                    )
                    lines_for_end_side = MultiLineString(
                        make_list_of_linestrings(
                            pt1_x=start_nodes_from_crossing_to_hard_pt_on_end_side[:,0],
                            pt1_y=start_nodes_from_crossing_to_hard_pt_on_end_side[:,1],
                            pt2_x=end_nodes_from_crossing_to_hard_pt_on_end_side[:,0],
                            pt2_y=end_nodes_from_crossing_to_hard_pt_on_end_side[:,1],
                        )
                    )            
                    # get length of pipe segments from crossing to the nearest hard points on start and end side
                    # start side
                    length_start_side = lines_for_start_side.length
                    # end side
                    length_end_side = lines_for_end_side.length
                    # determine anchorlage length as the shorter of the lengths on start and end side
                    anchorage_length.append(min(length_start_side,length_end_side))
            # add to dataframe
            # segment_by_qfault[each][f'l_anchor_{each}'] = anchorage_length
            segment_by_qfault[each]['l_anchor'] = anchorage_length
            
            # append column to denote if haz zone is primary or secondary
            segment_by_qfault[each]['qfault_type'] = each
            
            # store qfault
            gdf_qfault_crossed_export[each] = gdf_qfault_crossed.to_crs(epsg_wgs84).copy().reset_index()
        
        # ---
        # clean up segment crossing summary before export
        # 1) remove segments not tied to any UCERF scenario
        for each in ['primary','secondary']:
            inds = segment_by_qfault[each].index.values
            inds_to_keep = []
            # for i,scen in enumerate(segment_by_qfault[each][f'scenario_crossed_{each}'].values):
            for i,scen in enumerate(segment_by_qfault[each][f'event_ind_crossed'].values):
                if len(scen)>0:
                    inds_to_keep.append(inds[i])
            segment_by_qfault[each] = segment_by_qfault[each].loc[inds_to_keep]
        ########################
        # # 2) join dataframes of segment crossed table for primary and secondary
        # common_headers = list(set(list(segment_by_qfault['primary'].columns)).intersection(set(list(segment_by_qfault['secondary'].columns))))
        # all_segments_crossed = segment_by_qfault['primary'].merge(segment_by_qfault['secondary'], on=common_headers)
        # # drop geometry to make use of pd's concat function
        # seg_crossed_geoms = all_segments_crossed.geometry # keep a copy to make shapefile later
        # seg_crossed_geoms_crs = all_segments_crossed.crs
        # all_segments_crossed = pd.DataFrame(all_segments_crossed.drop('geometry',axis=1))
        # all_segments_crossed = pd.concat([
        #     all_segments_crossed,
        #     segment_by_qfault['primary'].drop('geometry',axis=1),
        #     segment_by_qfault['secondary'].drop('geometry',axis=1)
        # ])
        # # all_segments_crossed = all_segments_crossed.sort_values('ID').reset_index(drop=True)
        # # merge anchorlage length column
        # anchorage_length_merge = all_segments_crossed.l_anchor_primary.values
        # anchorage_length_merge[np.isnan(anchorage_length_merge)] = \
        #     all_segments_crossed.l_anchor_secondary.values[np.isnan(anchorage_length_merge)]
        # anchorage_length_merge = np.round(anchorage_length_merge,decimals=1)
        # all_segments_crossed['l_anchor'] = np.maximum(anchorage_length_merge,1e-2) # limit to 1 cm
        # # reset index and sort by ID column
        # all_segments_crossed.reset_index(drop=True,inplace=True)
        # all_segments_crossed.sort_values('ID',inplace=True)
        # # to flag crossing angles to get distribution later
        # all_segments_crossed['beta_crossing'] = 'sampling_dependent'
        ###########################################
        # 2) join dataframes of segment crossed table for primary and secondary
        all_segments_crossed = pd.concat([
            segment_by_qfault['primary'].drop('geometry',axis=1),
            segment_by_qfault['secondary'].drop('geometry',axis=1)
        ],axis=0)
        seg_crossed_geoms = np.hstack([
            segment_by_qfault['primary'].geometry.values,
            segment_by_qfault['secondary'].geometry.values
        ])
        seg_crossed_geoms_crs = segment_by_qfault['primary'].crs
        # sort by ID column and reset index
        all_segments_crossed.sort_values('ID',inplace=True)
        all_segments_crossed.reset_index(drop=True,inplace=True)
        
        # expand lists by repeating event IDs crossed
        all_segments_crossed_expand = pd.DataFrame(None,columns=all_segments_crossed.columns)
        prob_crossing_expand = []
        event_ind_expand = []
        event_id_expand = []
        norm_dist_expand = []
        geoms_expand = []
        for i in range(all_segments_crossed.shape[0]):
            n_event_crossed_for_seg_i = len(all_segments_crossed.event_id_crossed.values[i])
            for j in range(n_event_crossed_for_seg_i):
                prob_crossing_expand.append(all_segments_crossed.prob_crossing.iloc[i][j])
                event_ind_expand.append(all_segments_crossed.event_ind_crossed.iloc[i][j])
                event_id_expand.append(all_segments_crossed.event_id_crossed.iloc[i][j])
                norm_dist_expand.append(all_segments_crossed.norm_dist.iloc[i][j])
                geoms_expand.append(seg_crossed_geoms[i])
                all_segments_crossed_expand.loc[all_segments_crossed_expand.shape[0]] = \
                    all_segments_crossed.loc[i].values
        # append lists
        all_segments_crossed_expand.drop(columns=[
            'prob_crossing','event_ind_crossed','event_id_crossed','norm_dist'
        ],inplace=True)
        all_segments_crossed_expand['prob_crossing'] = prob_crossing_expand
        all_segments_crossed_expand['event_ind'] = event_ind_expand
        all_segments_crossed_expand['event_id'] = event_id_expand
        all_segments_crossed_expand['norm_dist'] = norm_dist_expand
        
        # ---
        # clean up rupture table before export
        # map crossing logic results from segment based to rupture based
        each = 'primary'
        scenarios_with_crossing_for_haz = np.unique(np.hstack([
            # np.hstack(segment_by_qfault[each][f'scenario_crossed_{each}'].values)
            np.hstack(segment_by_qfault[each][f'event_ind_crossed'].values)
            for each in ['primary','secondary']
        ])).astype(int)
        # sort lists under segment_crossed_by_rupture
        # get list of segment index crossed by rupture after reseting segment table
        # segment_crossed_by_rupture = {}
        # for each in ['primary','secondary']:
        #     segment_crossed_by_rupture[each] = {}
        #     # for scen_ind in list(segment_crossed_by_rupture[each]):
        #         # segment_crossed_by_rupture[each][scen_ind] = sorted(segment_crossed_by_rupture[each][scen_ind])
        #     for scen_ind in list(segment_id_crossed_by_rupture[each]):
        #         seg_ind_list = []
        #         for seg_id in segment_id_crossed_by_rupture[each][scen_ind]:
        #             seg_ind_list.append(np.where(all_segments_crossed.ID==seg_id)[0][0])
        #         segment_crossed_by_rupture[each][scen_ind] = seg_ind_list
        #         # segment_crossed_by_rupture[each][scen_ind] = sorted(segment_crossed_by_rupture[each][scen_ind])
        # get subset of rupture table with crossings only
        rupture_table_crossing_only = rupture_table.loc[scenarios_with_crossing_for_haz].copy()
        # append items to rupture table with crossings only
        col_headers_to_append = []
        for each in ['primary','secondary']:
            # segments_crossed_curr_haz = []
            segments_id_crossed_curr_haz = []
            prob_crossing_curr_haz = []
            norm_dist_curr_haz = []
            for ind in scenarios_with_crossing_for_haz:
                if ind in segment_id_crossed_by_rupture[each]:
                    # segments_crossed_curr_haz.append(segment_crossed_by_rupture[each][ind])
                    segments_id_crossed_curr_haz.append(segment_id_crossed_by_rupture[each][ind])
                    prob_crossing_curr_haz.append(prob_crossing_by_rupture[each][ind])
                    norm_dist_curr_haz.append(norm_dist_by_rupture[each][ind])
                else:
                    # segments_crossed_curr_haz.append([])
                    segments_id_crossed_curr_haz.append([])
                    prob_crossing_curr_haz.append([])
                    norm_dist_curr_haz.append([])
            # rupture_table_crossing_only[f'seg_ind_crossed_{each}'] = segments_crossed_curr_haz
            rupture_table_crossing_only[f'seg_id_crossed_{each}'] = segments_id_crossed_curr_haz
            rupture_table_crossing_only[f'prob_crossing_for_seg_{each}'] = prob_crossing_curr_haz
            rupture_table_crossing_only[f'norm_dist_for_seg_{each}'] = norm_dist_curr_haz
            # get columns created from this function
            col_headers_to_append.append([
                # f'seg_ind_crossed_{each}',
                f'seg_id_crossed_{each}',
                f'prob_crossing_for_seg_{each}',
                f'norm_dist_for_seg_{each}',
            ])
        col_headers_to_append = np.asarray(col_headers_to_append).flatten()
        # get concatenated list between primary and secondary
        segments_crossed_merge = []
        segments_id_crossed_merge = []
        prob_crossing_merge = []
        norm_dist_merge = []
        for i in range(rupture_table_crossing_only.shape[0]):
            # segments_crossed_merge.append(list(np.hstack([
            #     rupture_table_crossing_only['seg_ind_crossed_primary'].iloc[i],
            #     rupture_table_crossing_only['seg_ind_crossed_secondary'].iloc[i],
            # ]).astype(int)))
            segments_id_crossed_merge.append(list(np.hstack([
                rupture_table_crossing_only['seg_id_crossed_primary'].iloc[i],
                rupture_table_crossing_only['seg_id_crossed_secondary'].iloc[i],
            ]).astype(int)))
            prob_crossing_merge.append(list(np.hstack([
                rupture_table_crossing_only['prob_crossing_for_seg_primary'].iloc[i],
                rupture_table_crossing_only['prob_crossing_for_seg_secondary'].iloc[i],
            ])))
            norm_dist_merge.append(list(np.hstack([
                rupture_table_crossing_only['norm_dist_for_seg_primary'].iloc[i],
                rupture_table_crossing_only['norm_dist_for_seg_secondary'].iloc[i],
            ])))
        # rupture_table_crossing_only['seg_ind_crossed'] = segments_crossed_merge
        rupture_table_crossing_only['seg_id_crossed'] = segments_id_crossed_merge
        rupture_table_crossing_only['prob_crossing'] = prob_crossing_merge
        rupture_table_crossing_only['norm_dist'] = norm_dist_merge
        col_headers_to_append = np.hstack([
            col_headers_to_append,
            # ['seg_ind_crossed', 'seg_id_crossed', 'prob_crossing', 'norm_dist']
            ['seg_id_crossed', 'prob_crossing', 'norm_dist']
        ])
        rupture_table_crossing_only.reset_index(drop=True,inplace=True)
        event_ids_to_keep = np.unique(rupture_table_crossing_only.EventID.values)
        
        # ---
        # export
        # segment table
        # all_segments_crossed.to_csv(
        all_segments_crossed_expand.to_csv(
            os.path.join(processed_input_dir,'site_data_PROCESSED_CROSSING_ONLY.csv'),
            index=False
        )
        # rupture table
        rupture_table_crossing_only.drop('geometry',axis=1).to_csv(
            os.path.join(im_dir,'RUPTURE_METADATA_from_crossing_logic.csv'),
            index=False
        )
        # shapefile of segment table in case of empty geodataframe
        # recreate gdf to export to shapefile
        # all_segments_crossed_gdf = GeoDataFrame(
        all_segments_crossed_expand_gdf = GeoDataFrame(
            all_segments_crossed_expand,
            crs=seg_crossed_geoms_crs,
            # geometry=seg_crossed_geoms
            geometry=geoms_expand
        )
        all_segments_crossed_expand_gdf = all_segments_crossed_expand_gdf.to_crs(epsg_wgs84)
        all_segments_crossed_expand_gdf.to_file(
            os.path.join(processed_input_dir,'site_data_PROCESSED_CROSSING_ONLY.csv').replace(
                '.csv','.gpkg'
            ), layer='data', index=False
        )
        # all_segments_crossed_gdf = all_segments_crossed_gdf.to_crs(epsg_wgs84)
        
        # store qfault crossed
        for each in ['primary','secondary']:
            gdf_qfault_crossed_export[each].to_crs(epsg_wgs84).to_file(
                os.path.join(im_dir,'qfaults_crossed.csv').replace(
                    '.csv','.gpkg'
                ), layer=each, index=False
            )
        
        # ---
        # return
        return all_segments_crossed_expand_gdf, rupture_table_crossing_only, col_headers_to_append, event_ids_to_keep
        

# ---
def get_pipe_crossing_landslide_or_liq(
    opensra_dir,
    path_to_def_shp,
    infra_site_data,
    avail_data_summary,
    infra_site_data_geom=None,
    export_dir=None,
    def_type='landslide',
    dem_raster_fpath=None,
    flag_using_state_network=False,
    flag_using_cpt_based_methods=False,
    def_shp_crs=None,
    freeface_fpath=None,
):
    """
    function to determine crossing
    note: 1) a [pipe]line is composed  a series of segments
          2) def_type = 'lateral_spread', 'settlement', 'landslide', 'surface_fault_rupture'
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
        if def_shp_crs is None:
            def_shp_crs = 4326 # default to wgs84
        def_poly_gdf = read_file(path_to_def_shp,crs=def_shp_crs)
        if def_poly_gdf.crs != 4326:
            def_poly_gdf.to_crs(4326, inplace=True)

    # ---
    # convert site data DataFrame to GeoDataFrame
    if infra_site_data_geom is None:
        segment_gdf = GeoDataFrame(
            infra_site_data,
            crs=epsg_wgs84,
            geometry=make_list_of_linestrings(
                pt1_x=infra_site_data['LON_BEGIN'].values,
                pt1_y=infra_site_data['LAT_BEGIN'].values,
                pt2_x=infra_site_data['LON_END'].values,
                pt2_y=infra_site_data['LAT_END'].values,
            ),
        )
    else:
        segment_gdf = GeoDataFrame(
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
        if len(crossed_poly_index) == 0:
            logging.info(f'\n*****FATAL*****')
            logging.info(f'- No crossings identified using specified deformation polygons!')
            logging.info(f'- Preprocessing will now exit as the final risk metrics will all be zero.')
            logging.info(f'- Please revise the input infrastructure file and/or the landslide deformation shapefile and try preprocessing again.')
            logging.info(f'*****FATAL*****\n')
            return None
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
    # if using state network, hard points along pipelines have been precomputed
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
    # get DEM for unique deformation polygon boundaries to determine direction of slip
    if poly_exist:
        def_poly_crossed_unique_gdf_utm = def_poly_crossed_unique_gdf.to_crs(epsg_utm_zone10).copy()
        # for lateral spread and settlement, slip direction has been predetermined during CPT analysis
        if def_type == 'lateral_spread' or def_type == 'settlement':
            pass
        elif def_type == 'landslide':
            # get boundary coordinates for deformation polygons with crossings for DEM sampling
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
                crs=epsg_wgs84,
            )
            dem_raster_crs = epsg_wgs84
            if dem_raster_fpath is None:
                # sample DEM at grid nodes
                param = 'dem'
                dem_file_metadata = avail_data_summary['Parameters'][param]
                dem_raster_fpath = os.path.join(opensra_dir,dem_file_metadata['Datasets']['Set1']['Path'])
                dem_raster_crs = dem_file_metadata['Datasets']['Set1']['CRS']
            if not os.path.exists(dem_raster_fpath):
                raise ValueError("Path to DEM raster does not exist. Check pipe crossing function")
            bound_coord_flat_df.data = bound_coord_flat_df.sample_raster(
                input_table=bound_coord_flat_df.data,
                fpath=dem_raster_fpath,
                dtype='float',
                crs=dem_raster_crs
            )
            dem_basename = get_basename_without_extension(dem_raster_fpath)
            bound_coord_flat_df.data['x'] = bound_coord_utm_flat[:,0] # add utm x
            bound_coord_flat_df.data['y'] = bound_coord_utm_flat[:,1] # add utm y

    # ---
    # get slip direction (azimuth, relative to North) for deformation polygons
    if poly_exist:
        # for lateral spread and settlement, slip direction has been predetermined during CPT analysis
        if def_type == 'lateral_spread' or def_type == 'settlement':
            pass
        elif def_type == 'landslide':
            utm_coord_for_max_dem = []
            utm_coord_for_min_dem = []
            # for i,each in enumerate(def_poly_crossed_unique_gdf_utm.geometry.boundary):
            for i in range(def_poly_crossed_unique_gdf_utm.shape[0]):
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
        # slip_vect_dx = []
        # slip_vect_dy = []
        # for i,each in enumerate(def_poly_crossed_unique_gdf_utm.geometry.boundary):
        # for i in range(def_poly_crossed_unique_gdf_utm.shape[0]):
        #     dx, dy = get_dx_dy_for_vector_along_azimuth(def_poly_crossed_unique_gdf_utm.slip_dir[i])
        #     slip_vect_dx.append(dx)
        #     slip_vect_dy.append(dy)
        slip_vect_dx, slip_vect_dy = \
            get_dx_dy_for_vector_along_azimuth(def_poly_crossed_unique_gdf_utm.slip_dir.values)
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
        for i in range(def_poly_crossed_unique_gdf_utm.shape[0]):
            # for liquefaction, only need upper and lower halves
            if def_type == 'lateral_spread' or def_type == 'settlement':
                upper_i, lower_i = split_geom_into_halves_without_high_low_points(
                    geom=def_poly_crossed_unique_gdf_utm.geometry[i],
                    slip_dir=def_poly_crossed_unique_gdf_utm.slip_dir[i],
                )
                upper_list.append(upper_i)
                lower_list.append(lower_i)
            # for landslide, also need scarp, body, toe in addition to upper and lower halves
            elif def_type == 'landslide':
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
        # if no crossing, then make empty dataframe to proceed
        if len(def_poly_crossed_unique_gdf_utm) == 0:
            def_poly_crossed_gdf_utm = GeoDataFrame(
                columns=def_poly_crossed_unique_gdf_utm.columns,
                geometry=[]
            )
        else:
            def_poly_crossed_gdf_utm = \
                def_poly_crossed_unique_gdf_utm.loc[crossed_def_poly_map[:,2]].reset_index(drop=True)
        
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
        crossing_summary_gdf_utm = GeoDataFrame(
            pd.DataFrame(segment_crossed_gdf_utm.drop(columns='geometry')),
            crs=epsg_utm_zone10,
            geometry=crossings_list
        )
        # append to dataframe
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
        # create geodataframe of crossings
        crossing_summary_gdf_utm = GeoDataFrame(
            pd.DataFrame(segment_gdf.drop(columns='geometry')),
            crs=epsg_utm_zone10,
            geometry=points_from_xy(
                x=segment_full_mid_utm[:,0],
                y=segment_full_mid_utm[:,1],
                crs=epsg_utm_zone10
            )
        )
        
    # get l/h for at crossed segments if analyzing lateral spread and freeface freature exists
    if poly_exist and def_type == 'lateral_spread' and freeface_fpath is not None:
        gdf_freeface_wgs84 = read_file(freeface_fpath, crs=epsg_wgs84)
        gdf_freeface_utm = gdf_freeface_wgs84.to_crs(epsg_utm_zone10)
        # for each node, get shortest distance to each freeface feature
        crossing_to_freeface_dist = np.asarray([
            crossing_summary_gdf_utm.distance(gdf_freeface_utm.geometry[i])
            for i in range(gdf_freeface_utm.shape[0])
        ])
        # for each CPT, find minimum distance on all features
        crossing_summary_gdf_utm['freeface_dist_m'] = np.min(crossing_to_freeface_dist,axis=0)
        # for each CPT get closest freeface feature and get height from attribute
        _, nearest_freeface_feature = \
            GeoSeries(gdf_freeface_utm.geometry).sindex.nearest(crossing_summary_gdf_utm.to_crs(epsg_utm_zone10).geometry)
        crossing_summary_gdf_utm['freeface_height_m'] = gdf_freeface_utm['Height_m'].loc[nearest_freeface_feature].values
        # get freeface L/H for each CPT
        crossing_summary_gdf_utm['lh_ratio'] = \
            crossing_summary_gdf_utm['freeface_dist_m']/crossing_summary_gdf_utm['freeface_height_m']
        crossing_summary_gdf_utm['lh_ratio'] = np.maximum(crossing_summary_gdf_utm['lh_ratio'],4) # set lower limit to 4
                
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
            # deformation polygon index
            def_poly_index_i = crossing_summary_gdf_utm.def_poly_index_crossed[i]
            # crossing algo index for getting scarp, body, and toe geometries
            crossing_algo_index = crossing_summary_gdf_utm.crossing_algo_index[i]
            # first find crossing between segment and deformation geometry.boundary
            crossing_i = crossing_summary_gdf_utm.geometry[i]
            if def_type == 'landslide':
                # see which section the crossing lies on
                scarp_crossed_i = scarp_list[crossed_def_poly_map[crossing_algo_index,2]]
                body_crossed_i = body_list[crossed_def_poly_map[crossing_algo_index,2]]
                toe_crossed_i = toe_list[crossed_def_poly_map[crossing_algo_index,2]]
                if scarp_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                    section_crossed_list.append('scarp')
                    section_geom_list.append(scarp_crossed_i)
                elif toe_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                    section_crossed_list.append('toe')
                    section_geom_list.append(toe_crossed_i)
                else:
                    # assign to body
                    section_crossed_list.append('body')
                    section_geom_list.append(body_crossed_i)
            # see which half the crossing lies on
            upper_crossed_i = upper_list[crossed_def_poly_map[crossing_algo_index,2]]
            lower_crossed_i = lower_list[crossed_def_poly_map[crossing_algo_index,2]]
            if upper_crossed_i.boundary.buffer(buffer).contains(crossing_i):
                half_crossed_list.append('upper')
                half_geom_list.append(upper_crossed_i)
            else:
                # assign to lower
                half_crossed_list.append('lower')
                half_geom_list.append(lower_crossed_i)
        # append to dataframe
        if def_type == 'landslide':
            crossing_summary_gdf_utm['section_crossed'] = section_crossed_list
        crossing_summary_gdf_utm['which_half'] = half_crossed_list
    else:
        if def_type == 'landslide':
            crossing_summary_gdf_utm['section_crossed'] = None
        crossing_summary_gdf_utm['which_half'] = None
        
    # ---
    if poly_exist:
        # if crossing exists, continue, else set beta_crossing to null array
        if len(crossed_poly_index) > 0:
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
                if def_type == 'landslide':
                    geom_list_to_use = section_geom_list
                elif def_type == 'lateral_spread' or def_type == 'settlement':
                    geom_list_to_use = half_geom_list
                else:
                    raise ValueError('check def type')
                if geom_list_to_use[i].contains(pt):
                    direct.append(1)
                else:
                    # sanity check with startpoint
                    node2 = crossing_summary_segment_begin_utm[i]
                    vect = node2 - crossing_coords_i
                    vect_norm = vect/np.sqrt(np.dot(vect,vect))
                    pt = Point(crossing_coords_i + vect_norm*d_length)
                    if geom_list_to_use[i].contains(pt):
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
        else:
            beta_crossing = np.asarray([])
        
    # ---
    # get anchorage lengths
    # initialize
    anchorage_length = []
    length_tol = 1e-1 # m
    if poly_exist:
        # if crossing exists, continue, else set anchorage_length to null array
        if len(crossed_poly_index) > 0:
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
                lines_for_start_side = MultiLineString(
                    make_list_of_linestrings(
                        pt1_x=start_nodes_from_crossing_to_hard_pt_on_start_side[:,0],
                        pt1_y=start_nodes_from_crossing_to_hard_pt_on_start_side[:,1],
                        pt2_x=end_nodes_from_crossing_to_hard_pt_on_start_side[:,0],
                        pt2_y=end_nodes_from_crossing_to_hard_pt_on_start_side[:,1],
                    )
                )
                lines_for_end_side = MultiLineString(
                    make_list_of_linestrings(
                        pt1_x=start_nodes_from_crossing_to_hard_pt_on_end_side[:,0],
                        pt1_y=start_nodes_from_crossing_to_hard_pt_on_end_side[:,1],
                        pt2_x=end_nodes_from_crossing_to_hard_pt_on_end_side[:,0],
                        pt2_y=end_nodes_from_crossing_to_hard_pt_on_end_side[:,1],
                    )
                )            
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
            anchorage_length = np.asarray([])
    else:
        # see if infra data contains the column 'L_ANCHOR_FULL_METER', if so, use this values
        if 'L_ANCHOR_FULL_METER' in crossing_summary_gdf_utm:
            anchorage_length = crossing_summary_gdf_utm.L_ANCHOR_FULL_METER.values
        else:
            # get inputs
            sub_segment_index_list = crossing_summary_gdf_utm.SUB_SEGMENT_ID.values - 1
            segment_length_list = crossing_summary_gdf_utm.LENGTH_KM.values * 1000 # convert to meters
            # run function to calculate anchorage length from hard points
            anchorage_length = get_anchorage_length_full_system_landslide_or_liq(
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
        else:
            crossing_summary_gdf['beta_crossing'] = 135 # for SSComp
            crossing_summary_gdf['psi_dip'] = 15 # not used
        crossing_summary_gdf['theta_rake'] = 45 # not used
    # lateral spread or landslide
    elif def_type == 'lateral_spread' or def_type == 'settlement':
        crossing_summary_gdf['theta_rake'] = 45 # not used
        if def_type == 'lateral_spread':
            # for lateral spread, if deformation polygons exist, psi and theta are dependent on sampled beta
            crossing_summary_gdf['psi_dip'] = 45 # just a placeholder, to be determined in OpenSRA.py
        elif def_type == 'settlement':
            # for settlement, psi and theta can be assigned now, since only "normal-slip" is assumed for slip mechanism
            crossing_summary_gdf['psi_dip'] = 75 # always, for settlement
        if poly_exist:
            crossing_summary_gdf['beta_crossing'] = beta_crossing
            # get additional columns from deformation polygon file if used
            columns_to_get = [
                'event_id',
                'slip_dir',
                'pgdef_m',
                'sigma',
                'sigma_mu',
                'dist_type',
                'amu',
                'bmu',
                # 'lh_ratio'
            ]
            if def_type == 'lateral_spread':
                columns_to_get.append('ls_cond')
                # columns_to_get.append('lh_ratio')
            # set empty columns
            for col in columns_to_get:
                crossing_summary_gdf[col] = None
            # grab values based on crossing algo index
            for ind in np.unique(crossed_poly_index):
                rows_for_ind = np.where(crossing_summary_gdf.def_poly_index_crossed==ind)[0]
                for col in columns_to_get:
                    crossing_summary_gdf.loc[rows_for_ind,col] = def_poly_gdf[col].loc[ind]
        else:
            crossing_summary_gdf['beta_crossing'] = 90 # not used
    elif def_type == 'surface_fault_rupture':
        raise ValueError(f'For surface fault rupture, use the "get_pipe_crossing_fault_rup" function')
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
    # in case of empty geodataframe
    if crossing_summary_gdf.shape[0] == 0:
        # GeoSeries(crossing_summary_gdf.geometry).to_file(
        crossing_summary_gdf.to_file(
            os.path.join(export_dir,f'site_data_PROCESSED_CROSSING_ONLY.gpkg'),
            schema={"geometry": "LineString", "properties": {}},
            index=False,
            layer='data'
        )
        # pass
    else:
        # GeoSeries(crossing_summary_gdf.geometry).to_file(
        crossing_summary_gdf.to_crs(epsg_wgs84).to_file(
            os.path.join(export_dir,f'site_data_PROCESSED_CROSSING_ONLY.gpkg'),
            index=False,
            layer='data'
        )
        
    # export deformation polygons crossed
    if poly_exist:
        def_poly_crossed_unique_gdf_utm.to_crs(epsg_wgs84).to_file(
            os.path.join(export_dir,'deformation_polygons_crossed.gpkg'),
            index=False,
            layer='data'
        )
    
    # ---
    # return
    return crossing_summary_gdf