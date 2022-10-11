"""
"""

# scientific processing
import os
import json
import pandas as pd
import numpy as np

# geoprocessing
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from pyproj import Transformer

# precompile
from numba import njit, float64, boolean, int32, int64, typeof
from numba.types import Tuple, List


@njit(
    Tuple((float64[:],float64[:],float64[:],float64[:]))(
        float64[:,:],float64,float64,float64,float64
    ),
    fastmath=True,
    cache=True
)
def get_fault_vertices(
    fault_trace, # (x,y) m
    # strike, # deg
    dip, # deg
    dip_dir, # deg
    z_top, # m
    z_bot # m
):
    """get (x,y) vertices for a plane given trace and fault angles"""
    # fault geom calcs
    strike = dip_dir - 90
    dz = z_bot - z_top # m
    dx = dz/np.tan(np.radians(dip)) # m
    sin_dip_dir = np.sin(np.radians(dip_dir))
    cos_dip_dir = np.cos(np.radians(dip_dir))
    if strike >= 0:
        sign = -1
    else:
        sign = 1

    # get top of rupture vertices
    plane_pt1 = fault_trace[0]
    plane_pt2 = fault_trace[1]

    # get bottom of rupture vertices
    plane_pt3 = plane_pt2.copy()
    plane_pt4 = plane_pt1.copy()
    plane_pt3[0] = plane_pt3[0] + sign*dx*sin_dip_dir
    plane_pt3[1] = plane_pt3[1] + sign*dx*cos_dip_dir
    plane_pt4[0] = plane_pt4[0] + sign*dx*sin_dip_dir
    plane_pt4[1] = plane_pt4[1] + sign*dx*cos_dip_dir

    # return
    return plane_pt1, plane_pt2, plane_pt3, plane_pt4


@njit(
    float64(float64[:]),
    fastmath=True,
    cache=True
)
def get_vect_mag(vect):
    """get vector magnitude"""
    return np.sqrt(vect.dot(vect))


@njit(
    Tuple((boolean, int32, float64[:], float64))(
        float64[:,:],
        float64[:],float64[:],float64[:],float64[:],
    ),
    fastmath=True,
    cache=True
)
def get_well_fault_intersection_and_angle(
    # traces for well
    well_trace,
    # points 1-4 should be oriented in the same direction: CW or CCW
    plane_pt1,
    plane_pt2,
    plane_pt3,
    plane_pt4
):
    """for a given fault plane and well trace, find intersection and crossing angle"""
    # get unit normal vector to plane
    plane_normal = np.cross(plane_pt4 - plane_pt1, plane_pt2 - plane_pt1)
    plane_unit_normal = plane_normal / get_vect_mag(plane_normal)
    
    # loop through well trace to find segment with intersection
    # initialize default outputs
    intersection_on_well = False
    intersection_default = np.asarray([-999.,-999.,-999.])
    segment_ind_at_intersection_default = -999
    fault_angle = -999
    
    # loop through well trace and check each segment
    for i in range(len(well_trace)-1):
        # make well segment
        segment_pt1 = well_trace[i]
        segment_pt2 = well_trace[i+1]
        segment_vector = segment_pt2 - segment_pt1
        
        # get intersection between inf line and inf plane
        d = (plane_pt1 - segment_pt1).dot(plane_unit_normal / plane_unit_normal.dot(segment_vector))
        intersection = segment_pt1 + (d*segment_vector)
        
        # check if intersection is on segment
        intersection_on_segment = False
        if (intersection-segment_pt1).dot(intersection-segment_pt1) <= (segment_vector).dot(segment_vector):
            intersection_on_segment = True
        
        # check if intersection is on finite plane
        # set up determinant with three of the four vertices and the point of intersection
        pt_a = plane_pt1
        pt_c = plane_pt3
        for j in range(2):
            if j == 0:
                # 1) plane pts 1, 2, 3
                pt_b = plane_pt2
            elif j == 1:
                # 2) plane_pts 1, 4, 3
                pt_b = plane_pt4
            
            # vectors between vertices
            vect_ab = pt_b - pt_a
            vect_bc = pt_c - pt_b
            vect_ca = pt_a - pt_c
            
            # vectors from vertices to intersection
            vect_ap = intersection - pt_a
            vect_bp = intersection - pt_b
            vect_cp = intersection - pt_c

            # get cross-product
            cross_vect_ab_ap = np.cross(vect_ab,vect_ap)
            cross_vect_bc_bp = np.cross(vect_bc,vect_bp)
            cross_vect_ca_cp = np.cross(vect_ca,vect_cp)

            # check sign for last component; if all == same sign, then lies in plane
            intersection_within_plane = False
            if np.abs(np.sign(cross_vect_ab_ap[2]) - np.sign(cross_vect_bc_bp[2])) <= 1 and \
            np.abs(np.sign(cross_vect_bc_bp[2]) - np.sign(cross_vect_ca_cp[2])) <= 1 and \
            np.abs(np.sign(cross_vect_ca_cp[2]) - np.sign(cross_vect_ab_ap[2])) <= 1:
                intersection_within_plane = True
            # break loop if True
            if intersection_within_plane:
                break

        # check if intersection is on segment and within plane (valid)
        if intersection_on_segment and intersection_within_plane:
            intersection_on_well = True
            well_segment_vector_at_intersection = segment_vector
            segment_ind_at_intersection = i
            segment_pt1_at_intersection = segment_pt1
            segment_pt2_at_intersection = segment_pt2
            break

    # use segment vector and plane normal to get internal angle, then fault angle = complimentary of internal angle
    if intersection_on_well:
        internal_angle = np.degrees(np.arccos(
            segment_vector.dot(plane_unit_normal) / \
            get_vect_mag(segment_vector) / \
            get_vect_mag(plane_unit_normal)
        ))
        fault_angle = abs(90 - internal_angle)
    else:
        intersection = intersection_default
        segment_ind_at_intersection = segment_ind_at_intersection_default

    # return
    return intersection_on_well, segment_ind_at_intersection, intersection, fault_angle


@njit(
    Tuple((
            List(int64),
            List(float64[:]),
            List(int32),
            List(float64),
            List(float64)
    ))(
        float64[:,:],
        float64[:,:],
        float64[:,:],
        float64[:,:],
        float64[:,:],
        int32[:]
    ),
    fastmath=True,
    cache=True
)
def get_fault_crossing_for_well(
    well_trace,
    plane_pt1_arr,
    plane_pt2_arr,
    plane_pt3_arr,
    plane_pt4_arr,
    n_planes_list
):
    """get crossings between a well and a list of faults"""
    
    # track list of wells crossed for current fault
    faults_crossed_list = []
    intersection_list = []
    segment_ind_at_intersection_list = []
    fault_angle_list = []
    intersection_depth_list = []
    # for looping
    n_fault = len(n_planes_list)
    for i in range(n_fault):
        # get indices from list of plane vertices
        n_planes_curr_fault = n_planes_list[i]
        count_up_to_ith_fault = sum(n_planes_list[:i])
        count_after_ith_fault = count_up_to_ith_fault + n_planes_curr_fault
        # get preprocessed planes for current fault
        plane_pt1_curr_fault = plane_pt1_arr[count_up_to_ith_fault:count_after_ith_fault].copy()
        plane_pt2_curr_fault = plane_pt2_arr[count_up_to_ith_fault:count_after_ith_fault].copy()
        plane_pt3_curr_fault = plane_pt3_arr[count_up_to_ith_fault:count_after_ith_fault].copy()
        plane_pt4_curr_fault = plane_pt4_arr[count_up_to_ith_fault:count_after_ith_fault].copy()
        # loop through each fault plane
        for j in range(n_planes_curr_fault):
            # perform search for intersection
            intersection_on_well, segment_ind_at_intersection, intersection, fault_angle = \
                get_well_fault_intersection_and_angle(
                    # traces for well
                    well_trace,
                    # points 1-4 should be oriented in the same direction: CW or CCW
                    plane_pt1_curr_fault[j].copy(),
                    plane_pt2_curr_fault[j].copy(),
                    plane_pt3_curr_fault[j].copy(),
                    plane_pt4_curr_fault[j].copy()
                )
            # convert intersection from utm zone 10 to wgs 84 if crossing exists
            if intersection_on_well:                
                # append to list of well crossing metrics
                faults_crossed_list.append(i)
                intersection_list.append(intersection)
                segment_ind_at_intersection_list.append(segment_ind_at_intersection)
                fault_angle_list.append(fault_angle)
                intersection_depth_list.append(intersection[2])
                # break out of for loop through fault planes if there is intersection         
                break
    # return
    return faults_crossed_list, intersection_list, segment_ind_at_intersection_list, fault_angle_list, intersection_depth_list


def process_well_trace(well_trace_dir, wells_to_read):
    """process well traces given folder"""
    # default crs
    epsg_wgs84 = 4326
    epsg_utm_zone10 = 32610
    transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
    # get list of text files
    list_of_trace_files_in_dir = os.listdir(well_trace_dir)
    well_trace_list = []
    well_locs_geom = []
    # load traces for wells
    # for f in list_of_trace_files:
    for f in wells_to_read:
        if not f.endswith('.txt'):
            f += '.txt'
        # if not '_StatePlaneZone5' in f:
        if f in list_of_trace_files_in_dir:
            # load file
            curr_trace = np.loadtxt(
                os.path.join(well_trace_dir,f),
                skiprows=1 # first row is header
            )
            well_locs_geom.append(Point(curr_trace[0,0],curr_trace[0,1]))
            # transform from lat lon to UTM zone 10 meters
            curr_trace[:,0],curr_trace[:,1] = transformer_wgs84_to_utmzone10.transform(curr_trace[:,1],curr_trace[:,0])
            # append to dataframe
            well_trace_list.append(curr_trace)
    # number of wells
    n_well = len(well_trace_list)
    return well_trace_list, well_locs_geom, n_well


# workflow
def get_well_crossing(
    im_dir,
    infra_site_data,
    col_with_well_trace_file_names,
    well_trace_dir,
    export_dir=None,
    return_dict=False
):
    """function to determine crossing"""
    # default crs
    epsg_wgs84 = 4326
    epsg_utm_zone10 = 32610
    
    # load fault file
    # fault_list = pd.read_csv(rup_fpath)
    fault_list = pd.read_csv(os.path.join(im_dir,'RUPTURE_METADATA.csv'))
    n_fault = fault_list.shape[0]
    
    # create transformers for transforming coordinates
    transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
    transformer_utmzone10_to_wgs84 = Transformer.from_crs(epsg_utm_zone10, epsg_wgs84)
    
    # preprocess fault trace to get vertices for planes

    # arrays for tracking planes
    plane_pt1_arr = np.empty((0,3))
    plane_pt2_arr = np.empty((0,3))
    plane_pt3_arr = np.empty((0,3))
    plane_pt4_arr = np.empty((0,3))
    
    n_planes_list = []

    # for each fault scenario
    for i in range(n_fault):
        # get attributes
        # dip_dir = fault_list.loc[i,'DipDir'] # deg
        dip_dir = fault_list.loc[i,'dip_dir'] # deg
        # strike = dip_dir - 90 # deg
        # dip = fault_list.loc[i,'Dip'].copy() # deg
        dip = fault_list.loc[i,'dip'].copy() # deg
        # z_top = fault_list.loc[i,'UpperDepth'].copy() * 1000 # m
        # z_bot = fault_list.loc[i,'LowerDepth'].copy() * 1000 # m
        z_top = fault_list.loc[i,'z_tor'].copy() * 1000 # m
        z_bot = fault_list.loc[i,'z_bor'].copy() * 1000 # m
        # fault_trace_curr = np.asarray(json.loads(fault_list.loc[i,'FaultTrace'])) # lon, lat, z (m)
        fault_trace_curr = np.asarray(json.loads(fault_list.loc[i,'fault_trace'])) # lon, lat, z (m)

        # transform fault trace to x (m), y(m) under UTM zone 10
        fault_trace_curr_meter = fault_trace_curr[:,:2].copy()
        # fault_trace_curr_meter = fault_trace_curr.copy()
        fault_trace_curr_meter[:,0], fault_trace_curr_meter[:,1] = \
            transformer_wgs84_to_utmzone10.transform(fault_trace_curr[:,1],fault_trace_curr[:,0])
        
        # lists for tracking planes
        plane_pt1_curr_fault = []
        plane_pt2_curr_fault = []
        plane_pt3_curr_fault = []
        plane_pt4_curr_fault = []
        # loop through each fault and check plane by plane
        for j in range(len(fault_trace_curr_meter)-1):
            # get plane vertices
            plane_pt1, plane_pt2, plane_pt3, plane_pt4 = \
                get_fault_vertices(
                    fault_trace_curr_meter[j:j+2].copy(), # (x,y,z) m
                    # strike, # deg
                    dip, # deg
                    dip_dir, # deg
                    z_top, # m
                    z_bot # m
                )
            # append depths
            plane_pt1_curr_fault.append(np.hstack([plane_pt1, z_top]))
            plane_pt2_curr_fault.append(np.hstack([plane_pt2, z_top]))
            plane_pt3_curr_fault.append(np.hstack([plane_pt3, z_bot]))
            plane_pt4_curr_fault.append(np.hstack([plane_pt4, z_bot]))
        # stack to list of plane vertices
        n_planes_list.append(len(plane_pt1_curr_fault))
        plane_pt1_arr = np.vstack([plane_pt1_arr,plane_pt1_curr_fault])
        plane_pt2_arr = np.vstack([plane_pt2_arr,plane_pt2_curr_fault])
        plane_pt3_arr = np.vstack([plane_pt3_arr,plane_pt3_curr_fault])
        plane_pt4_arr = np.vstack([plane_pt4_arr,plane_pt4_curr_fault])
    
    # convert to numpy array
    n_planes_list = np.asarray(n_planes_list)
    
    # make fault plane geometry
    fault_trace_geom = []
    for i in range(n_fault):
        # get number of planes
        n_planes_curr_fault = n_planes_list[i]
        # get indices for rows
        count_up_to_ith_fault = sum(n_planes_list[:i])
        count_after_ith_fault = count_up_to_ith_fault + n_planes_curr_fault
        
        # get plane vertices
        plane_pt1_curr_fault = plane_pt1_arr[count_up_to_ith_fault:count_after_ith_fault]
        plane_pt2_curr_fault = plane_pt2_arr[count_up_to_ith_fault:count_after_ith_fault]
        plane_pt3_curr_fault = plane_pt3_arr[count_up_to_ith_fault:count_after_ith_fault]
        plane_pt4_curr_fault = plane_pt4_arr[count_up_to_ith_fault:count_after_ith_fault]
        # initialize
        coords_for_curr_fault = []
        # for every plane, get [pt1, pt2, pt3, pt4, pt1]
        for j in range(n_planes_curr_fault):
            coords_for_curr_fault += \
                [
                    plane_pt1_curr_fault[j,:2],
                    plane_pt2_curr_fault[j,:2],
                    plane_pt3_curr_fault[j,:2],
                    plane_pt4_curr_fault[j,:2],
                    plane_pt1_curr_fault[j,:2],
                ]
        coords_for_curr_fault = np.asarray(coords_for_curr_fault)
        # convert back to lat lon
        coords_for_curr_fault[:,1], coords_for_curr_fault[:,0] = \
            transformer_utmzone10_to_wgs84.transform(coords_for_curr_fault[:,0],coords_for_curr_fault[:,1])
        # make geometry
        fault_trace_geom.append(LineString(zip(coords_for_curr_fault[:,0],coords_for_curr_fault[:,1])))
        
    # run through each well trace
    if col_with_well_trace_file_names in infra_site_data.columns:
        wells_to_read = infra_site_data[col_with_well_trace_file_names].values
    else:
        wells_to_read = infra_site_data[col_with_well_trace_file_names.upper()].values
    well_trace_list, well_locs_geom, n_well = process_well_trace(well_trace_dir, wells_to_read)
    
    # set up for storage
    well_crossing_ordered_by_wells = pd.DataFrame(
        None,
        columns=[
            'fault_ind_crossed',
            'intersecting_well_segment_ind',
            'intersection_lon_lat',
            'fault_crossing_depth_m',
            'fault_angle_deg',
        ]
    )
    # for each well
    for i in range(n_well):        
        # get current well trace list
        well_trace_curr = well_trace_list[i].copy()
        # get crossings for curren well with all faults
        faults_crossed_list, intersection_list, segment_ind_at_intersection_list, \
            fault_angle_list, intersection_depth_list = \
            get_fault_crossing_for_well(
                well_trace_curr,
                plane_pt1_arr.copy(),
                plane_pt2_arr.copy(),
                plane_pt3_arr.copy(),
                plane_pt4_arr.copy(),
                n_planes_list
            )
        # append to dataframe
        well_crossing_ordered_by_wells.loc[well_crossing_ordered_by_wells.shape[0]] = [
            np.asarray(faults_crossed_list),
            np.asarray(segment_ind_at_intersection_list),
            np.asarray(intersection_list),
            np.asarray(intersection_depth_list),
            np.asarray(fault_angle_list),
        ]
    # convert intersection x y values to lat lon
    intersections = well_crossing_ordered_by_wells.intersection_lon_lat.values
    intersections_converted = []
    for each in intersections:
        if len(each) > 0:
            converted_lat, converted_lon = \
                transformer_utmzone10_to_wgs84.transform(each[:,0],each[:,1])
            intersections_converted.append(np.vstack([
                np.round(converted_lon,6),
                np.round(converted_lat,6),
                # np.round(each[:,2],1)
            ]).T.tolist())
        else:
            intersections_converted.append([])
    well_crossing_ordered_by_wells.intersection_lon_lat = intersections_converted
    # round fault angle and fault crossing depth to 1 decimals
    well_crossing_ordered_by_wells.fault_angle_deg = \
        well_crossing_ordered_by_wells.fault_angle_deg.apply(lambda x: np.round(x,1))
    well_crossing_ordered_by_wells.fault_crossing_depth_m = \
        well_crossing_ordered_by_wells.fault_crossing_depth_m.apply(lambda x: np.round(x,1))
    # add a column for if crossing exists
    crossing_exist = [
        True if len(val)>0 else False
        for val in well_crossing_ordered_by_wells.fault_ind_crossed
    ]
    well_crossing_ordered_by_wells['crossing_exist'] = crossing_exist
    # reorder to put crossing_exist column first
    cols_reorder = ['crossing_exist'] + list(well_crossing_ordered_by_wells.columns.drop('crossing_exist'))
    well_crossing_ordered_by_wells = well_crossing_ordered_by_wells[cols_reorder]
    # convert to geodataframe
    well_crossing_ordered_by_wells = gpd.GeoDataFrame(
        well_crossing_ordered_by_wells,
        crs=epsg_wgs84,
        geometry=well_locs_geom
    )

    # append column with index and reorder such that index is the first column
    # well_index = np.arange(n_well)
    well_crossing_ordered_by_wells['well_index'] = np.arange(n_well)
    well_crossing_ordered_by_wells = well_crossing_ordered_by_wells[
        ['well_index'] + list(well_crossing_ordered_by_wells.columns.drop('well_index'))
    ]
    
    # get list of wells with crossings
    wells_with_fault_crossing = np.where(well_crossing_ordered_by_wells.crossing_exist==True)[0]
    # faults_crossed = well_crossing_ordered_by_wells.fault_ind_crossed.values
    # wells_with_fault_crossing = [
    #     i for i,each in enumerate(faults_crossed) if len(each) > 0
    # ]

    # get unique well and fault crossings
    unique_well_fault_crossing = well_crossing_ordered_by_wells.loc[wells_with_fault_crossing].copy().reset_index(drop=True)
        
    # get table of well crossings ordered by faults.
    faults_crossed = np.asarray([each for each in unique_well_fault_crossing.fault_ind_crossed.values])   
    unique_faults_crossed = np.unique(faults_crossed.flatten())
    
    # for each unique fault crossed, find the well index that crosses it and the intersection info
    # initialize dictionary
    well_crossing_dict = {}
    faults_crossed = unique_well_fault_crossing.fault_ind_crossed.values
    for fault_id in unique_faults_crossed:
        # initialize dictionary
        well_crossing_dict[fault_id] = {}
        # search through fault_ind_crossed column to find which well it crosses
        unique_well_index_crossed_by_curr_fault = [
            i for i,each in enumerate(faults_crossed) if fault_id in each
        ]
        # get the position among all fault crossings for that well
        position_of_fault = [
            np.where(
                unique_well_fault_crossing.fault_ind_crossed[each]==fault_id
            )[0][0]
            for each in unique_well_index_crossed_by_curr_fault
        ]
        # retrieve info from other columns
        well_crossing_dict[fault_id]['well_ind_crossed'] = [
            # unique_well_fault_crossing.index[each]
            unique_well_fault_crossing.well_index[each]
            for each in unique_well_index_crossed_by_curr_fault
        ]
        for col in [
            'intersecting_well_segment_ind',
            'intersection_lon_lat',
            'fault_crossing_depth_m',
            'fault_angle_deg',
        ]:
            well_crossing_dict[fault_id][col] = [
                unique_well_fault_crossing[col][each][position_of_fault[each]]
                for each in unique_well_index_crossed_by_curr_fault
            ]
            
    # initialize
    well_crossing_ordered_by_faults = pd.DataFrame(
        None,
        columns=[
            'well_ind_crossed',
            'intersecting_well_segment_ind',
            'intersection_lon_lat',
            'fault_crossing_depth_m',
            'fault_angle_deg',
        ],
        index=np.arange(n_fault)
    )
    empty_list = [[]]*well_crossing_ordered_by_faults.shape[1]
    # source information
    for i in range(n_fault):
        if i in unique_faults_crossed:
            for col in well_crossing_ordered_by_faults.columns:
                well_crossing_ordered_by_faults.loc[i,col] = well_crossing_dict[i][col]
        else:
            well_crossing_ordered_by_faults.loc[i] = empty_list
    # add a column for if crossing exists
    crossing_exist = [
        True if len(val)>0 else False
        for val in well_crossing_ordered_by_faults.well_ind_crossed
    ]
    well_crossing_ordered_by_faults['crossing_exist'] = crossing_exist
    # reorder to put crossing_exist column first
    cols_reorder = ['crossing_exist'] + list(well_crossing_ordered_by_faults.columns.drop('crossing_exist'))
    well_crossing_ordered_by_faults = well_crossing_ordered_by_faults[cols_reorder]
    # convert to geodataframe
    well_crossing_ordered_by_faults = gpd.GeoDataFrame(
        well_crossing_ordered_by_faults,
        crs=epsg_wgs84,
        geometry=fault_trace_geom
    )
    
    # append rupture_metadata file with well crossings
    rup_meta_fpath = os.path.join(im_dir,'RUPTURE_METADATA.csv')
    rup_meta = pd.read_csv(rup_meta_fpath)
    for col in well_crossing_ordered_by_faults:
        if col != 'geometry':
            rup_meta[col] = well_crossing_ordered_by_faults[col].values
    rup_meta.to_csv(rup_meta_fpath,index=False)
    
    # return
    if return_dict:
        return well_crossing_ordered_by_faults, well_crossing_ordered_by_wells
    