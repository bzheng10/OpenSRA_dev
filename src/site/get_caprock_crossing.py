"""
"""

# Python base modules
import os
import json

# scientific processing
import pandas as pd
import numpy as np

# geoprocessing modules
# import geopandas as gpd
from geopandas import read_file, GeoDataFrame
from shapely.geometry import Polygon, LineString
from pyproj import Transformer

# precompile
from numba import njit, float64, boolean, int32, int64
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
    dip, # deg
    dip_dir, # deg
    z_top, # m
    z_bot, # m
):
    """get (x,y) vertices for a plane given trace and fault angles"""
    # fault geom calcs
    strike = dip_dir - 90 # deg
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

# workflow
def get_caprock_crossing(
    caprock_shp_file,
    im_dir,
    processed_input_dir=None,
    return_dict=False
):
    """function to determine crossing"""
    # default crs
    epsg_wgs84 = 4326
    epsg_utm_zone10 = 32610

    # load caprocks
    caprock_list = read_file(caprock_shp_file,crs=epsg_wgs84)
    n_caprock = caprock_list.shape[0]

    # load fault file
    # fault_list = pd.read_csv(rup_fpath)
    fault_list = pd.read_csv(os.path.join(im_dir,'RUPTURE_METADATA.csv'))
    n_fault = fault_list.shape[0]

    # create transformers for transforming coordinates
    transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
    transformer_utmzone10_to_wgs84 = Transformer.from_crs(epsg_utm_zone10, epsg_wgs84)

    # track caprock crossing
    caprock_crossing_flag = []
    caprock_crossing = []

    # for each caprock
    for h in range(n_caprock):
        z_depth_curr_caprock = caprock_list.depth_m[h]
        curr_caprock_polygon_wgs84 = caprock_list.geometry[h]
        curr_caprock_polygon_coord_utmzone10 = \
            transformer_wgs84_to_utmzone10.transform(*np.flipud(np.transpose(curr_caprock_polygon_wgs84.boundary.coords)))
        curr_caprock_polygon_utmzone10 = Polygon(np.transpose(curr_caprock_polygon_coord_utmzone10))
        
        # # just need to see if caprock crosses with ANY of the fault, if so, then move on to next caprock
        # crossing_with_curr_fault = False
        
        # track crossing for current caprock
        crossing_for_curr_caprock = []
        
        # for each fault scenario
        for i in range(n_fault):
            
            # just need to see if caprock crosses with ANY of the fault planes, if so, then move onto next fault
            crossing_with_curr_fault = False
            
            # get attributes
            # z_top = fault_list.loc[i,'UpperDepth'].copy() * 1000 # m
            # z_bot = fault_list.loc[i,'LowerDepth'].copy() * 1000 # m
            z_top = fault_list.loc[i,'z_tor'].copy() * 1000 # m
            z_bot = fault_list.loc[i,'z_bor'].copy() * 1000 # m
            # z_bot = z_depth_curr_caprock # m
            
            # if depth of caprock is outside the depths to top and bottom of rupture, then crossing does not exist for current rupture
            if z_depth_curr_caprock < z_top or z_depth_curr_caprock > z_bot:
                pass
            
            else:
                # get rest of fault attributes
                # dip_dir = fault_list.loc[i,'DipDir'] # deg
                dip_dir = fault_list.loc[i,'dip_dir'] # deg
                # strike = dip_dir - 90 # deg
                # dip = fault_list.loc[i,'Dip'].copy() # deg
                dip = fault_list.loc[i,'dip'].copy() # deg
                # fault_trace_curr = np.asarray(json.loads(fault_list.loc[i,'FaultTrace'])) # lon, lat (m)
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
                    # plane_pt1, plane_pt2, plane_pt3, plane_pt4 = \
                    _, _, plane_pt3, plane_pt4 = \
                        get_fault_vertices(
                            fault_trace=fault_trace_curr_meter.copy(), # (x,y,z) m
                            # strike=strike, # deg
                            dip=dip, # deg
                            dip_dir=dip_dir, # deg
                            z_top=z_top, # m
                            z_bot=z_depth_curr_caprock # m
                        )
                    
                    curr_trace_at_caprock_depth = LineString(
                        np.vstack([plane_pt3,plane_pt4])
                    )
                    # if curr_caprock_polygon_utmzone10.intersects(curr_fault_trace_geom):
                    if curr_caprock_polygon_utmzone10.intersects(curr_trace_at_caprock_depth):
                        crossing_with_curr_fault = True
                        crossing_for_curr_caprock.append(i)
                    break
                
                # break if crosses ANY fault
                # if crossing_with_curr_fault:
                #     break
        
        # append to list
        caprock_crossing_flag.append(crossing_with_curr_fault)
        caprock_crossing.append(crossing_for_curr_caprock)

    # append to caprock_list
    caprock_list['crossing_exist'] = caprock_crossing_flag
    caprock_list['faults_crossed'] = caprock_crossing

    # if processed_input_dir is not None, export
    if processed_input_dir is not None:
        # to csv
        spath_csv = os.path.join(processed_input_dir,'caprock_crossing.csv')
        caprock_list.to_csv(spath_csv,index=False)
        # to gpkg
        spath_gpkg = os.path.join(processed_input_dir,'caprock_crossing.gpkg')
        geoms = caprock_list.geometry.values # keep copy
        caprock_list_gdf = GeoDataFrame(
            pd.read_csv(spath_csv),
            crs=4326,
            geometry=geoms
        )
        caprock_list_gdf.to_file(spath_gpkg,index=False,layer='data')

    # return
    if return_dict:
        return caprock_list