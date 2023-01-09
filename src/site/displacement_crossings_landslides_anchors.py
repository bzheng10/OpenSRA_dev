# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:58:50 2022

@author: TomClifford

This version implements the anchor block length calculation
"""


import time
load_start = time.time()
import os
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import cascaded_union
import pyproj


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
import sys


wd = r"C:\Users\TomClifford\SlateGeotech\SlateDrive - Documents\_Projects\18-020.00_CEC_Seismic_Risk\04_Eval_Analysis\04 - Research\Task 4a\Shapefiles"
os.chdir(wd)
out_dir = os.path.dirname(wd)
# pipes_csv = pd.read_csv('California_Natural_Gas_Pipeline.csv')


#READ SHAPEFILE DATA
#pipeline shapefile
pipes = gpd.read_file('California_Natural_Gas_Pipeline.shp',  crs="EPSG:4326")
#pipeline split into 100 m segments
segs = pd.read_csv('segments_at_100m.csv')
#compile to geodataframe
lines = []
for i, l in segs.iterrows():
    line = LineString([(l['LON_BEGIN'], l['LAT_BEGIN']), (l['LON_END'], l['LAT_END'])])
    lines.append(line)
segs['geometry'] = lines
#convert to gdf
segs_gdf = gpd.GeoDataFrame(segs, crs="EPSG:4326")

def dip2strike(dipdirection):
    strike = dipdirection - 90
    if strike < 0:
        strike += 360
    return strike

#read fault shapefiles
pfaults = gpd.read_file('Primary_SC1_200m_FltHazZones_Poly.shp',  crs="EPSG:4326")
pfaults['Strike'] = pfaults['DipDirec_1'].apply(dip2strike)
sfaults = gpd.read_file('SC2_200m_FltHazZones_Poly.shp',  crs="EPSG:4326")
sfaults['Strike'] = sfaults['DipDirec_1'].apply(dip2strike)

#read UCERF3 rupture scenarios and sections
ruptures = os.path.join(os.path.dirname(wd), 'MEAN UCERF3 FM3.1', 'ruptures.csv')
ruptures = pd.read_csv(ruptures)
sections = os.path.join(os.path.dirname(wd), 'MEAN UCERF3 FM3.1', 'sections.csv')
sections = pd.read_csv(sections)

#clean up how ruptures/sections are listed
def cleanRups(rup):
    rup = rup.replace('\n','').replace('[','').replace(']','').split()
    rup = [int(i) for i in rup]
    return rup
def cleanSec(rup):
    rup = rup.replace('\n','').replace('[','').replace(']','').split()
    # rup = [int(i) for i in rup]
    return rup
ruptures['SectionsForRupture'] = ruptures['SectionsForRupture'].apply(cleanRups)
sections['RupturesForSection'] = sections['RupturesForSection'].apply(cleanSec)

def organizeCoords(c,return_val):
    #convert FaultTrace arrays into separate lat lon columns
    #return val is x or y
    c = c.replace(']','').replace('[','').replace('(','').replace(')','').replace('array','').replace(' ','').split(',')
    # c = sections['FaultTrace'][0].replace(']','').replace('[','').replace('(','').replace(')','').replace('array','').replace(' ','').split(',')
    x = []
    y = []
    num = 1
    for i in c:
        if num == 1:
            x.append(float(i))
            num = 2
            continue
        if num == 2:
            y.append(float(i))
            num = 3
            continue
        if num == 3:
            num = 1
            continue
    if return_val == 'x':
        return x
    else:
        return y 

sections['Lon'] = sections['FaultTrace'].apply(organizeCoords, args=('x'))
sections['Lat'] = sections['FaultTrace'].apply(organizeCoords, args=('y'))

def to_lines(xs, ys):
    #create list of gpd lines from lat/lon lists
    tupes = tuple(zip(xs, ys)) 
    lines = []
    for t in range(len(tupes)):
        if t == 0:
            continue
        l = LineString([tupes[t-1], tupes[t]])
        lines.append(l)
    return lines


def checkConsecutive(l):
    n = len(l) - 1
    return (sum(np.diff(sorted(l)) == 1) >= n) 

def splitSegments(l):
    #returns a list of lists of continuous indices
    # l = [1,2,3,5,7,6] > returns: [[1, 2, 3], [5, 6, 7]]
    l.sort()
    all_consec = []
    consec = []
    for i in range(len(l)):
        #always append first index
        if i == 0:
            consec.append(l[i])
        #if next index is consecutive, track
        elif l[i] == l[i-1] +1:
            consec.append(l[i])
        #if not consecutive, finish list and restart next segment
        else:
            all_consec.append(consec)
            consec = []
            consec.append(l[i])
        # print(i)
        # print(consec)
        # print(all_consec)
        # print()
        i+=1
    #append final segment
    all_consec.append(consec)
    
    return all_consec


def getUTM(longitude):
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
    # point1 =  Point(subseg.geometry[seg].xy[0][0], subseg.geometry[seg].xy[1][0])
    # point2 =  Point(subseg.geometry[seg].xy[0][1], subseg.geometry[seg].xy[1][1])
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

import math
## http://wikicode.wikidot.com/get-angle-of-line-between-two-points
## angle between two points
# def getAngle(pt1, pt2):
#     x_diff = pt2.x - pt1.x
#     y_diff = pt2.y - pt1.y
#     return math.degrees(math.atan2(y_diff, x_diff))




def az2ang(az):
    #convert geological azimuth (0 north) to unit cirle angle
    ang = (450-az)%360
    return ang
def angleDiff(a1, a2):
    a = a1 - a2
    a = (a + 180) % 360 - 180
    return a

def pipeVector(seg, fault, fault_id, crs='EPSG:4326'):
    #return vector that is oriented acutely with dip direction/ hanging wall
    #get azimuth of segment                
    pipe_az = azimuth(seg.geometry)
    #azimuth between pipe and dip direction
    pipe_dip_az = angleDiff(pipe_az, fault.DipDirec_1[fault_id])

    #if within 90 degrees, use the opposite-facing pipe vector
    if -90 < pipe_dip_az < 90:
        p1 = Point(seg.geometry.coords[0])
        p2 = Point(seg.geometry.coords[1])   
    else:
        p1 = Point(seg.geometry.coords[1])
        p2 = Point(seg.geometry.coords[0]) 
    
    return gpd.GeoDataFrame(geometry=[LineString([p1, p2])], crs=crs)


'''
def outwardVector(seg, fault, fault_id, crs='EPSG:4326'):
        #returns segment as a vector going from inside to outside polygon
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
            p_in = p2
            p_out = p1
        if p2dist > p1dist:
            p_in = p1
            p_out = p2

        return gpd.GeoDataFrame(geometry=[LineString([p_in, p_out])], crs=crs)
        # return LineString([p_in, p_out])
'''        
#get vector from point and azimuth
# def bearing(centroid, az, radius, side='both', crs='EPSG:4326'):
#     centroid = centroid.xy[0][0], centroid.xy[1][0]
#     # radius = fsize
#     ang = az2ang(az) #convert azimuth to math angles
#     ft2y = centroid[1] + radius * np.sin(np.radians(ang))
#     ft2x = centroid[0] + (radius * np.cos(np.radians(ang)))*np.cos(np.radians(ft2y))
#     if side == 'both':
#         ft1y = centroid[1] - radius * np.sin(np.radians(ang))
#         ft1x = centroid[0] - (radius * np.cos(np.radians(ang)))*np.cos(np.radians(ft1y))
#     #initial point needs to be centroid for rake
#     if side == 'one':
#         ft1y = centroid[1]
#         ft1x = centroid[0] 
#     bearing_line = gpd.GeoSeries(LineString([Point([ft1x, ft1y]), Point([ft2x, ft2y])]), crs=crs)
#     return bearing_line

# #new bearing using gpd.rotate for more acurate azimuth
def bearing(centroid, az, radius, side='both', crs='EPSG:4326'):
    # radius = 1
    # centroid = centroid.xy[0][0], centroid.xy[1][0]
    rad = LineString([centroid, (centroid.x, centroid.y + radius)])
    bearing_line = gpd.GeoSeries(rad, crs=crs)
    bearing = bearing_line.rotate(-az, origin=centroid)

    return bearing


def rake2comps(strike, dip, rake, rake_m=1):
    #strike, dip, rake in degrees as given in pfaults
    #returns map azimuth and displacement magnitude from a strike, dip, rake, and rake magnitude
    strike_m = (rake_m * np.cos(np.radians(rake))) #strike slip magnitude of rake
    # dip_m  = rake_m * np.sin(np.radians(rake)) #dip magnitude of rake - on fault plane
    vert_m = (np.sin(np.radians(dip))*np.sin(np.radians(rake)))
    dip_dir_m = (np.cos(np.radians(dip))*np.sin(np.radians(rake))) #mag disp in dip map direction (heave)
    #check mags equal 1:
    # print('Net distance of SS, DS, V components: '+str(np.sqrt((strike_m**2)+(vert_m**2)+(dip_dir_m**2))))
    horizontal_throw_m = np.sqrt((dip_dir_m**2) + (strike_m**2)) #map magnitude of rake
    # print('Net distance of H+V components: '+str(np.sqrt((horizontal_throw_m**2)+(vert_m**2))))

    strike_horizontal_throw_a = np.degrees(np.arctan(dip_dir_m/strike_m)) #angle between strike and rake azimuth - negative goes clockwise
    # print(strike_horizontal_throw_a)
    if rake > 90: #want the acute angle from strike
        rake_azimuth = strike + 180 - strike_horizontal_throw_a #add angle between rake and strike
    elif rake < -90: 
        #then find azimuth of rake in map view
        rake_azimuth = strike + 180 - strike_horizontal_throw_a 
    else:
        rake_azimuth = strike - strike_horizontal_throw_a
    #correct 0-360
    rake_azimuth = rake_azimuth%360
    #determine strike slip azimuth
    if -90 < rake < 90:
        strike_azimuth = strike
    else:
        strike_azimuth = (strike+180)%360
    #determine dip direction slip azimuth
    dip_direction_azimuth = (strike+90)%360
    if 0 < rake < 180:
        dip_direction_azimuth = (dip_direction_azimuth+180)%360
    #return abs of magnitude
    return {'horizontal_net_azimuth' : rake_azimuth, 
            'strike_slip_azimuth' : strike_azimuth,
            'dip_direction_slip_azimuth' : dip_direction_azimuth,
            
            'horizontal_net_magnitude' : np.abs(horizontal_throw_m),
            'strike_magnitude' : np.abs(strike_m),
            'dip_direction_magnitude': np.abs(dip_dir_m),
            'vertical_magnitude' : np.abs(vert_m)}


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
        # return LineString([p_in, p_out])
     


# read Cordelia Junction data (saved locally because NDA)

#post_route 100m
path = r"C:\Users\TomClifford\OneDrive - SlateGeotech\Desktop\Task4B_run_CordeliaJunction\Task4B_run_Cordelia_PostReroute\Input\site_data.csv"
segs = pd.read_csv(path)
#compile to geodataframe
lines = []
for i, l in segs.iterrows():
    line = LineString([(l['LON_BEGIN'], l['LAT_BEGIN']), (l['LON_END'], l['LAT_END'])])
    lines.append(line)
segs['geometry'] = lines
#convert to gdf
postroute = gpd.GeoDataFrame(segs, crs="EPSG:4326")


#post_route 100m
path = r"C:\Users\TomClifford\OneDrive - SlateGeotech\Desktop\Task4B_run_CordeliaJunction\Task4B_run_Cordelia_PreReroute\Input\site_data.csv"
segs = pd.read_csv(path)
#compile to geodataframe
lines = []
for i, l in segs.iterrows():
    line = LineString([(l['LON_BEGIN'], l['LAT_BEGIN']), (l['LON_END'], l['LAT_END'])])
    lines.append(line)
segs['geometry'] = lines
#convert to gdf
preroute = gpd.GeoDataFrame(segs, crs="EPSG:4326")

preroute_shp = gpd.read_file(r"C:\Users\TomClifford\SlateGeotech\SlateDrive - Documents\_Projects\18-020.00_CEC_Seismic_Risk\04_Eval_Analysis\04 - Research\Task 4a\CordeliaJunction_Pipeline\CordeliaJunctionPipeline_PostReRoute.shp")

# ca_landslides = gpd.read_file(r"C:\Users\TomClifford\SlateGeotech\SlateDrive - Documents\_Projects\18-020.00_CEC_Seismic_Risk\04_Eval_Analysis\04 - Research\Task 4a\CordeliaJunction_Pipeline\CA_LandslideInventory_CJ_SlipAz.shp")
# l3_landslides = gpd.read_file(r"C:\Users\TomClifford\SlateGeotech\SlateDrive - Documents\_Projects\18-020.00_CEC_Seismic_Risk\04_Eval_Analysis\04 - Research\Task 4a\CordeliaJunction_Pipeline\Level_3_Landslides_SlipAz.shp")

#rough estimate of landslide direction for now
landslides = gpd.read_file(r"C:\Users\TomClifford\SlateGeotech\SlateDrive - Documents\_Projects\18-020.00_CEC_Seismic_Risk\04_Eval_Analysis\04 - Research\Task 4a\CA_LandslideInventory_SlipDirection\CA_LandslideInventory_Intersect_SlipDirection.shp")


#%% FIND LANDSLIDE/PIPE INTERSECTIONS AND CALCULATE BETA/ANCHOR
start_time = time.time()
#set which pipeline route to use

intx = segs_gdf.sindex.query_bulk(landslides.geometry.boundary, predicate='intersects') #returns faults, then pipes
#listed in fault order, resort to pipes so pipes can be analyzed together
intx_df = pd.DataFrame({'landslide_index': intx[0], 'pipe_index':intx[1]})
#sort  by pipes
intx_df = intx_df.sort_values(by='pipe_index')

#landslide 44 is cordelia junction case study

intx_df = intx_df[intx_df.landslide_index == 44]


# len(intx_df.landslide_index.unique())
#536 polygons, 535 that intersect for me?
n = 1
dict_list = []
for i, r in intx_df.iterrows():
    # print(r)
    pipe_id, fault_id = r.pipe_index, r.landslide_index


    # print('Calculating probability of rupture at the intersection of pipeline {} and fault {}'.format(str(pipe_id), str(fault_id)))
    
    route = segs_gdf
    pipe = route.iloc[[pipe_id]] 
    fault = landslides.iloc[[fault_id]]
    
    
    #UTM Zone 11 3718, zone to 3740
    #choose CRS based on location    
    crs = getUTM(pipe.iloc[0].geometry.centroid.x)
    pipe = pipe.to_crs(crs) #UTM ZONE 10N
    fault = fault.to_crs(crs)  
    # route = route.to_crs(crs)   #this slows down - reduce to relevant pipeline

    
  
    ''' already using individual segment intersections so skip below intersection step'''
    # find all 100 segments that lie in the fault area
    # segments = segs[segs['OBJ_ID'] == pipe['OBJECTID'][pipe_id]]
    #reset index 
    # segments = segments.reset_index()
    # segments = segments.to_crs(crs) 
    
    #find all 100 m segments that intersection fault
    # seg_intx_indices = segments.sindex.query_bulk(fault.geometry, predicate='intersects')
    # seg_intx = segments.iloc[seg_intx_indices[1]]
    # seg_intx = route.copy()
    # # seg_intx = seg_intx.to_crs("EPSG:32633")  

    # #group continuous segment indices
    # cont_segments = splitSegments(seg_intx.index.to_list())
    
    # #find only 100 m segments that directly intersect polygon boundary
    # seg_intx_direct_i = segments.sindex.query_bulk(fault.geometry.boundary, predicate='intersects')
    # seg_intx_direct = segments.iloc[seg_intx_direct_i[1]]
    seg_intx_direct = pipe
    
    
    # #calculate La
    # La = 30 #default 30 meters
    # #check ahead in the direction into the fault zone
    # #UTM Zone 11 3718, zone to 3740
    
    # #find intersection point
    # intx_point = pipe.unary_union.intersection(fault.boundary.unary_union)
    # intx_point = gpd.GeoSeries(intx_point, crs=4326)
    # #pull pipe within 30 meters of intersection and within fault
    # anchor_buff = intx_point.to_crs(32610).buffer(La)
    # anchor_buff = gpd.GeoDataFrame(geometry=anchor_buff)
    # #clip pipe intersectin with buffer
    # pipe_buff = preroute_shp.to_crs(32610).clip(anchor_buff).clip(fault.to_crs(32610))
    
    
    
    
    # #check azimuth of each linstring in a gdf
    # #given geoseries of lines, find azimuth, if over 40 degrees, return point of their vertex
    # #shapefile is discontinuous by index
    # for i in preroute_shp.geometry.index:
    #     if i >0:
    #         azimuth(preroute_shp.geometry[i-1])
    #         azimuth(preroute_shp.geometry[i])
    


    
   #id if there are multiple segments or just one linestring
    
    #if are, find azimuth of each successive and check angle
    
    #break pipe up into 1 meter segments within next 30 meters
    
    


    # buffer = (maxx - minx)*2 

    #add columns to segments to store results
    # return_df = seg_intx.copy()
    # return_df['PoRP'] = 0
    # return_df['Buffer_ID'] = 'None'
    # return_df['TraceNo_1'] = 'None'
    # return_df['SectionId'] = 'None'

    ''' rake vector'''
    centroid = fault.geometry[fault_id].centroid

    dseg = seg_intx_direct.iloc[0]
    ''' use outwardVectorfuction to get vector of pipe pointing out of the landslide zone'''
    intx_pipe = InwardVector(dseg, fault, fault_id, crs) 
    end_seg = intx_pipe.iloc[0]
    vector_plot_length = intx_pipe.length[0]*5
    
    '''determine landslide direction here'''
    landslide_direction = fault.iloc[0]['Azimuth']

    #second pipe vector longer for plotting
    pipe_vector_plotting = bearing(end_seg.geometry.centroid, azimuth(end_seg.geometry), vector_plot_length, side='one', crs=crs)
    
    landslide_vector = bearing(end_seg.geometry.centroid, landslide_direction, vector_plot_length, side='one', crs=crs)
    #check that azimuth function works in both coords and utm
    # print('landslide azimuth in WGS 84 / UTM zone 10N')
    # print(azimuth(landslide_vector.iloc[0]))
    
    # landslide_vector_m = bearing(end_seg.geometry.centroid, landslide_direction, vector_plot_length, side='one', crs=32610)
    # landslide_vector_m = gpd.GeoSeries(landslide_vector_m, crs=4326)
    # landslide_vector_m = landslide_vector_m.to_crs(32610)
    # print('landslide azimuth in WGS 84 / UTM zone 10N')
    # print(azimuth(landslide_vector_m.iloc[0]))

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
    
    

    '''
    
    #set bounds to smaller of fault/pipe
    pb = pipe.geometry.total_bounds
    psize = np.sqrt((pb[0] - pb[2])**2 + (pb[1] - pb[3])**2)
    fb = fault.geometry.total_bounds
    fsize = np.sqrt((fb[0] - fb[2])**2 + (fb[1] - fb[3])**2)
    #determine bounds of smaller feature
    if fsize <= psize:
        smaller = fault
    else:
        smaller = pipe
    minx, miny, maxx, maxy = smaller.geometry.total_bounds
    
    minx, miny, maxx, maxy = fault.iloc[0].geometry.bounds
    fig, ax = plt.subplots(figsize=(5,5))  
    pipe.plot(ax=ax, color='blue', zorder=10, label='100 m Pipe Segment')
    # route.plot(ax=ax, color='blue', alpha=0.5, linestyle='dashed')
    fault.plot(ax=ax, color='green', alpha=0.5, label='Landslide Zone')
    # anchor_buff.to_crs(4326).boundary.plot(ax=ax)
    landslide_vector.plot(ax=ax, zorder=9, color='red', label='Landslide Direction')
    # pipe_buff.to_crs(4326).plot(ax=ax, color='orange', zorder=12)
    ax.text(pipe.iloc[0].geometry.centroid.xy[0][0], pipe.iloc[0].geometry.centroid.xy[1][0], '   β: '+str(round(beta,1)), zorder=11)
    ax.legend()

    ax.set_xlim(minx - 300, maxx + 300) # added/substracted value is to give some margin around total bounds
    ax.set_ylim(miny - 300, maxy + 300)
    # ax.scatter(landslide_vector.iloc[0].coords[1][0], landslide_vector.iloc[0].coords[1][1], marker=(3, 0, 45))
    '''
    temp = {
        'beta_crossing':beta,
        'strike':'NA',
        'psi_dip':'NA',
        'theta_slip':landslide_direction,
        'l_anchor':La
        # 'landslide':fault.iloc[0]. 
        }
    temp = {**dseg, **temp} #combine
    dict_list.append(temp)
    
    # if n % 10 == 0:
    print(time.time() - start_time)
    n+=1
landslide_geometries = pd.DataFrame.from_dict(dict_list)
landslide_geometries.to_csv(os.path.join(r'C:\Users\TomClifford\SlateGeotech\SlateDrive - Documents\_Projects\18-020.00_CEC_Seismic_Risk\04_Eval_Analysis\04 - Research\Task 4a',
                                          'cordelia_junction_crossing.csv'), index=False)
