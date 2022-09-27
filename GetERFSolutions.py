# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Retrieve solutions from UCERF3 through OpenSHA
#
# Created: December 10, 2020
# Updated: April 5, 2021
# @author: Barry Zheng (Slate Geotechnical Consultants)
#          
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import os, json
import pandas as pd
import numpy as np
import jpype
from jpype import imports
from jpype.types import *

# OpenSRA modules
from src.IM import OpenSHAInterface


# -----------------------------------------------------------
# INPUT
# list of models: Mean UCERF3, Mean UCERF3 FM3.1, Mean UCERF3 FM3.2
# erf_model = 'Mean UCERF3'
# erf_model = 'Mean UCERF3 FM3.1'
erf_model = 'Mean UCERF3 FM3.2'
print(f'ERF model: {erf_model}')
# export extension: csv or txt
ext = 'csv'


# -----------------------------------------------------------
# save directory
save_dir = os.path.join(os.getcwd(),'lib','OpenSHA','ERF',erf_model)
if os.path.isdir(save_dir) is False: # check if directory exists
    os.mkdir(save_dir) # create folder if it doesn't exist
        
# interface with OpenSHA 
erf = OpenSHAInterface.getERF(erf_model) # initialize ERF
nSources = erf.getNumSources() # get number of sources for ERF model
rupSet, rupSourceSections = OpenSHAInterface.get_rupture_set(erf) # get rupture set from ERF solution

# initialize dictionaries and lists
seg_unique = {}
# ps_unique = {}
src_connect = []
ps_id = []
ps_lon = []
ps_lat = []
ps_z = []

# loop through all sources and get information
print(f'Number of sources: {nSources}')
for src_i in range(nSources):
    # print current spot in loop
    if src_i%10000 == 0:
        print(f'\t...{src_i}')
    # set current source
    source = erf.getSource(src_i)
    # check if point source
    if source.getSourceSurface().isPointSurface():
        # point sources
        location = source.getSourceSurface().getLocation()
        ps_id.append(src_i)
        ps_lon.append(float(location.getLongitude()))
        ps_lat.append(float(location.getLatitude()))
        ps_z.append(float(location.getDepth()))
    else:
        # get list of segments
        listSeg = list(rupSourceSections[src_i].toArray())
        src_connect.append(listSeg)
        # get list of nodes for all segments in current source
        for j in range(len(listSeg)):
            if not str(listSeg[j]) in seg_unique:
                seg_unique[str(listSeg[j])] = {}
                section = rupSet.getFaultSectionData(listSeg[j])
                trace = section.getFaultTrace()
                nodes = []
                for point in trace:
                    nodes.append([float(point.getLongitude()),float(point.getLatitude()),float(point.getDepth())])
                #
                seg_unique[str(listSeg[j])].update({'name':str(section.getSectionName())})
                seg_unique[str(listSeg[j])].update({'trace':nodes})

# export segment connectivities for sources
dict_src_connect = {
    'SourceIndex': np.arange(len(src_connect)),
    'ListOfSegments': src_connect
}
df_src_connect = pd.DataFrame.from_dict(dict_src_connect)
df_src_connect.to_csv(os.path.join(save_dir,'ScenarioConnectivity.'+ext),index=False)

# export list of rupture segments
seg_id = []
seg_name = []
seg_trace = []
for key in seg_unique.keys():
    seg_id.append(int(key))
    seg_name.append(seg_unique[key]['name'])
    seg_trace.append(np.round(seg_unique[key]['trace'],decimals=5).tolist())
dict_rup_seg = {
    'SegmentID': seg_id,
    'SegmentName': seg_name,
    'SegmentTrace': seg_trace
}
df_rup_seg = pd.DataFrame.from_dict(dict_rup_seg)
df_rup_seg.to_csv(os.path.join(save_dir,'RuptureSegments.'+ext),index=False)
    
# export list of point sources
dict_ps = {
    'SourceIndex': ps_id,
    'Longitude': ps_lon,
    'Latitude': ps_lat,
    'Depth': ps_z
}
df_ps = pd.DataFrame.from_dict(dict_ps)
df_ps.to_csv(os.path.join(save_dir,'PointSources.'+ext),index=False)