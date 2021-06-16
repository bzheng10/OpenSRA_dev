# import modules
import os
import json
import time
import numpy as np
import pandas as pd

# import OpenSRA modules
from src.Fcn_Common import get_quick_dist

# input
erf_model = "Mean UCERF3"

# directories
erf_base_dir = r'C:\Users\barry\OneDrive - SlateGeotech\CEC\OpenSRA\lib\OpenSHA\ERF'
erf_dir = os.path.join(erf_base_dir,erf_model)
src_connect_file = os.path.join(erf_dir,'ScenarioConnectivity.csv')
rup_seg_file = os.path.join(erf_dir,'RuptureSegments.csv')

# load ERF solution files
df_src_connect = pd.read_csv(src_connect_file)
df_rup_seg = pd.read_csv(rup_seg_file)

# get list of source indices
src_list = df_src_connect['SourceIndex'].values

# get traces and add to list
# max_test = 100 # for testing
print_count = 1000
time_start = time.time()
time_curr = time_start
print('start')
print('---------')
print('.')
print('.')
print('.')
src_trace = []
# loop through source indices, first pull list of segment IDs, and then get list of traces for each segment
for count_src,src_i in enumerate(src_list):
    # get list of segment IDs from the "scenario_connectivity" file
    listSeg_src_i = json.loads(df_src_connect['ListOfSegments'][src_i])
    # initialize lists and counters for looping through list of segments
    trace_src_i = []
    # loop through list of segment IDs
    for count_seg,seg_j in enumerate(listSeg_src_i):
        # get list of traces for current segment ID
        trace_seg_j = json.loads(df_rup_seg['SegmentTrace'][np.where(df_rup_seg['SegmentID']==seg_j)[0][0]])
        trace_seg_j = [[trace[0],trace[1]] for trace in trace_seg_j]
        # it appears that in OpenSHA the directionality of segment traces are not consistently set up
        # below is a scheme to rearrange the order of traces such that segment endpoints are connected correctly
        if count_seg == 0: # no directionality enforced for first segment
            trace_src_i = trace_src_i + trace_seg_j
        elif count_seg == 1:
            dists = [
                get_quick_dist(trace_seg_jm1[-1][0],trace_seg_jm1[-1][1],trace_seg_j[0][0],trace_seg_j[0][1]),
                get_quick_dist(trace_seg_jm1[-1][0],trace_seg_jm1[-1][1],trace_seg_j[-1][0],trace_seg_j[-1][1]),
                get_quick_dist(trace_seg_jm1[0][0],trace_seg_jm1[0][1],trace_seg_j[0][0],trace_seg_j[0][1]),
                get_quick_dist(trace_seg_jm1[0][0],trace_seg_jm1[0][1],trace_seg_j[-1][0],trace_seg_j[-1][1])
            ]
            if dists.index(min(dists)) == 0: # no correction needed, directions of segments align
                trace_src_i = trace_seg_jm1 + trace_seg_j
            elif dists.index(min(dists)) == 1: # current segment is flipped relative to previous segment
                trace_src_i = trace_seg_jm1 + trace_seg_j[::-1]
            elif dists.index(min(dists)) == 2: # previous segment is flipped relative to current
                trace_src_i = trace_seg_jm1[::-1] + trace_seg_j
            elif dists.index(min(dists)) == 3: # both segments are oriented in opposite direction
                trace_src_i = trace_seg_jm1[::-1] + trace_seg_j[::-1]
        else:
            dists = [
                get_quick_dist(trace_seg_jm1[-1][0],trace_seg_jm1[-1][1],trace_seg_j[0][0],trace_seg_j[0][1]),
                get_quick_dist(trace_seg_jm1[-1][0],trace_seg_jm1[-1][1],trace_seg_j[-1][0],trace_seg_j[-1][1])
            ]
            if dists.index(min(dists)) == 0: # no correction needed, directions of segments align
                trace_src_i = trace_src_i + trace_seg_j
            else: # only need flip current segment since directionality has been established by previous segments
                trace_src_i = trace_src_i + trace_seg_j[::-1]
        # temporarily store traces of current segment for use in next iteration
        trace_seg_jm1 = trace_seg_j
    # remove consecutive repeating traces
    trace_src_i = [trace_src_i[i] for i in range(len(trace_src_i)) if trace_src_i[i] != trace_src_i[i-1]]
    # append traces for current source
    src_trace.append(trace_src_i)
    # for testing
    # if count_src > max_test-1:
        # break
    # print message
    if (count_src+1)%print_count == 0:
        time_end = time.time()
        print(f'{count_src+1}: {round(time_end-time_curr,2)} secs')
        time_curr = time_end
print('---------')
time_end = time.time()
print(f'done with {count_src+1} sources: {round(time_end-time_start,2)} secs')

# construct to pd.DataFrame
df = pd.DataFrame.from_dict({
    'SourceIndex': df_src_connect['SourceIndex'].values[:len(src_trace)],
    'ListOfTraces': src_trace
})

# export to zip file
df.to_csv(os.path.join(erf_dir,'ScenarioTraces.zip'),index=False,compression='zip')