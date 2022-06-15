# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Python interface to OpenSHA
#
# Created: December 10, 2020
# @author: Wael Elhaddad (SimCenter)
#          Kuanshi Zhong (Stanford University)
#          Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import os
import logging
import json
import time
# import sys
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import LineString
from shapely.ops import unary_union
# from multiprocessing.dummy import Pool as ThreadPool
# import itertools
import jpype
from jpype import imports
from jpype.types import *

# OpenSRA modules
from src.util import check_common_member, get_closest_pt, get_haversine_dist

# Using JPype to load OpenSHA in JVM
opensha_dir = os.path.join(os.path.dirname(os.getcwd()),'OpenSRA','lib','OpenSHA')
jpype.addClassPath(os.path.join(opensha_dir,'OpenSHA-1.5.2.jar'))
# jpype.addClassPath('../../lib/OpenSHA/OpenSHA-1.5.2.jar')
if not jpype.isJVMStarted():
    jpype.startJVM("-Xmx8G", convertStrings=False)

from java.io import *
from java.lang import *
from java.lang.reflect import *
from java.util import ArrayList

from org.opensha.commons.data import *
from org.opensha.commons.data.siteData import *
from org.opensha.commons.data.function import *
from org.opensha.commons.exceptions import ParameterException
from org.opensha.commons.geo import *
from org.opensha.commons.param import *
from org.opensha.commons.param.event import *
from org.opensha.commons.param.constraint import *
from org.opensha.commons.util import ServerPrefUtils

from org.opensha.sha.earthquake import *
from org.opensha.sha.earthquake.param import *
from org.opensha.sha.earthquake.rupForecastImpl.Frankel02 import Frankel02_AdjustableEqkRupForecast
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF1 import WGCEP_UCERF1_EqkRupForecast
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final import UCERF2
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final.MeanUCERF2 import MeanUCERF2
from org.opensha.sha.faultSurface import *
from org.opensha.sha.imr import *
from org.opensha.sha.imr.attenRelImpl import *
from org.opensha.sha.imr.attenRelImpl.ngaw2 import *
from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *
from org.opensha.sha.imr.param.IntensityMeasureParams import *
from org.opensha.sha.imr.param.OtherParams import *
from org.opensha.sha.imr.param.SiteParams import *
from org.opensha.sha.calc import *
from org.opensha.sha.calc.params import *
from org.opensha.sha.util import *
from scratch.UCERF3.erf.mean import MeanUCERF3

from org.opensha.sha.gcim.imr.attenRelImpl import *
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *
from org.opensha.sha.gcim.calc import *


class OpenSHA(object):
    
    
    
    
    
def 






# -----------------------------------------------------------
def get_fault_xing_opensha(src_list, start_loc, end_loc, trace_set, save_name,
                            to_write=True):
    """
    """
    
    # to read from save_name or to write to save_name
    if to_write is True:
        # initialize list
        crossing_seg_id = [] 
        crossing_location = []
        
        # create line shapes with start-end locations
        lines = [LineString([start_loc[i],end_loc[i]]) for i in range(len(start_loc))]
        # join all components
        lines_unary = unary_union(lines)
        # return events that intersect the entire network of components
        events_with_intersections = [i for i in range(len(trace_set)) if LineString(trace_set[i]).intersects(lines_unary)]

        # get crossings for events that intersect the network
        crossing_seg_id = []
        crossing_location = []
        # loop through unique source indices and get intersections
        for i in range(len(trace_set)):
            if i in events_with_intersections:
                rup_shape = LineString(trace_set[i])
                # get crossings
                ind_with_crossing_i = []
                crossings_i = []
                for ind, line in enumerate(lines):
                    if line.intersects(rup_shape):
                        ind_with_crossing_i.append(ind)
                        try: # for single intersection
                            # x.intersection returns Point object and cannot be looped
                            crossings_i.append([
                                round(line.intersection(rup_shape).x,5),
                                round(line.intersection(rup_shape).y,5)])
                        except: # for multiple intersections
                            # x.intersection returns Multipoint object and MUST be looped
                            for intersection in line.intersection(rup_shape):
                                crossings_i.append([
                                    round(intersection.x,5),
                                    round(intersection.y,5)])
                # add crossings and locations to list
                crossing_seg_id.append(ind_with_crossing_i)
                crossing_location.append(crossings_i)
            else:
                crossing_seg_id.append([])
                crossing_location.append([])
        
        # export traces
        dict_out = {
            'SourceIndex': src_list,
            'ListOfSegmentIDsWithCrossings': crossing_seg_id,
            'ListOfCrossingLocations': crossing_location
        }
        df_out = pd.DataFrame.from_dict(dict_out)
        df_out.to_csv(save_name,index=False)    
        logging.info(f"\t... list of fault crossings exported to:")
        logging.info(f"\t\t{save_name}")
        
        # dataframe to return
        dict_return = {
            'SourceIndex': src_list,
            'ListOfSegmentIDsWithCrossings': crossing_seg_id,
        }
        df_return = pd.DataFrame.from_dict(dict_return)
        #
        return df_return
        
    #
    else:
        # import fault crossings from "file_name"
        df_return = pd.read_csv(save_name)
        # remove the last column of data (unused in analysis)
        df_return.drop(columns=['ListOfCrossingLocations'], inplace=True)
        # convert string literals to lists
        df_return['ListOfSegmentIDsWithCrossings'] = \
            [json.loads(df_return['ListOfSegmentIDsWithCrossings'][i]) for i in range(len(df_return))]
        #
        return df_return
    

# -----------------------------------------------------------
def get_trace_opensha(src_list, finite_src_file, pt_src_file, save_name,
                        flag_include_pt_src=False, to_write=True):
    """
    """
    
    # to read from save_name or to write to save_name
    if to_write is True:
        # load ERF solution files
        # df_src_connect = pd.read_csv(src_connect_file)
        # df_rup_seg = pd.read_csv(rup_seg_file)
        df_finite_src = pd.read_csv(finite_src_file)
        if flag_include_pt_src:
            df_pt_src = pd.read_csv(pt_src_file)
            ind_end_of_finite_src = np.argmax(src_list>=df_pt_src.iloc[0,0]) # index where point sources
        else:
            ind_end_of_finite_src = len(src_list)
        
        # initialize list
        # trace_set = []
        
        # get traces for finite sources
        finite_src_to_get = src_list[:ind_end_of_finite_src]
        trace_set_finite_src = [json.loads(df_finite_src['ListOfTraces'][ind]) for ind in finite_src_to_get]
        trace_set = trace_set_finite_src
        
        # loop through source indices, first pull list of segment IDs, and then get list of traces for each segment
        #for src_i in src_list:
        #    # see if source index is in list of line sources, if not then it is a point source
        #    if src_i in df_src_connect['ListOfSegments']:
        #        # get list of segment IDs from the "scenario_connectivity" file
        #        listSeg_src_i = json.loads(df_src_connect['ListOfSegments'][src_i])
        #        # initialize lists and counters for looping through list of segments
        #        trace_src_i = []
        #        count_seg = 0
        #        # loop through list of segment IDs
        #        for seg_j in listSeg_src_i:
        #            # get list of traces for current segment ID
        #            trace_seg_j = json.loads(df_rup_seg['SegmentTrace'][np.where(df_rup_seg['SegmentID']==seg_j)[0][0]])
        #            # it appears that in OpenSHA the directionality of segment traces are not consistently set up
        #            # below is a scheme to rearrange the order of traces such that segment endpoints are connected correctly
        #            if count_seg == 0: # no directionality enforced for first segment
        #                trace_src_i = trace_src_i + trace_seg_j
        #            elif count_seg == 1:
        #                if trace_seg_jm1[-1] == trace_seg_j[0]: # no correction needed, directions of segments align
        #                    trace_src_i = trace_seg_jm1 + trace_seg_j
        #                elif trace_seg_jm1[-1] == trace_seg_j[-1]: # current segment is flipped relative to previous segment
        #                    trace_src_i = trace_seg_jm1 + trace_seg_j[::-1]
        #                elif trace_seg_jm1[0] == trace_seg_j[0]: # previous segment is flipped relative to current
        #                    trace_src_i = trace_seg_jm1[::-1] + trace_seg_j
        #                elif trace_seg_jm1[0] == trace_seg_j[-1]: # both segments are oriented in opposite direction
        #                    trace_src_i = trace_seg_jm1[::-1] + trace_seg_j[::-1]
        #            else:
        #                if trace_seg_jm1[-1] == trace_seg_j[0]:# no correction needed, directions of segments align
        #                    trace_src_i = trace_src_i + trace_seg_j
        #                else: # only need flip current segment since directionality has been established by previous segments
        #                    trace_src_i = trace_src_i + trace_seg_j[::-1]
        #            # temporarily store traces of current segment for use in next iteration
        #            trace_seg_jm1 = trace_seg_j
        #            # increment counter
        #            count_seg += 1
        #    # if source index is not in the "scenario_connectivity" file, then it's a point source
        #    # traces of point sources are stored in "point_sources.csv"
        #    else:
        #        # get trace for current source index, longitudes and latitudes are stored in column indices 1:2
        #        trace_src_i = df_pt_src.iloc[np.where(df_pt_src['SourceIndex']==src_i)[0][0],1:3].values.tolist()
        #    # append traces for current source index to overall list
        #    trace_set.append(trace_src_i)
        
        # get traces for point sources
        if flag_include_pt_src:
            pt_src_to_get = src_list[ind_end_of_finite_src:]
            ind_for_pt_src = [[i]*len(np.where(pt_src_to_get==df_pt_src['SourceIndex'][i])[0]) for i in range(len(df_pt_src['SourceIndex'])) if df_pt_src['SourceIndex'][i] in pt_src_to_get]
            trace_set_pt_src = list(np.round(df_pt_src.iloc[[val for sublist in ind_for_pt_src for val in sublist],1:3].values,3))
            trace_set_pt_src = [[trace.tolist()] for trace in trace_set_pt_src]
            trace_set = trace_set + trace_set_pt_src
        
        # export traces
        dict_out = {
            'SourceIndex': src_list,
            'ListOfTraces': trace_set
        }
        df = pd.DataFrame.from_dict(dict_out)
        df.to_csv(save_name,index=False)
        logging.info(f"\t... list of traces for rupture scenarios exported to:")
        logging.info(f"\t\t{save_name}")
        
        #
        return trace_set
    
    #
    else:
        # import fault traces from "file_name"
        df = pd.read_csv(save_name)
        # remove the last column of data (unused in analysis)
        trace_set = [json.loads(item) for item in df['ListOfTraces'].tolist()]
        # trace_set = [item if len(item) > 0 else [] for item in trace_set]
        #
        return trace_set
    

# -----------------------------------------------------------
def setup_opensha(setup_config, other_config_param, site_data):
    """
    Sets up interface to OpenSHA to retrieve GM predictions
    
    """
    
    #
    flag_get_vs30_from_opensha = False
    
    # Initialize ERF
    logging.info(f"\n\n     *****Runtime messages from OpenSHA*****\n")
    erf_name = setup_config['IntensityMeasure']['SourceForIM']['OpenSHA']['SeismicSourceModel']
    erf = getERF(erf_name)
    logging.info(f"\n\n     *****Runtime messages from OpenSHA*****\n")
    logging.info(f'\t... initialized ERF "{erf_name}"')
    
    # Set to include background source
    flagIncludeBackground = setup_config['IntensityMeasure']['SourcebarryForIM']['OpenSHA']['Filter']['PointSource']['ToInclude']
    if flagIncludeBackground:
        erf.setParameter(IncludeBackgroundParam.NAME, IncludeBackgroundOption.INCLUDE)
        logging.info(f"\t... point sources included in list of scenarios")
    else:
        erf.setParameter(IncludeBackgroundParam.NAME, IncludeBackgroundOption.EXCLUDE)
        logging.info(f"\t... point sources excluded from list of scenarios")
    
    # Initialize IMR
    gmpe_name = setup_config['IntensityMeasure']['SourceForIM']['OpenSHA']['GroundMotionModel']
    try:
        imr = getIMR(gmpe_name)
        logging.info(f'\t... created instance of IMR "{gmpe_name}"')
    except:
        print('Please check GMPE name.')
        return 1, station_info
    
    # Set site dictionary
    n_site = len(site_data)
    siteSpec = {
        'Latitude': site_data['Mid Latitude'].values,
        'Longitude': site_data['Mid Longitude'].values,
    }
    # get other parameters
    other_param_definition = setup_config['IntensityMeasure']['SourceForIM']['OpenSHA']
    for param in other_param_definition:
        # limit to Vs30,z1p0,z2p5 for now:
        if 'Vs30' in param or 'Z1p0' in param or 'Z2p5' in param:
            if param == 'Vs30':
                siteSpec['Vs30'] = site_data['Vs30 (m/s)'].values # get vs30
            else:
                if 'UserDefined' in other_param_definition[param]:
                    siteSpec[param] = site_data[other_param_definition[param]['UserDefined']['ColumnIDWithData']].values
                    # convert z1p0 from km to m
                    if param == 'Z1p0':
                        siteSpec[param] = siteSpec[param]*1000
                else:
                    siteSpec[param] = None
                
    # create Java objects of sites
    sites = get_site_prop(
        imr, siteSpec, n_site,
        outfile=os.path.join(other_config_param['Dir_IM_GroundMotion_Prediction'],'SiteDataUsedInGMM.csv')
    )
    # Set max distance
    if setup_config['IntensityMeasure']['SourceForIM']['OpenSHA']['Filter']['Distance']['ToInclude']:
        r_max = setup_config['IntensityMeasure']['SourceForIM']['OpenSHA']['Filter']['Distance']['Maximum']
        imr.setUserMaxDistance(r_max)
        logging.info(f"\t... set max distance to {r_max} km")
    
    # Set intensity measures
    #list_im = other_config_param['IM']
    #logging.info(f"\t... setting IMs of interest in GMM:")
    #for im in list_im:
    #    try:
    #        imr.setIntensityMeasure(im)
    #        logging.info(f"\t\t{im} - added")
    #    except:
    #        logging.info(f"\t\t{im} - not available for GMM")
    #
    return erf, imr, sites


# -----------------------------------------------------------
def filter_ruptures(erf, locs, filter_criteria, rupture_list, rup_save_name,
                    rup_seg_file, pt_src_file):
    """
    Filter rupture scenarios; locs should be in [lon,lat] order
    
    """

    # filter by rate_cutoff (or return period)
    if 'ReturnPeriod' in filter_criteria:
        rate_cutoff = 1/filter_criteria['ReturnPeriod']['Maximum']
        logging.info(f"\t\t- filtered scenarios with mean annual rates less than {rate_cutoff}")
        rupture_list = rupture_list[rupture_list['MeanAnnualRate']>=rate_cutoff]
        rupture_list.reset_index(inplace=True,drop=True)
        logging.info(f"\t\t\tnumber of scenarios after filter by rates = {len(rupture_list)}")

    # filter by magnitude range
    if 'Magnitude' in filter_criteria:
        mag_min = filter_criteria['Magnitude']['Minimum']
        mag_max = filter_criteria['Magnitude']['Maximum']
        logging.info(f"\t\t- filtering scenarios by magnitude range: min = {mag_min}, max = {mag_max}")
        if mag_min is not None:
            rupture_list = rupture_list[rupture_list['Magnitude']>=mag_min]
            rupture_list.reset_index(inplace=True,drop=True)
        if mag_max is not None:
            rupture_list = rupture_list[rupture_list['Magnitude']<mag_max]
            rupture_list.reset_index(inplace=True,drop=True)
        logging.info(f"\t\t\tnumber of scenarios after filter by magnitude = {len(rupture_list)}")

    # filter out inclusion of point sources
    df_pt_src = pd.read_csv(pt_src_file) # get list of point sources
    starting_of_pt_src = df_pt_src['SourceIndex'][0].copy()
    ind_end_of_finite_src = np.argmax(rupture_list['SourceIndex']>=df_pt_src.iloc[0,0]) # index where point sources start
    if not 'PointSource' in filter_criteria:
        rupture_list = rupture_list[:ind_end_of_finite_src] # get finite sources only

    # read list of rupture segments in source model
    if 'Distance' in filter_criteria:
        #
        r_max = filter_criteria['Distance']['Maximum']
        logging.info(f"\t\t- filtering scenarios that are more than {r_max} km from sites")
        df_rup_seg = pd.read_csv(rup_seg_file) # get list of rupture segments
        # check site locations against all rupture segments and see if shortest distance is within r_max                
        seg_pass_r_max = [df_rup_seg['SegmentID'][i] for i in range(len(df_rup_seg)) if check_rmax(locs,json.loads(df_rup_seg.iloc[i,2]),'seg',r_max)]
        logging.info(f"\t\t\t... obtained list of {len(seg_pass_r_max)} segments that are within {r_max} km of site locations")
    
        # check if point sources are needed
        pt_src_pass_r_max = []
        if 'PointSource' in filter_criteria:
            # check site locations against all point sources and see if shortest distance is within r_max                    
            pt_src_pass_r_max = [int(df_pt_src.iloc[i,0]) for i in range(len(df_pt_src)) if check_rmax(locs,df_pt_src.iloc[i,1:].values,'pt',r_max)]
            logging.info(f"\t\t\t... obtained list of {len(pt_src_pass_r_max)} point sources that are within {r_max} km of site locations")

        # get rupture section class from OpenSHA through EQHazard
        _, rupSourceSections = get_rupture_set(erf)
        
        # compare list of rupture segments that are wihtin r_max and with the rupture segments in each source
        if 'PointSource' in filter_criteria:
            pt_src_to_check = rupture_list['SourceIndex'][ind_end_of_finite_src:].values
        else:
            pt_src_to_check = []
        finite_src_to_check = rupture_list['SourceIndex'][:ind_end_of_finite_src].values
        # get rupture segments for finite sources
        set_seg_pass_r_max = set(seg_pass_r_max)
        rupture_list_finite = [i for i in finite_src_to_check if set(list(rupSourceSections[i].toArray())) & set_seg_pass_r_max]
        # rupture_list_finite = rupture_list.loc[rupture_list_finite]
        rupture_list_finite = rupture_list[np.in1d(rupture_list['SourceIndex'],rupture_list_finite)]
        # get locations of point sources
        common_index = sorted(list(set(pt_src_to_check).intersection(pt_src_pass_r_max)))
        rupture_list_pt_src = rupture_list[np.in1d(rupture_list['SourceIndex'],common_index)]
        rupture_list = pd.concat([rupture_list_finite,rupture_list_pt_src],ignore_index=True)
        logging.info(f"\t\t\tnumber of scenarios after filter by r_max = {len(rupture_list)}")
    
    # store filtered list of rupture metainfo
    rupture_list.to_csv(rup_save_name, index=False)
    logging.info(f"\t... filtered rupture scenarios exported to:")
    logging.info(f"\t\t{rup_save_name}")

    # store trace of filtered scenarios
    #df_out = pd.DataFrame(columns=['SourceID', 'RuptureID', 'Geometry'])
    #if len(rupture_list) > 0:
    #   for i in range(len(rupture_list)):
    #       # set current source and rupture indices and retrieve rupture surface
    #       source = erf.getSource(int(rupture_list[i][0]))
    #       rupture = source.getRupture(int(rupture_list[i][1]))
    #       ruptureSurface = rupture.getRuptureSurface()
    #       # get trace from rupture surface
    #       try:
    #           trace = ruptureSurface.getUpperEdge()
    #       except:
    #           trace = ruptureSurface.getEvenlyDiscritizedUpperEdge()
    #       # store coordinates into list
    #       coordinates = []
    #       for val in trace:
    #           coordinates.append([
    #               float(val.getLongitude()),
    #               float(val.getLatitude()),
    #               float(val.getDepth())
    #           ])
    #       # round decimals, convert to string, and remove extra whitespaces
    #       coordinates = str(np.round(coordinates,decimals=4).tolist()).replace(' ','')
    #       # store processed list of traces into dataframe
    #       df_out.loc[i] = [int(rupture_list[i][0]),int(rupture_list[i][1]),coordinates]
    ## export to CSV
    #df_out.to_csv(trace_save_name,index=False)
    #logging.info(f"\t... list of traces for rupture scenarios exported to:")
    #logging.info(f"\t\t{trace_save_name,}")
    
    # create return dictionary
    # output = {
        # 'SourceIndex':rupture_list[:,0].astype(int),
        # 'RuptureIndex':rupture_list[:,1].astype(int),
        # 'Magnitude':rupture_list[:,2],
        # 'MeanAnnualRate':rupture_list[:,3]}
        
    #
    # return output
    return rupture_list
    

# -----------------------------------------------------------
def check_rmax(locs, seg_trace, seg_type, r_max):
    """
    check for list of rupture segments that are within r_max; locs in (lon, lat)
    
    """

    #
    if seg_type == 'seg': # finite source
        for i in range(len(seg_trace)-1):
            seg_sub = [seg_trace[i][:2],seg_trace[i+1][:2]]
            for site_j in locs:
                _,dist = get_closest_pt(site_j,seg_sub)
                if dist <= r_max:
                    return True
        return False

    elif seg_type == 'pt': # point source
        for site_j in locs:
            dist = get_haversine_dist(site_j[0],site_j[1],seg_trace[0],seg_trace[1])
            if dist <= r_max:
                return True
        return False
    

# -----------------------------------------------------------
def get_rupture_set(erf):
    #Read the fault system solution
    sol = erf.getSolution()

    #Read the rupture set
    rupSet = sol.getRupSet()

    #Get the number of section
    numSections = rupSet.getNumSections()

    #Map rupture sources to sections
    rupSourceSections = rupSet.getSectionIndicesForAllRups()
    
    return rupSet, rupSourceSections


# -----------------------------------------------------------
def get_eq_rup_meta(erf=None, rupture_list_file=None, ind_range=['all'],
                    rate_cutoff=None, m_min=None, m_max=None, rup_group_file=None,
                    rup_per_group=None, file_type='txt'):
    """
    Gets sources, ruptures, rates, and mags from OpenSHA and metadata exported. If metadata file already exists, use existing.
    
    Parameters
    ----------
    rupture_list_file : str
        file name of the file containing rupture metadata
    proc : object
        OpenSHA processor object; used if **rupture_list_file** does not exist to obtain source parameters
    ind_range : str/list, optional
        define indices to extract source parameters for: options are **['all']**, **[index]** or **[low, high]** (brackets are required; replace index, low, and high with integers)
    
    Returns
    -------
    gm_source_info : dictionary
        contains list of source indices, rupture indices, moment magnitudes, and mean annual rates
    
    """
    
    # see rupture_list_file is given
    if rupture_list_file is None:
        # get and store rates and Mw
        nSources = erf.getNumSources()
        rup_meta = [[i, j]+get_mag_rate(erf,i,j)
                    for i in range(nSources) for j in range(erf.getNumRuptures(i))]
        logging.info(f"\t... total number of scenarios of {len(rup_meta)}")
        #
        return rup_meta
    else:
        # see if extension is provided, if not, add it
        # if not '.'+file_type in rupture_list_file:
            # rupture_list_file = rupture_list_file+'.'+file_type

        # initialize dictionaries
        # rup_meta = {'SourceIndex':None,'RuptureIndex':None,'Magnitude':None,'MeanAnnualRate':None}
        rup_meta = pd.read_csv(rupture_list_file)
        # load rupture_list_file
        # txt file format
        # if 'txt' in file_type:
            # f = np.loadtxt(rupture_list_file,unpack=True)
        if len(ind_range) == 1:
            if ind_range[0] == 'all':
                # rup_meta['SourceIndex'] = f[0].astype(int)
                # rup_meta['RuptureIndex'] = f[1].astype(int)
                # rup_meta['Magnitude'] = f[2]
                # rup_meta['MeanAnnualRate'] = f[3]
                pass
            else:
                # rup_meta['SourceIndex'] = f[0,ind_range[0]].astype(int)
                # rup_meta['RuptureIndex'] = f[1,ind_range[0]].astype(int)
                # rup_meta['Magnitude'] = f[2,ind_range[0]]
                # rup_meta['MeanAnnualRate'] = f[3,ind_range[0]]
                rup_meta = rup_meta.iloc[:,ind_range[0]]
                rup_meta.reset_index(inplace=True)
        elif len(ind_range) == 2:
            # rup_meta['SourceIndex'] = f[0,ind_range[0]:ind_range[1]].astype(int)
            # rup_meta['RuptureIndex'] = f[1,ind_range[0]:ind_range[1]].astype(int)
            # rup_meta['Magnitude'] = f[2,ind_range[0]:ind_range[1]]
            # rup_meta['MeanAnnualRate'] = f[3,ind_range[0]:ind_range[1]]
            rup_meta = rup_meta.iloc[:,ind_range[0]:ind_range[1]]
            rup_meta.reset_index(inplace=True)
        # ----------
        # convert from Pandas series to list
        # rup_meta = rup_meta.to_dict()
        # ---------- 
        #
        return rup_meta
    
    
# -----------------------------------------------------------
def get_mag_rate(erf, src, rup):
    """
    This extracts the mean annual rate and moment magnitude for target scenario (source + rupture index) with filters on rate and magnitude
    
    Parameters
    ----------
    erf : object
        OpenSHA ERF object
    src : int
        source index used by OpenSHA
    rup : int
        rupture index used by OpenSHA
        
    Returns
    -------
    mag : float
        moment magnitude of scenario
    rate : float
        mean annual rate for scenario
    
    """
    
    # Set current scenario
    rup = erf.getSource(src).getRupture(rup)
    # Return mean annual rate and moment magnitude
    # OpenSHA reference duration is accessed by erf.getTimeSpan().getDuration()
    return [rup.getMag(), rup.getMeanAnnualRate(erf.getTimeSpan().getDuration())]


# -----------------------------------------------------------
def get_IM(erf, imr, sites, src_list, rup_list, list_im=['PGA','PGV'],
    saveDir=None, store_file_type='txt', r_max=200):
    """
    Get IM from OpenSHA; developed by Kuanshi Zhong (SimCenter), modified by Barry Zheng
    
    """
    
    # IM param map to save name
    list_param = {
        'Median': 'median',
        'InterEvStdDev': 'stdev_inter',
        'IntraEvStdDev': 'stdev_intra',
        'TotalStdDev': 'stdev_total'}
    # Get available stdDev options
    # if imr.getShortName() == 'NGAWest_2014': # bypass check for inter and intra even stats since they exist for the base models
        # stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
        # hasIEStats = True
    # else:
    # try:
    stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
    if stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTER) and \
        stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA):
        hasIEStats = True
    else:
        hasIEStats = False
        # stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
        # hasIEStats = stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTER) and \
            # stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA)
    # except:
        # stdDevParam = None
        # hasIEStats = False
    # Set up output
    shape = (len(src_list),len(sites))
    output = {}
    for im in list_im:
        output[im] = {'Median': np.zeros(shape)}
        output[im].update({'TotalStdDev': np.zeros(shape)})
        if hasIEStats:
            output[im].update({'InterEvStdDev': np.zeros(shape)})
            output[im].update({'IntraEvStdDev': np.zeros(shape)})
    # Loop through scenarios
    for i in range(len(src_list)):
        # Setting up imr
        currentRupture = erf.getSource(src_list[i]).getRupture(rup_list[i])
        imr.setEqkRupture(currentRupture)
        # Looping over sites
        for j in range(len(sites)):
            # Set up the site in the imr
            imr.setSite(sites[j])
            # Get distance with rupture
            try:
                DistanceRup = imr.getStdDevIndependentParams().getValue('DistanceRup')
            except:
                DistanceRup = None
            try:
                DistanceJB = imr.getStdDevIndependentParams().getValue('DistanceJB')
            except:
                DistanceJB = None
            # Get max distance from rupture and JB distances
            if DistanceRup is None and DistanceJB is None:
                dist_max = 0
            elif DistanceRup is not None and DistanceJB is None:
                dist_max = DistanceRup
            elif DistanceRup is None and DistanceJB is not None:
                dist_max = DistanceJB
            else:
                dist_max = max(DistanceRup, DistanceJB)
            # loop through IM
            for im in list_im:
                # check if current IM is in list of possible IM output for GMM:
                # if im in imr.getIntensityMeasure().NAME:
                # Check if distances are within r_max, and get value only if true
                # if DistanceRup <= r_max and DistanceJB <= r_max:
                if dist_max <= r_max:
                    try:
                        imr.setIntensityMeasure(im)
                        output[im]['Median'][i,j] = float(np.exp(imr.getMean()))
                        output[im]['TotalStdDev'][i,j] = float(imr.getStdDev())
                        if hasIEStats:
                            stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
                            output[im]['InterEvStdDev'][i,j] = float(imr.getStdDev())
                            stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
                            output[im]['IntraEvStdDev'][i,j] = float(imr.getStdDev())
                    except:
                        pass
    # Convert to COO matrix and export if saveDir is given
    for im_i in list_im:
        for param_j in output[im_i].keys():
            output[im_i][param_j] = sparse.coo_matrix(output[im_i][param_j])
            if saveDir is not None:
                saveName = os.path.join(saveDir,im_i,param_j+'.'+store_file_type)
                np.savetxt(saveName,output[im_i][param_j].toarray(),fmt='%6.4e')
    #
    return output
    
    
# -----------------------------------------------------------
def get_site_prop(imr, siteSpec, n_site, outfile=None):
    """
    Get and update site properties; developed by Kuanshi Zhong (SimCenter), modified by Barry Zheng
    
    """

    # Available data sources in OpenSHA, in order of priority; index next to source is used in OpenSHA
    # Vs30:
    #     CGS/Wills VS30 Map (2015)                                             - 0
    #     Thompson VS30 Map (2018)                                              - 1
    #     CGS/Wills Site Classification Map (2006)                              - 2
    #     Global Vs30 from Topographic Slope (Wald & Allen 2008)                - 3
    # z2.5:
    #     SCEC Community Velocity Model Version 4, Iteration 26, Basin Depth    - 4
    #     SCEC CCA, Iteration 6, Basin Depth                                    - 6
    #     SCEC Community Velocity Model Version 4 Basin Depth                   - 8
    #     SCEC/Harvard Community Velocity Model Version 11.9.x Basin Depth      - 10
    #     SCEC CCA, Iteration 6, Basin Depth                                    - 12
    #     USGS Bay Area Velocity Model Release 8.3.0                            - 14
    # z1.0:
    #     SCEC Community Velocity Model Version 4, Iteration 26, Basin Depth    - 5
    #     SCEC CCA, Iteration 6, Basin Depth                                    - 7
    #     SCEC Community Velocity Model Version 4 Basin Depth                   - 9
    #     SCEC/Harvard Community Velocity Model Version 11.9.x Basin Depth      - 11
    #     SCEC CCA, Iteration 6, Basin Depth                                    - 13
    #     USGS Bay Area Velocity Model Release 8.3.0                            - 15

    # Site data
    sites = ArrayList()
    site_params = {
        'Vs30': {
            'Name': 'Vs30',
            'Index': [0,1,2,3]
        },
        'Z1p0': {
            'Name': 'Depth 1.0 km/sec',
            'Index': [5,7,9,11,13,15]
        },
        'Z2p5': {
            'Name': 'Depth 2.5 km/sec',
            'Index': [4,6,8,10,12,14]
        }
    }
    col_order = ['Latitude','Longitude']
    site_params_to_get = [key for key in siteSpec if siteSpec[key] is None]
    # setup for storage
    for param in site_params:
        col_order.append(param)
        if param in site_params_to_get:
            siteSpec[param] = []
        siteSpec[param+'_Source'] = []
        col_order.append(param+'_Source')
    # setup to get site params
    addSite = [sites.add(Site(Location(siteSpec['Latitude'][i], siteSpec['Longitude'][i]))) for i in range(n_site)]
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()
    availableSiteData = siteDataProviders.getAllAvailableData(sites)
    siteTrans = SiteTranslator()
    # Looping over all sites
    for i in range(n_site):
        # Current site
        site = sites.get(i)
        # Looping over site params
        for param in site_params:
            newParam = Parameter.clone(imr.getSiteParams().getParameter(site_params[param]['Name']))
            # if need to get from OpenSHA datasets
            if param in site_params_to_get:
                for j in site_params[param]['Index']:
                    if not np.isnan(float(availableSiteData.get(j).getValue(i).getValue())):
                        paramVal = availableSiteData.get(j).getValue(i).getValue()
                        paramSource = availableSiteData.get(j).getSourceName()
                        break
                newParam.setValue(paramVal)
                siteSpec[param].append(float(paramVal))
                siteSpec[param+'_Source'].append(str(paramSource))
            # if already in site_data
            else:
                try:
                    newParam.setValue(Double(siteSpec[param][i]))
                    if param == 'Vs30':
                        newParamType = Parameter.clone(imr.getSiteParams().getParameter('Vs30 Type'))
                        newParamType.setValue('Measured')
                        site.addParameter(newParamType)
                    siteSpec[param+'_Source'].append('User Defined')
                except jpype.JException as exception:
                    # catch exception, typically when values are outside the recommended range;
                    # if happens, get value from OpenSHA sources
                    for j in site_params[param]['Index']:
                        if not np.isnan(float(availableSiteData.get(j).getValue(i).getValue())):
                            paramVal = availableSiteData.get(j).getValue(i).getValue()
                            paramSource = availableSiteData.get(j).getSourceName()
                            break
                    newParam.setValue(paramVal)
                    siteSpec[param][i] = float(paramVal) # update value in siteSpec
                    siteSpec[param+'_Source'].append(str(paramSource))
            site.addParameter(newParam)
    # export siteSpec
    if outfile is not None:
        df_out = pd.DataFrame.from_dict(siteSpec)
        df_out.to_csv(outfile,index=False)
    #
    return sites


# -----------------------------------------------------------
def get_DataSource(paramName, siteData):
    """
    Fetch data source; developed by Kuanshi Zhong (SimCenter)
    
    """
    
    typeMap = SiteTranslator.DATA_TYPE_PARAM_NAME_MAP
    for dataType in typeMap.getTypesForParameterName(paramName):
        if dataType == SiteData.TYPE_VS30:
            for dataValue in siteData:
                if dataValue.getDataType() != dataType:
                    continue
                vs30 = Double(dataValue.getValue())
                if (not vs30.isNaN()) and (vs30 > 0.0):
                    return dataValue.getSourceName()
        elif (dataType == SiteData.TYPE_DEPTH_TO_1_0) or (dataType == SiteData.TYPE_DEPTH_TO_2_5):
             for dataValue in siteData:
                if dataValue.getDataType() != dataType:
                    continue
                depth = Double(dataValue.getValue())
                if (not depth.isNaN()) and (depth > 0.0):
                    return dataValue.getSourceName()
    return 1


# -----------------------------------------------------------
def getERF(erf_name, update_flag=True):
    """
    get ERF; developed by Kuanshi Zhong (SimCenter); modified by Barry Zheng
    
    """
    
    # Available SSC and names to use in setup configuration:
    #     WGCEP (2007) UCERF2 - Single Branch
    #     USGS/CGS 2002 Adj. Cal. ERF
    #     WGCEP UCERF 1.0 (2005)
    #     Mean UCERF3
    #     Mean UCERF3 FM3.1
    #     Mean UCERF3 FM3.2
    #     WGCEP Eqk Rate Model 2 ERF
    
    # Initialization
    erf = None
    # ERF model options
    if erf_name == 'WGCEP (2007) UCERF2 - Single Branch':
        erf = MeanUCERF2()
    elif erf_name == 'USGS/CGS 2002 Adj. Cal. ERF':
        erf = Frankel02_AdjustableEqkRupForecast()
    elif erf_name == 'WGCEP UCERF 1.0 (2005)':
        erf = WGCEP_UCERF1_EqkRupForecast()
    elif erf_name.startswith('Mean UCERF3'):
        erf = MeanUCERF3()
        if erf_name.endswith('FM3.1'): # Branch 3.1
            erf.setPreset(MeanUCERF3.Presets.FM3_1_BRANCH_AVG)
        elif erf_name.endswith('FM3.2'): # Branch 3.2
            erf.setPreset(MeanUCERF3.Presets.FM3_2_BRANCH_AVG)
        else: # Unspecified (both branches)
            erf.setPreset(MeanUCERF3.Presets.BOTH_FM_BRANCH_AVG)
    elif erf_name == 'WGCEP Eqk Rate Model 2 ERF':
        erf = UCERF2()
    else:
        raise ValueError(f'The ERF model "{erf_name}" is not supported.')
    if erf_name and update_flag:
        erf.updateForecast()
    # return
    return erf
    

# -----------------------------------------------------------
def getIMR(gmpe_name):
    """
    create IMR instance; developed by Kuanshi Zhong (SimCenter); modified by Barry Zheng
    
    """
    
    # Available GMMs and names to use in setup configuration:
    #     Abrahamson, Silva & Kamai (2014)
    #     Boore, Stewart, Seyhan & Atkinson (2014)
    #     Campbell & Bozorgnia (2014)
    #     Chiou & Youngs (2014)
    #     Idriss (2014)
    #     Bommer et al. (2009)
    #     Afshari & Stewart (2016)
    #     NGAWest2 2014 Averaged Attenuation Relationship
    #     NGAWest2 2014 Averaged No Idriss
    
    # GMPE name map
    gmpe_map = {str(ASK_2014.NAME): ASK_2014_Wrapper.class_.getName(),
                str(BSSA_2014.NAME): BSSA_2014_Wrapper.class_.getName(),
                str(CB_2014.NAME): CB_2014_Wrapper.class_.getName(),
                str(CY_2014.NAME): CY_2014_Wrapper.class_.getName(),
                str(Idriss_2014.NAME): Idriss_2014_Wrapper.class_.getName(),
                str(KS_2006_AttenRel.NAME): KS_2006_AttenRel.class_.getName(),
                str(BommerEtAl_2009_AttenRel.NAME): BommerEtAl_2009_AttenRel.class_.getName(),
                str(AfshariStewart_2016_AttenRel.NAME): AfshariStewart_2016_AttenRel.class_.getName(),
                str(NGAWest_2014_Averaged_AttenRel.NAME): NGAWest_2014_Averaged_AttenRel.class_.getName(),
                str(NGAWest_2014_Averaged_AttenRel.NGAWest_2014_Averaged_AttenRel_NoIdriss.NAME): NGAWest_2014_Averaged_AttenRel.NGAWest_2014_Averaged_AttenRel_NoIdriss.class_.getName()}
    # Check if NGAWest2 average relationship is requested
    if 'NGAWest2 2014 Averaged' in gmpe_name:
        # Different initialization for NGAWest2 Average IMR
        # Second arg in class call = boolean for Idriss
        if 'No Idriss' in gmpe_name:
            imr = NGAWest_2014_Averaged_AttenRel(None, False)
        else:
            imr = NGAWest_2014_Averaged_AttenRel(None, True)
    else:
        # Mapping GMPE name
        imrClassName = gmpe_map.get(gmpe_name, None)
        if imrClassName is None:
            raise ValueError(f'The GM model "{gmpe_name}" is not supported.')
            # return imrClassName
        # Getting the java class
        imrClass = Class.forName(imrClassName)
        ctor = imrClass.getConstructor()
        imr = ctor.newInstance()
    # Setting default parameters
    imr.setParamDefaults()
    # return
    return imr