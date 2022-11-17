# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for transient pipe strain
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python base modules
import argparse
import copy
import importlib
import json
import logging
import os
import shutil
import sys
import warnings

# scientific processing modules
import numpy as np
import pandas as pd
# suppress warning that may come up
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# for geospatial processing
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.errors import ShapelyDeprecationWarning
# suppress warning that may come up
warnings.simplefilter(action='ignore', category=ShapelyDeprecationWarning)

# OpenSRA modules
from src.im import haz
from src.site import geodata
from src.site.get_pipe_crossing import get_pipe_crossing
from src.site.get_well_crossing import get_well_crossing
from src.site.get_caprock_crossing import get_caprock_crossing
from src.util import set_logging


# -----------------------------------------------------------
# Main function
def main(work_dir, logging_level='info', logging_message_detail='simple'):
    """main function that runs the preprocess procedures"""
    
    # -----------------------------------------------------------
    # Setting logging level (e.g. DEBUG or INFO)
    set_logging(
        level=logging_level,
        msg_format=logging_message_detail
    )
    logging.info('\n---------------')
    
    # -----------------------------------------------------------
    # start of preprocess
    logging.info('Start of preprocessing for OpenSRA')
    counter = 1 # counter for stages of processing   
        
    # -----------------------------------------------------------
    # make directories
    # check current directory, if not at OpenSRA level, go up a level (happens during testing)
    
    if not os.path.basename(os.getcwd()) == 'OpenSRA' and not os.path.basename(os.getcwd()) == 'OpenSRABackEnd':
        os.chdir('..')
        
    opensra_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(work_dir,'Input')
    processed_input_dir = os.path.join(work_dir,'Processed_Input')
    if not os.path.isdir(processed_input_dir):
        os.mkdir(processed_input_dir)
    im_dir = os.path.join(work_dir,'IM')
    if not os.path.isdir(im_dir):
        os.mkdir(im_dir)
    logging.info(f'{counter}. Check and create file directories')
    logging.info('\tPerforming preprocessing of methods and input variables for OpenSRA')
    logging.info('\t\tOpenSRA backend directory')
    logging.info(f'\t\t\t- {opensra_dir}')
    logging.info('\t\tWorking directory given:')
    logging.info(f'\t\t\t- {work_dir}')
    logging.info('\t\tInput directory implied:')
    logging.info(f'\t\t\t- {input_dir}')
    logging.info('\t\tProcessed input directory for export of processed information:')
    logging.info(f'\t\t\t- {processed_input_dir}')
    logging.info('\t\tInensity measure directory:')
    logging.info(f'\t\t\t- {im_dir}')
    counter += 1
    
    # -----------------------------------------------------------
    # read important info from setup_config file
    setup_config_fpath = os.path.join(input_dir,'SetupConfig.json')
    with open(setup_config_fpath,'r') as f:
        setup_config = json.load(f)
    # General
    # work_dir = setup_config['General']['Directory']['Working']
    # Infrastructure
    infra_type = setup_config['Infrastructure']['InfrastructureType']
    # infra_type_map = {
    #     'Below Ground': 'below_ground',
    #     'Above Ground': 'above_ground',
    #     'Wells and Caprocks': 'wells_caprocks',
    # }
    # infra_type = infra_type_in[]
    infra_ftype = setup_config['Infrastructure']['DataType']
    infra_fname = setup_config['Infrastructure']['SiteDataFile']
    infra_geom_fpath = None
    if infra_fname == 'CA_Natural_Gas_Pipeline':
        # use internal preprocessed CSV file for state pipeline network
        infra_fpath = os.path.join(
            opensra_dir,
            r'lib\OtherData\Preprocessed\CA_Natural_Gas_Pipeline_Segments_WGS84',
            'CA_Natural_Gas_Pipeline_Segments_WGS84_Under100m_SUBSET.csv'
            # 'CA_Natural_Gas_Pipeline_Segments_WGS84_Under100m.csv'
        )
        infra_geom_fpath = infra_fpath.replace('.csv','_GeomOnly.shp')
        flag_using_state_network = True
    else:
        # create file path
        if infra_ftype == 'Shapefile':
            infra_fpath = os.path.join(
                # work_dir,
                input_dir,
                setup_config['Infrastructure']['SiteDataFile']
            )
            files = os.listdir(infra_fpath)
            for each in files:
                if each.endswith('shp'):
                    infra_fpath = os.path.join(infra_fpath,each)
                    break
        else:
            # infra_fpath = os.path.join(work_dir,setup_config['Infrastructure']['SiteDataFile'])
            print(input_dir,setup_config['Infrastructure']['SiteDataFile'])
            infra_fpath = os.path.join(input_dir,setup_config['Infrastructure']['SiteDataFile'])
        flag_using_state_network = False
    infra_loc_headers_in = setup_config['Infrastructure']['SiteLocationParams']
    if infra_type == 'below_ground':
        infra_loc_header_map = {
            "LatMid": 'lat_header',
            "LonMid": 'lon_header',
            "LatBegin": 'lat_begin_header',
            "LonBegin": 'lon_begin_header',
            "LatEnd": 'lat_end_header',
            "LonEnd": 'lon_end_header',
        }
    else:
        infra_loc_header_map = {
            "Lat": 'lat_header',
            "Lon": 'lon_header',
        }
    infra_loc_header = {}
    for each in infra_loc_header_map:
        if each in infra_loc_headers_in:
            infra_loc_header[infra_loc_header_map[each]] = infra_loc_headers_in[each]
    logging.info(f'{counter}. Processed setup configuration file')
    counter += 1
    
    # -----------------------------------------------------------
    # Intensity Measure
    im_source = list(setup_config['IntensityMeasure']['SourceForIM'])[0]
    if im_source == 'ShakeMap':
        sm_dir = os.path.join(
            # work_dir,
            input_dir,
            setup_config['IntensityMeasure']['SourceForIM']['ShakeMap']['Directory']
        )
        sm_events = setup_config['IntensityMeasure']['SourceForIM']['ShakeMap']['Events']
    elif im_source == 'UserDefinedRupture':
        rup_fpath = os.path.join(
            # work_dir,
            input_dir,
            setup_config['IntensityMeasure']['SourceForIM']['UserDefinedRupture']['FaultFile']
        )
    elif im_source == 'UCERF':
        rup_fpath = None
    else:
        raise NotImplementedError("To be added into preprocess...")
    # get filters if present
    if 'Filter' in setup_config['IntensityMeasure']['SourceForIM'][im_source]:
        im_filters = setup_config['IntensityMeasure']['SourceForIM'][im_source]['Filter']
    else:
        im_filters = {}
    logging.info(f'{counter}. Identified source for intensity measure: {im_source}')
    counter += 1
    
    # -----------------------------------------------------------
    # load json with available datasets, below-ground only for now
    avail_data_summary = None # initialize
    if infra_type == 'below_ground':
        avail_data_summary_fpath = os.path.join('lib','AvailableDataset.json')
        with open(avail_data_summary_fpath,'r') as f:
            avail_data_summary = json.load(f)
        logging.info(f'{counter}. Loaded JSON file with pre-packaged information of pre-packaged datasets (below-ground only)')
        counter += 1
    
    # -----------------------------------------------------------
    # preprocess infrastructure file
    preprocess_infra_file(
        infra_type, infra_fpath, infra_loc_header,
        processed_input_dir, flag_using_state_network, l_max=0.1,
    )
    logging.info(f'{counter}. Processed infrastructure file and exported site data table to directoy:')
    logging.info(f'\t{processed_input_dir}')
    counter += 1
    
    # -----------------------------------------------------------
    # get workflow for PC
    workflow, workflow_fpath = make_workflow(setup_config, processed_input_dir, to_export=True)
    logging.info(f'{counter}. Created workflow and exported to:')
    logging.info(f'\t{workflow_fpath}')
    logging.info(f'\n{json.dumps(workflow, indent=4)}\n')
    counter += 1
    
    # read input tables for random, fixed variables, and site data
    rvs_input, fixed_input, site_data, site_data_geom = \
        read_input_tables(input_dir, processed_input_dir, flag_using_state_network, infra_type, infra_geom_fpath)
    logging.info(f'{counter}. Read input tables for random, fixed variables, and infrastructure data in input directory')
    counter += 1   
    
    ##--------------------------
    # get crossings for below-ground infrastructure - may move to another location in Preprocess    
    if infra_type == 'below_ground':
        # landslide crossings
        cat = 'EDP'
        haz = 'landslide'
        if cat in workflow and haz in workflow[cat]:
            # get deformation polygon to use; if "statewide", then assign probability of 0.25 to all components instead
            landslide_meta = setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters']
            use_def_poly = landslide_meta['UseDeformationGeometry']
            if use_def_poly:
                def_poly_source = landslide_meta['SourceForDeformationGeometry']
                if def_poly_source == 'CA_LandslideInventory_WGS84':
                    fpath = avail_data_summary['Parameters']['ca_landslide_inventory']['Datasets']['Set1']['Path']
                else:
                    fdir = os.path.join(user_prov_gis_fdir,def_poly_source)
                    for f in os.listdir(fdir):
                        if f.endswith('.shp'):
                            fpath = os.path.join(fdir,f)
                            break
            else:
                fpath = None
            # site_data_crossing = get_pipe_crossing(
            # site_data_with_crossing_only = get_pipe_crossing(
            site_data = get_pipe_crossing(
                path_to_def_shp=fpath,
                infra_site_data=site_data.copy(),
                infra_site_data_geom=site_data_geom,
                opensra_dir=opensra_dir,
                export_dir=processed_input_dir,
                def_type=haz
            )
            logging.info(f'{counter}. Obtained pipeline crossing for landslide')
            counter += 1
    
    # -----------------------------------------------------------
    # rvs and fixed params split by preferred andf user provided
    pref_rvs, user_prov_table_rvs, user_prov_gis_rvs, \
    pref_fixed, user_prov_table_fixed, user_prov_gis_fixed = \
        separate_params_by_source(rvs_input, fixed_input)
    logging.info(f'{counter}. Separated random and fixed parameters by source')
    counter += 1
    
    # -----------------------------------------------------------
    # get preferred input distributions
    pref_param_dist, pref_param_dist_const_with_level, pref_param_fixed = \
        import_param_dist_table(opensra_dir, infra_type=infra_type)
    logging.info(f'{counter}. Read preferred distributions for variables')
    logging.info(f"\t{os.path.join('param_dist',f'{infra_type}.xlsx')}")
    counter += 1
    
    # -----------------------------------------------------------
    # get param_dist_meta from user-provided information
    if 'UserProvidedGISFolder' in setup_config['General']['Directory']:
        user_prov_gis_fdir = setup_config['General']['Directory']['UserProvidedGISFolder']
    else:
        user_prov_gis_fdir = ''
    param_dist_meta, param_dist_table = get_param_dist_from_user_prov_table(
        user_prov_table_rvs, user_prov_table_fixed,
        user_prov_gis_rvs, user_prov_gis_fixed,
        pref_rvs, pref_fixed, site_data, user_prov_gis_fdir
    )
    logging.info(f'{counter}. Retrieved user provided distributions from infrastructure table')
    counter += 1
    
    # -----------------------------------------------------------
    # get params with missing distribution metrics
    params_with_missing_dist_metric = get_params_with_missing_dist_metric(param_dist_meta)
    logging.info(f'{counter}. Determined parameters still with missing distribution metrics')
    counter += 1
    
    # -----------------------------------------------------------
    # get level to run
    if "EDP" in workflow and "Liquefaction" in workflow["EDP"] and "CPTBased" in workflow["EDP"]["Liquefaction"]:
        param_dist_table['level_to_run'] = np.ones(param_dist_table.shape[0])*3
    else:
        param_dist_table = get_level_to_run(
            param_dist_table,
            workflow,
            params_with_missing_dist_metric,
            param_dist_meta,
            setup_config,
            infra_type=infra_type
        )
    level_to_run = param_dist_table['level_to_run'][0]
    logging.info(f'{counter}. Determined level of analysis to run for each site')
    counter += 1

    # -----------------------------------------------------------
    # update to landslide params if level to run == 1
    if infra_type == 'below_ground':
        # landslide crossings
        cat = 'EDP'
        haz = 'landslide'
        if cat in workflow and haz in workflow[cat]:                
                # set to a generic value - will not be used at level 1.
                site_data.psi_dip = 15
                # export crossing summary table
                site_data.drop(columns='geometry').to_csv(
                    # os.path.join(export_dir,f'site_data_{def_type.upper()}_CROSSINGS_ONLY.csv'),
                    os.path.join(processed_input_dir,f'site_data_PROCESSED_CROSSING_ONLY.csv'),
                    index=False
                )
        logging.info(f'{counter}. Performed additional actions for landslide crossing')
        counter += 1
    
    # -----------------------------------------------------------
    # get rest of distribution metrics from preferred datasets
    param_dist_table, param_dist_meta, params_with_missing_dist_metric = get_pref_dist_for_params(
        params_with_missing_dist_metric,
        site_data,
        param_dist_table,
        param_dist_meta,
        pref_param_dist,
        pref_param_dist_const_with_level,
        pref_param_fixed,
        workflow,
        avail_data_summary,
        # site_data_with_crossing_only,
        export_path_dist_table=os.path.join(processed_input_dir,'param_dist.csv'),
        export_path_dist_json=os.path.join(processed_input_dir,'param_dist_meta.json'),
        infra_type=infra_type
    )
    logging.info(f'{counter}. Retrieved missing distribution metrics from preferred distributions')
    counter += 1
    
    # -----------------------------------------------------------
    # get IM predictions
    logging.info(f'\n')
    if im_source == "ShakeMap":
        get_im_pred(
            im_source, im_dir, site_data, infra_loc_header,
            # for ShakeMaps
            sm_dir=sm_dir,
            sm_events=sm_events,
        )
    elif im_source == "UserDefinedRupture" or im_source == 'UCERF':
        get_im_pred(
            im_source, im_dir, site_data, infra_loc_header, im_filters,
            # for user-defind ruptures
            opensra_dir=opensra_dir,
            processed_input_dir=processed_input_dir,
            rup_fpath=rup_fpath,
        )
    logging.info(f'\n')
    logging.info(f'{counter}. Obtained IM predictions from {im_source} and stored to:')
    logging.info(f"\t{im_dir}")
    counter += 1
    
    # -----------------------------------------------------------
    # get well and caprock crossings - may move to another location in Preprocess, but must be after getIM
    # well_crossing_ordered_by_faults = None           
    if infra_type == 'wells_caprocks':
        # get well crossings
        well_trace_dir = os.path.join(
            # work_dir,
            input_dir,
            setup_config['Infrastructure']['WellTraceDir']
        )
        # well_crossing_ordered_by_faults, _ = get_well_crossing(
        get_well_crossing(
            im_dir=im_dir,
            infra_site_data=site_data.copy(),
            col_with_well_trace_file_names='file_name',
            well_trace_dir=well_trace_dir,
        )
        logging.info(f'{counter}. Obtained well crossings for fault rupture')
        counter += 1
        
        # get caprock crossings
        if 'CaprockLeakage' in setup_config['DecisionVariable']['Type']:
            # get shapefile for caprock
            caprock_fdir = os.path.join(
                # work_dir,
                input_dir,
                setup_config['Infrastructure']['PathToCaprockShapefile']
            )
            for f in os.listdir(caprock_fdir):
                if f.endswith('.shp'):
                    caprock_shp_file = os.path.join(caprock_fdir,f)
                    break
            # project directory
            # rup_fpath = os.path.join(
            #     work_dir,
            #     rup_fpath
            # )
            # run caprock crossing algorith
            get_caprock_crossing(
                caprock_shp_file=caprock_shp_file,
                # rup_fpath=rup_fpath,
                im_dir=im_dir,
                processed_input_dir=processed_input_dir
            )
        logging.info(f'{counter}. Obtained caprock crossings for fault rupture')
        counter += 1
    
    # -----------------------------------------------------------
    # end of preprocess
    logging.info('... End of preprocessing for OpenSRA')


# -----------------------------------------------------------
def get_im_pred(
    im_source, im_dir, site_data, infra_loc_header, im_filters,
    # for ShakeMaps
    sm_dir=None, sm_events=None,
    # for user-defined ruptures
    opensra_dir=None, processed_input_dir=None, rup_fpath=None
):
    """get IM predictions from backend"""
    # initialize seismic hazard class
    seismic_hazard = getattr(haz, 'SeismicHazard')()
    
    # get IM predictions based on IM source
    if im_source == "ShakeMap":
        # set sites and site params
        if 'LON_MID' in site_data:
            seismic_hazard.set_site_data(
                # lon=site_data[infra_loc_header['lon_header']],
                # lat=site_data[infra_loc_header['lat_header']],
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values,
                vs30=np.zeros(site_data.shape[0])
            )
        elif 'LON' in site_data:
            seismic_hazard.set_site_data(
                # lon=site_data[infra_loc_header['lon_header']],
                # lat=site_data[infra_loc_header['lat_header']],
                lon=site_data.LON.values,
                lat=site_data.LAT.values,
                vs30=np.zeros(site_data.shape[0])
            )
        seismic_hazard.init_ssc(im_source,sm_dir=sm_dir,event_names=sm_events)  # initialize source
        
    elif im_source == 'UserDefinedRupture' or im_source == 'UCERF':
        # prepackaged site data
        gmc_site_data_dir = os.path.join(opensra_dir,'lib','OtherData','Preprocessed','Statewide_and_Regional_Grids')
        cols_to_get = ['vs30','vs30source','z1p0','z2p5']
        
        # create LocationData class to make use of nearest neighbor sampling schemes
        if 'LON_MID' in site_data:
            _infra = geodata.LocationData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values,
            )
        elif 'LON' in site_data:
            _infra = geodata.LocationData(
                lon=site_data.LON.values,
                lat=site_data.LAT.values,
            )
        # create component table, needed as a preprocessing phase
        _infra.make_component_table()
        
        # initialize temporary nearest_node_table as None
        temp_nearest_node_table = None
        # intiialize remaining_loc to track remaining locations
        remaining_loc = np.arange(_infra.data.shape[0])

        # go through each region and get site data from prepackaged files
        for region in ['BayArea','LosAngeles','Statewide']:
            # get spacing
            if 'Statewide' in region:
                spacing = 0.05
            elif 'LosAngeles' in region:
                spacing = 0.01
            elif 'BayArea' in region:
                spacing = 0.01
            # generate path to boundary file
            spacing_string = str(spacing).replace('.','p')
            bound_path = os.path.join(opensra_dir,'lib','OtherData','Boundaries',region,'Boundary.shp')
            region_dir = os.path.join(gmc_site_data_dir,region)
            gmc_site_data_path = os.path.join(region_dir,f'{spacing_string}_nodes_withSiteParam.csv')

            # make copy of _infra to track locations within each region
            _infra_curr_bound = copy.deepcopy(_infra)
            # reduce data table with locations that have already been accounted for in previous boundary
            _infra_curr_bound.data = _infra_curr_bound.data.loc[remaining_loc].reset_index(drop=True)
            # clip data with boundary
            _infra_curr_bound.clip_loc_with_bound(bound_fpath=bound_path)
            
            # if there are locations within the boundary, then proceed to get data
            if len(_infra_curr_bound.loc_in_bound) > 0:
                # get grid nodes from prepackaged site data files
                _grid_nodes = geodata.LocationData(fpath=gmc_site_data_path)
                _infra_curr_bound.get_grid_nodes(_grid_nodes.data)
                # find nearest grid and get values
                _infra_curr_bound.get_nearest_grid_nodes()
                # break
                # store results back to original_infra structure 
                if temp_nearest_node_table is None:
                    temp_nearest_node_table = _infra_curr_bound.nearest_node_table.copy()
                else:
                    temp_nearest_node_table = pd.concat(
                        [
                            temp_nearest_node_table,
                            _infra_curr_bound.nearest_node_table.copy()
                        ], ignore_index=True
                    )
            
            # track remaining locs that are outside current boundary
            remaining_loc = np.asarray(list(set(remaining_loc).difference(set(remaining_loc[_infra_curr_bound.loc_in_bound]))))
            # break out of boundary loop if no more locations remaining
            if len(remaining_loc) == 0:
                break
        
        # set _infra.nearest_node_table to temp_nearest_node_table
        _infra.nearest_node_table = temp_nearest_node_table.copy()
        # once the nearest_node_table has been populated, then sites from nearest node
        _infra.component_table = _infra.sample_csv(
            table=_infra.component_table.copy(),
            fpath=gmc_site_data_path,
            # cols_to_get=cols_to_get,
            use_hull=True
        )
        
        # export sampled basin params
        _infra.export_component_table(
            sdir=im_dir,
            sname='site_data_basin_params',
            to_replace=True
        )
        
        # pull site basin params to use as inputs to GMPEs
        col_with_data = {
            'lon': 'LON',
            'lat': 'LAT',
            'vs30': 'vs30',
            'z1p0': 'z1p0',
            'z2p5': 'z2p5',
            'vs30_source': 'vs30source'
        }
        site_kwargs = {
            key: _infra.component_table[col_with_data[key]].values for key in col_with_data
        }
        # set sites and initialize source
        seismic_hazard.set_site_data(**site_kwargs) # set site and site params
        if im_source == 'UserDefinedRupture':
            seismic_hazard.init_ssc(im_source, user_def_rup_fpath=rup_fpath)  # initialize source
        elif im_source == 'UCERF':
            seismic_hazard.init_ssc(im_source, opensra_dir=opensra_dir)  # initialize source

    else:
        raise NotImplementedError("to be added to preprocessing...")
    
    # default filters
    max_dist = 200
    rate_min = None
    mag_min = None
    mag_max = None
    # check for available filters
    if 'Distance' in im_filters:
        if im_filters['Distance']['ToInclude']:
            if 'Maximum' in im_filters['Distance']:
                max_dist = im_filters['Distance']['Maximum']
    if 'MeanAnnualRate' in im_filters:
        if im_filters['MeanAnnualRate']['ToInclude']:
            if 'Minimum' in im_filters['MeanAnnualRate']:
                rate_min = im_filters['MeanAnnualRate']['Minimum']
    if 'Magnitude' in im_filters:
        if im_filters['Magnitude']['ToInclude']:
            if 'Minimum' in im_filters['Magnitude']:
                mag_min = im_filters['Magnitude']['Minimum']
            if 'Maximum' in im_filters['Magnitude']:
                mag_max = im_filters['Magnitude']['Maximum']
    
    # rest of hazard calc
    seismic_hazard.init_gmpe() # initialize GMPE, even though not used for ShakeMaps
    seismic_hazard.process_rupture(
        max_dist=max_dist,
        mag_min=mag_min,
        mag_max=mag_max,
        rate_min=rate_min
    ) # process ruptures
    seismic_hazard.get_gm_pred_from_gmc() # get GM predictions
    seismic_hazard.export_gm_pred(sdir=im_dir) # export GM predictions


# -----------------------------------------------------------
def preprocess_infra_file(
    infra_type, infra_fpath, infra_loc_header,
    processed_input_dir, flag_using_state_network, l_max=0.1,
):
    """process infrastructure files"""
    
    # load infrastructure file
    if infra_type == 'below_ground':
        # if using state network, copy the preprocessed network into the "Processed_Input" folder to reduce processing effort
        if flag_using_state_network:
            # copy preprocessed infrastrcuture file
            shutil.copy(
                src=infra_fpath,
                dst=os.path.join(processed_input_dir,'site_data_PROCESSED.csv')
            )
        else:
            if infra_fpath.endswith('shp'):
                infra = geodata.NetworkData(fpath=infra_fpath)
            elif infra_fpath.endswith('csv'):
                infra = geodata.NetworkData(
                    fpath=infra_fpath,
                    lon_header=infra_loc_header["lon_header"],
                    lat_header=infra_loc_header["lat_header"],
                    lon_begin_header=infra_loc_header["lon_begin_header"],
                    lat_begin_header=infra_loc_header["lat_begin_header"],
                    lon_end_header=infra_loc_header["lon_end_header"],
                    lat_end_header=infra_loc_header["lat_end_header"],
                )
            else:
                raise ValueError('Only suports "shp" or "csv" as input file type')
            # network
            # l_max = 0.1 # km
            infra.split_network_by_max_length(l_max) # l_max in km
            infra.make_segment_table()
            infra.export_segment_table(
                # sdir=input_dir,
                # sname='site_data',
                # to_replace=False
                sdir=processed_input_dir,
                sname='site_data_PROCESSED',
                to_replace=True
            )
    else:
        if infra_fpath.endswith('shp'):
            infra = geodata.LocationData(fpath=infra_fpath)
        elif infra_fpath.endswith('csv'):
            infra = geodata.LocationData(
                fpath=infra_fpath,
                lon_header=infra_loc_header["lon_header"],
                lat_header=infra_loc_header["lat_header"],
            )
        else:
            raise ValueError('Only suports "shp" or "csv" as input file type')
        # process components/sites
        infra.make_component_table()
        infra.export_component_table(
            # sdir=input_dir,
            # sname='site_data',
            # to_replace=False
            sdir=processed_input_dir,
            sname='site_data_PROCESSED',
            to_replace=True
        )


# -----------------------------------------------------------
def make_workflow(setup_config, processed_input_dir, to_export=True):
    """makes workflow to be used for PC"""
    # now make workflow
    workflow = {}
    cat_map = {
        'EDP': 'EngineeringDemandParameter',
        'DM': 'DamageMeasure',
        'DV': 'DecisionVariable',
    }
    haz_map = {
        # below ground
        'liquefaction': "Liquefaction",
        'lateral_spread': "LateralSpread",
        'settlement': "Settlement",
        'landslide': "Landslide",
        'surface_fault_rupture': "SurfaceFaultRupture",
        # 'pipe_strain': "PipeStrain",
        'pipe_strain_settlement': "SettlementInducedPipeStrain",
        'pipe_strain_landslide': "LandslideInducedPipeStrain",
        'pipe_strain_lateral_spread': "LateralSpreadInducedPipeStrain",
        'pipe_comp_rupture': "PipeCompressiveRupture",
        'pipe_tensile_rupture': "PipeTensileRupture",
        'pipe_tensile_leakage': "PipeTensileLeakage",
        
        # above ground
        'wellhead_rotation': "WellheadRotation",
        'wellhead_strain': "WellheadStrain",
        'wellhead_rupture': "WellheadRupture",
        'wellhead_leakage': "WellheadLeakage",
        'vessel_moment_ratio': "VesselMomentRatio",
        'vessel_rupture': "VesselRupture",
        
        # wells and caprocks
        'surface_fault_rupture': "SurfaceFaultRupture",
        'well_strain': "WellStrain",
        'well_moment': "WellMoment",
        'well_rupture_shear': "ShearInducedWellRupture",
        'well_rupture_shaking': "ShakingInducedWellRupture",
        'caprock_leakage': "CaprockLeakage",
        
    }
    for cat in cat_map:
        workflow[cat] = {}
        haz_list = setup_config[cat_map[cat]]['Type']
        for haz in haz_map:
            if haz_map[haz] in haz_list and haz_list[haz_map[haz]]['ToInclude']:
                workflow[cat][haz] = {}
                method_list = haz_list[haz_map[haz]]['Method']
                # for getting model weights
                for method in method_list:
                    workflow[cat][haz][method] = {
                        'ModelWeight': method_list[method]['ModelWeight']
                    }
    # export workflow
    if to_export:
        workflow_fpath = os.path.join(processed_input_dir,'workflow.json')
        with open(workflow_fpath, 'w') as f:
            json.dump(workflow, f, indent=4)
    #
    return workflow, workflow_fpath


# -----------------------------------------------------------
# def get_rvs_and_fix_by_level(rv_input, fix_input, workflow, infra_fixed={}):
def get_rvs_and_fix_by_level(workflow, infra_fixed={}):
    """gets all the required RVs and fixed variables sorted by level"""
    
    all_rvs = [] # list of all RVs needed for selected methods
    req_rvs_by_level = {} # RVs needed by levels
    req_fixed_by_level = {} # fixed params needed by levels

    # for each category
    for cat in workflow:
        curr_cat = workflow[cat]
        # for each hazard
        for haz_type in curr_cat:
            curr_haz_type = curr_cat[haz_type]
            # if CPTBased, hard code in params for now:
            if "CPTBased" in curr_cat[haz_type]:
                # all_rvs += ['gw_depth', 'slope']
                all_rvs += []
            else:
                # load python file
                _file = importlib.import_module('.'.join(['src',cat.lower(),haz_type.lower()]))
                # for each method
                for method in curr_haz_type:
                    # create instance
                    _inst = copy.deepcopy(getattr(_file, method)())
                    # get all RVs for method
                    all_rvs += _inst._missing_inputs_rvs
                    # print(cat, haz_type, method)
                    # print(_inst._missing_inputs_rvs)
                    rvs_by_level, fix_by_level = _inst.get_req_rv_and_fix_params(infra_fixed)
                    # print(cat)
                    # print(haz_type)
                    # print(method)
                    # print(rvs_by_level)
                    # print(fix_by_level)
                    # gather required model parameters for each level
                    # print(rvs_by_level)
                    # print(fix_by_level)
                    for i in range(3):
                        # initialize list
                        if not f'level{i+1}' in req_rvs_by_level:
                            req_rvs_by_level[f'level{i+1}'] = rvs_by_level[f'level{i+1}']
                            req_fixed_by_level[f'level{i+1}'] = fix_by_level[f'level{i+1}']
                        else:
                            req_rvs_by_level[f'level{i+1}'] += rvs_by_level[f'level{i+1}']
                            req_fixed_by_level[f'level{i+1}'] += fix_by_level[f'level{i+1}']
                            
                            # req_rvs_by_level[f'level{i+1}'] = []
                            # req_fixed_by_level[f'level{i+1}'] = []
                        # if method varies with level
                        # if _inst.input_dist_vary_with_level:
                            # req_rvs_by_level[f'level{i+1}'] += rvs_by_level[f'level{i+1}']
                            # req_fixed_by_level[f'level{i+1}'] += fix_by_level[f'level{i+1}']
                        # else:
                        #     req_rvs_by_level[f'level{i+1}'] += rvs_by_level
                        #     req_fixed_by_level[f'level{i+1}'] += fix_by_level

    # get unique required params for each level
    for i in range(3):
        req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
        req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))

    # sort all rvs
    all_rvs = sorted(list(set(all_rvs)))
    
    # return
    return all_rvs, req_rvs_by_level, req_fixed_by_level


def import_param_dist_table(opensra_dir, infra_type='below_ground'):
    """loads table with param distributions, choose from 'below_ground', 'above_ground', and 'wells_caprocks'"""
    n_levels = 3
    # Append the opensra dir to make it a relative path
    pref_param_dist_path = os.path.join(opensra_dir, 'param_dist',f'{infra_type}.xlsx')
    pref_param_dist = {}
    # by levels
    for i in range(n_levels):
        curr_level = f'level{i+1}'
        curr_sheet = pd.read_excel(
            pref_param_dist_path,
            sheet_name=curr_level
        )
        if i == 0:
            pref_param_dist_const_with_level = curr_sheet[curr_sheet.vary_with_level==False].copy().reset_index(drop=True)
        pref_param_dist[curr_level] = curr_sheet[curr_sheet.vary_with_level==True].copy().reset_index(drop=True)
    # fixed
    pref_param_fixed = pd.read_excel(
        pref_param_dist_path,
        sheet_name='fixed'
    )
    return pref_param_dist, pref_param_dist_const_with_level, pref_param_fixed


# -----------------------------------------------------------
def read_input_tables(
    input_dir, processed_input_dir, flag_using_state_network,
    infra_type, infra_geom_fpath=None
):
    """read input tables"""
    rvs_input = pd.read_csv(os.path.join(input_dir,'rvs_input.csv'))
    fixed_input = pd.read_csv(os.path.join(input_dir,'fixed_input.csv'))
    # site_data = pd.read_csv(os.path.join(input_dir,'site_data.csv'))
    site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED.csv'))
    # convert "Yes" and "No" in site_data to "True/False
    site_data.replace({
        "Yes": True,
        "yes": True,
        "No": False,
        "no": False
    }, inplace=True)
    # if using preprocessed state pipeline network, update column to pull for diameter
    if flag_using_state_network:
        row_for_d_pipe = np.where(rvs_input.Name=='d_pipe')[0][0]
        if rvs_input.loc[row_for_d_pipe,'Source'] == 'Preferred':
            rvs_input.loc[row_for_d_pipe,'Source'] = 'From infrastructure table or enter value'
            rvs_input.loc[row_for_d_pipe,'Mean or Median'] = 'DIAMETER'
            rvs_input.loc[row_for_d_pipe,'Distribution Type'] = 'Normal'
    # preload infrastructure geometry if it exists, otherwise create it
    if infra_geom_fpath is not None:
        site_data_geom = gpd.read_file(infra_geom_fpath).geometry
    else:
        if infra_type == 'below_ground':
            site_data_geom = gpd.GeoDataFrame(
                None,
                crs=4326,
                geometry=[
                    LineString([
                        (site_data['LON_BEGIN'][i], site_data['LAT_BEGIN'][i]),
                        (site_data['LON_END'][i], site_data['LAT_END'][i])
                    ]) for i in range(site_data.shape[0])
                ]
            ).geometry
        else:
            site_data_geom = gpd.GeoDataFrame(
                None,
                crs=4326,
                geometry=[
                    Point((site_data['LON'][i], site_data['LAT'][i]))
                    for i in range(site_data.shape[0])
                ]
            ).geometry
            
        site_data_geom = None
    return rvs_input, fixed_input, site_data, site_data_geom


# -----------------------------------------------------------
def separate_params_by_source(rvs_input, fixed_input):
    """separate preferred vs user provided callouts in input tables"""
    pref_rvs = rvs_input[rvs_input.Source=='Preferred'].reset_index(drop=True).copy()
    user_prov_table_rvs = rvs_input[rvs_input.Source=='From infrastructure table or enter value'].reset_index(drop=True).copy()
    user_prov_gis_rvs = rvs_input[rvs_input.Source=='From user-provided GIS maps'].reset_index(drop=True).copy()
    pref_fixed = fixed_input[fixed_input.Source=='Preferred'].reset_index(drop=True).copy()
    user_prov_table_fixed = fixed_input[fixed_input.Source=='From infrastructure table or enter value'].reset_index(drop=True).copy()
    user_prov_gis_fixed = fixed_input[fixed_input.Source=='From user-provided GIS maps'].reset_index(drop=True).copy()
    return pref_rvs, user_prov_table_rvs, user_prov_gis_rvs, pref_fixed, user_prov_table_fixed, user_prov_gis_fixed


# -----------------------------------------------------------
def get_param_dist_from_user_prov_gis(
    # user_prov_gis_map_dir,
    # user_prov_gis_rvs,
    # user_prov_gis_fixed,
    # param_dist_meta,
    # site_data
):
    """gets inputs for parameters flagged as 'From user-provided GIS maps'"""
    logging.info(NotImplementedError("to be implemented"))
    return param_dist_meta, site_data


# -----------------------------------------------------------
def get_param_dist_from_user_prov_table(
    user_prov_table_rvs,
    user_prov_table_fixed,
    user_prov_gis_rvs,
    user_prov_gis_fixed,
    pref_rvs,
    pref_fixed,
    site_data,
    user_prov_gis_fdir,
    param_dist_meta={},
):
    """gets inputs for parameters flagged as 'From infrastructure table'"""
    # number of sites
    n_site = site_data.shape[0]
    
    # map names for distribution metrics
    metric_map = {
        'dist_type': 'Distribution Type',
        'mean': 'Mean or Median',
        'sigma': 'Sigma',
        'cov': 'CoV',
        'low': 'Distribution Min',
        'high': 'Distribution Max',
    }
    # dataframe for storing parameter distributions
    param_dist_table = pd.DataFrame(None)
    
    # first loop through user provided random params
    for i in range(user_prov_table_rvs.shape[0]):
        param = user_prov_table_rvs.Name[i]
        curr_param_dist = {
            'source': user_prov_table_rvs.Source[i]
        }
        # distribution type
        if isinstance(user_prov_table_rvs.loc[i,metric_map['dist_type']],float) and \
            np.isnan(user_prov_table_rvs.loc[i,metric_map['dist_type']]):
            curr_param_dist['dist_type'] = 'normal'
        else:
            curr_param_dist['dist_type'] = user_prov_table_rvs.loc[i,metric_map['dist_type']].lower()
        # mean
        try: # try converting to float, if can't then it's a column name
            curr_param_dist['mean'] = float(user_prov_table_rvs.loc[i,metric_map['mean']])
        except ValueError: # read from site data table
            curr_param_dist['mean'] = site_data[user_prov_table_rvs.loc[i,metric_map['mean']].upper()].values
        # apply ln, assuming values are given as medians
        if curr_param_dist['dist_type'] == 'lognormal':
            curr_param_dist['mean'] = np.log(curr_param_dist['mean'])
        # sigma/cov
        if np.isnan(user_prov_table_rvs.loc[i,metric_map['sigma']]) and \
            np.isnan(user_prov_table_rvs.loc[i,metric_map['cov']]):
            curr_param_dist['sigma'] = np.zeros(n_site)
        else:
            # check if sigma is given
            if not np.isnan(user_prov_table_rvs.loc[i,metric_map['sigma']]):
                try: # try converting to float, if can't then it's a column name
                    curr_param_dist['sigma'] = float(user_prov_table_rvs.loc[i,metric_map['sigma']])
                except ValueError:
                    curr_param_dist['sigma'] = site_data[user_prov_table_rvs.loc[i,metric_map['sigma']]].values
            # check if cov is given
            elif not np.isnan(user_prov_table_rvs.loc[i,metric_map['cov']]):
                try: # try converting to float, if can't then it's a column name
                    curr_cov = float(user_prov_table_rvs.loc[i,metric_map['cov']])
                except ValueError:
                    curr_cov = site_data[user_prov_table_rvs.loc[i,metric_map['cov']]].values
                # calculate sigma from cov
                if curr_param_dist['dist_type'] == 'normal':
                    curr_param_dist['sigma'] = curr_cov/100 * curr_param_dist['mean']
                elif curr_param_dist['dist_type'] == 'lognormal':
                    curr_param_dist['sigma'] = np.log(1+curr_cov/100)
        # low and high
        for each in ['low','high']:
            if np.isnan(user_prov_table_rvs.loc[i,metric_map[each]]):
                if each == 'low':
                    curr_param_dist[each] = -np.inf
                elif each == 'high':
                    curr_param_dist[each] = np.inf
            else:
                curr_param_dist[each] = float(user_prov_table_rvs.loc[i,metric_map[each]])
                if curr_param_dist['dist_type'] == 'lognormal':
                    if each == 'low' and curr_param_dist[each] == 0:
                        curr_param_dist[each] = -np.inf
                    else:
                        curr_param_dist[each] = np.log(curr_param_dist[each])
        # repeat for number of sites if dimension is 0, also check if still contains NaN
        curr_param_dist['still_need_pref'] = {}
        for each in ['mean','sigma','low','high']:
            # repeat for number of sites if dimension is 0
            if np.ndim(curr_param_dist[each]) == 0:
                curr_param_dist[each] = np.ones(n_site) * curr_param_dist[each]
            # convert to dtype to float
            curr_param_dist[each] = curr_param_dist[each].astype(float)
            # see if still contains np.isnan
            curr_param_dist['still_need_pref'][each] = False
            ind = np.where(np.isnan(curr_param_dist[each]))[0]
            if len(ind) > 0:
                curr_param_dist['still_need_pref'][each] = True
        # store to param_dist_meta and site_data
        for met in ['mean','sigma','low','high','dist_type']:
            col_name = f'{param}_{met}'
            param_dist_table[col_name] = curr_param_dist[met]
            curr_param_dist[met] = col_name
        param_dist_meta[param] = curr_param_dist.copy()
        
    # see if there are any random parameters where the values are from user defined GIS files
    if user_prov_gis_rvs.shape[0] > 0:
        # get coordinates
        if 'LON_MID' in site_data:
            locs = geodata.LocationData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values
            )
        elif 'LON' in site_data:
            locs = geodata.LocationData(
                lon=site_data.LON.values,
                lat=site_data.LAT.values
            )
        else:
            raise ValueError("Cannot locate lon/lat")
    # first loop through user provided GIS data
    for i in range(user_prov_gis_rvs.shape[0]):
        param = user_prov_gis_rvs.Name[i]
        curr_param_dist = {
            'source': user_prov_gis_rvs.Source[i]
        }
        # distribution type
        if isinstance(user_prov_gis_rvs.loc[i,metric_map['dist_type']],float) and \
            np.isnan(user_prov_gis_rvs.loc[i,metric_map['dist_type']]):
            curr_param_dist['dist_type'] = 'normal'
        else:
            curr_param_dist['dist_type'] = user_prov_gis_rvs.loc[i,metric_map['dist_type']].lower()
        # mean
        # currently can sample Shapefiles or raster
        # get full file path for GIS map
        curr_user_prov_gis_fdir = os.path.join(user_prov_gis_fdir,user_prov_gis_rvs['Mean or Median'][i])
        # find files under fdir
        files = os.listdir(curr_user_prov_gis_fdir)
        # look for either ".tif" for raster of ".shp" for Shapefile
        curr_user_prov_gis_fpath = None
        for f in files:
            if f.endswith('.tif') or f.endswith('.shp'):
                if f.endswith('.tif'):
                    gis_type = 'raster'
                else:
                    gis_type = 'shapefile'
                curr_user_prov_gis_fpath = os.path.join(curr_user_prov_gis_fdir,f)
                break
        if curr_user_prov_gis_fpath is None:
            logging.info('Cannot locate user provided GIS file: file must end with ".tif" or ".shp"')
        else:
            # with file path, now sample
            if gis_type == 'raster':
                # get sample from GIS file
                locs.data = locs.sample_raster(
                    table=locs.data,
                    fpath=curr_user_prov_gis_fpath,
                    store_name=param
                )
            elif gis_type == 'shapefile':
                locs.data = locs.sample_shapefile(
                    table=locs.data,
                    fpath=geo_unit_fpath,
                    attr=None,
                    store_name=param
                )
            # check for lognormal and apply correction
            if curr_param_dist['dist_type'] == 'lognormal':
                curr_param_dist['mean'] = np.log(locs.data[param].values)
            else:
                curr_param_dist['mean'] = locs.data[param].values
        # sigma/cov
        if np.isnan(user_prov_gis_rvs.loc[i,metric_map['sigma']]) and \
            np.isnan(user_prov_gis_rvs.loc[i,metric_map['cov']]):
            curr_param_dist['sigma'] = np.zeros(n_site)
        else:
            # check if sigma is given
            if not np.isnan(user_prov_gis_rvs.loc[i,metric_map['sigma']]):
                try: # try converting to float, if can't then it's a column name
                    curr_param_dist['sigma'] = float(user_prov_gis_rvs.loc[i,metric_map['sigma']])
                except ValueError:
                    curr_param_dist['sigma'] = site_data[user_prov_gis_rvs.loc[i,metric_map['sigma']]].values
            # check if cov is given
            elif not np.isnan(user_prov_gis_rvs.loc[i,metric_map['cov']]):
                try: # try converting to float, if can't then it's a column name
                    curr_cov = float(user_prov_gis_rvs.loc[i,metric_map['cov']])
                except ValueError:
                    curr_cov = site_data[user_prov_gis_rvs.loc[i,metric_map['cov']]].values
                # calculate sigma from cov
                if curr_param_dist['dist_type'] == 'normal':
                    curr_param_dist['sigma'] = curr_cov/100 * curr_param_dist['mean']
                elif curr_param_dist['dist_type'] == 'lognormal':
                    curr_param_dist['sigma'] = np.log(1+curr_cov/100)
        # low and high
        for each in ['low','high']:
            if np.isnan(user_prov_gis_rvs.loc[i,metric_map[each]]):
                if each == 'low':
                    curr_param_dist[each] = -np.inf
                elif each == 'high':
                    curr_param_dist[each] = np.inf
            else:
                curr_param_dist[each] = float(user_prov_gis_rvs.loc[i,metric_map[each]])
                if curr_param_dist['dist_type'] == 'lognormal':
                    if each == 'low' and curr_param_dist[each] == 0:
                        curr_param_dist[each] = -np.inf
                    else:
                        curr_param_dist[each] = np.log(curr_param_dist[each])
        # repeat for number of sites if dimension is 0, also check if still contains NaN
        curr_param_dist['still_need_pref'] = {}
        for each in ['mean','sigma','low','high']:
            # repeat for number of sites if dimension is 0
            if np.ndim(curr_param_dist[each]) == 0:
                curr_param_dist[each] = np.ones(n_site) * curr_param_dist[each]
            # convert to dtype to float
            curr_param_dist[each] = curr_param_dist[each].astype(float)
            # see if still contains np.isnan
            curr_param_dist['still_need_pref'][each] = False
            ind = np.where(np.isnan(curr_param_dist[each]))[0]
            if len(ind) > 0:
                curr_param_dist['still_need_pref'][each] = True
        # store to param_dist_meta and site_data
        for met in ['mean','sigma','low','high','dist_type']:
            col_name = f'{param}_{met}'
            param_dist_table[col_name] = curr_param_dist[met]
            curr_param_dist[met] = col_name
        param_dist_meta[param] = curr_param_dist.copy()

    # next loop through user provided fixed params
    for i in range(user_prov_table_fixed.shape[0]):
        param = user_prov_table_fixed.Name[i]
        curr_param_dist = {
            'source': user_prov_table_fixed.Source[i]
        }
        try: # try converting to float, if can't then it's a column name
            curr_param_dist['value'] = float(user_prov_table_fixed.loc[i,'Value'])
        except ValueError:
            # if is column name, then pull values from column, else use as value
            if user_prov_table_fixed.loc[i,'Value'].upper() in site_data.columns:
                curr_param_dist['value'] = site_data[user_prov_table_fixed.loc[i,'Value'].upper()].values
            else:
                curr_param_dist['value'] = [user_prov_table_fixed.loc[i,'Value'].lower()] * site_data.shape[0]
        curr_param_dist['dist_type'] = 'fixed'
        # some specific processing
        if param == 'soil_type':
            # make sure entries are lower case if possible
            curr_param_dist['value'] = np.array([
                x.lower() if isinstance(x, str) else x
                 for x in curr_param_dist['value']
            ])
        # check if still contains NaN
        curr_param_dist['still_need_pref'] = {}
        count = sum([1 for j,x in enumerate(curr_param_dist['value']) if isinstance(x, float) and np.isnan(x)])
        if count > 0:
            curr_param_dist['still_need_pref']['value'] = True
        # store to param_dist_meta and site_data
        param_dist_table[param] = curr_param_dist['value']
        curr_param_dist['value'] = param
        param_dist_meta[param] = curr_param_dist.copy()
    
    # see if there are any fixed parameters where the values are from user defined GIS files
    if user_prov_gis_fixed.shape[0] > 0 and not locs in locals():
        # get coordinates
        if 'LON_MID' in site_data:
            locs = geodata.LocationData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values
            )
        elif 'LON' in site_data:
            locs = geodata.LocationData(
                lon=site_data.LON.values,
                lat=site_data.LAT.values
            )
        else:
            raise ValueError("Cannot locate lon/lat")
    # next loop through user provided fixed params
    for i in range(user_prov_gis_fixed.shape[0]):
        param = user_prov_gis_fixed.Name[i]
        curr_param_dist = {
            'source': user_prov_gis_fixed.Source[i]
        }
        # value
        # currently can sample Shapefiles or raster
        # get full file path for GIS map
        curr_user_prov_gis_fdir = os.path.join(user_prov_gis_fdir,user_prov_gis_fixed['Value'][i])
        # find files under fdir
        files = os.listdir(curr_user_prov_gis_fdir)
        # look for either ".tif" for raster of ".shp" for Shapefile
        curr_user_prov_gis_fpath = None
        for f in files:
            if f.endswith('.tif') or f.endswith('.shp'):
                if f.endswith('.tif'):
                    gis_type = 'raster'
                else:
                    gis_type = 'shapefile'
                curr_user_prov_gis_fpath = os.path.join(curr_user_prov_gis_fdir,f)
                break
        if curr_user_prov_gis_fpath is None:
            logging.info('Cannot locate user provided GIS file: file must end with ".tif" or ".shp"')
        else:
            # with file path, now sample
            if gis_type == 'raster':
                # get sample from GIS file
                locs.data = locs.sample_raster(
                    table=locs.data,
                    fpath=curr_user_prov_gis_fpath,
                    store_name=param
                )
            elif gis_type == 'shapefile':
                locs.data = locs.sample_shapefile(
                    table=locs.data,
                    fpath=geo_unit_fpath,
                    attr=None,
                    store_name=param
                )
            # store to param_dist
            curr_param_dist['value'] = locs.data[param].values
        # dist type
        curr_param_dist['dist_type'] = 'fixed'
        # some specific processing
        if param == 'soil_type':
            # make sure entries are lower case if possible
            curr_param_dist['value'] = np.array([
                x.lower() if isinstance(x, str) else x
                 for x in curr_param_dist['value']
            ])
        # check if still contains NaN
        curr_param_dist['still_need_pref'] = {}
        count = sum([1 for j,x in enumerate(curr_param_dist['value']) if isinstance(x, float) and np.isnan(x)])
        if count > 0:
            curr_param_dist['still_need_pref']['value'] = True
        # store to param_dist_meta and site_data
        param_dist_table[param] = curr_param_dist['value']
        curr_param_dist['value'] = param
        param_dist_meta[param] = curr_param_dist.copy()
        
    # next loop through preferred random params
    for i in range(pref_rvs.shape[0]):
        param = pref_rvs.Name[i]
        curr_param_dist = {
            'source': pref_rvs.Source[i]
        }
        curr_param_dist['still_need_pref'] = {}
        # distribution type
        curr_param_dist['dist_type'] = None
        curr_param_dist['still_need_pref']['dist_type'] = True
        # mean and sigma
        for each in ['mean','sigma']:
            curr_param_dist[each] = np.ones(n_site) * np.nan
            curr_param_dist['still_need_pref'][each] = True
        # low and high
        curr_param_dist['low'] = np.ones(n_site) * np.nan
        curr_param_dist['still_need_pref']['low'] = True
        curr_param_dist['high'] = np.ones(n_site) * np.nan
        curr_param_dist['still_need_pref']['high'] = True
        # store to param_dist_meta and site_data
        for met in ['mean','sigma','low','high','dist_type']:
            col_name = f'{param}_{met}'
            param_dist_table[col_name] = curr_param_dist[met]
            curr_param_dist[met] = col_name
        param_dist_meta[param] = curr_param_dist.copy()
    
    # next loop through preferred fixed params
    for i in range(pref_fixed.shape[0]):
        param = pref_fixed.Name[i]
        curr_param_dist = {
            'source': pref_fixed.Source[i]
        }
        curr_param_dist['still_need_pref'] = {}
        # distribution type
        curr_param_dist['dist_type'] = 'fixed'
        # mean
        curr_param_dist['value'] = np.ones(n_site) * np.nan
        curr_param_dist['still_need_pref']['value'] = True
        # store to param_dist_meta and site_data
        param_dist_table[param] = curr_param_dist['value']
        curr_param_dist['value'] = param
        param_dist_meta[param] = curr_param_dist.copy()
    
    # return
    return param_dist_meta, param_dist_table


# -----------------------------------------------------------
def get_params_with_missing_dist_metric(param_dist_meta):
    # see which parameters are still missing dist metrics
    params_with_missing_dist_metric = {}
    for each in param_dist_meta:
        for met in param_dist_meta[each]['still_need_pref']:
            if param_dist_meta[each]['still_need_pref'][met]:
                if each in params_with_missing_dist_metric:
                    params_with_missing_dist_metric[each] += [met]
                else:
                    params_with_missing_dist_metric[each] = [met]
    return params_with_missing_dist_metric


# -----------------------------------------------------------
def get_level_to_run(
    param_dist_table,
    workflow,
    params_with_missing_dist_metric,
    param_dist_meta,
    setup_config,
    infra_type='below_ground'
):
    """determin level of analysis to run"""
    
    # number of sites
    n_site = param_dist_table.shape[0]
    
    # get param that are fixed
    param_fixed = []
    for param in param_dist_meta:
        if param_dist_meta[param]['dist_type'] == 'fixed':
            param_fixed.append(param)
    # make infra_fixed
    infra_fixed = {
        key: param_dist_table[key].values for key in param_fixed
    }
    # infra_fixed = {
        # key: param_dist_table[key].values for key in ['soil_type','steel_grade'] if key in param_dist_table.columns
    # }
    
    # determine RVs needed by level
    all_rvs, req_rvs_by_level, req_fixed_by_level = get_rvs_and_fix_by_level(workflow, infra_fixed)
    
    # print('\n')
    # print(all_rvs)
    # print('\n')
    # print(req_rvs_by_level)
    # print('\n')
    # print(req_fixed_by_level)
    # print('\n')
    
    # print(list(params_with_missing_dist_metric))
    
    # sys.exit()
    
    # for each site, determine level to run
    level_to_run = np.ones(n_site).astype(int)*3
    # loop through levels
    for i in range(3,1,-1):
        # parameters required for current level
        params_for_curr_level = req_rvs_by_level[f'level{i}']
        ind_for_cur_level = np.array([])
        for param in params_with_missing_dist_metric:
            if param in params_for_curr_level:
                for met in params_with_missing_dist_metric[param]:
                    ind_for_cur_level = np.hstack([
                        ind_for_cur_level,
                        # np.where(np.isnan(param_dist_meta[param][met]))[0]
                        np.where(param_dist_table[param+'_'+met].isnull())[0]
                    ])
        ind_for_cur_level = np.unique(ind_for_cur_level).astype(int)
        level_to_run[ind_for_cur_level] -= 1
        
    # for landslide and liquefaction, limit max level to run based on availability of deformation polygon
    # landslide
    cat = 'EDP'
    haz = 'landslide'
    min_level_with_no_def_poly = 1
    if cat in workflow and haz in workflow[cat]:        
        # get deformation polygon to use; if "statewide", then assign probability of 0.25 to all components instead
        landslide_meta = setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters']
        use_def_poly = landslide_meta['UseDeformationGeometry']
        if use_def_poly is False:
            level_to_run = min(min(level_to_run),min_level_with_no_def_poly) # level if no deformation polygon is to be used
    
    # store levels to run
    param_dist_table['level_to_run'] = level_to_run
    
    # get list of sites under each level
    # site_index_by_levels = {
    #     f'level{i+1}': np.where(param_dist_table.level_to_run==i+1)[0]
    #     for i in range(3)
    # }
    
    # return
    # return param_dist_table, site_index_by_levels
    return param_dist_table


# -----------------------------------------------------------
def get_pref_dist_for_params(
    params_with_missing_dist_metric,
    site_data,
    param_dist_table,
    param_dist_meta,
    pref_param_dist,
    pref_param_dist_const_with_level,
    pref_param_fixed,
    workflow,
    avail_data_summary,
    # site_data_with_crossing_only=None,
    export_path_dist_table=None,
    export_path_dist_json=None,
    infra_type='below_ground',
    default_statewide_geo_map='wills'
):
    """get rest of the missing distribution metrics"""
    
    # initialize
    met_list = ['dist_type','mean','sigma','low','high']
    crossing_params = ['l_anchor','beta_crossing','psi_dip']
    # crossing_params = ['l_anchor','beta_crossing','psi_dip','theta_rake']
    soil_prop_map = {}
    
    # first load geologic units from various geologic maps
    if infra_type == 'below_ground':
        
        # get coordinates
        if 'LON_MID' in site_data:
            locs = geodata.LocationData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values
            )
        elif 'LON' in site_data:
            locs = geodata.LocationData(
                lon=site_data.LON.values,
                lat=site_data.LAT.values
            )
        else:
            raise ValueError("Cannot locate lon/lat")
        
        # CGS geologic unit, may take this out
        if 'EDP' in workflow and 'landslide' in workflow['EDP']:
            if ('phi_soil' in param_dist_meta and param_dist_meta['phi_soil']['source'] == 'Preferred') or \
               ('coh_soil' in param_dist_meta and param_dist_meta['coh_soil']['source'] == 'Preferred'):
                if default_statewide_geo_map == 'wills':
                    file_key = 'level1_geo_unit_wills15'
                    store_name = avail_data_summary['Parameters'][file_key]['ColumnNameToStoreAs']
                    # print(1)
                    geo_unit_fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
                    locs.data = locs.sample_shapefile(
                        table=locs.data,
                        fpath=geo_unit_fpath,
                        attr='Geologic_U',
                        store_name=store_name,
                        missing_val='water'
                    )
                    # print(2)
                    param_dist_table[store_name] = locs.data[store_name].values
                    # print(locs.data[store_name].values)
                    # print(locs.data[store_name].values.dtype)
                    param_dist_table[store_name] = param_dist_table[store_name].astype('<U20')
                    # print(locs.data[store_name].values.dtype)
                    # load strength params from Bain et al. (2022)
                    default_geo_prop_fpath = avail_data_summary['Parameters']['phi_soil']['Datasets']['Set1']['Path']
                    default_geo_prop = pd.read_csv(default_geo_prop_fpath)
                    unique_geo_unit = np.unique(param_dist_table[store_name])
                    geo_unit_desc = np.empty_like(param_dist_table[store_name])
                    for each in unique_geo_unit:
                        geo_unit_desc[param_dist_table[store_name].values==each] = \
                            default_geo_prop['Unit Abbreviation'][default_geo_prop['Unit Abbreviation'].values==each]
                            # default_geo_prop['Unit Description'][default_geo_prop['Unit Abbreviation'].values==each]
                    param_dist_table['wills_geo_unit_desc'] = geo_unit_desc
                    if 'phi_soil' in param_dist_meta and param_dist_meta['phi_soil']['source'] == 'Preferred':
                        soil_prop_map['phi_soil'] = {
                            'mean': 'Friction Angle - Median (degrees)',
                            'cov': 'Friction Angle - CoV (%)',
                            'low': 'Friction Angle - Min (degrees)',
                            'high': 'Friction Angle - Max (degrees)',
                            'dist_type': 'lognormal'
                        }
                    if 'coh_soil' in param_dist_meta and param_dist_meta['coh_soil']['source'] == 'Preferred':
                        soil_prop_map['coh_soil'] = {
                            'mean': 'Cohesion - Median (kPa)',
                            'cov': 'Cohesion - CoV (%)',
                            'low': 'Cohesion - Min (kPa)',
                            'high': 'Cohesion - Max (kPa)',
                            'dist_type': 'lognormal'
                        }
                    
                # elif default_statewide_geo_map == 'cgs':
                #     file_key = 'level1_geo_unit_cgs10'
                #     store_name = avail_data_summary['Parameters'][file_key]['ColumnNameToStoreAs']
                #     geo_unit_fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
                #     locs.data = locs.sample_shapefile(
                #         table=locs.data,
                #         fpath=geo_unit_fpath,
                #         attr='PTYPE',
                #         store_name=store_name
                #     )
                #     param_dist_table[store_name] = locs.data[store_name].values
                #     print(f'\tRead CGS geologic units')
                
                #     # load Slate's site data table with soil properties based on geologic units, for temporary use
                #     slate_geo_prop_fpath = avail_data_summary['Parameters']['phi_soil']['Datasets']['Set1']['Path']
                #     slate_geo_prop = pd.read_csv(slate_geo_prop_fpath)
                #     unique_geo_unit = np.unique(param_dist_table[store_name])
                #     geo_unit_desc = np.empty_like(param_dist_table[store_name])
                #     for each in unique_geo_unit:
                #         geo_unit_desc[param_dist_table[store_name].values==each] = \
                #             slate_geo_prop['Unit Description CGS'][slate_geo_prop['Unit Abbreviation'].values==each]
                #     param_dist_table['cgs_geo_unit_desc'] = geo_unit_desc
                #     soil_prop_map = {
                #         'phi_soil': 'Friction Angle (degrees)',
                #         'coh_soil': 'Cohesion (kPa)',
                #         'gamma_soil': 'Unit Weight (kN/m3)',
                #     }
            
        # if liq susc is needed, then sample regional geologic units
        if 'liq_susc' in param_dist_meta and param_dist_meta['liq_susc']['source'] == 'Preferred':
            # Bedrossian et al. (2012)
            file_key = 'level2_geo_unit_witter06'
            store_name = avail_data_summary['Parameters'][file_key]['ColumnNameToStoreAs']
            geo_unit_fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
            locs.data = locs.sample_shapefile(
                table=locs.data,
                fpath=geo_unit_fpath,
                attr='PTYPE',
                store_name=store_name
            )
            param_dist_table[store_name] = locs.data[store_name].values
            witter_store_name = store_name
            logging.info(f'\tRead Witter et al. (2006) geologic units')
            
            # Witter et al. (2006)
            file_key = 'level2_geo_unit_bedrossian12'
            store_name = avail_data_summary['Parameters'][file_key]['ColumnNameToStoreAs']
            geo_unit_fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
            locs.data = locs.sample_shapefile(
                table=locs.data,
                fpath=geo_unit_fpath,
                attr='PTYPE',
                store_name=store_name
            )
            param_dist_table[store_name] = locs.data[store_name].values
            bedrossian_store_name = store_name
            logging.info(f'\tRead Bedrossian et al. (2012) geologic units')
            # drop liq susc from param dist table, to sample during run
            param_dist_table.drop(columns=['liq_susc'],inplace=True)
        
        # Merge Bedrossian et al. and Witter et al. into one column called "Regional_Geologic_Unit"
        # param_dist_table['Regional_Geologic_Unit'] = None
        # param_dist_table.loc[param_dist_table[witter_store_name].notna(),'Regional_Geologic_Unit'] = \
            # param_dist_table[witter_store_name][param_dist_table[witter_store_name].notna()].values
        # param_dist_table.loc[param_dist_table[bedrossian_store_name].notna(),'Regional_Geologic_Unit'] = \
            # param_dist_table[bedrossian_store_name][param_dist_table[bedrossian_store_name].notna()].values
        # print(f'\tMerged results from Witter et al. (2006) and Bedrossian et al. (2012) into column called "Regional_Geologic_Unit"')
    
    # first go through fixed params with missing values
    param_list = list(params_with_missing_dist_metric)
    for param in param_list:
        if param in pref_param_fixed.rv_label.values:
            row_for_param = np.where(pref_param_fixed.rv_label==param)[0][0]
            if pref_param_fixed['preferred exists?'][row_for_param]:
                # find which sites are np.nan and update values
                rows_nan = np.where(param_dist_table[param].isnull())[0]
                param_dist_table.loc[rows_nan,param] = pref_param_fixed.value[row_for_param]
                # remove param from missing param list
                params_with_missing_dist_metric.pop(param,None)
    
    # loop through rest of params with missing distribution metrics
    param_list = list(params_with_missing_dist_metric) # remaining params
    for param in param_list:
        # see if preferred distribution for parameter varies with levels
        if param in list(pref_param_dist['level1'].rv_label):
            # loop through levels
            for level in np.unique(param_dist_table.level_to_run.values):
                # find rows with current level
                rows_for_level = np.where(param_dist_table.level_to_run==level)[0]
                # row for param in preferred distribution table
                row_for_param = np.where(pref_param_dist[f'level{level}'].rv_label==param)[0][0]
                # loop through missing metrics
                for met in met_list:
                    if met in params_with_missing_dist_metric[param]:
                        # find rows where null
                        rows_nan = np.where(param_dist_table[f'{param}_{met}'].isnull())[0]
                        # intersection between rows_nan and rows_for_level
                        rows_comb = list(set(rows_nan).intersection(set(rows_for_level)))
                        # get preferred value
                        pref_val = pref_param_dist[f'level{level}'][met][row_for_param]
                        # specifically for sigma, check CoV
                        if met == 'sigma':
                            if np.isnan(pref_val):
                                # if nan, get from cov
                                pref_val = pref_param_dist[f'level{level}']['cov'][row_for_param]
                                # update distribution metric
                                if param_dist_table[f'{param}_dist_type'][0] == 'normal':
                                    param_dist_table.loc[rows_comb,f'{param}_sigma'] = \
                                        pref_val/100 * param_dist_table.loc[rows_comb,f'{param}_mean']
                                elif param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    param_dist_table.loc[rows_comb,f'{param}_sigma'] = \
                                        np.log(1+pref_val/100) * param_dist_table.loc[rows_comb,f'{param}_mean']
                            else:
                                # update distribution metric
                                param_dist_table.loc[rows_comb,f'{param}_sigma'] = pref_val
                        else:
                            # check for lognormal and apply correction
                            if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                pref_val = np.log(pref_val)
                            if met == 'low' or met == 'high':
                                if np.isnan(pref_val):
                                    if met == 'low':
                                        pref_val = -np.inf
                                    elif met == 'high':
                                        pref_val = np.inf
                            # update distribution metric
                            param_dist_table.loc[rows_comb,f'{param}_{met}'] = pref_val
                        
            # remove param from missing param list
            params_with_missing_dist_metric.pop(param,None)

        # param does not vary with level
        elif param in list(pref_param_dist_const_with_level.rv_label):
            # row for param in preferred distribution table
            row_for_param = np.where(pref_param_dist_const_with_level.rv_label==param)[0][0]
            
            # specific properties for landslide, use Wills et al. geo properties developed by Chris Bain
            if pref_param_dist_const_with_level['mean'][row_for_param] == 'depends' and \
                param in soil_prop_map:
                # for each unique geologic unit
                for each in unique_geo_unit:
                    # get rows with geo unit
                    rows_with_geo_unit = np.where(param_dist_table['wills_geo_unit_desc'].values==each)[0]
                    # intersection of geo unit and NaN
                    rows_comb = list(set(rows_nan).intersection(set(rows_with_geo_unit)))
                    # get preferred value from Chris Bain's table
                    rows_for_param = np.where(default_geo_prop['Unit Abbreviation'].values==each)[0][0]                    # get dist metrics
                    # dist type
                    param_dist_table.loc[rows_comb,f'{param}_dist_type'] = 'lognormal'
                    # mean
                    pref_val = default_geo_prop[soil_prop_map[param]['mean']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_mean'] = np.log(pref_val)
                    # sigma
                    pref_val = default_geo_prop[soil_prop_map[param]['cov']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_sigma'] = \
                        np.log(1+pref_val/100) * param_dist_table.loc[rows_comb,f'{param}_mean'].values
                    # low
                    pref_val = default_geo_prop[soil_prop_map[param]['low']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_low'] = np.log(pref_val)
                    # high
                    pref_val = default_geo_prop[soil_prop_map[param]['high']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_high'] = np.log(pref_val)
                    
                    # # get preferred value from Slate's table
                    # rows_for_param = np.where(slate_geo_prop['Unit Abbreviation'].values==each)[0][0]
                    # pref_val = default_geo_prop[soil_prop_map[param]][rows_for_param]
                    # # check for lognormal and apply correction
                    # if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                    #     pref_val = np.log(pref_val)
                    # # update value
                    # param_dist_table.loc[rows_comb,f'{param}_mean'] = pref_val

            # all other cases
            else:
                # loop through missing metrics
                for met in met_list:
                    if met in params_with_missing_dist_metric[param]:
                        # find rows where null
                        rows_nan = np.where(param_dist_table[f'{param}_{met}'].isnull())[0]
                        # get preferred value
                        pref_val = pref_param_dist_const_with_level[met][row_for_param]
                        # specifically for mean
                        if met == 'mean':
                            # get crossing from site datatable
                            if pref_val == 'depends':
                                # for pipe crossing parameters
                                if param in crossing_params:
                                    # pass
                                    # if param == 'l_anchor':
                                    #     print(site_data.columns)
                                    if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                        param_dist_table.loc[rows_nan,f'{param}_mean'] = np.log(site_data[param][rows_nan].values)
                                    else:
                                        param_dist_table.loc[rows_nan,f'{param}_mean'] = site_data[param][rows_nan].values
                                    
                                # for other cases that are assigned as "depends"
                                # leave value as NaN (likely imposed to be determined later)
                                else:
                                    # param_dist_table.loc[rows_nan,f'{param}_mean'] = np.nan
                                    param_dist_table.loc[rows_nan,f'{param}_mean'] = "event_dependent"
                            # using internal GIS maps
                            elif pref_val == 'internal gis dataset':
                                # path for GIS file
                                gis_fpath = avail_data_summary['Parameters'][param]['Datasets']['Set1']['Path']
                                locs.data = locs.sample_raster(
                                    table=locs.data,
                                    fpath=gis_fpath,
                                    store_name=param
                                )
                                # check for lognormal and apply correction
                                if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    pref_val = np.log(locs.data[param].values)
                                else:
                                    pref_val = locs.data[param].values
                                # pref_val = locs.data[param].values
                                param_dist_table.loc[rows_nan,f'{param}_mean'] = pref_val[rows_nan]
                            else:
                                # check for lognormal and apply correction
                                if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    pref_val = np.log(pref_val)
                                param_dist_table.loc[rows_nan,f'{param}_mean'] = pref_val
                        # specifically for sigma, check CoV
                        elif met == 'sigma':
                            if np.isnan(pref_val):
                                # if nan, get from cov
                                pref_val = pref_param_dist_const_with_level['cov'][row_for_param]
                                # update distribution metric
                                if param_dist_table[f'{param}_dist_type'][0] == 'normal':
                                    param_dist_table.loc[rows_nan,f'{param}_sigma'] = \
                                        pref_val/100 * param_dist_table.loc[rows_nan,f'{param}_mean'].values
                                elif param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    param_dist_table.loc[rows_nan,f'{param}_sigma'] = \
                                        np.log(1+pref_val/100) * param_dist_table.loc[rows_nan,f'{param}_mean'].values
                            else:
                                param_dist_table.loc[rows_nan,f'{param}_sigma'] = pref_val
                        else:
                            # for specific cases where it says "depends"
                            if pref_val == 'depends':
                                if param == 'dist_coast' or param == 'dist_river' or param == 'dist_water':
                                    if met == 'low':
                                        # check for lognormal and apply correction
                                        if param_dist_table[f'{param}_dist_type'][0] == 'normal':
                                            pref_val = np.maximum(param_dist_table[f'{param}_mean'].values - 50, 0) # limit to 0
                                        elif param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                            pref_val = np.log(np.maximum(np.exp(param_dist_table[f'{param}_mean'].values) - 50, 0)) # limit to 0
                                        # pref_val = np.maximum(param_dist_table[f'{param}_mean'].values - 50, 0) # limit to 0
                                        param_dist_table.loc[rows_nan,f'{param}_{met}'] = pref_val[rows_nan]
                                    elif met == 'high':
                                        # check for lognormal and apply correction
                                        if param_dist_table[f'{param}_dist_type'][0] == 'normal':
                                            pref_val = param_dist_table[f'{param}_mean'].values + 50 # 50 km over mean
                                        elif param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                            pref_val = param_dist_table[f'{param}_mean'] + 50 # 50 km over mean
                                        # pref_val = param_dist_table[f'{param}_mean'].values + 50 # 50 km over mean
                                        param_dist_table.loc[rows_nan,f'{param}_{met}'] = \
                                            param_dist_table.loc[rows_nan,f'{param}_mean'] + 50 # 50 km over mean
                                        param_dist_table.loc[rows_nan,f'{param}_{met}'] = pref_val[rows_nan]
                            else:
                                # for beta_crossing specifically at level 1
                                if param == 'beta_crossing' and param_dist_table['level_to_run'][0] == 1:
                                    if met == 'low':
                                        pref_val = 90 # limit low to 90
                                # check for lognormal and apply correction
                                if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    pref_val = np.log(pref_val)
                                if met == 'low' or met == 'high':
                                    if np.isnan(pref_val):
                                        if met == 'low':
                                            pref_val = -np.inf
                                        elif met == 'high':
                                            pref_val = np.inf
                                param_dist_table.loc[rows_nan,f'{param}_{met}'] = pref_val
            # remove param from missing param list
            params_with_missing_dist_metric.pop(param,None)          
    
    # additional backend limitations
    param_list = list(param_dist_meta) # all params
    for param in param_list:
        if param == 'n_param' or param == 'r_param':
            if met == 'low':
                param_dist_table[f'{param}_low'] = \
                    np.maximum(
                        param_dist_table[f'{param}_mean'] - \
                        2*param_dist_table[f'{param}_sigma']
                    , 0) # limit to 0
                param_dist_meta[param]['still_need_pref']['low'] = True
            if met == 'high':
                param_dist_table[f'{param}_low'] = \
                    param_dist_table[f'{param}_mean'] + \
                    2*param_dist_table[f'{param}_sigma']
                param_dist_meta[param]['still_need_pref']['high'] = True
    
    # export table
    if export_path_dist_table is not None:
        param_dist_table.to_csv(export_path_dist_table,index=False)
        # also export to hdf5 for access
        param_dist_table.to_hdf(export_path_dist_table.replace('.csv','.h5'),key='table',mode='w')
        # also export to txt
        # param_dist_table.to_csv(export_path_dist_table.replace('.csv','.txt'),index=False,sep='\t')
    # export dictionary
    if export_path_dist_json is not None:
        with open(export_path_dist_json, 'w') as f:
            json.dump(param_dist_meta, f, indent=4, sort_keys=True)
    
    # return
    return param_dist_table, param_dist_meta, params_with_missing_dist_metric


# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Preprocess methods for OpenSRA'
    )
    
    # Define arguments
    
    # analysis directory
    parser.add_argument('-w', '--work_dir', help='Path to working/analysis directory')
    
    # json with workflow
    # parser.add_argument('-w', '--workflow', help='Path to json file with workflow')
    
    # infrastructure type
    # parser.add_argument('-i', '--infra_type', help='Infrastructure type',
    #                     default='below_ground', type=str)
    
    # infrastructure file type
    # parser.add_argument('-f', '--file_path', help='Infrastructure file path',
    #                     default='shp', type=str)
    
    # infrastructure file type
    parser.add_argument('-l', '--logging_detail',
                        help='Logging message detail: "simple" (default) or "full"',
                        default='simple', type=str)
    
    # Parse command line input
    args = parser.parse_args()
    
    # Run "Main"
    main(
        work_dir = args.work_dir,
        # infra_fpath = args.file_type,
        logging_message_detail=args.logging_detail
    )
