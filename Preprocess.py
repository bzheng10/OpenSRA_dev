# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Preprocessing to risk calculations
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python base modules
import argparse
import copy
import glob
import importlib
import json
import logging
import os
import shutil
import sys
import time
import warnings
import zipfile
warnings.simplefilter(action='ignore', category=FutureWarning)

# scientific processing modules
import numpy as np
import pandas as pd
from scipy import sparse
# suppress warning that may come up
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# for geospatial processing
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from shapely.geometry import Point, LineString
from shapely.errors import ShapelyDeprecationWarning
from pyproj import Transformer
# suppress warning that may come up
warnings.simplefilter(action='ignore', category=ShapelyDeprecationWarning)

# OpenSRA modules
from src.edp import process_cpt_spt
from src.im import haz
from src.pc_func.pc_workflow import get_samples_for_params
from src.site import geodata
from src.site.get_pipe_crossing import get_pipe_crossing_landslide_and_liq, get_pipe_crossing_fault_rup
from src.site.get_well_crossing import get_well_crossing
from src.site.get_caprock_crossing import get_caprock_crossing
from src.site.site_util import make_list_of_linestrings, make_grid_nodes
from src.util import set_logging, check_and_get_abspath, get_shp_file_from_dir

# -----------------------------------------------------------
# Main function
def main(work_dir, logging_level='info', logging_message_detail='s',
         display_after_n_event=100, clean_prev_output=True):
    """main function that runs the preprocess procedures"""
    
    # -----------------------------------------------------------
    # Setting logging level (e.g. DEBUG or INFO)
    set_logging(
        level=logging_level,
        msg_format=logging_message_detail
    )
    
    # -----------------------------------------------------------
    # start of preprocess
    logging.info('---------------')
    logging.info('******** Start of preprocessing for OpenSRA ********')
    logging.info('---------------')
    config = {} # dictionary to store configuration params
    counter = 1 # counter for stages of processing   
    
    # -----------------------------------------------------------
    # get paths and make directories
    # check current directory, if not at OpenSRA level, go up a level (happens during testing)
    if not os.path.basename(os.getcwd()) == 'OpenSRA' and \
        not os.path.basename(os.getcwd()) == 'OpenSRABackEnd':
        os.chdir('..')
    # get paths to directories
    opensra_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(work_dir,'Input')
    processed_input_dir = os.path.join(work_dir,'Processed_Input')
    im_dir = os.path.join(work_dir,'IM')
    # clean prev outputs
    if clean_prev_output:
        if os.path.isdir(processed_input_dir):
            shutil.rmtree(processed_input_dir) # remove directory/start clean
        if os.path.isdir(im_dir):
            shutil.rmtree(im_dir) # remove directory/start clean
        os.mkdir(processed_input_dir)
        os.mkdir(im_dir)
    else:
        if not os.path.isdir(processed_input_dir):
            os.mkdir(processed_input_dir)
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
    logging.info(f'{counter}. Check and create file directories')
    counter += 1
    logging.info('\tPerforming preprocessing of methods and input variables for OpenSRA')
    logging.info('\t\tOpenSRA backend directory')
    logging.info(f'\t\t\t- {opensra_dir}')
    logging.info('\t\tWorking directory given:')
    logging.info(f'\t\t\t- {work_dir}')
    logging.info('\t\tInput directory implied:')
    logging.info(f'\t\t\t- {input_dir}')
    logging.info('\t\tProcessed input directory for export of processed information:')
    logging.info(f'\t\t\t- {processed_input_dir}')
    logging.info('\t\tIntensity measure directory:')
    logging.info(f'\t\t\t- {im_dir}')
    
    # ---------------------
    # first decompress precomputed statewide IMs (only done once)
    preprocess_lib_dir = os.path.join(opensra_dir,'lib','OtherData','Preprocessed')
    preprocess_im_dir = os.path.join(preprocess_lib_dir,'Precomputed_IMs_for_Statewide_Pipeline')
    zip_fpath = os.path.join(preprocess_lib_dir,'Precomputed_IMs_for_Statewide_Pipeline.zip')
    extracted_state_data = False
    # see if folder exists
    if os.path.exists(preprocess_im_dir):
        # if empty
        if len(os.listdir(preprocess_im_dir)) == 0:
            # unzip to directory
            with zipfile.ZipFile(zip_fpath,"r") as zip_ref:
                zip_ref.extractall(preprocess_im_dir)
            extracted_state_data = True
    else:
        # make folder
        os.mkdir(preprocess_im_dir)
        # unzip to directory
        with zipfile.ZipFile(zip_fpath,"r") as zip_ref:
            zip_ref.extractall(preprocess_im_dir)
        extracted_state_data = True
    if extracted_state_data:
        logging.info(f'{counter}. Extracted preprocessed IMs for state network (only performed once during first run after installation)')
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
    flag_using_state_network = False
    flag_using_state_full_set = False
    flag_using_region_network = False
    if infra_ftype == 'State_Network':
        # use internal preprocessed CSV file for state pipeline network
        if 'SUBSET' in infra_fname:
            infra_fpath = os.path.join(
                'CA_Natural_Gas_Pipeline_Segments_WGS84_SUBSET',
                'CA_Natural_Gas_Pipeline_Segments_WGS84_Under100m_SUBSET.csv'
            )
        else:
            infra_fpath = os.path.join(
                'CA_Natural_Gas_Pipeline_Segments_WGS84',
                'CA_Natural_Gas_Pipeline_Segments_WGS84_Under100m.csv'
            )
            flag_using_state_full_set = True
        infra_fpath = os.path.join(
            opensra_dir,
            'lib','OtherData','Preprocessed',
            infra_fpath
        )
        infra_geom_fpath = infra_fpath.replace('.csv','.gpkg')
        flag_using_state_network = True
    elif infra_ftype == 'Region_Network':
        # use internal preprocessed CSV file for state pipeline network
        if 'Bay_Area' in infra_fname:
            infra_fpath = os.path.join(
                'Bay_Area_Pipeline_Network_Clipped_From_Statewide',
                'Bay_Area_Pipeline_Network_Clipped_From_Statewide.gpkg'
            )
        elif 'Los_Angeles' in infra_fname:
            infra_fpath = os.path.join(
                'Los_Angeles_Pipeline_Network_Clipped_From_Statewide',
                'Los_Angeles_Pipeline_Network_Clipped_From_Statewide.gpkg'
            )
        infra_fpath = os.path.join(
            opensra_dir,
            'lib','OtherData','Preprocessed',
            infra_fpath
        )
        infra_fpath = os.path.abspath(infra_fpath) # get absolute path to avoid cross-platform pathing errors
        infra_fname = infra_fpath# update fname with fpath
        # update infra_ftype to shapefile for internal processing
        infra_ftype = "Shapefile"
        flag_using_region_network = True
    else:
        # check if infra_fname is already a valid filepath, if not then infer from input_dir
        infra_fpath = check_and_get_abspath(infra_fname, input_dir)
        # further search within fpath for shapefile
        if infra_ftype == 'Shapefile':
            infra_fpath = get_shp_file_from_dir(infra_fpath)
    if 'SiteLocationParams' in setup_config['Infrastructure']:
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
    else:
        infra_loc_header = {
            "lat_header": "LAT_MID",
            "lon_header": "LON_MID",
            "lat_begin_header": "LAT_BEGIN",
            "lon_begin_header": "LON_BEGIN",
            "lat_end_header": "LAT_END",
            "lon_end_header": "LON_END"
        }
    # user provided GIS folder
    user_prov_gis_dir = ''
    if 'UserSpecifiedData' in setup_config:
        if 'GISDatasets' in setup_config['UserSpecifiedData']:
            gis_data_params = setup_config['UserSpecifiedData']['GISDatasets']
            if 'Directory' in gis_data_params:
                user_prov_gis_dir = check_and_get_abspath(gis_data_params['Directory'], input_dir)
    logging.info(f'{counter}. Processed setup configuration file')
    counter += 1
    
    # -----------------------------------------------------------
    # Intensity Measure
    im_source = list(setup_config['IntensityMeasure']['SourceForIM'])[0]
    sm_dir = None
    sm_events = None
    rup_fpath = None
    if im_source == 'ShakeMap':
        sm_dir = setup_config['IntensityMeasure']['SourceForIM']['ShakeMap']['Directory']
        # check if user-input sm_dir is valid directory, if not then infer from input_dir
        sm_dir = check_and_get_abspath(sm_dir, input_dir)
        sm_events = setup_config['IntensityMeasure']['SourceForIM']['ShakeMap']['Events']
    elif im_source == 'UserDefinedRupture':
        rup_fpath = setup_config['IntensityMeasure']['SourceForIM']['UserDefinedRupture']['FaultFile']
        # check if user-input sm_dir is valid directory, if not then infer from input_dir
        rup_fpath = check_and_get_abspath(rup_fpath, input_dir)
    elif im_source == 'UCERF':
        pass
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
    opensra_dataset_dir = os.path.join(opensra_dir,'lib','Datasets')
    if infra_type == 'below_ground':
        # default path for development env
        if 'OpenSRAData' in setup_config['General']['Directory']:
            if len(setup_config['General']['Directory']['OpenSRAData']) > 0:
                opensra_dataset_dir = setup_config['General']['Directory']['OpenSRAData']
        # check if dataset dir is valid path, if not raise error
        if not os.path.exists(opensra_dataset_dir):
            raise ValueError(f"Path to OpenSRA pre-package datasets does not exist: {opensra_dataset_dir}")
        avail_data_summary_fpath = os.path.join(opensra_dataset_dir,'AvailableDataset.json')
        with open(avail_data_summary_fpath,'r') as f:
            avail_data_summary = json.load(f)
        logging.info(f'{counter}. Loaded JSON file with information of prepackaged datasets (below-ground only)')
        counter += 1
    
    # -----------------------------------------------------------
    # preprocess infrastructure file
    logging.info(f'{counter}. Processing infrastructure file...')
    counter += 1
    preprocess_infra_file(
        infra_type, infra_fpath, infra_loc_header,
        processed_input_dir, flag_using_state_network, l_max=0.1,
    )
    logging.info(f'... DONE - Processed infrastructure file and exported site data table to directoy:')
    logging.info(f'\t{processed_input_dir}')
    
    # -----------------------------------------------------------
    # get workflow for PC
    workflow, workflow_fpath = make_workflow(setup_config, input_dir, processed_input_dir, to_export=True)
    logging.info(f'{counter}. Created workflow and exported to:')
    logging.info(f'\t{workflow_fpath}')
    logging.info(f'\n{json.dumps(workflow, indent=4)}\n')
    counter += 1
    
    # -----------------------------------------------------------
    # import preferred input distributions
    pref_param_dist, pref_param_fixed = \
        import_param_dist_table(opensra_dir, infra_type=infra_type)
    logging.info(f'{counter}. Loaded backend tables with preferred variable distributions/values from the following spreadsheet')
    logging.info(f"\t{os.path.join('param_dist',f'{infra_type}.xlsx')}")
    counter += 1
    
    # -----------------------------------------------------------
    # read input tables for random, fixed variables, and site data
    rvs_input, fixed_input, site_data, site_data_geom = \
        read_input_tables(
            input_dir, processed_input_dir,
            flag_using_state_network, flag_using_region_network,
            infra_type, infra_geom_fpath
    )
    if site_data.shape[0] == 0:
        logging.info(f'\n')
        logging.info(f'*****FATAL*****')
        logging.info(f'- The number of components/segments is zero!')
        logging.info(f'- Preprocessing will now exit as the final risk metrics will all be zero.')
        logging.info(f'- Please revise the input infrastruture file and try preprocessing again.')
        # logging.info(f'- Preprocessing will now exit.')
        logging.info(f'*****FATAL*****')
        logging.info(f'\n')
        # sys.exit()
        raise ValueError("FATAL: The number of components/segments is zero!")
    else:
        logging.info(f'{counter}. Loaded input tables for random, fixed variables, and infrastructure data in input directory')
        counter += 1
    
    ##--------------------------
    # get crossings for below-ground infrastructure - may move to another location in Preprocess
    running_cpt_based_procedure = False
    event_ids_to_keep = None
    event_inds_to_keep = None
    rupture_table_from_crossing = None
    performed_crossing = False
    col_headers_to_append = []
    if infra_type == 'below_ground':
        # pipe crossings crossings
        if 'EDP' in workflow:
            # preprocessing for fault rupture crossings
            if 'surface_fault_rupture' in workflow['EDP']:
                hazard = 'surface_fault_rupture'
                # check for validity of fault crossing inputs
                if not im_source == 'UCERF':
                    raise ValueError('Surface fault rupture currently set up for QFault hazard zones from LCI, which interacts with UCERF3')
                # additional logic inputs
                fault_disp_model = list(workflow['EDP']['surface_fault_rupture'])[0]
                reduced_ucerf_fpath = os.path.join(opensra_dir,
                    'lib','UCERF3','ReducedEvents_Abrahamson2022','Mean UCERF3 FM3.1',
                    'UCERF3_reduced_senario_dM0.5_v2.shp'
                )
                # run get pipe crossing function
                logging.info(f'{counter}. Performing pipeline crossing algorithm for {hazard}...')
                counter += 1
                # site_data = 
                site_data, rupture_table_from_crossing, col_headers_to_append, event_ids_to_keep = \
                    get_pipe_crossing_fault_rup(
                        processed_input_dir=processed_input_dir,
                        im_dir=im_dir,
                        infra_site_data=site_data.copy(),
                        avail_data_summary=avail_data_summary,
                        opensra_dataset_dir=opensra_dataset_dir,
                        reduced_ucerf_fpath=reduced_ucerf_fpath,
                        fault_disp_model=fault_disp_model,
                        im_source=im_source,
                        infra_site_data_geom=site_data_geom,
                    )
                performed_crossing = True
                logging.info(f'... DONE - Obtained pipeline crossing for {hazard}')
                
            # liq and landslide share the same function for pipe crossing
            else:
                # preprocessing for liquefaction crossings
                if 'liquefaction' in workflow['EDP']:
                    # get geohazard
                    if 'lateral_spread' in workflow['EDP']:
                        hazard = 'lateral_spread'
                    elif 'settlement' in workflow['EDP']:
                        hazard = 'settlement'
                    # see if CPT based for method, if so, need to run CPT preprocessing
                    if 'CPTBased' in workflow['EDP']['liquefaction']:
                        running_cpt_based_procedure = True
                        # pass info into function to read and process CPTs and generated deformation polygons
                        # logging.info('\n---------------------------')
                        logging.info(f'{counter}. Running CPT preprocessing script to generate deformation polygons...')
                        counter += 1
                        spath_def_poly, freeface_fpath = preprocess_cpt_data(
                            # predetermined setup configuration parameters
                            setup_config, opensra_dir, im_dir, processed_input_dir, input_dir,
                            rvs_input, fixed_input, workflow,
                            # OpenSRA internal files
                            avail_data_summary, opensra_dataset_dir,
                            # for all IM sources
                            im_source, im_filters,
                            # for ShakeMaps
                            sm_dir=sm_dir, sm_events=sm_events,
                            # for user-defined and UCERF ruptures
                            rup_fpath=rup_fpath,
                            # misc.
                            display_after_n_event=display_after_n_event
                        )
                        logging.info('... DONE - Processed CPTs')
                        # logging.info('---------------------------\n')
                        spath_def_poly = spath_def_poly[0]
                        def_shp_crs = 4326
                    else:
                        # if not using CPT based methods, then assign probability of crossing of 0.25 to all components
                        spath_def_poly = None
                        def_shp_crs = None
                        freeface_fpath = None
                # preprocessing for landslide crossings
                if 'landslide' in workflow['EDP']:
                    hazard = 'landslide'
                    # get deformation polygon to use; if "statewide", then assign probability of crossing of 0.25 to all components
                    landslide_setup_meta = setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters']
                    use_def_poly = landslide_setup_meta['UseDeformationGeometry']
                    if use_def_poly:
                        def_poly_source = landslide_setup_meta['SourceForDeformationGeometry']
                        if def_poly_source == 'CA_LandslideInventory_WGS84':
                            file_metadata = avail_data_summary['Parameters']['ca_landslide_inventory']
                            def_shp_crs = file_metadata['Datasets']['Set1']['CRS']
                            spath_def_poly = os.path.join(opensra_dataset_dir,file_metadata['Datasets']['Set1']['Path'])
                        else:
                            # check if def_poly_source is a valid path
                            fdir = check_and_get_abspath(def_poly_source, user_prov_gis_dir)
                            # next check if fdir is a folder with shapefile or is already a shapefile
                            spath_def_poly = get_shp_file_from_dir(fdir)
                            # if fpath is still None, 
                            if spath_def_poly is None:
                                raise ValueError("A path was provided for user defined landslide deformation polygons, but path is invalid")
                            def_shp_crs = 4326
                    else:
                        spath_def_poly = None
                        def_shp_crs = None
                    freeface_fpath = None
                # run get pipe crossing function
                logging.info(f'{counter}. Performing pipeline crossing algorithm for {hazard}...')
                counter += 1
                site_data = get_pipe_crossing_landslide_and_liq(
                    path_to_def_shp=spath_def_poly,
                    infra_site_data=site_data.copy(),
                    avail_data_summary=avail_data_summary,
                    opensra_dataset_dir=opensra_dataset_dir,
                    infra_site_data_geom=site_data_geom,
                    export_dir=processed_input_dir,
                    def_type=hazard,
                    def_shp_crs=def_shp_crs,
                    freeface_fpath=freeface_fpath
                )
                # if can't find crossing, end preprocessing
                if site_data is None:
                    raise ValueError(f"FATAL: No crossings identified using specified deformation polygons!")
                # continue
                if spath_def_poly is not None:
                    performed_crossing = True
                if site_data.shape[0] == 0:
                    logging.info('\n')
                    logging.info(f'*****FATAL*****')
                    logging.info(f'- No crossings identified using CPT generated deformation polygons for {hazard}!')
                    logging.info(f'- Preprocessing will now exit as the final risk metrics will all be zero.')
                    if hazard == 'landslide':
                        logging.info(f'- Please revise the input infrastructure file and/or the landslide deformation shapefile and try preprocessing again.')
                    elif 'liquefaction' in workflow['EDP']:
                        logging.info(f'- Please revise the input infrastructure file and/or the site investigation data and try preprocessing again.')
                    # logging.info(f'- Preprocessing will now exit.')
                    logging.info(f'*****FATAL*****')
                    logging.info('\n')
                    raise ValueError(f"FATAL: No crossings identified using CPT generated deformation polygons for {hazard}!")
                    # sys.exit()
                else:
                    logging.info(f'... DONE - Obtained pipeline crossing for {hazard}')
    
    # -----------------------------------------------------------
    # rvs and fixed params split by preferred and user provided
    pref_rvs, user_prov_table_rvs, user_prov_gis_rvs, \
    pref_fixed, user_prov_table_fixed, user_prov_gis_fixed = \
        separate_params_by_source(rvs_input, fixed_input)
    logging.info(f'{counter}. Separated random and fixed parameters by source (i.e., Preferred vs user-specified)')
    counter += 1
    
    # -----------------------------------------------------------
    # get param_dist_meta from user-provided information
    param_dist_meta, param_dist_table = get_param_dist_from_user_prov_table(
        user_prov_table_rvs, user_prov_table_fixed,
        user_prov_gis_rvs, user_prov_gis_fixed,
        pref_rvs, pref_fixed, site_data, user_prov_gis_dir,
        infra_type, running_cpt_based_procedure
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
    if "EDP" in workflow and "liquefaction" in workflow["EDP"] and "CPTBased" in workflow["EDP"]["liquefaction"]:
        param_dist_table['level_to_run'] = np.ones(param_dist_table.shape[0])*3
        param_dist_table['level_to_run'] = param_dist_table['level_to_run'].values.astype(int)
    else:
        param_dist_table = get_level_to_run(
            param_dist_table,
            workflow,
            params_with_missing_dist_metric,
            param_dist_meta,
            setup_config,
            flag_using_state_network,
            infra_type=infra_type
        )
    level_to_run = param_dist_table['level_to_run'][0]
    logging.info(f'{counter}. Determined level of analysis to run (currently only varies for below-ground infrastructure)')
    logging.info(f'\t- level to run: {level_to_run}')
    counter += 1

    # -----------------------------------------------------------
    # update to landslide params if level to run == 1
    updated_site_data = False
    if infra_type == 'below_ground' and level_to_run == 1:
        # landslide crossings
        category = 'EDP'
        if category in workflow and 'landslide' in workflow[category]:
            # set to a generic value - will not be used at level 1.
            site_data.psi_dip = 15
            updated_site_data = True
        elif category in workflow and 'liquefaction' in workflow[category]:
            if 'lateral_spread' in workflow[category]:
                pass
            elif 'settlement' in workflow[category]:
                pass
        if updated_site_data:
            # export updated crossing summary table
            site_data.drop(columns='geometry').to_csv(
                # os.path.join(export_dir,f'site_data_{def_type.upper()}_CROSSINGS_ONLY.csv'),
                os.path.join(processed_input_dir,f'site_data_PROCESSED_CROSSING_ONLY.csv'),
                index=False
            )
            logging.info(f'{counter}. Performed additional actions for pipe crossing')
            counter += 1
    
    # -----------------------------------------------------------
    # get rest of distribution metrics from preferred datasets
    logging.info(f'{counter}. Retrieving missing distribution metrics from preferred distributions...')
    counter += 1
    param_dist_table, param_dist_meta, params_with_missing_dist_metric = \
        get_pref_dist_for_params(
            params_with_missing_dist_metric,
            site_data,
            param_dist_table,
            param_dist_meta,
            pref_param_dist,
            # pref_param_dist_const_with_level,
            pref_param_fixed,
            workflow,
            avail_data_summary,
            opensra_dataset_dir,
            level_to_run,
            flag_using_state_network,
            # running_cpt_based_procedure,
            # site_data_with_crossing_only,
            export_path_dist_table=os.path.join(processed_input_dir,'param_dist.csv'),
            export_path_dist_json=os.path.join(processed_input_dir,'param_dist_meta.json'),
            infra_type=infra_type
        )
    logging.info(f'... DONE - Retrieved missing distribution metrics from preferred distributions')
    
    # -----------------------------------------------------------
    # get IM predictions
    logging.info(f'{counter}. Getting ground motion (i.e., IM) predictions from {im_source}...')
    counter += 1
    if im_source == "ShakeMap":
        logging.info(f'********')
        get_im_pred(
            im_source, im_dir, site_data, infra_loc_header, im_filters,
            # for ShakeMaps
            sm_dir=sm_dir,
            sm_events=sm_events,
            event_ids_to_keep=event_ids_to_keep,
            # rupture_table=rupture_table_from_crossing,
            # col_headers_to_append=col_headers_to_append,
        )
        logging.info(f'********')
    elif im_source == "UserDefinedRupture" or im_source == 'UCERF':
        # running UCERF and using statewide pipeline, skip IM calc and use precomputed files
        # last statement to catch debugging/testing examples, which uses a subset of the statewide segments
        if im_source == 'UCERF' and flag_using_state_network and flag_using_state_full_set:
            # if rupture_table is created previously, which contains events to keep
            if performed_crossing:
                # for each IM file, keep only keep segment IDs with crossings
                site_ids_to_keep = site_data.ID.values
                # always 1 less for index for preprocessed state network 
                site_inds_to_keep = site_ids_to_keep - 1
                site_inds_to_keep = site_inds_to_keep.astype(int)
            if event_ids_to_keep is not None:
                # load rupture table with IM stage
                rupture_table_im_fpath = os.path.join(preprocess_im_dir,'RUPTURE_METADATA.csv')
                rupture_table_from_im = pd.read_csv(rupture_table_im_fpath)
                rupture_table_from_im.event_id = rupture_table_from_im.event_id.values.astype(int)
                event_inds_to_keep = np.asarray([
                    np.where(rupture_table_from_im.event_id==event_id)[0][0]
                    for event_id in event_ids_to_keep
                ])
            logging.info(f'\t-Copy/paste precomputed IMs to destination:')
            # copy each item in directory
            for each in os.listdir(preprocess_im_dir):
                src_path = os.path.join(preprocess_im_dir,each)
                dst_path = os.path.join(im_dir,each)
                if os.path.isdir(src_path):
                    if not os.path.exists(dst_path):
                        os.mkdir(dst_path)
                    if performed_crossing:
                        for f in os.listdir(src_path):
                            # get and read sparse data
                            curr_src_path = os.path.join(src_path,f)
                            data = sparse.load_npz(curr_src_path).toarray()
                            # get subset of datafile with sites to keep
                            data_with_sites_to_keep = data[:,site_inds_to_keep]
                            if event_ids_to_keep is not None:
                                data_with_sites_to_keep = data_with_sites_to_keep[event_inds_to_keep,:]
                            # convert back to sparse matrix and export
                            coo_out = sparse.coo_matrix(data_with_sites_to_keep)
                            sparse.save_npz(os.path.join(dst_path,f), coo_out)
                    else:
                        # remove if existing
                        if os.path.exists(dst_path):
                            shutil.rmtree(dst_path)
                        shutil.copytree(src=src_path,dst=dst_path)
                else:
                    if performed_crossing and 'site_data' in os.path.basename(src_path) and \
                        src_path.endswith('.csv'):
                        data = pd.read_csv(src_path)
                        # get subset of datafile with sites to keep
                        data_with_sites_to_keep = data.loc[site_inds_to_keep].copy()
                        # export
                        data_with_sites_to_keep.to_csv(dst_path,index=False)
                    else:
                        # remove if existing
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        shutil.copy(src=src_path,dst=dst_path)
                logging.info(f'\t\t-{dst_path}')
        else:
            logging.info(f'********')
            get_im_pred(
                im_source, im_dir, site_data, infra_loc_header, im_filters,
                # for user-defined and UCERF ruptures
                opensra_dir=opensra_dir,
                processed_input_dir=processed_input_dir,
                rup_fpath=rup_fpath,
                event_ids_to_keep=event_ids_to_keep,
                # rupture_table=rupture_table
            )
            logging.info(f'********')
    # merge rupture metadata from crossing (if exists) to that from IM
    if rupture_table_from_crossing is not None:
        # load rupture table with IM stage
        rupture_table_im_fpath = os.path.join(im_dir,'RUPTURE_METADATA.csv')
        rupture_table_from_im = pd.read_csv(rupture_table_im_fpath)
        rupture_table_from_im.event_id = rupture_table_from_im.event_id.values.astype(int)
        if event_ids_to_keep is not None:
            if event_inds_to_keep is None:
                event_inds_to_keep = np.asarray([
                    np.where(rupture_table_from_im.event_id.values==event_id)[0][0]
                    for event_id in event_ids_to_keep
                ])
            rupture_table_from_im = rupture_table_from_im.loc[event_inds_to_keep].reset_index(drop=True)
        # initialize empty list
        collect_list = {}
        for col in col_headers_to_append:
            collect_list[col] = [[]]*rupture_table_from_im.shape[0]
        # find common event IDs
        for i in range(rupture_table_from_im.shape[0]):
            event_i_in_im = rupture_table_from_im.event_id[i]
            if event_i_in_im in rupture_table_from_crossing.EventID.values:
                row = np.where(rupture_table_from_crossing.EventID.values==event_i_in_im)[0][0]
                for col in col_headers_to_append:
                    collect_list[col][i] = rupture_table_from_crossing.loc[row,col]
        # append to table
        for col in col_headers_to_append:
            rupture_table_from_im[col] = collect_list[col]
        # export and update rupture_metadata file
        rupture_table_from_im.to_csv(rupture_table_im_fpath,index=False)
        # export to shp
        save_name_shp = rupture_table_im_fpath.replace('.csv','.gpkg')
        geoms = []
        for i in range(rupture_table_from_im.shape[0]):
            # trace = np.asarray(json.loads(rup_meta.fault_trace.iloc[i]))
            if isinstance(rupture_table_from_im.fault_trace.iloc[i],str):
                trace = np.asarray(json.loads(rupture_table_from_im.fault_trace.iloc[i]))
            else:
                trace = np.asarray(rupture_table_from_im.fault_trace.iloc[i])                
            geoms.append(LineString(trace[:,:2]))
        rupture_table_from_im_gdf = GeoDataFrame(
            pd.read_csv(rupture_table_im_fpath), # reread dataframe to convert fields of lists into strings
            # rupture_table_from_im,
            crs=4326, geometry=geoms
        )
        rupture_table_from_im_gdf.to_file(save_name_shp,index=False,layer='data')
    logging.info(f'... DONE - Obtained IM predictions from {im_source} and stored to:')
    logging.info(f"\t{im_dir}")
    
    # -----------------------------------------------------------
    # get well and caprock crossings - may move to another location in Preprocess, but must be after getIM
    # well_crossing_ordered_by_faults = None           
    if infra_type == 'wells_caprocks':
        logging.info(f'{counter}. Getting well crossings for fault rupture...')
        counter += 1
        # get well crossings
        # check if path is already a valid filepath, if not then infer from input_dir
        well_trace_dir = setup_config['Infrastructure']['WellTraceDir']
        well_trace_dir = check_and_get_abspath(well_trace_dir, input_dir)
        # well_crossing_ordered_by_faults, _ = get_well_crossing(
        get_well_crossing(
            im_dir=im_dir,
            infra_site_data=site_data.copy(),
            col_with_well_trace_file_names='file_name',
            well_trace_dir=well_trace_dir,
        )
        logging.info(f'... DONE - Obtained well crossings for fault rupture')
        
        # get caprock crossings
        if 'CaprockLeakage' in setup_config['DecisionVariable']['Type']:
            logging.info(f'{counter}. Getting caprock crossings for fault rupture...')
            counter += 1
            # check if path is already a valid filepath, if not then infer from input_dir
            caprock_fdir = setup_config['Infrastructure']['PathToCaprockShapefile']
            caprock_fdir = check_and_get_abspath(caprock_fdir, input_dir)
            # get shapefile for caprock
            caprock_shp_file = get_shp_file_from_dir(caprock_fdir)
            # run caprock crossing algorith
            get_caprock_crossing(
                caprock_shp_file=caprock_shp_file,
                # rup_fpath=rup_fpath,
                im_dir=im_dir,
                processed_input_dir=processed_input_dir
            )
            logging.info(f'... DONE - Obtained caprock crossings for fault rupture')
    
    # -----------------------------------------------------------
    # end of preprocess
    logging.info('---------------')
    logging.info('******** End of preprocessing for OpenSRA ********')
    logging.info('---------------')


# -----------------------------------------------------------
def get_im_pred(
    im_source, im_dir, site_data, infra_loc_header, im_filters,
    # for ShakeMaps
    sm_dir=None, sm_events=None,
    # for user-defined ruptures
    opensra_dir=None, processed_input_dir=None, rup_fpath=None,
    event_ids_to_keep=None,
    # to return haz object for additional processing
    return_haz_obj=False
):
    """get IM predictions from backend"""
    # initialize seismic hazard class
    seismic_hazard = getattr(haz, 'SeismicHazard')()
    
    # get IM predictions based on IM source
    if im_source == "ShakeMap":
        # set sites and site params
        if 'LON_MID' in site_data:
            seismic_hazard.set_site_data(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values,
                vs30=np.zeros(site_data.shape[0])
            )
        elif 'LON' in site_data:
            seismic_hazard.set_site_data(
                lon=site_data.LON.values,
                lat=site_data.LAT.values,
                vs30=np.zeros(site_data.shape[0])
            )
        else:
            seismic_hazard.set_site_data(
                lon=site_data[infra_loc_header['lon_header']],
                lat=site_data[infra_loc_header['lat_header']],
                vs30=np.zeros(site_data.shape[0])
            )
        seismic_hazard.init_ssc(im_source,sm_dir=sm_dir,event_names=sm_events)  # initialize source
        
    elif im_source == 'UserDefinedRupture' or im_source == 'UCERF':
        # prepackaged site data
        gmc_site_data_dir = os.path.join(opensra_dir,'lib','OtherData','Preprocessed','Statewide_and_Regional_Grids')
        cols_to_get = ['vs30','vs30source','z1p0','z2p5']
        
        # create PointData class to make use of nearest neighbor sampling schemes
        if 'LON_MID' in site_data:
            _infra = geodata.PointData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values,
            )
        elif 'LON' in site_data:
            _infra = geodata.PointData(
                lon=site_data.LON.values,
                lat=site_data.LAT.values,
            )
        else:
            _infra = geodata.PointData(
                lon=site_data[infra_loc_header['lon_header']],
                lat=site_data[infra_loc_header['lat_header']],
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
                _grid_nodes = geodata.PointData(fpath=gmc_site_data_path)
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
        # once the nearest_node_table has been populated, then get sites from nearest node
        _infra.component_table = _infra.sample_table(
            input_table=_infra.component_table.copy(),
            sample_table=temp_nearest_node_table.copy(),
            # cols_to_get=cols_to_get,
            # use_hull=True
            use_hull=False
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
        rate_min=rate_min,
        event_ids_to_keep=event_ids_to_keep,
    ) # process ruptures
    seismic_hazard.get_gm_pred_from_gmc() # get GM predictions
    seismic_hazard.export_gm_pred(sdir=im_dir) # export GM predictions
    
    # return
    if return_haz_obj:
        return seismic_hazard


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
            # also make copy of shapefile
            infra_shp_fpath = infra_fpath.replace('.csv','.gpkg')
            shutil.copy(
                src=infra_shp_fpath,
                dst=os.path.join(processed_input_dir,'site_data_PROCESSED.gpkg')
            )
        else:
            if infra_fpath.endswith('shp') or infra_fpath.endswith('gpkg'):
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
                raise ValueError('Only supports "shp" or "gpkg" for shapefiles or "csv" as input file type')
            # network
            infra.split_network_by_max_length(l_max) # l_max in km
            infra.make_segment_table()
            infra.export_segment_table(
                sdir=processed_input_dir,
                sname='site_data_PROCESSED',
                to_replace=True
            )
    else:
        if infra_fpath.endswith('shp') or infra_fpath.endswith('gpkg'):
            infra = geodata.PointData(fpath=infra_fpath)
        elif infra_fpath.endswith('csv'):
            infra = geodata.PointData(
                fpath=infra_fpath,
                lon_header=infra_loc_header["lon_header"],
                lat_header=infra_loc_header["lat_header"],
            )
        else:
            raise ValueError('Only supports "shp" or "gpkg" for shapefiles or "csv" as input file type')
        # process components/sites
        infra.make_component_table()
        infra.export_component_table(
            sdir=processed_input_dir,
            sname='site_data_PROCESSED',
            to_replace=True
        )


# -----------------------------------------------------------
def make_workflow(setup_config, input_dir, processed_input_dir, to_export=True):
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
        'pipe_strain_settlement': "SettlementInducedPipeStrain",
        'pipe_strain_landslide': "LandslideInducedPipeStrain",
        'pipe_strain_lateral_spread': "LateralSpreadInducedPipeStrain",
        'pipe_strain_surface_fault_rupture': "SurfaceFaultRuptureInducedPipeStrain",
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
    # list of attributes to get if available
    model_att_to_get = [
        # should always exist
        "ModelWeight",
        # rest are for generic model
        "UpstreamCategory",
        "UpstreamParams",
        "ReturnCategory",
        "ReturnParams",
        "Aleatory",
        "Epistemic",
        "PathToModelInfo",
    ]
    for category in cat_map:
        workflow[category] = {}
        haz_list = setup_config[cat_map[category]]['Type']
        for hazard in haz_map:
            if haz_map[hazard] in haz_list and haz_list[haz_map[hazard]]['ToInclude']:
                workflow[category][hazard] = {}
                method_list = haz_list[haz_map[hazard]]['Method']
                # for getting model input attributes
                for method in method_list:
                    workflow[category][hazard][method] = {}
                    for att in model_att_to_get:
                        if att in method_list[method]:
                            if att == "PathToModelInfo": 
                                workflow[category][hazard][method][att] = \
                                    check_and_get_abspath(method_list[method][att], input_dir)
                            else:
                                workflow[category][hazard][method][att] = method_list[method][att]
                    # append current category to attribute list
                    workflow[category][hazard][method]['ReturnCategory'] = category
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

    # for each catesgory
    for category in workflow:
        curr_cat = workflow[category]
        # for each hazard
        for haz_type in curr_cat:
            curr_haz_type = curr_cat[haz_type]
            # if CPTBased, hard code in params for now:
            if "CPTBased" in curr_cat[haz_type]:
                all_rvs += []
            else:
                # load python file
                _file = importlib.import_module('.'.join(['src',category.lower(),haz_type.lower()]))
                # for each method
                for method in curr_haz_type:
                    # create instance
                    _inst = copy.deepcopy(getattr(_file, method)())
                    # get all RVs for method
                    all_rvs += _inst._missing_inputs_rvs
                    rvs_by_level, fix_by_level = _inst.get_req_rv_and_fix_params(infra_fixed)
                    for i in range(3):
                        # initialize list
                        if not f'level{i+1}' in req_rvs_by_level:
                            req_rvs_by_level[f'level{i+1}'] = rvs_by_level[f'level{i+1}']
                            req_fixed_by_level[f'level{i+1}'] = fix_by_level[f'level{i+1}']
                        else:
                            req_rvs_by_level[f'level{i+1}'] += rvs_by_level[f'level{i+1}']
                            req_fixed_by_level[f'level{i+1}'] += fix_by_level[f'level{i+1}']

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
        pref_param_dist[curr_level] = pd.read_excel(pref_param_dist_path,sheet_name=curr_level)
    # fixed
    pref_param_fixed = pd.read_excel(
        pref_param_dist_path,
        sheet_name='fixed'
    )
    return pref_param_dist, pref_param_fixed


# -----------------------------------------------------------
def read_input_tables(
    input_dir, processed_input_dir, flag_using_state_network, flag_using_region_network,
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
    if flag_using_state_network or flag_using_region_network:
        row_for_d_pipe = np.where(rvs_input.Name=='d_pipe')[0][0]
        if rvs_input.loc[row_for_d_pipe,'Source'] == 'Preferred':
            rvs_input.loc[row_for_d_pipe,'Source'] = 'From infrastructure table or enter value'
            rvs_input.loc[row_for_d_pipe,'Mean or Median'] = 'DIAMETER'
            rvs_input.loc[row_for_d_pipe,'Distribution Type'] = 'Normal'
    # preload infrastructure geometry if it exists, otherwise create it
    if infra_geom_fpath is not None:
        site_data_geom = read_file(infra_geom_fpath).geometry
    else:
        if infra_type == 'below_ground':
            site_data_geom = GeoDataFrame(
                None,
                crs=4326,
                geometry=make_list_of_linestrings(
                    pt1_x=site_data['LON_BEGIN'].values,
                    pt1_y=site_data['LAT_BEGIN'].values,
                    pt2_x=site_data['LON_END'].values,
                    pt2_y=site_data['LAT_END'].values,
                )
            ).geometry
        else:
            site_data_geom = GeoDataFrame(
                None,
                crs=4326,
                geometry=points_from_xy(
                    x=site_data['LON'],
                    y=site_data['LAT'],
                    crs=4326
                )
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
def get_param_dist_from_user_prov_table(
    user_prov_table_rvs,
    user_prov_table_fixed,
    user_prov_gis_rvs,
    user_prov_gis_fixed,
    pref_rvs,
    pref_fixed,
    site_data,
    user_prov_gis_dir,
    infra_type,
    running_cpt_based_procedure=False,
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
    
    # out of bound values
    # invalid_value_for_raster_sampling = {
    #     'gw_depth': 999 # m
    # }
    
    # if running CPTs, skip gw_depth and slope for inputs
    params_to_skip = []
    if running_cpt_based_procedure:
    #     pref_params_to_add = ['beta_crossing','psi_dip','theta_rake','gw_depth','slope']
        params_to_skip = ['gw_depth','slope']
    
    # dataframe for storing parameter distributions
    param_dist_table = pd.DataFrame(None)
    
    # first loop through user provided random params
    for i in range(user_prov_table_rvs.shape[0]):
        param = user_prov_table_rvs.Name[i]
        if not param in params_to_skip:
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
            mean_val = user_prov_table_rvs.loc[i,metric_map['mean']]
            # if param == 's_u_backfill':
            #     print('here')
            try: # try converting to float, if can't then it's a column name
                curr_param_dist['mean'] = float(mean_val)
            except ValueError: # read from site data table
                if mean_val in site_data:
                    col_name = mean_val
                elif mean_val.upper() in site_data:
                    col_name = mean_val.upper()
                else:
                    raise ValueError(f'Cannot identify column in site_data given column name for {param}')
                curr_param_dist['mean'] = site_data[col_name].values
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
                        curr_param_dist['sigma'] = np.sqrt(np.log((curr_cov/100)**2 + 1))
                        # curr_param_dist['sigma'] = np.log(1+curr_cov/100)
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
                # if len(ind) > 0:
                if len(ind) == n_site:
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
            locs = geodata.PointData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values
            )
        elif 'LON' in site_data:
            locs = geodata.PointData(
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
        # curr_user_prov_gis_dir = os.path.join(user_prov_gis_dir,user_prov_gis_rvs['Mean or Median'][i])
        # find files under fdir
        # files = os.listdir(curr_user_prov_gis_dir)
        # look for either ".tif" for raster of ".shp" for Shapefile
        # curr_user_prov_gis_fpath = None
        files = glob.glob(os.path.join(user_prov_gis_dir,'*',user_prov_gis_rvs['Mean or Median'][i]))
        curr_user_prov_gis_fpath = files[0]
        # for f in files:
        #     if f.endswith('.tif') or f.endswith('.shp') or f.endswith('.gpkg'):
        #         if f.endswith('.tif'):
        #             gis_type = 'raster'
        #         else:
        #             gis_type = 'shapefile'
        #         curr_user_prov_gis_fpath = os.path.join(curr_user_prov_gis_dir,f)
        #         break
        # if curr_user_prov_gis_fpath is None:
        #     logging.info('Cannot locate user provided GIS file: file must end with ".tif", ".shp", or ".gpkg"')
        if curr_user_prov_gis_fpath.endswith('.tif'):
            gis_type = 'raster'
        elif curr_user_prov_gis_fpath.endswith('.shp') or \
            curr_user_prov_gis_fpath.endswith('.gpkg'):
            gis_type = 'shapefile'
        else:
            raise ValueError('GIS file must end with ".tif", ".shp", or ".gpkg"')
        # with file path, now sample
        if gis_type == 'raster':
            # get sample from GIS file
            locs.data = locs.sample_raster(
                input_table=locs.data,
                fpath=curr_user_prov_gis_fpath,
                store_name=param
            )
        elif gis_type == 'shapefile':
            locs.data = locs.sample_shapefile(
                input_table=locs.data,
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
                    # curr_param_dist['sigma'] = np.log(1+curr_cov/100)
                    curr_param_dist['sigma'] = np.sqrt(np.log((curr_cov/100)**2 + 1))
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
            locs = geodata.PointData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values
            )
        elif 'LON' in site_data:
            locs = geodata.PointData(
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
        curr_user_prov_gis_dir = os.path.join(user_prov_gis_dir,user_prov_gis_fixed['Value'][i])
        # find files under fdir
        files = os.listdir(curr_user_prov_gis_dir)
        # look for either ".tif" for raster of ".shp" for Shapefile
        curr_user_prov_gis_fpath = None
        for f in files:
            if f.endswith('.tif') or f.endswith('.shp'):
                if f.endswith('.tif'):
                    gis_type = 'raster'
                else:
                    gis_type = 'shapefile'
                curr_user_prov_gis_fpath = os.path.join(curr_user_prov_gis_dir,f)
                break
        if curr_user_prov_gis_fpath is None:
            logging.info('Cannot locate user provided GIS file: file must end with ".tif" or ".shp"')
        else:
            # with file path, now sample
            if gis_type == 'raster':
                # get sample from GIS file
                locs.data = locs.sample_raster(
                    input_table=locs.data,
                    fpath=curr_user_prov_gis_fpath,
                    store_name=param
                )
            elif gis_type == 'shapefile':
                locs.data = locs.sample_shapefile(
                    input_table=locs.data,
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
        
    # some RV are determined internally and are not presented through the GUI
    if infra_type == 'below_ground':
        pref_params_to_add = ['beta_crossing','psi_dip','theta_rake']
    else:
        pref_params_to_add = []
    for param in pref_params_to_add:
        curr_param_dist = {
            'source': 'Preferred'
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
    flag_using_state_network=False,
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
    
    # determine RVs needed by level
    all_rvs, req_rvs_by_level, req_fixed_by_level = get_rvs_and_fix_by_level(workflow, infra_fixed)
    
    param_to_skip_for_determining_level = {
        'level1': ['prob_liq','liq_susc'],
        'level2': ['prob_liq','liq_susc','gw_depth'],
        'level3': ['prob_liq','liq_susc','gw_depth'],
    }
    if flag_using_state_network:
        for i in range(3):
            param_to_skip_for_determining_level[f'level{i+1}'] += ['d_pipe', 't_pipe']
    
    # for each site, determine level to run
    level_to_run = np.ones(n_site).astype(int)*3
    # loop through levels
    for i in range(3,1,-1):
        level_str = f'level{i}'
        # parameters required for current level
        params_for_curr_level = req_rvs_by_level[level_str]
        ind_for_cur_level = np.array([])
        for param in params_with_missing_dist_metric:
            if param in params_for_curr_level and \
                not param in param_to_skip_for_determining_level[level_str]:
                for met in params_with_missing_dist_metric[param]:
                    # ind_for_cur_level = np.hstack([
                    ind_for_cur_level = np.unique(np.hstack([
                        ind_for_cur_level,
                        # np.where(np.isnan(param_dist_meta[param][met]))[0]
                        np.where(param_dist_table[param+'_'+met].isnull())[0]
                    ]))
        ind_for_cur_level = np.unique(ind_for_cur_level).astype(int)
        level_to_run[ind_for_cur_level] -= 1
        
    # for landslide and liquefaction, limit max level to run based on availability of deformation polygon
    # landslide
    category = 'EDP'
    hazard = 'landslide'
    min_level_with_no_def_poly = 1
    if category in workflow and hazard in workflow[category]:        
        # get deformation polygon to use; if "statewide", then assign probability of 0.25 to all components instead
        landslide_setup_meta = setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters']
        use_def_poly = landslide_setup_meta['UseDeformationGeometry']
        if use_def_poly is False:
            level_to_run = min(min(level_to_run),min_level_with_no_def_poly) # level if no deformation polygon is to be used
    
    # store levels to run
    param_dist_table['level_to_run'] = level_to_run
    
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
    # pref_param_dist_const_with_level,
    pref_param_fixed,
    workflow,
    avail_data_summary,
    opensra_dataset_dir,
    level_to_run,
    flag_using_state_network,
    # running_cpt_based_procedure,
    # site_data_with_crossing_only=None,
    export_path_dist_table=None,
    export_path_dist_json=None,
    infra_type='below_ground',
    default_statewide_geo_map='wills'
):
    """get rest of the missing distribution metrics"""
    
    # initialize
    met_list = ['dist_type','mean','sigma','low','high']
    if 'EDP' in workflow:
        # if 'surface_fault_rupture' in workflow['EDP']:
        #     crossing_params = [
        #         'l_anchor',
        #         'beta_crossing_primary','psi_dip_primary','theta_rake_primary',
        #         'beta_crossing_secondary','psi_dip_secondary','theta_rake_secondary',
        #     ]
        # else:
        crossing_params = ['l_anchor','beta_crossing','psi_dip','theta_rake']
        if level_to_run == 3:
            crossing_params.append('def_length')
        
        # if running_cpt_based_procedure:
        #     pref_params_to_add = ['beta_crossing','psi_dip','theta_rake','gw_depth','slope']
        # else:
        #     pref_params_to_add = []
        # for param in pref_params_to_add:
        #     params_with_missing_dist_metric[param] = ['dist_type', 'mean', 'sigma', 'low', 'high']
        
    # crossing_params = ['l_anchor','beta_crossing','psi_dip','theta_rake']
    soil_prop_map = {}
    
    # first load geologic units from various geologic maps
    if infra_type == 'below_ground':
        
        # get coordinates
        if 'LON_MID' in site_data:
            locs = geodata.PointData(
                lon=site_data.LON_MID.values,
                lat=site_data.LAT_MID.values
            )
        elif 'LON' in site_data:
            locs = geodata.PointData(
                lon=site_data.LON.values,
                lat=site_data.LAT.values
            )
        else:
            raise ValueError("Cannot locate lon/lat")
        
        # Sample statewide geologic map for soil properties
        if 'EDP' in workflow and 'landslide' in workflow['EDP']:
            if ('phi_soil' in param_dist_meta and param_dist_meta['phi_soil']['source'] == 'Preferred') or \
               ('coh_soil' in param_dist_meta and param_dist_meta['coh_soil']['source'] == 'Preferred'):
                if default_statewide_geo_map == 'wills':
                    file_key = 'level1_geo_unit_wills15'
                    file_metadata = avail_data_summary['Parameters'][file_key]
                    store_name = file_metadata['ColumnNameToStoreAs']
                    geo_unit_fpath = os.path.join(opensra_dataset_dir,file_metadata['Datasets']['Set1']['Path'])
                    geo_unit_crs = file_metadata['Datasets']['Set1']['CRS']
                    locs.data = locs.sample_shapefile(
                        input_table=locs.data,
                        fpath=geo_unit_fpath,
                        crs=geo_unit_crs,
                        attr='PTYPE',
                        store_name=store_name,
                        missing_val='water'
                    )
                    param_dist_table[store_name] = locs.data[store_name].values
                    param_dist_table[store_name] = param_dist_table[store_name].astype('<U20')
                    # load strength params from Bain et al. (2022)
                    default_geo_prop_fpath = os.path.join(
                        opensra_dataset_dir,
                        avail_data_summary['Parameters']['phi_soil']['Datasets']['Set1']['Path']
                    )
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
                    
                # CGS geologic unit, may take this out
                # elif default_statewide_geo_map == 'cgs':
                #     file_key = 'level1_geo_unit_cgs10'
                #     store_name = avail_data_summary['Parameters'][file_key]['ColumnNameToStoreAs']
                #     geo_unit_fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
                #     locs.data = locs.sample_shapefile(
                #         input_table=locs.data,
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
        
        if level_to_run <= 2:
            # if liq susc is needed, then sample regional geologic units
            if 'liq_susc' in param_dist_meta and param_dist_meta['liq_susc']['source'] == 'Preferred':
                # read from Witter et al. (2006) and Bedrossian et al. (2012)
                logging.info(f'\tFor getting liquefaction susceptibility category from geologic maps...')
                for each in ['witter06','bedrossian12']:
                    file_key = f'level2_geo_unit_{each}'
                    file_metadata = avail_data_summary['Parameters'][file_key]
                    store_name = file_metadata['ColumnNameToStoreAs']
                    geo_unit_fpath = os.path.join(opensra_dataset_dir,file_metadata['Datasets']['Set1']['Path'])
                    geo_unit_crs = file_metadata['Datasets']['Set1']['CRS']
                    src_name = file_metadata['Datasets']['Set1']['Source']
                    locs.data = locs.sample_shapefile(
                        input_table=locs.data,
                        fpath=geo_unit_fpath,
                        crs=geo_unit_crs,
                        attr='PTYPE',
                        store_name=store_name
                    )
                    param_dist_table[store_name] = locs.data[store_name].values
                    witter_store_name = store_name
                    logging.info(f'\t\t- read {src_name} geologic units')
                # file_key = 'level2_geo_unit_bedrossian12'
                # file_metadata = avail_data_summary['Parameters'][file_key]
                # store_name = file_metadata['ColumnNameToStoreAs']
                # geo_unit_fpath = os.path.join(opensra_dir,file_metadata['Datasets']['Set1']['Path'])
                # locs.data = locs.sample_shapefile(
                #     input_table=locs.data,
                #     fpath=geo_unit_fpath,
                #     crs=geo_unit_crs,
                #     attr='PTYPE',
                #     store_name=store_name
                # )
                # param_dist_table[store_name] = locs.data[store_name].values
                # bedrossian_store_name = store_name
                # logging.info(f'\tRead Bedrossian et al. (2012) geologic units')
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
                
    # given level to run, get sheet with preferred param dist for level
    pref_param_dist_for_level = pref_param_dist[f'level{level_to_run}'].copy()
    
    # loop through rest of params with missing distribution metrics
    param_list = list(params_with_missing_dist_metric) # remaining params
    for param in param_list:
        if param in list(pref_param_dist_for_level.rv_label):
            # row for param in preferred distribution table
            row_for_param = np.where(pref_param_dist_for_level.rv_label==param)[0][0]

            # specific properties for landslide, use Wills et al. geo properties developed by Chris Bain
            if pref_param_dist_for_level['mean'][row_for_param] == 'depends' and \
                param in soil_prop_map:
                # for each unique geologic unit
                for each in unique_geo_unit:
                    # get rows with geo unit
                    rows_with_geo_unit = np.where(param_dist_table['wills_geo_unit_desc'].values==each)[0]
                    # intersection of geo unit and NaN
                    rows_comb = list(set(rows_nan).intersection(set(rows_with_geo_unit)))
                    # get preferred value from Chris Bain's table
                    rows_for_param = np.where(default_geo_prop['Unit Abbreviation'].values==each)[0][0]
                    # dist type
                    param_dist_table.loc[rows_comb,f'{param}_dist_type'] = 'lognormal'
                    # mean
                    pref_val = default_geo_prop[soil_prop_map[param]['mean']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_mean'] = np.log(pref_val)
                    # sigma
                    pref_val = default_geo_prop[soil_prop_map[param]['cov']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_sigma'] = \
                        np.sqrt(np.log((pref_val/100)**2 + 1))
                        # np.log(1+pref_val/100) * param_dist_table.loc[rows_comb,f'{param}_mean'].values
                        # np.log(1+pref_val/100) * param_dist_table.loc[rows_comb,f'{param}_mean'].values
                    # low
                    pref_val = default_geo_prop[soil_prop_map[param]['low']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_low'] = np.log(pref_val)
                    # high
                    pref_val = default_geo_prop[soil_prop_map[param]['high']][rows_for_param]
                    param_dist_table.loc[rows_comb,f'{param}_high'] = np.log(pref_val)

            # all other cases
            else:
                # loop through missing metrics
                for met in met_list:
                    if met in params_with_missing_dist_metric[param]:
                        # find rows where null
                        rows_nan = np.where(param_dist_table[f'{param}_{met}'].isnull())[0]
                        # get preferred value
                        pref_val = pref_param_dist_for_level[met][row_for_param]
                        # specifically for mean
                        if met == 'mean':
                            # get crossing from site datatable
                            if pref_val == 'depends':
                                # for pipe crossing parameters
                                if param in crossing_params:
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
                                if param == 'gw_depth':
                                    pass
                                # path for GIS file
                                file_metadata = avail_data_summary['Parameters'][param]
                                gis_fpath = os.path.join(opensra_dataset_dir,file_metadata['Datasets']['Set1']['Path'])
                                gis_crs = file_metadata['Datasets']['Set1']['CRS']
                                locs.data = locs.sample_raster(
                                    input_table=locs.data,
                                    fpath=gis_fpath,
                                    crs=gis_crs,
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
                                # if running statewide pipeline and level<=2, catch t_pipe and op_press:
                                if flag_using_state_network and level_to_run <= 2:
                                    if param == 'op_press' and pref_val == 'user provided':
                                        pref_val = pref_param_dist['level1'].loc[row_for_param,'mean'] # set to level 1 default
                                    if param == 't_pipe' and pref_val == 'user provided':
                                        pref_val = param_dist_table['d_pipe_mean'].values*0.05 # mm, use 5% of diameter
                                # check for lognormal and apply correction
                                if param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    pref_val = np.log(pref_val)
                                param_dist_table.loc[rows_nan,f'{param}_mean'] = pref_val
                        # specifically for sigma, check CoV
                        elif met == 'sigma':
                            if np.isnan(pref_val):
                                # if nan, get from cov
                                pref_val = pref_param_dist_for_level['cov'][row_for_param]
                                # update distribution metric
                                if param_dist_table[f'{param}_dist_type'][0] == 'normal':
                                    param_dist_table.loc[rows_nan,f'{param}_sigma'] = \
                                        pref_val/100 * param_dist_table.loc[rows_nan,f'{param}_mean'].values
                                elif param_dist_table[f'{param}_dist_type'][0] == 'lognormal':
                                    param_dist_table.loc[rows_nan,f'{param}_sigma'] = \
                                        np.sqrt(np.log((pref_val/100)**2 + 1))
                                        # np.log(1+pref_val/100) * param_dist_table.loc[rows_nan,f'{param}_mean'].values
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
def preprocess_cpt_data(
    # predetermined setup configuration parameters
    setup_config, opensra_dir, im_dir, processed_input_dir, input_dir,
    rvs_input, fixed_input, workflow,
    # OpenSRA internal files
    avail_data_summary, opensra_dataset_dir,
    # for all IM sources
    im_source, im_filters,
    # for ShakeMaps
    sm_dir=None, sm_events=None,
    # for user-defined ruptures
    rup_fpath=None,
    # for sampling and Forward Euler differentiation for PC
    num_epi_input_samples=50, forward_euler_multiplier=1.01,
    # misc.
    display_after_n_event=100
):
    """wrapper to use functions from src.edp.process_cpt_spt"""
    # -----------------------------------------------------------
    # coordinate systems
    epsg_wgs84 = 4326 # lat lon, deg
    epsg_utm_zone10 = 32610 # m
    transformer_wgs84_to_utmzone10 = Transformer.from_crs(epsg_wgs84, epsg_utm_zone10)
    transformer_utmzone10_to_wgs84 = Transformer.from_crs(epsg_utm_zone10, epsg_wgs84)
    
    # -----------------------------------------------------------
    # setup config shorthands
    im_setup_config = setup_config['IntensityMeasure']
    edp_setup_config = setup_config['EngineeringDemandParameter']
    cpt_setup_params = setup_config['UserSpecifiedData']['CPTParameters']
    
    # -----------------------------------------------------------
    # get summary file
    cpt_summary_fpath = cpt_setup_params['PathToCPTSummaryCSV']
    # get folder with CPT data
    cpt_data_fdir = cpt_setup_params['PathToCPTDataFolder']
    # check if fpath is already a valid filepath, if not then infer from user provided GIS directory
    cpt_summary_fpath = check_and_get_abspath(cpt_summary_fpath, input_dir)
    cpt_data_fdir = check_and_get_abspath(cpt_data_fdir, input_dir)
    # get column with groundwater table depth
    col_with_gw_depth = cpt_setup_params['ColumnInCPTSummaryWithGWTable']
    # read CPT data
    cpt_meta_wgs84, cpt_meta_utm, cpt_data, n_cpt = \
        process_cpt_spt.read_cpt_data(cpt_summary_fpath, cpt_data_fdir, col_with_gw_depth)
    cpt_loc_headers = {
        'lon_header': 'Longitude',
        'lat_header': 'Latitude'
    }
    
    # -----------------------------------------------------------
    # make a processed folder for CPTs
    processed_cpt_base_dir = os.path.join(processed_input_dir,'CPTs')
    processed_cpt_im_dir = os.path.join(im_dir,'CPTs')
    if not os.path.exists(processed_cpt_base_dir):
        os.mkdir(processed_cpt_base_dir)
        logging.info(f'- CPTs: directory for intermediate files:')
        logging.info(f'\t- {processed_cpt_base_dir}')
    if not os.path.exists(processed_cpt_im_dir):
        os.mkdir(processed_cpt_im_dir)
        logging.info(f'- CPTs: Intensity measure directory:')
        logging.info(f'\t- {processed_cpt_im_dir}')
    
    # -----------------------------------------------------------
    logging.info(f'********')
    # get ground motion predictions to use for CPT analysis
    if im_source == "ShakeMap":
        get_im_pred(
            im_source, processed_cpt_im_dir, cpt_meta_wgs84, cpt_loc_headers, im_filters,
            # for ShakeMaps
            sm_dir=sm_dir,
            sm_events=sm_events,
        )
    elif im_source == "UserDefinedRupture" or im_source == 'UCERF':
        get_im_pred(
            im_source, processed_cpt_im_dir, cpt_meta_wgs84, cpt_loc_headers, im_filters,
            # for user-defined ruptures
            opensra_dir=opensra_dir,
            processed_input_dir=processed_input_dir,
            rup_fpath=rup_fpath,
        )
    logging.info(f'********')
    
    # -----------------------------------------------------------
    # read IM distributions for CPTs
    cpt_im_import = {}
    for each in ['pga','pgv']:
        cpt_im_import[each] = {        
            # note that sparse matrix files require -10 to all values for correct magnitude;
            # during ground motion phase, locations with 0 intensities are reported as -10 instead of -np.inf;
            # for storage, in order to be memory efficient, ground motions are stored as sparse matrx by adding
            # 10 to all values, thus convert the -10 intensity magnitudes to 0;
            # ---> this means when using the sparse datafiles, they must be -10 to get the correct magnitude
            # 'mean_table': np.round(sparse.load_npz(os.path.join(im_dir,each.upper(),'MEAN.npz')).toarray()-10,decimals=3),
            # 'sigma_table': np.round(sparse.load_npz(os.path.join(im_dir,each.upper(),'ALEATORY.npz')).toarray(),decimals=3),
            # 'sigma_mu_table': np.round(sparse.load_npz(os.path.join(im_dir,each.upper(),'EPISTEMIC.npz')).toarray(),decimals=3)
            'mean_table': np.round(
                    sparse.load_npz(os.path.join(processed_cpt_im_dir,each.upper(),'MEAN.npz')).toarray()-10,
                decimals=3),
            'sigma_table': np.round(
                    sparse.load_npz(os.path.join(processed_cpt_im_dir,each.upper(),'ALEATORY.npz')).toarray(),
                decimals=3),
            'sigma_mu_table': np.round(
                    sparse.load_npz(os.path.join(processed_cpt_im_dir,each.upper(),'EPISTEMIC.npz')).toarray(),
                decimals=3),
        }    
    # use mean of IM intensity for all CPTs
    for each in ['pga','pgv']:
        cpt_im_import[each]['mean_table'] = \
            np.ma.masked_invalid(cpt_im_import[each]['mean_table']).mean(axis=1)
            # cpt_im_import[each]['mean_table'].mean(axis=1)
        cpt_im_import[each]['sigma_table'] = \
            np.mean(cpt_im_import[each]['sigma_table']**2,axis=1)**(0.5)
        cpt_im_import[each]['sigma_mu_table'] = \
            np.mean(cpt_im_import[each]['sigma_mu_table']**2,axis=1)**(0.5)
    # import rupture information
    cpt_rupture_table = pd.read_csv(os.path.join(processed_cpt_im_dir,'RUPTURE_METADATA.csv'))

    # -----------------------------------------------------------
    # form input_dist for CPTs
    cpt_input_dist = {}
    ones_arr = np.ones(cpt_meta_wgs84.shape[0])
    # get rest of dist for GW depth
    param_dist_pref = pd.read_excel(
        os.path.join(opensra_dir,'param_dist','below_ground.xlsx'),
        sheet_name='level3'
    )
    # make use of sampling functions in PointData class for sampling GIS files
    cpt_locs = geodata.PointData(
        lon=cpt_meta_wgs84.Longitude.values,
        lat=cpt_meta_wgs84.Latitude.values,
    )
    # mapping used in rvs_input.csv
    low_high_map = {
        'low': 'Distribution Min',
        'high': 'Distribution Max'
    }
    store_names = {
        'slope': 'slope_deg',
        'gw_depth': 'gw_depth_m'
    }
    # loop through additional CPT analysis params)
    for param in ['gw_depth','slope']:
        try:
            row = np.where(rvs_input.Name.values==param)[0][0]
                
            # if param == 'gw_depth':
            #     param_source = 'From infrastructure table or enter value'
            # else:
            #     param_source = rvs_input.loc[row,'Source']
            param_source = rvs_input.loc[row,'Source']
        except:
            param_source = 'Preferred'
        param_metadata = avail_data_summary['Parameters'][param]
        # store_name = param_metadata['ColumnNameToStoreAs']
        store_name = store_names[param]
        cpt_input_dist[param] = {}
        # # get input dist for GW depth
        # if param_source == 'Preferred':
        # get preferred distribution
        curr_param_dist_pref = param_dist_pref.loc[param_dist_pref.rv_label==param].to_dict('list')
        for each in curr_param_dist_pref:
            curr_param_dist_pref[each] = curr_param_dist_pref[each][0]
        # 1) dist type
        cpt_input_dist[param]['dist_type'] = curr_param_dist_pref['dist_type']
        # 2) get mean
        #    - for gw_depth, use data from column
        #    - for slope, use internal gis map
        # gw_depth
        if param == 'gw_depth':
            # store_name = col_with_gw_depth
            # store_name = store_names[param]
            cpt_meta_wgs84[store_name] = cpt_meta_wgs84[col_with_gw_depth].values
        elif param == 'slope':
            # 2) sample mean from map
            pref_gis_fpath = os.path.join(opensra_dataset_dir,param_metadata['Datasets']['Set1']['Path'])
            pref_gis_crs = param_metadata['Datasets']['Set1']['CRS']
            cpt_meta_wgs84[store_name] = cpt_locs.sample_raster(
                input_table=cpt_locs.data.copy(),
                fpath=pref_gis_fpath,
                store_name=store_name,
                dtype='float',
                crs=pref_gis_crs
            )[store_name].values
        cpt_input_dist[param]['mean'] = cpt_meta_wgs84[store_name].values
        # if lognormal, apply ln
        if cpt_input_dist[param]['dist_type'] == 'lognormal':
            cpt_input_dist[param]['mean'] = np.log(cpt_input_dist[param]['mean'])
        # 3) sigma
        # check if sigma or cov is given in preferred dist table
        if ~np.isnan(curr_param_dist_pref['sigma']):
            # sigma is available, use it directly
            cpt_input_dist[param]['sigma'] = ones_arr.copy() * curr_param_dist_pref['sigma']
        else:
            # calculate sigma from cov
            if cpt_input_dist[param]['dist_type'] == 'normal':
                cpt_input_dist[param]['sigma'] = \
                    ones_arr.copy() * curr_param_dist_pref['cov']/100 * cpt_meta_wgs84['meanGWT'].values
            elif cpt_input_dist[param]['dist_type'] == 'lognormal':
                # cpt_input_dist[param]['sigma'] = ones_arr.copy() * np.log(1+curr_param_dist_pref['cov']/100)
                cpt_input_dist[param]['sigma'] = \
                    ones_arr.copy() * np.sqrt(np.log((curr_param_dist_pref['cov']/100)**2 + 1))
        # 4) low and high
        for each in ['low','high']:
            if np.isnan(curr_param_dist_pref[each]):
                if each == 'low':
                    cpt_input_dist[param][each] = -np.inf
                elif each == 'high':
                    cpt_input_dist[param][each] = np.inf
            else:
                cpt_input_dist[param][each] = float(curr_param_dist_pref[each])
                if cpt_input_dist[param]['dist_type'] == 'lognormal':
                    if each == 'low' and cpt_input_dist[param][each] == 0:
                        cpt_input_dist[param][each] = -np.inf
                    else:
                        cpt_input_dist[param][each] = np.log(cpt_input_dist[param][each])
            # multiply by ones array
            cpt_input_dist[param][each] = cpt_input_dist[param][each] * ones_arr.copy()
        # elif param_source == 'From infrastructure table or enter value':
        #     # 1) dist_type
        #     cpt_input_dist[param]['dist_type'] = rvs_input.loc[row,'Distribution Type']
        #     # 2) collect mean for CPTs from input table
        #     # check if source is a float, if so then it's a value, otherwise treat it as a column header
        #     if param == 'gw_depth':
        #         mean_val = 'Mean GWT'
        #     else:
        #         mean_val = rvs_input.loc[row,'Mean or Median']
        #     try:
        #         # val_float = float(rvs_input.loc[row,'Mean or Median'])
        #         val_float = float(mean_val)
        #         cpt_meta_wgs84[store_name] = val_float
        #         cpt_input_dist[param]['mean'] = val_float * ones_arr.copy()
        #     except:
        #         # treat as column and look for values from site data table
        #         if mean_val in cpt_meta_wgs84:
        #             col_name = mean_val
        #         elif mean_val.upper() in cpt_meta_wgs84:
        #             col_name = mean_val.upper()
        #         else:
        #             raise ValueError(f'Cannot identify column in cpt_meta_wgs84 given column name for "{param}"')
        #         cpt_meta_wgs84[store_name] = cpt_meta_wgs84[mean_val].values
        #         cpt_input_dist[param]['mean'] = cpt_meta_wgs84[mean_val].values
        #     # if lognormal, apply ln
        #     if cpt_input_dist[param]['dist_type'] == 'lognormal':
        #         cpt_input_dist[param]['mean'] = np.log(cpt_input_dist[param]['mean'])
        #     # 3) sigma
        #     # check if sigma or cov is given in preferred dist table
        #     if ~np.isnan(rvs_input.loc[row,'Sigma']):
        #         # sigma is available, use it directly
        #         cpt_input_dist[param]['sigma'] = ones_arr.copy() * rvs_input.loc[row,'Sigma']
        #     else:
        #         # calculate sigma from cov
        #         if cpt_input_dist[param]['dist_type'] == 'normal':
        #             # cpt_input_dist[param]['sigma'] = rvs_input.loc[row,'CoV']/100 * cpt_input_dist[param]['mean']
        #             cpt_input_dist[param]['sigma'] = \
        #                 ones_arr.copy() * rvs_input.loc[row,'CoV']/100 * cpt_input_dist[param]['mean']
        #         elif cpt_input_dist[param]['dist_type'] == 'lognormal':
        #             # cpt_input_dist[param]['sigma'] = ones_arr.copy() * np.log(1+rvs_input.loc[row,'CoV']/100)
        #             cpt_input_dist[param]['sigma'] = \
        #                 ones_arr.copy() * np.sqrt(np.log((curr_param_dist_pref['CoV']/100)**2 + 1))
        #     # 4) low and high
        #     for each in ['low','high']:
        #         if np.isnan(rvs_input.loc[row,low_high_map[each]]):
        #             if each == 'low':
        #                 cpt_input_dist[param][each] = -np.inf
        #             elif each == 'high':
        #                 cpt_input_dist[param][each] = np.inf
        #         else:
        #             cpt_input_dist[param][each] = float(rvs_input.loc[row,low_high_map[each]])
        #             if cpt_input_dist[param]['dist_type'] == 'lognormal':
        #                 if each == 'low' and cpt_input_dist[param][each] == 0:
        #                     cpt_input_dist[param][each] = -np.inf
        #                 else:
        #                     cpt_input_dist[param][each] = np.log(cpt_input_dist[param][each])
        #         # multiply by ones array
        #         cpt_input_dist[param][each] = cpt_input_dist[param][each] * ones_arr.copy()
    logging.info(f'- Obtained distributions for CPT input parameters: "gw_depth" and "slope"')
    
    # -----------------------------------------------------------
    # get samples for CPT input dist
    n_cpt_params = len(cpt_input_dist)
    cpt_param_names = list(cpt_input_dist)
    cpt_input_samples = get_samples_for_params(cpt_input_dist, num_epi_input_samples, n_cpt)
    logging.info(f'- Generated samples using distributions of input parameters')
    
    # -----------------------------------------------------------
    # apply buffer to CPT locations
    buffer = 100 # m
    cpt_meta_utm_buffer = cpt_meta_utm.buffer(buffer)
    # get extents after applying buffer
    cpt_meta_utm_buffer_bounds = np.asarray(cpt_meta_utm_buffer.unary_union.bounds)
    # create grid given extent of CPT with buffer
    grid_spacing = 20 # m
    cpt_meta_utm_buffer_bounds[:2] = np.floor(cpt_meta_utm_buffer_bounds[:2]/grid_spacing)*grid_spacing
    cpt_meta_utm_buffer_bounds[2:] = np.ceil(cpt_meta_utm_buffer_bounds[2:]/grid_spacing)*grid_spacing
    # make grid nodes
    nodes = make_grid_nodes(
        cpt_meta_utm_buffer_bounds[0], cpt_meta_utm_buffer_bounds[1], # x and y min
        cpt_meta_utm_buffer_bounds[2], cpt_meta_utm_buffer_bounds[3], # x and y max
        grid_spacing, grid_spacing # dx and dy
    )
    # dimension of grid
    n_col = int((cpt_meta_utm_buffer_bounds[2]-cpt_meta_utm_buffer_bounds[0])/grid_spacing)+1
    n_row = int((cpt_meta_utm_buffer_bounds[3]-cpt_meta_utm_buffer_bounds[1])/grid_spacing)+1
    # make geodataframe with grid nodes
    gdf_nodes_utm = GeoSeries(
        points_from_xy(x=nodes[:,0],y=nodes[:,1]),
        crs=epsg_utm_zone10
    )
    # find nodes that intersect the CPT buffers
    cpt_to_node_mapping = cpt_meta_utm_buffer.sindex.query_bulk(gdf_nodes_utm,predicate='intersects')
    # for each node, get list of CPT buffers it intersects with
    cpts_per_node = [
        list(cpt_to_node_mapping[1][cpt_to_node_mapping[0]==i])
        for i in range(len(gdf_nodes_utm))
    ]
    # get lat lon of nodes
    nodes_lat_1d, nodes_lon_1d = transformer_utmzone10_to_wgs84.transform(
        gdf_nodes_utm.x.values, gdf_nodes_utm.y.values)
    logging.info(f'- Generated grid around CPTs:')
    logging.info(f'\t- buffer: {buffer} m')
    logging.info(f'\t- grid spacing: {grid_spacing} m')
    
    # -----------------------------------------------------------
    # for sampling at grid nodes
    grid_locs = geodata.PointData(
        lon=nodes_lon_1d,
        lat=nodes_lat_1d,
    )
    # sample aspect at grid nodes
    param = 'aspect'
    aspect_file_metadata = avail_data_summary['Parameters'][param]
    aspect_store_name = aspect_file_metadata['ColumnNameToStoreAs']
    aspect_gis_fpath = os.path.join(
        opensra_dataset_dir,
        aspect_file_metadata['Datasets']['Set1']['Path']
    )
    aspect_gis_crs = aspect_file_metadata['Datasets']['Set1']['CRS']
    grid_locs.data = grid_locs.sample_raster(
        input_table=grid_locs.data.copy(),
        fpath=aspect_gis_fpath,
        store_name=aspect_store_name,
        dtype='float',
        crs=aspect_gis_crs
    )
    logging.info(f'- Sampled aspect at grid nodes')
    
    # -----------------------------------------------------------
    # additional setup information for liquefaction methods
    liq_haz_map = {
        'LateralSpread': 'lateral_spread',
        'Settlement': 'settlement'
    }
    haz_to_run = []
    # for each in liq_haz_map:
    for each in ['lateral_spread','settlement']:
        if each in workflow['EDP']:
            haz_to_run.append(each)
    # weights for methods
    weight_r09 = 0.5
    if 'WeightRobertson09' in cpt_setup_params:
        weight_r09 = cpt_setup_params['WeightRobertson09']
    weight_z04 = 0.5
    if 'lateral_spread' in workflow['EDP']:
        if 'WeightZhang04' in cpt_setup_params:
            weight_z04 = cpt_setup_params['WeightZhang04']
    depth_scale_method = 'Nonlinear (Bain and Bray, 2022)'
    if 'DepthScalingMethod' in cpt_setup_params:
        depth_scale_method = cpt_setup_params['DepthScalingMethod']
    # get free-face feature if provided
    freeface_fpath = None
    if 'lateral_spread' in workflow['EDP']:
        if 'PathToFreefaceDir' in cpt_setup_params:
            freeface_fpath = cpt_setup_params['PathToFreefaceDir']
            # check length of filepath: if ==0, then assume nothing was provided
            if len(freeface_fpath) == 0:
                freeface_fpath = None
            else:
                freeface_fpath = check_and_get_abspath(freeface_fpath, input_dir)
    # for liquefaction calcs
    z_cutoff = 15 # m
    null_arr_cpt_sample = np.zeros((n_cpt, num_epi_input_samples))
    null_arr_sample = np.zeros((num_epi_input_samples))
    # weighting scheme for connecting CPT-based deforma+tion estimate to nodes
    # weight_scheme = 'distance' # distance for distance weighted or average of CPTs
    weight_scheme = 'average' # distance for distance weighted or average of CPTs
    # initialize screening params
    pgdef_cutoff = 5/100 # m
    buffer_size = 0.0001 # ~10 km
    logging.info(f'- Obtained additional liquefaction analysis paramteres from setup_config:')

    # -----------------------------------------------------------
    # processing for freeface feature or L/H ratios
    if freeface_fpath is not None:
        gdf_freeface_wgs84 = read_file(freeface_fpath, crs=epsg_wgs84)
        gdf_freeface_utm = gdf_freeface_wgs84.to_crs(epsg_utm_zone10)
        # for each node, get shortest distance to each freeface feature
        cpt_to_freeface_dist = np.asarray([
            cpt_meta_wgs84.to_crs(epsg_utm_zone10).distance(gdf_freeface_utm.geometry[i])
            for i in range(gdf_freeface_utm.shape[0])
        ])
        # for each CPT, find minimum distance on all features
        cpt_meta_wgs84['FreefaceDist_m'] = np.min(cpt_to_freeface_dist,axis=0)
        # for each CPT get closest freeface feature and get height from attribute
        _, nearest_freeface_feature = GeoSeries(gdf_freeface_utm.geometry).sindex.nearest(cpt_meta_wgs84.to_crs(epsg_utm_zone10).geometry)
        cpt_meta_wgs84['FreefaceHeight_m'] = gdf_freeface_utm['Height_m'].loc[nearest_freeface_feature].values
        # get freeface L/H for each CPT
        cpt_meta_wgs84['LH_Ratio'] = cpt_meta_wgs84['FreefaceDist_m']/cpt_meta_wgs84['FreefaceHeight_m']
        cpt_meta_wgs84['LH_Ratio'] = np.maximum(cpt_meta_wgs84['LH_Ratio'],4) # set lower limit to 4
        logging.info(f'- Loaded free-face feature and computed L/H ratios')

    # -----------------------------------------------------------
    # go through each liquefaction consequence
    gdf_poly_hull = {}
    lateral_spread_control_case_by_event = []
    amu_by_event = []
    bmu_by_event = []
    time_init = time.time()
    logging.info(f'- Looping through events to calculate liquefaction-induced deformation for CPTs...')
    event_count = 0
    for event_i in range(cpt_rupture_table.shape[0]):
        #####################################################
        # event_id for storage
        event_id = cpt_rupture_table.event_id[event_i]
        
        #####################################################
        # im
        cpt_im_dist_info = {}
        for each in ['pga','pgv']:
            cpt_im_dist_info[each] = {
                'mean': cpt_im_import[each]['mean_table'][event_i],
                # 'sigma': 0.0,
                'sigma': cpt_im_import[each]['sigma_table'][event_i],
                # 'sigma_mu': 0.0,
                'sigma_mu': cpt_im_import[each]['sigma_mu_table'][event_i],
                'dist_type': 'lognormal'
            }
            # avoid sigma = 0 for PC
            cpt_im_dist_info[each]['sigma'] = np.maximum(
                cpt_im_dist_info[each]['sigma'],0.001
            )
        # forward IM for gettning derivatives
        cpt_im_dist_info_forward = {}
        for each in ['pga','pgv']:
            cpt_im_dist_info_forward[each] = {
                'mean': cpt_im_dist_info[each]['mean'].copy() * forward_euler_multiplier,
                'dist_type': cpt_im_dist_info[each]['dist_type']
            }
            for met in ['sigma','sigma_mu']:
                cpt_im_dist_info_forward[each][met] = cpt_im_dist_info[each][met].copy()
        # special mapping keys for rupture metadata
        rup_map_key = {
            'magnitude': 'mag',
            'annual_rate': 'rate'
        }
        cpt_rup_info = {}
        for key in list(cpt_rupture_table.columns):
            if key in rup_map_key:
                cpt_rup_info[rup_map_key[key]] = cpt_rupture_table[key][event_i]
            else:
                cpt_rup_info[key] = cpt_rupture_table[key][event_i]
        
        #####################################################
        # initialize displacement array
        pgdef = {}
        pgdef_forward = {}
        lateral_spread_control_case = np.empty((n_cpt, num_epi_input_samples),dtype="<U20")
        lateral_spread_control_case[:,:] = 'na' # defautl to ground slope
        # l_h_at_nodes_mapped_from_cpts = []
        for each in haz_to_run:
            pgdef[each] = null_arr_cpt_sample.copy()
            pgdef_forward[each] = null_arr_cpt_sample.copy()
        # additional processing that depends on other input params
        for i,each in enumerate(cpt_data):
            # dimensions
            n_depth = each.shape[0]
            # get params for current cpt
            qc = each.qc.values # MPa
            fs = each.fs.values # MPa
            z = each.z.values # m
            
            # get density and stresses
            tot_sig_v0, pore_press, eff_sig_v0 = \
                process_cpt_spt.get_cpt_stress(qc, fs, z, cpt_input_samples['gw_depth'][i], gamma_water=9.81)
            _, Qtn, fn, ic, _ = process_cpt_spt.get_sbt_index(qc, fs, tot_sig_v0, eff_sig_v0, pa=101.3)
            
            # additional preprocessing before calculating FS
            # additional screening for fs_liq and also get sum of thickness with FS<1
            # liq_thick = null_arr_sample.copy()
            # get layer thickness
            dz = np.hstack([z[0],np.diff(each.z.values)])
            # if above gw_depth or below z_cutoff, also set to 4
            z_repeat = z.repeat(num_epi_input_samples).reshape((-1, num_epi_input_samples))
            dz_repeat = dz.repeat(num_epi_input_samples).reshape((-1, num_epi_input_samples))
            # depth scale factor
            if depth_scale_method == 'None (ScaleFactor=1)':
                z_sf = np.ones(z_repeat.shape)
            elif depth_scale_method == 'Linear (Zhang et al., 2004)':
                z_sf = 1 - z_repeat/15
            elif depth_scale_method == 'Nonlinear (Bain and Bray, 2022)':
                z_sf = 1 - np.sinh(z_repeat/13.615)**2.5
            # reshape groundwater depth
            gw_depth_repeat = cpt_input_samples['gw_depth'][i].repeat(n_depth).reshape((-1, num_epi_input_samples))
            # get relative density
            dr = process_cpt_spt.get_cpt_dr(qc, tot_sig_v0, eff_sig_v0, pa=float(101.3))
            
            #############################
            # at current PGA
            # get weighted fs_liq and qc1ncs
            fs_liq, _ = process_cpt_spt.get_cpt_based_fs_liq(
                qc, fs, z, tot_sig_v0, eff_sig_v0,
                Qtn, fn, ic,
                pga=np.exp(cpt_im_dist_info['pga']['mean']),
                mag=cpt_rup_info['mag'],
                weight_r09=weight_r09, pa=101.3
            )
            fs_liq = np.minimum(fs_liq,4) # limit to 4
            
            # continue rest of analysis for current CPT for deformation if min(fs_liq) M 4:
            if np.min(fs_liq) < 4:
                # limit factor of safety to 4 based on cutoff depth and groundwater depth
                fs_liq[np.logical_or(z_repeat<=gw_depth_repeat,z_repeat>z_cutoff)] = 4
                # get thickness of potentially liquefiable layeres
                liq_thick = dz_repeat.copy()
                liq_thick[fs_liq>=1] = 0 # set liquefiable layer thickness to 0 if FS > 1
                # sum up to get total liquefiable thickness
                liq_thick = np.sum(liq_thick,axis=0)
                # run lateral spread
                if 'lateral_spread' in haz_to_run:
                    # get weighted maximum shear strain
                    gamma_max = process_cpt_spt.get_cpt_based_shear_strain(fs_liq.copy(), dr.copy(), weight_z04)
                    # get slope in percents
                    slope_percent = np.arctan(np.radians(cpt_input_samples['slope'][i]))*100
                    # get lateral spread index
                    ldi = np.sum(gamma_max/100*dz_repeat*z_sf,axis=0)
                    # if total liquefiable layer thickness < 0.3 meter, set LDI to 0
                    ldi[liq_thick<0.3] = 0
                    # get lateral spread displacement by ground slope, m
                    # limit ground slope displacement to between 0.2% and 3.5%
                    # for slope = 3.5 to 5%, use 3.5%
                    # for slope < 0.2% and slope > 5%, no displacement
                    slope_percent[np.logical_and(slope_percent>3.5,slope_percent<=5)] = 3.5
                    ld_ground_slope = ldi * (slope_percent + 0.2) # eq. 6, Zhang et al. (2004)
                    ld_ground_slope[np.logical_and(slope_percent<0.2,slope_percent>5)] = 0                
                    # if freeface feature is not given, set PGDef to ground slope displacement
                    if freeface_fpath is None:
                        pgdef['lateral_spread'][i] = ld_ground_slope
                        lateral_spread_control_case[i] = 'ground_slope'
                    else:
                        lh_ratio_curr_cpt = cpt_meta_wgs84['LH_Ratio'].values[i]
                        # if no L/H ratio is less than 50, then skip calculations
                        if lh_ratio_curr_cpt > 50:
                            pgdef['lateral_spread'][i] = ld_ground_slope
                            lateral_spread_control_case[i] = 'ground_slope'
                        else:
                            # get freeface displacement and use max of ground slope and freeface disp
                            ld_freeface = 6*(lh_ratio_curr_cpt**(-0.8)) * ldi
                            # set to PGDef                        
                            pgdef['lateral_spread'][i] = np.maximum(ld_ground_slope,ld_freeface)
                            # track controlling condition for lateral spread: freeface or ground slope
                            where_freeface_pgdef_higher = np.where(ld_freeface>ld_ground_slope)[0]
                            lateral_spread_control_case[i][where_freeface_pgdef_higher] = 'freeface'
                            where_ground_slope_pgdef_higher = np.where(ld_ground_slope>ld_freeface)[0]
                            lateral_spread_control_case[i][where_ground_slope_pgdef_higher] = 'ground_slope'
                if 'settlement' in haz_to_run:
                    # get volumetric strain
                    eps_vol = process_cpt_spt.get_cpt_based_vol_strain(fs_liq.copy(), dr.copy(), weight_z04)
                    # get settlement, m
                    pgdef['settlement'][i] = np.sum(eps_vol/100*dz_repeat*z_sf,axis=0)
                    # if total liquefiable layer thickness < 0.3 meter, set deformation to 0
                    pgdef['settlement'][i][liq_thick<0.3] = 0
            
            ######
            # at forward PGA (for getting derivatives)
            # get weighted fs_liq and qc1ncs
            fs_liq_forward, _ = process_cpt_spt.get_cpt_based_fs_liq(
                qc, fs, z, tot_sig_v0, eff_sig_v0,
                Qtn, fn, ic,
                pga=np.exp(cpt_im_dist_info_forward['pga']['mean']),
                mag=cpt_rup_info['mag'],
                weight_r09=weight_r09, pa=101.3
            )
            fs_liq_forward = np.minimum(fs_liq_forward,4) # limit to 4
            # continue rest of analysis for current CPT for deformation if min(fs_liq) M 4:
            if np.min(fs_liq_forward) < 4:
                # limit factor of safety to 4 based on cutoff depth and groundwater depth
                fs_liq_forward[np.logical_or(z_repeat<=gw_depth_repeat,z_repeat>z_cutoff)] = 4
                # get thickness of potentially liquefiable layeres
                liq_thick = dz_repeat.copy()
                liq_thick[fs_liq_forward>=1] = 0 # set liquefiable layer thickness to 0 if FS > 1
                # sum up to get total liquefiable thickness
                liq_thick = np.sum(liq_thick,axis=0)
                # run lateral spread
                if 'lateral_spread' in haz_to_run:
                    # get weighted maximum shear strain
                    gamma_max = process_cpt_spt.get_cpt_based_shear_strain(fs_liq_forward.copy(), dr.copy(), weight_z04)
                    # get slope in percents
                    slope_percent = np.arctan(np.radians(cpt_input_samples['slope'][i]))*100
                    # get lateral spread index
                    ldi = np.sum(gamma_max/100*dz_repeat*z_sf,axis=0)
                    # if total liquefiable layer thickness < 0.3 meter, set LDI to 0
                    ldi[liq_thick<0.3] = 0
                    # get lateral spread displacement by ground slope, m
                    # limit ground slope displacement to between 0.2% and 3.5%
                    # for slope = 3.5 to 5%, use 3.5%
                    # for slope < 0.2% and slope > 5%, no displacement
                    slope_percent[np.logical_and(slope_percent>3.5,slope_percent<=5)] = 3.5
                    ld_ground_slope = ldi * (slope_percent + 0.2) # eq. 6, Zhang et al. (2004)
                    ld_ground_slope[np.logical_and(slope_percent<0.2,slope_percent>5)] = 0                
                    # if freeface feature is not given, set PGDef to ground slope displacement
                    if freeface_fpath is None:
                        pgdef_forward['lateral_spread'][i] = ld_ground_slope
                        lateral_spread_control_case[i] = 'ground_slope'
                    else:
                        lh_ratio_curr_cpt = cpt_meta_wgs84['LH_Ratio'].values[i]
                        # if no L/H ratio is less than 50, then skip calculations
                        if lh_ratio_curr_cpt > 50:
                            pgdef_forward['lateral_spread'][i] = ld_ground_slope
                        else:
                            # get freeface displacement and use max of ground slope and freeface disp
                            ld_freeface = 6*(lh_ratio_curr_cpt**(-0.8)) * ldi
                            # set to PGDef                        
                            pgdef_forward['lateral_spread'][i] = np.maximum(ld_ground_slope,ld_freeface)
                if 'settlement' in haz_to_run:
                    # get volumetric strain
                    eps_vol = process_cpt_spt.get_cpt_based_vol_strain(fs_liq_forward.copy(), dr.copy(), weight_z04)
                    # get settlement, m
                    pgdef_forward['settlement'][i] = np.sum(eps_vol/100*dz_repeat*z_sf,axis=0)
                    # if total liquefiable layer thickness < 0.3 meter, set deformation to 0
                    pgdef_forward['settlement'][i][liq_thick<0.3] = 0
        
        
        # go through each liquefaction consequence
        mean_pgdef_per_node_over_samples = {}
        mean_pgdef_forward_per_node_over_samples = {}
        for each in pgdef:
            # processing on LDs
            # get mean LD for each node
            if weight_scheme == 'average':
                # average deformation on node - current PGA
                mean_pgdef_per_node = [
                    np.ma.masked_invalid(pgdef[each][ind,:]).mean(axis=0)
                    # pgdef[each][ind,:].mean(axis=0)
                    if len(ind)>0 else np.zeros(num_epi_input_samples)
                    for i,ind in enumerate(cpts_per_node)
                ]
                # average deformation on node - target PGA
                mean_pgdef_forward_per_node = [
                    np.ma.masked_invalid(pgdef_forward[each][ind,:]).mean(axis=0)
                    # pgdef_forward[each][ind,:].mean(axis=0)
                    if len(ind)>0 else np.zeros(num_epi_input_samples)
                    for i,ind in enumerate(cpts_per_node)
                ]
            # elif weight_scheme == 'distance':
            #     # determine distance between each node to the CPTs it is tied to
            #     node_to_cpt_dists = [
            #         [gdf_nodes_utm[i].distance(cpt_meta_utm.geometry[each_cpt]) for each_cpt in ind]
            #         for i,ind in enumerate(cpts_per_node)
            #     ]
            #     # distance weighted deformation on node
            #     mean_pgdef_per_node = []
            #     for i,ind in enumerate(cpts_per_node):
            #         mean_pgdef_i = np.zeros((num_epi_input_samples))
            #         if len(ind)>0:
            #             ld_for_ind = pgdef[each][ind,:]
            #             dist_for_ind = np.asarray(node_to_cpt_dists[i])
            #             sum_inv_sqrt_dist = np.sum(1/dist_for_ind**2)
            #             for j in range(len(ind)):
            #                 mean_pgdef_i += ld_for_ind[j,:]/node_to_cpt_dists[i][j]**2
            #             mean_pgdef_i = mean_pgdef_i / sum_inv_sqrt_dist
            #         mean_pgdef_per_node.append(mean_pgdef_i)
            # get mean LD over all samples
            mean_pgdef_per_node_over_samples[each] = \
                np.ma.masked_invalid(mean_pgdef_per_node).mean(axis=1)
            # mean_pgdef_per_node_over_samples[each] = np.mean(mean_pgdef_per_node,axis=1)
            mean_pgdef_forward_per_node_over_samples[each] = \
                np.ma.masked_invalid(mean_pgdef_forward_per_node).mean(axis=1)
            # mean_pgdef_forward_per_node_over_samples[each] = np.mean(mean_pgdef_forward_per_node,axis=1)
            
        # collect all controlling cases for lateral spread and find most governing (most frequent)
        if 'lateral_spread' in haz_to_run:
            lateral_spread_control_case_per_node = [
                list(lateral_spread_control_case[ind].flatten())
                if len(ind)>0 else []
                for i,ind in enumerate(cpts_per_node)
            ]
            
        # estimate d(pgdef)/d(pga) at each node
        tangent_intercept = {}
        tangent_slope = {}
        amu = {}
        bmu = {}
        # get PGAs        
        curr_ln_pga = cpt_im_dist_info['pga']['mean']
        curr_ln_pga_forward = cpt_im_dist_info_forward['pga']['mean']
        tangent_slope_denom = curr_ln_pga_forward - curr_ln_pga
        # for each hazard
        for each in mean_pgdef_per_node_over_samples:
            # pull values out
            pgdef_at_curr_pga = mean_pgdef_per_node_over_samples[each].copy()
            pgdef_at_forward_pga = mean_pgdef_forward_per_node_over_samples[each].copy()
            # set pgdef == 0 to 1e-5 to avoid ln(0)
            pgdef_at_curr_pga[pgdef_at_curr_pga==0] = 1e-5
            pgdef_at_forward_pga[pgdef_at_forward_pga==0] = 1e-5
            # calculate and store slope
            tangent_slope[each] = (np.log(pgdef_at_forward_pga) - np.log(pgdef_at_curr_pga))/tangent_slope_denom
            # store intercept
            tangent_intercept[each] = np.log(pgdef_at_curr_pga)
            amu[each] = tangent_slope[each].copy()
            bmu[each] = tangent_slope[each] * (-pgdef_at_curr_pga) + tangent_intercept[each]

        # perform the following to get hull (i.e., deformation polygon):
        # 1) gdf of points for current sample given ld_cutoff
        # 2) average aspect for points within hull
        # 3) average ld for points within hull
        # 4) convex hull polygon
        for each in pgdef:
            if not each in gdf_poly_hull:
                # initialize gdf for deformation polygon
                gdf_poly_hull[each] = GeoDataFrame(
                    None,
                    crs=4326,
                    columns=['FID','event_id','slip_dir','pgdef_m','geometry']
                )
            # 1) gdf of points with mean deformation
            preproc_hull = process_cpt_spt.preprocess_for_hull(
                mean_pgdef_per_node_over_samples[each].copy(),
                grid_locs.data,
                aspect_store_name,
                pgdef_cutoff
            )
            # if len(preproc_hull) == 0, set geometry to None
            if len(preproc_hull) == 0:
                # create shapefile of lateral spread using hull
                gdf_poly_hull[each].loc[gdf_poly_hull[each].shape[0]] = [
                    event_count,
                    int(event_id),
                    0,
                    0, # mm
                    None
                ]
                amu_by_event.append(0)
                bmu_by_event.append(0)
                if each == 'lateral_spread':
                    lateral_spread_control_case_by_event.append('')
            else:
                # 2) average aspect for points within hull
                avg_aspect = np.nanmean(preproc_hull.aspect)
                # avg_aspect = preproc_hull.aspect.mean()
                # 3) average deformation for points within hull
                avg_pgdef = np.nanmean(preproc_hull.pgdef_m)
                # avg_pgdef = preproc_hull.pgdef_m.mean()
                # 4) convex hull polygon
                poly = preproc_hull.unary_union.convex_hull
                poly_hull = poly.buffer(buffer_size)
                # create shapefile of lateral spread using hull
                gdf_poly_hull[each].loc[gdf_poly_hull[each].shape[0]] = [
                    event_count,
                    int(event_id),
                    np.round(avg_aspect,decimals=1),
                    np.round(avg_pgdef,decimals=4), # mm
                    poly_hull
                ]
                # for each deformation polygon, find all grid nodes within in
                intersect_node = gdf_nodes_utm.to_crs(epsg_wgs84).sindex.query(
                    # geometry=list(def_poly_gdf.geometry.boundary),
                    geometry=gdf_poly_hull[each].geometry.values[-1],
                    predicate='intersects'
                )
                # get average amu and bmu for nodes in hull
                amu_intersect_node = amu[each][intersect_node].copy()
                bmu_intersect_node = bmu[each][intersect_node].copy()
                where_bmu_nonzero = np.where(bmu_intersect_node>-10)[0]
                amu_by_event.append(np.ma.masked_invalid(amu_intersect_node[where_bmu_nonzero]).mean())
                # amu_by_event.append(np.mean(amu_intersect_node[where_bmu_nonzero]))
                bmu_by_event.append(np.ma.masked_invalid(bmu_intersect_node[where_bmu_nonzero]).mean())
                # bmu_by_event.append(np.mean(bmu_intersect_node[where_bmu_nonzero]))
                # find controlling lateral spread case 
                if each == 'lateral_spread':
                    all_lateraL_spread_control_cases = [
                        case
                        for node_ind in intersect_node
                        for case in lateral_spread_control_case_per_node[node_ind]
                    ]
                    unique, pos = np.unique(all_lateraL_spread_control_cases,return_inverse=True)
                    counts = np.bincount(pos)
                    maxpos = counts.argmax()
                    lateral_spread_control_case_by_event.append(unique[maxpos])           
                
        # update counter
        event_count += 1

        # for displaying progress in analysis
        if (event_i+1)%display_after_n_event == 0:
            logging.info(f"\t\t- after {event_i+1} events: {np.round(time.time()-time_init,decimals=2)} sec")
            time_init=time.time()
    logging.info(f'\t- Finished calculation for all {cpt_rupture_table.shape[0]} events')
    
    # -----------------------------------------------------------
    # some cleanup
    for each in gdf_poly_hull:
        # only keep rows where pgdef > 0
        rows_with_zero_pgdef = np.where(gdf_poly_hull[each]['pgdef_m'].values>0)[0]
        gdf_poly_hull[each] = gdf_poly_hull[each].loc[rows_with_zero_pgdef].reset_index(drop=True)
        # assign sigmas and sigma mus
        gdf_poly_hull[each]['sigma'] = 0.6
        gdf_poly_hull[each]['sigma_mu'] = 0.25
        gdf_poly_hull[each]['dist_type'] = 'lognormal'
        # after removing rows with pgdef == 0, reset FID = index + 1
        gdf_poly_hull[each]['FID'] = np.arange(gdf_poly_hull[each].shape[0])
        # make sure event_id is int
        gdf_poly_hull[each].FID = gdf_poly_hull[each].FID.values.astype(int)
        gdf_poly_hull[each].slip_dir = gdf_poly_hull[each].slip_dir.round(decimals=1)
        # append controlling lateral spread case and average l/h for free face controlled cases
        if each == 'lateral_spread':
            lateral_spread_control_case_by_event = np.asarray(lateral_spread_control_case_by_event)
            gdf_poly_hull[each]['ls_cond'] = lateral_spread_control_case_by_event[rows_with_zero_pgdef]
        # append amu and bmu estimates
        gdf_poly_hull[each]['amu'] = np.asarray(amu_by_event)[rows_with_zero_pgdef]
        gdf_poly_hull[each]['bmu'] = np.asarray(bmu_by_event)[rows_with_zero_pgdef]
    # define schema in case of empty geodataframe
    schema = {
        "geometry": "Polygon",
        "properties": {
            "FID": "int",
            "slip_dir": "float",
            "pgdef_m": "float",
        }
    }
    # export to processed CPT dir
    logging.info(f'- Exported processed CPT deformation polygons to:')
    spath_def_poly = []
    for each in gdf_poly_hull:
        spath_def_poly.append(os.path.join(processed_cpt_base_dir,f'cpt_based_deformation_{each}.shp'))
        if os.path.exists(spath_def_poly[-1]):
            os.remove(spath_def_poly[-1])
        logging.info(f'\t- {spath_def_poly[-1]}')
        if gdf_poly_hull[each].shape[0] == 0:
            gdf_poly_hull[each].to_file(
                spath_def_poly[-1], schema=schema, crs=epsg_wgs84, layer='data')
            # also export to csv, without geometry
            gdf_poly_hull[each].drop(columns=['geometry']).to_csv(
                spath_def_poly[-1].replace('.shp','.csv'),
                index=False)
            logging.info(f'\t\t- file is empty')
        else:
            gdf_poly_hull[each].to_file(
                spath_def_poly[-1], crs=epsg_wgs84, layer='data')
            gdf_poly_hull[each].drop(columns=['geometry']).to_csv(
                spath_def_poly[-1].replace('.shp','.csv'),
                index=False)
    # updated CPT summary spreadsheet
    spath_cpt_meta = os.path.join(processed_cpt_base_dir,'cpt_data_PROCESSED.csv')
    cpt_meta_wgs84.drop('geometry',axis=1).to_csv(spath_cpt_meta,index=False)
    # export to gpkg
    cpt_meta_wgs84_copy = cpt_meta_wgs84.copy()
    if 'UTM_x' in cpt_meta_wgs84:
        cpt_meta_wgs84_copy.drop(columns=['UTM_x'],inplace=True)
    if 'UTM_y' in cpt_meta_wgs84:
        cpt_meta_wgs84_copy.drop(columns=['UTM_y'],inplace=True)
    cpt_meta_wgs84_copy.to_file(spath_cpt_meta.replace('.csv','.gpkg'), crs=epsg_wgs84, layer='data')
    logging.info(f'- Exported processed CPT summary spreadsheet to:')
    logging.info(f'\t- {spath_cpt_meta}')
    logging.info(f"\t- {spath_cpt_meta.replace('.csv','.gpkg')}")
    
    # -----------------------------------------------------------
    # return
    return spath_def_poly, freeface_fpath

# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Preprocess methods for OpenSRA'
    )
    
    # Define arguments
    
    # analysis directory
    parser.add_argument('-w', '--workdir', help='Path to working/analysis directory')
    
    # json with workflow
    # parser.add_argument('-w', '--workflow', help='Path to json file with workflow')
    
    # infrastructure type
    # parser.add_argument('-i', '--infra_type', help='Infrastructure type',
    #                     default='below_ground', type=str)
    
    # infrastructure file type
    # parser.add_argument('-f', '--file_path', help='Infrastructure file path',
    #                     default='shp', type=str)
    
    # infrastructure file type
    parser.add_argument('-l', '--logging',
                        help='Logging message detail: "s" for simple or "d" for detailed',
                        default='s', type=str)
    
    # infrastructure file type
    parser.add_argument('-d', '--display',
                        help='Display a message every n scenarios (CPT processing only)',
                        default=100, type=int)
    
    # infrastructure file type
    parser.add_argument('-c', '--clean',
                        help='Clean "IM" and "Processed_Input" directories from previous preprocessing if exists',
                        default=True, type=bool)
    
    # Parse command line input
    args = parser.parse_args()
    
    # Run "Main"
    main(
        work_dir = args.workdir,
        # infra_fpath = args.file_type,
        logging_message_detail=args.logging,
        display_after_n_event=args.display,
        clean_prev_output=args.clean,
    )
