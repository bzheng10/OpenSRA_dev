# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Run risk calculations
#
# Created: December 10, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
#
# Instructions
# ------------
# Currently, the program should be run in four phases:
# 1. Get rupture scenarios, and fault-crossing and geologic units and properties at sites
# 2. Get ground motion predictions at target sites given rupture scenarios
# 3. Generate and store intensity measure realizations
# 4. Assess EDPs, DMs, DVs using IM realizations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Python base modules
import argparse
import json
import logging
import os
import shutil
import sys
import time

# scientific processing modules
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import truncnorm, norm

# for geospatial processing
from geopandas import GeoDataFrame, points_from_xy, read_file
from shapely.geometry import Polygon

# precompiling
# from numba_stats import truncnorm as nb_truncnorm
# from numba_stats import norm as nb_norm

# OpenSRA modules and functions
from src import numba_cache_config
from src.site.site_util import make_list_of_linestrings, get_regional_liq_susc
from src.pc_func import pc_util, pc_workflow
from src.pc_func.pc_coeffs_single_int import pc_coeffs_single_int
from src.pc_func.pc_coeffs_double_int import pc_coeffs_double_int
from src.pc_func.pc_coeffs_triple_int import pc_coeffs_triple_int
from src.util import set_logging, lhs, get_cdf_given_pts, check_and_get_abspath, remap_str


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main function
def main(work_dir, logging_level='info', logging_message_detail='s',
         display_after_n_event=100, clean_prev_output=True, get_timer=False):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setting logging level (e.g. DEBUG or INFO)
    set_logging(
        level=logging_level,
        msg_format=logging_message_detail
    )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logging.info('---------------')
    logging.info('******** Start of OpenSRA Analysis ********')
    logging.info('---------------')
    counter = 1 # counter for stages in analysis

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define primary directories - these should be created during Preprocess
    input_dir = os.path.join(work_dir,'Input')
    processed_input_dir = os.path.join(work_dir,'Processed_Input')
    im_dir = os.path.join(work_dir,'IM')
    if os.path.exists(input_dir) is False:
        logging.info('URGENT: Missing input directory; create the folder and run "Preprocess".')
        logging.info(f'\t- OpenSRA will now exit.')
        sys.exit()
    if os.path.exists(processed_input_dir) is False or os.path.exists(im_dir) is False:
        logging.info('URGENT: Missing generated directories; first run "Preprocess".')
        logging.info(f'\t- OpenSRA will now exit.')
        sys.exit()
    logging.info(f'{counter}. Identified primary directories')
    counter += 1
    logging.info('\tWorking directory given:')
    logging.info(f'\t\t- {work_dir}')
    logging.info('\tInput directory implied:')
    logging.info(f'\t\t- {input_dir}')
    logging.info('\tProcessed input directory for export of processed information:')
    logging.info(f'\t\t- {processed_input_dir}')
    logging.info('\tIntensity measure directory:')
    logging.info(f'\t\t- {im_dir}')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import site data
    logging.info(f'{counter}. Loading site data file...')
    counter += 1
    # check for files with crossing
    if 'site_data_PROCESSED_CROSSING_ONLY.csv' in os.listdir(processed_input_dir):
        site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED_CROSSING_ONLY.csv'))
        site_data_full = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED.csv'))
        flag_crossing_file_exists = True
    else:
        site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED.csv'))
        flag_crossing_file_exists = False
    logging.info(f'... DONE - Loaded site data file')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Seismic Events
    logging.info(f'{counter}. Loading seismic event files (e.g., rupture scenarios, ground motions)')
    counter += 1
    # Import IM distributions
    im_import = {}
    for each in ['pga','pgv']:
        im_import[each] = {
            # note that sparse matrix files require -10 to all values for correct magnitude;
            # during ground motion phase, locations with 0 intensities are reported as -10 instead of -np.inf;
            # for storage, in order to be memory efficient, ground motions are stored as sparse matrx by adding
            # 10 to all values, thus convert the -10 intensity magnitudes to 0;
            # ---> this means when using the sparse datafiles, they must be -10 to get the correct magnitude
            # 'mean_table': np.round(sparse.load_npz(os.path.join(im_dir,each.upper(),'MEAN.npz')).toarray()-10,decimals=3),
            # 'sigma_table': np.round(sparse.load_npz(os.path.join(im_dir,each.upper(),'ALEATORY.npz')).toarray(),decimals=3),
            # 'sigma_mu_table': np.round(sparse.load_npz(os.path.join(im_dir,each.upper(),'EPISTEMIC.npz')).toarray(),decimals=3)
            'mean_table': sparse.load_npz(os.path.join(im_dir,each.upper(),'MEAN.npz')).toarray(),
            'sigma_table': sparse.load_npz(os.path.join(im_dir,each.upper(),'ALEATORY.npz')).toarray(),
            'sigma_mu_table': sparse.load_npz(os.path.join(im_dir,each.upper(),'EPISTEMIC.npz')).toarray()
        }
    # Import rupture information
    rupture_table = pd.read_csv(os.path.join(im_dir,'RUPTURE_METADATA.csv'))
    rupture_table.event_id = rupture_table.event_id.astype(int) # set as integers if not already
    event_ids_to_run = rupture_table.event_id.values # for looping through events to run
    event_ind_relative_to_rupture_table = rupture_table.index.values.astype(int)
    # Get Number of sites
    n_site = im_import[list(im_import)[0]]['mean_table'].shape[1]
    # Make some arrays to be used later
    n_site_ind_arr = np.arange(n_site)
    null_arr_nsite = np.zeros(n_site)
    ones_arr_nsite = np.ones(n_site)
    logging.info(f'... DONE - Loaded seismic event files')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load setup configuration file
    setup_config_file = os.path.join(input_dir,'SetupConfig.json')
    with open(setup_config_file, 'r') as f:
        setup_config = json.load(f)
    infra_type = setup_config['Infrastructure']['InfrastructureType']
    im_source = im_source = list(setup_config['IntensityMeasure']['SourceForIM'])[0]
    logging.info(f'{counter}. Loaded setup configuration file')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load and process workflow
    workflow_file = os.path.join(processed_input_dir,'workflow.json')
    if os.path.exists(workflow_file):
        with open(workflow_file, 'r') as f:
            workflow = json.load(f)
    # Process workflow
    methods_dict, additional_params = pc_workflow.prepare_methods(workflow, n_site)
    workflow_order_list = pc_workflow.get_workflow_order_list(methods_dict, infra_type=infra_type)
    logging.info(f'{counter}. Loaded and processed workflow')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # standard epsgs
    epsg_wgs84 = 4326 # lat lon, deg
    epsg_utm_zone10 = 32610 # m

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if CPT is needed - make toggle if running CPT-based procedure
    running_cpt_based_procedure = False
    if "edp" in methods_dict and \
        "liquefaction" in methods_dict['edp'] and \
        "CPTBased" in methods_dict['edp']['liquefaction']['method']:
        running_cpt_based_procedure = True
        # get preprocessed CPT directories
        processed_cpt_base_dir = os.path.join(processed_input_dir,'CPTs')
        processed_cpt_im_dir = os.path.join(im_dir,'CPTs')
        # get preprocessed CPT files
        for each in ['lateral_spread','settlement']:
            if each in methods_dict['edp']:
                cpt_pgdef_dist = pd.read_csv(os.path.join(processed_cpt_base_dir,f'cpt_based_deformation_{each}.csv'))
                cpt_pgdef_dist.event_id = cpt_pgdef_dist.event_id.astype(int)
                break
        processed_cpt_metadata = pd.read_csv(os.path.join(processed_cpt_base_dir,'cpt_data_PROCESSED.csv'))
        logging.info(f'{counter}. Flagged OpenSRA to run CPT procedure and loaded preprocessed CPT files')
        counter += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get various flags
    # check infrastructure type to run
    running_below_ground = False
    running_wells_caprocks = False
    running_above_ground = False
    if infra_type == 'below_ground':
        running_below_ground = True
    elif infra_type == 'wells_caprocks':
        running_wells_caprocks = True
    elif infra_type == 'above_ground':
        running_above_ground = True
        
    # check PBEE categories
    has_edp = False
    has_dm = False
    has_dv = False
    if 'EDP' in workflow:
        has_edp = True
    if 'DM' in workflow:
        has_dm = True
    if 'DV' in workflow:
        has_dv = True
    
    # Check geohazard to run
    running_below_ground_fault_rupture = False
    running_below_ground_landslide = False
    running_below_ground_liquefaction = False
    running_below_ground_lateral_spread = False
    running_below_ground_settlement = False
    if running_below_ground and has_edp:
        if 'surface_fault_rupture' in workflow['EDP']:
            running_below_ground_fault_rupture = True
        if 'landslide' in workflow['EDP']:
            running_below_ground_landslide = True
        if 'liquefaction' in workflow['EDP']:
            running_below_ground_liquefaction = True
        if 'lateral_spread' in workflow['EDP']:
            running_below_ground_lateral_spread = True
        if 'settlement' in workflow['EDP']:
            running_below_ground_settlement = True
        
    # Check if running caprocks for wells_caprocks, requires special processing
    running_caprock = False
    if 'caprock_leakage' in workflow['DV']:
        running_caprock = True
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for analysis with crossings with deformation polygons
    crossing_event_dependency = {
        # geohazard: analysis levels in list
        'lateral_spread': [3],
        'surface_fault_rupture': [1,2,3],
    }
    flag_possible_repeated_crossings = False # default
    # additional processing depending if the deformation polygons are event dependent or not
    flag_event_dependent_crossing = False
    if running_cpt_based_procedure:
        flag_event_dependent_crossing = True
    if running_below_ground_fault_rupture:
        flag_event_dependent_crossing = True
    # if crossing file exists
    if flag_crossing_file_exists:
        # get probability of crossing
        prob_crossing = site_data.prob_crossing.values
        # if prob crossing == 1, then using deformation polygons and multiple crossings per segment is possible
        # if prob crossing == 0.25, then no geometry was used and only 1 crossing per segment
        if prob_crossing[0] == 1 or running_below_ground_fault_rupture:
            flag_possible_repeated_crossings = True
        # get segment IDs
        segment_ids_full = site_data_full.ID.values
        # get table indices corresponding to IDs
        segment_index_full = site_data_full.index.values
        segment_index_crossed = site_data.index.values
        # get repeated index
        if flag_possible_repeated_crossings:
            # if deformation polygons are event specific
            if flag_event_dependent_crossing:
                # initialize
                segment_ids_crossed_by_event_id = {}
                rows_to_run_by_event_id = {}
                segment_ids_crossed_repeat_by_event_id = {}
                segment_index_repeat_in_full_by_event_id = {}
                segment_index_single_in_full_by_event_id = {}
                # different ref column for event index for CPT vs fault rupture
                if running_cpt_based_procedure:
                    event_ind_col = 'def_poly_index_crossed'
                elif running_below_ground_fault_rupture:
                    event_ind_col = 'event_ind'
                # for CPT-informed deformation polygons, polygon ID is the event ID
                # check to see if multiple crossings per segment for the same polygon ID
                unique_event_index_with_crossing = np.unique(site_data[event_ind_col].values)
                if running_cpt_based_procedure:
                    unique_event_id_with_crossing = cpt_pgdef_dist.event_id.loc[unique_event_index_with_crossing].values
                elif running_below_ground_fault_rupture:
                    unique_event_id_with_crossing = np.unique(site_data.event_id.values)
                # for each unique event, see if there are repeating crossings
                for i,event_ind in enumerate(unique_event_index_with_crossing):
                    # get current event id
                    if running_cpt_based_procedure:
                        curr_event_id = unique_event_id_with_crossing[i]
                    elif running_below_ground_fault_rupture:
                        curr_event_id = site_data.event_id.values[np.where(site_data.event_ind==event_ind)[0][0]]
                    # get rows relative to site_data for segments with crossings for current event
                    rows_to_run_by_event_id[curr_event_id] = np.where(site_data[event_ind_col]==event_ind)[0]
                    # store subset of site_data for current event
                    # site_data_by_event_id[curr_event_id] = site_data.loc[rows_with_segments_for_event_ind].copy().reset_index(drop=True)
                    # get all segments with crossings for current event
                    segment_ids_curr_event = site_data.ID.loc[rows_to_run_by_event_id[curr_event_id]].values
                    segment_ids_crossed_by_event_id[curr_event_id] = segment_ids_curr_event
                    # from segments list above, find repeating segments if any
                    segment_ids_crossed_unique_curr_event, counts = \
                        np.unique(segment_ids_curr_event, return_counts=True)
                    segment_ids_crossed_repeat_curr_event = segment_ids_crossed_unique_curr_event[np.where(counts>1)[0]]
                    segment_ids_crossed_single_curr_event = np.asarray(
                        list(set(segment_ids_crossed_unique_curr_event).difference(set(segment_ids_crossed_repeat_curr_event))))
                    segment_ids_crossed_single_curr_event = np.sort(segment_ids_crossed_single_curr_event)
                    # find row index corresponding to repeated IDS in full table
                    segment_index_repeat_in_full_curr_event = np.asarray([
                        np.where(segment_ids_full==seg_id)[0][0]
                        for seg_id in segment_ids_crossed_repeat_curr_event
                    ])
                    segment_index_single_in_full_curr_event = np.asarray([
                        np.where(segment_ids_full==seg_id)[0][0]
                        for seg_id in segment_ids_crossed_single_curr_event
                    ])
                    # store to dictionary
                    segment_ids_crossed_repeat_by_event_id[curr_event_id] = segment_ids_crossed_repeat_curr_event
                    segment_index_repeat_in_full_by_event_id[curr_event_id] = segment_index_repeat_in_full_curr_event
                    segment_index_single_in_full_by_event_id[curr_event_id] = segment_index_single_in_full_curr_event
                # update events to run, only those with crossings
                event_ids_to_run = unique_event_id_with_crossing # update to this list of event ids
                event_ind_relative_to_rupture_table = np.asarray([
                    np.where(rupture_table.event_id==event_id)[0][0]
                    for event_id in unique_event_id_with_crossing
                ])
            # otherwise
            else:
                segment_ids_crossed = site_data.ID.values
                # get unique crossings
                segment_ids_crossed_unique, counts = np.unique(segment_ids_crossed, return_counts=True)
                segment_ids_crossed_repeat = segment_ids_crossed_unique[np.where(counts>1)[0]]
                # segments with only 1 crossing
                segment_ids_crossed_single = np.asarray(
                    list(set(segment_ids_crossed_unique).difference(set(segment_ids_crossed_repeat))))
                segment_ids_crossed_single = np.sort(segment_ids_crossed_single)
                # find row index corresponding to repeated IDS in full table
                segment_index_repeat_in_full = np.asarray([
                    np.where(segment_ids_full==seg_id)[0][0]
                    for seg_id in segment_ids_crossed_repeat
                ])
                segment_index_single_in_full = np.asarray([
                    np.where(segment_ids_full==seg_id)[0][0]
                    for seg_id in segment_ids_crossed_single
                ])
        else:
            # no possibility of segments with repeated crossings
            segment_ids_crossed = site_data.ID.values
            segment_ids_crossed_repeat = np.asarray([])
            segment_index_repeat_in_full = np.asarray([])
            segment_ids_crossed_single = segment_ids_crossed.copy()
            segment_index_single_in_full = segment_index_crossed.copy()
        logging.info(f'{counter}. Identified indices between full instrastructure inventory and subset of inventory with crossings')
        counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get list of input params
    cat_params = {}
    all_params = []
    for cat in methods_dict:
        cat_params[cat] = [
            methods_dict[cat.lower()][haz]['input_params']
            for haz in methods_dict[cat.lower()]
        ]
        if len(cat_params[cat]) > 0:
            cat_params[cat] = list(np.unique(np.hstack(cat_params[cat])))
        all_params += cat_params[cat]
    all_params = list(set(all_params))
    logging.info(f'{counter}. Loaded required parameters for analysis')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input param distributions
    # Load input param distr table
    input_table = pd.read_hdf(os.path.join(work_dir,'Processed_Input','param_dist.h5'),'table')
    input_cols = input_table.columns
    # Get input_distributions
    input_dist = {}
    missing_params = all_params.copy()
    for param in all_params:
        if f'{param}_mean' in input_cols:
            input_dist[param] = {
                'mean': input_table[f'{param}_mean'].values,
                'sigma': input_table[f'{param}_sigma'].values,
                'low': input_table[f'{param}_low'].values,
                'high': input_table[f'{param}_high'].values,
                'dist_type': input_table[f'{param}_dist_type'].values[0],
            }
            missing_params.remove(param)
        else:
            if param in input_cols:
                input_dist[param] = {
                    'value': input_table[param].values,
                    'dist_type': 'fixed',
                }
                missing_params.remove(param)
    # add in single-valued parameters loaded through setup_config
    for param in additional_params:
        input_dist[param] = {
            'value': np.asarray([additional_params[param]]*n_site),
            'dist_type': 'fixed',
        }
    # get level to run
    if 'level_to_run' in input_table.columns:
        level_to_run = input_table.level_to_run[0]
    else:
        level_to_run = 3
    input_dist = pc_workflow.clean_up_input_params(input_dist)
    logging.info(f'{counter}. Loaded input parameter distributions')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize params for analysis
    n_cases = len(workflow_order_list)
    last_cdf_integrated = {}
    # Forward Euler differentiation
    forward_euler_multiplier = 1.01
    # Number of Epistemic samples for inputs
    if flag_possible_repeated_crossings:
        num_epi_input_samples = 1000
    else:
        num_epi_input_samples = 100
    # Number of Epistemic samples for fractiles
    num_epi_fractile_samples = 1000
    # Make some arrays to be used later
    null_arr_nsite_by_ninput = np.zeros((n_site,num_epi_input_samples))
    ones_arr_nsite_by_ninput = np.ones((n_site,num_epi_input_samples))
    twos_arr_nsite_by_ninput = ones_arr_nsite_by_ninput.copy() * 2
    str_arr_nsite_by_ninput = np.empty((n_site,num_epi_input_samples),dtype='<U40')
    ones_arr_nfractile_sample = np.ones(num_epi_fractile_samples)
    logging.info(f'{counter}. Initialized analysis metrics')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get samples for input parameters
    logging.info(f'{counter}. Performing sampling of input parameters that are not event-dependent...')
    counter += 1
    n_params = len(input_dist)
    param_names = list(input_dist)
    input_samples = \
        pc_workflow.get_samples_for_params(input_dist, num_epi_input_samples, n_site)
    
    # Additional sampling of inputs with more complex/dependent conditions
    logging.info(f'\t Performing additional procedures for other input parameters with more complex/dependent conditions...')
    # for angles that are continuous rotationally (e.g., -181 = 179) but is capped by limits in models
    if running_wells_caprocks:
        if 'theta' in input_samples:
            # target range = 0 to 90 degrees, but
            # distribution limits are extended to -90 and 180 to allow - and >90 values
            cond = input_samples['theta']<0
            if True in cond:
                input_samples['theta'][cond] = np.abs(input_samples['theta'][cond])
            cond = input_samples['theta']>90
            if True in cond:
                input_samples['theta'][cond] = np.abs(90-input_samples['theta'][cond])
            logging.info(f'\t\t- added "theta" to input samples for {infra_type} analysis')
    # for beta_crossing in below ground
    if running_below_ground:
        if 'beta_crossing' in input_samples:
            # target range = 0 to 180 degrees, but
            # distribution limits are extended to -180 and 360 to allow - and >180 values
            cond = input_samples['beta_crossing']<0
            if True in cond:
                input_samples['beta_crossing'][cond] = np.abs(input_samples['beta_crossing'][cond])
            cond = input_samples['beta_crossing']>180
            if True in cond:
                input_samples['beta_crossing'][cond] = np.abs(180-input_samples['beta_crossing'][cond])
            logging.info(f'\t\t- added "beta_crossing" to input samples for {infra_type} analysis')
        else:
            input_samples['beta_crossing'] = 0*ones_arr_nsite_by_ninput # dummy - not used but to carry on with analysis
        if 'theta_rake' in input_samples:
            # target range = -180 to 180 degrees, but
            # distribution limits are extended to -360 and 360
            cond = input_samples['theta_rake']<-180
            if True in cond:
                # e.g., -185 -> 175
                input_samples['theta_rake'][cond] = input_samples['theta_rake'][cond]+360
            cond = input_samples['theta_rake']>180
            if True in cond:
                # e.g., 185 -> -175
                input_samples['theta_rake'][cond] = input_samples['theta_rake'][cond]-360
            logging.info(f'\t\t- added "theta_rake" to input samples for {infra_type} analysis')
        else:
            input_samples['theta_rake'] = 0*ones_arr_nsite_by_ninput # dummy - not used but to carry on with analysis
                
    # get liquefaction susceptibility categories if obtained from regional geologic maps
    # for liq susc cat
    if has_edp:
        if running_below_ground_liquefaction and \
            'Hazus2020' in workflow['EDP']['liquefaction'] and \
            not 'liq_susc' in input_dist:
            if 'gw_depth' in input_samples:
                input_samples['liq_susc'] = get_regional_liq_susc(
                    input_table.GeologicUnit_Witter2006.copy(),
                    input_table.GeologicUnit_BedrossianEtal2012.copy(),
                    input_samples['gw_depth'],
                    default='none'
                )
                gw_depth_mean = input_dist['gw_depth']['mean'].copy()
                if input_dist['gw_depth']['dist_type'] == 'lognormal':
                    gw_depth_mean = np.exp(gw_depth_mean)
                input_dist['liq_susc'] = {
                    'value': get_regional_liq_susc(
                        input_table.GeologicUnit_Witter2006.copy(),
                        input_table.GeologicUnit_BedrossianEtal2012.copy(),
                        gw_depth_mean,
                        get_mean=True,
                        default='none'
                    ),
                    'dist_type': 'fixed'
                }
                logging.info(f'\t\t- added "liq_susc" to input samples for {infra_type} analysis')

    # generate additional inputs if crossing algorithm is performed
    # initialize params for storing additional sampling
    addl_input_dist = {}
    crossing_params_intermediate = []
    if flag_crossing_file_exists:
        # if below ground, then perform additional sampling using crossing angles
        if running_below_ground and has_dm:
            transition_weight_factor = None
            if running_below_ground_lateral_spread:
                # if running CPTs, then deformation polygons are produced and true beta_crossings exist
                if running_cpt_based_procedure:
                    # get additional crossing params
                    which_half = site_data.which_half.values
                    # repeat above crossing params over number of samples
                    which_half = np.tile(which_half,(num_epi_input_samples,1)).T
                    # get beta_crossing_samples
                    beta_crossing_samples = input_samples['beta_crossing']
                    # additional logic values given crossing conditions
                    cond_upper = which_half=='upper'
                    cond_lower = which_half=='lower'
                    # next get crossing angle conditions
                    # 1) crossing within +=/-= 30 deg
                    cond_le30_or_ge_150 = np.logical_or(beta_crossing_samples <= 30,beta_crossing_samples >= 150)
                    # 2) transition (30 to 45 or 135 to 150)
                    cond_transition_beta = np.logical_or(
                        np.logical_and(beta_crossing_samples > 30,beta_crossing_samples < 45),
                        np.logical_and(beta_crossing_samples > 135,beta_crossing_samples < 150)
                    )
                    # 3) between 45 and 135 (inclusive)
                    cond_btw_45_135 = np.logical_and(beta_crossing_samples >= 45,beta_crossing_samples <= 135)
                    cond_ge_90 = beta_crossing_samples >= 90
                    cond_lt_90 = beta_crossing_samples < 90
                    # next get controlling condition between freeface and ground slope
                    ls_cond = site_data.ls_cond.values
                    ls_cond = ls_cond.repeat(num_epi_input_samples).reshape((-1, num_epi_input_samples))
                    lh_ratio = site_data.lh_ratio.values
                    lh_ratio = lh_ratio.repeat(num_epi_input_samples).reshape((-1, num_epi_input_samples))
                    # 1) get cond where ground_slope only
                    cond_gs = ls_cond == 'ground_slope'
                    # get cond where freeface
                    cond_ff = ls_cond == 'freeface'
                    # 2) get cond where freeface and l/h <= 10
                    cond_ff_and_lh_le_10 = np.logical_and(cond_ff, lh_ratio<=10)
                    # 3) get cond where freeface and l/h > 10
                    cond_ff_and_lh_gt_10 = np.logical_and(cond_ff, lh_ratio>10)
                    # initialize additional crossing logic params
                    primary_mech = str_arr_nsite_by_ninput.copy()
                    # for weighting between mechanisms in transition zone
                    transition_weight_factor = np.ones(beta_crossing_samples.shape)
                    # determine primary mechanism(s) based on above conditions
                    # 1) crossing within +/- 30 deg
                    # -- ground slope
                    # ---- upper
                    cond_joint = cond_le30_or_ge_150 & cond_gs
                    cond_joint2 = cond_joint & cond_upper
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'BainTens'
                    # ---- lower
                    cond_joint2 = cond_joint & cond_lower
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'BainComp'
                    # -- freeface and l/h <= 10
                    cond_joint = cond_le30_or_ge_150 & cond_ff_and_lh_le_10
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'BainTens_and_HutabaratNormal'
                    # -- freeface and l/h > 10
                    cond_joint = cond_le30_or_ge_150 & cond_ff_and_lh_gt_10
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'BainTens'
                    # 2) transition (30 to 45 or 135 to 150)
                    # -- ground slope
                    # ---- upper
                    cond_joint = cond_transition_beta & cond_gs
                    cond_joint2 = cond_joint & cond_upper
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'BainTens_and_HutabaratSSTens'
                    # ---- lower
                    cond_joint2 = cond_joint & cond_lower
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'BainComp_and_HutabaratSSComp'
                    # -- freeface and l/h <= 10
                    cond_joint = cond_transition_beta & cond_ff_and_lh_le_10
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'BainTens_and_HutabaratNormal'
                    # -- freeface and l/h > 10
                    cond_joint = cond_transition_beta & cond_ff_and_lh_gt_10
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'BainTens_and_HutabaratSSTens'
                    # 3) between 45 and 135 (inclusive)
                    # -- ground slope
                    # ---- upper
                    cond_joint = cond_btw_45_135 & cond_gs
                    cond_joint2 = cond_joint & cond_upper
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'HutabaratSSTens'
                    # ---- lower
                    cond_joint2 = cond_joint & cond_lower
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'HutabaratSSComp'
                    # -- freeface and l/h <= 10
                    cond_joint = cond_btw_45_135 & cond_ff_and_lh_le_10
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'HutabaratNormal'
                    # -- freeface and l/h > 10
                    cond_joint = cond_btw_45_135 & cond_ff_and_lh_gt_10
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'HutabaratSSTens'
                    # get linear scale factor for transition zones (30 to 45, 135 to 150)
                    cond_joint = cond_transition_beta & cond_ge_90
                    if True in cond_joint:
                        beta_at_transition_ge90 = beta_crossing_samples[cond_joint]
                        transition_weight_factor[cond_joint] = 1 - (150-beta_at_transition_ge90)/(150-135)
                    cond_joint = cond_transition_beta & cond_lt_90
                    if True in cond_joint:
                        beta_at_transition_lt90 = beta_crossing_samples[cond_joint]
                        transition_weight_factor[cond_joint] = (45-beta_at_transition_lt90)/(45-30)
                else:
                    # get default values
                    primary_mech = str_arr_nsite_by_ninput.copy()
                    primary_mech[:] = 'HutabaratSSComp'
                    # for weighting between mechanisms in transition zone
                    transition_weight_factor = ones_arr_nsite_by_ninput.copy()
                    
            elif running_below_ground_settlement:
                # get default values
                primary_mech = str_arr_nsite_by_ninput.copy()
                primary_mech[:] = 'Normal' # always normal
                # for weighting between mechanisms in transition zone - does not apply to settlement
                transition_weight_factor = ones_arr_nsite_by_ninput.copy()
            
            elif running_below_ground_landslide:
                # if with possible crossings, then deformation polygons was used to determine crossings.
                if flag_possible_repeated_crossings:
                    # get additional crossing params
                    section_crossed = site_data.section_crossed.values
                    which_half = site_data.which_half.values
                    # repeat above crossing params over number of samples
                    section_crossed = np.tile(section_crossed,(num_epi_input_samples,1)).T
                    which_half = np.tile(which_half,(num_epi_input_samples,1)).T
                    # get beta_crossing_samples
                    beta_crossing_samples = input_samples['beta_crossing']
                    # additional logic values given crossing conditions
                    # first get conditions for sections crossed
                    cond_scarp = section_crossed=='scarp'
                    cond_body = section_crossed=='body'
                    cond_toe = section_crossed=='toe'
                    # next get conditions for upper vs lower half
                    cond_upper = which_half=='upper'
                    cond_lower = which_half=='lower'
                    # next get crossing angle conditions
                    # 1) crossing within +/- 20 deg
                    cond_le20_or_ge_160 = np.logical_or(beta_crossing_samples <= 20,beta_crossing_samples >= 160)
                    # 2) transition (20 to 45 or 135 to 160)
                    cond_transition_beta = np.logical_or(
                        np.logical_and(beta_crossing_samples > 20,beta_crossing_samples < 45),
                        np.logical_and(beta_crossing_samples > 135,beta_crossing_samples < 160)
                    )
                    # 3) pure strike-slip (45 to 135)
                    cond_btw_45_135 = np.logical_and(beta_crossing_samples >= 45,beta_crossing_samples <= 135)
                    cond_ge_90 = beta_crossing_samples >= 90
                    cond_lt_90 = beta_crossing_samples < 90
                    # initialize additional crossing logic params
                    primary_mech = str_arr_nsite_by_ninput.copy()
                    # for weighting between mechanisms in transition zone
                    transition_weight_factor = np.ones(beta_crossing_samples.shape)
                    # determine primary mechanism(s) based on above conditions
                    # 1) crossing within +=/-= 20 deg
                    # -- head scarp
                    cond_joint = cond_le20_or_ge_160 & cond_scarp
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'Normal'
                    # -- toe
                    cond_joint = cond_le20_or_ge_160 & cond_toe
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'Reverse'
                    # -- body, further determine by whether above or below 90
                    cond_joint = cond_le20_or_ge_160 & cond_body
                    cond_joint2 = cond_joint & cond_ge_90
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'SSComp'
                    cond_joint2 = cond_joint & cond_lt_90
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'SSTens'
                    # 2) transition (20 to 45 or 135 to 160)
                    # -- upper half
                    cond_joint = cond_transition_beta & cond_upper
                    # ---- determine by whether above or below 90
                    cond_joint2 = cond_joint & cond_ge_90
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'Normal_SSComp'
                    cond_joint2 = cond_joint & cond_lt_90
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'Reverse_SSComp'
                    # -- lower half
                    cond_joint = cond_transition_beta & cond_lower
                    # ---- determine by whether above or below 90
                    cond_joint2 = cond_joint & cond_ge_90
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'Normal_SSTens'
                    cond_joint2 = cond_joint & cond_lt_90
                    if True in cond_joint2:
                        primary_mech[cond_joint2] = 'Reverse_SSTens'
                    # 3) pure strike-slip (45 to 135, inclusive)
                    # --- determine by whether above or below 90
                    cond_joint = cond_btw_45_135 & cond_ge_90
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'SSComp'
                    cond_joint = cond_btw_45_135 & cond_lt_90
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'SSTens'
                    # get linear scale factor for transition zones (20 to 45, 135 to 160)
                    cond_joint = cond_transition_beta & cond_ge_90
                    if True in cond_joint:
                        beta_at_transition_ge90 = beta_crossing_samples[cond_joint]
                        transition_weight_factor[cond_joint] = 1 - (160-beta_at_transition_ge90)/(160-135)
                    cond_joint = cond_transition_beta & cond_lt_90
                    if True in cond_joint:
                        beta_at_transition_lt90 = beta_crossing_samples[cond_joint]
                        transition_weight_factor[cond_joint] = (45-beta_at_transition_lt90)/(45-20)
                    # store to input samples
                    # input_samples['primary_mech'] = primary_mech
                    # input_samples['transition_weight_factor'] = transition_weight_factor
                    # crossing_params_intermediate.append('primary_mech')
                    # crossing_params_intermediate.append('transition_weight_factor')
                    # get/update distributions based on above conditions
                    addl_input_dist = {}
                    # missing crossing params that need to be sampled given samples of beta_crossing
                    crossing_params = [
                        'psi_dip',
                    ]
                    # go through input dist and see which input params require sampling            
                    for param in crossing_params:
                        # get dist_type
                        param_dist_type = input_dist[param]['dist_type']
                        # get mean/value from input dist and see if any site is flagged with "sampling_dependent"
                        param_mean = input_dist[param]['mean']
                        if param_mean[0] == 'sampling_dependent':
                            # sigma
                            sigmas = np.tile(
                                input_dist[param]['sigma'].copy(),
                                (num_epi_input_samples,1)
                            ).T
                            # mean, low, and high are crossing angle dependent
                            # initialize
                            means = ones_arr_nsite_by_ninput.copy()
                            lows = null_arr_nsite_by_ninput.copy()
                            highs = twos_arr_nsite_by_ninput.copy()
                            # for any case with normal slip (pure or transition)
                            cond = np.logical_or(
                                primary_mech=='Normal',
                                np.logical_or(primary_mech=='Normal_SSComp',primary_mech=='Normal_SSTens')
                            )
                            if True in cond:
                                means[cond] = 65
                                lows[cond] = 45
                                highs[cond] = 90
                            # for any case with reverse slip (pure or transition)
                            cond = np.logical_or(
                                primary_mech=='Reverse',
                                np.logical_or(primary_mech=='Reverse_SSComp',primary_mech=='Reverse_SSTens')
                            )
                            if True in cond:
                                means[cond] = 35
                                lows[cond] = 25
                                highs[cond] = 50
                            # make dist
                            addl_input_dist[param] = {
                                'mean': means,
                                'sigma': sigmas,
                                'low': lows,
                                'high': highs,
                                'dist_type': param_dist_type
                            }
                            # generate residuals and get samples
                            res = lhs(n_site=n_site, n_var=1, n_samp=num_epi_input_samples)[:,:,0]
                            samples = truncnorm.ppf(
                                q=norm.cdf(res,0,1),
                                a=(lows-means)/sigmas,
                                b=(highs-means)/sigmas,
                                loc=means,
                                scale=sigmas
                            )
                            # store samples to input_samples
                            input_samples[param] = samples
                else:
                    # get default values
                    primary_mech = str_arr_nsite_by_ninput.copy()
                    primary_mech[:] = 'SSComp'
                    # for weighting between mechanisms in transition zone
                    transition_weight_factor = ones_arr_nsite_by_ninput.copy()
                    
            elif running_below_ground_fault_rupture:
                # get crossing and fault angle samples
                beta_crossing_samples = input_samples['beta_crossing']
                theta_rake_samples = input_samples['theta_rake']
                # beta crossing conditions
                cond_beta_le_90 = beta_crossing_samples <= 90
                cond_beta_gt_90 = beta_crossing_samples > 90
                # theta rake conditions
                cond_theta_case_1 = np.logical_and(theta_rake_samples>=-14,theta_rake_samples<=14)
                cond_theta_case_2 = np.logical_and(theta_rake_samples>-76,theta_rake_samples<-14)
                cond_theta_case_3 = np.logical_and(theta_rake_samples>=-104,theta_rake_samples<=-76)
                cond_theta_case_4 = np.logical_and(theta_rake_samples>-166,theta_rake_samples<-104)
                cond_theta_case_5 = np.logical_or(theta_rake_samples>=166,theta_rake_samples<=-166)
                cond_theta_case_6 = np.logical_and(theta_rake_samples>14,theta_rake_samples<76)
                cond_theta_case_7 = np.logical_and(theta_rake_samples>=76,theta_rake_samples<=104)
                cond_theta_case_8 = np.logical_and(theta_rake_samples>104,theta_rake_samples<166)
                # initialize additional crossing logic params
                primary_mech = str_arr_nsite_by_ninput.copy()
                # determine primary mechanism(s) based on above conditions
                # 1) left lateral strike-slip
                #       a)  0 <= beta_crossing <= 90    ss tens
                cond_joint = cond_theta_case_1 & cond_beta_le_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'SSTens'
                #       b)  90 < beta_crossing <= 180   ss comp
                cond_joint = cond_theta_case_1 & cond_beta_gt_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'SSComp'
                # 2) Oblique normal with left lateral strike-slip
                #       a)  0 <= beta_crossing <= 90    ss tens
                cond_joint = cond_theta_case_2 & cond_beta_le_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Normal_SSTens'
                #       b)  90 < beta_crossing <= 180   ss comp
                cond_joint = cond_theta_case_2 & cond_beta_gt_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Normal_SSComp'
                # 3) normal slip
                if True in cond_theta_case_3:
                    primary_mech[cond_theta_case_3] = 'Normal'
                # 4) Oblique normal with right lateral strike-slip
                #       a)  0 <= beta_crossing <= 90    ss tens
                cond_joint = cond_theta_case_4 & cond_beta_le_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Normal_SSTens'
                #       b)  90 < beta_crossing <= 180   ss comp
                cond_joint = cond_theta_case_4 & cond_beta_gt_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Normal_SSComp'
                # 5) right lateral strike-slip
                #       a)  0 <= beta_crossing <= 90    ss tens
                cond_joint = cond_theta_case_5 & cond_beta_le_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'SSTens'
                #       b)  90 < beta_crossing <= 180   ss comp
                cond_joint = cond_theta_case_5 & cond_beta_gt_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'SSComp'
                # 6) Oblique reverse with left lateral strike-slip
                #       a)  0 <= beta_crossing <= 90    ss tens
                cond_joint = cond_theta_case_6 & cond_beta_le_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Reverse_SSTens'
                #       b)  90 < beta_crossing <= 180   ss comp
                cond_joint = cond_theta_case_6 & cond_beta_gt_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Reverse_SSComp'
                # 7) reverse slip
                if True in cond_theta_case_7:
                    primary_mech[cond_theta_case_7] = 'Reverse'
                # 8) Oblique reverse with right lateral strike-slip
                #       a)  0 <= beta_crossing <= 90    ss tens
                cond_joint = cond_theta_case_8 & cond_beta_le_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Reverse_SSTens'
                #       b)  90 < beta_crossing <= 180   ss comp
                cond_joint = cond_theta_case_8 & cond_beta_gt_90
                if True in cond_joint:
                    primary_mech[cond_joint] = 'Reverse_SSComp'
                # mannually sample additional crossing scale factors recommended by LCI
                # params:
                # --- f_r: faulting frequency (nonzero for secondary hazard)
                # --- f_ds: displacement scale factor (nonzero for secondary hazard)
                # n site with secondary hazard
                qfault_type = site_data.qfault_type.values
                # make ones table for each param
                f_r_samples = ones_arr_nsite_by_ninput.copy()
                f_ds_samples = ones_arr_nsite_by_ninput.copy()
                # see crossing with secondary qfault exists
                cond_secondary = qfault_type=='secondary'
                if True in cond_secondary:
                    where_secondary = np.where(cond_secondary)[0]
                    n_secondary = len(where_secondary)
                    # performing sampling
                    # get LHS samples
                    res = lhs(n_site=n_secondary, n_var=2, n_samp=num_epi_input_samples)
                    res_cdf = norm.cdf(res,0,1)
                    # first get samples of f_r
                    f_r_trian_dist_pts = np.array([
                        [0.2,0.8],
                        [0.5,2],
                        [0.8,0.8],
                    ]) # prob beyond first and last points = 0
                    # get discrete cdf
                    f_r_trian_dist_cdf = get_cdf_given_pts(f_r_trian_dist_pts)
                    # get samples
                    f_r_samples[where_secondary] = np.transpose([
                        np.interp(res_cdf[:,samp_ind,0],f_r_trian_dist_cdf[:,1],f_r_trian_dist_cdf[:,0])
                        for samp_ind in range(num_epi_input_samples)
                    ])
                    # next get samples of f_ds
                    f_ds_trian_dist_pts = np.array([
                        [0.05,2/3],
                        [0.15,2],
                        [1,0],
                    ]) # prob beyond first and last points = 0
                    # get discrete cdf
                    f_ds_trian_dist_cdf = get_cdf_given_pts(f_ds_trian_dist_pts)
                    # get samples
                    f_ds_samples[where_secondary] = np.transpose([
                        np.interp(res_cdf[:,samp_ind,1],f_ds_trian_dist_cdf[:,1],f_ds_trian_dist_cdf[:,0])
                        for samp_ind in range(num_epi_input_samples)
                    ])
                # store samples to input_samples
                input_samples['f_r'] = f_r_samples
                input_samples['f_ds'] = f_ds_samples
            # store to input samples
            input_samples['primary_mech'] = primary_mech
            crossing_params_intermediate.append('primary_mech')
            if transition_weight_factor is not None:
                input_samples['transition_weight_factor'] = transition_weight_factor
                crossing_params_intermediate.append('transition_weight_factor')
            logging.info(f'\t\t- added "primary_mech" to input samples for {infra_type} analysis')
            logging.info(f'\t\t- added "transition_weight_factor" to input samples for {infra_type} analysis')
    logging.info(f'... DONE - Finished sampling input parameters that are not event-dependent')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup and run PC
    # initialize for tracking
    df_frac = {}
    sites_with_no_crossing = None
    track_im_dependency_for_output = {}
    prev_case_params = {}
    pc_coeffs = {}
    last_haz_param_full = {}
    pbee_dim = {}
    null_arr_pc_terms = {}
    hermite_prob_table_indep = {}
    # minimum mean values for various hazards to be considered "zero"
    # e.g., for ground deformation, 5 cm
    # used for below ground reduce scale of problem; not necessary for other infrastructure
    min_mean_for_zero_val = {
        'im': {
            'pga': -7, # pga, lognormal
            'pgv': -5, # pgv, lognormal
        },
        'edp': {
            # 'pgdef': 0.05, # ground deformation, < 5 cm = 0, per Task 4B
            'pgdef': 1e-3, # ground deformation, < 0.1 cm = 0
        },
        'dm': {
            'eps_pipe_comp': 1e-5, # %, pipe strain, compressive, 1e-5%
            'eps_pipe_tens': 1e-5, # %, pipe strain, compressive, 1e-5%
        },
        'dv': {
        }
    }
    
    #----------------------
    if get_timer:
        print('\n')
        time_initial = time.time()
        time_start = time_initial
    #----------------------
    
    # event to run
    logging.info('---------------------------')
    logging.info(f'{counter}. Starting risk analysis using Polynomial Chaos...')
    counter += 1
    time_loop_init = time.time()
    time_loop = time.time()
    logging.info(f'\t>>>>> number of events: {len(event_ids_to_run)}')
    event_counter = 0
    for ind, event_id in enumerate(event_ids_to_run):
    # for event_ind in range(1):
        # current event index
        event_ind = event_ind_relative_to_rupture_table[ind]
        # for tracking sites with nonzero mean values to reduce scale
        sites_to_keep = {}
        rows_to_keep_rel_to_nonzero_step0 = {}
        rows_to_keep_index = {}
        # start with assumption to use all of the sites
        sites_with_nonzero_step0 = n_site_ind_arr.copy()
        # initialize for tracking primary PC inputs
        mean_of_mu = {}
        sigma_of_mu = {}
        sigma = {}
        amu = {}
        bmu = {}
        # additional terms used for PC
        index_mean_of_mu_up = {}
        intercept = {}
        slope = {}
        tangent_vector = {}
        sigma_mu_intercept = {}
        liq_results = None
        
        ###########################################
        # run if CPT-based
        if running_cpt_based_procedure or running_below_ground_fault_rupture:
            # also get the segments crossing deformation polygon developed for current event
            sites_to_run_curr_event = rows_to_run_by_event_id[event_id]
            # find sites with nonzero PGA
            pga_curr_event = np.round(im_import['pga']['mean_table'][event_ind,sites_to_run_curr_event].copy()-10,decimals=3)
        else:
            # find sites with nonzero PGA
            pga_curr_event = np.round(im_import['pga']['mean_table'][event_ind,:].copy()-10,decimals=3)
        # get sites with nonzero IMs
        sites_with_nonzero_step0 = np.where(pga_curr_event>min_mean_for_zero_val['im']['pga'])[0]
        n_site_curr_event = len(sites_with_nonzero_step0)
        # get nonzero im
        im_dist_info = {}
        for each in ['pga','pgv']:
            im_dist_info[each] = {
                # 'mean': im_import[each]['mean_table'].loc[event_ind,:].values,
                # 'sigma': im_import[each]['sigma_table'].loc[event_ind,:].values,
                # 'sigma_mu': im_import[each]['sigma_mu_table'].loc[event_ind,:].values,
                
                # note that sparse matrix files require -10 to all values for correct magnitude;
                # during ground motion phase, locations with 0 intensities are reported as -10 instead of -np.inf;
                # for storage, in order to be memory efficient, ground motions are stored as sparse matrx by adding
                # 10 to all values, thus convert the -10 intensity magnitudes to 0;
                # ---> this means when using the sparse datafiles, they must be -10 to get the correct magnitude
                'mean': np.round(im_import[each]['mean_table'][event_ind,sites_with_nonzero_step0].copy()-10,decimals=3),
                'sigma': np.round(im_import[each]['sigma_table'][event_ind,sites_with_nonzero_step0].copy(),decimals=3),
                'sigma_mu': np.round(im_import[each]['sigma_mu_table'][event_ind,sites_with_nonzero_step0].copy(),decimals=3),
                'dist_type': 'lognormal'
            }
            # avoid sigma = 0 for PC
            im_dist_info[each]['sigma'] = np.maximum(im_dist_info[each]['sigma'],0.001)
        
        #----------------------
        if get_timer:
            print(f'\t-2. time: {time.time()-time_start} seconds')
            time_start = time.time()
        #----------------------
        
        # special mapping keys for rupture metadata
        rup_map_key = {
            'magnitude': 'mag',
            'annual_rate': 'rate',
            'fault_angle_deg': 'theta',
            'fault_crossing_depth_m': 'z_crossing'
        }
        rup_info = {}
        for key in list(rupture_table.columns):
            if key in rup_map_key:
                if isinstance(rupture_table[key][event_ind], str):
                    rup_info[rup_map_key[key]] = np.asarray(json.loads(rupture_table[key][event_ind]))
                else:
                    rup_info[rup_map_key[key]] = rupture_table[key][event_ind]
            else:
                if isinstance(rupture_table[key][event_ind], str):
                    rup_info[key] = np.asarray(json.loads(rupture_table[key][event_ind]))
                else:
                    rup_info[key] = rupture_table[key][event_ind]
        
        # get additional rupture params for fault rupture specifically
        if running_below_ground_fault_rupture:
            # also get 'norm_dist' and 'prob_disp_sf' from site_data if available
            for col in ['norm_dist']:
                if col in site_data:
                    input_samples[col] = \
                        site_data[col].values.repeat(num_epi_input_samples).reshape(
                            (-1, num_epi_input_samples))
                            
        # get list of sites with no well crossing
        if running_wells_caprocks:
            sites_with_crossing = rup_info['well_ind_crossed'].copy()
            if sites_with_no_crossing is None:
                for key in list(rup_info):
                    if key == 'theta':
                        sites_with_no_crossing = list(set(sites_with_nonzero_step0).difference(set(sites_with_crossing)))
                        break
        
        #----------------------
        if get_timer:
            print(f'\t-1. time: {time.time()-time_start} seconds')
            time_start = time.time()
        #----------------------
        
        # if crossing algorithm is performed in preprocessing OR if wells
        if flag_crossing_file_exists or running_wells_caprocks:
            # initialize params for storing additional sampling
            addl_input_dist = {}
            null_arr_nsite_nonzero = np.zeros(n_site_curr_event)
            ones_arr_nsite_nonzero = np.ones(n_site_curr_event)
            # if wells and caprocks, then perform additional sampling using fault depths and crossing angles
            if running_wells_caprocks:
                crossing_params = [
                    'theta', # fault angle (deg)
                    'z_crossing', # fault depth (m)
                    'e_rock', # Young's modulus of rock (GPa)
                ]
                # go through input dist and see which input params require sampling
                for param in crossing_params:
                    # get dist_type
                    param_dist_type = input_dist[param]['dist_type']
                    # get mean/value from input dist and see if any site is flagged with "event_dependent"
                    if param_dist_type == 'fixed':
                        param_value = input_dist[param]['value']
                        if param_value[0] == 'event_dependent':
                            if param == 'e_rock':
                                addl_input_dist[param] = {
                                    'value': addl_input_dist['z_crossing']['value']*1.6E-3 + 15.2,
                                    'dist_type': 'fixed'
                                }
                            else:
                                param_val_arr = null_arr_nsite.copy()
                                param_val_arr[sites_with_crossing] = rup_info[param]
                                addl_input_dist[param] = {
                                    'value': param_val_arr,
                                    'dist_type': 'fixed'
                                }
                            # also update the mean value in input_dist
                            input_dist[param]['value'] = addl_input_dist[param]['value'].copy()
                    else:
                        param_mean = input_dist[param]['mean']
                        if param_mean[0] == 'event_dependent':
                            # sigma
                            sigmas = input_dist[param]['sigma'].copy()
                            if len(sites_with_crossing) > 0:
                                # low
                                lows = null_arr_nsite.copy()
                                lows[sites_with_crossing] = input_dist[param]['low'][sites_with_crossing]
                                # high
                                highs = null_arr_nsite.copy()
                                highs[sites_with_crossing] = input_dist[param]['high'][sites_with_crossing]
                            else:
                                if param_dist_type == 'normal':
                                    lows = null_arr_nsite.copy()
                                elif param_dist_type == 'lognormal':
                                    lows = ones_arr_nsite.copy() * -np.inf
                                highs = ones_arr_nsite.copy() * np.inf
                            # if e_rock
                            if param == 'e_rock':
                                # mean
                                param_val_arr = addl_input_dist['z_crossing']['mean']*1.6E-3 + 15.2
                                param_val_arr[addl_input_dist['theta']['mean']==0] = 0
                            else:
                                # mean
                                param_val_arr = null_arr_nsite.copy()
                                if len(sites_with_crossing) > 0:
                                    param_val_arr[sites_with_crossing] = rup_info[param]
                                else:
                                    param_val_arr = null_arr_nsite.copy()
                            # make dist
                            addl_input_dist[param] = {
                                'mean': param_val_arr,
                                'sigma': sigmas,
                                'low': lows,
                                'high': highs,
                                'dist_type': param_dist_type
                            }
                            # also update the metrics in input_dist
                            input_dist[param]['mean'] = addl_input_dist[param]['mean'].copy()
                            input_dist[param]['low'] = addl_input_dist[param]['low'].copy()
                            input_dist[param]['high'] = addl_input_dist[param]['high'].copy()
            # perform sampling on additional params and append to existing samples
            if len(addl_input_dist) > 0:
                addl_input_samples = \
                    pc_workflow.get_samples_for_params(addl_input_dist, num_epi_input_samples, n_site)
                for each in addl_input_samples:
                    if each == 'theta':
                        addl_input_samples[each][sites_with_no_crossing,:] = 0
                    input_samples[each] = addl_input_samples[each]
        
        #----------------------
        if get_timer:
            print(f'\t-1a. time: {time.time()-time_start} seconds')
            time_start = time.time()
        #----------------------
        
        # get inputs for sites with nonzero step0 values
        input_samples_nsite_nonzero = {
            param: input_samples[param][sites_with_nonzero_step0]
            for param in input_samples
        }
        
        #----------------------
        if get_timer:
            print(f'\t-1b. time: {time.time()-time_start} seconds')
            time_start = time.time()
        #----------------------
            
        # print(input_samples_nsite_nonzero['d_pipe'])
        input_dist_nsite_nonzero = {}
        for param in input_dist:
            input_dist_nsite_nonzero[param] = {}
            for met in list(input_dist[param]):
                if met == 'dist_type':
                    input_dist_nsite_nonzero[param][met] = input_dist[param][met]
                else:
                    input_dist_nsite_nonzero[param][met] = input_dist[param][met][sites_with_nonzero_step0]
        # make ones array to be used later
        ones_arr_nsite_nonzero_by_ninput = np.ones((n_site_curr_event,num_epi_input_samples))
        
        #----------------------
        if get_timer:
            print(f'\t0. time: {time.time()-time_start} seconds')
            time_start = time.time()
        #----------------------

        # loop through cases to run
        for case_to_run in range(1,n_cases+1):
            # string for current case
            curr_case_str = f"case_{case_to_run}"
            if case_to_run > 1:
                prev_case_str = f"case_{case_to_run-1}"
            else:
                prev_case_str = None
            # get workflow current case
            workflow_i = workflow_order_list[curr_case_str]
            # initialize sub-dictionaries
            mean_of_mu[curr_case_str] = {}
            sigma_of_mu[curr_case_str] = {}
            sigma[curr_case_str] = {}
            amu[curr_case_str] = {}
            bmu[curr_case_str] = {}
            # track IM dependence for generating output file
            # if event_ind == 0:
            if event_counter == 0:
                track_im_dependency_for_output[case_to_run-1] = []
            # for caprock specifically
            if 'caprock_leakage' in workflow_i['haz_list']:
                # pass and run this later since caprock leakage is not dependent on IMs
                pass
            else:
                # only do this once to initialize PC background params
                # if event_ind == 0:
                if event_counter == 0:
                    # inputs scenario
                    pbee_dim[curr_case_str] = workflow_i['n_pbee_dim']
                    pc_order = 4
                    num_pc_terms_indep = pc_util.num_pc_terms(pbee_dim[curr_case_str],pc_order)
                    index_pc_table_indep = pc_util.index_table_function(pbee_dim[curr_case_str],pc_order)
                    # make null array for number of sites x number of PC terms
                    null_arr_pc_terms[curr_case_str] = np.zeros((n_site,num_pc_terms_indep))
                    # pre-calculate hermite probs for epistemic samples
                    epi_samples_for_pc = np.random.normal(size=(num_epi_fractile_samples,pbee_dim[curr_case_str]))
                    hermite_prob_table_indep[curr_case_str] = np.zeros((num_pc_terms_indep,num_epi_fractile_samples))
                    # loop through number of independent pc terms to get Hermite probability
                    for i in range(num_pc_terms_indep):
                        hermite_prob_table_indep_i = ones_arr_nfractile_sample.copy()
                        for j in range(pbee_dim[curr_case_str]):
                            hermite_prob_table_indep_i *= \
                                pc_util.hermite_prob(
                                    epi_samples_for_pc[:,pbee_dim[curr_case_str]-1-j],
                                    index_pc_table_indep[i,j]
                                )
                        hermite_prob_table_indep[curr_case_str][i,:] = hermite_prob_table_indep_i
                    # additional tracking params
                    pc_coeffs[curr_case_str] = {}
                
                ###########################################
                # continue with rest of PC setup
                # default to PGA if CPT-based
                # if running_cpt_based_procedure:
                #     prev_haz_param = ['pga']
                # # run if not CPT-based
                # else:
                # set up for step 1
                # previous step (step 0) in PBEE
                step0 = 0
                step0_str = f'step_{step0}'
                step0_cat = workflow_i['cat_list'][step0].lower()
                step0_haz = workflow_i['haz_list'][step0]
                if step0_haz != 'im':
                    step0_haz_dict = methods_dict[step0_cat][step0_haz].copy()
                    step0_methods = step0_haz_dict['method']
                else:
                    step0_methods = {}
                # if step0 setup params are the same as the previous case, then skip analysis
                if prev_case_str is not None and\
                    step0_str in prev_case_params and \
                    prev_case_params[step0_str]['cat'] == step0_cat and \
                    prev_case_params[step0_str]['haz'] == step0_haz and \
                    prev_case_params[step0_str]['methods'] == list(step0_methods):
                    # store distribution metrics
                    mean_of_mu[curr_case_str][step0_str] = mean_of_mu[prev_case_str][step0_str].copy()
                    sigma_of_mu[curr_case_str][step0_str] = sigma_of_mu[prev_case_str][step0_str].copy()
                    sigma[curr_case_str][step0_str] = sigma[prev_case_str][step0_str].copy()
                    track_im_dependency_for_output[case_to_run-1] = track_im_dependency_for_output[case_to_run-2].copy()
                else:
                    # metadata for step 1 in PBEE
                    step1 = 1
                    step1_str = f'step_{step1}'
                    step1_cat = workflow_i['cat_list'][step1].lower()
                    step1_haz = workflow_i['haz_list'][step1]
                    step1_haz_param = methods_dict[step1_cat][step1_haz]['return_params']
                    # number of methods for hazard
                    step1_haz_dict = methods_dict[step1_cat][step1_haz].copy()
                    step1_methods = step1_haz_dict['method']
                    # if start with IM
                    if step0_cat == 'im':
                        # find primary IM intensity
                        mean_of_mu[curr_case_str][step0_str] = {}
                        sigma_of_mu[curr_case_str][step0_str] = {}
                        sigma[curr_case_str][step0_str] = {}
                        prev_haz_param = []
                        # for method in list(step1_methods):
                        for param in step1_haz_dict['upstream_params']:
                            # get param dists
                            if param == 'pga' or param == 'pgv':
                                mean_of_mu[curr_case_str][step0_str][param] = im_dist_info[param]['mean']
                                sigma_of_mu[curr_case_str][step0_str][param] = im_dist_info[param]['sigma_mu']
                                sigma[curr_case_str][step0_str][param] = im_dist_info[param]['sigma']
                                prev_haz_param.append(param)
                        # default to PGA as domain
                        if not 'pga' in step1_haz_dict['upstream_params'] and \
                            not 'pgv' in step1_haz_dict['upstream_params']:
                                mean_of_mu[curr_case_str][step0_str]['pga'] = im_dist_info['pga']['mean']
                                sigma_of_mu[curr_case_str][step0_str]['pga'] = im_dist_info['pga']['sigma_mu']
                                sigma[curr_case_str][step0_str]['pga'] = im_dist_info['pga']['sigma']
                                prev_haz_param.append('pga')
                        # if event == 0
                        # if event_ind == 0:
                        if event_counter == 0:
                            if 'pga' in prev_haz_param:
                                track_im_dependency_for_output[case_to_run-1].append('pga'.upper())
                            elif 'pgv' in prev_haz_param:
                                track_im_dependency_for_output[case_to_run-1].append('pgv'.upper())
                    # if doesn't start with IM
                    else:    
                        # get param for domain vector
                        param_for_domain = step1_haz_dict['upstream_params'][0]
                        # get additional params for evaluation
                        step0_param_names_all = methods_dict[step0_cat][step0_haz]['input_params']
                        step0_param_internal = step0_param_names_all.copy()
                        step0_param_external = []
                        n_step0_params = len(step0_param_names_all)
                        step0_input_samples = {}
                        # step0_input_dist = {}
                        for param in step0_param_names_all:
                            if param in input_samples:
                                step0_input_samples[param] = input_samples_nsite_nonzero[param].copy()
                                # step0_input_dist[param] = input_dist_nsite_nonzero[param].copy()
                                step0_param_external.append(param)
                                step0_param_internal.remove(param)
                            else:
                                raise ValueError(f'Cannot find {param} in input_samples')
                        # pull upstream params for full analysis
                        step0_upstream_params = {}
                        for param in step0_haz_dict['upstream_params']:
                            step0_upstream_params[param] = ones_arr_nsite_nonzero_by_ninput.copy()*rup_info[param]
                        # pull internal params, e.g., prob_liq and liq_susc
                        step0_internal_params = {}
                        # get mean of mu for domain vector
                        _, step0_results = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                            haz_dict=step0_haz_dict,
                            upstream_params=step0_upstream_params,
                            internal_params=step0_internal_params,
                            input_samples=step0_input_samples, 
                            n_sample=num_epi_input_samples,
                            n_site=n_site_curr_event,
                            level_to_run=level_to_run,
                        )
                        # get other metrics
                        mean_of_mu[curr_case_str][step0_str] = {param_for_domain: step0_results[param_for_domain]['mean_of_mu']}
                        sigma_of_mu[curr_case_str][step0_str] = {param_for_domain: step0_results[param_for_domain]['sigma_of_mu']}
                        sigma[curr_case_str][step0_str] = {param_for_domain: step0_results[param_for_domain]['sigma']}
                        prev_haz_param = [param_for_domain]
                
                #----------------------
                if get_timer:
                    print(f'\t1. time: {time.time()-time_start} seconds')
                    time_start = time.time()
                #----------------------
                
                ###########################################
                # for tracking if results from current step has sites with nonzero mean values
                has_nonzero_mean = False
                sites_to_keep[step0_str] = sites_with_nonzero_step0.copy()
                rows_to_keep_rel_to_nonzero_step0[step0_str] = np.arange(n_site_curr_event).astype(int)
                rows_to_keep_index[step0_str] = rows_to_keep_rel_to_nonzero_step0[step0_str].copy()
                # loop through steps
                for step in range(1,pbee_dim[curr_case_str]-1):
                    # previous category in PBEE
                    prev_step_str = f'step_{step-1}'
                    prev_cat = workflow_i['cat_list'][step-1].lower()
                    prev_haz = workflow_i['haz_list'][step-1]
                    if step > 1:
                        prev_haz_param = methods_dict[prev_cat][prev_haz]['return_params']
                    # current category in PBEE
                    curr_step_str = f'step_{step}'
                    curr_cat = workflow_i['cat_list'][step].lower()
                    curr_haz = workflow_i['haz_list'][step]
                    curr_haz_param = methods_dict[curr_cat][curr_haz]['return_params']
                    # number of methods for hazard
                    curr_haz_dict = methods_dict[curr_cat][curr_haz].copy()
                    curr_methods = curr_haz_dict['method']
                    
                    ###########################################
                    # if current step setup params are the same as the previous case, then skip analysis
                    if prev_case_str is not None and\
                        curr_step_str in prev_case_params and \
                        prev_case_params[curr_step_str]['cat'] == curr_cat and \
                        prev_case_params[curr_step_str]['haz'] == curr_haz and \
                        prev_case_params[curr_step_str]['methods'] == list(step0_methods):
                        # use results from analysis of previous case
                        mean_of_mu[curr_case_str][curr_step_str] = mean_of_mu[prev_case_str][curr_step_str].copy()
                        sigma_of_mu[curr_case_str][curr_step_str] = sigma_of_mu[prev_case_str][curr_step_str].copy()
                        sigma[curr_case_str][curr_step_str] = sigma[prev_case_str][curr_step_str].copy()
                        amu[curr_case_str][curr_step_str] = amu[prev_case_str][curr_step_str].copy()
                        bmu[curr_case_str][curr_step_str] = amu[prev_case_str][curr_step_str].copy()
                        # use indices from previous case
                        sites_to_keep[curr_step_str] = prev_sites_to_keep[curr_step_str].copy()
                        rows_to_keep_rel_to_nonzero_step0[curr_step_str] = \
                            prev_rows_to_keep_rel_to_nonzero_step0[curr_step_str].copy()
                        rows_to_keep_index[curr_step_str] = prev_rows_to_keep_index[curr_step_str].copy()
                    else:
                        # setup and run analysis
                        # get n_site to use
                        n_site_to_use = len(rows_to_keep_rel_to_nonzero_step0[prev_step_str])
                        # if running CPT-based
                        if 'CPTBased' in curr_methods:
                            # get mean, sigma, sigmu_mu from CPT analysis
                            mean_of_mu[curr_case_str][curr_step_str] = {
                                'pgdef': np.log(site_data.loc[sites_to_run_curr_event].pgdef_m.values)}
                            sigma_of_mu[curr_case_str][curr_step_str] = {
                                'pgdef': site_data.loc[sites_to_run_curr_event].sigma_mu.values}
                            sigma[curr_case_str][curr_step_str] = {
                                'pgdef': site_data.loc[sites_to_run_curr_event].sigma.values}
                            amu[curr_case_str][curr_step_str] = {
                                'pgdef': site_data.loc[sites_to_run_curr_event].amu.values}
                            bmu[curr_case_str][curr_step_str] = {
                                'pgdef': site_data.loc[sites_to_run_curr_event].bmu.values}
                            # store indices
                            rows_to_keep_index[curr_step_str] = rows_to_keep_index[prev_step_str].copy()
                            rows_to_keep_rel_to_nonzero_step0[curr_step_str] = \
                                rows_to_keep_rel_to_nonzero_step0[prev_step_str].copy()
                            sites_to_keep[curr_step_str] = sites_to_keep[prev_step_str].copy()
                            # by analysis approach, CPT-based runs will start with nonzere values
                            has_nonzero_mean = True
                        # if running_cpt_based_procedure is False:
                        else:
                            # special case for "lateral spread" and "settlement", which require "liquefaction" to be first assessed.
                            if (curr_haz == 'lateral_spread' or curr_haz == 'settlement'):
                                liq_haz_dict = methods_dict['edp']['liquefaction'].copy()
                                liq_methods = liq_haz_dict['method']
                                
                                # step1_param_names_liq = ['vs30','precip','dist_coast','gw_depth','dist_river','dist_water']
                                liq_input_param_names = methods_dict['edp']['liquefaction']['input_params']
                                liq_input_samples = {}
                                # liq_input_dist = {}
                                for param in liq_input_param_names:
                                    liq_input_samples[param] = input_samples_nsite_nonzero[param].copy()
                                    # liq_input_dist[param] = input_dist_nsite_nonzero[param].copy()
                                # pull upstream params by method for full analysis
                                liq_upstream_params = {} # for mean
                                for param in liq_haz_dict['upstream_params']:
                                    # intensities
                                    if param == 'pga' or param == 'pgv':
                                        liq_upstream_params[param] = np.tile(
                                            np.exp(im_dist_info[param]['mean'].copy())
                                            ,(num_epi_input_samples,1)
                                        ).T
                                    # rupture params
                                    else:
                                        liq_upstream_params[param] = ones_arr_nsite_nonzero_by_ninput.copy()*rup_info[param]
                                # no internal params since nothing has been evaluated
                                liq_internal_params = {}
                                # preprocess methods with input samples
                                if 'liq_susc' in input_dist:
                                    liq_results, _, = pc_workflow.process_methods_for_mean_and_sigma_of_mu_for_liq(
                                        haz_dict=liq_haz_dict,
                                        upstream_params=liq_upstream_params,
                                        internal_params=liq_internal_params,
                                        input_samples=liq_input_samples, 
                                        n_sample=num_epi_input_samples,
                                        n_site=n_site_curr_event,
                                        get_liq_susc=False,
                                        level_to_run=level_to_run,
                                    )
                                    if not 'liq_susc' in input_samples:
                                        liq_susc = np.tile(input_dist['liq_susc']['value'],(num_epi_input_samples,1)).T
                                else:
                                    liq_results, liq_susc, = pc_workflow.process_methods_for_mean_and_sigma_of_mu_for_liq(
                                        haz_dict=liq_haz_dict,
                                        upstream_params=liq_upstream_params,
                                        internal_params=liq_internal_params,
                                        input_samples=liq_input_samples, 
                                        n_sample=num_epi_input_samples,
                                        n_site=n_site_curr_event,
                                        get_liq_susc=True,
                                        level_to_run=level_to_run,
                                    )
                                # rerun with with upstream param * factor for getting slope using forward Euler
                                liq_upstream_params_forward = liq_upstream_params.copy()
                                for param in liq_haz_dict['upstream_params']:
                                    # intensities
                                    if param == 'pga' or param == 'pgv':
                                        liq_upstream_params_forward[param] = liq_upstream_params_forward[param] * forward_euler_multiplier
                                # preprocess methods with input samples
                                if 'liq_susc' in input_dist:
                                    liq_results_forward, _, = pc_workflow.process_methods_for_mean_and_sigma_of_mu_for_liq(
                                        haz_dict=liq_haz_dict,
                                        upstream_params=liq_upstream_params_forward,
                                        internal_params=liq_internal_params,
                                        input_samples=liq_input_samples, 
                                        n_sample=num_epi_input_samples,
                                        n_site=n_site_curr_event,
                                        get_liq_susc=False,
                                        level_to_run=level_to_run,
                                    )
                                    if not 'liq_susc' in input_samples:
                                        liq_susc_forward = np.tile(
                                            input_dist_nsite_nonzero['liq_susc']['value'],
                                            (num_epi_input_samples,1)
                                        ).T
                                else:
                                    liq_results_forward, liq_susc_forward, = pc_workflow.process_methods_for_mean_and_sigma_of_mu_for_liq(
                                        haz_dict=liq_haz_dict,
                                        upstream_params=liq_upstream_params_forward,
                                        internal_params=liq_internal_params,
                                        input_samples=liq_input_samples, 
                                        n_sample=num_epi_input_samples,
                                        n_site=n_site_curr_event,
                                        get_liq_susc=True,
                                        level_to_run=level_to_run,
                                    )
                            
                            ###########################################
                            # get additional params for evaluation
                            curr_param_names_all = methods_dict[curr_cat][curr_haz]['input_params']
                            curr_param_internal = curr_param_names_all.copy()
                            curr_param_external = []
                            n_curr_params = len(curr_param_names_all)
                            curr_input_samples = {}
                            # curr_input_dist = {}
                            # rows to use for inputs
                            rows_inputs = rows_to_keep_rel_to_nonzero_step0[prev_step_str].copy()
                            # go through params
                            for param in curr_param_names_all:
                                if param in input_samples_nsite_nonzero:
                                    curr_input_samples[param] = input_samples_nsite_nonzero[param][rows_inputs].copy()
                                    curr_param_external.append(param)
                                    curr_param_internal.remove(param)
                            # pull upstream params for full analysis
                            curr_upstream_params = {}
                            for param in curr_haz_dict['upstream_params']:
                                # if in previous mean of mu
                                if param in mean_of_mu[curr_case_str][prev_step_str]:
                                    if np.ndim(mean_of_mu[curr_case_str][prev_step_str][param]) == 0:
                                        curr_upstream_params[param] = \
                                            ones_arr_nsite_nonzero_by_ninput.copy() * \
                                            np.exp(mean_of_mu[curr_case_str][prev_step_str][param].copy())
                                    else:
                                        curr_upstream_params[param] = np.tile(
                                            np.exp(mean_of_mu[curr_case_str][prev_step_str][param].copy())
                                            ,(num_epi_input_samples,1)
                                        ).T
                                # from rupture params
                                else:
                                    curr_upstream_params[param] = ones_arr_nsite_nonzero_by_ninput[rows_inputs].copy()*rup_info[param]
                            # pull internal params, e.g., prob_liq and liq_susc
                            curr_internal_params = {}
                            for param in curr_param_internal:
                                if param == 'liq_susc':
                                    if not 'liq_susc' in input_samples:
                                        curr_internal_params[param] = liq_susc.copy()
                                else:
                                    if liq_results is not None:
                                        curr_internal_params[param] = liq_results[param]['mean_of_mu'].copy()
                            
                            #----------------------
                            if get_timer:
                                print(f'\t2a-1. time: {time.time()-time_start} seconds')
                                time_start = time.time()
                            #----------------------
                            
                            ###########################################
                            # run methods with samples of input parameters to get samples of mean predictions
                            _, curr_results = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                                haz_dict=curr_haz_dict,
                                upstream_params=curr_upstream_params,
                                internal_params=curr_internal_params,
                                input_samples=curr_input_samples, 
                                n_sample=num_epi_input_samples,
                                n_site=n_site_to_use,
                                level_to_run=level_to_run,
                            )
                            
                            #----------------------
                            if get_timer:
                                print(f'\t2a. time: {time.time()-time_start} seconds')
                                time_start = time.time()
                            #----------------------
                            
                            # track nonzero sites for step0
                            nonzero_ind_from_out = np.arange(n_site_to_use).astype(int)
                            # reduce problem scale by keeping only values greater than threshold
                            for param in curr_results:
                                # print(param, curr_results[param]['mean_of_mu'])
                                if param in min_mean_for_zero_val[curr_cat]:
                                    threshold_val = min_mean_for_zero_val[curr_cat][param]
                                    if curr_results[param]['dist_type'] == 'lognormal':
                                        threshold_val = np.log(threshold_val)
                                    rows_nonzero_mean_for_curr_param = np.where(
                                        curr_results[param]['mean_of_mu']>threshold_val
                                    )[0]
                                    # store to tracking dictionary
                                    nonzero_ind_from_out = np.hstack([
                                        nonzero_ind_from_out,
                                        rows_nonzero_mean_for_curr_param,
                                    ])                            
                            nonzero_ind_from_out = np.unique(nonzero_ind_from_out).astype(int)
                            # store indices
                            rows_to_keep_index[curr_step_str] = nonzero_ind_from_out.copy()
                            rows_to_keep_rel_to_nonzero_step0[curr_step_str] = \
                                rows_to_keep_rel_to_nonzero_step0[prev_step_str][nonzero_ind_from_out].copy()
                            sites_to_keep[curr_step_str] = sites_to_keep[prev_step_str][nonzero_ind_from_out].copy()
                            # see if any of the output params have non-"zero" means
                            # if not, then skip complete because prob = 0
                            if len(nonzero_ind_from_out) == 0:
                                # break out of loop for getting mean, sigma mu, sigma if there are no sites with nonzero mean values
                                break
                            # if at least one site with non-"zero" means, reduce inputs for subsequent analysis 
                            else:
                                # continue with loop
                                has_nonzero_mean = True
                                # rerun with with upstream param * factor for getting slope using forward Euler
                                # pull upstream params for full analysis
                                curr_upstream_params_forward = curr_upstream_params.copy()
                                for param in curr_haz_dict['upstream_params']:
                                    # if in previous mean of mu
                                    if param in mean_of_mu[curr_case_str][prev_step_str]:
                                        curr_upstream_params_forward[param] = curr_upstream_params_forward[param] * forward_euler_multiplier
                                # pull internal params, e.g., prob_liq and liq_susc
                                curr_internal_params_forward = {}
                                for param in curr_param_internal:
                                    if param == 'liq_susc':
                                        if not 'liq_susc' in input_samples:
                                            curr_internal_params_forward[param] = liq_susc_forward.copy()
                                    else:
                                        if liq_results is not None:
                                            curr_internal_params_forward[param] = liq_results_forward[param]['mean_of_mu'].copy()
                                
                                #----------------------
                                if get_timer:
                                    print(f'\t2b. time: {time.time()-time_start} seconds')
                                    time_start = time.time()
                                #----------------------
                                    
                                # preprocess methods with input samples
                                _, curr_results_forward = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                                    haz_dict=curr_haz_dict,
                                    upstream_params=curr_upstream_params_forward,
                                    internal_params=curr_internal_params_forward,
                                    input_samples=curr_input_samples, 
                                    n_sample=num_epi_input_samples,
                                    n_site=n_site_to_use,
                                    level_to_run=level_to_run,
                                )
                                
                                #----------------------
                                if get_timer:
                                    print(f'\t2c. time: {time.time()-time_start} seconds')
                                    time_start = time.time()
                                #----------------------
                                    
                            # get tangent params
                            curr_intercept = {}
                            curr_slope = {}
                            curr_tangent_vector = {}
                            curr_sigma_mu_intercept = {}
                            for i,param in enumerate(curr_results):
                                # get intercept
                                curr_intercept[param] = curr_results[param]['mean_of_mu'][rows_to_keep_index[curr_step_str]].copy()
                                # get slope
                                if len(mean_of_mu[curr_case_str][prev_step_str]) == len(curr_haz_param):
                                    prev_param_to_use = list(mean_of_mu[curr_case_str][prev_step_str])[i]
                                else:
                                    prev_param_to_use = list(mean_of_mu[curr_case_str][prev_step_str])[0]
                                if prev_param_to_use in curr_upstream_params:
                                    numer = \
                                        curr_results_forward[param]['mean_of_mu'][rows_to_keep_index[curr_step_str]].copy() - \
                                        curr_results[param]['mean_of_mu'][rows_to_keep_index[curr_step_str]].copy()
                                    denom = \
                                        np.log(curr_upstream_params_forward[prev_param_to_use][rows_to_keep_index[curr_step_str],0].copy()) - \
                                        np.log(curr_upstream_params[prev_param_to_use][rows_to_keep_index[curr_step_str],0].copy())
                                    curr_slope[param] = numer / denom
                                else:
                                    # curr_slope[param] = np.zeros(n_site)
                                    curr_slope[param] = null_arr_nsite_nonzero[rows_to_keep_index[curr_step_str]].copy()
                            # read mean, sigma, sigmu_mu, tangent a and b from results
                            curr_mean_of_mu = {}
                            curr_sigma_of_mu = {}
                            curr_sigma = {}
                            curr_amu = {}
                            curr_bmu = {}
                            for i,param in enumerate(curr_haz_param):
                                # mean of mu
                                curr_mean_of_mu[param] = curr_results[param]['mean_of_mu'][rows_to_keep_index[curr_step_str]].copy()
                                # sigma of mu
                                curr_sigma_of_mu[param] = curr_results[param]['sigma_of_mu'][rows_to_keep_index[curr_step_str]].copy()
                                # sigma
                                curr_sigma[param] = curr_results[param]['sigma'][rows_to_keep_index[curr_step_str]].copy()
                                # amu
                                curr_amu[param] = curr_slope[param]
                                # bmu
                                if len(mean_of_mu[curr_case_str][prev_step_str]) == len(curr_haz_param):
                                    prev_param_to_use = list(mean_of_mu[curr_case_str][prev_step_str])[i]
                                else:
                                    prev_param_to_use = list(mean_of_mu[curr_case_str][prev_step_str])[0]
                                curr_bmu[param] = \
                                    curr_slope[param]*(
                                        -mean_of_mu[curr_case_str][prev_step_str][prev_param_to_use][rows_to_keep_index[curr_step_str]]
                                    ) + curr_intercept[param]
                                # if wells and caprocks, then set a = 0 and b = 1e-10 for no crossings
                                if infra_type == 'wells_caprocks' and 'well_strain' in workflow_i['haz_list']:
                                    curr_amu[param][rows_to_keep_index[curr_step_str]] = 0.0
                                    curr_bmu[param][rows_to_keep_index[curr_step_str]] = 1.e-10
                            # store mean of mu, sigma of mu, sigma
                            mean_of_mu[curr_case_str][curr_step_str] = curr_mean_of_mu.copy()
                            sigma_of_mu[curr_case_str][curr_step_str] = curr_sigma_of_mu.copy()
                            sigma[curr_case_str][curr_step_str] = curr_sigma.copy()
                            if step >= 1:
                                # get amu and bmu for PC
                                amu[curr_case_str][curr_step_str] = curr_amu.copy()
                                bmu[curr_case_str][curr_step_str] = curr_bmu.copy()
                
                #----------------------
                if get_timer:
                    print(f'\t2. time: {time.time()-time_start} seconds')
                    time_start = time.time()
                #----------------------
                
                ###########################################
                # set up for final step
                last_step = pbee_dim[curr_case_str]-1
                # last step string
                last_step_str = f'step_{last_step}'
                # current part in PBEE
                last_cat = workflow_i['cat_list'][last_step].lower()
                last_haz = workflow_i['haz_list'][last_step]
                last_haz_param = methods_dict[last_cat][last_haz]['return_params']
                # number of methods for hazard
                last_haz_dict = methods_dict[last_cat][last_haz].copy()
                last_methods = last_haz_dict['method']
                # previous category in PBEE
                prev_step_str = f'step_{last_step-1}'
                prev_cat = workflow_i['cat_list'][last_step-1].lower()
                prev_haz = workflow_i['haz_list'][last_step-1]
                # flag to run last step
                run_last_step = False
                if has_nonzero_mean:
                    run_last_step = True
                else:
                    if pbee_dim[curr_case_str] == 2:
                        run_last_step = True
                        curr_cat = last_cat
                        curr_hat = last_haz
                        curr_haz_param = last_haz_param
                # only run if there are sites with nonzero means
                if run_last_step:
                    # get n_site to use
                    n_site_to_use = len(rows_to_keep_rel_to_nonzero_step0[prev_step_str])
                    # get inputs for last step
                    last_param_names_all = methods_dict[last_cat][last_haz]['input_params']
                    last_param_internal = last_param_names_all.copy()
                    last_param_external = []
                    n_last_params = len(last_param_names_all)
                    last_input_samples = {}
                    # last_input_dist = {}
                    rows_inputs = rows_to_keep_rel_to_nonzero_step0[prev_step_str].copy()
                    for param in last_param_names_all:
                        if param in input_samples_nsite_nonzero:
                            # last_input_samples[param] = input_samples[param]
                            # last_input_samples[param] = input_samples[param][sites_with_nonzero_step0]
                            last_input_samples[param] = input_samples_nsite_nonzero[param][rows_inputs].copy()
                            last_param_external.append(param)
                            last_param_internal.remove(param)
                            # last_input_dist[param] = input_dist[param].copy()
                        # if param in input_dist_nsite_nonzero:
                        #     last_input_dist[param] = {}
                        #     for met in list(input_dist_nsite_nonzero[param]):
                        #         if met == 'dist_type':
                        #             last_input_dist[param][met] = input_dist_nsite_nonzero[param][met]
                        #         else:
                        #             last_input_dist[param][met] = input_dist_nsite_nonzero[param][met][rows_inputs].copy()
                    # upstream and internal params
                    last_upstream_params = {}
                    last_internal_params = {}
                    last_upstream_mean_params = {}
                    last_internal_mean_params = {}
                    # preprocess methods with input samples
                    _, last_results = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                        haz_dict=last_haz_dict,
                        upstream_params=last_upstream_params,
                        internal_params=last_internal_params,
                        input_samples=last_input_samples, 
                        n_sample=num_epi_input_samples,
                        n_site=n_site_to_use,
                        level_to_run=level_to_run,
                    )
                    # track nonzero sites for step0
                    nonzero_ind_from_out = []
                    # read from results
                    last_mean_of_mu = {}
                    last_sigma_of_mu = {}
                    last_sigma = {}
                    for i,param in enumerate(last_results):
                        # mean of mu
                        last_mean_of_mu[param] = last_results[param]['mean_of_mu'][rows_to_keep_index[prev_step_str]].copy()
                        # sigma of mu
                        last_sigma_of_mu[param] = last_results[param]['sigma_of_mu'][rows_to_keep_index[prev_step_str]].copy()
                        # sigma
                        last_sigma[param] = last_results[param]['sigma'][rows_to_keep_index[prev_step_str]].copy()
                    # store mean of mu, sigma of mu, sigma
                    mean_of_mu[curr_case_str][last_step_str] = last_mean_of_mu.copy()
                    sigma_of_mu[curr_case_str][last_step_str] = last_sigma_of_mu.copy()
                    sigma[curr_case_str][last_step_str] = last_sigma.copy()
                
                #----------------------
                if get_timer:
                    print(f'\t3. time: {time.time()-time_start} seconds')
                    time_start = time.time()
                #----------------------

                ###########################################
                # set up for and run PC
                # only run if last step is run
                if run_last_step is False:
                    # for each return parameter
                    for i, param_i in enumerate(last_haz_param):
                        # aggregate pc coefficients
                        if not param_i in pc_coeffs[curr_case_str]:
                            pc_coeffs[curr_case_str][param_i] = null_arr_pc_terms[curr_case_str].copy()
                        else:
                            # no action needed since just adding zeros
                            pass
                else:
                    # pc letter for step
                    pc_letter_for_step = {
                        0: 'Y',
                        1: 'Z',
                        2: 'T',
                        3: 'V'
                    }
                    # rows to use for PC
                    rows_to_use_pc = {}
                    # loop through steps
                    for step in range(pbee_dim[curr_case_str]-1, -1, -1):
                        step_str = f'step_{step}'
                        if step == pbee_dim[curr_case_str]-1:
                            step_up_str = f'step_{step-1}'
                            final_size = len(rows_to_keep_index[step_up_str])
                            rows_to_use_pc[step_str] = np.arange(final_size).astype(int)
                        elif step == pbee_dim[curr_case_str]-2:
                            rows_to_use_pc[step_str] = np.arange(final_size).astype(int)
                            rows_rel_to_nonzero_ref = rows_to_keep_rel_to_nonzero_step0[step_str]
                        else:
                            rows_rel_to_nonzero_curr = rows_to_keep_rel_to_nonzero_step0[step_str]
                            rows_to_use_pc[step_str] = np.asarray([
                                np.where(rows_rel_to_nonzero_curr==row)[0][0]
                                for row in rows_rel_to_nonzero_ref
                            ]).astype(int)
                    # initial step string
                    init_step_str = 'step_0'
                    # for each return parameter
                    for i, param_i in enumerate(last_haz_param):
                        # build input kwargs for pc function
                        pc_kwargs = {}
                        # loop through steps
                        for step in range(1, pbee_dim[curr_case_str]):
                            if step == 1:
                                params_for_step = list(mean_of_mu[curr_case_str][init_step_str])
                                if len(params_for_step) == len(curr_haz_param) and len(curr_haz_param) > 1:
                                    param_to_use = params_for_step[i]
                                else:
                                    param_to_use = params_for_step[0]
                                pc_kwargs[f'mu{pc_letter_for_step[0]}'] = \
                                    mean_of_mu[curr_case_str][init_step_str][param_to_use][rows_to_use_pc[init_step_str]]
                                pc_kwargs[f'sigmaMu{pc_letter_for_step[0]}'] = \
                                    sigma_of_mu[curr_case_str][init_step_str][param_to_use][rows_to_use_pc[init_step_str]]
                                pc_kwargs[f'sigma{pc_letter_for_step[0]}'] = \
                                    sigma[curr_case_str][init_step_str][param_to_use][rows_to_use_pc[init_step_str]]
                                # make sure sigmas are > 0 to avoid dividing by 0
                                pc_kwargs[f'sigma{pc_letter_for_step[0]}'] = np.maximum(pc_kwargs[f'sigma{pc_letter_for_step[0]}'],1e-4)
                            if step == pbee_dim[curr_case_str]-1:
                                pc_kwargs[f'mu{pc_letter_for_step[step]}'] = \
                                    mean_of_mu[curr_case_str][last_step_str][param_i][rows_to_use_pc[last_step_str]]
                                pc_kwargs[f'sigmaMu{pc_letter_for_step[step]}'] = \
                                    sigma_of_mu[curr_case_str][last_step_str][param_i][rows_to_use_pc[last_step_str]]
                                pc_kwargs[f'sigma{pc_letter_for_step[step]}'] = \
                                    sigma[curr_case_str][last_step_str][param_i][rows_to_use_pc[last_step_str]]
                            else:
                                step_str = f'step_{step}'
                                params_for_step = list(mean_of_mu[curr_case_str][step_str])
                                if running_below_ground and step == pbee_dim[curr_case_str]-2:
                                    if 'comp' in param_i:
                                        str_to_check = 'comp'
                                    elif 'tens' in param_i:
                                        str_to_check = 'tens'
                                    else:
                                        str_to_check = None
                                    if str_to_check is None:
                                        param_to_use = params_for_step[0] # use first param                                        
                                    else:
                                        for each in params_for_step:
                                            if str_to_check in each:
                                                param_to_use = each
                                                break
                                else:
                                    if len(params_for_step) == len(curr_haz_param) and len(curr_haz_param) > 1:
                                        param_to_use = params_for_step[i]
                                    else:
                                        param_to_use = params_for_step[0]
                                pc_kwargs[f'amu{pc_letter_for_step[step]}'] = \
                                    amu[curr_case_str][step_str][param_to_use][rows_to_use_pc[step_str]]
                                pc_kwargs[f'bmu{pc_letter_for_step[step]}'] = \
                                    bmu[curr_case_str][step_str][param_to_use][rows_to_use_pc[step_str]]
                                pc_kwargs[f'sigmaMu{pc_letter_for_step[step]}'] = \
                                    sigma_of_mu[curr_case_str][step_str][param_to_use][rows_to_use_pc[step_str]]
                                pc_kwargs[f'sigma{pc_letter_for_step[step]}'] = \
                                    sigma[curr_case_str][step_str][param_to_use][rows_to_use_pc[step_str]]
                            # make sure sigmas are > 0 to avoid dividing by 0
                            pc_kwargs[f'sigma{pc_letter_for_step[step]}'] = np.maximum(pc_kwargs[f'sigma{pc_letter_for_step[step]}'],1e-4)
                        # run function to get pc coefficients
                        # 1 integral
                        if pbee_dim[curr_case_str] == 2:
                            try:
                                pc_coeffs_param_i = pc_coeffs_single_int(**pc_kwargs)
                            except ZeroDivisionError:
                                raise ValueError("Somewhere in mean_of_mu there are invalid values; double check distribution")
                        # 2 integral
                        elif pbee_dim[curr_case_str] == 3:
                            try:
                                pc_coeffs_param_i = pc_coeffs_double_int(**pc_kwargs)
                            except ZeroDivisionError:
                                raise ValueError("Somewhere in mean_of_mu there are invalid values; double check distribution")
                        # 3 integrals
                        elif pbee_dim[curr_case_str] == 4:
                            try:
                                pc_coeffs_param_i = pc_coeffs_triple_int(**pc_kwargs)
                            except ZeroDivisionError:
                                raise ValueError("Somewhere in mean_of_mu there are invalid values; double check distribution")
                        # multiply by annual rate
                        pc_coeffs_param_i = pc_coeffs_param_i * rup_info['rate']
                        # map from n_site_curr_event to n_site
                        pc_coeffs_param_i_full = null_arr_pc_terms[curr_case_str].copy()
                        if running_cpt_based_procedure:
                            inds_from_step0_to_full = sites_to_run_curr_event[sites_with_nonzero_step0[rows_to_use_pc[init_step_str]]]
                        else:
                            inds_from_step0_to_full = sites_with_nonzero_step0[rows_to_use_pc[init_step_str]]
                        pc_coeffs_param_i_full[inds_from_step0_to_full,:] = pc_coeffs_param_i
                        # aggregate pc coefficients
                        if not param_i in pc_coeffs[curr_case_str]:
                            pc_coeffs[curr_case_str][param_i] = pc_coeffs_param_i_full
                        else:
                            pc_coeffs[curr_case_str][param_i] = pc_coeffs[curr_case_str][param_i] + pc_coeffs_param_i_full

                #----------------------
                if get_timer:
                    print(f'\t4. time: {time.time()-time_start} seconds')
                    time_start = time.time()
                    print(f'total time: {time.time()-time_initial} seconds')
                    print('\n')
                #----------------------
                
                ###########################################
                # store indices
                prev_rows_to_keep_index = rows_to_keep_index.copy()
                prev_rows_to_keep_rel_to_nonzero_step0 = rows_to_keep_rel_to_nonzero_step0.copy()
                prev_sites_to_keep = sites_to_keep.copy()
                # keep copies of the analysis setup params to determine if if need to rerun analysis
                prev_case_params = {}
                # step 0 params to track for next case
                prev_case_params[step0_str] = {
                    'cat': step0_cat,
                    'haz': step0_haz,
                    'methods': list(step0_methods)
                }
                # intermediate steps
                for step in range(1,pbee_dim[curr_case_str]-1):
                    # current category in PBEE
                    curr_step_str = f'step_{step}'
                    curr_cat = workflow_i['cat_list'][step].lower()
                    curr_haz = workflow_i['haz_list'][step]
                    curr_haz_param = methods_dict[curr_cat][curr_haz]['return_params']
                    # number of methods for hazard
                    curr_haz_dict = methods_dict[curr_cat][curr_haz].copy()
                    curr_methods = curr_haz_dict['method']
                    prev_case_params[curr_step_str] = {
                        'cat': curr_cat,
                        'haz': curr_haz,
                        'methods': list(curr_methods)
                    }
                # last step
                prev_case_params[last_step_str] = {
                    'cat': last_cat,
                    'haz': last_haz,
                    'methods': list(last_methods)
                }
        
        #----------------------
        # event_ind += 1
        event_counter += 1
        # if event_ind % display_after_n_event == 0:
        if event_counter % display_after_n_event == 0:
            # logging.info(f'\t\t- finished {event_ind} events: {np.round(time.time()-time_loop,decimals=1)} seconds...')
            logging.info(f'\t\t- finished {event_counter} events: {np.round(time.time()-time_loop,decimals=1)} seconds...')
            time_loop = time.time()
        #----------------------
            
    #----------------------
    total_time = time.time()-time_loop_init
    if total_time > 60:
        logging.info(f'\t>>>>> DONE - finished all {rupture_table.shape[0]} events: {np.round(total_time/60,decimals=1)} min')
    else:
        logging.info(f'\t>>>>> DONE - finished all {rupture_table.shape[0]} events: {np.round(total_time,decimals=1)} sec')
        # logging.info(f'\t---------------------------')
    time_loop = time.time()
    #----------------------
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # once pc coeffs are computed for all events, now go through cases again to generate samples and compute fractles

    for case_to_run in range(1,n_cases+1):
        # initialize
        pc_samples = {}
        # string for current case
        curr_case_str = f"case_{case_to_run}"
        # get workflow current case
        workflow_i = workflow_order_list[curr_case_str]
        # skip if caprock
        if 'caprock_leakage' in workflow_i['haz_list']:
            # load caprock crossing file
            # caprock_crossing = pd.read_csv(os.path.join(processed_input_dir,'caprock_crossing.csv'))
            caprock_crossing = read_file(os.path.join(processed_input_dir,'caprock_crossing.gpkg'))
            # get constant prob of leakage distribution
            output = methods_dict['dv']['caprock_leakage']['method']['ZhangEtal2022']._model()
            prob_leak_dist = np.asarray([
                -1.65, -1, 0, 1, 1.65, 0
            ]) * output['prob_leak']['sigma_mu'] + output['prob_leak']['mean']
            prob_leak_dist = np.round(prob_leak_dist/100,decimals=3) # convert to decimals and leave at 3 decimals
            # get list of annual rates
            ann_rates = rupture_table.annual_rate.values
            # initialize df_frac
            df_frac[curr_case_str] = pd.DataFrame(
                np.zeros((caprock_crossing.shape[0],len(prob_leak_dist))),
                columns=[
                    '5th_leakage',
                    '16th_leakage',
                    '50th_leakage',
                    '84th_leakage',
                    '95th_leakage',
                    'mean_leakage',
                ]
            )
            # go through list of caprock crossings to see which crossed a fault
            for i in range(caprock_crossing.shape[0]):
                if caprock_crossing.crossing_exist[i]:
                    # get list of faults crossed
                    fault_crossed = json.loads(caprock_crossing.faults_crossed[i])
                    sum_rates = sum(ann_rates[fault_crossed])
                    df_frac[curr_case_str].loc[i] = prob_leak_dist * sum_rates
            # create caprock ID list
            df_index = [f'caprock_{i+1}' for i in range(caprock_crossing.shape[0])]
            df_frac[curr_case_str].index = df_index
            df_geom = pd.DataFrame(
                caprock_crossing.geometry.values,
                columns=['geometry'],
                index=df_index
            )
            # concat
            df_frac[curr_case_str] = pd.concat([df_geom,df_frac[curr_case_str]],axis=1)
        else:
            # second to last step
            second_to_last_last_step = pbee_dim[curr_case_str]-2
            second_to_last_last_cat = workflow_i['cat_list'][second_to_last_last_step].lower()
            second_to_last_last_haz = workflow_i['haz_list'][second_to_last_last_step]
            if len(methods_dict) == 1:
                second_to_last_haz_param = {'pga': {}}
            else:
                second_to_last_haz_param = methods_dict[second_to_last_last_cat][second_to_last_last_haz]['return_params']
            # final step
            last_step = pbee_dim[curr_case_str]-1
            last_cat = workflow_i['cat_list'][last_step].lower()
            last_haz = workflow_i['haz_list'][last_step]
            last_haz_param = methods_dict[last_cat][last_haz]['return_params']
            # loop through params
            for param_i in last_haz_param:
                # multiply hermite polynomials to PC coefficients and sum up terms
                pc_samples[param_i] = np.inner(
                    hermite_prob_table_indep[curr_case_str].T,
                    pc_coeffs[curr_case_str][param_i]
                )
                # keep sum within 0 and 1
                pc_samples[param_i] = np.maximum(np.minimum(pc_samples[param_i],1),0)

            #----------------------
            if get_timer:   
                print(f'\t5. time: {time.time()-time_start} seconds')
                time_start = time.time()
            #----------------------
            
            # get fractiles
            df_frac[curr_case_str] = pd.DataFrame(None)
            for i,param_i in enumerate(last_haz_param):
                if running_below_ground:
                    comp_dir = None
                    for each in ['comp','tens']:
                        if each in param_i:
                            comp_dir = each
                            break
                    if comp_dir is None:
                        for each in second_to_last_haz_param:
                            if each in param_i:
                                param_to_use = each
                                break
                    else:
                        for each in second_to_last_haz_param:
                            if comp_dir in each:
                                param_to_use = each
                                break
                else:
                    if len(second_to_last_haz_param) > 1:
                        param_to_use = second_to_last_haz_param[i]
                    else:
                        param_to_use = second_to_last_haz_param[0]
                return_frac = pc_workflow.get_fractiles(
                    pc_samples[param_i],
                    n_sig_fig=4,
                )
                # add param to column name
                return_frac.columns = [f'{col}_{param_to_use}' for col in return_frac.columns]
                df_frac[curr_case_str] = pd.concat([df_frac[curr_case_str],return_frac],axis=1)
        
            #----------------------
            if get_timer:
                print(f'\t6. time: {time.time()-time_start} seconds')
                time_start = time.time()
            #----------------------
                
            # multiply results by probablity of crossing
            if flag_crossing_file_exists:
                df_frac[curr_case_str] = df_frac[curr_case_str] * np.tile(prob_crossing,(6,1)).T
            
            # if using crossings, check for multiple crossings per segment and pick worst case for segment
            if flag_possible_repeated_crossings:
            # if infra_type == 'below_ground':
                # initialize fractile table with all locations
                frac_full = pd.DataFrame(
                    0,
                    index=segment_index_full,
                    columns=df_frac[curr_case_str].columns
                )
                # remapping if possible for multiple crossings for same segment
                # algorithm changes if crossing is dependent on event
                if flag_event_dependent_crossing:
                    # loop through events
                    for event_ind, event_id in enumerate(event_ids_to_run):
                        # get segment id and ind for current event
                        rows_to_run = rows_to_run_by_event_id[event_id]
                        segment_ids_crossed = segment_ids_crossed_by_event_id[event_id]
                        segment_ids_crossed_repeat = segment_ids_crossed_repeat_by_event_id[event_id]
                        segment_index_repeat_in_full = segment_index_repeat_in_full_by_event_id[event_id]
                        segment_index_single_in_full = segment_index_single_in_full_by_event_id[event_id]
                        # get index to track segments with multiple crossings
                        df_frac_index = df_frac[curr_case_str].index.values[rows_to_run]
                        rows_to_run_index = list(range(len(rows_to_run)))
                        for ind, segment_id in enumerate(segment_ids_crossed_repeat):
                            ind_in_full_for_segment_id = segment_index_repeat_in_full[ind]
                            rows_with_repeat_seg = np.where(segment_ids_crossed==segment_id)[0]
                            # df_frac_index = np.asarray(list(set(df_frac_index).difference(set(rows))))
                            rows_to_run_index = np.asarray(list(set(rows_to_run_index).difference(set(rows_with_repeat_seg))))
                            # frac_repeat_curr_segment = df_frac[curr_case_str].loc[rows].reset_index(drop=True)
                            frac_repeat_curr_segment = df_frac[curr_case_str].loc[df_frac_index[rows_with_repeat_seg]].reset_index(drop=True)
                            # if fault rupture and below ground sum up between repeatin (primary and secondary)
                            if running_below_ground_fault_rupture:
                                frac_full.loc[ind_in_full_for_segment_id] = frac_repeat_curr_segment.values.sum(axis=0)
                            # else find worst case
                            else:
                                # pick case with higher mean value
                                worst_row = np.argmax(frac_repeat_curr_segment.iloc[:,-1])
                                frac_full.loc[ind_in_full_for_segment_id] = frac_repeat_curr_segment.loc[worst_row].values
                        # for all the segments with only 1 crossing
                        frac_full.loc[segment_index_single_in_full] = df_frac[curr_case_str].loc[df_frac_index[rows_to_run_index]].values
                # if not dependent on event
                else:
                    # get index to track segments with multiple crossings
                    df_frac_index = df_frac[curr_case_str].index.values
                    for ind, segment_id in enumerate(segment_ids_crossed_repeat):
                        ind_in_full_for_segment_id = segment_index_repeat_in_full[ind]
                        rows = np.where(segment_ids_crossed==segment_id)[0]
                        df_frac_index = np.asarray(list(set(df_frac_index).difference(set(rows))))
                        frac_repeat_curr_segment = df_frac[curr_case_str].loc[rows].reset_index(drop=True)
                        # pick case with higher mean value
                        worst_row = np.argmax(frac_repeat_curr_segment.iloc[:,-1])
                        frac_full.loc[ind_in_full_for_segment_id] = frac_repeat_curr_segment.loc[worst_row].values
                    # for all the segments with only 1 crossing
                    frac_full.loc[segment_index_single_in_full] = df_frac[curr_case_str].loc[df_frac_index].values
                # update df_frac
                df_frac[curr_case_str] = frac_full.copy()
            
            #----------------------
            if get_timer:
                print(f'\t7. time: {time.time()-time_start} seconds')
                time_start = time.time()
            #----------------------

            # update index for fractile dataframe
            if running_below_ground:
                tag = 'segment'
            elif running_wells_caprocks:
                tag = 'well'
            elif running_above_ground:
                tag = 'component'
            else:
                tag = 'site'
            # get row index names
            if flag_crossing_file_exists:
                index = [f'{tag}_{each}' for each in site_data_full.ID.values]
            else:
                index = [f'{tag}_{each}' for each in site_data.ID.values]
            df_frac[curr_case_str].index = index
            # set anything below 1e-10 to 0
            df_frac[curr_case_str][df_frac[curr_case_str]<1e-10] = 0
    
    logging.info(f'... DONE - Finished risk analysis')
    logging.info('---------------------------')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate summary table of analysis for export
    df_workflow = pd.DataFrame(None,index=list(workflow_order_list),columns=['IM','EDP','DM','DV'])
    for case in workflow_order_list:
        workflow_i = workflow_order_list[case]
        for j, col in enumerate(workflow_i['cat_list']):
            df_workflow[col][case] = workflow_i['haz_list'][j]
    logging.info(f'{counter}. Generated summary table for export')
    counter += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Additional processing for above ground components
    site_ind = np.arange(n_site)
    if running_above_ground:
        logging.info(f'{counter}. Performing additional postprocessing for above ground components...')
        counter += 1
        all_pc_case = list(df_frac)
        rot_type = {
            'elbow': ['4E90'],
            'tee': ['4TOP','4TIP'],
        }
        frac_strs = ['5th','16th','50th','84th','95th','mean']
        # find worst case
        for pc_case in all_pc_case:
            if df_workflow.loc[pc_case,'DV'] != 'vessel_rupture':
                cases = list(df_frac[pc_case])
                # get headers not about fractiles
                non_frac_headers = []
                for case in cases:
                    check = False
                    for frac in frac_strs:
                        if case.startswith(frac):
                            check = True
                            break
                    if check == False:
                        non_frac_headers.append(case)                
                # convert headers to np.array
                cases = np.asarray(cases)
                # initialize output dataframe
                df_frac[pc_case+'_worst_case'] = df_frac[pc_case][non_frac_headers].copy()
                # loop through subsystem cases
                for comp in ['elbow','tee']:
                    # create output string
                    out_str_comp = [f'{frac}_{comp}' for frac in frac_strs]
                    # get columns for current component
                    check_cols_with_mean_sys_comp = []
                    for h in cases:
                        check_if_has_str = False
                        for num in [2,3,4]:
                            for each_comp in rot_type[comp]:
                                if f"mean_eps_sys{num}_{each_comp}" in h:
                                    check_if_has_str = True
                                    break
                        check_cols_with_mean_sys_comp.append(check_if_has_str)
                    check_cols_with_mean_sys_comp = np.asarray(check_cols_with_mean_sys_comp)
                    cols_with_mean_sys_comp = cases[check_cols_with_mean_sys_comp]
                    # get means and find worst case = highest mean
                    worst_case_ind = np.argmax(df_frac[pc_case][cases[check_cols_with_mean_sys_comp]].values,axis=1)
                    worst_case = cols_with_mean_sys_comp[worst_case_ind]
                    worst_case_joint = [
                        string[string.find('joint'):string.find('joint')+len('joint')+1]
                        for string in worst_case
                    ]
                    # get values for worst case for all fractiles
                    worst_case_vals = []
                    for frac in frac_strs:
                        col_names_for_curr_frac = [string.replace('mean', frac) for string in cols_with_mean_sys_comp]
                        df_frac_sub = df_frac[pc_case][col_names_for_curr_frac].values
                        worst_case_vals.append([df_frac_sub[pair] for pair in list(zip(site_ind,worst_case_ind))])
                    # append to output dataframe
                    df_frac[pc_case+'_worst_case'][out_str_comp] = np.transpose(worst_case_vals)
                    # append column with joint with highest prob of failure
                    df_frac[pc_case+'_worst_case'][f'{comp}_worst_case_joint'] = worst_case_joint
        logging.info(f'... DONE - Finished performing additional postprocessing for above ground components')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create a table for PBEE notations:
    df_notation = pd.DataFrame(None,columns=['abbr','desc'])
    notation_dict = {
        'Abbrevations': 'Description',
        'IM': 'Intensity measure (e.g., PGA)',
        'EDP': 'Engineering demand parameter (e.g., ground displacement)',
        'DM': 'Damage measure (e.g., pipe strain)',
        'DV': 'Decision variable (e.g., probability of rupture)',
    }
    df_notation['abbr'] = list(notation_dict)
    df_notation['desc'] = [val for key,val in notation_dict.items()]
    # create a table with notes:
    df_notes = pd.DataFrame(None,columns=['index','notes'])
    notes = ['']
    notes.append('Results are provided as annual rate of occurrence (i.e., number of ocurrence per year).')
    notes.append('Results are given at the 5th, 16th, 50th (median), 84th, and 95th percentiles. The mean of the distribution is also provided.')
    notes.append('The decision variable for each case is given in the table above, along with its upstream dependencies.')
    df_notes['index'] = ['Notes'] + list(np.arange(len(notes)-1) + 1)
    df_notes['notes'] = notes
    # provide source for IM (e.g., UCERF, Shakemap)
    im_source_map = {
        'UCERF': 'Mean UCERF FM3.1 scenarios',
        'ShakeMap': 'ShakeMap scenario(s)',
        'UserDefinedRupture': 'User defined rupture scenario(s)',
        'UserDefinedGM': 'User defined ground motions',
    }
    df_im_source = pd.DataFrame(['Source for IM', im_source_map[im_source]],)
    # clean up df_workflow
    # update 'im' description in df_workflow with actual IM
    for i in range(df_workflow.shape[0]):
        if df_workflow['IM'].iloc[i] == 'im':
            if len(track_im_dependency_for_output[i]) == 1:
                df_workflow['IM'].iloc[i] = track_im_dependency_for_output[i][0]
            elif len(track_im_dependency_for_output[i]) > 1:
                df_workflow['IM'].iloc[i] = ", ".join(*track_im_dependency_for_output[i])
    # replace underscore with space in strings
    for i in range(df_workflow.shape[0]):
        for j in range(df_workflow.shape[1]):
            if isinstance(df_workflow.iloc[i,j],str):
                df_workflow.iloc[i,j] = df_workflow.iloc[i,j].replace('_',' ')
    # export and features
    dict_for_df_export = {
        'df_workflow': {
            'index': True,
            'header': True,
        },
        'df_notation': {
            'index': False,
            'header': False,
        },
        'df_im_source': {
            'index': False,
            'header': False,
        },
        'df_notes': {
            'index': False,
            'header': False,
        },
    }
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create a table for infrastructure locations:
    if flag_crossing_file_exists:
        if 'LON_MID' in site_data_full:
            df_locs = site_data_full[[
                'LON_BEGIN','LAT_BEGIN',
                'LON_END','LAT_END',
                'LON_MID','LAT_MID',
            ]].copy()
        else:
            df_locs = site_data_full[['LON','LAT']].copy()
    else:
        if 'LON_MID' in site_data:
            df_locs = site_data[[
                'LON_BEGIN','LAT_BEGIN',
                'LON_END','LAT_END',
                'LON_MID','LAT_MID',
            ]].copy()
        else:
            df_locs = site_data[['LON','LAT']].copy()
    # use same index from df_frac for df_locs
    df_locs.index = index
    # round lat lons
    df_locs = df_locs.round(decimals=6)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Export summary file
    logging.info(f'{counter}. Preparing result tables to export...')
    counter += 1
    # results directory
    sdir = os.path.join(work_dir,'Results')
    # clean prev outputs
    if clean_prev_output:
        if os.path.isdir(sdir):
            shutil.rmtree(sdir) # remove directory/start clean
        os.mkdir(sdir) # make directory
    else:
        if not os.path.isdir(sdir):
            os.mkdir(sdir) # make directory
    # formats
    rows_pad_btw_table = 2
    # store to one file
    flag_store_to_one_file = False
    if flag_store_to_one_file:
        # file path
        spath = os.path.join(sdir,f'results.excel')
        if os.path.exists(spath):
            os.remove(spath)
        # more formats for using one file
        float_format = r'%.3e'
        # float_format = r'%e'
        # set formatter to None for export
        pd.io.formats.excel.ExcelFormatter.header_style = None # remove
        with pd.ExcelWriter(spath) as writer:
            # case_summary tab
            row_start = 0
            for key,val in dict_for_df_export.items():
                locals()[key].to_excel(
                    writer,
                    sheet_name='case_summary',
                    startrow=row_start,
                    index=val['index'],
                    header=val['header']
                )
                row_end = row_start + locals()[key].shape[0] + val['header']
                row_start = row_end + rows_pad_btw_table
            # site location tab
            df_locs.to_excel(writer, sheet_name='locations')
            # case results
            for i,case in enumerate(workflow_order_list):
                df_frac[case].to_excel(writer, sheet_name=case, float_format=float_format)
                # df_frac[case].to_excel(writer, sheet_name=case)
            # for special cases for above ground:
            if case+'_combined' in df_frac:
                df_frac[case+'_combined'].to_excel(writer, sheet_name=case+'_combined', float_format=float_format)
                # df_frac[case+'_combined'].to_excel(writer, sheet_name=case+'_combined')
            if case+'_worst_case' in df_frac:
                df_frac[case+'_worst_case'].to_excel(writer, sheet_name=case+'_worst_case', float_format=float_format)
                # df_frac[case+'_worst_case'].to_excel(writer, sheet_name=case+'_worst_case')        
    else:
        # file path for notes
        spath = os.path.join(sdir,f'notes_for_cases.csv')
        if os.path.exists(spath):
            os.remove(spath)
        with open(spath,'a') as writer:
            for key,val in dict_for_df_export.items():
                try:
                    locals()[key].to_csv(
                        writer,
                        index=val['index'],
                        header=val['header'],
                        lineterminator='\n'
                    )
                except TypeError:
                    # catch backward compatibility with pandas for lineterminator
                    locals()[key].to_csv(
                        writer,
                        index=val['index'],
                        header=val['header'],
                        line_terminator='\n'
                    )
                for _ in range(rows_pad_btw_table):
                    writer.write("\n")
        # site location tab
        df_locs.to_csv(os.path.join(sdir,f'locations.csv'))
        # case results
        for i,case in enumerate(workflow_order_list):
            dv_str = df_workflow['DV'].iloc[i]
            dv_str = dv_str.replace(' ','_')
            df_frac[case].to_csv(os.path.join(sdir,f'{case}_{dv_str}.csv'))
            # for special cases for above ground:
            if case+'_combined' in df_frac:
                df_frac[case+'_combined'].to_csv(os.path.join(sdir,f'{case}_{dv_str}_combined.csv'))
            if case+'_worst_case' in df_frac:
                df_frac[case+'_worst_case'].to_csv(os.path.join(sdir,f'{case}_{dv_str}_worst_case.csv'))
    logging.info(f'... Exported result tables to directory:')
    logging.info(f'\t{sdir}')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Export gpkg file with mean fractiles
    logging.info(f'{counter}. Preparing a summary geopackage with primary results...')
    counter += 1
    col_name_remap = {
        'comp': 'compressive',
        'tee': 'tee-joint',
        'worst_case': '(worst_combination)',
        'vessel': 'pressure_vessel',
    }
    # export path
    spath = os.path.join(sdir,'analysis_summary.gpkg')
    if os.path.exists(spath):
        os.remove(spath)
    # get list of cases in df_frac, get all mean columns
    cases_in_df_frac = sorted(list(df_frac)) # also sort by alphabetical order
    # tracking what is in gpkg
    gpkg_contains = []
    
    # for below ground, everything fits into one summary sheet (same number of rows):
    if running_below_ground:
        # create a gpkg file to store mean fractiles
        if 'LON_MID' in df_locs:
            gdf_frac_mean = GeoDataFrame(
                None,
                crs=4326,
                geometry=make_list_of_linestrings(
                    pt1_x=df_locs.LON_BEGIN.values,
                    pt1_y=df_locs.LAT_BEGIN.values,
                    pt2_x=df_locs.LON_END.values,
                    pt2_y=df_locs.LAT_END.values,
                )
            )
        else:
            gdf_frac_mean = GeoDataFrame(
                None,
                crs=4326,
                geometry=points_from_xy(
                    x=df_locs.LON.values,
                    y=df_locs.LAT.values,
                )
            )
        gdf_frac_mean['SegmentID'] = index
        # for each case in df_frac, get all mean columns
        for i, case in enumerate(cases_in_df_frac):
            case_num = int(case[-1])
            dv_str = df_workflow['DV'].iloc[case_num-1]
            dv_str = dv_str.replace(' ','_')
            for col in df_frac[case].columns:
                if 'mean_' in col:
                    # new_col_name = f'{case}-{dv_str}-{col}'
                    new_col_name = f'{case}: {dv_str}'
                    new_col_name = remap_str(new_col_name, col_name_remap)
                    new_col_name = new_col_name.replace('_',' ')
                    gdf_frac_mean[new_col_name] = df_frac[case][col].values
        # export
        gdf_frac_mean.to_file(spath, layer='mean_annual_rate_of_failure', index=False, crs=4326)
        
    # for wells and caprocks - one sheet for wells, one sheet for caprocks if exists
    if running_wells_caprocks:
        gdf_frac_mean = {}
        # first get mean fractile summary for wells
        gdf_frac_mean['mean_annual_rate_of_failure_for_wells'] = GeoDataFrame(
            None,
            crs=4326,
            geometry=points_from_xy(
                x=df_locs.LON.values,
                y=df_locs.LAT.values,
            )
        )
        gdf_frac_mean['mean_annual_rate_of_failure_for_wells']['WellID'] = index
        # for each case in df_frac, get all mean columns
        for i, case in enumerate(cases_in_df_frac):
            case_num = int(case[-1])
            dv_str = df_workflow['DV'].iloc[case_num-1]
            dv_str = dv_str.replace(' ','_')
            if not 'caprock_leakage' in workflow_order_list[case]['haz_list']:
                for col in df_frac[case].columns:
                    if 'mean_' in col:
                        # new_col_name = f'{case}-{dv_str}-{col}'
                        new_col_name = f'{case}: {dv_str}'
                        new_col_name = remap_str(new_col_name, col_name_remap)
                        new_col_name = new_col_name.replace('_',' ')
                        gdf_frac_mean['mean_annual_rate_of_failure_for_wells'][new_col_name] = df_frac[case][col].values
        # for each case in df_frac, get all mean columns
        for i, case in enumerate(cases_in_df_frac):
            if 'caprock' in workflow_order_list[case]['haz_list']:
                case_num = int(case[-1])
                dv_str = df_workflow['DV'].iloc[case_num-1]
                dv_str = dv_str.replace(' ','_')
                # next get mean fractile summary for caprocks
                if not 'mean_annual_rate_of_failure_for_caprocks' in gdf_frac_mean:
                    gdf_frac_mean['mean_annual_rate_of_failure_for_caprocks'] = GeoDataFrame(
                        None,
                        crs=4326,
                        geometry=caprock_crossing.geometry.values
                    )
                for col in df_frac[case].columns:
                    if 'mean_' in col:
                        # new_col_name = f'{case}-{dv_str}-{col}'
                        new_col_name = f'{case}: {dv_str} - {col}'
                        new_col_name = remap_str(new_col_name, col_name_remap)
                        new_col_name = new_col_name.replace('_',' ')
                        gdf_frac_mean['mean_annual_rate_of_failure_for_caprocks'][new_col_name] = df_frac[case][col].values
        # export
        for layer in gdf_frac_mean:
            gdf_frac_mean[layer].to_file(spath, layer=layer, index=False, crs=4326)
    
    # for above ground, everything fits into one summary sheet (same number of rows):
    if running_above_ground:
        # create a gpkg file to store mean fractiles
        gdf_frac_mean = GeoDataFrame(
            None,
            crs=4326,
            geometry=points_from_xy(
                x=df_locs.LON.values,
                y=df_locs.LAT.values,
            )
        )
        gdf_frac_mean['ComponentID'] = index
        # for each case in df_frac, get all mean columns
        for case in cases_in_df_frac:
            if case+'_worst_case' in cases_in_df_frac:
                pass
            else:
                if 'worst_case' in case:
                    case_num = int(case[case.find('worst_case')-2])
                else:
                    case_num = int(case[-1])
                dv_str = df_workflow['DV'].iloc[case_num-1]
                dv_str = dv_str.replace(' ','_')
                for col in df_frac[case].columns:
                    # mean fractile
                    if 'mean_' in col:
                        if 'elbow' in col or 'tee' in col:
                            # new_col_name = f'{case}-{dv_str}-{col}'
                            new_col_name = f'{case}: {dv_str} @ {col.replace("mean_","")}'
                        else:
                            new_col_name = f'{case}: {dv_str}'
                        new_col_name = remap_str(new_col_name, col_name_remap)
                        new_col_name = new_col_name.replace('_',' ')
                        gdf_frac_mean[new_col_name] = df_frac[case][col].values
                    # worst case joint
                    # if 'worst_case' in col:
                    #     new_col_name = f'{case}-{dv_str}-{col}'
                    #     new_col_name = f'{case}: {dv_str}'
                    #     new_col_name = remap_str(new_col_name, col_name_remap)
                    #     new_col_name = new_col_name.replace('_',' ')
                    #     gdf_frac_mean[new_col_name] = df_frac[case][col].values
        # export
        gdf_frac_mean.to_file(spath, layer='mean_annual_rate_of_failure', index=False, crs=4326)
    gpkg_contains.append('mean fractiles from call cases')
    
    # append other gpkg to gdf_frac_mean if they exist
    # 1) rupture metadata
    gdf_rupture_table = read_file(os.path.join(im_dir,'RUPTURE_METADATA.gpkg'))
    gdf_rupture_table.to_file(spath, layer='rupture_traces', index=False, crs=4326)
    gpkg_contains.append('rupture scenarios close to sites')
    
    # 2) site data table with crossings only
    if flag_crossing_file_exists:
        # update previous string in gpkg_contains
        gpkg_contains[-1] = 'rupture scenarios crossed and/or close to sites'
        gdf_crossings_only = read_file(os.path.join(processed_input_dir,'site_data_PROCESSED_CROSSING_ONLY.gpkg'))
        gdf_crossings_only.to_file(spath, layer=f'locations_with_crossings', index=False, crs=4326)
        gpkg_contains.append('locations with crossings')
        if os.path.exists(os.path.join(processed_input_dir,'deformation_polygons_crossed.gpkg')):
            gdf_def_poly = read_file(os.path.join(processed_input_dir,'deformation_polygons_crossed.gpkg'))
            gdf_def_poly.to_file(spath, layer=f'deformation_polygons_crossed', index=False, crs=4326)
            gpkg_contains.append('deformation polygons with crossings')
    
    # 3) qfaults if running below_ground and surface_fault_rupture
    if running_below_ground_fault_rupture:
        for each in ['primary','secondary']:
            gdf_qfault_primary = read_file(os.path.join(im_dir,'qfaults_crossed.gpkg'),layer=each)
            gdf_qfault_primary.to_file(spath, layer=f'qfault_crossed_{each}', index=False, crs=4326)
        gpkg_contains.append('qfaults crossed')
    
    # 4) if running CPTBased
    if running_cpt_based_procedure:
        # processed CPT
        gdf_processed_cpt = read_file(os.path.join(processed_input_dir,'CPTs','cpt_data_PROCESSED.gpkg'))
        gdf_processed_cpt.to_file(spath, layer=f'processed_cpts', index=False, crs=4326)
        gpkg_contains.append('processed CPTs')
        # if freeface feature is given
        if 'PathToFreefaceDir' in setup_config['UserSpecifiedData']['CPTParameters']:
            freeface_fpath = setup_config['UserSpecifiedData']['CPTParameters']['PathToFreefaceDir']
            # check length of filepath: if ==0, then assume nothing was provided
            if len(freeface_fpath) == 0:
                freeface_fpath = None
            else:
                freeface_fpath = check_and_get_abspath(freeface_fpath, input_dir)
        if freeface_fpath is not None:
            gdf_freeface_wgs84 = read_file(freeface_fpath, crs=epsg_wgs84)
            gdf_freeface_wgs84.to_file(spath, layer=f'freeface_features', index=False, crs=4326)
            gpkg_contains.append('freeface features')
            
    # 5) if running caprock analysis
    if running_caprock:
        if os.path.exists(os.path.join(processed_input_dir,'caprock_crossing.gpkg')):
            gdf_caprock_crossing = read_file(os.path.join(processed_input_dir,'caprock_crossing.gpkg'))
            gdf_caprock_crossing.to_file(spath, layer=f'caprocks_with_crossings', index=False, crs=4326)
            gpkg_contains.append('caprocks with crossings')            

    logging.info(f'{counter}. Exported a summary geopackage to:')
    logging.info(f'\t{spath}')
    logging.info(f'The summary geopackage contains the following:')
    for each in gpkg_contains:
        logging.info(f'\t- {each}')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # End of analysis
    logging.info('---------------')
    logging.info('******** End of OpenSRA Analysis ********')
    logging.info('---------------')
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Main function for OpenSRA'
    )
    
    # Define arguments
    
    # input directory
    # parser.add_argument('-i', '--input', help='Path of input directory"]')
    parser.add_argument('-w', '--workdir', help='Path to working/analysis directory')
    
    # clean previous analysis
    # parser.add_argument('-c', '--clean', help='Clean previous analysis: "y" or "n" (default)', default='n')
    
    # logging
    # parser.add_argument('-l', '--logging', help='Logging level: "info"(default) or "debug"', default='info')
    
    # infrastructure file type
    parser.add_argument('-l', '--logging',
                        help='Logging message detail: "s" for simple or "d" for detailed',
                        default='s', type=str)
    
    # infrastructure file type
    parser.add_argument('-d', '--display',
                        help='Display a message every n scenarios',
                        default=10, type=int)
    
    # infrastructure file type
    parser.add_argument('-c', '--clean',
                        help='Clean "Results" directory from previous OpenSRA run if exists',
                        default=True, type=bool)
    
    ###########################
    # display temporary timer messages
    parser.add_argument('-t', '--timer',
                        help='temporary: get timer',
                        default=False, type=bool)
    
    ###########################
    
    # Parse command line input
    args = parser.parse_args()
    
    # Run "Main"
    main(
        work_dir = args.workdir,
        # clean_prev_run = args.clean,
        # logging_level = args.logging,
        display_after_n_event=args.display,
        logging_message_detail=args.logging,
        clean_prev_output=args.clean,
        get_timer=args.timer,
    )