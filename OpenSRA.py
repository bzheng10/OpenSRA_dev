# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Creates workflow
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
# -----------------------------------------------------------


# Python Interpreter
#! /usr/bin/env python3


# -----------------------------------------------------------
# Python base modules
import argparse
import json
import logging
import os
import sys
import time

# scientific processing modules
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import truncnorm, norm

# precompiling
from numba_stats import truncnorm as nb_truncnorm
from numba_stats import norm as nb_norm

# OpenSRA modules and functions
from src.site import site_util
from src.util import set_logging, lhs
from src.pc_func import pc_util, pc_workflow
from src.pc_func.pc_coeffs_single_int import pc_coeffs_single_int
from src.pc_func.pc_coeffs_double_int import pc_coeffs_double_int
from src.pc_func.pc_coeffs_triple_int import pc_coeffs_triple_int


# -----------------------------------------------------------
# Main function
# def main(input_dir, clean_prev_run, logging_level):
def main(work_dir, logging_level='info', logging_message_detail='simple',
         display_after_n_event=100, get_timer=False):

    # -----------------------------------------------------------
    # Setting logging level (e.g. DEBUG or INFO)
    set_logging(
        level=logging_level,
        msg_format=logging_message_detail
    )
    logging.info('\n---------------')
    
    # -----------------------------------------------------------
    logging.info('******** OpenSRA Analysis********\n')
    counter = 1 # counter for stages of processing

    # -----------------------------------------------------------
    # Define primary directories - these should be created during Preprocess
    input_dir = os.path.join(work_dir,'Input')
    processed_input_dir = os.path.join(work_dir,'Processed_Input')
    im_dir = os.path.join(work_dir,'IM')
    if os.path.exists(input_dir) is False:
        logging.info('URGENT: Missing input directory; create the folder and run "Proprocess".')
        logging.info(f'\t- OpenSRA will now exit.')
        sys.exit()
    if os.path.exists(processed_input_dir) is False or os.path.exists(im_dir) is False:
        logging.info('URGENT: Missing generated directories; first run "Preprocess".')
        logging.info(f'\t- OpenSRA will now exit.')
        sys.exit()
    logging.info(f'{counter}. Identified primary directories')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import site data
    # check for files with crossing
    if 'site_data_PROCESSED_CROSSING_ONLY.csv' in os.listdir(processed_input_dir):
        site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED_CROSSING_ONLY.csv'))
        site_data_full = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED.csv'))
        flag_crossing_file = True
        # get probability of crossing
        prob_crossing = site_data.prob_crossing.values
        # if prob crossing == 1, then using deformation polygons and multiple crossings per segment is possible
        # if prob crossing == 0.25, then no geometry was used and only 1 crossing per segment
        if prob_crossing[0] == 1:
            possible_repeated_crossings = True
        else:
            possible_repeated_crossings = False
        
        # get segment IDs
        segment_ids_full = site_data_full.ID.values
        segment_ids_crossed = site_data.ID.values
        # get table indices corresponding to IDs
        segment_index_full = site_data_full.index.values
        segment_index_crossed = site_data.index.values
        
        # get repeated index
        if possible_repeated_crossings:
            # get unique crossings
            segment_ids_crossed_unique, counts = np.unique(segment_ids_crossed, return_counts=True)
            segment_ids_crossed_repeat = segment_ids_crossed_unique[np.where(counts>1)[0]]
            # segments with only 1 crossing
            segment_ids_crossed_single = np.asarray(
                list(set(segment_ids_crossed_unique).difference(set(segment_ids_crossed_repeat))))
            segment_ids_crossed_single = np.sort(segment_ids_crossed_single)
            
            # find row index corresponding to repeated IDS in full table
            segment_index_repeat_in_full = np.asarray([
                np.where(segment_ids_full==seg_ind)[0][0]
                for seg_ind in segment_ids_crossed_repeat
            ])
            segment_index_single_in_full = np.asarray([
                np.where(segment_ids_full==seg_ind)[0][0]
                for seg_ind in segment_ids_crossed_single
            ])
        else:
            # no repeated 
            segment_ids_crossed_repeat = np.asarray([])
            segment_index_repeat_in_full = np.asarray([])
            segment_ids_crossed_single = segment_ids_crossed.copy()
            segment_index_single_in_full = segment_index_crossed.copy()

        # print('\n')
        # print(segment_ids_full)
        # print(segment_ids_crossed)
        # print(segment_ids_crossed_repeat)
        # print(segment_ids_crossed_single)
        
        # print('\n')
        # print(segment_index_full)
        # print(segment_index_crossed)
        # print(segment_index_repeat_in_full)
        # print(segment_index_single_in_full)
        
    else:
        site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED.csv'))
        flag_crossing_file = False
    logging.info(f'{counter}. Loaded site data file')
    counter += 1
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Seismic Events
    # Import IM distributions
    im_import = {}
    for each in ['pga','pgv']:
        im_import[each] = {
            # 'mean_table': pd.read_csv(os.path.join(im_dir,each.upper(),'MEAN.csv'),header=None),
            # 'sigma_table': pd.read_csv(os.path.join(im_dir,each.upper(),'ALEATORY.csv'),header=None),
            # 'sigma_mu_table': pd.read_csv(os.path.join(im_dir,each.upper(),'EPISTEMIC.csv'),header=None)
            
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
    # get sites with 0 intensities across all locations
    # print(len(np.unique(np.where(im_import['pga']['mean_table']-10<=-7)[1])))
    # sys.exit()
    # Import rupture information
    rupture_table = pd.read_csv(os.path.join(im_dir,'RUPTURE_METADATA.csv'))
    # Get Number of sites
    n_site = im_import[list(im_import)[0]]['mean_table'].shape[1]
    # Make some arrays to be used later
    n_site_ind_arr = np.arange(n_site)
    null_arr_nsite = np.zeros(n_site)
    ones_arr_nsite = np.ones(n_site)
    logging.info(f'{counter}. Loaded seismic event files')
    counter += 1
    
    # -----------------------------------------------------------
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
    # Check if CPT is needed - make toggle if running CPT-based procedure
    if "edp" in methods_dict and \
        "liquefaction" in methods_dict['edp'] and \
        "CPTBased" in methods_dict['edp']['liquefaction']['method']:
        running_cpt_based_procedure = True
        logging.info(f'{counter}. Flagged to run CPT procedure')
        counter += 1
    else:
        running_cpt_based_procedure = False
    
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
    logging.info(f'{counter}. Loaded required parameters')
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
    if '0sigma' in work_dir:
        num_epi_input_samples = 1
    else:    
        num_epi_input_samples = 50
    # Number of Epistemic samples for fractiles
    num_epi_fractile_samples = 1000
    # Make some arrays to be used later
    null_arr_nsite_by_ninput = np.zeros((n_site,num_epi_input_samples))
    ones_arr_nsite_by_ninput = np.ones((n_site,num_epi_input_samples))
    twos_arr_nsite_by_ninput = ones_arr_nsite_by_ninput.copy() * 2
    ones_arr_nfractile_sample = np.ones(num_epi_fractile_samples)
    logging.info(f'{counter}. Initialized analysis metrics')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get samples for input parameters
    n_params = len(input_dist)
    param_names = list(input_dist)
    
    time_start = time.time()
    input_samples = \
        pc_workflow.get_samples_for_params(input_dist, num_epi_input_samples, n_site)
    if get_timer:
        print(f'\ttime to get samples for input: {time.time()-time_start} seconds')
    
    # for liq susc cat
    if 'liquefaction' in workflow['EDP'] and \
        'Hazus2020' in workflow['EDP']['liquefaction'] and \
        not 'liq_susc' in input_dist:
        if 'gw_depth' in input_samples:
            input_samples['liq_susc'] = site_util.get_regional_liq_susc(
                input_table.GeologicUnit_Witter2006.copy(),
                input_table.GeologicUnit_BedrossianEtal2012.copy(),
                input_samples['gw_depth']
            )
            gw_depth_mean = input_dist['gw_depth']['mean'].copy()
            if input_dist['gw_depth']['dist_type'] == 'lognormal':
                gw_depth_mean = np.exp(gw_depth_mean)
            input_dist['liq_susc'] = {
                'value': site_util.get_regional_liq_susc(
                    input_table.GeologicUnit_Witter2006.copy(),
                    input_table.GeologicUnit_BedrossianEtal2012.copy(),
                    gw_depth_mean,
                    get_mean=True
                ),
                'dist_type': 'fixed'
            }

    # if crossing algorithm is performed
    # initialize params for storing additional sampling
    addl_input_dist = {}
    crossing_params_intermediate = []
    if flag_crossing_file:
        # if below ground, then perform additional sampling using crossing angles
        if infra_type == 'below_ground':
            if 'landslide' in workflow['EDP']:
                # check level to run
                if level_to_run == 1:
                    # get default values
                    primary_mech = np.empty((n_site,num_epi_input_samples), dtype="<U20")
                    primary_mech[:] = 'SSComp'
                    # for weighting between mechanisms in transition zone
                    transition_weight_factor = ones_arr_nsite_by_ninput.copy()
                    # store to input samples
                    input_samples['primary_mech'] = primary_mech
                    input_samples['transition_weight_factor'] = transition_weight_factor
                    crossing_params_intermediate.append('primary_mech')
                    crossing_params_intermediate.append('transition_weight_factor')
                
                else:
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
                    cond_lt20_or_gt_160 = np.logical_or(
                        beta_crossing_samples <= 20,
                        beta_crossing_samples >= 160
                    )
                    # 2) transition (20 to 45 or 135 to 160)
                    cond_transition_beta = np.logical_or(
                        np.logical_and(
                            beta_crossing_samples > 20,
                            beta_crossing_samples <= 45
                        ),
                        np.logical_and(
                            beta_crossing_samples >= 135,
                            beta_crossing_samples < 160
                        )
                    )
                    # 3) pure strike-slip (45 to 135)
                    cond_btw_45_135 = np.logical_and(
                        beta_crossing_samples > 45,
                        beta_crossing_samples < 135
                    )
                    cond_ge_90 = beta_crossing_samples >= 90
                    cond_lt_90 = beta_crossing_samples < 90
                    
                    # initialize additional crossing logic params
                    primary_mech = np.empty(beta_crossing_samples.shape, dtype="<U20")
                    # for weighting between mechanisms in transition zone
                    transition_weight_factor = np.ones(beta_crossing_samples.shape)
                    # determine primary mechanism(s) based on above conditions
                    # 1) crossing within +/- 20 deg
                    # -- head scarp
                    cond_joint = cond_lt20_or_gt_160 & cond_scarp
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'Normal'
                    # -- toe
                    cond_joint = cond_lt20_or_gt_160 & cond_toe
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'Reverse'
                    # -- body, further determine by whether above or below 90
                    cond_joint = cond_lt20_or_gt_160 & cond_body
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
                    # 3) pure strike-slip (45 to 135)
                    # --- determine by whether above or below 90
                    cond_joint = cond_btw_45_135 & cond_ge_90
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'SSComp'
                    cond_joint = cond_btw_45_135 & cond_lt_90
                    if True in cond_joint:
                        primary_mech[cond_joint] = 'SSTens'
                    
                    # get linear scale factor for transition zones
                    cond_joint = cond_transition_beta & cond_ge_90
                    if True in cond_joint:
                        beta_at_transition_ge90 = beta_crossing_samples[cond_joint]
                        transition_weight_factor[cond_joint] = 1 - (160-beta_at_transition_ge90)/(160-135)
                    cond_joint = cond_transition_beta & cond_lt_90
                    if True in cond_joint:
                        beta_at_transition_lt90 = beta_crossing_samples[cond_joint]
                        transition_weight_factor[cond_joint] = (45-beta_at_transition_lt90)/(45-20)
                    
                    # store to input samples
                    input_samples['primary_mech'] = primary_mech
                    input_samples['transition_weight_factor'] = transition_weight_factor
                    crossing_params_intermediate.append('primary_mech')
                    crossing_params_intermediate.append('transition_weight_factor')
                    
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
                                np.logical_or(
                                    primary_mech=='Normal_SSComp',
                                    primary_mech=='Normal_SSTens',
                                )
                            )
                            if True in cond:
                                means[cond] = 65
                                lows[cond] = 45
                                highs[cond] = 90
                            # for any case with reverse slip (pure or transition)
                            cond = np.logical_or(
                                primary_mech=='Reverse',
                                np.logical_or(
                                    primary_mech=='Reverse_SSComp',
                                    primary_mech=='Reverse_SSTens',
                                )
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
                            res = lhs(
                                n_site=n_site,
                                n_var=1,
                                n_samp=num_epi_input_samples
                            )[:,:,0]
                            samples = truncnorm.ppf(
                                q=norm.cdf(res,0,1),
                                a=(lows-means)/sigmas,
                                b=(highs-means)/sigmas,
                                loc=means,
                                scale=sigmas
                            )
                            
                            # store samples to input_samples
                            input_samples[param] = samples
    
    #
    # print(input_samples['primary_mech'])
    # print(input_samples['psi_dip'])
    # print(input_samples['transition_weight_factor'])

    logging.info(f'{counter}. Sampled input parameters')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup and run PC
    if 'OBJ_ID' in site_data:
        sites_to_run = list(site_data.OBJ_ID.values)
    else:
        sites_to_run = list(range(site_data.shape[0]))
    # mean_of_mu_track = []
    
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
    time_loop_init = time.time()
    time_loop = time.time()
    logging.info('\t---------------------------')
    logging.info('\t>>>>> Starting PC workflow...')
    for event_ind in range(rupture_table.shape[0]):
    # for event_ind in range(1):
        
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
        
        # run if not CPT-based
        if running_cpt_based_procedure is False:
            
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
                im_dist_info[each]['sigma'] = np.maximum(
                    im_dist_info[each]['sigma'],0.001
                )        
        
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
        
        # get list of sites with no well crossing
        if infra_type == 'wells_caprocks':
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
        if flag_crossing_file or infra_type == 'wells_caprocks':
            # initialize params for storing additional sampling
            addl_input_dist = {}
            null_arr_nsite_nonzero = np.zeros(n_site_curr_event)
            ones_arr_nsite_nonzero = np.ones(n_site_curr_event)
        
            # if wells and caprocks, then perform additional sampling using fault depths and crossing angles
            if infra_type == 'wells_caprocks':
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
            
            if event_ind == 0:
                track_im_dependency_for_output[case_to_run-1] = []
            
            # for caprock specifically
            if 'caprock_leakage' in workflow_i['haz_list']:
                
                # pass and run this later since caprock leakage is not dependent on IMs
                pass
            
                # only do this once, since caprock leakage is not dependent on IMs
                # if event_ind == 0:
                #     # load caprock crossing file
                #     caprock_crossing = pd.read_csv(os.path.join(
                #         processed_input_dir,'caprock_crossing.csv'
                #     ))
                #     # get constant prob of leakage distribution
                #     output = methods_dict['dv']['caprock_leakage']['method']['ZhangEtal2022']._model()
                #     prob_leak_dist = np.asarray([
                #         -1.65, -1, 0, 1, 1.65, 0
                #     ]) * output['prob_leak']['sigma_mu'] + output['prob_leak']['mean']
                #     prob_leak_dist = np.round(prob_leak_dist/100,decimals=3) # convert to decimals and leave at 3 decimals
                #     # get list of annual rates
                #     ann_rates = rupture_table.annual_rate.values
                #     # initialize df_frac
                #     df_frac[curr_case_str] = pd.DataFrame(
                #         np.zeros((caprock_crossing.shape[0],len(prob_leak_dist))),
                #         columns=[
                #             '5th_leakage',
                #             '16th_leakage',
                #             '50th_leakage',
                #             '84th_leakage',
                #             '95th_leakage',
                #             'mean_leakage',
                #         ]
                #     )
                #     # go through list of caprock crossings to see which crossed a fault
                #     for i in range(caprock_crossing.shape[0]):
                #         if caprock_crossing.crossing_exist[i]:
                #             # get list of faults crossed
                #             fault_crossed = json.loads(caprock_crossing.faults_crossed[i])
                #             sum_rates = sum(ann_rates[fault_crossed])
                #             df_frac[curr_case_str].loc[i] = prob_leak_dist * sum_rates
                #     # create caprock ID list
                #     df_index = [
                #         f'caprock_{i+1}'
                #         for i in range(caprock_crossing.shape[0])
                #     ]
                #     df_frac[curr_case_str].index = df_index
                #     df_geom = pd.DataFrame(
                #         caprock_crossing.geometry.values,
                #         columns=['geometry'],
                #         index=df_index
                #     )
                #     # concat
                #     df_frac[curr_case_str] = pd.concat([df_geom,df_frac[curr_case_str]],axis=1)

            else:
                # only do this once to initialize PC background params
                if event_ind == 0:
                    
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
                
                # continue with rest of PC setup
                
                # default to PGA if CPT-based
                if running_cpt_based_procedure:
                    prev_haz_param = ['pga']
                    
                # run if not CPT-based
                else:
                    # set up for step 1
                    # previous category in PBE
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
                        
                        # print(f'skipping {step0_str} for {curr_case_str}')
                        
                        mean_of_mu[curr_case_str][step0_str] = mean_of_mu[prev_case_str][step0_str].copy()
                        sigma_of_mu[curr_case_str][step0_str] = sigma_of_mu[prev_case_str][step0_str].copy()
                        sigma[curr_case_str][step0_str] = sigma[prev_case_str][step0_str].copy()
                        track_im_dependency_for_output[case_to_run-1] = track_im_dependency_for_output[case_to_run-2].copy()
                        
                    else:
                        # setup and run analysis
                    
                        # current category in PBEE
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
                            if event_ind == 0:
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
                            step0_input_dist = {}
                            for param in step0_param_names_all:
                                if param in input_samples:
                                    step0_input_samples[param] = input_samples_nsite_nonzero[param].copy()
                                    step0_input_dist[param] = input_dist_nsite_nonzero[param].copy()
                                    step0_param_external.append(param)
                                    step0_param_internal.remove(param)
                            
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
                                n_site=n_site_curr_event
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
                            curr_results = cpt_based_curr_results.copy()
                            curr_mean_for_tangent = cpt_based_curr_mean_for_tangent.copy()
                        # if running_cpt_based_procedure is False:
                        else:
                            # special case for "lateral spread" and "settlement", which require "liquefaction" to be first assessed.
                            if curr_haz == 'lateral_spread' or curr_haz == 'settlement':
                                liq_haz_dict = methods_dict['edp']['liquefaction'].copy()
                                liq_methods = liq_haz_dict['method']
                                
                                # step1_param_names_liq = ['vs30','precip','dist_coast','gw_depth','dist_river','dist_water']
                                liq_input_param_names = methods_dict['edp']['liquefaction']['input_params']
                                liq_input_samples = {}
                                liq_input_dist = {}
                                for param in liq_input_param_names:
                                    liq_input_samples[param] = input_samples_nsite_nonzero[param].copy()
                                    liq_input_dist[param] = input_dist_nsite_nonzero[param].copy()
                                
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
                                        get_liq_susc=False
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
                                        get_liq_susc=True
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
                                        get_liq_susc=False
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
                                        get_liq_susc=True
                                    )
                            
                            # get additional params for evaluation
                            curr_param_names_all = methods_dict[curr_cat][curr_haz]['input_params']
                            curr_param_internal = curr_param_names_all.copy()
                            curr_param_external = []
                            n_curr_params = len(curr_param_names_all)
                            curr_input_samples = {}
                            curr_input_dist = {}
                            
                            # rows to use for inputs
                            rows_inputs = rows_to_keep_rel_to_nonzero_step0[prev_step_str].copy()
                            # go through params
                            for param in curr_param_names_all:
                                if param in input_samples_nsite_nonzero:
                                    curr_input_samples[param] = input_samples_nsite_nonzero[param][rows_inputs].copy()
                                    curr_param_external.append(param)
                                    curr_param_internal.remove(param)
                                if param in input_dist_nsite_nonzero:
                                    if param == 'liq_susc' and param in input_samples_nsite_nonzero:
                                        curr_input_dist[param] = {}
                                        for met in list(input_dist_nsite_nonzero[param]):
                                            if met == 'dist_type':
                                                curr_input_dist[param][met] = input_dist_nsite_nonzero[param][met]
                                            else:
                                                curr_input_dist[param][met] = input_dist_nsite_nonzero[param][met][rows_inputs].copy()
                                    else:
                                        if not param in crossing_params_intermediate:
                                            curr_input_dist[param] = {}
                                            for met in list(input_dist_nsite_nonzero[param]):
                                                if met == 'dist_type':
                                                    curr_input_dist[param][met] = input_dist_nsite_nonzero[param][met]
                                                else:
                                                    curr_input_dist[param][met] = input_dist_nsite_nonzero[param][met][rows_inputs].copy()
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
                            
                            if get_timer:
                                print(f'\t2a-1. time: {time.time()-time_start} seconds')
                                time_start = time.time()
                            
                            # preprocess methods with input samples
                            _, curr_results = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                                haz_dict=curr_haz_dict,
                                upstream_params=curr_upstream_params,
                                internal_params=curr_internal_params,
                                input_samples=curr_input_samples, 
                                n_sample=num_epi_input_samples,
                                n_site=n_site_to_use,
                            )
                            
                            if get_timer:
                                print(f'\t2a. time: {time.time()-time_start} seconds')
                                time_start = time.time()
                            
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
                                
                                # print(f'skipping {curr_step_str} for {curr_case_str}')
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
                                
                                if get_timer:
                                    print(f'\t2b. time: {time.time()-time_start} seconds')
                                    time_start = time.time()
                                    
                                # preprocess methods with input samples
                                _, curr_results_forward = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                                    haz_dict=curr_haz_dict,
                                    upstream_params=curr_upstream_params_forward,
                                    internal_params=curr_internal_params_forward,
                                    input_samples=curr_input_samples, 
                                    n_sample=num_epi_input_samples,
                                    n_site=n_site_to_use,
                                )
                                
                                if get_timer:
                                    print(f'\t2c. time: {time.time()-time_start} seconds')
                                    time_start = time.time()
                        
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
                            
                        # read from results
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
                                curr_slope[param]*(-mean_of_mu[curr_case_str][prev_step_str][prev_param_to_use][rows_to_keep_index[curr_step_str]]) + curr_intercept[param]
                        
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
                
                if get_timer:
                    print(f'\t2. time: {time.time()-time_start} seconds')
                    time_start = time.time()
                
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

                # only run if there are sites with nonzero means
                if has_nonzero_mean:
                    # get n_site to use
                    n_site_to_use = len(rows_to_keep_rel_to_nonzero_step0[prev_step_str])

                    # get inputs for last step
                    last_param_names_all = methods_dict[last_cat][last_haz]['input_params']
                    last_param_internal = last_param_names_all.copy()
                    last_param_external = []
                    n_last_params = len(last_param_names_all)
                    last_input_samples = {}
                    last_input_dist = {}
                    rows_inputs = rows_to_keep_rel_to_nonzero_step0[prev_step_str].copy()
                    for param in last_param_names_all:
                        if param in input_samples_nsite_nonzero:
                            # last_input_samples[param] = input_samples[param]
                            # last_input_samples[param] = input_samples[param][sites_with_nonzero_step0]
                            last_input_samples[param] = input_samples_nsite_nonzero[param][rows_inputs].copy()
                            last_param_external.append(param)
                            last_param_internal.remove(param)
                            # last_input_dist[param] = input_dist[param].copy()
                        if param in input_dist_nsite_nonzero:
                            last_input_dist[param] = {}
                            for met in list(input_dist_nsite_nonzero[param]):
                                if met == 'dist_type':
                                    last_input_dist[param][met] = input_dist_nsite_nonzero[param][met]
                                else:
                                    last_input_dist[param][met] = input_dist_nsite_nonzero[param][met][rows_inputs].copy()
                            
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
                        n_site=n_site_to_use
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
                
                if get_timer:
                    print(f'\t3. time: {time.time()-time_start} seconds')
                    time_start = time.time()

                ########################
                # set up for and run PC

                # only run if there are sites with nonzero means
                if has_nonzero_mean is False:
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
                        # build kwargs for pc
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
                                if infra_type == 'below_ground' and step == pbee_dim[curr_case_str]-2:
                                    if 'comp' in param_i:
                                        str_to_check = 'comp'
                                    elif 'tens' in param_i:
                                        str_to_check = 'tens'
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

                        # get pc coefficients
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
                                for each in input_samples:
                                    print(each, input_samples[each])
                                print(mean_of_mu)
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
                        pc_coeffs_param_i_full[sites_with_nonzero_step0[rows_to_use_pc[init_step_str]],:] = pc_coeffs_param_i
                        
                        # aggregate pc coefficients
                        if not param_i in pc_coeffs[curr_case_str]:
                            pc_coeffs[curr_case_str][param_i] = pc_coeffs_param_i_full
                        else:
                            pc_coeffs[curr_case_str][param_i] = pc_coeffs[curr_case_str][param_i] + pc_coeffs_param_i_full

                if get_timer:
                    print(f'\t4. time: {time.time()-time_start} seconds')
                    time_start = time.time()
                    print(f'total time: {time.time()-time_initial} seconds')
                    print('\n')
                
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
                                
        event_ind += 1
        if event_ind % display_after_n_event == 0:
            logging.info(f'\t\t- finished {event_ind} events: {np.round(time.time()-time_loop,decimals=1)} seconds...')
            time_loop = time.time()
            
    total_time = time.time()-time_loop_init
    if total_time > 60:
        logging.info(f'\t>>>>> finished all {rupture_table.shape[0]} events: {np.round(total_time/60,decimals=1)} min')
    else:
        logging.info(f'\t>>>>> finished all {rupture_table.shape[0]} events: {np.round(total_time,decimals=1)} sec')
        logging.info(f'\t---------------------------')
    time_loop = time.time()
            
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
            caprock_crossing = pd.read_csv(os.path.join(
                processed_input_dir,'caprock_crossing.csv'
            ))
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
            df_index = [
                f'caprock_{i+1}'
                for i in range(caprock_crossing.shape[0])
            ]
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
            second_to_last_haz_param = methods_dict[second_to_last_last_cat][second_to_last_last_haz]['return_params']
            
            # final step
            last_step = pbee_dim[curr_case_str]-1
            last_cat = workflow_i['cat_list'][last_step].lower()
            last_haz = workflow_i['haz_list'][last_step]
            last_haz_param = methods_dict[last_cat][last_haz]['return_params']
                    
            # loop through params
            for param_i in last_haz_param:
                # sum up pc terms
                pc_samples[param_i] = np.inner(
                    hermite_prob_table_indep[curr_case_str].T,
                    pc_coeffs[curr_case_str][param_i]
                )
                # keep sum within 0 and 1
                pc_samples[param_i] = np.maximum(np.minimum(pc_samples[param_i],1),0)
                        
            if get_timer:   
                print(f'\t5. time: {time.time()-time_start} seconds')
                time_start = time.time()
            
            # get fractiles
            df_frac[curr_case_str] = pd.DataFrame(None)
            for i,param_i in enumerate(last_haz_param):
                if infra_type == 'below_ground':
                    for each in ['comp','tens']:
                        if each in param_i:
                            comp_dir = each
                            break
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
                    # infra_type=infra_type,
                    # site_id=site_data.ID.values
                    n_sig_fig=4,
                )
                # add param to column name
                return_frac.columns = [f'{col}_{param_to_use}' for col in return_frac.columns]
                df_frac[curr_case_str] = pd.concat([df_frac[curr_case_str],return_frac],axis=1)
        
            if get_timer:
                print(f'\t6. time: {time.time()-time_start} seconds')
                time_start = time.time()
                
            # if using crossings, check for multiple crossings per segment and pick worst case for segment
            # also multiply by prob of crossing
            if flag_crossing_file:
                if infra_type == 'below_ground':
                    if 'landslide' in workflow['EDP']:
                        # multiply results by probablity of crossing
                        df_frac[curr_case_str] = df_frac[curr_case_str] * np.tile(prob_crossing,(6,1)).T
                        
                        # initialize fractile table with all locations
                        frac_full = pd.DataFrame(
                            0,
                            index=segment_index_full,
                            columns=df_frac[curr_case_str].columns
                        )
                        
                        # get index to track segments with single crossings
                        df_frac_index = df_frac[curr_case_str].index.values
                        
                        # for segment_id in segment_ids_crossed_unique:
                        for ind, segment_id in enumerate(segment_ids_crossed_repeat):
                            ind_in_full_for_segment_id = segment_index_repeat_in_full[ind]
                            rows = np.where(segment_ids_crossed==segment_id)[0]
                            df_frac_index = np.delete(df_frac_index,rows)
                            frac_repeat_curr_segment = df_frac[curr_case_str].loc[rows].reset_index(drop=True)
                            # pick case with higher mean value
                            worst_row = np.argmax(frac_repeat_curr_segment.iloc[:,-1])
                            frac_full.loc[ind_in_full_for_segment_id] = frac_repeat_curr_segment.loc[worst_row].values
                            
                        # for all the segments with only 1 crossing
                        frac_full.loc[segment_index_single_in_full] = df_frac[curr_case_str].loc[df_frac_index].values

                        # update df_frac
                        df_frac[curr_case_str] = frac_full.copy()
            
            if get_timer:
                print(f'\t7. time: {time.time()-time_start} seconds')
                time_start = time.time()
            
            # update index for fractile dataframe
            if infra_type == 'below_ground':
                tag = 'segment'
            elif infra_type == 'wells_caprocks':
                tag = 'well'
            elif infra_type == 'above_ground':
                tag = 'component'
            else:
                tag = 'site'
            if flag_crossing_file:
                index = [f'{tag}_{each}' for each in site_data_full.ID.values]
            else:
                index = [f'{tag}_{each}' for each in site_data.ID.values]
            df_frac[curr_case_str].index = index

            # set anything below 1e-10 to 0
            df_frac[curr_case_str][df_frac[curr_case_str]<1e-10] = 0
    
    logging.info(f'{counter}. Performed risk analysis using PC')
    counter += 1
    
    # -----------------------------------------------------------
    # Generate summary table of analysis for export
    df_workflow = pd.DataFrame(None,index=list(workflow_order_list),columns=['IM','EDP','DM','DV'])
    for case in workflow_order_list:
        workflow_i = workflow_order_list[case]
        for j, col in enumerate(workflow_i['cat_list']):
            df_workflow[col][case] = workflow_i['haz_list'][j]
    logging.info(f'{counter}. Generated summary table for export')
    counter += 1

    # -----------------------------------------------------------
    # Additional processing for above ground components
    site_ind = np.arange(n_site)
    if infra_type == 'above_ground':
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

                # intiialize
                # df_curr_pc_case = df_frac[pc_case][non_frac_headers].copy()
                
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
        logging.info(f'{counter}. Performed additional processing for above ground components')
        counter += 1


    # -----------------------------------------------------------
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
    df_im_source = pd.DataFrame(
        ['Source for IM', im_source_map[im_source]],
    )
    
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
    

    # -----------------------------------------------------------
    # create a table for infrastructure locations:
    if flag_crossing_file:
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
    # get annotated index
    # for case_to_run in range(1,n_cases+1):
    #     if df_frac[curr_case_str].shape[0] == df_locs.shape[0]:
    df_locs.index = index
    # round lat lons
    df_locs = df_locs.round(decimals=6)
    
    
    # -----------------------------------------------------------
    # Export summary file
    sdir = os.path.join(work_dir,'Results')
    if not os.path.isdir(sdir):
        os.mkdir(sdir)
    ###################################
    # spath = os.path.join(sdir,f'results.xlsx')
    # spath = os.path.join(sdir,f'results.csv')
    # formats
    rows_pad_btw_table = 2
    # float_format = r'%.3e'
    # float_format = r'%e'
    # set formatter to None for export
    # pd.io.formats.excel.ExcelFormatter.header_style = None # remove
    # with pd.ExcelWriter(spath) as writer:
    #     # case_summary tab
    #     row_start = 0
    #     for key,val in dict_for_df_export.items():
    #         locals()[key].to_excel(
    #             writer,
    #             sheet_name='case_summary',
    #             startrow=row_start,
    #             index=val['index'],
    #             header=val['header']
    #         )
    #         row_end = row_start + locals()[key].shape[0] + val['header']
    #         row_start = row_end + rows_pad_btw_table
    
    spath = os.path.join(sdir,f'notes_for_cases.csv')
    if os.path.exists(spath):
        os.remove(spath)
    with open(spath,'a') as writer:
        for key,val in dict_for_df_export.items():
            locals()[key].to_csv(
                writer,
                index=val['index'],
                header=val['header'],
                line_terminator='\n'
            )
            for _ in range(rows_pad_btw_table):
                writer.write("\n")
    ###################################
    
    # site location tab
    # df_locs.to_excel(writer, sheet_name='locations')
    df_locs.to_csv(os.path.join(sdir,f'locations.csv'))
    # case tab
    for i,case in enumerate(workflow_order_list):
        # df_frac[case].to_excel(writer, sheet_name=case, float_format=float_format)
        # df_frac[case].to_excel(writer, sheet_name=case)
        dv_str = df_workflow['DV'].iloc[i]
        dv_str = dv_str.replace(' ','_')
        df_frac[case].to_csv(os.path.join(sdir,f'{case}_{dv_str}.csv'))
    # for case in workflow_order_list:
        if case+'_combined' in df_frac:
            # df_frac[case+'_combined'].to_excel(writer, sheet_name=case+'_combined', float_format=float_format)
            # df_frac[case+'_combined'].to_excel(writer, sheet_name=case+'_combined')
            df_frac[case+'_combined'].to_csv(os.path.join(sdir,f'{case}_{dv_str}_combined.csv'))
        if case+'_worst_case' in df_frac:
            # df_frac[case+'_worst_case'].to_excel(writer, sheet_name=case+'_worst_case', float_format=float_format)
            # df_frac[case+'_worst_case'].to_excel(writer, sheet_name=case+'_worst_case')
            df_frac[case+'_worst_case'].to_csv(os.path.join(sdir,f'{case}_{dv_str}_worst_case.csv'))
    
                
    logging.info(f'{counter}. Exported results table to:')
    logging.info(f'\t{spath}')
    counter += 1

    # -----------------------------------------------------------
    # Exit Program
    logging.info('\n******** OpenSRA Analysis********')
    logging.info('---------------\n')
    

# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Main function for OpenSRA'
    )
    
    # Define arguments
    
    # input directory
    # parser.add_argument('-i', '--input', help='Path of input directory"]')
    parser.add_argument('-w', '--work_dir', help='Path to working/analysis directory')
    
    # clean previous analysis
    # parser.add_argument('-c', '--clean', help='Clean previous analysis: "y" or "n" (default)', default='n')
    
    # logging
    # parser.add_argument('-l', '--logging', help='Logging level: "info"(default) or "debug"', default='info')
    
    # infrastructure file type
    parser.add_argument('-l', '--logging_detail',
                        help='Logging message detail: "simple" (default) or "full"',
                        default='simple', type=str)
    
    # infrastructure file type
    parser.add_argument('-d', '--display_after_n_event',
                        help='Display a message every n scenarios',
                        default=100, type=int)
    
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
        work_dir = args.work_dir,
        # clean_prev_run = args.clean,
        # logging_level = args.logging,
        display_after_n_event=args.display_after_n_event,
        logging_message_detail=args.logging_detail,
        get_timer=args.timer,
    )