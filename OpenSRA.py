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
import os
import logging
import sys
import argparse
import json
import time

# scientific processing modules
import numpy as np
import pandas as pd

# OpenSRA modules and functions
from src.site import site_util
from src.util import set_logging
from src.pc_func import pc_util, pc_workflow
from src.pc_func.pc_coeffs_single_int import pc_coeffs_single_int
from src.pc_func.pc_coeffs_double_int import pc_coeffs_double_int
from src.pc_func.pc_coeffs_triple_int import pc_coeffs_triple_int


# -----------------------------------------------------------
# Main function
# def main(input_dir, clean_prev_run, logging_level):
def main(work_dir, logging_level='info'):

    # -----------------------------------------------------------
    # Setting logging level (e.g. DEBUG or INFO)
    set_logging(logging_level)
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
    if 'site_data_PROCESSED_CROSSING_ONLY.csv' in os.listdir(processed_input_dir):
        site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED_CROSSING_ONLY.csv'))
    else:
        site_data = pd.read_csv(os.path.join(processed_input_dir,'site_data_PROCESSED.csv'))
    logging.info(f'{counter}. Loaded site data file')
    counter += 1
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Seismic Events
    # Import IM distributions
    im_import = {}
    for each in ['pga','pgv']:
        im_import[each] = {
            'mean_table': pd.read_csv(os.path.join(im_dir,each.upper(),'MEAN.csv'),header=None),
            'sigma_table': pd.read_csv(os.path.join(im_dir,each.upper(),'ALEATORY.csv'),header=None),
            'sigma_mu_table': pd.read_csv(os.path.join(im_dir,each.upper(),'EPISTEMIC.csv'),header=None)
        }
    # Import rupture information
    rupture_table = pd.read_csv(os.path.join(im_dir,'RUPTURE_METADATA.csv'))
    # Get Number of sites
    n_site = im_import[list(im_import)[0]]['mean_table'].shape[1]
    logging.info(f'{counter}. Loaded seismic event files')
    counter += 1
    
    # -----------------------------------------------------------
    # Start setup configuration file
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
    logging.info(f'{counter}. Initialized analysis metrics')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get samples for input parameters
    n_params = len(input_dist)
    param_names = list(input_dist)
    input_samples = \
        pc_workflow.get_samples_for_params(input_dist, num_epi_input_samples, n_site)
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
    logging.info(f'{counter}. Sampled input parameters')
    counter += 1
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup and run PC
    if 'OBJ_ID' in site_data:
        sites_to_run = list(site_data.OBJ_ID.values)
    else:
        sites_to_run = list(range(site_data.shape[0]))
    mean_of_mu_track = []
    
    # initialize for tracking
    df_frac = {}
    sites_with_no_crossing = None
    track_im_dependency_for_output = []
    
    # loop through cases to run
    for case_to_run in range(1,n_cases+1):

        # get workflow current case
        workflow_i = workflow_order_list[f"case_{case_to_run}"]
        print(f'\tRunning case {case_to_run}...')
        
        # don't need to run PC if caprock
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
            df_frac[f'case_{case_to_run}'] = pd.DataFrame(
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
                    df_frac[f'case_{case_to_run}'].loc[i] = prob_leak_dist * sum_rates
            # create caprock ID list
            df_index = [
                f'caprock_{i+1}'
                for i in range(caprock_crossing.shape[0])
            ]
            df_frac[f'case_{case_to_run}'].index = df_index
            df_geom = pd.DataFrame(
                caprock_crossing.geometry.values,
                columns=['geometry'],
                index=df_index
            )
            # concat
            df_frac[f'case_{case_to_run}'] = pd.concat([df_geom,df_frac[f'case_{case_to_run}']],axis=1)
        
        else:
            # inputs scenario
            pbee_dim = workflow_i['n_pbee_dim']
            pc_order = 4
            num_pc_terms_indep = pc_util.num_pc_terms(pbee_dim,pc_order)
            index_pc_table_indep = pc_util.index_table_function(pbee_dim,pc_order)
            xi_order_linear = np.arange(pbee_dim) + 1
            
            # pre-calculate hermite probs for epistemic samples
            epi_samples_for_pc = np.random.normal(size=(num_epi_fractile_samples,pbee_dim))
            hermite_prob_table_indep = np.zeros((num_pc_terms_indep,num_epi_fractile_samples))

            # loop through number of independent pc terms to get Hermite probability
            for i in range(num_pc_terms_indep):
                hermite_prob_table_indep_i = np.ones(num_epi_fractile_samples)
                for j in range(pbee_dim):
                    hermite_prob_table_indep_i *= \
                        pc_util.hermite_prob(epi_samples_for_pc[:,pbee_dim-1-j], index_pc_table_indep[i,j])
                hermite_prob_table_indep[i,:] = hermite_prob_table_indep_i
            
            # initialize pc_coeffs dictionary
            pc_coeffs = {}
            
            # print('\n')
            # time_initial = time.time()
            # time_start = time_initial
            
            # event to run
            for event_ind in range(rupture_table.shape[0]):
                
                # run if not CPT-based
                if running_cpt_based_procedure is False:
                    # im
                    im_dist_info = {}
                    for each in ['pga','pgv']:
                        im_dist_info[each] = {
                            'mean': im_import[each]['mean_table'].loc[event_ind,:].values,
                            'sigma': im_import[each]['sigma_table'].loc[event_ind,:].values,
                            'sigma_mu': im_import[each]['sigma_mu_table'].loc[event_ind,:].values,
                            'dist_type': 'lognormal'
                        }
                        # avoid sigma = 0 for PC
                        im_dist_info[each]['sigma'] = np.maximum(
                            im_dist_info[each]['sigma'],0.001
                        )

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
                        # if key == 'intersecting_well_segment_ind':
                        #     print(type(rupture_table[key][event_ind]))
                        #     print(type(json.loads(rupture_table[key][event_ind])))
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
                                sites_with_no_crossing = list(set(np.arange(n_site)).difference(set(sites_with_crossing)))
                                break
                
                # if wells and caprocks, then perform additional sampling using fault depths and crossing angles
                if infra_type == 'wells_caprocks':
                    well_crossing_params = [
                        'theta', # fault angle (deg)
                        'z_crossing', # fault depth (m)
                        'e_rock', # Young's modulus of rock (GPa)
                    ]
                    addl_input_dist = {}
                    null_array = np.zeros(n_site)
                    ones_array = np.ones(n_site)
                    # go through input dist and see which input params require sampling
                    for param in well_crossing_params:
                        # get dist_type
                        param_dist_type = input_dist[param]['dist_type']
                        # get mean/value from input dist and see if any site is flagged with "event_dependent"
                        if param_dist_type == 'fixed':
                            param_value = input_dist[param]['value']
                            if param_value[0] == 'event_dependent':
                                if param == 'e_rock':
                                    # param_val_arr = null_array.copy()
                                    # param_val_arr[rup_info['well_ind_crossed']] = rup_info['well_ind_crossed']
                                    addl_input_dist[param] = {
                                        'value': addl_input_dist['z_crossing']['value']*1.6E-3 + 15.2,
                                        'dist_type': 'fixed'
                                    }
                                else:
                                    param_val_arr = null_array.copy()
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
                                # sigmas = ones_array.copy()
                                # sigmas[rup_info['well_ind_crossed']] = input_dist[param]['sigma'][rup_info['well_ind_crossed']]
                                sigmas = input_dist[param]['sigma'].copy()
                                if len(sites_with_crossing) > 0:
                                    # low
                                    lows = null_array.copy()
                                    lows[sites_with_crossing] = input_dist[param]['low'][sites_with_crossing]
                                    # high
                                    highs = null_array.copy()
                                    highs[sites_with_crossing] = input_dist[param]['high'][sites_with_crossing]
                                else:
                                    if param_dist_type == 'normal':
                                        lows = np.zeros((n_site))
                                    elif param_dist_type == 'lognormal':
                                        lows = np.ones((n_site)) * -np.inf
                                    highs = np.ones((n_site)) * np.inf
                                # if e_rock
                                if param == 'e_rock':
                                    # mean
                                    param_val_arr = addl_input_dist['z_crossing']['mean']*1.6E-3 + 15.2
                                    param_val_arr[addl_input_dist['theta']['mean']==0] = 0
                                else:
                                    # mean
                                    param_val_arr = null_array.copy()
                                    if len(sites_with_crossing) > 0:
                                        param_val_arr[sites_with_crossing] = rup_info[param]
                                    else:
                                        param_val_arr = np.zeros((n_site))
                                # make dist
                                addl_input_dist[param] = {
                                    'mean': param_val_arr,
                                    'sigma': sigmas,
                                    'low': lows,
                                    'high': highs,
                                    'dist_type': param_dist_type
                                }
                                # additional metrics
                                # for each in ['low','high','dist_type']:
                                    # addl_input_dist[param][each] = input_dist[param][each]
                                # also update the metrics in input_dist
                                input_dist[param]['mean'] = addl_input_dist[param]['mean'].copy()
                                input_dist[param]['low'] = addl_input_dist[param]['low'].copy()
                                input_dist[param]['high'] = addl_input_dist[param]['high'].copy()
                    # perform sampling on additional params and append to existing samples
                    # if case_to_run == 2:
                    #     for each in addl_input_dist:
                    #         print(each)
                    #         for met in addl_input_dist[each]:
                    #             print(addl_input_dist[each][met])
                    #     # print('##############################################')
                    #     for each in input_dist:
                    #         print(each)
                    #         for met in input_dist[each]:
                    #             print(input_dist[each][met])
                    if len(addl_input_dist) > 0:
                        addl_input_samples = \
                            pc_workflow.get_samples_for_params(addl_input_dist, num_epi_input_samples, n_site)
                        for each in addl_input_samples:
                            if each == 'theta':
                                addl_input_samples[each][sites_with_no_crossing,:] = 0
                            input_samples[each] = addl_input_samples[each]
                
                # run if not CPT-based
                if running_cpt_based_procedure:
                    prev_haz_param = ['pga']
                else:
                    # initialize
                    param_dist_type = {}
                    mean_of_mu = {}
                    sigma_of_mu = {}
                    sigma = {}

                    # set up for step 1
                    # previous category in PBE
                    step0 = 0
                    step0_str = f'step_{step0}'
                    step0_cat = workflow_i['cat_list'][step0].lower()
                    step0_haz = workflow_i['haz_list'][step0]
                    if step0_haz != 'im':
                        step0_haz_dict = methods_dict[step0_cat][step0_haz].copy()
                        step0_methods = step0_haz_dict['method']

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
                        mean_of_mu[step0_str] = {}
                        sigma_of_mu[step0_str] = {}
                        sigma[step0_str] = {}
                        prev_haz_param = []
                        # for method in list(step1_methods):
                        for param in step1_haz_dict['upstream_params']:
                            # get param dists
                            if param == 'pga' or param == 'pgv':
                                mean_of_mu[step0_str][param] = im_dist_info[param]['mean']
                                sigma_of_mu[step0_str][param] = im_dist_info[param]['sigma_mu']
                                sigma[step0_str][param] = im_dist_info[param]['sigma']
                                prev_haz_param.append(param)
                        # default to PGA as domain
                        if not 'pga' in step1_haz_dict['upstream_params'] and \
                            not 'pgv' in step1_haz_dict['upstream_params']:
                                mean_of_mu[step0_str]['pga'] = im_dist_info['pga']['mean']
                                sigma_of_mu[step0_str]['pga'] = im_dist_info['pga']['sigma_mu']
                                sigma[step0_str]['pga'] = im_dist_info['pga']['sigma']
                                prev_haz_param.append('pga')
                        if 'pga' in prev_haz_param:
                            track_im_dependency_for_output.append('pga'.upper())
                        elif 'pgv' in prev_haz_param:
                            track_im_dependency_for_output.append('pgv'.upper())

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
                                step0_input_samples[param] = input_samples[param]
                                step0_input_dist[param] = input_dist[param].copy()
                                step0_param_external.append(param)
                                step0_param_internal.remove(param)
                        # pull upstream params for full analysis
                        step0_upstream_params = {}
                        for param in step0_haz_dict['upstream_params']:
                            step0_upstream_params[param] = np.ones((n_site,num_epi_input_samples))*rup_info[param]
                            # print(step0_upstream_params[param])
                            # print(step0_upstream_params[param].shape)
                        
                        # pull internal params, e.g., prob_liq and liq_susc
                        step0_internal_params = {}
                        
                        # get mean of mu for domain vector
                        _, step0_results = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                            haz_dict=step0_haz_dict,
                            upstream_params=step0_upstream_params,
                            internal_params=step0_internal_params,
                            input_samples=step0_input_samples, 
                            n_sample=num_epi_input_samples,
                            n_site=n_site
                        )
                        # print(step0_results['pgdef']['mean_of_mu'].shape)
                        
                        # if contains epistemic sampling
                        # if np.ndim(step0_results[''])
                        # collapse mean_of_mu vector
                        # step0_results[param_for_domain]['mean_of_mu'] = \
                        #     np.mean(step0_results[param_for_domain]['mean_of_mu'],axis=0)
                        # # collapse sigma_mu vector
                        # step0_results[param_for_domain]['sigma_of_mu'] = \
                        #     np.mean(step0_results[param_for_domain]['sigma_of_mu'],axis=0)
                        # get other metrics
                        mean_of_mu[step0_str] = {param_for_domain: step0_results[param_for_domain]['mean_of_mu']}
                        sigma_of_mu[step0_str] = {param_for_domain: step0_results[param_for_domain]['sigma_of_mu']}
                        sigma[step0_str] = {param_for_domain: step0_results[param_for_domain]['sigma']}
                        prev_haz_param = [param_for_domain]
                
                # for tracking tangent line and sigma_mu to use in PC
                index_mean_of_mu_up = {}
                intercept = {}
                slope = {}
                tangent_vector = {}
                sigma_mu_intercept = {}
                amu = {}
                bmu = {}
                liq_results = None
                
                # print(f'\t1. time: {time.time()-time_start} seconds')
                # time_start = time.time()
                
                # loop through steps
                for step in range(1,pbee_dim-1):
                    
                    # previous category in PBE
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

                    # run if not CPT-based
                    if 'CPTBased' in curr_methods: 
                        curr_results = cpt_based_curr_results.copy()
                        curr_mean_for_tangent = cpt_based_curr_mean_for_tangent.copy()
                    else:
                    # if running_cpt_based_procedure is False:
                        # special case for "lateral spread" and "settlement", which require "liquefaction" to be first assessed.
                        if curr_haz == 'lateral_spread' or curr_haz == 'settlement':
                            liq_haz_dict = methods_dict['edp']['liquefaction'].copy()
                            liq_methods = liq_haz_dict['method']
                            
                            # step1_param_names_liq = ['vs30','precip','dist_coast','gw_depth','dist_river','dist_water']
                            liq_input_param_names = methods_dict['edp']['liquefaction']['input_params']
                            liq_input_samples = {}
                            liq_input_dist = {}
                            for param in liq_input_param_names:
                                liq_input_samples[param] = input_samples[param]
                                liq_input_dist[param] = input_dist[param].copy()
                            
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
                                    liq_upstream_params[param] = np.ones((n_site,num_epi_input_samples))*rup_info[param]
                            
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
                                    n_site=n_site,
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
                                    n_site=n_site,
                                    get_liq_susc=True
                                )
                                
                            # rerun with with upstream param * factor for getting slope using forward Euler
                            liq_upstream_params_forward = liq_upstream_params.copy()
                            for param in liq_haz_dict['upstream_params']:
                                # intensities
                                if param == 'pga' or param == 'pgv':
                                    # liq_upstream_params_forward[param] *= forward_euler_multiplier
                                    liq_upstream_params_forward[param] = liq_upstream_params_forward[param] * forward_euler_multiplier

                            # preprocess methods with input samples
                            if 'liq_susc' in input_dist:
                                liq_results_forward, _, = pc_workflow.process_methods_for_mean_and_sigma_of_mu_for_liq(
                                    haz_dict=liq_haz_dict,
                                    upstream_params=liq_upstream_params_forward,
                                    internal_params=liq_internal_params,
                                    input_samples=liq_input_samples, 
                                    n_sample=num_epi_input_samples,
                                    n_site=n_site,
                                    get_liq_susc=False
                                )
                                if not 'liq_susc' in input_samples:
                                    liq_susc_forward = np.tile(input_dist['liq_susc']['value'],(num_epi_input_samples,1)).T
                            else:
                                liq_results_forward, liq_susc_forward, = pc_workflow.process_methods_for_mean_and_sigma_of_mu_for_liq(
                                    haz_dict=liq_haz_dict,
                                    upstream_params=liq_upstream_params_forward,
                                    internal_params=liq_internal_params,
                                    input_samples=liq_input_samples, 
                                    n_sample=num_epi_input_samples,
                                    n_site=n_site,
                                    get_liq_susc=True
                                )
                        
                        # get additional params for evaluation
                        curr_param_names_all = methods_dict[curr_cat][curr_haz]['input_params']
                        curr_param_internal = curr_param_names_all.copy()
                        curr_param_external = []
                        n_curr_params = len(curr_param_names_all)
                        curr_input_samples = {}
                        curr_input_dist = {}
                        for param in curr_param_names_all:
                            if param in input_samples:
                                if param == 'liq_susc' and param in input_samples:
                                    curr_input_samples[param] = input_samples[param]
                                    curr_input_dist[param] = input_dist[param].copy()
                                else:
                                    curr_input_samples[param] = input_samples[param]
                                    curr_input_dist[param] = input_dist[param].copy()
                                curr_param_external.append(param)
                                curr_param_internal.remove(param)
                        # pull upstream params for full analysis
                        curr_upstream_params = {}
                        for param in curr_haz_dict['upstream_params']:
                            # if in previous mean of mu
                            if param in mean_of_mu[prev_step_str]:
                                if np.ndim(mean_of_mu[prev_step_str][param]) == 0:
                                    curr_upstream_params[param] = \
                                        np.ones((n_site, num_epi_input_samples)) * \
                                        np.exp(mean_of_mu[prev_step_str][param].copy())
                                else:
                                    curr_upstream_params[param] = np.tile(
                                        np.exp(mean_of_mu[prev_step_str][param].copy())
                                        ,(num_epi_input_samples,1)
                                    ).T
                            # from rupture params
                            else:
                                curr_upstream_params[param] = np.ones((n_site,num_epi_input_samples))*rup_info[param]
                        
                        # pull internal params, e.g., prob_liq and liq_susc
                        curr_internal_params = {}
                        for param in curr_param_internal:
                            if param == 'liq_susc':
                                if not 'liq_susc' in input_samples:
                                    curr_internal_params[param] = liq_susc.copy()
                            else:
                                if liq_results is not None:
                                    curr_internal_params[param] = liq_results[param]['mean_of_mu'].copy()
                        
                        # preprocess methods with input samples
                        _, curr_results = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                            haz_dict=curr_haz_dict,
                            upstream_params=curr_upstream_params,
                            internal_params=curr_internal_params,
                            input_samples=curr_input_samples, 
                            n_sample=num_epi_input_samples,
                            n_site=n_site,
                        )
                        
                        # print(f'\t2a. time: {time.time()-time_start} seconds')
                        # time_start = time.time()
                        
                        # if step == 1:
                        #     print(curr_results)
                        #     print(curr_results['eps_tubing']['sigma_of_mu'])
                        
                        # rerun with with upstream param * factor for getting slope using forward Euler
                        # pull upstream params for full analysis
                        curr_upstream_params_forward = curr_upstream_params.copy()
                        for param in curr_haz_dict['upstream_params']:
                            # if in previous mean of mu
                            if param in mean_of_mu[prev_step_str]:
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
                        
                        # preprocess methods with input samples
                        _, curr_results_forward = pc_workflow.process_methods_for_mean_and_sigma_of_mu(
                            haz_dict=curr_haz_dict,
                            upstream_params=curr_upstream_params_forward,
                            internal_params=curr_internal_params_forward,
                            input_samples=curr_input_samples, 
                            n_sample=num_epi_input_samples,
                            n_site=n_site,
                        )
                        
                        # print(f'\t2c. time: {time.time()-time_start} seconds')
                        # time_start = time.time()

                    # get tangent params
                    curr_intercept = {}
                    curr_slope = {}
                    curr_tangent_vector = {}
                    curr_sigma_mu_intercept = {}
                    for i,param in enumerate(curr_results):
                        # get intercept
                        curr_intercept[param] = curr_results[param]['mean_of_mu'].copy()
                        # get slope
                        if len(mean_of_mu[prev_step_str]) == len(curr_haz_param):
                            prev_param_to_use = list(mean_of_mu[prev_step_str])[i]
                        else:
                            prev_param_to_use = list(mean_of_mu[prev_step_str])[0]
                        if prev_param_to_use in curr_upstream_params:
                            numer = \
                                curr_results_forward[param]['mean_of_mu'].copy() - \
                                curr_results[param]['mean_of_mu'].copy()
                            denom = \
                                np.log(curr_upstream_params_forward[prev_param_to_use][:,0].copy()) - \
                                np.log(curr_upstream_params[prev_param_to_use][:,0].copy())
                            curr_slope[param] = numer / denom
                        else:
                            curr_slope[param] = np.zeros(n_site)
                        
                    # read from results
                    curr_mean_of_mu = {}
                    curr_sigma_of_mu = {}
                    curr_sigma = {}
                    curr_amu = {}
                    curr_bmu = {}
                    for i,param in enumerate(curr_haz_param):
                        # mean of mu
                        curr_mean_of_mu[param] = curr_results[param]['mean_of_mu'].copy()
                        # sigma of mu
                        curr_sigma_of_mu[param] = curr_results[param]['sigma_of_mu'].copy()
                        # sigma
                        curr_sigma[param] = curr_results[param]['sigma'].copy()
                        # amu
                        curr_amu[param] = curr_slope[param]
                        # bmu
                        if len(mean_of_mu[prev_step_str]) == len(curr_haz_param):
                            prev_param_to_use = list(mean_of_mu[prev_step_str])[i]
                        else:
                            prev_param_to_use = list(mean_of_mu[prev_step_str])[0]
                        curr_bmu[param] = \
                            curr_slope[param]*(-mean_of_mu[prev_step_str][prev_param_to_use]) + curr_intercept[param]
                    
                        # if wells and caprocks, then set a = 0 and b = 1e-10 for no crossings
                        # if infra_type == 'wells_caprocks':
                        if infra_type == 'wells_caprocks' and 'well_strain' in workflow_i['haz_list']:
                            curr_amu[param][sites_with_no_crossing] = 0.0
                            curr_bmu[param][sites_with_no_crossing] = 1.e-10
                            
                    # store mean of mu, sigma of mu, sigma
                    mean_of_mu[curr_step_str] = curr_mean_of_mu.copy()
                    sigma_of_mu[curr_step_str] = curr_sigma_of_mu.copy()
                    sigma[curr_step_str] = curr_sigma.copy()
                    
                    if step >= 1:
                        # get amu and bmu for PC
                        amu[curr_step_str] = curr_amu.copy()
                        bmu[curr_step_str] = curr_bmu.copy()
                
                    if step == pbee_dim-2:
                        # save curr_haz_param as second_to_last_haz_param
                        second_to_last_haz_param = curr_haz_param
                
                # print(f'\t2. time: {time.time()-time_start} seconds')
                # time_start = time.time()
                
                ###########################################
                # set up for final step
                last_step = pbee_dim-1
                # last step string
                last_step_str = f'step_{last_step}'

                # current part in PBEE
                last_cat = workflow_i['cat_list'][last_step].lower()
                last_haz = workflow_i['haz_list'][last_step]
                last_haz_param = methods_dict[last_cat][last_haz]['return_params']

                # number of methods for hazard
                last_haz_dict = methods_dict[last_cat][last_haz].copy()
                last_methods = last_haz_dict['method']

                # get inputs for last step
                last_param_names_all = methods_dict[last_cat][last_haz]['input_params']
                last_param_internal = last_param_names_all.copy()
                last_param_external = []
                n_last_params = len(last_param_names_all)
                last_input_samples = {}
                last_input_dist = {}
                for param in last_param_names_all:
                    if param in input_samples:
                        last_input_samples[param] = input_samples[param]
                        last_input_dist[param] = input_dist[param].copy()
                        last_param_external.append(param)
                        last_param_internal.remove(param)
                        
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
                    n_site=n_site
                ) 
                
                # read from results
                last_mean_of_mu = {}
                last_sigma_of_mu = {}
                last_sigma = {}
                for i,param in enumerate(last_results):
                    # mean of mu
                    last_mean_of_mu[param] = last_results[param]['mean_of_mu'].copy()
                    # sigma of mu
                    last_sigma_of_mu[param] = last_results[param]['sigma_of_mu'].copy()
                    # sigma
                    last_sigma[param] = last_results[param]['sigma'].copy()
                # store mean of mu, sigma of mu, sigma
                mean_of_mu[last_step_str] = last_mean_of_mu.copy()
                sigma_of_mu[last_step_str] = last_sigma_of_mu.copy()
                sigma[last_step_str] = last_sigma.copy()
                
                # print(f'\t3. time: {time.time()-time_start} seconds')
                # time_start = time.time()
                
                ########################
                # set up for PC

                # pc letter for step
                pc_letter_for_step = {
                    0: 'Y',
                    1: 'Z',
                    2: 'T',
                    3: 'V'
                }

                # initial step string
                init_step_str = 'step_0'
                
                # for each return parameter
                for i, param_i in enumerate(last_haz_param):
                    # build kwargs for pc
                    pc_kwargs = {}
                    # loop through steps
                    for step in range(1, pbee_dim):
                        if step == 1:
                            params_for_step = list(mean_of_mu[init_step_str])
                            if len(params_for_step) == len(curr_haz_param) and len(curr_haz_param) > 1:
                                param_to_use = params_for_step[i]
                            else:
                                param_to_use = params_for_step[0]
                            pc_kwargs[f'mu{pc_letter_for_step[0]}'] = mean_of_mu[init_step_str][param_to_use]
                            pc_kwargs[f'sigmaMu{pc_letter_for_step[0]}'] = sigma_of_mu[init_step_str][param_to_use]
                            pc_kwargs[f'sigma{pc_letter_for_step[0]}'] = sigma[init_step_str][param_to_use]        
                        if step == pbee_dim-1:
                            pc_kwargs[f'mu{pc_letter_for_step[step]}'] = mean_of_mu[last_step_str][param_i]
                            pc_kwargs[f'sigmaMu{pc_letter_for_step[step]}'] = sigma_of_mu[last_step_str][param_i]
                            pc_kwargs[f'sigma{pc_letter_for_step[step]}'] = sigma[last_step_str][param_i]
                        else:
                            step_str = f'step_{step}'
                            params_for_step = list(mean_of_mu[step_str])
                            if infra_type == 'below_ground' and step == pbee_dim-2:
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
                            pc_kwargs[f'amu{pc_letter_for_step[step]}'] = amu[step_str][param_to_use]
                            pc_kwargs[f'bmu{pc_letter_for_step[step]}'] = bmu[step_str][param_to_use]
                            pc_kwargs[f'sigmaMu{pc_letter_for_step[step]}'] = sigma_of_mu[step_str][param_to_use]
                            pc_kwargs[f'sigma{pc_letter_for_step[step]}'] = sigma[step_str][param_to_use]

                    # get pc coefficients
                    # 1 integral
                    if pbee_dim == 2:
                        pc_coeffs_param_i = pc_coeffs_single_int(**pc_kwargs)
                    # 2 integral
                    elif pbee_dim == 3:
                        pc_coeffs_param_i = pc_coeffs_double_int(**pc_kwargs)
                    # 3 integrals
                    elif pbee_dim == 4:
                        pc_coeffs_param_i = pc_coeffs_triple_int(**pc_kwargs)
                
                    # aggregate pc coefficients
                    if not param_i in pc_coeffs:
                        pc_coeffs[param_i] = pc_coeffs_param_i * rup_info['rate'] # weighted by annual rate
                    else:
                        pc_coeffs[param_i] = pc_coeffs[param_i] + pc_coeffs_param_i * rup_info['rate'] # weighted by annual rate
                
                
                # print(f'\t4. time: {time.time()-time_start} seconds')
                # time_start = time.time()
                # print(f'total time: {time.time()-time_initial} seconds')
                # print('\n')
            
            # initialize
            pc_samples = {}
            # loop through params
            for param_i in last_haz_param:
                # sum up pc terms
                pc_samples[param_i] = np.inner(hermite_prob_table_indep.T,pc_coeffs[param_i])
                # keep sum within 0 and 1
                pc_samples[param_i] = np.maximum(np.minimum(pc_samples[param_i],1),0)
            
            # get fractiles
            df_frac[f'case_{case_to_run}'] = pd.DataFrame(None)
            for i,param_i in enumerate(last_haz_param):
                if len(curr_haz_param) > 1:
                    param_to_use = second_to_last_haz_param[i]
                else:
                    param_to_use = second_to_last_haz_param[0]
                return_frac = pc_workflow.get_fractiles(
                    pc_samples[param_i],
                    infra_type=infra_type,
                    site_id=site_data.ID.values
                )
                # add param to column name
                return_frac.columns = [f'{col}_{param_to_use}' for col in return_frac.columns]
                df_frac[f'case_{case_to_run}'] = pd.concat([df_frac[f'case_{case_to_run}'],return_frac],axis=1)
            # get site locations to add to fractile dataframe
            if 'LON_MID' in site_data:
                df_locs = site_data[[
                            'LON_BEGIN','LAT_BEGIN',
                            'LON_END','LAT_END',
                            'LON_MID','LAT_MID',
                        ]].copy()
            else:
                df_locs = site_data[['LON','LAT']].copy()
            df_locs.index = df_frac[f'case_{case_to_run}'].index
            # concat
            df_frac[f'case_{case_to_run}'] = pd.concat([df_locs,df_frac[f'case_{case_to_run}']],axis=1)
            mean_of_mu_track.append(mean_of_mu)
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
                df_curr_pc_case = df_frac[pc_case][non_frac_headers].copy()
                
                # convert headers to np.array
                cases = np.asarray(cases)
                
                # initialize output dataframe
                df_frac[pc_case+'_worst_case'] = df_frac[pc_case][non_frac_headers].copy()
                
                # loop through subsystem cases
                for comp in ['elbow','tee']:
                    # create output string
                    out_str_comp = [
                        f'{frac}_{comp}' for frac in frac_strs
                    ]
                    
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
            df_workflow['IM'].iloc[i] = track_im_dependency_for_output[i]
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
    # Export summary file
    sdir = os.path.join(work_dir,'Results')
    if not os.path.isdir(sdir):
        os.mkdir(sdir)
    spath = os.path.join(sdir,f'results.xlsx')
    # formats
    rows_pad_btw_table = 2
    float_format = r'%.3e'
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
        # case tab
        for case in workflow_order_list:
            df_frac[case].to_excel(writer, sheet_name=case, float_format=float_format)
        for case in workflow_order_list:
            if case+'_combined' in df_frac:
                df_frac[case+'_combined'].to_excel(writer, sheet_name=case+'_combined', float_format=float_format)
            if case+'_worst_case' in df_frac:
                df_frac[case+'_worst_case'].to_excel(writer, sheet_name=case+'_worst_case', float_format=float_format)

        
    # add a page for notes
    # from openpyxl import openpyxl
    # wb = load_workbook('template.xlsx')
    # workbook = Workbook()
    # worksheet = workbook.worksheets[0]
    # worksheet.title = "Sheet1"

    # worksheet.cell('A1').style.alignment.wrap_text = True
    # worksheet.cell('A1').value = "Line 1\nLine 2\nLine 3"

    # workbook.save('wrap_text1.xlsx')
                
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
    
    # Parse command line input
    args = parser.parse_args()
    
    # Run "Main"
    main(
        work_dir = args.work_dir,
        # clean_prev_run = args.clean,
        # logging_level = args.logging,
    )