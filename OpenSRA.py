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
# Python modules
import importlib
import os
import time
import warnings
import logging
import sys
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from scipy import sparse

# OpenSRA modules and functions
from src import Model, PreProcess, Fcn_Common
# from src.fcn_common import set_logging, make_dir
# from src.im import fcn_im
# from opensra_input import *
# import input
# from src.site.geologic_unit import get_geologic_unit

# SimCenter modules
# import lib.simcenter.OpenSHAInterface as OpenSHAInterface


# -----------------------------------------------------------
# Main function
def main(input_dir, clean_prev_run, logging_level):

    # -----------------------------------------------------------
    # Setting logging level (e.g. DEBUG or INFO)
    Fcn_Common.set_logging(logging_level)
    logging.info('---------------')

    # -----------------------------------------------------------
    # Check if setup_file is provided, and import if provided
    setup_file = os.path.join(input_dir,'SetupConfig.json')
    if setup_file is None:
        logging.info(f"Setup file is not provided; OpenSRA will now exit.")
        sys.exit()
        
    else:
        logging.info(f"Setup file to use:")
        logging.info(f"\t{setup_file}")
        
        # import setup_file
        with open(setup_file) as json_file:
            setup_config = json.load(json_file)
    
    # -----------------------------------------------------------
    # Initialize dictionary for storing created setup params
    other_config_param = {}
    other_config_param['Dir_Input'] = input_dir # add input_dir to config_param
    other_config_param['File_SetupConfig'] = setup_file # add setup_file to config_param
    
    # Initialize list of folders to create
    folders_to_create = []
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # List of inputs that are fixed currently
    other_config_param['Shp_GeologicUnit'] = r'lib/other/GDM_002_GMC_750k_v2_GIS/shapefiles/GMC_geo_poly.shp' # shapefile with geologic units, set path relative to OpenSRA base directory
    other_config_param['File_GeologicUnitParam'] = r'lib/slate/Seismic_Hazard_CGS_Unit Strengths.csv' # Micaela's file with properties for geologic units, set path relative to OpenSRA base directory
    # other_config_param['im_model'] = 'mean_ngawest2' # method for ground motion prediction; Mean NGAWest2 (without I-14)
    other_config_param['Flag_SampleWithStDevTotal'] = False # True to sample from total sigma; False to sample intra- and inter-event sigmas separately then add together
    other_config_param['UniformSigmaAleatory'] = None # set value for uniform aleatory sigma; set to None to use GM predicted total sigma
    other_config_param['Num_Decimals'] = 3 # number of decimals in log10 space for export
    other_config_param['Flag_ResampleIM'] = False # sample IM even if samples already exist
    other_config_param['RupturePerGroup'] = 100 # number of ruptures to store per group
    other_config_param['Num_Threads'] = 1 # number of threads for regional_processor
    other_config_param['ColumnsToUseForKy'] = {
        'friction_angle': 'Friction Angle (deg)',
        'cohesion': 'Cohesion (kPa)',
        'thickness': 'Thickness (m)',
        'unit_weight': 'Unit Weight (kN/m3)',
        }
    other_config_param['ListOfIMParams'] = ['mean','stdev_inter','stdev_intra','stdev_total']
    other_config_param['ApproxPeriod'] = {
        'PGA': 0.01,
        'PGV': 1
    }
        
    # Other parameters required for now but will likely be removed
    other_config_param['Flag_EDPSampleExist'] = False
    other_config_param['Flag_DMSampleExist'] = False
    other_config_param['Flag_SaveDV'] = False
    inc1 = 5
    inc2 = 100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # -----------------------------------------------------------
    # Pre-process
    logging.info('---------------')
    logging.info('\t******** Preprocess ********')
    # Get directory of OpenSRA.py, in case if user is working in different directory and calling OpenSRA.py
    other_config_param['Dir_OpenSRA'] = os.path.dirname(os.path.realpath(__file__))
    # dir_opensra = setup_config['General']['Directory']['OpenSRA']
    os.chdir(other_config_param['Dir_OpenSRA'])
    
    # -----------------------------------------------------------
    # Get working directory and see if user wants previous run cleaned
    other_config_param['Dir_Working'] = setup_config['General']['Directory']['Working']
    if clean_prev_run.lower() in 'yes':
        list_of_items = os.listdir(other_config_param['Dir_Working'])
        for item in list_of_items:
            if not item == 'Input':
                if os.path.isdir(os.path.join(other_config_param['Dir_Working'],item)):
                    shutil.rmtree(os.path.join(other_config_param['Dir_Working'],item))
                elif os.path.isfile(os.path.join(other_config_param['Dir_Working'],item)):
                    os.remove(os.path.join(other_config_param['Dir_Working'],item))
        logging.info('Cleaned previous run')

    
    # Create other variables from and append to setup_config
    other_config_param, folders_to_create = PreProcess.get_other_setup_config(
        setup_config, other_config_param, folders_to_create
    )
    
    # Create additional folders in working directory
    for item in folders_to_create:
        Fcn_Common.make_dir(item)    
    
    # load rupture groups
    # if os.path.exists(other_config_param['File_RuptureGroups']):
        # list_rup_group = np.loadtxt(other_config_param['File_RuptureGroups'], dtype=str, ndmin=1)
    
    # Import and update site data file
    if os.path.exists(other_config_param['Path_SiteData_Updated']):
        site_data = pd.read_csv(other_config_param['Path_SiteData_Updated']) # read site data file
    else:
        site_data = pd.read_csv(other_config_param['Path_SiteData']) # read site data file
    other_config_param['Num_Sites'] = len(site_data)
    
    # Update site_data 
    site_data = PreProcess.update_site_data_file(site_data, setup_config, other_config_param)
    
    # Append site_data to method parameters
    setup_config = PreProcess.add_site_data_to_method_param(setup_config, site_data, other_config_param)
        
    # Export site locations and vs30 to text files for OpenSHA interface
    PreProcess.export_site_loc_and_vs30(site_data, other_config_param)
    
    # Export setup_file for checks
    # with open(setup_file.replace('.json','_updated.json'), 'w') as outfile:
        # json.dump(setup_config, outfile, indent=4, separators=(',', ': '))
    
    # -----------------------------------------------------------
    # Create assessment class object
    model = Model.assessment()
    logging.info(f'Created "Model.assessment" class object named "model"')
    
    # -----------------------------------------------------------
    # Start workflow
    # 1. Get rupture scenarios, and fault-crossing and geologic units and properties at sites
    # 2. Get ground motion predictions at target sites given rupture scenarios
    # 3. Generate and store intensity measure realizations
    # 4. Assess EDPs, DMs, DVs using IM realizations
    
    # -----------------------------------------------------------
    # Phases 1&2:
    # Get rupture scenarios and fault crossings given site
    logging.info('---------------')
    logging.info('\t******** Get IM means and StdDevs ********')
    model.get_IM_means(setup_config, other_config_param, site_data)
    other_config_param['Num_Events'] = len(model._EVENT_dict['Scenarios']['mag'])
    
    # -----------------------------------------------------------
    # Phase 3:
    # Generate realizations of IMs (i.e., sampling); assess EDPs, DMs, DVs
    logging.info('---------------')
    logging.info('\t******** Simulate IM samples ********')
    model.sim_IM(setup_config, other_config_param, site_data)
    
    # sys.exit()
    
    # -----------------------------------------------------------
    # Phase 3:
    # Generate realizations of IMs (i.e., sampling); assess EDPs, DMs, DVs
    # -----------------------------------------------------------
    # generate/import samples
    # if flag_get_IM or flag_gen_sample:
    # directory name for current rupture group
    #sample_dir = os.path.join(im_dir,rup_group)
    #fcn_common.make_dir(sample_dir) # create folder if it doesn't exist
    ## Check if samples exist for current rupture group, default to True (samples exist and will not generate again)
    #flag_im_sample_exist = True # default
    #list_dir_im = os.listdir(sample_dir) # get list of files under "sample_dir"
    ## First check if the number of files in "sample_dir" is consistent with the expected number of files (# of IMs * # of samples per IM)
    #if len(list_dir_im) < len(imt)*input.n_samp_im:
    #    flag_im_sample_exist = False # if file count is inconsistent, then set flag for IM existing to False such that IM samples will be saved
    ## Second check to see if sample 0 (first sample) exist for each IM; if not, then set flag to False
    #for im in imt:
    #    im_samp_file_name = im+'_samp_0'+'.'+input.output_file_type
    #    if not im_samp_file_name in list_dir_im:
    #        flag_im_sample_exist = False
    #        break
    ## flag to force resample of IM
    #if input.Flag_ResampleIM:
    #    flag_im_sample_exist = False
    ##
    #model.sim_IM(n_samp_im=input.n_samp_im, ims=imt,
    #        flag_spatial_corr=input.flag_spatial_corr, flag_cross_corr=input.flag_cross_corr, 
    #        Flag_SampleWithSigmaTotal=input.Flag_SampleWithSigmaTotal,
    #        sigma_aleatory=input.sigma_aleatory,
    #        flag_clear_dict=True, flag_im_sample_exist=flag_im_sample_exist,
    #        sample_dir=sample_dir, output_file_type=input.output_file_type)
    #        
    # If IM samples exist in the (i.e., phase_to_run == 4)
    #if flag_im_sample_exist:
    #    logging.debug(f'\tIM_sim: Updated "_IM_dict" using path to samples:')
    #else:
    #    # for im in imt:
    #        # for samp_i in range(input.n_samp_im):
    #            # save_name = os.path.join(sample_dir,im+'_samp_'+str(samp_i)+'.'+input.output_file_type)
    #            # samp = model._IM_dict[im][samp_i] # pull sparse matrix for sample from class
    #            # file_io.store_im_samp(save_name, samp, input.output_file_type, input.n_demicals)
    #            # sparse_mat = model._IM_dict[im][samp_i].log1p().tocsc() # pull sparse matrix for sample from class
    #            # sparse_mat = model._IM_dict[im][samp_i] # pull sparse matrix for sample from class
    #            # sparse_mat.data = np.round(sparse_mat.data,decimals=input.Num_Decimals)
    #            # if input.output_file_type == 'npz':
    #                
    #            # sparse.save_npz(os.path.join(sample_dir,im+'_samp_'+str(samp_i)+'.npz'),sparse_mat)
    #    logging.debug(f'\tIM_sim: Updated "_IM_dict" by generating samples and storing to:')
    #    logging.debug(f'\t\t{sample_dir}')
    
    
    # -----------------------------------------------------------
    # Phase 2:
    # Get predictions of IMs from GMMs
    #model.get_IM_pred()
    #
    ## initialize processor
    #logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
    #                                        path_vs30=vs30_file, numThreads=input.Num_Threads,
    #                                        rmax_cutoff=input.r_max)
    #logging.info(f"\n")
    #
    #logging.info(f"\tLooping through all {len(list_rup_group)} groups of ruptures and getting GM predictions")
    ## loop through every rup group and get predictions
    #for rup_group in list_rup_group:
    ## rup_group = list_rup_group[0]
    #    # folder to store predictions for current group
    #    gm_pred_rup_group_dir = os.path.join(gm_pred_dir,rup_group)
    #    fcn_common.make_dir(gm_pred_rup_group_dir) # create folder if it doesn't exist
    #    # see if files already exist
    #    # if len(os.listdir(gm_pred_rup_group_dir)) != 6:
    #    # current range of ruptures for current group
    #    rup_start = int(rup_group[:rup_group.find('_')])
    #    rup_end = int(rup_group[rup_group.find('_')+1:])+1
    #    #
    #    # logging.info(f"\tGetting GM predictions for rup_group {rup_group}")
    #    OpenSHAInterface.runHazardAnalysis(processor=reg_proc,rup_meta_file=rup_meta_file_tr_rmax,
    #                                        ind_range=[rup_start,rup_end],saveDir=gm_pred_rup_group_dir,
    #                                        output_file_type=input.output_file_type)
    #    
    #logging.info(f"\tGenerated predictions for all {len(list_rup_group)} groups of ruptures; results stored under:")
    #logging.info(f"\t\t{gm_pred_dir}")
    
    
    
    
    #
    #if 'regionalprocessor' in other_config_param['im_tool']:
    #
    #    #
    #    if not os.path.exists(rup_meta_file_tr) or not os.path.exists(rup_meta_file_tr_rmax) or len(os.listdir(trace_dir)) == 0 or len(os.listdir(intersect_dir)) == 0:
    #    
    #        # initialize processor
    #        logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #        reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
    #                                                path_vs30=vs30_file, numThreads=input.Num_Threads,
    #                                                rmax_cutoff=input.r_max)
    #        logging.info(f"\n")
    #    
    #        # check if rupture metafile screened by return period is already created
    #        if not os.path.exists(rup_meta_file_tr):
    #            logging.info(f"\t...generating list of rupture scenarios given cutoff on return period (this may take some time)")
    #            # run to get a list of ruptures given criteria on tr and rmax
    #            OpenSHAInterface.load_src_rup_M_rate(rup_meta_file=rup_meta_file_tr, ind_range=['all'],
    #                                                proc=reg_proc, rate_cutoff=1/input.tr_max)
    #            logging.info(f"\tList of rupture scenarios filtered by return period exported to:")
    #            logging.info(f"\t\t{rup_meta_file_tr}")
    #            
    #        # check if rupture metafile screened by return period and maximum distance is already created
    #        if not os.path.exists(rup_meta_file_tr_rmax):
    #            OpenSHAInterface.filter_src_by_rmax(site_data.loc[:,['Longitude','Latitude']].values,
    #                                                input.r_max, rup_meta_file_tr, rup_meta_file_tr_rmax,
    #                                                rup_seg_file, pt_src_file,File_RuptureGroups, input.RupturePerGroup,
    #                                                input.flag_include_point_source)
    #            logging.info(f"\t...further filter rupture scenarios by rmax and exported to:")
    #            logging.info(f"\t\t{rup_meta_file_tr_rmax}")
    #            logging.info(f"\tList of rupture groups exported to:")
    #            logging.info(f"\t\t{File_RuptureGroups}")
    #            
    #        # check if trace and intersect directories are empty
    #        if input.flag_get_fault_xing is True & (len(os.listdir(trace_dir)) == 0 or len(os.listdir(intersect_dir)) == 0):
    #            logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #            OpenSHAInterface.get_fault_xing(reg_proc, site_data.loc[:,['Lon_start','Lat_start']].values,
    #                                            site_data.loc[:,['Lon_end','Lat_end']].values,
    #                                            trace_dir, intersect_dir, rup_meta_file_tr_rmax,
    #                                            src_model=input.src_model)
    #            logging.info(f"\n")
    #            logging.info(f"\tRupture (segment) traces exported to:")
    #            logging.info(f"\t\t{trace_dir}")
    #            logging.info(f"\tFault crossings exported to:")
    #            logging.info(f"\t\t{intersect_dir}")
    
    
    #
    #if 'regionalprocessor' in input.im_tool.lower():
    #    
    #    # initialize processor
    #    logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #    reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
    #                                            path_vs30=vs30_file, numThreads=input.Num_Threads,
    #                                            rmax_cutoff=input.r_max)
    #    logging.info(f"\n")
    #    
    #    logging.info(f"\tLooping through all {len(list_rup_group)} groups of ruptures and getting GM predictions")
    #    # loop through every rup group and get predictions
    #    for rup_group in list_rup_group:
    #    # rup_group = list_rup_group[0]
    #        # folder to store predictions for current group
    #        gm_pred_rup_group_dir = os.path.join(gm_pred_dir,rup_group)
    #        fcn_common.make_dir(gm_pred_rup_group_dir) # create folder if it doesn't exist
    #        # see if files already exist
    #        # if len(os.listdir(gm_pred_rup_group_dir)) != 6:
    #        # current range of ruptures for current group
    #        rup_start = int(rup_group[:rup_group.find('_')])
    #        rup_end = int(rup_group[rup_group.find('_')+1:])+1
    #        #
    #        # logging.info(f"\tGetting GM predictions for rup_group {rup_group}")
    #        OpenSHAInterface.runHazardAnalysis(processor=reg_proc,rup_meta_file=rup_meta_file_tr_rmax,
    #                                            ind_range=[rup_start,rup_end],saveDir=gm_pred_rup_group_dir,
    #                                            output_file_type=input.output_file_type)
    #        
    #    logging.info(f"\tGenerated predictions for all {len(list_rup_group)} groups of ruptures; results stored under:")
    #    logging.info(f"\t\t{gm_pred_dir}")
    
    # General
    ##Dir_OpenSRA = os.path.dirname(os.path.realpath(__file__)) # get directory of OpenSRA.py, in case if user is working in different directory and calling main
    ##os.chdir(Dir_OpenSRA) # change current directory to the OpenSRA directory
    ##site_dir = os.path.join(input.work_dir, 'site_out') # set full path for site output folder
    ##fcn_common.make_dir(site_dir) # create folder if it doesn't exist
    ##Path_SiteData  = os.path.join(input.work_dir,input.site_data_file_name) # create full path for site data file
    ##site_data = pd.read_csv(Path_SiteData) # read site data file
    ##flag_update_site_data = False # default
    ##
    ### Check if segment lengths are given in the file; if not then assume lengths of 1 km and add to "site_data"
    ##if not 'l_seg (km)' in site_data.columns:
    ##    site_data['l_seg (km)'] = np.ones(len(site_data))
    ##    flag_update_site_data = True # updated site data
    ##    
    ### Check if start and end locations are given in the file; if not set equal to site locations (i.e., points)
    ##if not 'Lon_start' in site_data.columns:
    ##    site_data['Lon_start'] = site_data['Longitude'].values
    ##    site_data['Lat_start'] = site_data['Latitude'].values
    ##    site_data['Lon_end'] = site_data['Longitude'].values
    ##    site_data['Lat_end'] = site_data['Latitude'].values
    ##    flag_update_site_data = True # updated site data
        
    # Pad OpenSRA path to the user-defined relative paths
    # input.geo_shp_file = os.path.join(os.getcwd(),input.geo_shp_file)
    # input.geo_unit_param_file = os.path.join(os.getcwd(),input.geo_unit_param_file)
    
    # Check if files for site locations and vs30 exist in "site_dir", these files are exported from "site_data" for easier access by "im_tool"
    #flag_export_site_loc = False # default
    #site_loc_file = os.path.join(site_dir,'site_loc.txt')
    #if not os.path.exists(site_loc_file):
    #    np.savetxt(site_loc_file,site_data.loc[:,['Longitude','Latitude']].values,fmt='%10.6f,%10.6f') # export vs30 to file for im_tool
    #    flag_export_site_loc = True
    #flag_export_vs30 = False # default
    #vs30_file = os.path.join(site_dir,'vs30.txt')
    #if not os.path.exists(vs30_file):
    #    np.savetxt(vs30_file,site_data['VS30 (m/s)'].values,fmt='%6.2f') # export vs30 to file for im_tool
    #    flag_export_vs30 = True

    # Source and ground motion predictions
    # gm_dir = os.path.join(input.work_dir, 'gm_out') # set full path for GM output folder
    # fcn_common.make_dir(gm_dir) # create folder if it doesn't exist
    # gm_pred_dir = os.path.join(gm_dir, 'gm_pred') # set full path for GM prediction output folder
    # fcn_common.make_dir(gm_pred_dir) # create folder if it doesn't exist
    # rup_seg_file = os.path.join(os.getcwd(),'lib','simcenter','erf',input.src_model,'rup_seg.json')
    # pt_src_file = os.path.join(os.getcwd(),'lib','simcenter','erf',input.src_model,'point_source.txt')
    # rup_meta_file_tr = os.path.join(gm_dir,'rup_meta_'+str(input.tr_max)+'yr.txt') # create full path for rupture metafile, which contains magnitudes and rates for rupture scenarios screened by return period
    # rup_meta_file_tr_rmax = os.path.join(gm_dir,'rup_meta_'+str(input.tr_max)+'yr_'+str(input.r_max)+'km.txt') # create full path for rupture metafile, which contains magnitudes and rates for rupture scenarios screened by return period
    # File_RuptureGroups = os.path.join(gm_dir,'list_rup_group_'+str(input.tr_max)+'yr_'+str(input.r_max)+'km.txt') # create full path for list of groups of ruptures
    # check if need to extract geologic units and properties, get the params if not
    # if not 'Geologic Unit' in site_data.columns:
        # run script to get the properties
        # site_data = get_geo_unit(site_data, input.geo_shp_file, input.geo_unit_param_file,
                                # input.ColumnsToUseForKy, input.flag_use_default_param_for_ky)
        # flag_update_site_data = True # updated site data
    # check if site data is updated, if so then save to site file
    # if flag_update_site_data:
        # site_data.to_csv(Path_SiteData,index=False)
        
    # load rupture groups
    # if os.path.exists(File_RuptureGroups):
        # list_rup_group = np.loadtxt(File_RuptureGroups,dtype=str,ndmin=1)
    
    # check for directories for storing fault traces and intersections; create if don't exist
    # trace_dir = os.path.join(gm_dir,'src_trace')
    # fcn_common.make_dir(trace_dir) # create folder if it doesn't exist
    # intersect_dir = os.path.join(gm_dir,'src_intersect')
    # fcn_common.make_dir(intersect_dir) # create folder if it doesn't exist
    
    # Intensity measures
    # im_dir = os.path.join(input.work_dir, 'im_out') # set full path for IM output folder
    # fcn_common.make_dir(im_dir) # create folder if it doesn't exist
    # flag_im_sample_exist = True # default
        # flag_im_sample_exist = False # True if IM samples have been generated, else False
    # else:
        # if len(os.listdir(im_dir)) == 0: # see if IM folder is empty (do samples exist)
            # flag_im_sample_exist = False # True if IM samples have been generated, else False
    # check if IM samples are needed
    # imt = [] # target intensity measure type (IMT) to get
    # if input.flag_include_pga:
        # imt.append('pga')
    # if input.flag_use_pgv:
        # imt.append('pgv')
    # flag_get_IM = False # default
    # if 'rr_pgv' in input.dvs:
        # flag_get_IM = True
    # if 'rr_pgd' in input.dvs:
        # if 'hazus' in input.dv_procs:
            # flag_get_IM = True
        # else:
            # if 'ls' in input.edps or 'land' in input.edps or 'surf' in input.edps:
                # flag_get_IM = True
    # update required 'edps' and 'ims' if only 'rr_pgv' is requested
    # if 'rr_pgv' in input.dvs and len(input.dvs) == 1:
        # input.edps = []
        # imt = ['pgv']
    # if flag_get_IM is False:
        # imt = []
                    

    # Engineering demand parameters
    # note: OpenSRA currently is not set up to store dm results
    # edp_dir = os.path.join(input.work_dir, 'edp_out') # set full path for EDP output folder
    # fcn_common.make_dir(edp_dir) # create folder if it doesn't exist
        
    # if EDP samples exist
    # Flag_EDPSampleExist = False
    
    # check if probability of liquefaction is needed
    # flag_p_liq = False # default
    # flag_liq_susc = False # default
    # if 'liq' in input.edps:
        # flag_liq_susc = True
    # if 'ls' in input.edps:
        # flag_p_liq = True
    
    #
    #if input.phase_to_run == 4:
    #    # for liquefaction-induced demands
    #    if flag_liq_susc or flag_p_liq:
    #    
    #        # import general inputs
    #        wtd = site_data['WTD_300m (m)'].values
    #        dr = site_data['Dist_River (km)'].values
    #        dc = site_data['Dist_Coast (km)'].values
    #        dw = site_data['Dist_Any Water (km)'].values
    #        precip = site_data['CA_Precip (mm)'].values
    #        vs30 = site_data['VS30 (m/s)'].values
    #        
    #        # change all -9999 entries to NaN
    #        find_val = -9999
    #        set_val = np.nan
    #        wtd = fcn_common.set_elements_to_nan(wtd,find_val,set_val)
    #        dr = fcn_common.set_elements_to_nan(dr,find_val,set_val)
    #        dc = fcn_common.set_elements_to_nan(dc,find_val,set_val)
    #        dw = fcn_common.set_elements_to_nan(dw,find_val,set_val)
    #        precip = fcn_common.set_elements_to_nan(precip,find_val,set_val)
    #        vs30 = fcn_common.set_elements_to_nan(vs30,find_val,set_val)
    #        
    #        # elevation from DEM maps
    #        z = np.ones(vs30.shape)*10 # set to 10 for now, get data from DEM map later
    #
    #        # add params to input
    #        input.edp_other_params.update({'wtd':wtd, 'dr':dr, 'dc':dc, 'dw':dw,
    #                                    'precip':precip, 'vs30':vs30, 'z':z})
    #
    #        # for ground settlement
    #        if 'ls' in input.edps:
    #            dwWB = site_data['Dist_Any Water (km)'].values
    #            dwWB = fcn_common.set_elements_to_nan(dwWB,find_val,set_val)
    #            dwWB = dwWB*1000 # convert to meters
    #            input.edp_other_params.update({'dwWB':dwWB})
    #
    #    # check if probability of landslide is needed
    #    flag_p_land = False
    #    if 'rr_pgd' in input.dvs and 'land' in input.edps:
    #        flag_p_land = True
    #
    #    # for landslide-induced demands
    #    if 'land' in input.edps:
    #        input.edp_other_params.update({'ky':site_data['ky_inf_bray'].values})
    #        
    #    # for surface fault rupture
    #    if 'surf' in input.edps:
    #        input.edp_other_params.update({'l_seg':site_data['l_seg (km)'].values})
    #        input.dv_other_params.update({'l_seg':site_data['l_seg (km)'].values})
    #        # intersect_dir = os.path.join(input.work_dir,'gm_resource','rup_intersect')
    #        # logging.info(f"Defined directory with fault crossings: {intersect_dir}.")

    # Damage measures
    # note: OpenSRA currently is not set up to store dm results
    # dm_dir = os.path.join(input.work_dir, 'dm_out') # set full path for DM output folder
    # fcn_common.make_dir(dm_dir) # create folder if it doesn't exist
    # if DM samples exist
    # Flag_DMSampleExist = False

    # Decision variables
    # dv_dir = os.path.join(input.work_dir, 'dv_out') # set full path for DV output folder
    # fcn_common.make_dir(dv_dir) # create folder if it doesn't exist
    # increment to print output messages
    # Flag_SaveDV = False # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # if len(input.dvs) == 0:
        # Flag_SaveDV = False
    # for printing message per group
    # inc1 = 5 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # inc2 = 100 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    # Other setups
    

    # -----------------------------------------------------------
    ## Print messages for...
    ## General
    #logging.info(f"Analysis Details")
    #logging.info("-----------------------------------------------------------")
    #logging.info(f"Setup for phase {input.phase_to_run}: {input.phase_message}")
    #
    ## Site data
    #logging.info(f"Site locations and data")
    #logging.info(f"\tLoaded site locations and data from:")
    #logging.info(f"\t\t{Path_SiteData}")
    #if flag_update_site_data:
    #    logging.info(f"\tUpdated and saved site data to the original file")
    #if flag_export_site_loc:
    #    logging.info(f"\tExported site locations for {gm_dir}")
    #if flag_export_vs30:
    #    logging.info(f"\tExported vs30 for {gm_dir}")
    #
    ## Source and ground motion predictions
    #logging.info(f"EQ sources and GMs")
    #logging.info(f"\tDirectory for source and GM outputs:")
    #logging.info(f"\t\t{gm_dir}")
    #logging.info(f"\tSource and ground motion tool: {input.im_tool}")
    #logging.info(f"\tSource model: {input.src_model}")
    #logging.info(f"\tCutoff return period: {input.tr_max} years")
    #logging.info(f"\tCutoff distance: {input.r_max} km")
    #
    ## for phases 2+
    #if input.phase_to_run >= 2:
    #    logging.info(f"\tGM model: {input.im_model}")
    #    # logging.info(f"Number of groups of ruptures = {len(list_rup_group)} (each group contains {RupturePerGroup} ruptures)")
    #    
    #    if flag_get_IM or flag_gen_sample:
    #        logging.info(f"\tWill print messages every {inc1} groups (each group contains {input.RupturePerGroup} ruptures)")
    #    if Flag_SaveDV:
    #        logging.info(f"\tWill save damage outputs every {inc2} groups (each group contains {input.RupturePerGroup} ruptures)")
    #
    ## for phases 3+
    #if input.phase_to_run >= 3:
    #    logging.info(f"IMs")
    #    if input.phase_to_run == 3:
    #        logging.info(f"\tRequested IMs include: {imt}")
    #        logging.info(f"\tPerform spatial correlation = {input.flag_spatial_corr}")
    #        logging.info(f"\tPerform cross-correlation = {input.flag_cross_corr}")
    #    elif input.phase_to_run == 4:
    #        logging.info(f"\tRequired IMs include: {imt} (list may be altered based on requested DVs)")
    #    logging.info(f"\tNumber of IM samples = {input.n_samp_im}")
    #
    ## for phase 4
    #if input.phase_to_run == 4:
    #    #
    #    logging.info(f"EDPs")
    #    logging.info(f"\tRequired EDPs include: {input.edps} (list may be altered based on requested DVs)")
    #    # logging.info(f"\tNumber of EDP samples = {input.n_samp_edp} (for uniform distributions use only)")
    #    logging.info(f"\tFlag for probability of liquefaction = {flag_p_liq}")
    #    logging.info(f"\tFlag for liquefaction susceptibility = {flag_liq_susc}")
    #    logging.info(f"\tFlag for probability of landslide = {flag_p_land}")
    #    # if flag_p_liq or flag_liq_susc:
    #        # logging.info(f"\tLoaded site parameters for liquefaction into 'edp_other_params'")
    #    # if flag_calc_ky:
    #        # logging.info(f"Calculated ky using Bray (2007) for infinite slope for landslide-induced demands.")
    #    # else:
    #        # logging.info(f"Loaded ky into 'edp_other_params'")
    #            
    #    #
    #    logging.info(f"DMs")
    #    logging.info(f"\tRequired DMs include: {input.dms} (list may be altered based on requested DVs)")
    #    logging.info(f"\tNumber of DM samples = {input.n_samp_dm} (for uniform distributions use only)")
    #    #
    #    logging.info(f"DVs")
    #    logging.info(f"\tRequested DVs include: {input.dvs}")
    #    logging.info(f"\tDirectory for DV outputs:")
    #    logging.info(f"\t\t{dv_dir}")
    #
    #
    ##
    ## Print messages for...
    ## reload model.py if it has been modified
    ## importlib.reload(model)
    ## logging.info(f'Load/reloaded "model.py".')
    #
    ##
    #logging.info("-----------------------------------------------------------")
    #logging.info(f"Performing phase {input.phase_to_run} analysis")
    

    # -----------------------------------------------------------
    # Get rupture scenarios and fault crossings given site
    #if input.phase_to_run == 1:
    #    
    #    #
    #    if 'regionalprocessor' in input.im_tool.lower():
    #    
    #        #
    #        if not os.path.exists(rup_meta_file_tr) or not os.path.exists(rup_meta_file_tr_rmax) or len(os.listdir(trace_dir)) == 0 or len(os.listdir(intersect_dir)) == 0:
    #        
    #            # initialize processor
    #            logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #            reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
    #                                                    path_vs30=vs30_file, numThreads=input.Num_Threads,
    #                                                    rmax_cutoff=input.r_max)
    #            logging.info(f"\n")
    #        
    #            # check if rupture metafile screened by return period is already created
    #            if not os.path.exists(rup_meta_file_tr):
    #                logging.info(f"\t...generating list of rupture scenarios given cutoff on return period (this may take some time)")
    #                # run to get a list of ruptures given criteria on tr and rmax
    #                OpenSHAInterface.load_src_rup_M_rate(rup_meta_file=rup_meta_file_tr, ind_range=['all'],
    #                                                    proc=reg_proc, rate_cutoff=1/input.tr_max)
    #                logging.info(f"\tList of rupture scenarios filtered by return period exported to:")
    #                logging.info(f"\t\t{rup_meta_file_tr}")
    #                
    #            # check if rupture metafile screened by return period and maximum distance is already created
    #            if not os.path.exists(rup_meta_file_tr_rmax):
    #                OpenSHAInterface.filter_src_by_rmax(site_data.loc[:,['Longitude','Latitude']].values,
    #                                                    input.r_max, rup_meta_file_tr, rup_meta_file_tr_rmax,
    #                                                    rup_seg_file, pt_src_file,File_RuptureGroups, input.RupturePerGroup,
    #                                                    input.flag_include_point_source)
    #                logging.info(f"\t...further filter rupture scenarios by rmax and exported to:")
    #                logging.info(f"\t\t{rup_meta_file_tr_rmax}")
    #                logging.info(f"\tList of rupture groups exported to:")
    #                logging.info(f"\t\t{File_RuptureGroups}")
    #                
    #            # check if trace and intersect directories are empty
    #            if input.flag_get_fault_xing is True & (len(os.listdir(trace_dir)) == 0 or len(os.listdir(intersect_dir)) == 0):
    #                logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #                OpenSHAInterface.get_fault_xing(reg_proc, site_data.loc[:,['Lon_start','Lat_start']].values,
    #                                                site_data.loc[:,['Lon_end','Lat_end']].values,
    #                                                trace_dir, intersect_dir, rup_meta_file_tr_rmax,
    #                                                src_model=input.src_model)
    #                logging.info(f"\n")
    #                logging.info(f"\tRupture (segment) traces exported to:")
    #                logging.info(f"\t\t{trace_dir}")
    #                logging.info(f"\tFault crossings exported to:")
    #                logging.info(f"\t\t{intersect_dir}")
        

    # -----------------------------------------------------------
    # Get predictions of IMs from GMMs
    #if input.phase_to_run == 2:
    #    
    #    #
    #    if 'regionalprocessor' in input.im_tool.lower():
    #        
    #        # initialize processor
    #        logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
    #        reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
    #                                                path_vs30=vs30_file, numThreads=input.Num_Threads,
    #                                                rmax_cutoff=input.r_max)
    #        logging.info(f"\n")
    #        
    #        logging.info(f"\tLooping through all {len(list_rup_group)} groups of ruptures and getting GM predictions")
    #        # loop through every rup group and get predictions
    #        for rup_group in list_rup_group:
    #        # rup_group = list_rup_group[0]
    #            # folder to store predictions for current group
    #            gm_pred_rup_group_dir = os.path.join(gm_pred_dir,rup_group)
    #            fcn_common.make_dir(gm_pred_rup_group_dir) # create folder if it doesn't exist
    #            # see if files already exist
    #            # if len(os.listdir(gm_pred_rup_group_dir)) != 6:
    #            # current range of ruptures for current group
    #            rup_start = int(rup_group[:rup_group.find('_')])
    #            rup_end = int(rup_group[rup_group.find('_')+1:])+1
    #            #
    #            # logging.info(f"\tGetting GM predictions for rup_group {rup_group}")
    #            OpenSHAInterface.runHazardAnalysis(processor=reg_proc,rup_meta_file=rup_meta_file_tr_rmax,
    #                                                ind_range=[rup_start,rup_end],saveDir=gm_pred_rup_group_dir,
    #                                                output_file_type=input.output_file_type)
    #            
    #        logging.info(f"\tGenerated predictions for all {len(list_rup_group)} groups of ruptures; results stored under:")
    #        logging.info(f"\t\t{gm_pred_dir}")
    
    
    # -----------------------------------------------------------
    # Generate realizations of IMs (i.e., sampling); assess EDPs, DMs, DVs
    #if input.phase_to_run >= 3:
    #
    #    # create assessment class object
    #    model = model.assessment()
    #    logging.info(f'Created "model" class object named "model"')
    #
    #    # -----------------------------------------------------------
    #    # define multiplier; grops to run = multiplier * RupturePerGroup
    #    # multi_start = 14
    #    # n_multi = 1 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #    # if flag_get_IM or flag_gen_sample:
    #        # multi_end = multi_start+n_multi
    #        # logging.info(f"Multiplers to run: {multi_start} to {multi_end-1}.")
    #    # else:
    #        # multi_end = multi_start+1
    #        # logging.info(f"Samples not needed, no need to loop multi.")
    #
    #    time_start = time.time()
    #
    #    # loop through multipliers
    #    # for multi in range(multi_start,multi_end):
    #    
    #    # define range of ruptures for IM
    #    # range_start = RupturePerGroup*multi
    #    # if flag_get_IM or flag_gen_sample:
    #    #     range_end = range_start+RupturePerGroup
    #    #     # range_end = range_start+1
    #    #     logging.debug(f"Multi {multi}...")
    #    # else:
    #    #     range_end = range_start+1
    #    # count = range_start
    #    count = 0
    #
    #    # loop through groups
    #    # for rup_group in list_rup_group[range_start:range_end]:
    #    for rup_group in list_rup_group:
    #        count += 1
    #        count_EDP = 0
    #        logging.debug(f"-----------------------------------------------------------")
    #        # logging.info(f'\tCount = {count}: rupture group = {rup_group}...')
    #        logging.info(f'\t...running rupture group = {rup_group}')
    #        
    #        # -----------------------------------------------------------
    #        logging.debug(f"\t-------------Intensity Measures-------------")
    #        # -----------------------------------------------------------
    #        # load GM predictions and create random variable
    #        model.get_src_GM_site(input.phase_to_run, site_data, input.im_tool, gm_pred_dir,
    #                    imt, rup_meta_file_tr_rmax, flag_clear_dict=True,
    #                    output_file_type=input.output_file_type, rup_group=rup_group)
    #        logging.debug(f'\tIM_rv: Updated "_GM_pred_dict"')
    #        
    #        # -----------------------------------------------------------
    #        # generate/import samples
    #        # if flag_get_IM or flag_gen_sample:
    #        # directory name for current rupture group
    #        sample_dir = os.path.join(im_dir,rup_group)
    #        fcn_common.make_dir(sample_dir) # create folder if it doesn't exist
    #        # Check if samples exist for current rupture group, default to True (samples exist and will not generate again)
    #        flag_im_sample_exist = True # default
    #        list_dir_im = os.listdir(sample_dir) # get list of files under "sample_dir"
    #        # First check if the number of files in "sample_dir" is consistent with the expected number of files (# of IMs * # of samples per IM)
    #        if len(list_dir_im) < len(imt)*input.n_samp_im:
    #            flag_im_sample_exist = False # if file count is inconsistent, then set flag for IM existing to False such that IM samples will be saved
    #        # Second check to see if sample 0 (first sample) exist for each IM; if not, then set flag to False
    #        for im in imt:
    #            im_samp_file_name = im+'_samp_0'+'.'+input.output_file_type
    #            if not im_samp_file_name in list_dir_im:
    #                flag_im_sample_exist = False
    #                break
    #        # flag to force resample of IM
    #        if input.Flag_ResampleIM:
    #            flag_im_sample_exist = False
    #        #
    #        model.sim_IM(n_samp_im=input.n_samp_im, ims=imt,
    #                flag_spatial_corr=input.flag_spatial_corr, flag_cross_corr=input.flag_cross_corr, 
    #                Flag_SampleWithSigmaTotal=input.Flag_SampleWithSigmaTotal,
    #                sigma_aleatory=input.sigma_aleatory,
    #                flag_clear_dict=True, flag_im_sample_exist=flag_im_sample_exist,
    #                sample_dir=sample_dir, output_file_type=input.output_file_type)
    #                
    #       # If IM samples exist in the (i.e., phase_to_run == 4)
    #        if flag_im_sample_exist:
    #            logging.debug(f'\tIM_sim: Updated "_IM_dict" using path to samples:')
    #        else:
    #            # for im in imt:
    #                # for samp_i in range(input.n_samp_im):
    #                    # save_name = os.path.join(sample_dir,im+'_samp_'+str(samp_i)+'.'+input.output_file_type)
    #                    # samp = model._IM_dict[im][samp_i] # pull sparse matrix for sample from class
    #                    # file_io.store_im_samp(save_name, samp, input.output_file_type, input.n_demicals)
    #                    # sparse_mat = model._IM_dict[im][samp_i].log1p().tocsc() # pull sparse matrix for sample from class
    #                    # sparse_mat = model._IM_dict[im][samp_i] # pull sparse matrix for sample from class
    #                    # sparse_mat.data = np.round(sparse_mat.data,decimals=input.Num_Decimals)
    #                    # if input.output_file_type == 'npz':
    #                        
    #                    # sparse.save_npz(os.path.join(sample_dir,im+'_samp_'+str(samp_i)+'.npz'),sparse_mat)
    #            logging.debug(f'\tIM_sim: Updated "_IM_dict" by generating samples and storing to:')
    #            logging.debug(f'\t\t{sample_dir}')
    #
    #        # print(model._IM_dict)
    #        # sys.exit()
    #    # -----------------------------------------------------------
    #    # Assess EDPs, DMs, DVs
    #    if input.phase_to_run == 4:
    #        # -----------------------------------------------------------
    #        logging.debug(f"\t-------------Engineering Demand Parameters-------------")
    #        # -----------------------------------------------------------
    #        # Loop through EDPs
    #        for edp in input.edps:
    #            #
    #            count_EDP += 1 # counter for number of EDP evaluated
    #            
    #            # Liquefaction
    #            if 'liq' in edp.lower():
    #                # check for need for probability of liquefaction and liquefactino susceptibility
    #                if flag_p_liq and 'p_liq' not in input.edp_procs[edp]['return_param']:
    #                    input.edp_procs[edp]['return_param'].append('p_liq')
    #                if flag_liq_susc and 'liq_susc' not in input.edp_procs[edp]['return_param']:
    #                    input.edp_procs[edp]['return_param'].append('liq_susc')
    #                
    #            # Landslide
    #            elif 'land' in edp.lower():
    #                # See if probability of landslide is needed
    #                if flag_p_land and 'p_land' not in input.edp_procs[edp]['return_param']:
    #                    input.edp_procs[edp]['return_param'].append('p_land')
    #
    #            # Surface fault rupture
    #            elif 'surf' in edp.lower():
    #                # set up flags for fault crossings
    #                rows = []
    #                cols = []
    #                n_rup = len(model._GM_pred_dict['rup']['src'])
    #                n_site = len(model._src_site_dict['site_lon'])
    #                count_seg = 0
    #                for src in model._GM_pred_dict['rup']['src']:
    #                    with warnings.catch_warnings():
    #                        warnings.simplefilter("ignore")
    #                        seg_list = np.loadtxt(os.path.join(intersect_dir,'src_'+str(src)+'.txt'),dtype=int,ndmin=1)
    #                    for seg in seg_list:
    #                        rows.append(count_seg)
    #                        cols.append(seg)
    #                    count_seg += 1
    #                rows = np.asarray(rows)
    #                cols = np.asarray(cols)
    #                mat = sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n_rup,n_site))
    #                logging.debug(f'\tEDP_{edp}: Generated matrix of flags for fault crossing')
    #                input.edp_other_params.update({'mat_seg2calc':mat})
    #                
    #
    #            # Print setup for current EDP
    #            logging.debug(f"\tEDP_{edp}: Calculate {input.edp_procs[edp]['return_param']} using {input.edp_procs[edp]['method']}, with {input.edp_procs[edp]['source_param']} from {input.edp_procs[edp]['source_method']}")
    #            logging.debug(f"\t\t-->epsilons for aleatory variability = {input.edp_procs[edp]['eps_aleatory']}, epsilons for epistemic uncertainty = {input.edp_procs[edp]['eps_epistemic']} (if 999, then use lumped sigma)")
    #            
    #            # if first EDP, clear _EDP_dict
    #            flag_clear_dict = True if count_EDP == 1 else False
    #        
    #            # evaluate EDP with catch on Numpy warnings
    #            with warnings.catch_warnings():
    #                warnings.simplefilter("ignore")
    #                model.assess_EDP(edp, input.edp_procs[edp], input.edp_other_params, input.n_samp_im, flag_clear_dict)
    #            
    #            # pop any keys added to dictionary while in loop
    #            try:
    #                input.edp_other_params.pop('mat_seg2calc')
    #            except:
    #                pass
    #            
    #        
    #        
    #        # -----------------------------------------------------------
    #        # temporary; for checks
    #        # print(type(model._EDP_dict['pgd_gs']['method1']['output']) )
    #           
    #        for edp_i in model._EDP_dict:
    #            for method_i in model._EDP_dict[edp_i]:
    #                if isinstance(model._EDP_dict[edp_i][method_i]['output'],dict):
    #                    for sample_i in model._EDP_dict[edp_i][method_i]['output']:
    #                        print(f"edp = {edp_i}, sample = {sample_i}, output = {model._EDP_dict[edp_i][method_i]['output'][sample_i].toarray()}")
    #                elif isinstance(model._EDP_dict[edp_i][method_i]['output'],sparse.coo.coo_matrix):
    #                    print(f"edp = {edp_i}, output = {model._EDP_dict[edp_i][method_i]['output'].toarray()}")
    #                else:
    #                    print(f"edp = {edp_i}, output = {model._EDP_dict[edp_i][method_i]['output']}")
    #        # -----------------------------------------------------------
    #
    #        # -----------------------------------------------------------
    #        logging.debug(f"\t-------------Damage Measures-------------")
    #        # -----------------------------------------------------------
    #        # Nothing here at the moment
    #
    #        # -----------------------------------------------------------
    #        logging.debug(f"\t-------------Decision Variables-------------")
    #        # -----------------------------------------------------------
    #        # Loop through DVs
    #        for dv in input.dvs:
    #        
    #            # Loop through DV methods
    #            for proc_i in input.dv_procs[dv]['method']:
    #            
    #                # for rr_pgv
    #                if 'rr_pgv' in dv.lower():
    #                
    #                    # Print setup for current DV
    #                    logging.debug(f"\tDV_{dv}: Calculate {input.dv_procs[dv]['return_param']} using '{proc_i}'")
    #                    
    #                    # evaluate DV with catch on Numpy warnings
    #                    with warnings.catch_warnings():
    #                        warnings.simplefilter("ignore")
    #                        model.assess_DV(dv[:dv.find('_')], proc_i, input.dv_procs[dv], 
    #                                                input.dv_other_params, n_samp_im=input.n_samp_im, 
    #                                                n_samp_edp=input.n_samp_edp)
    #                
    #                # for rr_pgd
    #                elif 'rr_pgd' in dv.lower():
    #                    
    #                    # generate store ID for DV
    #                    input.dv_other_params['pgd_label'] = {}
    #                    
    #                    # Loop through EDPs
    #                    for edp in input.edps:
    #                        
    #                        if edp.lower() != 'liq':
    #                            # ID for labeling DV with EDP
    #                            input.dv_other_params.update({'pgd_label':'pgd_'+edp})
    #                            
    #                            # Print setup for current DV and EDP
    #                            logging.debug(f"\tDV_{dv}: Calculate {input.dv_procs[dv]['return_param']} using '{proc_i}', with '{input.dv_procs[dv]['source_param'][edp]} from {input.dv_procs[dv]['source_method'][edp]}.")
    #                            
    #                            # evaluate DV with catch on Numpy warnings
    #                            with warnings.catch_warnings():
    #                                warnings.simplefilter("ignore")
    #                                model.assess_DV(dv[:dv.find('_')], proc_i, input.dv_procs[dv], 
    #                                                        input.dv_other_params, edp=edp, n_samp_im=input.n_samp_im, n_samp_edp=input.n_samp_edp)
    #                                                  
    #                            # pop any keys added to dictionary while in loop
    #                            try:
    #                                input.dv_other_params.pop('pgd_label')
    #                            except:
    #                                pass


# -----------------------------------------------------------
# Messages to print
# -----------------------------------------------------------

    #    # -----------------------------------------------------------
    #    # print message for number of groups run
    #    if input.phase_to_run >= 3 and count % inc1 == 0:
    #            logging.info(f"-------------After {count*input.RupturePerGroup} groups: {np.round(time.time()-time_start,decimals=2)} secs-------------")
    #            time_start = time.time()
    #    
    #    # -----------------------------------------------------------
    #    if input.phase_to_run == 4 and Flag_SaveDV and model._DV_dict is not None:
    #        if count % inc2 == 0 or count == len(list_rup_group):
    #            logging.info(f"-----------------------------------------------------------")
    #            logging.info(f"\tSaving damages...")
    #            logging.info(f"\tExport directory:")
    #            logging.info(f"\t\t{dv_dir}")
    #            if input.flag_export_to_csv:
    #                logging.info(f"\tFiles (Also exported to txt):")
    #            else:
    #                logging.info(f"\tFiles:")
    #            for dv_i in model._DV_dict.keys():
    #                logging.info(f"\t\t----------")
    #                count_proc = 0
    #                for proc_i in model._DV_dict[dv_i].keys():
    #                    if 'rr_pgd' in dv_i:
    #                        for i in model._DV_dict[dv_i][proc_i]['source_param']:
    #                            if 'pgd' in i:
    #                                edp_i = i
    #                        str_edp_i = edp_i[edp_i.find('_')+1:]+'_'
    #                    else:
    #                        str_edp_i = ''
    #                    str_proc_i = model._DV_dict[dv_i][proc_i]['method'][0:model._DV_dict[dv_i][proc_i]['method'].find('_')]
    #                    if model._DV_dict[dv_i][proc_i]['eps_epistemic'] == 999:
    #                        str_eps = 'epiTotal_'
    #                    else:
    #                        str_eps = 'epi'+str(model._DV_dict[dv_i][proc_i]['eps_epistemic'])+'_'
    #                        str_eps = str_eps.replace('.','o')
    #                        str_eps = str_eps.replace('-','m')
    #                    # str_range = 'rup_'+str(range_start)+'_'+str(range_end-1)
    #                    str_range = 'rup_all'
    #                    save_name_npz = dv_i+'_'+str_proc_i+'_'+str_edp_i+str_eps+str_range+'.npz'
    #                    save_path_npz = os.path.join(dv_dir,save_name_npz)
    #                    logging.info(f"\t\t{save_name_npz}")
    #                    sparse.save_npz(save_path_npz,np.transpose(model._DV_dict[dv_i][proc_i]['output']))
    #                    #
    #                    if input.flag_export_to_csv:
    #                        save_name_csv = save_name_npz[:save_name_npz.find('.npz')]+'.txt'
    #                        save_path_csv = os.path.join(dv_dir,save_name_csv)
    #                        np.savetxt(save_path_csv,np.transpose(model._DV_dict[dv_i][proc_i]['output'].toarray()),fmt='%5.3e')
    #                    count_proc += 1
    #
    ## 
    #logging.info(f"-----------------------------------------------------------")
    #logging.info(f"...done with current run - program will now exit")

    
# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Open-source Seismic Risk Analysis (OpenSRA)'
        )
    
    # Define arguments
    # input directory
    parser.add_argument('-i', '--input', help='Path of input directory"]')
    # clean previous analysis
    parser.add_argument('-c', '--clean', help='Clean previous analysis: "y" or "n" (default)', default='n')
    # logging
    parser.add_argument('-l', '--logging', help='Logging level: "info"(default) or "debug"', default='info')
    
    # Parse command line input
    args = parser.parse_args()
    
    # Run "Main"
    main(
        input_dir = args.input,
        clean_prev_run = args.clean,
        logging_level = args.logging,
    )
