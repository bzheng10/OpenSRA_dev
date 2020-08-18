#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### Main function
#####
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####
#####
##### Instructions
##### ------------
##### Currently, the program should be run in four phases:
##### 1. Get rupture scenarios, and fault-crossing and geologic units and properties at sites
##### 2. Get ground motion predictions at target sites given rupture scenarios
##### 3. Generate and store intensity measure realizations
##### 4. Assess EDPs, DMs, DVs using IM realizations
#####################################################################################################################


#####################################################################################################################
##### Python modules
import importlib, os, time, warnings, logging, sys, argparse, json
import numpy as np
import pandas as pd
from scipy import sparse

##### OpenSRA modules and functions
from src import model, fcn_gen
from src.fcn_gen import setLogging
# from src.im import fcn_im
# from opensra_input import *
import input
from src.site.get_geo_unit import get_geo_unit

##### SimCenter modules
import lib.simcenter.OpenSHAInterface as OpenSHAInterface



#####################################################################################################################
## Main function
def main(logging_level):


    #####################################################################################################################
    ##### Setting logging level (e.g. DEBUG or INFO)
    setLogging(logging_level)
    logging.info('===========================================================')


    #####################################################################################################################
    ##### Additional set up given user inputs
    
    ##### General
    site_dir = os.path.join(input.work_dir, 'site_out') # set full path for site output folder
    if os.path.isdir(site_dir) is False: # check if directory exists
        os.mkdir(site_dir) # create folder if it doesn't exist
    site_file  = os.path.join(input.work_dir,input.site_file_name) # create full path for site data file
    site_data = pd.read_csv(site_file) # read site data file
    flag_update_site_data = False # default
    
    ## check if segment lengths are given in the file; if not then assume 1 km and add to site_data
    if not 'l_seg (km)' in site_data.columns:
        site_data['l_seg (km)'] = np.ones(len(site_data))
        flag_update_site_data = True # updated site data
        
    ## check if start and end locations are given in the file; if not set equal to site locations (i.e., points)
    if not 'Lon_start' in site_data.columns:
        site_data['Lon_start'] = site_data['Longitude'].values
        site_data['Lat_start'] = site_data['Latitude'].values    
        site_data['Lon_end'] = site_data['Longitude'].values
        site_data['Lat_end'] = site_data['Latitude'].values
        flag_update_site_data = True # updated site data
        
    ## pad OpenSRA path to the user-defined relative paths
    input.geo_shp_file = os.path.join(os.getcwd(),input.geo_shp_file)
    input.geo_unit_param_file = os.path.join(os.getcwd(),input.geo_unit_param_file)
    ## check if files for site locations and vs30 exist in site_dir
    flag_export_site_loc = False # default
    site_loc_file = os.path.join(site_dir,'site_loc.txt')
    if not os.path.exists(site_loc_file):
        np.savetxt(site_loc_file,site_data.loc[:,['Longitude','Latitude']].values,fmt='%10.6f,%10.6f') # export vs30 to file for gm_tool
        flag_export_site_loc = True
    flag_export_vs30 = False # default
    vs30_file = os.path.join(site_dir,'vs30.txt')
    if not os.path.exists(vs30_file):
        np.savetxt(vs30_file,site_data['VS30 (m/s)'].values,fmt='%6.2f') # export vs30 to file for gm_tool
        flag_export_vs30 = True

    ##### Source and ground motion predictions
    gm_dir = os.path.join(input.work_dir, 'gm_out') # set full path for GM output folder
    if os.path.isdir(gm_dir) is False:
        os.mkdir(gm_dir) # create folder if it doesn't exist
    gm_pred_dir = os.path.join(gm_dir, 'gm_pred') # set full path for GM prediction output folder
    if os.path.isdir(gm_pred_dir) is False:
        os.mkdir(gm_pred_dir) # create folder if it doesn't exist
    rup_seg_file = os.path.join(os.getcwd(),'lib','simcenter',input.src_model,'rup_seg.json')
    pt_src_file = os.path.join(os.getcwd(),'lib','simcenter',input.src_model,'point_source.txt')
    rup_meta_file_tr = os.path.join(gm_dir,'rup_meta_'+str(input.tr)+'yr.hdf5') # create full path for rupture metafile, which contains magnitudes and rates for rupture scenarios screened by return period
    rup_meta_file_tr_rmax = os.path.join(gm_dir,'rup_meta_'+str(input.tr)+'yr_'+str(input.rmax)+'km.hdf5') # create full path for rupture metafile, which contains magnitudes and rates for rupture scenarios screened by return period
    rup_group_file = os.path.join(gm_dir,'list_rup_group_'+str(input.tr)+'yr_'+str(input.rmax)+'km.txt') # create full path for list of groups of ruptures
    ## check if need to extract geologic units and properties, get the params if not
    if not 'Geologic Unit' in site_data.columns:
        ## run script to get the properties
        site_data = get_geo_unit(site_data, input.geo_shp_file, input.geo_unit_param_file)
        flag_update_site_data = True # updated site data
    ## check if site data is updated, if so then save to site file
    if flag_update_site_data:
        site_data.to_csv(site_file,index=False)
        
    ## load rupture groups
    if os.path.exists(rup_group_file):
        list_rup_group = np.loadtxt(rup_group_file,dtype=str)
    
    ## check for directories for storing fault traces and intersections; create if don't exist
    trace_dir = os.path.join(gm_dir,'src_trace')
    if os.path.isdir(trace_dir) is False:
        os.mkdir(trace_dir) # create folder if it doesn't exist
    intersect_dir = os.path.join(gm_dir,'src_intersect')
    if os.path.isdir(intersect_dir) is False:
        os.mkdir(intersect_dir) # create folder if it doesn't exist
    
    ##### Intensity measures
    im_dir = os.path.join(input.work_dir, 'im_out') # set full path for IM output folder
    # flag_im_sample_exist = True # default
    if os.path.isdir(im_dir) is False: # check if directory exists
        os.mkdir(im_dir) # create folder if it doesn't exist
        # flag_im_sample_exist = False # True if IM samples have been generated, else False
    # else:
        # if len(os.listdir(im_dir)) == 0: # see if IM folder is empty (do samples exist)
            # flag_im_sample_exist = False # True if IM samples have been generated, else False
    ## check if IM samples are needed
    flag_get_IM = False # default
    if 'rr_pgv' in input.dvs:
        flag_get_IM = True
    if 'rr_pgd' in input.dvs:
        if 'hazus' in input.dv_procs:
            flag_get_IM = True
        else:
            if 'ls' in input.edps or 'land' in input.edps or 'surf' in input.edps:
                flag_get_IM = True
    ## update required 'edps' and 'ims' if only 'rr_pgv' is requested
    if 'rr_pgv' in input.dvs and len(input.dvs) == 1:
        input.edps = []
        input.ims = ['pgv']
    if flag_get_IM is False:
        input.ims = []
                    

    ##### Engineering demand parameters
    ##### note: OpenSRA currently is not set up to store dm results
    edp_dir = os.path.join(input.work_dir, 'edp_out') # set full path for EDP output folder
    if os.path.isdir(edp_dir) is False: # check if directory exists
        os.mkdir(edp_dir) # create folder if it doesn't exist
        
    ## if EDP samples exist
    flag_edp_sample_exist = False
    
    ## check if probability of liquefaction is needed
    flag_p_liq = False # default
    flag_liq_susc = False # default
    if 'liq' in input.edps:
        flag_liq_susc = True
    if 'ls' in input.edps:
        flag_p_liq = True
    
    ##
    if input.phase_to_run == 4:
        ## for liquefaction-induced demands
        if flag_liq_susc or flag_p_liq:
        
            ## import general inputs
            wtd = site_data['WTD_300m (m)'].values
            dr = site_data['Dist_River (km)'].values
            dc = site_data['Dist_Coast (km)'].values
            dw = site_data['Dist_Any Water (km)'].values
            precip = site_data['CA_Precip (mm)'].values
            vs30 = site_data['VS30 (m/s)'].values
            
            ## change all -9999 entries to NaN
            find_val = -9999
            set_val = np.nan
            wtd = fcn_gen.find_set_nan(wtd,find_val,set_val)
            dr = fcn_gen.find_set_nan(dr,find_val,set_val)
            dc = fcn_gen.find_set_nan(dc,find_val,set_val)
            dw = fcn_gen.find_set_nan(dw,find_val,set_val)
            precip = fcn_gen.find_set_nan(precip,find_val,set_val)
            vs30 = fcn_gen.find_set_nan(vs30,find_val,set_val)
            
            ## elevation from DEM maps
            z = np.ones(vs30.shape)*10 ## set to 10 for now, get data from DEM map later

            ## add params to input
            input.edp_other_params.update({'wtd':wtd, 'dr':dr, 'dc':dc, 'dw':dw,
                                        'precip':precip, 'vs30':vs30, 'z':z})

            ## for ground settlement
            if 'ls' in input.edps:
                dwWB = site_data['Dist_Any Water (km)'].values
                dwWB = fcn_gen.find_set_nan(dwWB,find_val,set_val)
                dwWB = dwWB*1000 # convert to meters
                input.edp_other_params.update({'dwWB':dwWB})

        ## check if probability of landslide is needed
        flag_p_land = False
        if 'rr_pgd' in input.dvs and 'land' in input.edps:
            flag_p_land = True

        ## for landslide-induced demands
        if 'land' in input.edps:
            input.edp_other_params.update({'ky':site_data['ky_inf_bray'].values})
            
        ## for surface fault rupture
        if 'surf' in input.edps:
            input.edp_other_params.update({'l_seg':site_data['l_seg (km)'].values})
            input.dv_other_params.update({'l_seg':site_data['l_seg (km)'].values})
            # intersect_dir = os.path.join(input.work_dir,'gm_resource','rup_intersect')
            # logging.info(f"Defined directory with fault crossings: {intersect_dir}.")

    ##### Damage measures
    ##### note: OpenSRA currently is not set up to store dm results
    dm_dir = os.path.join(input.work_dir, 'dm_out') # set full path for DM output folder
    if os.path.isdir(im_dir) is False: # check if directory exists
        os.mkdir(im_dir) # create folder if it doesn't exist
    ## if EDP samples exist
    flag_dm_sample_exist = False

    ##### Decision variables
    dv_dir = os.path.join(input.work_dir, 'dv_out') # set full path for DV output folder
    if os.path.isdir(dv_dir) is False: # check if directory exists
        os.mkdir(dv_dir) # create folder if it doesn't exist
    ## increment to print output messages
    flag_save_dv = True # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if len(input.dvs) == 0:
        flag_save_dv = False
    ## for printing message per group
    inc1 = 5 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    inc2 = 100 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    ##### Other setups
    

    #####################################################################################################################
    ##### Print messages for...
    ## General
    logging.info(f"Analysis Details")
    logging.info("===========================================================")
    logging.info(f"Setup for phase {input.phase_to_run}: {input.phase_message}")
    
    ## Site data
    logging.info(f"Site locations and data")
    logging.info(f"\tLoaded site locations and data from:")
    logging.info(f"\t\t{site_file}")
    if flag_update_site_data:
        logging.info(f"\tUpdated and saved site data to the original file")
    if flag_export_site_loc:
        logging.info(f"\tExported site locations for {gm_dir}")
    if flag_export_vs30:
        logging.info(f"\tExported vs30 for {gm_dir}")

    ## Source and ground motion predictions
    logging.info(f"EQ sources and GMs")
    logging.info(f"\tDirectory for source and GM outputs:")
    logging.info(f"\t\t{gm_dir}")
    logging.info(f"\tSource and ground motion tool: {input.gm_tool}")
    logging.info(f"\tSource model: {input.src_model}")
    logging.info(f"\tCutoff return period: {input.tr} years")
    logging.info(f"\tCutoff distance: {input.rmax} km")

    ## for phases 2+
    if input.phase_to_run >= 2:
        logging.info(f"\tGM model: {input.gm_model}")
        # logging.info(f"Number of groups of ruptures = {len(list_rup_group)} (each group contains {rup_per_group} ruptures)")
        
        if flag_get_IM or flag_gen_sample:
            logging.info(f"\tWill print messages every {inc1} groups (each group contains {input.rup_per_group} ruptures)")
        if flag_save_dv:
            logging.info(f"\tWill save damage outputs every {inc2} groups (each group contains {input.rup_per_group} ruptures)")
    
    ## for phase 3
    if input.phase_to_run >= 3:
        logging.info(f"IMs")
        if input.phase_to_run == 3:
            logging.info(f"\tRequested IMs include: {input.ims}")
            logging.info(f"\tPerform spatial correlation = {input.flag_spatial_corr}")
            logging.info(f"\tPerform cross-correlation = {input.flag_cross_corr}")
        elif input.phase_to_run == 4:
            logging.info(f"\tRequired IMs include: {input.ims} (list may be altered based on requested DVs)")
        logging.info(f"\tNumber of IM samples = {input.n_samp_im}")

    ## for phase 4
    if input.phase_to_run == 4:
        ##
        logging.info(f"EDPs")
        logging.info(f"\tRequired EDPs include: {input.edps} (list may be altered based on requested DVs)")
        logging.info(f"\tNumber of EDP samples = {input.n_samp_edp} (for uniform distributions use only)")
        logging.info(f"\tFlag for probability of liquefaction = {flag_p_liq}")
        logging.info(f"\tFlag for liquefaction susceptibility = {flag_liq_susc}")
        logging.info(f"\tFlag for probability of landslide = {flag_p_land}")
        # if flag_p_liq or flag_liq_susc:
            # logging.info(f"\tLoaded site parameters for liquefaction into 'edp_other_params'")
        # if flag_calc_ky:
            # logging.info(f"Calculated ky using Bray (2007) for infinite slope for landslide-induced demands.")
        # else:
            # logging.info(f"Loaded ky into 'edp_other_params'")
                
        ##
        logging.info(f"DMs")
        logging.info(f"\tRequired DMs include: {input.dms} (list may be altered based on requested DVs)")
        logging.info(f"\tNumber of DM samples = {input.n_samp_dm} (for uniform distributions use only)")
        ##
        logging.info(f"DVs")
        logging.info(f"\tRequested DVs include: {input.dvs}")
        logging.info(f"\tDirectory for DV outputs:")
        logging.info(f"\t\t{dv_dir}")
    
    
    #####################################################################################################################
    ##### Print messages for...
    ## reload model.py if it has been modified
    # importlib.reload(model)
    # logging.info(f'Load/reloaded "model.py".')

    ##
    logging.info("===========================================================")
    logging.info(f"Performing phase {input.phase_to_run} analysis")
    
    ## get list of rupture scenarios filtered by return period and shortest distance
    if input.phase_to_run == 1:
        
        ##
        if 'regionalprocessor' in input.gm_tool.lower():
        
            ##
            if not os.path.exists(rup_meta_file_tr) or not os.path.exists(rup_meta_file_tr_rmax) or len(os.listdir(trace_dir)) == 0 or len(os.listdir(intersect_dir)) == 0:
            
                ## initialize processor            
                logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
                reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
                                                        path_vs30=vs30_file, numThreads=input.num_thread,
                                                        rmax_cutoff=input.rmax)
                logging.info(f"\n")
            
                ## check if rupture metafile screened by return period is already created
                if not os.path.exists(rup_meta_file_tr):
                    logging.info(f"\t...generating list of rupture scenarios given cutoff on return period (this may take some time)")
                    ## run to get a list of ruptures given criteria on tr and rmax
                    OpenSHAInterface.load_src_rup_M_rate(rup_meta_file=rup_meta_file_tr, ind_range=['all'],
                                                        proc=reg_proc, rate_cutoff=1/input.tr)
                    logging.info(f"\tList of rupture scenarios filtered by return period exported to:")
                    logging.info(f"\t\t{rup_meta_file_tr}")
                    
                ## check if rupture metafile screened by return period and maximum distance is already created
                if not os.path.exists(rup_meta_file_tr_rmax):
                    OpenSHAInterface.filter_src_by_rmax(site_data.loc[:,['Longitude','Latitude']].values,
                                                        input.rmax, rup_meta_file_tr, rup_meta_file_tr_rmax,
                                                        rup_seg_file, pt_src_file,rup_group_file, input.rup_per_group,
                                                        input.flag_include_point_source)
                    logging.info(f"\t...further filter rupture scenarios by rmax and exported to:")
                    logging.info(f"\t\t{rup_meta_file_tr_rmax}")
                    logging.info(f"\tList of rupture groups exported to:")
                    logging.info(f"\t\t{rup_group_file}")
                    
                ## check if trace and intersect directories are empty
                if len(os.listdir(trace_dir)) == 0 or len(os.listdir(intersect_dir)) == 0:
                    logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
                    OpenSHAInterface.get_fault_xing(reg_proc, site_data.loc[:,['Lon_start','Lat_start']].values,
                                                    site_data.loc[:,['Lon_end','Lat_end']].values,
                                                    trace_dir, intersect_dir, rup_meta_file_tr_rmax)
                    logging.info(f"\n")
                    logging.info(f"\tRupture (segment) traces exported to:")
                    logging.info(f"\t\t{trace_dir}")
                    logging.info(f"\tFault crossings exported to:")
                    logging.info(f"\t\t{intersect_dir}")
        
    ## get GM prediction
    if input.phase_to_run == 2:
        
        ##
        if 'regionalprocessor' in input.gm_tool.lower():
                
            ## initialize processor
            logging.info(f"\n----------------------------------\n-----Runtime messages from OpenSHA\n")
            reg_proc = OpenSHAInterface.init_processor(case_to_run=2, path_siteloc=site_loc_file,
                                                    path_vs30=vs30_file, numThreads=input.num_thread,
                                                    rmax_cutoff=input.rmax)
            logging.info(f"\n")
            
            logging.info(f"\tLooping through all {len(list_rup_group)} groups of ruptures and getting GM predictions")
            ## loop through every rup group and get predictions
            for rup_group in list_rup_group:
            # rup_group = list_rup_group[0]
                ## folder to store predictions for current group
                gm_pred_rup_group_dir = os.path.join(gm_pred_dir,rup_group)
                if os.path.isdir(gm_pred_rup_group_dir) is False: # check if directory exists
                    os.mkdir(gm_pred_rup_group_dir) # create folder if it doesn't exist
                ## see if files already exist
                # if len(os.listdir(gm_pred_rup_group_dir)) != 6:
                ## current range of ruptures for current group
                rup_start = int(rup_group[:rup_group.find('_')])
                rup_end = int(rup_group[rup_group.find('_')+1:])+1
                ##
                # logging.info(f"\tGetting GM predictions for rup_group {rup_group}")
                OpenSHAInterface.runHazardAnalysis(reg_proc,rup_meta_file_tr_rmax,[rup_start,rup_end],gm_pred_rup_group_dir)
                
            logging.info(f"\tGenerated predictions for all {len(list_rup_group)} groups of ruptures; results stored under:")
            logging.info(f"\t\t{gm_pred_dir}")
    
    
    #####################################################################################################################
    ##
    if input.phase_to_run >= 3:
    
        ##### create assessment class object
        model_assess = model.assessment()
        logging.info(f'Created model class object named "model_assess"')


        #####################################################################################################################
        ##### define multiplier; grops to run = multiplier * rup_per_group
        # multi_start = 14
        # n_multi = 1 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # if flag_get_IM or flag_gen_sample:
            # multi_end = multi_start+n_multi
            # logging.info(f"Multiplers to run: {multi_start} to {multi_end-1}.")
        # else:
            # multi_end = multi_start+1
            # logging.info(f"Samples not needed, no need to loop multi.")

        time_start = time.time()

        ## loop through multipliers
        # for multi in range(multi_start,multi_end):
            
        ## define range of ruptures for IM
        #range_start = rup_per_group*multi
        #if flag_get_IM or flag_gen_sample:
        #    range_end = range_start+rup_per_group
        #    # range_end = range_start+1
        #    logging.debug(f"Multi {multi}...")
        #else:
        #    range_end = range_start+1
        # count = range_start
        count = 0

        ## loop through groups
        # for rup_group in list_rup_group[range_start:range_end]:
        for rup_group in list_rup_group:
            count += 1
            count_EDP = 0
            logging.debug(f"========================================================")
            logging.info(f'Count = {count}: rupture group = {rup_group}...')
            
            #####################################################################
            #####################################################################
            logging.debug(f"\t-------------Intensity Measures-------------")
            #####################################################################
            ## load GM predictions and create random variable
            model_assess.get_src_GM_site(input.phase_to_run, site_data, input.gm_tool, gm_pred_dir,
                        input.ims, rup_meta_file_tr_rmax, flag_clear_dict=True,
                        rup_group=rup_group)
            logging.debug(f'\tIM_rv: Updated "_GM_pred_dict"')
            
            #####################################################################
            ## generate/import samples
            # if flag_get_IM or flag_gen_sample:
            ## directory name for current rupture group
            path_sample = os.path.join(im_dir,rup_group)
            if os.path.isdir(path_sample) is False: # check if directory exists
                os.mkdir(path_sample) # create folder if it doesn't exist
            ## check if samples exist for current rupture group
            flag_im_sample_exist = True # default
            list_dir_im = os.listdir(path_sample)
            if len(list_dir_im) < len(input.ims)*input.n_samp_im:
                flag_im_sample_exist = False        
            for im in input.ims:
                if im+'_samp_0' in list_dir_im:
                    flag_im_sample_exist = False
                    break
            ## flag to force resample of IM
            if input.flag_force_resample_im:
                flag_im_sample_exist = False
            ##
            model_assess.sim_IM(input.n_samp_im, input.ims, input.flag_spatial_corr, input.flag_cross_corr, 
                    flag_sample_with_sigma_total=input.flag_sample_with_sigma_total,
                    sigma_aleatory=input.sigma_aleatory,
                    flag_clear_dict=True, flag_im_sample_exist=flag_im_sample_exist,
                    path_sample=path_sample)
            if flag_im_sample_exist:
                logging.debug(f'\tIM_sim: Updated "_IM_dict" using path to samples:')
            else:
                # if not os.path.isdir(os.path.join(im_dir,rup_group)):
                    # os.mkdir(os.path.join(im_dir,rup_group))
                for im in input.ims:
                    for samp_i in range(input.n_samp_im):
                        sparse_mat = model_assess._IM_dict[im][samp_i].log1p().tocsc()
                        sparse_mat.data = np.round(sparse_mat.data,decimals=input.n_decimals)
                        sparse.save_npz(os.path.join(path_sample,im+'_samp_'+str(samp_i)+'.npz'),sparse_mat)
                logging.debug(f'\tIM_sim: Updated "_IM_dict" by generating samples and storing to:')
            logging.debug(f'\t\t{path_sample}')


        ##
        if input.phase_to_run == 4:
            #####################################################################
            #####################################################################
            logging.debug(f"\t-------------Engineering Demand Parameters-------------")
            #####################################################################
            ## Loop through EDPs
            for edp in input.edps:
                ## 
                count_EDP += 1 # counter for number of EDP evaluated
                
                ## Liquefaction
                if 'liq' in edp.lower():
                    ## check 
                    if flag_p_liq and 'p_liq' not in input.edp_procs[edp]['return_param']:
                        input.edp_procs[edp]['return_param'].append('p_liq')
                    if flag_liq_susc and 'liq_susc' not in input.edp_procs[edp]['return_param']:
                        input.edp_procs[edp]['return_param'].append('liq_susc')
                    
                ## Landslide
                elif 'land' in edp.lower():
                    ## See if probability of landslide is needed
                    if flag_p_land and 'p_land' not in input.edp_procs[edp]['return_param']:
                        input.edp_procs[edp]['return_param'].append('p_land')

                ## Surface fault rupture
                elif 'surf' in edp.lower():
                    ## set up flags for fault crossings
                    rows = []
                    cols = []
                    n_rup = len(model_assess._GM_pred_dict['rup']['src'])
                    n_site = len(model_assess._src_site_dict['site_lon'])
                    count_seg = 0
                    for src in model_assess._GM_pred_dict['rup']['src']:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            seg_list = np.loadtxt(os.path.join(intersect_dir,'src_'+str(src)+'.txt'),dtype=int,ndmin=1)
                        for seg in seg_list:
                            rows.append(count_seg)
                            cols.append(seg)
                        count_seg += 1
                    rows = np.asarray(rows)
                    cols = np.asarray(cols)
                    mat = sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n_rup,n_site))
                    logging.debug(f'\tEDP_{edp}: Generated matrix of flags for fault crossing')
                    input.edp_other_params.update({'mat_seg2calc':mat})
                    

                ## Print setup for current EDP
                logging.debug(f"\tEDP_{edp}: Calculate {input.edp_procs[edp]['return_param']} using {input.edp_procs[edp]['method']}, with {input.edp_procs[edp]['source_param']} from {input.edp_procs[edp]['source_method']}")
                logging.debug(f"\t\t-->epsilons for aleatory variability = {input.edp_procs[edp]['eps_aleatory']}, epsilons for epistemic uncertainty = {input.edp_procs[edp]['eps_epistemic']} (if 999, then use lumped sigma)")
                
                ## if first EDP, clear _EDP_dict
                flag_clear_dict = True if count_EDP == 1 else False
            
                ## evaluate EDP with catch on Numpy warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_assess.assess_EDP(edp, input.edp_procs[edp], input.edp_other_params, input.n_samp_im, flag_clear_dict)
                
                ## pop any keys added to dictionary while in loop
                try:
                    input.edp_other_params.pop('mat_seg2calc')
                except:
                    pass

            #####################################################################
            #####################################################################
            logging.debug(f"\t-------------Damage Measures-------------")
            #####################################################################
            ## Nothing here at the moment

            #####################################################################
            #####################################################################
            logging.debug(f"\t-------------Decision Variables-------------")
            #####################################################################
            ## Loop through DVs
            for dv in input.dvs:
            
                ## Loop through DV methods
                for proc_i in input.dv_procs[dv]['method']:
                
                    ## for rr_pgv
                    if 'rr_pgv' in dv.lower():
                    
                        ## Print setup for current DV
                        logging.debug(f"\tDV_{dv}: Calculate {input.dv_procs[dv]['return_param']} using '{proc_i}'")
                        
                        ## evaluate DV with catch on Numpy warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_assess.assess_DV(dv[:dv.find('_')], proc_i, input.dv_procs[dv], 
                                                    input.dv_other_params, n_samp_im=input.n_samp_im, 
                                                    n_samp_edp=input.n_samp_edp)
                    
                    ## for rr_pgd
                    elif 'rr_pgd' in dv.lower():
                        
                        ## generate store ID for DV
                        input.dv_other_params['pgd_label'] = {}
                        
                        ## Loop through EDPs
                        for edp in input.edps:
                            
                            if edp.lower() != 'liq':
                                ## ID for labeling DV with EDP
                                input.dv_other_params.update({'pgd_label':'pgd_'+edp})
                                
                                ## Print setup for current DV and EDP
                                logging.debug(f"\tDV_{dv}: Calculate {input.dv_procs[dv]['return_param']} using '{proc_i}', with '{input.dv_procs[dv]['source_param'][edp]} from {input.dv_procs[dv]['source_method'][edp]}.")
                                
                                ## evaluate DV with catch on Numpy warnings
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    model_assess.assess_DV(dv[:dv.find('_')], proc_i, input.dv_procs[dv], 
                                                            input.dv_other_params, edp=edp, n_samp_im=input.n_samp_im, n_samp_edp=input.n_samp_edp)
                                                      
                                ## pop any keys added to dictionary while in loop
                                try:
                                    input.dv_other_params.pop('pgd_label')
                                except:
                                    pass

    
        #####################################################################
        #####################################################################
        ## print message for number of groups run
        if input.phase_to_run >= 3 and count % inc1 == 0:
                logging.info(f"-------------After {count*input.rup_per_group} groups: {np.round(time.time()-time_start,decimals=2)} secs-------------")
                time_start = time.time()
        
        #####################################################################
        #####################################################################
        if input.phase_to_run == 4 and flag_save_dv and model_assess._DV_dict is not None:
            if count % inc2 == 0 or count == len(list_rup_group):
                logging.info(f"========================================================")
                logging.info(f"\tSaving damages...")
                logging.info(f"\tExport directory:")
                logging.info(f"\t\t{dv_dir}")
                if input.flag_export_to_csv:
                    logging.info(f"\tFiles (Also exported to txt):")
                else:
                    logging.info(f"\tFiles:")
                for dv_i in model_assess._DV_dict.keys():
                    logging.info(f"\t\t----------")
                    count_proc = 0
                    for proc_i in model_assess._DV_dict[dv_i].keys():
                        if 'rr_pgd' in dv_i:
                            for i in model_assess._DV_dict[dv_i][proc_i]['source_param']:
                                if 'pgd' in i:
                                    edp_i = i
                            str_edp_i = edp_i[edp_i.find('_')+1:]+'_'
                        else:
                            str_edp_i = ''
                        str_proc_i = model_assess._DV_dict[dv_i][proc_i]['method'][0:model_assess._DV_dict[dv_i][proc_i]['method'].find('_')]
                        if model_assess._DV_dict[dv_i][proc_i]['eps_epistemic'] == 999:
                            str_eps = 'epiTotal_'
                        else:
                            str_eps = 'epi'+str(model_assess._DV_dict[dv_i][proc_i]['eps_epistemic'])+'_'
                            str_eps = str_eps.replace('.','o')
                            str_eps = str_eps.replace('-','m')
                        # str_range = 'rup_'+str(range_start)+'_'+str(range_end-1)
                        str_range = 'rup_all'
                        save_name_npz = dv_i+'_'+str_proc_i+'_'+str_edp_i+str_eps+str_range+'.npz'
                        save_path_npz = os.path.join(dv_dir,save_name_npz)
                        logging.info(f"\t\t{save_name_npz}")
                        sparse.save_npz(save_path_npz,np.transpose(model_assess._DV_dict[dv_i][proc_i]['output']))
                        ##
                        if input.flag_export_to_csv:
                            save_name_csv = save_name_npz[:save_name_npz.find('.npz')]+'.txt'
                            save_path_csv = os.path.join(dv_dir,save_name_csv)
                            np.savetxt(save_path_csv,np.transpose(model_assess._DV_dict[dv_i][proc_i]['output'].toarray()),fmt='%5.3e')
                        count_proc += 1

    ## 
    logging.info(f"========================================================")
    logging.info(f"...done with current run - program will now exit")

    
#####################################################################################################################
## Proc main
if __name__ == "__main__":

    ## Create parser for command prompt arguments
    parser = argparse.ArgumentParser(
                                    description='Open-source Seismic Risk Analysis (OpenSRA)'
                                    )
    
    ## Define arguments
    parser.add_argument('-l', '--logging', help='logging level [info(default), debug]', default='info')
    
    ## Parse command prompt input
    args = parser.parse_args()
    
    ## Run "Main"
    main(args.logging)