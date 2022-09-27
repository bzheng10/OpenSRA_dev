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
# Python modules
import importlib
import copy
import sys
import os
import json
import argparse
import warnings

# data manipulation modules
import numpy as np
import pandas as pd
# suppress warning that may come up
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# append OpenSRA base dir to path
# sys.path.append('..')

# OpenSRA modules
from src.site import geodata
from src.im import haz
from src.site.get_pipe_crossing import get_pipe_crossing
from src.site.get_well_crossing import *


def main(work_dir):
    """main function that runs the preprocess procedures"""
    
    # start of preprocess
    print('Start of preprocessing for OpenSRA')
    counter = 1 # counter for stages of processing   
        
    # make directories
    # check current directory, if not at OpenSRA level, go up a level (happens during testing)
    if not os.path.basename(os.getcwd()) == 'OpenSRA' and not os.path.basename(os.getcwd()) == 'OpenSRABackEnd':
        os.chdir('..')
        print(work_dir)
    input_dir = os.path.join(work_dir,'Input')
    processed_input_dir = os.path.join(work_dir,'Processed_Input')
    if not os.path.isdir(processed_input_dir):
        os.mkdir(processed_input_dir)
    im_dir = os.path.join(work_dir,'IM')
    if not os.path.isdir(im_dir):
        os.mkdir(im_dir)
    print(f'{counter}. Check and create file directories')
    print('\tPerforming preprocessing of methods and input variables for OpenSRA')
    print('\t\tWorking directory given:')
    print(f'\t\t\t- {work_dir}')
    print('\t\tInput directory implied:')
    print(f'\t\t\t- {input_dir}')
    print('\t\tProcessed input directory for export of processed information:')
    print(f'\t\t\t- {processed_input_dir}')
    print('\t\tInensity measure directory:')
    print(f'\t\t\t- {im_dir}')
    counter += 1
    
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
    if infra_ftype == 'Shapefile':
        infra_fpath = os.path.join(work_dir,setup_config['Infrastructure']['SiteDataFile'])
        files = os.listdir(infra_fpath)
        for each in files:
            if each.endswith('shp'):
                infra_fpath = os.path.join(infra_fpath,each)
                break
    else:
        infra_fpath = os.path.join(work_dir,setup_config['Infrastructure']['SiteDataFile'])
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
    # Intensity Measure
    im_source = list(setup_config['IntensityMeasure']['SourceForIM'])[0]
    if im_source == 'ShakeMap':
        sm_dir = os.path.join(work_dir,setup_config['IntensityMeasure']['SourceForIM']['ShakeMap']['Directory'])
        sm_events = setup_config['IntensityMeasure']['SourceForIM']['ShakeMap']['Events']
    else:
        raise NotImplementedError("To be added into preprocess...")
    print(f'{counter}. Processed setup configuration file')
    counter += 1
    
    # preprocess infrastructure file
    preprocess_infra_file(infra_type, infra_fpath, infra_loc_header, processed_input_dir, l_max=0.1)
    print(f'{counter}. Processed infrastructure file and exported site data table to directoy:')
    print(f'\t{processed_input_dir}')
    counter += 1
    
    # get workflow for PC
    workflow, workflow_fpath = make_workflow(setup_config, processed_input_dir, to_export=True)
    print(f'{counter}. Created workflow and exported to:')
    print(f'\t{workflow_fpath}')
    print(f'{json.dumps(workflow, indent=4)}')
    counter += 1
    
    # read input tables for random, fixed variables, and site data
    rvs_input, fixed_input, site_data = read_input_tables(input_dir, processed_input_dir)
    print(f'{counter}. Read input tables for random, fixed variables, and infrastructure data in input directory')
    counter += 1
    
    ##--------------------------
    # get crossings - may move to another location in Preprocess
    avail_data_summary = None # initialize
    
    if infra_type == 'below_ground':
        # load json with available datasets
        avail_data_summary_fpath = os.path.join('lib','AvailableDataset.json')
        with open(avail_data_summary_fpath,'r') as f:
            avail_data_summary = json.load(f)
        # landslide
        cat = 'EDP'
        haz = 'landslide'
        if cat in workflow and haz in workflow[cat]:
            file_key = 'ca_landslide_inventory'
            fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
            # site_data_crossing = get_pipe_crossing(
            site_data = get_pipe_crossing(
                path_to_def_shp=fpath,
                infra_site_data=site_data.copy(),
                export_dir=processed_input_dir,
                def_type=haz
            )
            print(f'{counter}. Obtained pipeline crossing for landslide')
            counter += 1
            
    elif infra_type == 'wells_caprocks':
        # load json with available datasets
        avail_data_summary_fpath = os.path.join('lib','AvailableDataset.json')
        with open(avail_data_summary_fpath,'r') as f:
            avail_data_summary = json.load(f)
        # landslide
        cat = 'EDP'
        haz = 'landslide'
        if cat in workflow and haz in workflow[cat]:
            file_key = 'ca_landslide_inventory'
            fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
            # site_data_crossing = get_pipe_crossing(
            site_data = get_pipe_crossing(
                path_to_def_shp=fpath,
                infra_site_data=site_data.copy(),
                export_dir=processed_input_dir,
                def_type=haz
            )
            print(f'{counter}. Obtained pipeline crossing for landslide')
            counter += 1
        
    ##--------------------------
    
    # rvs and fixed params split by preferred andf user provided
    pref_rvs, user_prov_table_rvs, user_prov_gis_rvs, \
    pref_fixed, user_prov_table_fixed, user_prov_gis_fixed = \
        separate_params_by_source(rvs_input, fixed_input)
    print(f'{counter}. Separate random and fixed parameters by source')
    counter += 1
    
    # get preferred input distributions
    pref_param_dist, pref_param_dist_const_with_level, pref_param_fixed = \
        import_param_dist_table(infra_type=infra_type)
    print(f'{counter}. Read preferred distributions for variables')
    print(f"\t{os.path.join('param_dist',f'{infra_type}.xlsx')}")
    counter += 1
    
    # get param_dist_meta from user-provided information
    if 'UserGISFile' in setup_config['General']['Directory']:
        user_prov_gis_fdir = setup_config['General']['Directory']['UserGISFile']
    else:
        user_prov_gis_fdir = ''
    param_dist_meta, param_dist_table = get_param_dist_from_user_prov_table(
        user_prov_table_rvs, user_prov_table_fixed,
        user_prov_gis_rvs, user_prov_gis_fixed,
        pref_rvs, pref_fixed, site_data, user_prov_gis_fdir
    )
    print(f'{counter}. Get user provided distributions from infrastructure table')
    counter += 1
    
    # get params with missing distribution metrics
    params_with_missing_dist_metric = get_params_with_missing_dist_metric(param_dist_meta)
    print(f'{counter}. Track parameters still with missing distribution metrics')
    counter += 1

    # get level to run
    if "EDP" in workflow and "Liquefaction" in workflow["EDP"] and "CPTBased" in workflow["EDP"]["Liquefaction"]:
        param_dist_table['level_to_run'] = np.ones(param_dist_table.shape[0])*3
    else:
        param_dist_table = get_level_to_run(
            param_dist_table,
            workflow,
            pref_param_dist,
            pref_param_dist_const_with_level,
            params_with_missing_dist_metric,
            param_dist_meta,
            infra_type=infra_type
        )
    print(f'{counter}. Determine level of analysis to run for each site')
    counter += 1
    
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
        export_path_dist_table=os.path.join(processed_input_dir,'param_dist.csv'),
        export_path_dist_json=os.path.join(processed_input_dir,'param_dist_meta.json'),
        infra_type=infra_type
    )
    print(f'{counter}. Get missing distribution metrics from preferred distributions')
    counter += 1
    
    # get IM predictions
    get_im_pred(im_source, sm_dir, sm_events, im_dir, site_data, infra_loc_header)
    print(f'{counter}. Obtained IM predictions from {im_source} and stored to:')
    print(f"\t{im_dir}")
    counter += 1
    
    # end of preprocess
    print('... End of preprocessing for OpenSRA')


def get_im_pred(im_source, sm_dir, sm_events, im_dir, site_data, infra_loc_header):
    """get IM predictions from backend"""
    seismic_hazard = getattr(haz, 'SeismicHazard')()
    if im_source == "ShakeMap":
        # print(f'\tRunning ShakeMaps scenarios:')
        # for each in sm_events:
        #     print(f'\t\t- {each}')
        seismic_hazard.init_ssc('ShakeMap',sm_dir=sm_dir,event_names=sm_events)
        seismic_hazard.init_gmpe()
        seismic_hazard.process_rupture()
        if 'LON_MID' in site_data:
            seismic_hazard.set_site_data(
                # lon=site_data[infra_loc_header['lon_header']],
                # lat=site_data[infra_loc_header['lat_header']],
                lon=site_data.LON_MID,
                lat=site_data.LAT_MID,
                vs30=np.zeros(site_data.shape[0])
            )
        elif 'LON' in site_data:
            seismic_hazard.set_site_data(
                # lon=site_data[infra_loc_header['lon_header']],
                # lat=site_data[infra_loc_header['lat_header']],
                lon=site_data.LON,
                lat=site_data.LAT,
                vs30=np.zeros(site_data.shape[0])
            )
        seismic_hazard.get_gm_pred_from_gmc()
        seismic_hazard.export_gm_pred(
            sdir=im_dir,
            stype=['sparse','csv']
        )
    else:
        raise NotImplementedError("to be added to preprocessing...")


def preprocess_infra_file(infra_type, infra_fpath, infra_loc_header, processed_input_dir, l_max=0.1):
    """process infrastructure files"""
    # load infrastructure file
    if infra_type == 'below_ground':
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
                    rvs_by_level, fix_by_level = _inst.get_req_rv_and_fix_params(infra_fixed)
                    # gather required model parameters for each level
                    for i in range(3):
                        # initialize list
                        if not f'level{i+1}' in req_rvs_by_level:
                            req_rvs_by_level[f'level{i+1}'] = []
                            req_fixed_by_level[f'level{i+1}'] = []
                        # if method varies with level
                        if _inst.input_dist_vary_with_level:
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


def import_param_dist_table(infra_type='below_ground'):
    """loads table with param distributions, choose from 'below_ground', 'above_ground', and 'wells_caprocks'"""
    n_levels = 3
    pref_param_dist_path = os.path.join('param_dist',f'{infra_type}.xlsx')
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


def read_input_tables(input_dir,processed_input_dir):
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
    return rvs_input, fixed_input, site_data


def separate_params_by_source(rvs_input, fixed_input):
    """separate preferred vs user provided callouts in input tables"""
    pref_rvs = rvs_input[rvs_input.Source=='Preferred'].reset_index(drop=True).copy()
    user_prov_table_rvs = rvs_input[rvs_input.Source=='From infrastructure table or enter value'].reset_index(drop=True).copy()
    user_prov_gis_rvs = rvs_input[rvs_input.Source=='From user-provided GIS maps'].reset_index(drop=True).copy()
    pref_fixed = fixed_input[fixed_input.Source=='Preferred'].reset_index(drop=True).copy()
    user_prov_table_fixed = fixed_input[fixed_input.Source=='From infrastructure table or enter value'].reset_index(drop=True).copy()
    user_prov_gis_fixed = fixed_input[fixed_input.Source=='From user-provided GIS maps'].reset_index(drop=True).copy()
    return pref_rvs, user_prov_table_rvs, user_prov_gis_rvs, pref_fixed, user_prov_table_fixed, user_prov_gis_fixed


def get_param_dist_from_user_prov_gis(
    # user_prov_gis_map_dir,
    # user_prov_gis_rvs,
    # user_prov_gis_fixed,
    # param_dist_meta,
    # site_data
):
    """gets inputs for parameters flagged as 'From user-provided GIS maps'"""
    print(NotImplementedError("to be implemented"))
    return param_dist_meta, site_data


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
            print('Cannot locate user provided GIS file: file must end with ".tif" or ".shp"')
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
            if user_prov_table_fixed.loc[i,'Value'].upper() in site_data:
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
            print('Cannot locate user provided GIS file: file must end with ".tif" or ".shp"')
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


def get_level_to_run(
    param_dist_table,
    workflow,
    pref_param_dist,
    pref_param_dist_const_with_level,
    params_with_missing_dist_metric,
    param_dist_meta,
    infra_type='below_ground'
):
    """get preferred distribution metrics from internal tables"""
    
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
    export_path_dist_table=None,
    export_path_dist_json=None,
    infra_type='below_ground',
    default_statewide_geo_map='wills'
):
    """get rest of the missing distribution metrics"""
    
    met_list = ['dist_type','mean','sigma','low','high']
    crossing_params = ['l_anchor','beta_crossing','psi_dip','theta_slip']
    
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
        soil_prop_map = {}
        if 'EDP' in workflow and 'landslide' in workflow['EDP']:
            if ('phi_soil' in param_dist_meta and param_dist_meta['phi_soil']['source'] == 'Preferred') or \
               ('coh_soil' in param_dist_meta and param_dist_meta['coh_soil']['source'] == 'Preferred'):
                if default_statewide_geo_map == 'wills':
                    file_key = 'level1_geo_unit_wills15'
                    store_name = avail_data_summary['Parameters'][file_key]['ColumnNameToStoreAs']
                    geo_unit_fpath = avail_data_summary['Parameters'][file_key]['Datasets']['Set1']['Path']
                    locs.data = locs.sample_shapefile(
                        table=locs.data,
                        fpath=geo_unit_fpath,
                        attr='Geologic_U',
                        store_name=store_name
                    )
                    param_dist_table[store_name] = locs.data[store_name].values
                
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
            print(f'\tRead Witter et al. (2006) geologic units')
            
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
            print(f'\tRead Bedrossian et al. (2012) geologic units')
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
                                print(param, pref_val)
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
                            if pref_val == 'depends' and param in crossing_params:
                                param_dist_table.loc[rows_nan,f'{param}_mean'] = site_data[param][rows_nan].values
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
        param_dist_table.to_hdf(export_path_dist_table.replace('.csv','.h5'),key='table')
        # also export to hdf5 for reading
        param_dist_table.to_csv(export_path_dist_table.replace('.csv','.txt'),index=False,sep='\t')
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
    
    # Parse command line input
    args = parser.parse_args()
    
    # Run "Main"
    main(
        work_dir = args.work_dir,
        # infra_type = args.infra_type,
        # infra_fpath = args.file_type
    )