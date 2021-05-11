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
import os
import logging
import sys
import argparse
import json
import zipfile
import numpy as np
import pandas as pd

# Prior to importing OpenSRA backend, see if lib/OpenSHA folder exists
# and contains files, if not, unzip from lib/OpenSHA.zip
opensra_dir = os.path.dirname(os.path.realpath(__file__))
flag_to_unzip_opensha_zip = False
if os.path.isdir(os.path.join(opensra_dir,'lib','OpenSHA')):
    if len(os.listdir(os.path.join(opensra_dir,'lib','OpenSHA'))) == 0:
        flag_to_unzip_opensha_zip = True
else:
    flag_to_unzip_opensha_zip = True
if flag_to_unzip_opensha_zip:
    with zipfile.ZipFile(os.path.join(opensra_dir,'lib','OpenSHA.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(opensra_dir,'lib'))

# OpenSRA modules and functions
from src import Model, PreProcess, Fcn_Common


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
    
    # Initialize dictionary for storing model parameters
    method_assess_param = {}
    
    # -----------------------------------------------------------
    # Pre-process
    logging.info('---------------')
    logging.info('\t******** Preprocess ********')
    # Get directory of OpenSRA.py, in case if user is working in different directory and calling OpenSRA.py
    other_config_param['Dir_OpenSRA'] = os.path.dirname(os.path.realpath(__file__))
    # dir_opensra = setup_config['General']['Directory']['OpenSRA']
    os.chdir(other_config_param['Dir_OpenSRA'])
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # For memory efficiency, store and operate internal arrays and matrices in SciPy.sparse as much as possible, over NumPy.ndarray
    other_config_param['InternalMatrixType'] = 'sparse' # or 'ndarray'
    # Create additional configuration parameters; will be transferred to the GUI in future versions
    other_config_param['Flag_SampleWithStdDevTotal'] = False # True to sample from total sigma; False to sample intra- and inter-event sigmas separately then add together
    other_config_param['Flag_UseUniformStdDev'] = False # set value for uniform aleatory sigma; set to None to use GM predicted total sigma
    other_config_param['Flag_CombineAleatoryAndEpistemic'] = False # instead of using aleatory and epistemic stddevs, used total stddev
    other_config_param['Flag_UseSpecificAleatoryCases'] = True # instead of using aleatory and epistemic stddevs, used total stddev
    other_config_param['UniformStdDev'] = 0.65 # set value for uniform aleatory sigma; set to None to use GM predicted total sigma
    other_config_param['Num_Decimals'] = 3 # number of decimals in log10 space for export
    other_config_param['Flag_ResampleIM'] = False # sample IM even if samples already exist
    other_config_param['RupturePerGroup'] = 100 # number of ruptures to store per group
    other_config_param['Num_Threads'] = 1 # number of threads for regional_processor
    other_config_param['ListOfIMParams'] = ['Mean','InterEvStdDev','IntraEvStdDev','TotalStdDev']
    other_config_param['ApproxPeriod'] = {
        'PGA': 0.01,
        'PGV': 1
    }
    other_config_param['Flag_IMSamplesEventDependent'] = False
    other_config_param['Flag_DVEventDependent'] = True
    # Additional params for PEER groups (to be moved)
    other_config_param['PEERCategories'] = {
        'IntensityMeasure': 'IM',
        'EngineeringDemandParameter': 'EDP',
        'DamageMeasure': 'DM',
        'DecisionVariable': 'DV',
    }
    other_config_param['ID_EDP'] = {
        'Liquefaction': 'liq',
        'LateralSpread': 'ls',
        'GroundSettlement': 'gs',
        'Landslide': 'land',
        'SurfaceFaultRupture': 'surf',
        'SubsurfaceFaultRupture': 'subsurf',
        'TransientGroundStrain': 'tgs'
    }
    other_config_param['ID_DM'] = {
        'TransientPipeStrain': 'tps'
    }
    other_config_param['ID_DV'] = {
        'RepairRatePGV': 'rr_pgv',
        'RepairRatePGD': 'rr_pgd'
    }
    # Other site param
    other_config_param['ColumnsToKeepInFront'] = [
        'SITE_NUM', 'LAT_BEGIN', 'LONG_BEGIN', 'LAT_END',
        'LONG_END', 'LAT_MIDDLE', 'LONG_MIDDLE', 'SHAPE_LENGTH']
    other_config_param['ColumnsAppended'] = []
    other_config_param['Dir_Library'] = os.path.join(other_config_param['Dir_OpenSRA'],'lib')
    other_config_param['Dir_GeneralSiteData'] = os.path.join(other_config_param['Dir_Library'],'OtherData')
    other_config_param['SiteParams'] = {
        'Vs30': {
            'PathToRaster': os.path.join(
                other_config_param['Dir_GeneralSiteData'], 'Vs30_Wills_et_al_2015',
                'California_vs30_Wills15_hybrid_7p5c','California_vs30_Wills15_hybrid_7p5c.tif'),
            'ColumnNameToStoreAs': 'Vs30 (m/s)'
        },
        'Slope': {
            'PathToRaster': os.path.join(
                other_config_param['Dir_GeneralSiteData'],
                'Slope_Interpretted_From_CA_DEM', 'CA_Slope_Degrees_UTM_clip.tif'),
            'ColumnNameToStoreAs': 'Slope (deg)'
        }
    }
    # Zhu et al. (2017) model parameters
    other_config_param['Dir_ZhuEtal2017_param'] = os.path.join(
        other_config_param['Dir_GeneralSiteData'],'Zhu_etal_2017_Model_Inputs')
    other_config_param['ZhuEtal2017_param'] = {
        'dwWB': {
            'FolderName': 'CA_DistAnyWater_NoWB_WGS84_clip_km',
            'ColumnNameToStoreAs': 'Dist_Any Water (km)'
        },
        'dc': {
            'FolderName': 'CA_DistCoast_WGS84_clip_km',
            'ColumnNameToStoreAs': 'Dist_Coast (km)'
        },
        'dr': {
            'FolderName': 'CA_DistRivers_WGS84_clip_km',
            'ColumnNameToStoreAs': 'Dist_River (km)'
        },
        'wtd': {
            'FolderName': 'CA_ModeledWTD_1km_WGS84_merge_clip_m',
            'ColumnNameToStoreAs': 'Water Table Depth (m)'
        },
        'precip': {
            'FolderName': 'CA_Precip_1981-2010_WGS84_clip_mm',
            'ColumnNameToStoreAs': 'CA_Precip (mm)'
        }
    }
    # Params for landslide
    # shapefile with geologic units
    other_config_param['Shp_GeologicUnit'] = os.path.join(
        other_config_param['Dir_GeneralSiteData'],
        'CGS_CA_Geologic_Map_2010','shapefiles','GMC_geo_poly.shp')
    # Slate's temporary file with properties for geologic units
    other_config_param['File_GeologicUnitParam'] = os.path.join(
        other_config_param['Dir_Library'], 
        'Slate', 'Seismic_Hazard_CGS_Unit Strengths.csv')
    other_config_param['ColumnsToUseForKy'] = {
        'friction_angle': 'Friction Angle (deg)',
        'cohesion': 'Cohesion (kPa)',
        'thickness': 'Thickness (m)',
        'unit_weight': 'Unit Weight (kN/m3)',
        }
    # Probability distribution types
    other_config_param['DistributionType'] = {
        'Uniform': {
            "ListOfHazard": [
                'Liquefaction', 'LateralSpread', 'GroundSettlement',
                'RepairRatePGD'
            ],
            'Aleatory': {
                'Cases': [0.125, 0.375, 0.625, 0.875],
                'Weights': [0.25, 0.25, 0.25, 0.25]
            },
            'Epistemic': {
                'Branch': [-1, 0, 1],
                'Weight': [1/3]*3
            }
        }, 
        'Lognormal': {
            "ListOfHazard": [
                'Landslide', 'SurfaceFaultRupture', 'SubsurfaceFaultRupture',
                'TransientGroundStrain', 'TransientPipeStrain', 'RepairRatePGV'
            ],
            'Aleatory': {
                'Cases': [-1.65, 0, 1.65],
                'Weights': [0.2, 0.6, 0.2]
            },
            'Epistemic': {
                'Branch': [-1.65, 0, 1.65],
                'Weight': [1/3]*3
            }
        }
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
    # Get working directory and see if user wants previous run cleaned
    other_config_param['Dir_Working'] = setup_config['General']['Directory']['Working']
    if clean_prev_run.lower() in 'yes':
        for root, dirs, files in os.walk(other_config_param['Dir_Working']):
            if 'Input' in root:
                pass
            else:
                for file in files:
                    os.remove(os.path.join(root,file))
        logging.info('Cleaned outputs previous run')
    
    # Create other variables from and append to setup_config
    other_config_param, folders_to_create, method_assess_param = PreProcess.get_other_setup_config(
        setup_config, other_config_param, folders_to_create, method_assess_param
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
    # Limit number of sites to 5000 if including spatial correlation
    if setup_config['IntensityMeasure']['Correlation']['Spatial']['ToInclude']:
        site_data = site_data.loc[:5000,:]
    other_config_param['Num_Sites'] = len(site_data)
    
    # Update site_data 
    site_data = PreProcess.update_site_data_file(site_data, setup_config, other_config_param)
    
    # Append site_data to method parameters
    method_assess_param = PreProcess.add_site_data_to_method_param(setup_config, site_data, other_config_param, method_assess_param)
    
    # Set up other parameters for method assessment
    method_assess_param = PreProcess.setup_other_method_param(other_config_param, method_assess_param)
    
    # Export site locations and vs30 to text files for OpenSHA interface
    PreProcess.export_site_loc_and_vs30(site_data, other_config_param)
    
    logging.info('\t******** Preprocess ********')
    logging.info('---------------\n\n')
    
    
    # -----------------------------------------------------------
    # Start workflow
    # 1. Get events from seismic source
    # 2. Get ground motion predictions at sites given events
    # 3. Generate intensity measure realizations
    # 4. Assess EDPs, DMs, DVs using IM realizations
    # 5. Export DVs
    # -----------------------------------------------------------
    
    logging.info('\t******** OpenSRA Analysis ********')
    
    # Create assessment class object
    model = Model.assessment()
    logging.info(f'Created "Model.assessment" class object named "model\n')
    
    # -----------------------------------------------------------
    # Get rupture scenarios and fault crossings given site
    logging.info('---------------')
    logging.info('\t------------- Get Means and StdDevs for Intensity Measures -------------')
    model.get_IM_means(setup_config, other_config_param, site_data)
    other_config_param['Num_Events'] = len(model._EVENT_dict['Scenarios']['mag'])
    
    # -----------------------------------------------------------
    # Generate realizations of IMs (i.e., sampling)
    logging.info('---------------')
    logging.info('\t------------- Simulate Intensity Measure Realizations -------------')
    model.sim_IM(setup_config, other_config_param, site_data)
    
    # -----------------------------------------------------------
    # Assess EDPs
    logging.info('---------------')
    logging.info('\t------------- Assess Engineering Demand Parameters -------------')
    model.assess_EDP(setup_config, other_config_param, site_data, method_assess_param)
    
    # -----------------------------------------------------------
    # Assess DMs
    logging.info('---------------')
    logging.info('\t------------- Assess Damage Measures -------------')
    model.assess_DM(setup_config, other_config_param, site_data, method_assess_param)
    
    # -----------------------------------------------------------
    # Assess DVs
    logging.info('---------------')
    logging.info('\t------------- Assess Decision Variables -------------')
    model.assess_DV(setup_config, other_config_param, site_data, method_assess_param)
    
    # -----------------------------------------------------------
    # Export
    logging.info('---------------')
    logging.info('\t------------- Export Results -------------')
    model.export_DV(setup_config, other_config_param, site_data)
    
    # -----------------------------------------------------------
    # Exit Program
    logging.info("\t------------- OpenSRA Analysis -------------")
    logging.info('---------------')
    

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