# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Performs pre-processing tasks
#
# Created: December 10, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import os
import json
import logging
import numpy as np
from src.site.GetGeologicUnit import get_geologic_unit


# -----------------------------------------------------------
def get_other_setup_config(setup_config, other_config_param, folders_to_create):
    """
    Creates new and updates existing dictionary of additional setup parameters
    
    """
    
    # Get important directories
    dir_working = other_config_param['Dir_Working']
    dir_opensra = other_config_param['Dir_OpenSRA']
    dir_input = other_config_param['Dir_Input']

    # Set full path for site output folder
    other_config_param['Dir_Site'] = os.path.join(setup_config['General']['Directory']['Working'], 'Site') 
    folders_to_create.append(other_config_param['Dir_Site'])
    
    # Create full file paths for vairous files by appending the appropriate directory to the file name
    other_config_param['Path_SiteData'] = os.path.join(dir_input, setup_config['General']['SiteDataFile']) # create full path for site data file
    other_config_param['Path_SiteData_Updated'] = os.path.join(
        other_config_param['Dir_Site'], 
        os.path.basename(other_config_param['Path_SiteData']).replace('.csv','_Updated.csv')
    ) # create full path for updated site data file
    other_config_param['Dir_IM'] = os.path.join(dir_working, 'IM') # set full path for GM output folder
    folders_to_create.append(other_config_param['Dir_IM'])
    
    # Intensity measures
    # Create additional parameters for use of source model
    other_config_param['Dir_IM_SeismicSource'] = os.path.join(other_config_param['Dir_IM'], 'SeismicSource') # set full path for GM output folder
    folders_to_create.append(other_config_param['Dir_IM_SeismicSource'])
    other_config_param['Dir_IM_GroundMotion'] = os.path.join(other_config_param['Dir_IM'], 'GroundMotion') # set full path for GM prediction output folder
    folders_to_create.append(other_config_param['Dir_IM_GroundMotion'])
    other_config_param['Dir_IM_GroundMotion_Prediction'] = os.path.join(other_config_param['Dir_IM_GroundMotion'], 'Prediction') # set full path for GM prediction output folder
    folders_to_create.append(other_config_param['Dir_IM_GroundMotion_Prediction'])
    other_config_param['Dir_IM_GroundMotion_Simulation'] = os.path.join(other_config_param['Dir_IM_GroundMotion'], 'Simulation') # set full path for GM prediction output folder
    folders_to_create.append(other_config_param['Dir_IM_GroundMotion_Simulation'])
    
    # Base name for file with rupture metadata
    other_config_param['File_ListOfScenarios'] = os.path.join(other_config_param['Dir_IM_SeismicSource'],'ListOfScenarios')
    # Add extension
    other_config_param['File_ListOfScenarios'] = other_config_param['File_ListOfScenarios'] + '.txt'
    
    # Check the source for intensity measures
    if setup_config['IntensityMeasure']['SourceForIM'] == 'OpenSHA':
        # Create additional parameters for use of source model
        src_model = setup_config['IntensityMeasure']['SourceParameters']['SeismicSourceModel']
        other_config_param['Path_RuptureSegment'] = os.path.join(dir_opensra,'lib','simcenter','opensha','erf',src_model,'rupture_segments.json')
        other_config_param['Path_PointSource'] = os.path.join(dir_opensra,'lib','simcenter','opensha','erf',src_model,'point_sources.txt')
        
        # Check for filters to use
        other_config_param['Flag_IncludeFilter_TrMax'] = setup_config['IntensityMeasure']['SourceParameters']['Filter']['ReturnPeriod']['ToInclude']
        other_config_param['Flag_IncludeFilter_RMax'] = setup_config['IntensityMeasure']['SourceParameters']['Filter']['Distance']['ToInclude']
        other_config_param['Flag_IncludeFilter_PtSrc'] = setup_config['IntensityMeasure']['SourceParameters']['Filter']['PointSource']['ToInclude']
        
        # Base name for file with rupture metadata
        other_config_param['File_ListOfScenarios_Full'] = os.path.join(other_config_param['Dir_IM_SeismicSource'],'ListOfScenarios_Full')
        # other_config_param['File_RuptureGroups'] = os.path.join(other_config_param['Dir_IM_SeismicSource'],'ListOfRuptureGroups')
        
        # Append additional identifiers to File_RuptureMetadata_filter based on filters to use
        # if other_config_param['Flag_IncludeFilter_TrMax']:
            # str_to_append = '_' + str(setup_config['IntensityMeasure']['SourceParameters']['Filter']['ReturnPeriod']['Maximum']) + 'yr' # append string with maximum return period
            # other_config_param['File_RuptureMetadata'] = other_config_param['File_RuptureMetadata'] + str_to_append
            # other_config_param['File_RuptureGroups'] = other_config_param['File_RuptureGroups'] + str_to_append
        # if other_config_param['Flag_IncludeFilter_RMax']:
            # str_to_append = '_' + str(setup_config['IntensityMeasure']['SourceParameters']['Filter']['Distance']['Maximum']) + 'km' # append string with maximum distance
            # other_config_param['File_RuptureMetadata'] = other_config_param['File_RuptureMetadata'] + str_to_append
            # other_config_param['File_RuptureGroups'] = other_config_param['File_RuptureGroups'] + str_to_append
        # if other_config_param['Flag_IncludeFilter_PtSrc']:
            # str_to_append = '_' + 'wPtSrc' # append string for point sources
            # other_config_param['File_RuptureMetadata'] = other_config_param['File_RuptureMetadata'] + str_to_append
            # other_config_param['File_RuptureGroups'] = other_config_param['File_RuptureGroups'] + str_to_append
        other_config_param['File_ListOfScenarios_Filtered'] = os.path.join(other_config_param['Dir_IM'],'ListOfScenarios_Filtered')
        
        # Add extension
        other_config_param['File_ListOfScenarios_Filtered'] = other_config_param['File_ListOfScenarios_Filtered'] + '.txt'
        other_config_param['File_ListOfScenarios_Full'] = other_config_param['File_ListOfScenarios_Full'] + '.txt'
        # other_config_param['File_RuptureGroups'] = other_config_param['File_RuptureGroups'] + '.txt'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Section reserved for ShakeMap
    elif setup_config['IntensityMeasure']['SourceForIM'] == 'ShakeMap':
        other_config_param['Dir_ShakeMap'] = setup_config['IntensityMeasure']['SourceParameters']['Directory']
        try:
            list_events = setup_config['IntensityMeasure']['SourceParameters']['Events']
            other_config_param['ShakeMapEvents'] = []
            for event_i in list_events:
                if event_i in os.listdir(other_config_param['Dir_ShakeMap']):
                    other_config_param['ShakeMapEvents'].append(event_i)
        except:
            other_config_param['ShakeMapEvents'] = os.listdir(other_config_param['Dir_ShakeMap'])
        other_config_param['Num_Events'] = len(other_config_param['ShakeMapEvents'])
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Get list of target intensity measure Types
    other_config_param['IM'] = [] 
    for key in setup_config['IntensityMeasure']['Type'].keys():
       if setup_config['IntensityMeasure']['Type'][key]['ToAssess']:
            other_config_param['IM'].append(key)
    other_config_param['Num_IM'] = len(other_config_param['IM'])
            
    # Check if IM samples are needed based on DVs
    other_config_param['Flag_GetIM'] = False # default
    if setup_config['DecisionVariable']['Type']['RepairRatePGV']['ToAssess']:
        other_config_param['Flag_GetIM'] = True
    if setup_config['DecisionVariable']['Type']['RepairRatePGD']['ToAssess']:
        if setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['ToAssess'] or \
            setup_config['EngineeringDemandParameter']['Type']['Landslide']['ToAssess']  or \
            setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['ToAssess']:
                other_config_param['Flag_GetIM'] = True
                
    # If IM samples are not needed, then clear list of IM
    if other_config_param['Flag_GetIM'] is False:
        other_config_param['IM'] = [] 
    # Clear list of EDP required if only RepairRatePGV is to be assessed
    if setup_config['DecisionVariable']['Type']['RepairRatePGV']['ToAssess'] and \
        not setup_config['DecisionVariable']['Type']['RepairRatePGD']['ToAssess']:
            for key in setup_config['EngineeringDemandParameter']['Type'].keys():
                setup_config['EngineeringDemandParameter']['Type'][key]['ToAssess'] = False
            other_config_param['IM'] = ['PGV'] # set IM to only PGV
    
    # Engineering demand parameters
    # Note: OpenSRA currently is not set up to store EDP results
    # Check if probability of liquefaction and liquefaction susceptibility are needed
    other_config_param['Flag_LiqSusc'] = False # default
    other_config_param['Flag_PLiq'] = False # default
    if setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['ToAssess']:
        other_config_param['Flag_LiqSusc'] = True
    if setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['ToAssess']:
        other_config_param['Flag_PLiq'] = True
    
    # Check if probability of landslide is needed
    other_config_param['Flag_PLand'] = False # default
    if setup_config['EngineeringDemandParameter']['Type']['Landslide']['ToAssess'] & \
        setup_config['DecisionVariable']['Type']['RepairRatePGD']['ToAssess']:
        other_config_param['Flag_PLand'] = True
        
    # Damage measures
    # Note: OpenSRA currently is not set up to store DM results
    
    # Decision variables
    
    # Create sub-directories within the PEER categories based on IMs, hazards, and loss variables to assess
    peer_categories = {
        'IntensityMeasure': 'IM',
        'EngineeringDemandParameter': 'EDP',
        'DamageMeasure': 'DM',
        'DecisionVariable': 'DV',
    }
    # Loop through PEER categories
    for category in peer_categories.keys():
        if category == 'IntensityMeasure':
            # Strings for names
            base_param_name = [
                'Dir_' + peer_categories[category] + '_GroundMotion_Prediction',
                'Dir_' + peer_categories[category] + '_GroundMotion_Simulation'
            ]
        else:
            # Strings for names
            base_param_name = ['Dir_' + peer_categories[category]]
            # Set full path for DV output folder
            other_config_param['Dir_' + peer_categories[category]] = os.path.join(dir_working, peer_categories[category])
            folders_to_create.append(other_config_param['Dir_' + peer_categories[category]])
        # Loop through IMs, hazards, and loss variables
        for key in setup_config[category]['Type'].keys():
            if setup_config[category]['Type'][key]['ToAssess']:
                for name in base_param_name:
                    # Set full path for GM prediction output folder
                    other_config_param[name + '_' + key] = os.path.join(other_config_param[name], key)
                    folders_to_create.append(other_config_param[name + '_' + key])
    
    # Append OpenSRA path to the user-defined relative paths
    other_config_param['Shp_GeologicUnit'] = os.path.join(dir_opensra,other_config_param['Shp_GeologicUnit']) # location of geologic unit shapefile
    other_config_param['File_GeologicUnitParam'] = os.path.join(dir_opensra,other_config_param['File_GeologicUnitParam']) # location of Slate's parameters for geologic units
    
    # Create directories for storing fault traces and intersections if fault ruptures are to be assessed
    if setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['ToAssess'] or \
        setup_config['EngineeringDemandParameter']['Type']['subSurfaceFaultRupture']['ToAssess']:
        other_config_param['Dir_FaultCrossing'] = os.path.join(other_config_param['Dir_IM'],'FaultSource')
        other_config_param['Dir_FaultTrace'] = os.path.join(other_config_param['Dir_FaultCrossing'],'Trace')
        other_config_param['Dir_Intersection'] = os.path.join(other_config_param['Dir_FaultCrossing'],'Intersection')
        folders_to_create.append(other_config_param['Dir_FaultCrossing'])
        folders_to_create.append(other_config_param['Dir_FaultTrace'])
        folders_to_create.append(other_config_param['Dir_Intersection'])
    
    # Export other_config_param dictionary to JSON for checks
    other_config_param = {key:other_config_param[key] for key in sorted(other_config_param)} # sort for export
    with open(os.path.join(dir_working,'CheckPreProcess.json'), 'w') as outfile:
        json.dump(other_config_param, outfile, indent=4, separators=(',', ': '))
    
    # Eeturn updated dictionary of other_config_param
    return other_config_param, folders_to_create
    
    
# -----------------------------------------------------------
def update_site_data_file(site_data, setup_config, other_config_param):
    """
    Updates existing site data file with additional information necessary for run
    
    """
    
    #
    logging.info(f"Updating site data file...")
    # logging.info(f"\t{other_config_param['Path_SiteData']}")
    
    # Flag for whether the site data matrix is updated, default = False
    flag_update_site_data_file = False # default
    
    # Check if segment lengths are given in the file; if not then assume lengths of 1 km and add to "site_data"
    if not 'l_seg (km)' in site_data.columns:
        site_data['l_seg (km)'] = [1]*len(site_data)
        flag_update_site_data_file = True # updated site data
        
    # Check if start and end locations are given in the file; if not set equal to site locations (i.e., points)
    if not 'Lon_start' in site_data.columns:
        site_data['Lon_start'] = site_data['Longitude'].values
        site_data['Lat_start'] = site_data['Latitude'].values
        site_data['Lon_end'] = site_data['Longitude'].values
        site_data['Lat_end'] = site_data['Latitude'].values
        flag_update_site_data_file = True # updated site data
    
    # Check if need to extract geologic units and properties, get the params if not
    # Only if landslide is required for assessment
    if setup_config['EngineeringDemandParameter']['Type']['Landslide']['ToAssess']:
        if not 'Geologic Unit' in site_data.columns:
            # run script to get the properties
            site_data = get_geologic_unit(
                site_data, 
                other_config_param['Shp_GeologicUnit'],
                other_config_param['File_GeologicUnitParam'],
                other_config_param['ColumnsToUseForKy'],
                setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters']['SourceParametersForKy'],
            )
            flag_update_site_data_file = True # updated site data
    
    # Check flag to see if site data matrix is updated, if so then save to site file
    if flag_update_site_data_file:
        site_data.to_csv(other_config_param['Path_SiteData_Updated'],index=False)
        logging.info(f"... Site data file updated to:")
        logging.info(f"\t{other_config_param['Path_SiteData_Updated']}")
    else:
        logging.info(f"\tSite data file is up-to-date")
    
    # Return site_data dataframe and flag of whether site_data has been updated
    return site_data

    
# -----------------------------------------------------------
def add_site_data_to_method_param(setup_config, site_data, other_config_param):
    """
    Add parameters in site_data to method parameters in setup_config. Currently method dependent
    
    """
    
    #
    logging.info(f"Adding site data to list of model parameters in setup_config...")
    
    # Define values to substitute if parameters are not included in site_data
    sub_value = [np.nan]*len(site_data)
    find_val = -9999
    set_val = np.nan
    
    # Initalize dictionary that tracks which methods have been updated
    AppendMethod = {
        'EngineeringDemandParameter': {},
        'DamageMeasure': {},
        'DecisionVariable': {}
    }
    
    # For specific PEER categories, geohazards, and methods:
    
    # Liquefaction:
    # if Flag_LiqSusc or Flag_PLiq:
    if setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['ToAssess']:
    
        # For Zhu et al. (2017)
        if 'zhu_etal_2017' in setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['ListOfMethods']:
            # Get values from site_data, set to list of NaN if doesn't exist
            wtd = np.asarray(site_data.get('WTD_300m (m)', sub_value))
            dr = np.asarray(site_data.get('Dist_River (km)', sub_value))
            dc = np.asarray(site_data.get('Dist_Coast (km)', sub_value))
            dw = np.asarray(site_data.get('Dist_Any Water (km)', sub_value))
            precip = np.asarray(site_data.get('CA_Precip (mm)', sub_value))
            vs30 = np.asarray(site_data.get('VS30 (m/s)', sub_value))
            # Find elements = find_val and set to set_val
            wtd = set_elements_to_nan(wtd, find_val, set_val)
            dr = set_elements_to_nan(dr, find_val, set_val)
            dc = set_elements_to_nan(dc, find_val, set_val)
            dw = set_elements_to_nan(dw, find_val, set_val)
            precip = set_elements_to_nan(precip, find_val, set_val)
            vs30 = set_elements_to_nan(vs30, find_val, set_val)
            
            # Temporary values for elevations
            z = np.asarray(site_data.get('Elev (m)', [10]*len(site_data))) # set to 10 for now, get data from DEM map later
            # Update setup_config with parameters
            setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['OtherParameters'].update({
                'wtd':wtd, 'dr':dr, 'dc':dc, 'dw':dw, 'precip':precip, 'vs30':vs30, 'z':z,
            })

    # Lateral Spread
    if setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['ToAssess']:
    
        # For Hazus (2014)
        if 'hazus_2014' in setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['ListOfMethods']:
            # Get parameters from site_data
            dwWB = np.asarray(site_data.get('Dist_Any Water (km)', sub_value))
            dwWB = set_elements_to_nan(dwWB, find_val, set_val)
            dwWB = dwWB*1000 # convert to meters
            # Update setup_config with parameters
            setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['OtherParameters'].update({
                'dwWB':dwWB,
            })

    # Landslide
    if setup_config['EngineeringDemandParameter']['Type']['Landslide']['ToAssess']:
    
        # For Bray & Macedo (2019)
        if 'bray_and_macedo_2019' in setup_config['EngineeringDemandParameter']['Type']['Landslide']['ListOfMethods']:
            # Update setup_config with parameters
            setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters'].update({
                'ky': np.asarray(site_data.get('ky_inf_bray', sub_value)),
            })
        
    # Surface fault rupture
    if setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['ToAssess']:
    
        # Update setup_config with parameters
        setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['OtherParameters'].update({
            'l_seg': np.asarray(site_data.get('l_seg (km)', sub_value)),
        })
    
    # Repair Rate PGD:
    if setup_config['DecisionVariable']['Type']['RepairRatePGD']['ToAssess']:
    
        # Update setup_config with parameters
        setup_config['DecisionVariable']['Type']['RepairRatePGD']['OtherParameters'].update({
            'l_seg': np.asarray(site_data.get('l_seg (km)', sub_value)),
        })
        
    #
    return setup_config
    
        
# -----------------------------------------------------------
def export_site_loc_and_vs30(site_data, other_config_param):
    """
    Exports site locations and vs30 values to two text files; for use with OpenSHA interface (may be removed later)
    
    """
    #
    logging.info(f"Exporting site locations and Vs30 to text files for OpenSHA interface...")
    
    #
    # other_config_param['flag_export_site_loc'] = False # default
    site_loc_file = os.path.join(other_config_param['Dir_Site'],'site_loc.txt')
    if os.path.exists(site_loc_file):
        logging.info("\tSite location file already exists; did not export")
    else:
        np.savetxt(site_loc_file,site_data.loc[:,['Longitude','Latitude']].values,fmt='%10.6f,%10.6f') # export vs30 to file for im_tool
        # other_config_param['flag_export_site_loc'] = True
        logging.info("\tSite location file exported to:")
        logging.info(f"\t\t{site_loc_file}")
    # other_config_param['flag_export_vs30'] = False # default
    
    #
    if 'VS30 (m/s)' in site_data:
        vs30_file = os.path.join(other_config_param['Dir_Site'],'vs30.txt')
        if os.path.exists(vs30_file):
            logging.info("\tVs30 file already exists; did not export")
        else:
            np.savetxt(vs30_file,site_data['VS30 (m/s)'].values,fmt='%6.2f') # export vs30 to file for im_tool
            # other_config_param['flag_export_vs30'] = True
            logging.info("\tVs30 file exported to:")
            logging.info(f"\t\t{vs30_file}")
    else:
        logging.info("\t... vs30 not provided as input")
        

# -----------------------------------------------------------
def print_setup_config_to_logging(arr,find_val,set_val):
    """
    Function that creates and prints a summary message to cmd based on setup configuration
    (under construction)
    
    """

    # -----------------------------------------------------------
    # Print messages for...
    # General
    logging.info(f"Analysis Configuration")
    logging.info("-----------------------------------------------------------")
    # logging.info(f"Setup for phase {input.phase_to_run}: {input.phase_message}")
    
    # Site data
    logging.info(f"Site locations and data")
    logging.info(f"\tLoaded site locations and data from:")
    logging.info(f"\t\t{Path_SiteData}")
    if flag_update_site_data_file:
        logging.info(f"\tUpdated and saved site data to the original file")
    if flag_export_site_loc:
        logging.info(f"\tExported site locations for {gm_dir}")
    if flag_export_vs30:
        logging.info(f"\tExported vs30 for {gm_dir}")

    # Source and ground motion predictions
    logging.info(f"EQ sources and GMs")
    logging.info(f"\tDirectory for source and GM outputs:")
    logging.info(f"\t\t{gm_dir}")
    logging.info(f"\tSource and ground motion tool: {input.im_tool}")
    logging.info(f"\tSource model: {input.src_model}")
    logging.info(f"\tCutoff return period: {input.tr_max} years")
    logging.info(f"\tCutoff distance: {input.r_max} km")

    # for phases 2+
    if input.phase_to_run >= 2:
        logging.info(f"\tGM model: {input.im_model}")
        # logging.info(f"Number of groups of ruptures = {len(ListOfRuptureGroups)} (each group contains {rup_per_group} ruptures)")
        
        if Flag_GetIM or flag_gen_sample:
            logging.info(f"\tWill print messages every {inc1} groups (each group contains {input.rup_per_group} ruptures)")
        if flag_save_dv:
            logging.info(f"\tWill save damage outputs every {inc2} groups (each group contains {input.rup_per_group} ruptures)")
    
    # for phases 3+
    if input.phase_to_run >= 3:
        logging.info(f"IMs")
        if input.phase_to_run == 3:
            logging.info(f"\tRequested IMs include: {IM}")
            logging.info(f"\tPerform spatial correlation = {input.flag_spatial_corr}")
            logging.info(f"\tPerform cross-correlation = {input.flag_cross_corr}")
        elif input.phase_to_run == 4:
            logging.info(f"\tRequired IMs include: {IM} (list may be altered based on requested DVs)")
        logging.info(f"\tNumber of IM samples = {input.n_samp_im}")

    # for phase 4
    if input.phase_to_run == 4:
        #
        logging.info(f"EDPs")
        logging.info(f"\tRequired EDPs include: {input.edps} (list may be altered based on requested DVs)")
        # logging.info(f"\tNumber of EDP samples = {input.n_samp_edp} (for uniform distributions use only)")
        logging.info(f"\tFlag for probability of liquefaction = {Flag_PLiq}")
        logging.info(f"\tFlag for liquefaction susceptibility = {Flag_LiqSusc}")
        logging.info(f"\tFlag for probability of landslide = {Flag_PLand}")
        # if Flag_PLiq or Flag_LiqSusc:
            # logging.info(f"\tLoaded site parameters for liquefaction into 'edp_OtherParameterss'")
        # if flag_calc_ky:
            # logging.info(f"Calculated ky using Bray (2007) for infinite slope for landslide-induced demands.")
        # else:
            # logging.info(f"Loaded ky into 'edp_OtherParameterss'")
                
        #
        logging.info(f"DMs")
        logging.info(f"\tRequired DMs include: {input.DMs} (list may be altered based on requested DVs)")
        logging.info(f"\tNumber of DM samples = {input.n_samp_DM} (for uniform distributions use only)")
        #
        logging.info(f"DVs")
        logging.info(f"\tRequested DVs include: {input.dvs}")
        logging.info(f"\tDirectory for DV outputs:")
        logging.info(f"\t\t{dv_dir}")
    
    
    #
    # Print messages for...
    # reload model.py if it has been modified
    # importlib.reload(model)
    # logging.info(f'Load/reloaded "model.py".')

    #
    logging.info("-----------------------------------------------------------")
    logging.info(f"Performing phase {input.phase_to_run} analysis")
    
    
# -----------------------------------------------------------
def set_elements_to_nan(arr,find_val,set_val):
    """
    Finds all the elements in an array **arr** that equals the value **find_val** and sets these elements to the value **set_val**.
    
    Parameters
    ----------
    arr: float, array
        an array of values
    find_val: float
        value to search for
    set_val: float
        value to set to
        
    Returns
    -------
    arr: float, array
        modified array
    
    """
	
    #
    filters = np.where(arr==find_val)
    if len(filters[0]) > 0:
        arr[filters] = set_val
    
    #
    return arr