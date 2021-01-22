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

# OpenSRA modules and functions
from src import Fcn_Common
from src.Site import Fcn_Site


def setup_other_method_param(other_config_param, method_assess_param):
    """
    Get other parameters for methods
    
    """
    
    # Loop through keys to update method_assess_param
    for category in method_assess_param.keys():
        # Get the types of hazard to assess:
        for hazard in method_assess_param[category].keys():
            # Return parameters
            method_assess_param[category][hazard]['ReturnParameter'] = []
            if category == 'EDP':
                if 'liq' in hazard.lower():
                    if other_config_param['Flag_LiqSusc']:
                        method_assess_param[category][hazard]['ReturnParameter'].append('liq_susc')
                    if other_config_param['Flag_PLiq']:
                        method_assess_param[category][hazard]['ReturnParameter'].append('p_liq')
                else:
                    if 'land' in hazard.lower():
                        if other_config_param['Flag_PLand']:
                            method_assess_param[category][hazard]['ReturnParameter'].append('p_land')
                    if 'surf' in hazard.lower():
                        pass
                        # -----------------------------------------------------------
                        # -----------------------------------------------------------                    
                        ## set up flags for fault crossings
                        #rows = []
                        #cols = []
                        #n_rup = len(model._GM_pred_dict['rup']['src'])
                        #n_site = len(model._src_site_dict['site_lon'])
                        #count_seg = 0
                        #for src in model._GM_pred_dict['rup']['src']:
                        #    with warnings.catch_warnings():
                        #        warnings.simplefilter("ignore")
                        #        seg_list = np.loadtxt(os.path.join(intersect_dir,'src_'+str(src)+'.txt'),dtype=int,ndmin=1)
                        #    for seg in seg_list:
                        #        rows.append(count_seg)
                        #        cols.append(seg)
                        #    count_seg += 1
                        #rows = np.asarray(rows)
                        #cols = np.asarray(cols)
                        #mat = sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n_rup,n_site))
                        #logging.debug(f'\tEDP_{edp_i}: Generated matrix of flags for fault crossing')
                        #input.edp_other_params.update({'mat_seg2calc':mat})
                        # -----------------------------------------------------------
                        # -----------------------------------------------------------    
                        
                    method_assess_param[category][hazard]['ReturnParameter'].append(
                        'pgd_' + other_config_param['ID_EDP'][hazard])
            #
            elif category == 'DV':
                method_assess_param[category][hazard]['ReturnParameter'].append(
                    other_config_param['ID_DV'][hazard])
            # Uncertainty
            method_assess_param[category][hazard]['Uncertainty'] = {}
            if hazard in other_config_param['DistributionType']['Uniform']['ListOfHazard']:
                method_assess_param[category][hazard]['Uncertainty']['DistributionType'] = 'Uniform'
                method_assess_param[category][hazard]['Uncertainty']['Epistemic'] = {
                    'Branch': other_config_param['DistributionType']['Uniform']['Epistemic']['Branch'],
                    'Weight': other_config_param['DistributionType']['Uniform']['Epistemic']['Weight']
                }
            else:
                method_assess_param[category][hazard]['Uncertainty']['DistributionType'] = 'Lognormal'
                method_assess_param[category][hazard]['Uncertainty']['Epistemic'] = {
                    'Branch': other_config_param['DistributionType']['Lognormal']['Epistemic']['Branch'],
                    'Weight': other_config_param['DistributionType']['Lognormal']['Epistemic']['Weight']
                }
    
    #
    return method_assess_param


# -----------------------------------------------------------
def get_other_setup_config(setup_config, other_config_param, folders_to_create, method_assess_param):
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
    other_config_param['Path_SiteData'] = setup_config['Infrastructure']['SiteDataFile'] # create full path for site data file
    other_config_param['Path_SiteData_Updated'] = os.path.join(
        other_config_param['Dir_Site'], 
        os.path.basename(other_config_param['Path_SiteData']).replace('.csv','_Updated.csv')
    ) # create full path for updated site data file
    other_config_param['Dir_IM'] = os.path.join(dir_working, 'IM') # set full path for GM output folder
    folders_to_create.append(other_config_param['Dir_IM'])
    
    # Source for Vs30
    # other_config_param['SourceForVs30'] = setup_config['Infrastructure']['SourceForVs30']
    other_config_param['SourceForVs30'] = setup_config['IntensityMeasure']['SourceParameters']['SourceForVs30']
    

    # -----------------------------------------------------------
    # PEER categories
    peer_categories = other_config_param['PEERCategories']
    # Loop through PEER categories to update other_config_param and method_assess_param
    for category in peer_categories.keys():
        # Initialize category in other_config_param and method_assess_param 
        other_config_param[peer_categories[category]] = []
        method_assess_param[peer_categories[category]] = {}
        
        # Get the types of hazard to assess:
        if category == 'EngineeringDemandParameter':
            hazard_list = list(other_config_param['ID_EDP'])
        else:
            hazard_list = list(setup_config[category]['Type'])
        for hazard in hazard_list:
            # Get flag for 'ToAssess'
            Flag_ToAssess = setup_config[category]['Type'][hazard]['ToAssess']
            # For Liquefaction force True if either LateralSpread or GroundSettlement is requested
            if hazard == 'Liquefaction':
                if setup_config[category]['Type']['LateralSpread']['ToAssess'] \
                    or setup_config[category]['Type']['GroundSettlement']['ToAssess']:
                    Flag_ToAssess = True
            # If 'Flag_ToAssess' is True, update dictionaries
            if Flag_ToAssess:
                # For other_config_param
                other_config_param[peer_categories[category]].append(hazard)
                # For method_assess_param
                if not category == 'IntensityMeasure':
                    # Initialize hazard
                    method_assess_param[peer_categories[category]][hazard] = {'Method':{}}
                    count = 0
                    weights_to_use = setup_config[category]['Type'][hazard]['ListOfWeights']
                    if sum(weights_to_use) != 1:
                        weights_to_use = list(np.asarray(weights_to_use)/sum(weights_to_use))
                    # Initialize key for method and get weight; set default flags for need of IMs and magnitude
                    for method in setup_config[category]['Type'][hazard]['ListOfMethods']:
                        method_assess_param[peer_categories[category]][hazard]['Method'][method] = {
                            'Weight': weights_to_use[count],
                            'Flags': {'PGA': False, 'PGV': False, 'mag': False},
                            'InputParameters': {}}
                        # Pull 'OtherParameters' into method_assess_param
                        for param in setup_config[category]['Type'][hazard]['OtherParameters'].keys():
                            method_assess_param[peer_categories[category]][hazard]['Method'][method]['InputParameters'].update({
                                param: setup_config[category]['Type'][hazard]['OtherParameters'][param]})
                        count+=1
        other_config_param['Num_'+peer_categories[category]] = len(other_config_param[peer_categories[category]])
       
       
    # -----------------------------------------------------------
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
        other_config_param['Path_RuptureSegment'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'rupture_segments.json')
        other_config_param['Path_PointSource'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'point_sources.txt')
        
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
    
    # Check if IM samples are needed based on DVs
    other_config_param['Flag_GetIM'] = False # default
    if 'RepairRatePGV' in other_config_param['DV']:
        other_config_param['Flag_GetIM'] = True
    if 'RepairRatePGD' in other_config_param['DV']:
        if 'LateralSpread' in other_config_param['EDP'] or \
            'Landslide' in other_config_param['EDP'] or \
            'SurfaceFaultRupture' in other_config_param['EDP']:
                other_config_param['Flag_GetIM'] = True
                
    # If IM samples are not needed, then clear list of IM
    # if other_config_param['Flag_GetIM'] is False:
        # other_config_param['IM'] = []
    # Clear list of EDP required if only RepairRatePGV is to be assessed
    if 'RepairRatePGV' in other_config_param['DV'] and \
        not 'RepairRatePGD' in other_config_param['DV']:
            for key in setup_config['EngineeringDemandParameter']['Type'].keys():
                other_config_param['EDP'] = []
            # other_config_param['IM'] = ['PGV'] # set IM to only PGV
    
    
    # -----------------------------------------------------------
    # Engineering demand parameters
    # Note: OpenSRA currently is not set up to store EDP results
    
    # Check if probability of liquefaction and liquefaction susceptibility are needed
    other_config_param['Flag_LiqSusc'] = False # default
    other_config_param['Flag_PLiq'] = False # default
    if 'Liquefaction' in other_config_param['EDP']:
        other_config_param['Flag_LiqSusc'] = True
    if 'LateralSpread' in other_config_param['EDP'] \
        or 'GroundSettlement' in other_config_param['EDP']:
        other_config_param['Flag_PLiq'] = True
    
    # Check if probability of landslide is needed
    other_config_param['Flag_PLand'] = False # default
    if 'Landslide' in other_config_param['EDP'] and \
        'RepairRatePGD' in other_config_param['DV']:
        other_config_param['Flag_PLand'] = True
    
    # -----------------------------------------------------------
    # Damage measures
    # Note: OpenSRA currently is not set up to store DM results
    
    
    # -----------------------------------------------------------
    # Decision variables
    other_config_param['EDPDemandForRepairRatePGD'] = {}
    if 'RepairRatePGD' in other_config_param['DV']:
        for edp_i in other_config_param['EDP']:
            # make label for PGD type
            if edp_i == 'LateralSpread' or edp_i == 'GroundSettlement':
                prob_label = 'p_liq'
            elif edp_i == 'Landslide':
                prob_label = 'p_land'
            else:
                prob_label = None
            # add labels
            if not edp_i == 'Liquefaction':
                other_config_param['EDPDemandForRepairRatePGD'][edp_i] = {
                    'Label_PGD': 'pgd_'+other_config_param['ID_EDP'][edp_i],
                    'Label_Prob': prob_label
                }
    
    # -----------------------------------------------------------    
    # Create sub-directories within the PEER categories based on IMs, hazards, and loss variables to assess
    # Loop through PEER categories
    for category in peer_categories.keys():        
        # Make directories for storage
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
        for item in other_config_param[peer_categories[category]]:
            if not category == "DecisionVariable":
                for name in base_param_name:
                    # Set full path for GM prediction output folder
                    other_config_param[name + '_' + item] = os.path.join(other_config_param[name], item)
                    folders_to_create.append(other_config_param[name + '_' + item])
    
    
    # -----------------------------------------------------------    
    # Append OpenSRA path to the user-defined relative paths
    other_config_param['Shp_GeologicUnit'] = os.path.join(dir_opensra,other_config_param['Shp_GeologicUnit']) # location of geologic unit shapefile
    other_config_param['File_GeologicUnitParam'] = os.path.join(dir_opensra,other_config_param['File_GeologicUnitParam']) # location of Slate's parameters for geologic units
    
    # Create directories for storing fault traces and intersections if fault ruptures are to be assessed
    if 'SurfaceFaultRupture' in other_config_param['EDP'] or \
        'SubsurfaceFaultRupture' in other_config_param['EDP']:
        other_config_param['Dir_FaultCrossing'] = os.path.join(other_config_param['Dir_IM'],'FaultSource')
        other_config_param['Dir_FaultTrace'] = os.path.join(other_config_param['Dir_FaultCrossing'],'Trace')
        other_config_param['Dir_Intersection'] = os.path.join(other_config_param['Dir_FaultCrossing'],'Intersection')
        folders_to_create.append(other_config_param['Dir_FaultCrossing'])
        folders_to_create.append(other_config_param['Dir_FaultTrace'])
        folders_to_create.append(other_config_param['Dir_Intersection'])
    
    # Export other_config_param dictionary to JSON for checks
    other_config_param = {key:other_config_param[key] for key in sorted(other_config_param)} # sort for export
    with open(os.path.join(dir_working,'CreatedConfigParams.json'), 'w') as outfile:
        json.dump(other_config_param, outfile, indent=4, separators=(',', ': '))
    
    # Eeturn updated dictionary of other_config_param
    return other_config_param, folders_to_create, method_assess_param
    
    
# -----------------------------------------------------------
def update_site_data_file(site_data, setup_config, other_config_param):
    """
    Updates existing site data file with additional information necessary for run
    
    """
    
    #
    logging.info(f"Updating site data file...")
    logging.info(f"\tExtracting info from...")
    # logging.info(f"\t{other_config_param['Path_SiteData']}")
    
    # Flag for whether the site data matrix is updated, default = False
    flag_update_site_data_file = False # default
    
    # Check if segment lengths are given in the file; if not then assume lengths of 1 km and add to "site_data"
    if not 'SHAPE_LENGTH' in site_data.columns:
        site_data['SHAPE_LENGTH'] = [1]*len(site_data)
        flag_update_site_data_file = True # updated site data
        
    # Check if start and end locations are given in the file; if not set equal to site locations (i.e., points)
    if not 'LONG_BEGIN' in site_data.columns:
        site_data['LONG_BEGIN'] = site_data['LONG_MIDDLE'].values
        site_data['LAT_BEGIN'] = site_data['LAT_MIDDLE'].values
        site_data['LONG_END'] = site_data['LONG_MIDDLE'].values
        site_data['LAT_END'] = site_data['LAT_MIDDLE'].values
        flag_update_site_data_file = True # updated site data
    
    # General site parameters: Vs30, slope, DEM
    for param in other_config_param['SiteParams']:
        column_id = other_config_param['SiteParams'][param]['ColumnNameToStoreAs']
        if not column_id in site_data:
            path_to_raster = other_config_param['SiteParams'][param]['PathToRaster']
            if path_to_raster is None:
                if 'slope' in column_id.lower():
                    site_data[column_id] = np.zeros(len(site_data))*0.1
                elif 'vs30' in column_id.lower():
                    site_data[column_id] = np.ones(len(site_data))*999
            else:
                raster_interp_data = Fcn_Common.interp_from_raster(
                    raster_path = path_to_raster,
                    x = site_data.loc[:,'LONG_MIDDLE'].values,
                    y = site_data.loc[:,'LAT_MIDDLE'].values)
                if 'slope' in column_id.lower():
                    raster_interp_data[raster_interp_data<=0] = 0.1
                elif 'vs30' in column_id.lower():
                    raster_interp_data[raster_interp_data<=0] = 999
                site_data[column_id] = raster_interp_data
            other_config_param['ColumnsAppended'].append(column_id)
            flag_update_site_data_file = True # updated site data
    
    # Liquefaction
    if 'Liquefaction' in other_config_param['EDP']:
        # For Zhu et al. (2017)
        if 'ZhuEtal2017' in setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['ListOfMethods'] and \
            not other_config_param['ZhuEtal2017_param']['precip']['ColumnNameToStoreAs'] in site_data.columns:
            # Loop through all params for Zhu et al. (2017)
            for key in other_config_param['ZhuEtal2017_param'].keys():
                param_info = other_config_param['ZhuEtal2017_param'][key]
                file = os.path.join(
                    other_config_param['Dir_ZhuEtal2017_param'],
                    param_info['FolderName'],
                    param_info['FolderName']+'.tif')
                site_data[param_info['ColumnNameToStoreAs']] = \
                    Fcn_Common.interp_from_raster(
                        raster_path = file,
                        x = site_data.loc[:,'LONG_MIDDLE'].values,
                        y = site_data.loc[:,'LAT_MIDDLE'].values)
                other_config_param['ColumnsAppended'].append(param_info['ColumnNameToStoreAs'])
            flag_update_site_data_file = True # updated site data
    
    # Landslide
    # Check if need to extract geologic units and properties, get the params if not
    if 'Landslide' in other_config_param['EDP']:
        if not 'Geologic Unit (CGS, 2010)' in site_data.columns:
            # run script to get the properties
            site_data, other_config_param['ColumnsAppended'] = Fcn_Site.get_geologic_param(
                site_data = site_data, 
                geo_shp_file = other_config_param['Shp_GeologicUnit'],
                geo_unit_param_file = other_config_param['File_GeologicUnitParam'],
                ky_param_to_use = other_config_param['ColumnsToUseForKy'],
                source_param_for_ky = setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters']['SourceParametersForKy'],
                appended_columns = other_config_param['ColumnsAppended']
            )
            flag_update_site_data_file = True # updated site data
    
    # Check flag to see if site data matrix is updated, if so then save to site file
    if flag_update_site_data_file:
        # rearrange appended columns
        columns_to_place_at_the_end = []
        for col in site_data.columns:
            if not col in other_config_param['ColumnsToKeepInFront'] and \
                not col in other_config_param['ColumnsAppended']:
                columns_to_place_at_the_end.append(col)
        columns_reorder = \
            other_config_param['ColumnsToKeepInFront'] + \
            other_config_param['ColumnsAppended'] + \
            columns_to_place_at_the_end
        site_data = site_data[columns_reorder]
        site_data.to_csv(other_config_param['Path_SiteData_Updated'],index=False)
        logging.info(f"... Site data file updated to:")
        logging.info(f"\t{other_config_param['Path_SiteData_Updated']}")
    else:
        logging.info(f"\tSite data file is up-to-date")
    
    # Return site_data dataframe and flag of whether site_data has been updated
    return site_data

    
# -----------------------------------------------------------
def add_site_data_to_method_param(setup_config, site_data, other_config_param, method_assess_param):
    """
    Add parameters in site_data to method parameters in method_assess_param. Currently method dependent
    
    """
    
    #
    logging.info(f"Adding site data to list of model parameters in setup_config...")

    # Flag for whether the site data matrix is updated, default = False
    flag_update_site_data_file = False # default
    
    # Define values to substitute if parameters are not included in site_data
    sub_value = [0]*len(site_data)
    find_val = -9999
    set_val1 = 0
    set_val2 = 2000
    
    # For specific PEER categories, geohazards, and methods:
    
    # Liquefaction:
    # if Flag_LiqSusc or Flag_PLiq:
    if 'Liquefaction' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'Liquefaction'
    
        # For Zhu et al. (2017)
        if 'ZhuEtal2017' in setup_config['EngineeringDemandParameter']['Type'][hazard]['ListOfMethods']:
            method = 'ZhuEtal2017'
            # Get values from site_data, set to list of NaN if doesn't exist
            wtd = np.asarray(site_data.get('Water Table Depth (m)', [0]*len(site_data)))
            dr = np.asarray(site_data.get('Dist_River (km)', [0]*len(site_data)))
            dc = np.asarray(site_data.get('Dist_Coast (km)', [0]*len(site_data)))
            dw = np.asarray(site_data.get('Dist_Any Water (km)', [0]*len(site_data)))
            precip = np.asarray(site_data.get('CA_Precip (mm)', [0]*len(site_data)))
            vs30 = np.asarray(site_data.get('Vs30 (m/s)', [999]*len(site_data)))
            # Find elements = find_val and set to set_val
            wtd = set_elements_to_nan(wtd, find_val, 0)
            dr = set_elements_to_nan(dr, find_val, 0)
            dc = set_elements_to_nan(dc, find_val, 0)
            dw = set_elements_to_nan(dw, find_val, 0)
            precip = set_elements_to_nan(precip, find_val, 0)
            vs30 = set_elements_to_nan(vs30, find_val, 999)
            
            # Temporary values for elevations
            z = np.asarray(site_data.get('Elev (m)', [10]*len(site_data))) # set to 10 for now, get data from DEM map later
            # Update method_assess_param with parameters
            # setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['OtherParameters'].update({
            method_assess_param[category][hazard]['Method'][method]['InputParameters'].update({
                # 'wtd': list(wtd), 'dr': list(dr), 'dc': list(dc),
                # 'dw': list(dw), 'precip': list(precip), 'vs30': list(vs30), 'z': list(z),
                'wtd': wtd, 'dr': dr, 'dc': dc,
                'dw': dw, 'precip': precip, 'vs30': vs30, 'z': z,
            })
            method_assess_param[category][hazard]['Method'][method]['Flags']['PGV'] = True
            method_assess_param[category][hazard]['Method'][method]['Flags']['mag'] = True

    # Lateral Spread
    if 'LateralSpread' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'LateralSpread'
    
        # For Hazus (2014)
        if 'Hazus2014' in setup_config['EngineeringDemandParameter']['Type'][hazard]['ListOfMethods']:
            method = 'Hazus2014'
            # Get parameters from site_data
            dwWB = np.asarray(site_data.get('Dist_Any Water (km)', sub_value))
            dwWB = set_elements_to_nan(dwWB, find_val, 999)
            dwWB = dwWB*1000 # convert to meters
            # Update method_assess_param with parameters
            # setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['OtherParameters'].update({
            method_assess_param[category][hazard]['Method'][method]['InputParameters'].update({
                'dw':dwWB,
            })
            method_assess_param[category][hazard]['Method'][method]['Flags']['PGA'] = True
            method_assess_param[category][hazard]['Method'][method]['Flags']['mag'] = True

    # Landslide
    if 'Landslide' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'Landslide'
    
        # For Bray & Macedo (2019)
        if 'BrayMacedo2019' in setup_config['EngineeringDemandParameter']['Type'][hazard]['ListOfMethods']:
            method = 'BrayMacedo2019'
            # Update method_assess_param with parameters
            # setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters'].update({
            method_assess_param[category][hazard]['Method'][method]['InputParameters'].update({
                # 'ky': np.asarray(site_data.get('Ky for Infinite Slope_Bray', sub_value)),
                'ky': site_data.get('Ky for Infinite Slope_Bray', sub_value).to_numpy(),
            })
            method_assess_param[category][hazard]['Method'][method]['Flags']['PGA'] = True
            method_assess_param[category][hazard]['Method'][method]['Flags']['PGV'] = True
            method_assess_param[category][hazard]['Method'][method]['Flags']['mag'] = True
        
    # Surface fault rupture
    if 'SurfaceFaultRupture' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'SurfaceFaultRupture'
    
        # For every method
        for method_i in setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['ListOfMethods']:
            # Update method_assess_param with parameters
            # setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['OtherParameters'].update({
            method_assess_param[category][hazard]['Method'][method_i]['InputParameters'].update({
                'l_seg': np.asarray(site_data.get('SHAPE_LENGTH', sub_value)),
            })
            method_assess_param[category][hazard]['Method'][method_i]['Flags']['mag'] = True
    
    # Repair Rate PGD:
    if 'RepairRatePGV' in other_config_param['DV']:
        category = 'DV'
        hazard = 'RepairRatePGV'
        
        # For every method
        for method_i in setup_config['DecisionVariable']['Type'][hazard]['ListOfMethods']:
            method_assess_param[category][hazard]['Method'][method_i]['Flags']['PGV'] = True
    
    # Repair Rate PGD:
    if 'RepairRatePGD' in other_config_param['DV']:
        category = 'DV'
        hazard = 'RepairRatePGD'
    
        # For every method
        for method_i in setup_config['DecisionVariable']['Type'][hazard]['ListOfMethods']:
            # Update method_assess_param with parameters
            # setup_config['DecisionVariable']['Type']['RepairRatePGD']['OtherParameters'].update({
            method_assess_param[category][hazard]['Method'][method_i]['InputParameters'].update({
                'l_seg': np.asarray(site_data.get('SHAPE_LENGTH', sub_value)),
            })
            method_assess_param[category][hazard]['Method'][method_i]['Flags']['PGA'] = True
            method_assess_param[category][hazard]['Method'][method_i]['Flags']['mag'] = True
        
    #
    return method_assess_param
    
        
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
        np.savetxt(site_loc_file,site_data.loc[:,['LONG_MIDDLE','LAT_MIDDLE']].values,fmt='%10.6f,%10.6f') # export vs30 to file for im_tool
        # other_config_param['flag_export_site_loc'] = True
        logging.info("\tSite location file exported to:")
        logging.info(f"\t\t{site_loc_file}")
    
    #
    vs30_file = os.path.join(other_config_param['Dir_Site'],'vs30.txt')
    if os.path.exists(vs30_file):
        logging.info("\tVs30 file already exists; will not export")
    else:
        # if other_config_param['SourceForVs30'] == 'Wills et al. (2015)':
            # logging.info("Vs30 will be inferred from Wills et al. (2015)")
        # elif other_config_param['SourceForVs30'] == 'User Defined':
        # if 'Vs30 (m/s)' in site_data:
        np.savetxt(vs30_file,site_data['Vs30 (m/s)'].values,fmt='%6.2f') # export vs30 to file for im_tool
        logging.info("\tVs30 file exported to:")
        logging.info(f"\t\t{vs30_file}")
        # else:
            # logging.info("\t... Vs30 not provided as input, will infer from Wills et al. (2015)")
            # other_config_param['SourceForVs30'] = 'Wills et al. (2015)'
        # else:
            # logging.info("Source for Vs30 is invalid; currently supported sources:")
            # logging.info('\t"User Defined" or "Wills et al. (2015)"')
        

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