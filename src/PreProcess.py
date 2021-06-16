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
from src.EDP import Fcn_EDP


def setup_other_method_param(other_config_param, method_param_for_assess):
    """
    Get other parameters for methods
    
    """
    
    # Loop through keys to update method_param_for_assess
    for category in method_param_for_assess:
        # Get the types of hazard to assess:
        for hazard in method_param_for_assess[category]:
            # Return parameters
            method_param_for_assess[category][hazard]['ReturnParameter'] = []
            if category == 'EDP':
                if 'liq' in hazard.lower():
                    if other_config_param['Flag_LiqSusc']:
                        method_param_for_assess[category][hazard]['ReturnParameter'].append('liq_susc')
                    if other_config_param['Flag_PLiq']:
                        method_param_for_assess[category][hazard]['ReturnParameter'].append('p_liq')
                else:
                    if 'land' in hazard.lower():
                        if other_config_param['Flag_PLand']:
                            method_param_for_assess[category][hazard]['ReturnParameter'].append('p_land')
                    if 'surf' in hazard.lower():
                        # if other_config_param['Flag_PSurf']:
                            # method_param_for_assess[category][hazard]['ReturnParameter'].append('p_surf')
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
                        
                    method_param_for_assess[category][hazard]['ReturnParameter'].append(
                        'pgd_' + other_config_param['ID_EDP'][hazard])
            #
            elif category == 'DV':
                method_param_for_assess[category][hazard]['ReturnParameter'].append(
                    other_config_param['ID_DV'][hazard])
            # Uncertainty
            method_param_for_assess[category][hazard]['Uncertainty'] = {}
            if hazard in other_config_param['DistributionType']['Uniform']['ListOfHazard']:
                method_param_for_assess[category][hazard]['Uncertainty']['DistributionType'] = 'Uniform'
                method_param_for_assess[category][hazard]['Uncertainty']['Epistemic'] = {
                    'Branch': other_config_param['DistributionType']['Uniform']['Epistemic']['Branch'],
                    'Weight': other_config_param['DistributionType']['Uniform']['Epistemic']['Weight']
                }
            else:
                method_param_for_assess[category][hazard]['Uncertainty']['DistributionType'] = 'Lognormal'
                method_param_for_assess[category][hazard]['Uncertainty']['Epistemic'] = {
                    'Branch': other_config_param['DistributionType']['Lognormal']['Epistemic']['Branch'],
                    'Weight': other_config_param['DistributionType']['Lognormal']['Epistemic']['Weight']
                }
    
    #
    return method_param_for_assess


# -----------------------------------------------------------
def get_other_setup_config(setup_config, other_config_param, folders_to_create, method_param_for_assess):
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
    other_config_param['Path_DatasetMetadata'] = os.path.join(
        other_config_param['Dir_Site'], 'DatasetMetadata.json') # create full path for JSON file with dataset metadata
    other_config_param['Dir_IM'] = os.path.join(dir_working, 'IM') # set full path for GM output folder
    folders_to_create.append(other_config_param['Dir_IM'])
    
    # Source for Vs30
    # other_config_param['SourceForVs30'] = setup_config['Infrastructure']['SourceForVs30']
    # other_config_param['SourceForVs30'] = setup_config['IntensityMeasure']['SourceParameters']['SourceForVs30']
    

    # -----------------------------------------------------------
    # PEER categories
    peer_categories = other_config_param['PEERCategories']
    other_config_param['ParamsToGetFromDatasets'] = []
    # Loop through PEER categories to update other_config_param and method_param_for_assess
    for category in peer_categories:
        # Initialize category in other_config_param and method_param_for_assess 
        other_config_param[peer_categories[category]] = []
        method_param_for_assess[peer_categories[category]] = {}
        
        # Check if Vs30,z1p0,z2p5 needs to be obtained from pre-packaged dataset
        if category == 'IntensityMeasure':
            if setup_config[category]['SourceForIM'] == 'OpenSHA':
                for param in setup_config[category]['SourceParameters']['OtherParameters']:
                    if 'z1p0' in param or 'z2p5' in param:
                        pass # to get with OpenSHAInterface
                    else:
                        if setup_config[category]['SourceParameters']['OtherParameters'][param]['Source'] != 'UserDefined':
                            other_config_param['ParamsToGetFromDatasets'].append(param)
                # -----------------------------------------------------------
                # to add z1p0
                # to add z2p5
                # -----------------------------------------------------------
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
                # For method_param_for_assess
                if category != 'IntensityMeasure':
                    # Initialize hazard
                    method_param_for_assess[peer_categories[category]][hazard] = {'Method':{}}
                    count = 0
                    # get method weight, and if sum of weights do not add to 1, normalize weights to sum of 1
                    # weights_to_use = setup_config[category]['Type'][hazard]['ListOfWeights']
                    weights_to_use = []
                    for method in setup_config[category]['Type'][hazard]['Method']:
                        method_info = setup_config[category]['Type'][hazard]['Method'][method]
                        weights_to_use.append(method_info['ModelWeight'])
                    if sum(weights_to_use) != 1:
                        weights_to_use = list(np.asarray(weights_to_use)/sum(weights_to_use))
                    # Initialize key for method and get weight; set default flags for need of IMs and magnitude
                    for method in setup_config[category]['Type'][hazard]['Method']:
                        method_info = setup_config[category]['Type'][hazard]['Method'][method]
                        method_param_for_assess[peer_categories[category]][hazard]['Method'][method] = {
                            'Weight': weights_to_use[count], # store method weights
                            'Flags': {'PGA': False, 'PGV': False, 'mag': False}, # store flags for PGA, PGV, M (obtained from IM section)
                            'InputParameters': method_info # get method input params
                        }
                        # Check which input parameter(s) need to be obtained from pre-packaged dataset
                        for param in method_info:
                            param_info = method_info[param]
                            if isinstance(param_info,dict):
                                if 'Source' in param_info:
                                    if param_info['Source'] == 'Preferred' and \
                                        not param in other_config_param['ParamsToGetFromDatasets']:
                                        other_config_param['ParamsToGetFromDatasets'].append(param)
                        # Pull 'OtherParameters' into method_param_for_assess
                        for param in setup_config[category]['Type'][hazard]['OtherParameters']:
                            method_param_for_assess[peer_categories[category]][hazard]['Method'][method]['InputParameters'].update({
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
    other_config_param['File_ListOfScenarios'] = other_config_param['File_ListOfScenarios'] + '.csv'
    # File with scenario traces
    other_config_param['File_ScenarioTraces'] = os.path.join(other_config_param['Dir_IM_SeismicSource'],'ScenarioTraces.csv')
    
    # Check the source for intensity measures
    if setup_config['IntensityMeasure']['SourceForIM'] == 'OpenSHA':
        # Create additional parameters for use of source model
        src_model = setup_config['IntensityMeasure']['SourceParameters']['SeismicSourceModel']
        other_config_param['Path_FiniteSourceConnectivity'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'FiniteSourceConnectivity.csv')
        other_config_param['Path_RuptureSegment'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'RuptureSegments.csv')
        other_config_param['Path_PointSource'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'PointSourceTraces.csv')
        other_config_param['Path_RuptureMetadata'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'RuptureMetadata.csv')
        other_config_param['Path_FiniteSource'] = os.path.join(dir_opensra,'lib','OpenSHA','ERF',src_model,'FiniteSourceTraces.zip')
        
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
        other_config_param['File_ListOfScenarios_Filtered'] = other_config_param['File_ListOfScenarios_Filtered'] + '.csv'
        other_config_param['File_ListOfScenarios_Full'] = other_config_param['File_ListOfScenarios_Full'] + '.csv'
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
            for hazard in setup_config['EngineeringDemandParameter']['Type']:
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
    
    # Check if probability of landslide is needed
    # other_config_param['Flag_PSurf'] = False # default
    # if 'SurfaceFaultRupture' in other_config_param['EDP'] and \
        # 'RepairRatePGD' in other_config_param['DV']:
        # other_config_param['Flag_PSurf'] = True
        
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
            if edp_i != 'Liquefaction':
                other_config_param['EDPDemandForRepairRatePGD'][edp_i] = {
                    'Label_PGD': 'pgd_'+other_config_param['ID_EDP'][edp_i],
                    'Label_Prob': prob_label
                }
    
    # -----------------------------------------------------------    
    # Create sub-directories within the PEER categories based on IMs, hazards, and loss variables to assess
    # Loop through PEER categories
    for category in peer_categories:        
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
            if category != "DecisionVariable":
                for name in base_param_name:
                    # Set full path for GM prediction output folder
                    other_config_param[name + '_' + item] = os.path.join(other_config_param[name], item)
                    folders_to_create.append(other_config_param[name + '_' + item])
    
    
    # -----------------------------------------------------------    
    # Append OpenSRA path to the user-defined relative paths
    # other_config_param['Shp_GeologicUnit'] = os.path.join(dir_opensra,other_config_param['Shp_GeologicUnit']) # location of geologic unit shapefile
    # other_config_param['File_GeologicUnitParam'] = os.path.join(dir_opensra,other_config_param['File_GeologicUnitParam']) # location of Slate's parameters for geologic units
    
    # Create directories for storing fault traces and intersections if fault ruptures are to be assessed
    if 'SurfaceFaultRupture' in other_config_param['EDP'] or \
        'SubsurfaceFaultRupture' in other_config_param['EDP']:
        other_config_param['File_FaultCrossing'] = os.path.join(other_config_param['Dir_IM_SeismicSource'],'FaultCrossings.csv')
        # other_config_param['Dir_FaultCrossing'] = os.path.join(other_config_param['Dir_IM'],'FaultSource')
        # other_config_param['Dir_FaultTrace'] = os.path.join(other_config_param['Dir_FaultCrossing'],'Trace')
        # other_config_param['Dir_Intersection'] = os.path.join(other_config_param['Dir_FaultCrossing'],'Intersection')
        # folders_to_create.append(other_config_param['Dir_FaultCrossing'])
        # folders_to_create.append(other_config_param['Dir_FaultTrace'])
        # folders_to_create.append(other_config_param['Dir_Intersection'])
    
    # Export other_config_param dictionary to JSON for checks
    other_config_param = {key:other_config_param[key] for key in sorted(other_config_param)} # sort for export
    with open(os.path.join(dir_working,'CreatedConfigParams.json'), 'w') as outfile:
        json.dump(other_config_param, outfile, indent=4, separators=(',', ': '))
    
    # Eeturn updated dictionary of other_config_param
    return other_config_param, folders_to_create, method_param_for_assess
    
    
# -----------------------------------------------------------
def update_site_data_file(site_data, setup_config, other_config_param):
    """
    Updates existing site data file with additional information necessary for run
    
    """
    
    #
    logging.info(f"Updating site data file...")
    logging.info(f"\tExtracting info from...")
    # logging.info(f"\t{other_config_param['Path_SiteData']}")
    
    # Load existing dictionary for storing metadata for datasets if exists
    if os.path.exists(other_config_param['Path_DatasetMetadata']):
        with open(other_config_param['Path_DatasetMetadata'], 'r') as f:
            dataset_meta_dict = json.load(f)
    else:
        # Initialize dictionary for storing metadata for datasets used
        dataset_meta_dict = {}
    
    # Flag for whether the site data matrix is updated, default = False
    flag_update_site_data_file = False # default
    
    # Create a column for Component ID/num
    if not 'Component Index' in site_data.columns:
        if 'SITE_NUM' in site_data.columns:
            site_data.rename({
                'SITE_NUM': 'Component Index'
            }, axis='columns',inplace=True)
        else:
            # site_data['Site Index'] = list(range(1,len(site_data)+1))
            site_data['Component Index'] = list(range(len(site_data)))
        flag_update_site_data_file = True # updated site data
    
    # Check which site location parameters are given:
    flag_endpoints_exist = True
    flag_midpoint_exists = True
    flag_length_exists = True
    site_loc_params = setup_config['Infrastructure']['SiteLocationParams']
    site_loc_params_to_get = []
    
    # if importing updated site data, actions in site locations should have
    #     been performed; do not perform again
    if other_config_param['Flag_LoadFromUpdatedSiteData']:
        pass
    else:
        for param in site_loc_params:
            if site_loc_params[param]['DataExists'] and \
                site_loc_params[param]['ColumnIDWithData'] in site_data:
                pass
            else:
                site_loc_params_to_get.append(param)
        if 'LatMid' in site_loc_params_to_get or 'LonMid' in site_loc_params_to_get:
            flag_midpoint_exists = False
        if 'LatBegin' in site_loc_params_to_get or 'LonBegin' in site_loc_params_to_get or \
            'LatEnd' in site_loc_params_to_get or 'LonEnd' in site_loc_params_to_get:
            flag_endpoints_exist = False
        logging.info("Site Location parameters to be set:")
        logging.info(f"\t{(', '.join(map(str,site_loc_params_to_get)))}")
        
        # check if either midpoints or endpoints are provided
        if flag_midpoint_exists is False and flag_endpoints_exist is False:
            logging.info(f"Not enough inputs for site locations")
            logging.info(f"At least one of the following conditions is required:")
            logging.info(f"\t (1) Specify midpoint coordinates")
            logging.info(f"\t (2) Specify endpoint (start AND end) coordinates")
        else:
            # update column names to standardized names
            if flag_midpoint_exists:
                site_data.rename({
                    site_loc_params['LonMid']['ColumnIDWithData']: 'Mid Longitude',
                    site_loc_params['LatMid']['ColumnIDWithData']: 'Mid Latitude'
                }, axis='columns',inplace=True)
                flag_update_site_data_file = True # updated site data
            if flag_endpoints_exist:
                site_data.rename({
                    site_loc_params['LonBegin']['ColumnIDWithData']: 'Start Longitude',
                    site_loc_params['LatBegin']['ColumnIDWithData']: 'Start Latitude',
                    site_loc_params['LonEnd']['ColumnIDWithData']: 'End Longitude',
                    site_loc_params['LatEnd']['ColumnIDWithData']: 'End Latitude'
                }, axis='columns',inplace=True)
                flag_update_site_data_file = True # updated site data
                # logging.info("Endpoints of components are set to equal the given midpoints")
            # set endpoints to equal midpoint
            if flag_endpoints_exist is False:
                site_data['Start Longitude'] = site_data['Mid Longitude'].values
                site_data['Start Latitude'] = site_data['Mid Latitude'].values
                site_data['End Longitude'] = site_data['Mid Longitude'].values
                site_data['End Latitude'] = site_data['Mid Latitude'].values
                flag_update_site_data_file = True # updated site data
            # compute midpoint based on endpoints
            if flag_midpoint_exists is False:
                site_data['Mid Longitude'], site_data['Mid Latitude'] = Fcn_Common.get_midpoint(
                    lon1 = site_data['Start Longitude'].values,
                    lat1 = site_data['Start Latitude'].values,
                    lon2 = site_data['End Longitude'].values,
                    lat2 = site_data['End Latitude'].values
                )
                flag_update_site_data_file = True # updated site data
            
        # calculate length of component if not given
        if 'Length' in site_loc_params_to_get:
            if flag_endpoints_exist:
                site_data['Component Length (km)'] = Fcn_Common.get_haversine_dist(
                    lon1 = site_data['Start Longitude'].values,
                    lat1 = site_data['Start Latitude'].values,
                    lon2 = site_data['End Longitude'].values,
                    lat2 = site_data['End Latitude'].values
                )
            else:
                site_data['Component Length (km)'] = [1]*len(site_data)
        else:
            site_data.rename({
                    site_loc_params['Length']['ColumnIDWithData']: 'Component Length (km)'
                }, axis='columns',inplace=True)
            flag_update_site_data_file = True # updated site data
    
    # Check if start and end locations are given in the file; if not set equal to site locations (i.e., points)
    # if not 'LONG_BEGIN' in site_data.columns:
        # site_data['LONG_BEGIN'] = site_data['LONG_MIDDLE'].values
        # site_data['LAT_BEGIN'] = site_data['LAT_MIDDLE'].values
        # site_data['LONG_END'] = site_data['LONG_MIDDLE'].values
        # site_data['LAT_END'] = site_data['LAT_MIDDLE'].values
        # flag_update_site_data_file = True # updated site data   
        
    # Check if segment lengths are given in the file; if not then assume lengths of 1 km and add to "site_data"
    # if not 'SHAPE_LENGTH' in site_data.columns:
        # site_data['SHAPE_LENGTH'] = [1]*len(site_data)
        # flag_update_site_data_file = True # updated site data
    
    # Loop through ParamsToGetFromDatasets and obtain data
    params_to_get = other_config_param['ParamsToGetFromDatasets']
    logging.info("Parameters to obtain from internal datasets:")
    # logging.info(f"\t{(', '.join(map(str,other_config_param['ParamsToGetFromDatasets'])))}")
    if len(params_to_get) == 0:
        logging.info("\tNumber of parameters = 0; skipping import of datasets")
    else:
        logging.info(f"\t{(', '.join(map(str,other_config_param['ParamsToGetFromDatasets'])))}")
        with open(other_config_param['File_AvailableDataset'], 'r') as f:
            available_data = json.load(f)

        # -----
        # specific to FrictionAngle, Cohesion, Thickness, and UnitWeight
        params_to_obtain_together = ['FrictionAngle', 'Cohesion', 'SlopeThickness', 'UnitWeight']
        params_to_get_specific = [param for param in params_to_get if param in params_to_obtain_together]
        params_to_get_specific = [param for param in params_to_get_specific if not param in dataset_meta_dict]
        param_info_specific = {}
        for param in params_to_get_specific:
            param_info_specific[param] = available_data['Parameters'][param]
        flag_params_to_obtain_together = False
        # -----
        
        for param in available_data['Parameters']:
            # See if site param with available data is requested
            if param in other_config_param['ParamsToGetFromDatasets']:
                # column to save information as
                column_name = available_data['Parameters'][param]['ColumnNameToStoreAs']
                
                # for FrictionAngle, Cohesion, Thickness, and UnitWeight
                if param in params_to_obtain_together:
                    if flag_params_to_obtain_together is False:
                       geo_unit_info = available_data['Parameters']['GeologicUnit']
                       site_data, dataset_meta_dict, appended_columns, flag_to_update = \
                           Fcn_Site.get_geologic_param(
                               site_data = site_data,
                               dataset_main_dir = other_config_param['Dir_Library'],
                               geo_unit_info = geo_unit_info,
                               geo_param_info = param_info_specific,
                               geo_params_to_get = params_to_get_specific,
                               csv_param_name = other_config_param['ColumnsFromParamCSVForKy'],
                               dataset_meta_dict = dataset_meta_dict,
                               flag_to_update = flag_params_to_obtain_together,
                           )
                       # logging.info(f"\t - {param}: {current_dataset['Source']}")
                       other_config_param['ColumnsAppended'] = other_config_param['ColumnsAppended'] + appended_columns
                       flag_update_site_data_file = flag_to_update
                       flag_params_to_obtain_together = True
                
                # for other parameters:
                else:
                    # If column with header ending with "_Existing" exists, then already performed
                    #   data search - don't perform search for the param again
                    # if column_name+'_Existing' in site_data:
                    if param in dataset_meta_dict:
                        pass
                    else:
                        # If column with header already exists, save it to another column
                        if column_name in site_data:
                            site_data[column_name+'_Existing'] = site_data[column_name].values
                            
                        # ------------------------------------------------------------------------------------
                        # To add feature to go through multiple datasets and get best data for each location
                        
                        # Loop through available datasets for current param until data is available
                        for dataset_i in available_data['Parameters'][param]['Datasets']:
                            current_dataset_info = available_data['Parameters'][param]['Datasets'][dataset_i]
                            # sample dataset for site param values
                            results = Fcn_Site.get_site_data(
                                param = column_name,
                                dataset_info = current_dataset_info,
                                dataset_main_dir = other_config_param['Dir_Library'],
                                x = site_data.loc[:,'Mid Longitude'].values,
                                y = site_data.loc[:,'Mid Latitude'].values
                            )
                            # see what is returned from the sampling process
                            if results is None:
                                pass
                            else:
                                site_data[column_name] = results
                                dataset_meta_dict[param] = current_dataset_info
                                dataset_meta_dict[param].update({'ColumnNameStoredAs': column_name})
                                if dataset_meta_dict[param]['Path'] is not None:
                                    dataset_meta_dict[param].update({
                                        'Path': '[OpenSRA_Dir]/lib/'+dataset_meta_dict[param]['Path']
                                    })
                                logging.info(f"\t - {param}: {current_dataset_info['Source']}")
                                other_config_param['ColumnsAppended'].append(column_name)
                                flag_update_site_data_file = True # updated site data
                                break
                        # ------------------------------------------------------------------------------------
    
        # For parameters without prepackaged datasets
        # logging.info(f"Requested parameters without internal datasets, default values to 0:")
        # for param in other_config_param['ParamsToGetFromDatasets']:
            # if not param in available_data['Parameters']:
                # site_data[column_name] = np.zeros(len(site_data))
                # dataset_meta_dict[param] = {
                    # "FileType": None,
                    # "Source": None,
                    # "Path": None,
                    # "UnitForData": "deg",
                    # "TargetUnit": "deg",
                    # "ConversionToOutputUnit": 1,
                    # "ColumnNameStoredAs": "Friction Angle (deg)"
                # }
    
    ## General site parameters: Vs30, slope, DEM
    #for param in other_config_param['SiteParams']:
    #    # See if site param with available datasets is requested
    #    if param in other_config_param['ParamsToGetFromDatasets']:
    #        column_name = other_config_param['SiteParams'][param]['ColumnNameToStoreAs']
    #        # If column with header ending with "_Existing" exists, then already performed
    #        #   data search - don't perform search for the param again
    #        if column_name+'_Existing' in site_data:
    #            pass
    #        else:
    #            # If column with header already exists, save it to another column
    #            if column_name in site_data:
    #                site_data[column_name+'_Existing'] = site_data[column_name].values
    #            # sample dataset for site param values
    #            path_to_raster = other_config_param['SiteParams'][param]['PathToRaster']
    #            if path_to_raster is None:
    #                if 'slope' in column_name.lower():
    #                    site_data[column_name] = np.zeros(len(site_data))*0.1
    #                elif 'vs30' in column_name.lower():
    #                    site_data[column_name] = np.ones(len(site_data))*999
    #            else:
    #                raster_interp_data = Fcn_Common.interp_from_raster(
    #                    raster_path = path_to_raster,
    #                    x = site_data.loc[:,'LONG_MIDDLE'].values,
    #                    y = site_data.loc[:,'LAT_MIDDLE'].values)
    #                if 'slope' in column_name.lower():
    #                    raster_interp_data[raster_interp_data<=0] = 0.1
    #                elif 'vs30' in column_name.lower():
    #                    raster_interp_data[raster_interp_data<=0] = 999
    #                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                    # OpenSHA sets a check for minimum Vs30
    #                    # if Vs30 sampled < 150, OpenSHA raises warning and ends
    #                    # opensha-core/src/org/opensha/sha/imr/attenRelImpl/ngaw2/NGAW2_WrapperFullParam.java
    #                    raster_interp_data[raster_interp_data<150] = 150
    #                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                site_data[column_name] = raster_interp_data
    #            other_config_param['ColumnsAppended'].append(column_name)
    #            flag_update_site_data_file = True # updated site data
    
    ## Liquefaction
    #if 'Liquefaction' in other_config_param['EDP']:
    #    # For Zhu et al. (2017)
    #    if 'ZhuEtal2017' in setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['Method']:
    #    # if 'ZhuEtal2017' in setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['Method'] and \
    #        # not other_config_param['ZhuEtal2017_param']['precip']['ColumnNameToStoreAs'] in site_data:
    #        # Loop through all params for Zhu et al. (2017)
    #        for param in other_config_param['ZhuEtal2017_param']:
    #            param_info = other_config_param['ZhuEtal2017_param'][param]
    #            column_name = param_info['ColumnNameToStoreAs']
    #            # If column with header ending with "_Existing" exists, then already performed
    #            #   data search - don't perform search for the param again
    #            if column_name in site_data:
    #                pass
    #            else:
    #                file = os.path.join(
    #                    other_config_param['Dir_ZhuEtal2017_param'],
    #                    param_info['FolderName'],
    #                    param_info['FolderName']+'.tif')
    #                site_data[column_name] = Fcn_Common.interp_from_raster(
    #                    raster_path = file,
    #                    x = site_data.loc[:,'LONG_MIDDLE'].values,
    #                    y = site_data.loc[:,'LAT_MIDDLE'].values)
    #                other_config_param['ColumnsAppended'].append(param_info['ColumnNameToStoreAs'])
    #                flag_update_site_data_file = True # updated site data
    
    ## Landslide
    ## Check if need to extract geologic units and properties, get the params if not
    #if 'Landslide' in other_config_param['EDP']:
    #    # check if any of the ky parameters are needed, if so, set flag to true
    #    flag_get_site_data = False
    #    for param in other_config_param['ColumnNamesForKy']:
    #        if param in other_config_param['ParamsToGetFromDatasets']:
    #            flag_get_site_data = True
    #            break
    #    # if not 'Geologic Unit (CGS, 2010)' in site_data.columns:
    #    if flag_get_site_data:
    #        # run script to get the ky properties
    #        site_data, other_config_param['ColumnsAppended'] = Fcn_Site.get_geologic_param(
    #            site_data = site_data, 
    #            geo_shp_file = other_config_param['Shp_GeologicUnit'],
    #            geo_unit_param_file = other_config_param['File_GeologicUnitParam'],
    #            param_to_get_from_data = other_config_param['ParamsToGetFromDatasets'],
    #            param_for_ky = other_config_param['ColumnNamesForKy'],
    #            csv_param_name = other_config_param['ColumnsFromParamCSVForKy'],
    #            appended_columns = other_config_param['ColumnsAppended'],
    #            flag_update_site_data_file = flag_update_site_data_file
    #        )
    #        # flag_update_site_data_file = True # updated site data
    
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
        
        # JSON with dataset metadata
        with open(other_config_param['Path_DatasetMetadata'], 'w') as f:
            json.dump(dataset_meta_dict, f, indent=4)
        logging.info(f"... Dataset meta information saved to:")
        logging.info(f"\t{other_config_param['Path_DatasetMetadata']}")
    else:
        logging.info(f"...Site data file is up-to-date")
    
    # Return site_data dataframe and flag of whether site_data has been updated
    return site_data, dataset_meta_dict

    
# -----------------------------------------------------------
def add_site_data_to_method_param(setup_config, site_data, other_config_param, method_param_for_assess, dataset_meta_dict):
    """
    Add parameters in site_data to method parameters in method_param_for_assess. Currently method dependent
    
    """
    
    #
    logging.info(f"Adding site data to list of model parameters in setup_config...")
    
    # Update column names for sourcing data
    for category in method_param_for_assess:
        for hazard in method_param_for_assess[category]:
            hazard_info = method_param_for_assess[category][hazard]
            for method in hazard_info['Method']:
                method_info = hazard_info['Method'][method]
                for param in method_info['InputParameters']:
                    if isinstance(method_info['InputParameters'][param],dict):
                        if method_info['InputParameters'][param]['Source'] == 'Preferred':
                            if param in dataset_meta_dict:
                                method_param_for_assess[category][hazard]['Method'][method]['InputParameters'][param]['ColumnIDWithData'] = \
                                    dataset_meta_dict[param]['ColumnNameStoredAs']
                            else:
                                method_param_for_assess[category][hazard]['Method'][method]['InputParameters'][param] = 'Preferred'
                        elif method_info['InputParameters'][param]['Source'] == 'UserDefined':
                            if param in dataset_meta_dict:
                                method_param_for_assess[category][hazard]['Method'][method]['InputParameters'][param]['ColumnIDWithData'] = \
                                    dataset_meta_dict[param]['ColumnNameStoredAs'] + '_Existing'
    
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
        if 'ZhuEtal2017' in setup_config['EngineeringDemandParameter']['Type'][hazard]['Method']:
            method = 'ZhuEtal2017'
            # Get values from site_data, set to list of NaN if doesn't exist
            #wtd = np.asarray(site_data.get('Water Table Depth (m)', [0]*len(site_data)))
            #dr = np.asarray(site_data.get('Distance to River (km)', [0]*len(site_data)))
            #dc = np.asarray(site_data.get('Distance to Coast (km)', [0]*len(site_data)))
            #dw = np.asarray(site_data.get('Distance to Water Body (km)', [0]*len(site_data)))
            #precip = np.asarray(site_data.get('Mean Annual Precipitation (mm)', [0]*len(site_data)))
            #vs30 = np.asarray(site_data.get('Vs30 (m/s)', [999]*len(site_data)))
            ## Find elements = find_val and set to set_val
            #wtd = set_elements_to_nan(wtd, find_val, 0)
            #dr = set_elements_to_nan(dr, find_val, 0)
            #dc = set_elements_to_nan(dc, find_val, 0)
            #dw = set_elements_to_nan(dw, find_val, 0)
            #precip = set_elements_to_nan(precip, find_val, 0)
            #vs30 = set_elements_to_nan(vs30, find_val, 999)
            #
            ## Temporary values for elevations
            #z = np.asarray(site_data.get('Elev (m)', [10]*len(site_data))) # set to 10 for now, get data from DEM map later
            ## Update method_param_for_assess with parameters
            ## setup_config['EngineeringDemandParameter']['Type']['Liquefaction']['OtherParameters'].update({
            #method_param_for_assess[category][hazard]['Method'][method]['InputParameters'].update({
            #    # 'wtd': list(wtd), 'dr': list(dr), 'dc': list(dc),
            #    # 'dw': list(dw), 'precip': list(precip), 'vs30': list(vs30), 'z': list(z),
            #    'WaterTableDepth': wtd,
            #    'DistanceToRiver': dr,
            #    'DistanceToCoast': dc,
            #    'DistanceToWaterBody': dw,
            #    'Precipitation': precip,
            #    'Vs30': vs30,
            #    'z': z,
            #})
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGA'] = False
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGV'] = True
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['mag'] = True

    # Lateral Spread
    if 'LateralSpread' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'LateralSpread'
    
        # For Hazus (2014)
        if 'GrantEtal2016' in setup_config['EngineeringDemandParameter']['Type'][hazard]['Method'] or \
            'Hazus2014' in setup_config['EngineeringDemandParameter']['Type'][hazard]['Method'] :
            method = 'GrantEtal2016'
            ## Get parameters from site_data
            #dwWB = np.asarray(site_data.get('Dist Any Water (km)', sub_value))
            #dwWB = set_elements_to_nan(dwWB, find_val, 999)
            #dwWB = dwWB*1000 # convert to meters
            ## Update method_param_for_assess with parameters
            ## setup_config['EngineeringDemandParameter']['Type']['LateralSpread']['OtherParameters'].update({
            #method_param_for_assess[category][hazard]['Method'][method]['InputParameters'].update({
            #    'DistanceToWaterBody':dwWB,
            #})
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGA'] = True
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGV'] = False
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['mag'] = True

    # Landslide
    if 'Landslide' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'Landslide'
        # Calculate Yield Acceleration, ky, for each method before passing into method_param_for_assess
        ky_dict = {}
        for method in method_param_for_assess[category][hazard]['Method']:
            method_info = method_param_for_assess[category][hazard]['Method'][method]
            # Start dictionary for input parameters
            kwargs = {}
            # Get ky input parameters
            for param in method_info['InputParameters']:
                param_info = method_info['InputParameters'][param]
                # check if item is dictionary
                if isinstance(method_info['InputParameters'][param], dict):
                    # specifically for Vs30
                    if param == 'Vs30' and param_info['Source'] == 'Preferred':
                        kwargs[param] = site_data['Vs30 (m/s)'].values
                    else:
                        kwargs[param] = site_data[param_info['ColumnIDWithData']].values
                        ## See if param is one of the parameters with pre-packaged data
                        #if param in other_config_param['ColumnNamesForKy']:
                        #    if method_info[param]['Source'] == 'Preferred': # if preferred data (obtained in OpenSRA)
                        #        if param in other_config_param['ColumnNamesForKy']:
                        #            if other_config_param['ColumnNamesForKy'][param] in site_data:
                        #                kwargs[param] = site_data[other_config_param['ColumnNamesForKy'][param]].values
                        #    elif method_info[param]['Source'] == 'UserDefined': # if user-defined
                        #        if param in other_config_param['ColumnNamesForKy']:
                        #            if other_config_param['ColumnNamesForKy'][param]+'_Existing' in site_data:
                        #                kwargs[param] = site_data[other_config_param['ColumnNamesForKy'][param]+'_Existing'].values
                        #            else:
                        #                if other_config_param['ColumnNamesForKy'][param] in site_data:
                        #                    kwargs[param] = site_data[other_config_param['ColumnNamesForKy'][param]].values
                        #else:
                        #    # see if ColumnIDWithData is None
                        #    if method_info[param]['ColumnIDWithData'] is None:
                        #        kwargs[param] = method_info[param]['Source']
                        #    else:
                        #        # get data from site_data file
                        #        kwargs[param] = site_data[method_info[param]['ColumnIDWithData']].values
                else:
                    kwargs[param] = param_info
                # also send n_site
                kwargs['n_site'] = other_config_param['Num_Sites']
            # Calculate and process ky
            ky = Fcn_EDP.get_ky(**kwargs)
            # if ky less than 0, set to 99
            ky[ky<0] = 99
            # store ky into method_param_for_assess
            method_param_for_assess[category][hazard]['Method'][method]['InputParameters'].update({
                'ky': ky
            })
    
        # For Bray & Macedo (2019)
        if 'BrayMacedo2019' in setup_config['EngineeringDemandParameter']['Type'][hazard]['Method']:
            method = 'BrayMacedo2019'
            # Update method_param_for_assess with parameters
            # setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters'].update({
            # method_param_for_assess[category][hazard]['Method'][method]['InputParameters'].update({
                # 'ky': np.asarray(site_data.get('Ky for Infinite Slope_Bray', sub_value)),
                # 'ky': site_data.get('Ky for Infinite Slope_Bray', sub_value).to_numpy(),
                # 'ky': ky_dict[method]
            # })
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGA'] = True
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGV'] = True
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['mag'] = True
        
        # For Bray & Macedo (2019)
        if 'Jibson2007' in setup_config['EngineeringDemandParameter']['Type'][hazard]['Method']:
            method = 'Jibson2007'
            # Update method_param_for_assess with parameters
            # setup_config['EngineeringDemandParameter']['Type']['Landslide']['OtherParameters'].update({
            # method_param_for_assess[category][hazard]['Method'][method]['InputParameters'].update({
                # 'ky': np.asarray(site_data.get('Ky for Infinite Slope_Bray', sub_value)),
                # 'ky': site_data.get('Ky for Infinite Slope_Grant', sub_value).to_numpy(),
                # 'ky': ky_dict[method]
            # })
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGA'] = True
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['PGV'] = False
            method_param_for_assess[category][hazard]['Method'][method]['Flags']['mag'] = True
        
    # Surface fault rupture
    if 'SurfaceFaultRupture' in other_config_param['EDP']:
        category = 'EDP'
        hazard = 'SurfaceFaultRupture'
    
        # For every method
        for method_i in setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['Method']:
            # Update method_param_for_assess with parameters
            # setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['OtherParameters'].update({
            method_param_for_assess[category][hazard]['Method'][method_i]['InputParameters'].update({
                'ComponentLength': np.asarray(site_data.get('Component Length (km)', sub_value)),
            })
            method_param_for_assess[category][hazard]['Method'][method_i]['Flags']['mag'] = True
    
    # Repair Rate PGD:
    if 'RepairRatePGV' in other_config_param['DV']:
        category = 'DV'
        hazard = 'RepairRatePGV'
        
        # For every method
        for method_i in setup_config['DecisionVariable']['Type'][hazard]['Method']:
            method_param_for_assess[category][hazard]['Method'][method_i]['Flags']['PGV'] = True
    
    # Repair Rate PGD:
    if 'RepairRatePGD' in other_config_param['DV']:
        category = 'DV'
        hazard = 'RepairRatePGD'
    
        # For every method
        for method_i in setup_config['DecisionVariable']['Type'][hazard]['Method']:
            # Update method_param_for_assess with parameters
            # setup_config['DecisionVariable']['Type']['RepairRatePGD']['OtherParameters'].update({
            method_param_for_assess[category][hazard]['Method'][method_i]['InputParameters'].update({
                'ComponentLength': np.asarray(site_data.get('Component Length (km)', sub_value)),
            })
            method_param_for_assess[category][hazard]['Method'][method_i]['Flags']['PGA'] = True
            method_param_for_assess[category][hazard]['Method'][method_i]['Flags']['mag'] = True
        
    #
    return method_param_for_assess
    
        
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
        np.savetxt(site_loc_file,site_data[['Mid Longitude','Mid Latitude']].values,fmt='%10.6f,%10.6f') # export vs30 to file for im_tool
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
        # elif other_config_param['SourceForVs30'] == 'UserDefined':
        # if 'Vs30 (m/s)' in site_data:
        np.savetxt(vs30_file,site_data['Vs30 (m/s)'].values,fmt='%6.2f') # export vs30 to file for im_tool
        logging.info("\tVs30 file exported to:")
        logging.info(f"\t\t{vs30_file}")
        # else:
            # logging.info("\t... Vs30 not provided as input, will infer from Wills et al. (2015)")
            # other_config_param['SourceForVs30'] = 'Wills et al. (2015)'
        # else:
            # logging.info("Source for Vs30 is invalid; currently supported sources:")
            # logging.info('\t"UserDefined" or "Wills et al. (2015)"')
        

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
        # logging.info(f"\tFlag for probability of surface fault rupture = {Flag_PSurf}")
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