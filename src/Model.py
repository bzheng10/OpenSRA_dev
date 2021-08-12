# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Risk model class
#
# Created: April 27, 2020
# Updated: July 14, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import importlib
# import sys
import os
import logging
import numpy as np
import pandas as pd
from scipy import sparse
# import json

# OpenSRA modules and functions
from src import Fcn_Common
# from src import Fcn_InputOutput
from src.IM import Fcn_IM, OpenSHAInterface


# -----------------------------------------------------------
class assessment(object):
    """
    Assess the damages and decision values at target sites using ground motion predictions and demand parameters.
    
    .. autosummary::
    
        get_IM_means
        sim_IM
        assess_EDP
        assess_DM
        assess_DV
        export_DV
    
    """
    

    # -----------------------------------------------------------
    def __init__(self):
        """
        Initialize class object.
        
        Parameters
        ----------
        
        Returns
        -------
    
        """
        # General setup information
        # self._setup_dict = None # store setup information such as directories
        
        # Site information
        # self._SITE_dict = None # site data
        
        # Source and GM 
        self._EVENT_dict = None # for rupture scenarios
        
        # PBEE workflow
        self._IM_dict = None # simulated intensity measures
        self._EDP_dict = None # assessed engineering demand parameters
        self._DM_dict = None # assessed damage measures
        self._DV_dict = None # assessed decision values/variables
    
        # results
        # self._SUMMARY = None # summary of results
    
    
    # -----------------------------------------------------------
    def get_IM_means(self, setup_config, other_config_param, site_data):
        """
        Get IM predictions from user-defined sources (e.g, OpenSHA, ShakeMap, User-Input) and stores in **_IM_dict**. Event information is stored in **_EVENT_dict**
        
        Parameters
        ----------
        setup_config : dictionary
            dictionary with configuration parameters from GUI
        other_config_param : dictionary
            dictionary with additional configuration parameters created in OpenSRA
        site_data : pandas.DataFrame
            table with user-defined site data
        
        Returns
        -------
    
        """

        # Initialize dictionary if it is none
        if self._EVENT_dict is None:
            self._EVENT_dict = {}
            
        if self._IM_dict is None:
            self._IM_dict = {}

        # Setup for _EVENT_dict
        self._EVENT_dict['Scenarios'] = {}
        event_keys = ['SourceIndex','RuptureIndex','Magnitude','MeanAnnualRate']

        # Initialize variables to be used
        trace_set = None

        # Check if files are present before interfacing with IM sources
        exist_ListOfScenarios = False
        exist_Predictions = False
        exist_FaultCrossings = False
        # Check if file with list of ruptures exists, if so, load and store into _EVENT_dict
        if os.path.exists(other_config_param['File_ListOfScenarios']):
            logging.info(f"File with list of rupture scenarios exist in:")
            logging.info(f"\t{other_config_param['Dir_IM']}")
            exist_ListOfScenarios = True
            rupture_list = OpenSHAInterface.get_eq_rup_meta(
                erf=None, rupture_list_file=other_config_param['File_ListOfScenarios'], ind_range=['all'])
            for key in event_keys:
                self._EVENT_dict['Scenarios'][key] = rupture_list[key].values
            self._EVENT_dict['Scenarios']['Num_Events'] = len(self._EVENT_dict['Scenarios']['SourceIndex'])

        # Check if files with predictions already exist
        for im in other_config_param['IM']:
            list_of_files = os.listdir(os.path.join(other_config_param['Dir_IM_GroundMotion_Prediction'],im))
            if 'Median.txt' in list_of_files:
                if 'TotalStdDev.txt' in list_of_files or \
                    ('InterEvStdDev.txt' in list_of_files and 'IntraEvStdDev.txt' in list_of_files):
                    exist_Predictions = True
        # Load predictions
        if exist_Predictions:
            logging.info(f"IM predictions already exist under:")
            logging.info(f"\t{other_config_param['Dir_IM_GroundMotion_Prediction']}")
            self._IM_dict.update({
                'Prediction': Fcn_IM.read_IM_means(
                    im_pred_dir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                    list_im = other_config_param['IM'],
                    list_param = other_config_param['ListOfIMParams'],
                    # store_file_type = setup_config['General']['OutputFileType']
                    store_file_type = 'txt'
                )
            })

        # Get predictions and rupture lists from OpenSHA
        if exist_ListOfScenarios is False or exist_Predictions is False:
            # Setup - general
            self._EVENT_dict['Platform'] = setup_config['IntensityMeasure']['SourceForIM']
            
            # Interface with OpenSHA
            if setup_config['IntensityMeasure']['SourceForIM'] == 'OpenSHA':
                logging.info(f"\n------------------------------------------------------\n-----Interfacing with OpenSHA for GMs\n")
                # Setup up OpenSHA
                # erf, imr, sites = OpenSHAInterface.setup_opensha(setup_config, other_config_param, site_data)
                erf, imr, sites = OpenSHAInterface.setup_opensha(setup_config, other_config_param, site_data)
                self._EVENT_dict['RuptureForecast'] = {
                    'ModelName': setup_config['IntensityMeasure']['SourceParameters']['SeismicSourceModel'],
                    'JavaInstance': erf
                }
                self._EVENT_dict['GroundMotion'] = {
                    'ModelName': setup_config['IntensityMeasure']['SourceParameters']['GroundMotionModel'],
                    'JavaInstance': imr
                }
                self._EVENT_dict['SiteParamForModel'] = sites
                
                # Get list of ruptures
                if exist_ListOfScenarios is False:
                    # Get list with rate cutoff and distance filters
                    logging.info(f"Getting full list of rupture scenarios")
                    rupture_list = pd.read_csv(other_config_param['Path_RuptureMetadata'])
                    # Filter list of ruptures
                    filters = setup_config['IntensityMeasure']['SourceParameters']['Filter']
                    filters_to_perform = {}
                    for item in filters.keys():
                        if filters[item]['ToInclude']:
                            filters_to_perform[item] = filters[item]
                    logging.info(f"\t... list of filters to perform: {list(filters_to_perform)}")
                    rupture_list = OpenSHAInterface.filter_ruptures(
                        erf = erf,
                        locs = site_data[['Mid Longitude','Mid Latitude']].values,
                        filter_criteria = filters_to_perform,
                        rupture_list = rupture_list,
                        rup_save_name = other_config_param['File_ListOfScenarios'],
                        # trace_save_name = other_config_param['File_ScenarioTraces'],
                        rup_seg_file = other_config_param['Path_RuptureSegment'],
                        pt_src_file = other_config_param['Path_PointSource'])
                    # Store into _EVENT_dict
                    for key in event_keys:
                        if np.ndim(rupture_list[key]) < 1:
                            self._EVENT_dict['Scenarios'][key] = np.expand_dims(rupture_list[key],axis=0)
                        else:
                            self._EVENT_dict['Scenarios'][key] = rupture_list[key].values
                    # Get list of traces for each source scenario
                    trace_set = OpenSHAInterface.get_trace_opensha(
                        src_list = np.unique(rupture_list['SourceIndex']),
                        finite_src_file = other_config_param['Path_FiniteSource'],
                        pt_src_file = other_config_param['Path_PointSource'],
                        save_name = other_config_param['File_ScenarioTraces'],
                        flag_include_pt_src = other_config_param['Flag_IncludeFilter_PtSrc']
                    )
                
                # Get IM predictions
                if exist_Predictions is False:
                    logging.info(f"Looping through ruptures and get IM predictions")
                    self._IM_dict['Prediction'] = OpenSHAInterface.get_IM(
                        erf = erf,
                        imr = imr,
                        sites = sites,
                        src_list = rupture_list['SourceIndex'],
                        rup_list = rupture_list['RuptureIndex'],
                        list_im = other_config_param['IM'],
                        saveDir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                        # store_file_type = setup_config['General']['OutputFileType']
                        store_file_type = 'txt'
                    )
                    logging.info(f"... generated predictions for ruptures; results stored under:")
                    logging.info(f"\t{other_config_param['Dir_IM_GroundMotion_Prediction']}")
        
                logging.info(f"\n\n-----Interfacing with OpenSHA for GMs\n------------------------------------------------------")
                
            elif setup_config['IntensityMeasure']['SourceForIM'] == 'ShakeMap':
                logging.info(f"\n------------------------------------------------------\n-----Getting GMs from ShakeMap\n")
                #
                Fcn_IM.read_ShakeMap_data(
                    sm_dir = other_config_param['Dir_ShakeMap'],
                    event_names = other_config_param['ShakeMapEvents'],
                    sites = site_data[['Mid Longitude','Mid Latitude']].values,
                    IM_dir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                    store_events_file = other_config_param['File_ListOfScenarios'],
                    trace_save_name = other_config_param.get('File_ScenarioTraces',None)
                )
                #
                self._IM_dict.update({
                    'Prediction': Fcn_IM.read_IM_means(
                        im_pred_dir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                        list_im = other_config_param['IM'],
                        list_param = other_config_param['ListOfIMParams'],
                        # store_file_type = setup_config['General']['OutputFileType']
                        store_file_type = 'txt'
                    )
                })
                #
                rupture_list = OpenSHAInterface.get_eq_rup_meta(
                    erf=None, rupture_list_file=other_config_param['File_ListOfScenarios'])
                for key in event_keys:
                    if np.ndim(rupture_list[key]) < 1:
                        self._EVENT_dict['Scenarios'][key] = np.expand_dims(rupture_list[key],axis=0)
                    else:
                        self._EVENT_dict['Scenarios'][key] = rupture_list[key]
                self._EVENT_dict['Scenarios']['Num_Events'] = len(self._EVENT_dict['Scenarios']['SourceIndex'])
                logging.info(f"\n\n-----Getting GMs from ShakeMap\n------------------------------------------------------")
        
        #
        logging.info(f'Added listOfRuptures to "model._EVENT_dict" and IM Means and StdDevs to "model._IM_dict"\n')
        
        # Get Fault Crossings
        if setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['ToAssess']:
            logging.info(f"SurfaceFaultRupture is requested; getting fault crossings...")
            if os.path.exists(other_config_param['File_FaultCrossing']):
                logging.info(f"\tFile with fault crossings exists:")
                logging.info(f"\t\t{other_config_param['File_FaultCrossing']}")
                # exist_FaultCrossings = True
                self._EVENT_dict['FaultCrossings'] = OpenSHAInterface.get_fault_xing_opensha(
                    src_list = None,
                    start_loc = None,
                    end_loc = None,
                    trace_set = None,
                    save_name = other_config_param['File_FaultCrossing'],
                    to_write = False,
                )
            else:
                # exist_FaultCrossings is False:
                logging.info(f"\tFile with fault crossings does not exist, computing now...")
                if trace_set is None:
                    # Get list of traces for each source scenario
                    trace_set = OpenSHAInterface.get_trace_opensha(
                        src_list = None,
                        finite_src_file = None,
                        pt_src_file = None,
                        save_name = other_config_param['File_ScenarioTraces'],
                        to_write = False,
                        flag_include_pt_src = False
                    )
                self._EVENT_dict['FaultCrossings'] = OpenSHAInterface.get_fault_xing_opensha(
                    src_list = np.unique(rupture_list['SourceIndex']),
                    start_loc = site_data[['Start Longitude','Start Latitude']].values,
                    end_loc = site_data[['End Longitude','End Latitude']].values,
                    trace_set = trace_set,
                    save_name = other_config_param['File_FaultCrossing']
                )
        
        
    # -----------------------------------------------------------
    def sim_IM(self, setup_config, other_config_param, site_data):
        """
        Perform multivariate random sampling of **PGA** and **PGV** using means and sigmas for all scenarios. Spatial and spectral orrelations can be applied.
    
        Parameters
        ----------
        n_samp_im : float
            number of samples/realizations for intensity measures
        ims : str, list, optional
            list of **IM** variables to create: **pga** or **pgv** (default = both)
        flag_spatial_corr : boolean, optional
            decision on performing correlation between sites (distance); default = True
        flag_cross_corr : boolean, optional
            decision on performing correlation between periods; default = True (only if **ims** contains **pga** and **pgv**)
        T_pga : float, optional
            [sec] approximate period for **PGA** to be used in spectral correlation; default = 0.01 sec
        T_pgv : float, optional
            [sec] approximate period for **PGV** to be used in spectral correlation; default = 1.0 sec
        method_d = str, optional
            method to use for spatial correlation; default = jayaram_baker_2009
        method_T = str, optional
            method to use for spectral correlation; default = baker_jayaram_2008
        flag_clear_dict : boolean, optional
            **True** to clear the calculated demands or **False** to keep them; default = **False**
        flag_im_sample_exist : boolean, optional
            **True** if IM samples are available or **False** to import statistical parameters and create random variables; default = **False**
        sample_dir : str, optional
            directory with samples; default = None
        store_file_type : str, optional
            file type used to store the GM predictions (**npz** or **txt**), default = **npz**
        
        
        For additional parameters, see the respective methods under :func:`im.corr_spatial.py` and :func:`im.corr_spectral.py`
    
        Returns
        -------
        _IM_dict : dict
            stored samples for all intensity measures and rupture scenarios
        
        """
    
        # setup
        self._IM_dict['Simulation'] = {'Correlation':{}, 'Sample':{}}
        n_site = other_config_param['Num_Sites']
        n_event = other_config_param['Num_Events']
        n_IM = other_config_param['Num_IM']
        sampling_method = setup_config['SamplingMethod']['Method']
        if 'monte' in sampling_method.lower():
            algorithm = setup_config['SamplingMethod']['Algorithm']
            n_sample = setup_config['SamplingMethod']['NumberOfSamples']
            seed_num = setup_config['SamplingMethod']['Seed']
        elif 'logic' in sampling_method.lower():
            algorithm = None
            logic_branch = setup_config['SamplingMethod']['Branch']
            logic_branch_weights = setup_config['SamplingMethod']['Weights']
            n_sample = len(logic_branch)
        elif 'polynomial' in sampling_method.lower():
            logging.info("Polynomial chaos under development: restart analysis with another method")
        # store_file_type = setup_config['General']['OutputFileType']
        store_file_type = 'txt'
        n_decimals = other_config_param['Num_Decimals']
        
        # Check if files are present before interfacing with IM sources
        exist_IMSamples = True
        # If requested to resample IMs, then set to False
        if other_config_param['Flag_ResampleIM']:
            exist_IMSamples = False
        # Check if files with samples already exist
        for im in other_config_param['IM']:
            if len(os.listdir(os.path.join(other_config_param['Dir_IM_GroundMotion_Simulation'],im))) < n_sample:
                exist_IMSamples = False
                break
        
        # if samples exist, import, if not, sample
        if exist_IMSamples:
            logging.info(f"IM samples already exist under:")
            logging.info(f"\t{other_config_param['Dir_IM_GroundMotion_Simulation']}")
            for im_i in other_config_param['IM']:
                self._IM_dict['Simulation']['Sample'][im_i] = {}
                for j in range(n_sample):
                    file_name = os.path.join(
                        other_config_param['Dir_IM_GroundMotion_Simulation'],
                        im_i, 'Sample_'+str(j)+'.'+store_file_type)
                    self._IM_dict['Simulation']['Sample'][im_i][j] = Fcn_IM.read_IM_sample(
                        file = file_name,
                        store_file_type = store_file_type,
                        n_event = n_event,
                        n_site = n_site
                    )
            
        else:
            # import correlation procedures to be applied
            proc_for_corr = {}
            for corr_type in ['Spatial', 'Spectral']:
                if setup_config['IntensityMeasure']['Correlation'][corr_type]['ToInclude']:
                    proc_for_corr[corr_type] = getattr(
                        importlib.import_module('src.IM.'+corr_type+'Correlation'),
                        setup_config['IntensityMeasure']['Correlation'][corr_type]['Method']
                    )
                    self._IM_dict['Simulation']['Correlation'].update({
                        corr_type:{
                            'Method': setup_config['IntensityMeasure']['Correlation'][corr_type]['Method'],
                            'ProcInstance': proc_for_corr[corr_type]
                        }
                    })
                else:
                    proc_for_corr[corr_type] = None
                    self._IM_dict['Simulation']['Correlation'].update({
                        corr_type:{
                            'Method': None,
                            'ProcInstance': None
                        }
                    })
            
            # compute intra-event spectral correlation between PGA and PGV
            self._IM_dict['Simulation']['Correlation']['Spectral']['Matrix'] = None
            self._IM_dict['Simulation']['Correlation']['Spectral']['Cholesky'] = None
            logging.info(f"Computing spatial and spectral correlations...")
            if other_config_param['Num_IM'] <= 1:
                logging.info(f"\tSpectral (cross) correlation is only applicable for 2 or more IMs:")
            else:
                if proc_for_corr['Spectral'] is None:
                    logging.info(f"\tSpectral (cross) correlation not requested; correlation between IMs set to 0")
                    self._IM_dict['Simulation']['Correlation']['Spectral']['Matrix'] = np.eye(n_IM)
                    self._IM_dict['Simulation']['Correlation']['Spectral']['Cholesky'] = sparse.eye(n_IM)
                else:
                    logging.info(f"\tSpectral (cross) correlation:")
                    if self._IM_dict['Simulation']['Correlation']['Spectral']['Method'] == 'BakerJayaram2008':
                        logging.info(f"\t\tmethod = {self._IM_dict['Simulation']['Correlation']['Spectral']['Method']}")
                        logging.info(f"\t\tperiods to assess with = {other_config_param['ApproxPeriod']}")
                        corr_val = proc_for_corr['Spectral'](
                            T1=other_config_param['ApproxPeriod']['PGA'],
                            T2=other_config_param['ApproxPeriod']['PGV'])
                        corr_mat = np.array([[1,corr_val],[corr_val,1]])
                        self._IM_dict['Simulation']['Correlation']['Spectral']['Matrix'] = corr_mat
                        self._IM_dict['Simulation']['Correlation']['Spectral']['Cholesky'] = \
                            sparse.coo_matrix(np.linalg.cholesky(corr_mat))
                    else:
                        logging.info(f"\tinvalid method requested; correlation between IMs set to 0")
                        self._IM_dict['Simulation']['Correlation']['Spectral']['Matrix'] = np.eye(n_IM)
                        self._IM_dict['Simulation']['Correlation']['Spectral']['Cholesky'] = sparse.eye(n_IM)
                        

            # compute inter-event spatial correlations between sites
            self._IM_dict['Simulation']['Correlation']['Spatial']['Matrix'] = {}
            self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'] = {}
            self._IM_dict['Simulation']['Correlation']['Spatial']['CheckForUniqueSites'] = {'FlagRepeatingSites': False}
            n_site_spatial = n_site
            site_lon_spatial = site_data['Mid Longitude'].values
            site_lat_spatial = site_data['Mid Latitude'].values
            if proc_for_corr['Spatial'] is None:
                logging.info(f"\tSpatial correlation not requested; correlation between sites set to 0")
                for im_i in other_config_param['IM']:
                    self._IM_dict['Simulation']['Correlation']['Spatial']['Matrix'].update({im_i: np.eye(n_site_spatial)})
                    self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'].update({im_i: sparse.eye(n_site_spatial)})
            else:
                logging.info(f"\tSpatial correlation:")
                if self._IM_dict['Simulation']['Correlation']['Spatial']['Method'] == 'JayaramBaker2009':
                    logging.info(f"\t\tmethod = {self._IM_dict['Simulation']['Correlation']['Spatial']['Method']}")
                    # -----------------------------------------------------------
                    # working procedure to calculate spatial correlation matrix
                    # site_lon = site_data['LONG_MIDDLE'].values
                    # site_lat = site_data['LAT_MIDDLE'].values
                    
                    # # -----------------------------------------------------------
                    # working section to get spatial correlations for unique sites only
                    #   keep matrix positive definite for Choleksy decomposition
                    site_mat = np.vstack([site_lon_spatial,site_lat_spatial]).T
                    site_mat_unique, ind_rev = np.unique(site_mat, axis=0, return_inverse=True)
                    n_site_unique = site_mat_unique.shape[0]
                    
                    # store check for repeating sites
                    if n_site_unique < n_site:
                        self._IM_dict['Simulation']['Correlation']['Spatial']['CheckForUniqueSites'].update({
                            'FlagRepeatingSites': True,
                            'UniqueSites': site_mat_unique,
                            'IndicesForReconstruction': ind_rev
                        })
                        n_site_spatial = n_site_unique
                        site_lon_spatial = site_mat_unique[:,0]
                        site_lat_spatial = site_mat_unique[:,1]
                    
                    # get distances between sites for upper triangle
                    ind1,ind2 = Fcn_Common.fast_triu_indices(n_site_spatial)
                    d = Fcn_Common.get_haversine_dist(
                        site_lon_spatial[ind1], site_lat_spatial[ind1], 
                        site_lon_spatial[ind2], site_lat_spatial[ind2])
                    # compute intra-event correlations between sites for IMs
                    self._IM_dict['Simulation']['Correlation']['Spatial']['Value'] = {}
                    for i in range(other_config_param['Num_IM']):
                        corr_val = proc_for_corr['Spatial'](d=d, T=other_config_param['ApproxPeriod'][other_config_param['IM'][i]])
                        corr_mat = Fcn_Common.convert_triu_to_sym_mat(corr_val,n_site_spatial)
                        self._IM_dict['Simulation']['Correlation']['Spatial']['Matrix'].update({
                            other_config_param['IM'][i]: corr_mat})
                        self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'].update({
                            other_config_param['IM'][i]: sparse.coo_matrix(np.linalg.cholesky(corr_mat.toarray()))})
                    # # -----------------------------------------------------------
                        
                    # -----------------------------------------------------------
                else:
                    logging.info(f"\t\tinvalid method requested; correlation between sites set to 0")
                    for im_i in other_config_param['IM']:
                        self._IM_dict['Simulation']['Correlation']['Spatial']['Matrix'].update({im_i: np.eye(n_site_spatial)})
                        self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'].update({im_i: sparse.eye(n_site_spatial)})
            
            # get flag for sampling with total stdevs
            Flag_SampleWithStdDevTotal = other_config_param['Flag_SampleWithStdDevTotal']
            
            # check if inter and intra event stdevs are available, if not, default to total stdev
            for im_i in other_config_param['IM']:
                if self._IM_dict['Prediction'][im_i]['InterEvStdDev'] is None or \
                    self._IM_dict['Prediction'][im_i]['IntraEvStdDev'] is None:
                # if not 'InterEvStdDev' in self._IM_dict['Prediction'][im_i].keys() or \
                    # not 'IntraEvStdDev' in self._IM_dict['Prediction'][im_i].keys():
                    Flag_SampleWithStdDevTotal = True
                    break
            
            # perform sampling
            logging.info(f"Performing sampling of IMs...")
            if Flag_SampleWithStdDevTotal: # for total stdevs
                # get residuals for first IM, then apply IM correlation and sample the next IM
                Flag_UseUniformStdDev = other_config_param['Flag_UseUniformStdDev']
                if self._IM_dict['Prediction'][im_i]['TotalStdDev'] is None:
                    Flag_UseUniformStdDev = True
                #
                residuals = {}
                spectral_corr = self._IM_dict['Simulation']['Correlation']['Spectral']['Matrix']
                for im_counter, im_i in enumerate(other_config_param['IM']):
                    # get residuals and then sample
                    if 'logic' in sampling_method.lower():
                        residuals[im_i] = np.ones((n_sample,n_event,n_site))
                        for j in range(n_sample):
                            residuals[im_i][j,:,:] = logic_branch[j]
                    elif 'monte' in sampling_method.lower():
                        # get spatially-correlated residuals for first IM
                        if im_counter == 0:
                            residuals[im_i] = Fcn_IM.get_correlated_residuals(
                                chol_mat = self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'][im_i],
                                n_sample = n_sample,
                                n_event = n_event,
                                dim3 = n_site_spatial,
                                algorithm = algorithm
                            )
                            prev_im = im_i
                        # get spatially-correlated residuals of second IM conditioned on first IM
                        else:
                            residuals[im_i] = Fcn_IM.get_correlated_residuals(
                                chol_mat = self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'][im_i],
                                n_sample = n_sample,
                                n_event = n_event,
                                dim3 = n_site_spatial,
                                algorithm = algorithm,
                                cross_corr = spectral_corr[im_counter-1,im_counter],
                                prev_residuals = residuals[prev_im]
                            )
                            
                            # # -----------------------------------------------------------
                            # reconstruct full matrix given repeating sites
                            if self._IM_dict['Simulation']['Correlation']['Spatial']['CheckForUniqueSites']['FlagRepeatingSites']:
                                ind_rev = self._IM_dict['Simulation']['Correlation']['Spatial']['CheckForUniqueSites']['IndicesForReconstruction']
                                residuals_full = np.zeros((residuals[im_i].shape[0],residuals[im_i].shape[1],len(ind_rev)))
                                for i in range(len(ind_rev)):
                                    residuals_full[:,:,i] = residuals[im_i][:,:,ind_rev[i]]
                                residuals[im_i] = residuals_full
                            # # -----------------------------------------------------------
                            
                    # set up and get distributions
                    self._IM_dict['Simulation']['Sample'][im_i] = {}
                    median = self._IM_dict['Prediction'][im_i]['Median']
                    if Flag_UseUniformStdDev:
                        stdev_total = sparse.coo_matrix(np.ones((n_event,n_site))*other_config_param['UniformStdDev'])
                    else:
                        stdev_total = self._IM_dict['Prediction'][im_i]['TotalStdDev']
                    # loop through each sample
                    for j in range(n_sample):
                        sample_j = stdev_total.multiply(residuals[im_i][j])
                        sample_j = median.multiply(sample_j.expm1() + np.ones((n_event,n_site)))
                        self._IM_dict['Simulation']['Sample'][im_i][j] = sample_j
                logging.info(f"\tPerformed sampling using total StdDev")
            else: # for inter and intra stdevs
                # get spatially-correlated residuals for each IM
                eps_intra = {}
                for im_i in other_config_param['IM']:
                    if 'logic' in sampling_method.lower():
                        eps_intra[im_i] = np.ones((n_sample,n_event,n_site))
                        for j in range(n_sample):
                            eps_intra[im_i][j,:,:] = logic_branch[j]
                    elif 'monte' in sampling_method.lower():
                        # get spatially-correlated residuals for each IM, cross-correlation is considered in eta
                        eps_intra[im_i] = Fcn_IM.get_correlated_residuals(
                            chol_mat = self._IM_dict['Simulation']['Correlation']['Spatial']['Cholesky'][im_i],
                            n_sample = n_sample,
                            n_event = n_event,
                            dim3 = n_site_spatial,
                            algorithm = algorithm
                        )
                        
                        # # -----------------------------------------------------------
                        # reconstruct full matrix given repeating sites
                        if self._IM_dict['Simulation']['Correlation']['Spatial']['CheckForUniqueSites']['FlagRepeatingSites']:
                            ind_rev = self._IM_dict['Simulation']['Correlation']['Spatial']['CheckForUniqueSites']['IndicesForReconstruction']
                            residuals_full = np.zeros((eps_intra[im_i].shape[0],eps_intra[im_i].shape[1],len(ind_rev)))
                            for i in range(len(ind_rev)):
                                residuals_full[:,:,i] = eps_intra[im_i][:,:,ind_rev[i]]
                            eps_intra[im_i] = residuals_full
                        # # -----------------------------------------------------------
                        
                # get spectrally-correlated residuals for each event
                if 'logic' in sampling_method.lower():
                    eta_inter = np.ones((n_sample,n_event,n_IM))
                    for j in range(n_sample):
                        eta_inter[j,:,:] = logic_branch[j]
                elif 'monte' in sampling_method.lower():
                    # get spatially-correlated residuals for each IM, cross-correlation is considered in eta
                    eta_inter = Fcn_IM.get_correlated_residuals(
                        chol_mat = self._IM_dict['Simulation']['Correlation']['Spectral']['Cholesky'],
                        n_sample = n_sample,
                        n_event = n_event,
                        dim3 = n_IM,
                        algorithm = algorithm
                    )
                # get samples
                for im_counter, im_i in enumerate(other_config_param['IM']):
                    # set up and get distributions
                    self._IM_dict['Simulation']['Sample'][im_i] = {}
                    median = self._IM_dict['Prediction'][im_i]['Median']
                    stdev_intra = self._IM_dict['Prediction'][im_i]['IntraEvStdDev']
                    stdev_inter = self._IM_dict['Prediction'][im_i]['InterEvStdDev']
                    # loop through each sample
                    for j in range(n_sample):
                        sample_j = stdev_intra.multiply(eps_intra[im_i][j]) + \
                            stdev_inter.multiply(np.tile(eta_inter[j,:,im_counter],[n_site,1]).T)
                        sample_j = median.multiply(sample_j.expm1() + np.ones((n_event,n_site)))
                        self._IM_dict['Simulation']['Sample'][im_i][j] = sample_j
                logging.info(f"\tPerformed sampling using Inter- and IntraStdDevs")

            # export to files
            for im_i in other_config_param['IM']:
                for j in range(n_sample):
                    sample = self._IM_dict['Simulation']['Sample'][im_i][j]
                    save_name = os.path.join(
                        other_config_param['Dir_IM_GroundMotion_Simulation'],
                        im_i, 'Sample_'+str(j)+'.'+store_file_type)
                    Fcn_IM.store_IM_sample(save_name, sample, store_file_type, n_decimals)
            logging.info(f"IM samples stored under:")
            logging.info(f"\t{other_config_param['Dir_IM_GroundMotion_Simulation']}")
        
        # Empty _IM_dict['Prediction'] dictionary to clear up memoryview
        for key in self._IM_dict['Prediction'].keys():
            self._IM_dict['Prediction'][key] = {}
        
        #
        logging.info(f'Added IM simulations to "model._IM_dict\n')
    
    # -----------------------------------------------------------
    def assess_EDP(self, setup_config, other_config_param, site_data, method_param_for_assess):
        """
        Using the simulated intensity measures to calculate engineering demand parameters.

        Parameters
        ----------
        edp_category : str
            demand category to calculate; options are **corr_spt**, **liq**, **ls**, **gs**, etc. (see :func:`edp` for all **EDP** categories)
        method : str
            method/procedure to use to calculate the demand; see :func:`edp` for available methods
        return_param : str, list
            single of a list of parameters to return, see the return variables under each function (:func:`edp`)
        store_name : str, list, optional
            names to store parameter as; default = **return_param**
        flag_clear_dict : boolean, optional
            **True** to clear the calculated demands or **False** to keep them; default = **False**
        flag_pga : boolean, optional
            **True** include simulated **PGA**; default = **False**
        flag_pgv : boolean, optional
            **True** include simulated **PGV**; default = **False**
        flag_mag : boolean, optional
            **True** include moment magnitude **mag** from rupture metadata; default = **False**
        flag_rup_depend : boolean, optional
            **True** if dependent on rupture scenario; default = **False**
        source_dict : str, list, optional
            dictionary that contains **source_param** and **source_method**; default = None
        source_param : str, list, optional
            parameter to get from **existing** stored parameters (e.g., **liq_susc**); default = None
        source_method : str, list, optional
            method used to obtain the source_param (e.g., **zhu_etal_2017**); default = None

        For input parameters to each method, refer to the method documentation under :func:`edp`.

        Returns
        -------
        output : varies
          [varies] output depends on the target demand and methods.

        """

        # Initialize dictionary if it is none
        if self._EDP_dict is None:
            self._EDP_dict = {}
        
        # setup
        n_site = other_config_param['Num_Sites']
        n_event = other_config_param['Num_Events']
        n_EDP = other_config_param['Num_EDP']
        sampling_method = setup_config['SamplingMethod']['Method']
        if 'monte' in sampling_method.lower():
            algorithm = setup_config['SamplingMethod']['Algorithm']
            n_sample = setup_config['SamplingMethod']['NumberOfSamples']
            seed_num = setup_config['SamplingMethod']['Seed']
        elif 'logic' in sampling_method.lower():
            algorithm = None
            logic_branch = setup_config['SamplingMethod']['Branch']
            logic_branch_weights = setup_config['SamplingMethod']['Weights']
            n_sample = len(logic_branch)
        elif 'polynomial' in sampling_method.lower():
            logging.info("Polynomial chaos under development: restart analysis with another method")
        # store_file_type = setup_config['General']['OutputFileType']
        store_file_type = 'txt'
        n_decimals = other_config_param['Num_Decimals']
        
        # set up kwargs to pass inputs into method function
        kwargs = {}
        kwargs['n_site'] = n_site
        kwargs['n_event'] = n_event
        kwargs['n_sample'] = n_sample
        
        # get current PEER category info from model_assess_param
        curr_peer_category = method_param_for_assess['EDP']
        
        # dictionary for parameters computed in run
        param_computed_in_run_to_add = {}
        
        # loop through list of EDPs=
        temp_out = {}
        for edp_counter, edp_i in enumerate(curr_peer_category):
            logging.info(f"Current EDP: {edp_i}")
            edp_info = curr_peer_category[edp_i]
            # initialize storage for hazard
            self._EDP_dict[edp_i] = {}
            
            # pull method params from model
            method = list(edp_info['Method'].keys())
            return_param = edp_info['ReturnParameter']
            dist_type = edp_info['Uncertainty']['DistributionType']
            epistemic_branch = edp_info['Uncertainty']['Epistemic']['Branch']
            epistemic_weight = edp_info['Uncertainty']['Epistemic']['Weight']
            
            # use return params as keys for output dictionary
            for param_j in return_param:
                self._EDP_dict[edp_i][param_j] = {'Value': {}}
                
            # set up kwargs to pass inputs into method function
            kwargs['return_param'] = return_param
            
            # dictionary for parameters computed in run
            param_computed_in_run_to_add[edp_i] = {}
            # for LateralSpread and GroundSettlement, add liq_susc
            if edp_i == 'LateralSpread' or edp_i == 'GroundSettlement':
                param_computed_in_run_to_add[edp_i].update({
                    'liq_susc': self._EDP_dict['Liquefaction']['liq_susc']['Value']
                })
            # for SurfaceFaultRupture, add list of fault crossings
            if edp_i == 'SurfaceFaultRupture':
                param_computed_in_run_to_add[edp_i].update({
                    'fault_crossings': self._EVENT_dict['FaultCrossings']
                })
            # add uncertainty information
            if dist_type == 'Uniform':
                self._EDP_dict[edp_i]['Uncertainty'] = {
                    'DistributionType': dist_type,
                    'Aleatory': {
                        'ScaleFactor': 1,
                    },
                    'Epistemic': {
                        'ScaleFactor': 1,
                        'Branches': epistemic_branch,
                        'Weights': epistemic_weight,
                    }
                }
            elif dist_type == 'Lognormal':
                self._EDP_dict[edp_i]['Uncertainty'] = {
                    'DistributionType': dist_type,
                    'Aleatory': {
                        'StdDev': 0,
                    },
                    'Epistemic': {
                        'StdDev': 0,
                        'Branches': epistemic_branch,
                        'Weights': epistemic_weight,
                    },
                    'Total': {
                        'StdDev': 0,
                    }
                }
            
            # loop through list of methods for hazard
            for method_counter, method_j in enumerate(method):
                logging.info(f"\tCurrent method: {method_j}")
                method_info = edp_info['Method'][method_j]
                # add mag to kwargs
                if method_info['Flags']['mag'] is True:
                    kwargs['M'] = self._EVENT_dict['Scenarios']['Magnitude']
                # add IM simulations to kwargs
                if method_info['Flags']['PGA'] is True:
                    kwargs['pga'] = self._IM_dict['Simulation']['Sample']['PGA']
                if method_info['Flags']['PGV'] is True:
                    kwargs['pgv'] = self._IM_dict['Simulation']['Sample']['PGV']
                # add additional parameters computed in this run into kwargs
                for key in param_computed_in_run_to_add[edp_i]:
                    kwargs[key] = param_computed_in_run_to_add[edp_i][key]
                # add method specific input parameters to kwargs
                for param_k in method_info['InputParameters']:
                    param_info = method_info['InputParameters'][param_k]
                    if isinstance(param_info, dict):
                        # specifically for Vs30         
                        if param_k == 'Vs30' and param_info['Source'] == 'SameAsIntensityMeasure':
                            kwargs[param_k] = site_data['Vs30 (m/s)'].values
                        else:
                            kwargs[param_k] = site_data[param_info['ColumnIDWithData']].values
                    else:
                        kwargs[param_k] = param_info
                
                # if edp_i == 'SurfaceFaultRupture':
                    # temp_file = os.path.join(r'C:\Users\barry\Desktop\New folder\test2','input_param.json')
                    # keys = curr_peer_category[edp_i]['Method'][method_j]['Flags'].keys()
                    # print(curr_peer_category[edp_i]['Method'][method_j]['Flags'])
                    # np.savetxt(temp_file, keys)
                    # with open(os.path.join(r'C:\Users\barry\Desktop\New folder\test2','input_param.json'),'w') as f:
                        # json.dump(curr_peer_category[edp_i],f,indent=4)
                
                # import procedure and run method
                proc_for_method = getattr(
                    importlib.import_module('src.EDP.'+edp_i), method_j)
                # evaluate DV with catch on Numpy warnings
                # with warnings.catch_warnings():
                    # warnings.simplefilter("ignore")
                output = proc_for_method(**kwargs)

                # if edp_i == 'LateralSpread':
                    # return output
                
                # store outputs
                weight = method_info['Weight']
                # loop through return parameters
                for param_k in return_param:
                    if param_k in output.keys():
                        logging.info(f"\t\tGetting return parameter: {param_k}")
                        # specific to liq_susc, which is a category
                        if param_k == 'liq_susc':
                            self._EDP_dict[edp_i][param_k]['Value'] = output[param_k]
                        # for p_land, only rely on BrayMacedo2007 for now
                        elif param_k == 'p_land':
                            # loop through samples
                            for sample_l in range(n_sample):
                                    self._EDP_dict[edp_i][param_k]['Value'][sample_l] = output[param_k][sample_l]
                        else:
                            # loop through samples
                            for sample_l in range(n_sample):
                                # logging.info(f"\t\t\tSample {sample_l}")
                                # if output is sample-dependent, will be in dict
                                if isinstance(output[param_k], dict):
                                    # store directly into _EDP_dict or update/combine using weights
                                    if method_counter == 0:
                                        self._EDP_dict[edp_i][param_k]['Value'][sample_l] = \
                                            output[param_k][sample_l].power(weight)
                                    else:
                                        self._EDP_dict[edp_i][param_k]['Value'][sample_l] = \
                                            self._EDP_dict[edp_i][param_k]['Value'][sample_l].multiply(
                                                output[param_k][sample_l].power(weight))
                                else:
                                    # store directly into _EDP_dict or update/combine using weights
                                    if method_counter == 0:
                                        self._EDP_dict[edp_i][param_k]['Value'][sample_l] = \
                                            output[param_k].power(weight)
                                    else:
                                        self._EDP_dict[edp_i][param_k]['Value'][sample_l] = \
                                            self._EDP_dict[edp_i][param_k]['Value'][sample_l].multiply(
                                                output[param_k].power(weight))
                
                # get uncertainty params
                if dist_type == 'Uniform':
                    self._EDP_dict[edp_i]['Uncertainty']['Aleatory']['ScaleFactor'] = \
                        output['prob_dist']['factor_aleatory']
                    self._EDP_dict[edp_i]['Uncertainty']['Epistemic']['ScaleFactor'] = \
                        output['prob_dist']['factor_epistemic']
                elif dist_type == 'Lognormal':
                    self._EDP_dict[edp_i]['Uncertainty']['Aleatory']['StdDev'] = np.sqrt(
                        self._EDP_dict[edp_i]['Uncertainty']['Aleatory']['StdDev']**2 + \
                        output['prob_dist']['sigma_aleatory']**2 * weight**2)
                    self._EDP_dict[edp_i]['Uncertainty']['Epistemic']['StdDev'] = np.sqrt(
                        self._EDP_dict[edp_i]['Uncertainty']['Epistemic']['StdDev']**2 + \
                        output['prob_dist']['sigma_epistemic']**2 * weight**2)
                    self._EDP_dict[edp_i]['Uncertainty']['Total']['StdDev'] = np.sqrt(
                        self._EDP_dict[edp_i]['Uncertainty']['Total']['StdDev']**2 + \
                        output['prob_dist']['sigma_total']**2 * weight**2)
                
                # remove method specific input parameters from kwargs for next iteration
                kwargs.pop('M',None)
                kwargs.pop('pga',None)
                kwargs.pop('pgv',None)
                for key in param_computed_in_run_to_add[edp_i]:
                    kwargs.pop(key,None)
                for param_k in method_info['InputParameters']:
                    kwargs.pop(param_k,None)
            
            # remove hazard specific input parameters from kwargs for next iteration
            kwargs.pop('return_param',None)
            
        #
        if len(method_param_for_assess['EDP']) == 0:
            logging.info(f'No EDP requested for this analysis\n')
        else:
            logging.info(f'Added EDP results to "model._EDP_dict\n')

        # return temp_out

    # -----------------------------------------------------------
    def assess_DM(self, setup_config, other_config_param, site_data, method_param_for_assess):
        """
        Using the simulated intensity measures and engineering demand parameters to calculate damage measures.

        Parameters
        ----------

        For input parameters to each method, refer to the method documentation under :func:`dm`.

        Returns
        -------
        output : varies
            [varies] output depends on the target demand and methods.

        """
        
        # Initialize dictionary if it is none
        if self._DM_dict is None:
            self._DM_dict = {}
    
        #
        if len(method_param_for_assess['DM']) == 0:
            logging.info(f'No DM requested for this analysis\n')
        else:
            logging.info(f'Added DM results to "model._DM_dict\n')
    
    
    # -----------------------------------------------------------
    def assess_DV(self, setup_config, other_config_param, site_data, method_param_for_assess):
        """
        Using the simulated intensity measures, engineering demand parameters, and damage measures to calculate decision variables/values.

        Parameters
        ----------
        dv_category : str
           decision variable category to calculate; options are **rr** (more to be added, see :func:`dv` for meaning of the options)
        dv_method : str
            method/procedure to use to calculate the damage; see :func:`dv` for available methods.
        return_param : str, list
            single of a list of parameters to return, see the return variables under each function (:func:`dv`)
        store_name : str, list, optional
           names to store parameter as; default = **return_param**
        ims : str, list, optional
           list of **IM** variables to create: **pga** or **pgv** (default = both)
        flag_clear_dict : boolean, optional
           **True** to clear the calculated damages or **False** to keep them; default = **False**
        flag_pga : boolean, optional
           **True** include simulated **PGA**; default = **False**
        flag_pgv : boolean, optional
           **True** include simulated **PGV**; default = **False**
        flag_mag : boolean, optional
           **True** include moment magnitude **mag** from rupture metadata; default = **False**
        flag_rup_depend : boolean, optional
           **True** if dependent on rupture scenario; default = **False**
        source_dict : str, list, optional
           dictionary that contains **source_param** and **source_method**; default = None
        source_param : str, list, optional
           parameter to get from **existing** stored parameters in (e.g., **liq_susc**); default = None
        source_method : str, list, optional
           method used to obtain the source_param (e.g., **zhu_etal_2017**); default = None

        For method parameters, refer to the method documentation under :func:`dv`.

        Returns
        -------
        output : varies
            [varies] output depends on the target decision variables and methods.

        """
        
        # Initialize dictionary if it is none
        if self._DV_dict is None:
            self._DV_dict = {}
        
        # setup
        n_site = other_config_param['Num_Sites']
        n_event = other_config_param['Num_Events']
        n_EDP = other_config_param['Num_EDP']
        sampling_method = setup_config['SamplingMethod']['Method']
        if 'monte' in sampling_method.lower():
            algorithm = setup_config['SamplingMethod']['Algorithm']
            n_sample = setup_config['SamplingMethod']['NumberOfSamples']
            seed_num = setup_config['SamplingMethod']['Seed']
        elif 'logic' in sampling_method.lower():
            algorithm = None
            logic_branch = setup_config['SamplingMethod']['Branch']
            logic_branch_weights = setup_config['SamplingMethod']['Weights']
            n_sample = len(logic_branch)
        elif 'polynomial' in sampling_method.lower():
            logging.info("Polynomial chaos under development: restart analysis with another method")
        # store_file_type = setup_config['General']['OutputFileType']
        store_file_type = 'txt'
        n_decimals = other_config_param['Num_Decimals']
        
        # get rates and reformat to shape = n_event x n_site
        rate = self._EVENT_dict['Scenarios']['MeanAnnualRate']
        rate = np.tile(rate,[n_site, 1]).T
        
        # set up kwargs to pass inputs into method function
        kwargs = {}
        kwargs['n_site'] = n_site
        kwargs['n_event'] = n_event
        kwargs['n_sample'] = n_sample
        
        
        # -----------------------------------------------------------
        # think about how to loop this with more random variables
        # -----------------------------------------------------------
        # setup additional information for DV
        demand_dict = {}
        dist_type_dv = {}
        for dv_i in method_param_for_assess['DV']:
            demand_dict[dv_i] = {}
            # get DV distribution type
            if dv_i in other_config_param['DistributionType']['Uniform']['ListOfHazard']:
                dist_type_dv[dv_i] = 'Uniform'
            elif dv_i in other_config_param['DistributionType']['Lognormal']['ListOfHazard']:
                dist_type_dv[dv_i] = 'Lognormal'
            # get other params
            if dv_i == 'RepairRatePGV':
                demand_dict[dv_i]['PGV'] = {
                    'Demand': {
                        'Type': 'pgv',
                        'Label': None,
                        'Value': self._IM_dict['Simulation']['Sample']['PGV'],
                        'DistType': 'Lognormal',
                        'EpiBranch': other_config_param['DistributionType']['Lognormal']['Epistemic']['Branch'],
                        'EpiWgt': other_config_param['DistributionType']['Lognormal']['Epistemic']['Weight'],
                        'StdDevAle': 0,
                        'StdDevEpi': other_config_param['UniformStdDev'],
                        'StdDevTotal': other_config_param['UniformStdDev']
                    },
                    'Prob': None
                }
            elif dv_i == 'RepairRatePGD':
                for edp_j in other_config_param['EDPDemandForRepairRatePGD']:
                    pgd_label = other_config_param['EDPDemandForRepairRatePGD'][edp_j]['Label_PGD']
                    dist_type = self._EDP_dict[edp_j]['Uncertainty']['DistributionType']
                    demand_dict[dv_i][edp_j] = {
                        'Demand': {
                            'Type': 'pgd',
                            'Label': pgd_label,
                            'Value': self._EDP_dict[edp_j][pgd_label]['Value'],
                            'DistType': dist_type,
                            'EpiBranch': self._EDP_dict[edp_j]['Uncertainty']['Epistemic']['Branches'],
                            'EpiWgt': other_config_param['DistributionType']['Lognormal']['Epistemic']['Weight'],
                        }
                    }
                    if dist_type == 'Uniform':
                        demand_dict[dv_i][edp_j]['Demand'].update({
                            'ScaleFactorAle': self._EDP_dict[edp_j]['Uncertainty']['Aleatory']['ScaleFactor'],
                            'ScaleFactorEpi': self._EDP_dict[edp_j]['Uncertainty']['Epistemic']['ScaleFactor']
                        })
                    elif dist_type == 'Lognormal':
                        demand_dict[dv_i][edp_j]['Demand'].update({
                            'StdDevAle': self._EDP_dict[edp_j]['Uncertainty']['Aleatory']['StdDev'],
                            'StdDevEpi': self._EDP_dict[edp_j]['Uncertainty']['Epistemic']['StdDev'],
                            'StdDevTotal': self._EDP_dict[edp_j]['Uncertainty']['Total']['StdDev']
                        })
                    # prob_hazard params to import
                    if edp_j == 'LateralSpread' or edp_j == 'GroundSettlement':
                        demand_dict[dv_i][edp_j]['Prob'] = {
                            'Label': 'p_liq',
                            'Value': self._EDP_dict['Liquefaction']['p_liq']['Value'],
                            'DistType': 'Uniform',
                            'EpiBranch': other_config_param['DistributionType']['Uniform']['Epistemic']['Branch'],
                            'EpiWgt': other_config_param['DistributionType']['Uniform']['Epistemic']['Weight'],
                            'ScaleFactorAle': self._EDP_dict['Liquefaction']['Uncertainty']['Aleatory']['ScaleFactor'],
                            'ScaleFactorEpi': self._EDP_dict['Liquefaction']['Uncertainty']['Epistemic']['ScaleFactor']
                        }
                    elif edp_j == 'Landslide':
                        demand_dict[dv_i][edp_j]['Prob'] = {
                            'Label': 'p_land',
                            'Value': self._EDP_dict['Landslide']['p_land']['Value'],
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Probability of landslide deformation is set to be uniformly distributed for now
                            'DistType': 'Uniform',
                            'EpiBranch': other_config_param['DistributionType']['Uniform']['Epistemic']['Branch'],
                            'EpiWgt': other_config_param['DistributionType']['Uniform']['Epistemic']['Weight'],
                            'ScaleFactorAle': self._EDP_dict['Liquefaction']['Uncertainty']['Aleatory']['ScaleFactor'],
                            'ScaleFactorEpi': self._EDP_dict['Liquefaction']['Uncertainty']['Epistemic']['ScaleFactor']
                            # 'DistType': 'Lognormal',
                            # 'EpiBranch': other_config_param['DistributionType']['Lognormal']['Epistemic']['Branch'],
                            # 'EpiWgt': other_config_param['DistributionType']['Lognormal']['Epistemic']['Weight'],
                            # 'StdDevAle': self._EDP_dict['Landslide']['Uncertainty']['Aleatory']['StdDev'],
                            # 'StdDevEpi': self._EDP_dict['Landslide']['Uncertainty']['Epistemic']['StdDev'],
                            # 'StdDevTotal': self._EDP_dict['Landslide']['Uncertainty']['Total']['StdDev']
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        }
                    elif edp_j == 'SurfaceFaultRupture':
                        demand_dict[dv_i][edp_j]['Prob'] = {
                            'Label': 'p_surf',
                            'Value': None,
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            'DistType': 'Uniform',
                            'EpiBranch': [1],
                            'EpiWgt': [1],
                            'ScaleFactorAle': 1,
                            'ScaleFactorEpi': 1
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        }
                    else:
                        demand_dict[dv_i][edp_j]['Prob'] = None
                        
        # -----------------------------------------------------------
        # think about how to loop this with more random variables
        # -----------------------------------------------------------
        
        # get current PEER category info from model_assess_param
        curr_peer_category = method_param_for_assess['DV']
        # Loop through DVs
        dv_counter = 0
        for dv_i in curr_peer_category:
            logging.info(f"Current DV: {dv_i}")
            dv_info = curr_peer_category[dv_i]
            self._DV_dict[dv_i] = {}

            # get DV ID
            dv_id = other_config_param['ID_DV'][dv_i]
            
            # pull method params from method_param_for_assess
            method = list(dv_info['Method'].keys())
            return_param = dv_info['ReturnParameter']
            
            # get demand_dict for current dv
            curr_demand_dict = demand_dict[dv_i]
            
            # get DV source name for loading procedure
            if 'RepairRate' in dv_i:
                dv_source_name = 'RepairRate'
            else:
                dv_source_name = dv_i
            
            # set up kwargs to pass inputs into method function
            kwargs['return_param'] = return_param
            
            # loop through list of methods
            method_counter = 0
            for method_j in method:
                logging.info(f"\tCurrent method: {method_j}")
                method_info = dv_info['Method'][method_j]
                method_weight = method_info['Weight']
                
                # add method specific input parameters to kwargs
                for param_k in method_info['InputParameters']:
                    param_info = method_info['InputParameters'][param_k]
                    if isinstance(param_info, dict):
                        # specifically for Vs30         
                        if param_k == 'Vs30' and param_info['Source'] == 'SameAsIntensityMeasure':
                            kwargs[param_k] = site_data['Vs30 (m/s)'].values
                        else:
                            kwargs[param_k] = site_data[param_info['ColumnIDWithData']].values
                    else:
                        kwargs[param_k] = param_info
            
                # import procedure for current method
                proc_for_method = getattr(
                    importlib.import_module('src.DV.'+dv_source_name), method_j)
                

                # -----------------------------------------------------------
                # for demands
                # -----------------------------------------------------------
                # loop through demands for current DV
                for demand_k in curr_demand_dict:
                    logging.info(f"\t\tCurrent demand for DV: {demand_k}")
                    demand_info = curr_demand_dict[demand_k]
                    self._DV_dict[dv_i][demand_k] = {}
                    # self._DV_dict[dv_i][demand_k] = None
                
                    # get info for current demand
                    demand_dist_type = demand_info['Demand']['DistType']
                    demand_epi_branch = demand_info['Demand']['EpiBranch']
                    demand_epi_wgt = demand_info['Demand']['EpiWgt']
                    if demand_dist_type == 'Uniform':
                        demand_ale_sf = demand_info['Demand']['ScaleFactorAle']
                        demand_epi_sf = demand_info['Demand']['ScaleFactorEpi']
                        demand_epi_str = []
                        for demand_epi_l in demand_epi_branch:
                            demand_epi_str.append(f'SF^{demand_epi_l}')
                    elif demand_dist_type == 'Lognormal':
                        demand_ale_stddev = demand_info['Demand']['StdDevAle']
                        demand_epi_stddev = demand_info['Demand']['StdDevEpi']
                        demand_total_stddev = demand_info['Demand']['StdDevTotal']
                        demand_epi_str = []
                        for demand_epi_l in demand_epi_branch:
                            demand_epi_str.append(f'{demand_epi_l}Sigma')
                    
                    # add to kwargs
                    kwargs['pgd_label'] = demand_info['Demand']['Label']
                    # loop through epistemic branches for demand
                    demand_epi = {}
                    for demand_epi_counter, demand_epi_l in enumerate(demand_epi_branch):
                        # logging.info(f"\t\t\tCurrent epistemic branch for demand: {demand_epi_str[demand_epi_counter]}")
                        # setup
                        # epi_name = 'epi_'+str(demand_epi_l)
                        # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]] = {}
                        # specific for RepairRatePGV
                        if dv_i == 'RepairRatePGV':
                            # adjust for epistemic
                            scale_factor = np.exp(demand_epi_stddev*demand_epi_l)
                            for sample_n in range(n_sample):
                                # default = lognormal
                                # get demand for current sample
                                demand_epi[sample_n] = demand_info['Demand']['Value'][sample_n]
                                # apply scale_factor to mean demand and covert back to coo_matrix
                                demand_epi[sample_n] = demand_epi[sample_n].multiply(scale_factor).tocoo()
                            # add adjusted demand to kwargs
                            kwargs[demand_info['Demand']['Type']] = demand_epi
                            # -----------------------------------------------------------
                            # run method
                            output = proc_for_method(**kwargs)
                            # -----------------------------------------------------------
                            # store into temporary dict
                            demand_epi = output[dv_id]
                            # remove adjusted demand kwargs for next iteration
                            kwargs.pop(demand_info['Demand']['Type'],None)
                            # clear dictionaries
                            output = None
                        
                        # specific for RepairRatePGD
                        else:
                            # see if using specific aleatory cases for demand
                            if other_config_param['Flag_UseSpecificAleatoryBranch']:
                                # get fixed aleatory cases 
                                demand_ale_case = other_config_param['DistributionType'][demand_dist_type]['Aleatory']['Branch']
                                demand_ale_wgt = other_config_param['DistributionType'][demand_dist_type]['Aleatory']['Weights']
                            else:
                                demand_ale_case = np.transpose(Fcn_Common.lhs(n_var=1, n_samp=min(5,n_sample), dist=demand_dist_type))[0]
                                demand_ale_wgt = np.ones(min(5,n_sample))/min(5,n_sample)
                            
                            # loop through aleatory cases for demand
                            demand_epi_ale = {}
                            for demand_ale_counter, demand_ale_m in enumerate(demand_ale_case):
                                # for each PGV branch in the demand; sample the demand on aleatory variability
                                demand_epi_ale_val = {}
                                for sample_n in range(n_sample):
                                    # logging.info(f"\t\t\t\t\tSample {sample_n} of demand")
                                    # get demand for current sample
                                    demand_epi_ale_val[sample_n] = demand_info['Demand']['Value'][sample_n]
                                    if demand_dist_type == 'Uniform':
                                        # adjust for epistemic
                                        scale_factor = demand_epi_sf**demand_epi_l
                                        # adjust for aleatory
                                        scale_factor = scale_factor*demand_ale_sf**demand_ale_m
                                    elif demand_dist_type == 'Lognormal':
                                        if other_config_param['Flag_CombineAleatoryAndEpistemic']:
                                            # adjust for epistemic with total stddev
                                            scale_factor = np.exp(demand_total_stddev*demand_ale_m)
                                        else:
                                            # adjust for epistemic
                                            scale_factor = np.exp(demand_epi_stddev*demand_epi_l)
                                            # adjust for aleatory
                                            scale_factor = np.exp(demand_ale_stddev*demand_ale_m)
                                    # apply scale_factor to mean demand and covert back to coo_matrix
                                    demand_epi_ale_val[sample_n] = demand_epi_ale_val[sample_n].multiply(scale_factor).tocoo()
                                # add adjusted demand to kwargs
                                kwargs[demand_info['Demand']['Label']] = demand_epi_ale_val
                                
                                
                                # -----------------------------------------------------------
                                # for probability
                                # -----------------------------------------------------------
                                # see if using specific aleatory cases for probability
                                if demand_info['Prob'] is None:
                                    prob_dist_type = 'Uniform'
                                    #
                                    prob_epi_sf = 1
                                    prob_epi_branch = [1]
                                    prob_epi_wgt = [1]
                                    #
                                    prob_ale_sf = 1
                                    prob_ale_case = [1]
                                    prob_ale_wgt = [1]
                                else:                                        
                                    # get info for current demand probability
                                    prob_dist_type = demand_info['Prob']['DistType']
                                    prob_epi_sf = demand_info['Prob']['ScaleFactorEpi']
                                    prob_epi_branch = demand_info['Prob']['EpiBranch']
                                    prob_epi_wgt = demand_info['Prob']['EpiWgt']
                                    #
                                    prob_ale_sf = demand_info['Prob']['ScaleFactorAle']
                                    if other_config_param['Flag_UseSpecificAleatoryBranch']:
                                        # get fixed aleatory cases 
                                        prob_ale_case = other_config_param['DistributionType']['Uniform']['Aleatory']['Branch']
                                        prob_ale_wgt = other_config_param['DistributionType']['Uniform']['Aleatory']['Weights']
                                    else:
                                        prob_ale_case = np.transpose(Fcn_Common.lhs(n_var=1, n_samp=min(5,n_sample), dist='Uniform'))[0]
                                        prob_ale_wgt = np.ones(min(5,n_sample))/min(5,n_sample)
                                
                                # loop through epistemic branches for probability
                                prob_epi = {}
                                for prob_epi_counter, prob_epi_l in enumerate(prob_epi_branch):
                                    # logging.info(f"\t\t\t\tCurrent epistemic branch for prob: {prob_epi_branch[prob_epi_counter]}")
                                    # loop through aleatory cases for demand
                                    prob_epi_ale = {}
                                    for prob_ale_counter, prob_ale_m in enumerate(prob_ale_case):
                                        # for each PGV branch in the demand; sample the demand on aleatory variability
                                        prob_epi_ale_val = {}
                                        for sample_n in range(n_sample):
                                            # logging.info(f"\t\t\t\t\tSample {sample_n} of demand")
                                            # get demand for current sample
                                            if demand_info['Prob']['Value'] is None:
                                                ref_sparse_mat = demand_epi_ale_val[sample_n].tocoo()
                                                prob_epi_ale_val[sample_n] = sparse.coo_matrix(
                                                    (np.ones(len(ref_sparse_mat.data)),(ref_sparse_mat.row,ref_sparse_mat.col)),
                                                    shape=ref_sparse_mat.shape
                                                )
                                            else:
                                                prob_epi_ale_val[sample_n] = demand_info['Prob']['Value'][sample_n]
                                            # adjust for epistemic
                                            scale_factor = prob_epi_sf**prob_epi_l
                                            # adjust for aleatory
                                            scale_factor = scale_factor*prob_ale_sf**prob_ale_m
                                            # apply scale_factor to mean demand and covert back to coo_matrix
                                            prob_epi_ale_val[sample_n] = prob_epi_ale_val[sample_n].multiply(scale_factor).tocoo()
                                            # limit maximum probability to 100%
                                            prob_epi_ale_val[sample_n].data = np.minimum(prob_epi_ale_val[sample_n].data,100)
                                        # add adjusted demand to kwargs
                                        kwargs[demand_info['Prob']['Label']] = prob_epi_ale_val
                                    
                                        # -----------------------------------------------------------
                                        # run method
                                        output = proc_for_method(**kwargs)
                                        # -----------------------------------------------------------

                                        # for probability, loop through PGV branches and scale by prob_ale_wgt
                                        for sample_n in range(n_sample):
                                            if sample_n in prob_epi_ale:
                                                prob_epi_ale[sample_n] = prob_epi_ale[sample_n] + \
                                                    output[dv_id][sample_n].multiply(prob_ale_wgt[prob_ale_counter])
                                            else:
                                                prob_epi_ale[sample_n] = output[dv_id][sample_n].multiply(prob_ale_wgt[prob_ale_counter])
                                        # remove adjusted prob kwargs for next iteration
                                        kwargs.pop(demand_info['Prob']['Label'],None)
                                        # clear dictionaries
                                        output = None

                                    # for probability, loop through PGV branches and scale by prob_epi_wgt
                                    for sample_n in range(n_sample):
                                        if sample_n in prob_epi:
                                            prob_epi[sample_n] = prob_epi[sample_n] + \
                                                prob_epi_ale[sample_n].multiply(prob_epi_wgt[prob_epi_counter])
                                        else:
                                            prob_epi[sample_n] = prob_epi_ale[sample_n].multiply(prob_epi_wgt[prob_epi_counter])
                                    # clear dictionaries
                                    prob_epi_ale = None
                                # -----------------------------------------------------------
                                # for probability
                                # -----------------------------------------------------------


                                # for demand, loop through PGV branches and scale by demand_ale_wgt
                                for sample_n in range(n_sample):
                                    if sample_n in demand_epi_ale:
                                        if demand_dist_type == 'Uniform':
                                            demand_epi_ale[sample_n] = prob_epi[sample_n] + \
                                                prob_epi[sample_n].multiply(demand_ale_wgt[demand_ale_counter])
                                        elif demand_dist_type == 'Lognormal':
                                            demand_epi_ale[sample_n] = prob_epi[sample_n].multiply(
                                                prob_epi[sample_n].power(demand_ale_wgt[demand_ale_counter]))
                                    else:
                                        if demand_dist_type == 'Uniform':
                                            demand_epi_ale[sample_n] = prob_epi[sample_n].multiply(demand_ale_wgt[demand_ale_counter])
                                        elif demand_dist_type == 'Lognormal':
                                            demand_epi_ale[sample_n] = prob_epi[sample_n].power(demand_ale_wgt[demand_ale_counter])
                                # remove adjusted demand kwargs for next iteration
                                kwargs.pop(demand_info['Demand']['Label'],None)
                                # clear dictionaries
                                prob_epi = None

                            # for demand, loop through PGV branches and scale by demand_ale_wgt
                            for sample_n in range(n_sample):
                                if sample_n in demand_epi:
                                    if demand_dist_type == 'Uniform':
                                        demand_epi[sample_n] = demand_epi[sample_n] + \
                                            demand_epi_ale[sample_n].multiply(demand_epi_wgt[demand_epi_counter])
                                    elif demand_dist_type == 'Lognormal':
                                        demand_epi[sample_n] = demand_epi[sample_n].multiply(
                                            demand_epi_ale[sample_n].power(demand_epi_wgt[demand_epi_counter]))
                                else:
                                    if demand_dist_type == 'Uniform':
                                        demand_epi[sample_n] = demand_epi_ale[sample_n].multiply(demand_epi_wgt[demand_epi_counter])
                                    elif demand_dist_type == 'Lognormal':
                                        demand_epi[sample_n] = demand_epi_ale[sample_n].power(demand_epi_wgt[demand_epi_counter])
                            # clear dictionaries
                            demand_epi_ale = None
                
                        # combine by methods using method weight and store into _DV_dict
                        # -----------------------------------------------------------
                        # for without combining epistemic uncertainty
                        # for sample_n in range(n_sample):
                            # if sample_n in self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]]:
                                # if dist_type_dv[dv_i] == 'Uniform':
                                    # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]][sample_n] = \
                                        # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]] + \
                                        # demand_epi[sample_n].multiply(method_weight)
                                # elif dist_type_dv[dv_i] == 'Lognormal':
                                    # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]][sample_n] = \
                                        # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]].multiply(
                                        # demand_epi[sample_n].power(method_weight))
                            # else:
                                # if dist_type_dv[dv_i] == 'Uniform':
                                    # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]][sample_n] = \
                                        # demand_epi[sample_n].multiply(method_weight)
                                # elif dist_type_dv[dv_i] == 'Lognormal':
                                    # self._DV_dict[dv_i][demand_k][demand_epi_str[demand_epi_counter]][sample_n] = \
                                        # demand_epi[sample_n].power(method_weight)
                        # -----------------------------------------------------------
                        for sample_n in range(n_sample):
                            if sample_n in self._DV_dict[dv_i][demand_k]:
                                if dist_type_dv[dv_i] == 'Uniform':
                                    self._DV_dict[dv_i][demand_k][sample_n] = \
                                        self._DV_dict[dv_i][demand_k][sample_n] + demand_epi[sample_n].multiply(method_weight)
                                elif dist_type_dv[dv_i] == 'Lognormal':
                                    self._DV_dict[dv_i][demand_k][sample_n] = \
                                        self._DV_dict[dv_i][demand_k][sample_n].multiply(demand_epi[sample_n].power(method_weight))
                            else:
                                if dist_type_dv[dv_i] == 'Uniform':
                                    self._DV_dict[dv_i][demand_k][sample_n] = demand_epi[sample_n].multiply(method_weight)
                                elif dist_type_dv[dv_i] == 'Lognormal':
                                    self._DV_dict[dv_i][demand_k][sample_n] = demand_epi[sample_n].power(method_weight)
                                               
                    # clear dictionaries
                    demand_epi = None
                    # -----------------------------------------------------------
                    # for demands
                    # -----------------------------------------------------------
                                    
                    # remove demand specific input parameters from kwargs for next iteration
                    kwargs.pop('pgd_label',None)
                    
            # combine events and update _DV_dict
            for demand_k in self._DV_dict[dv_i]:
                # -----------------------------------------------------------
                # for without combining epistemic uncertainty
                # for epi_l in self._DV_dict[dv_i][demand_k]:
                    # temp_dict = self._DV_dict[dv_i][demand_k][epi_l]
                    
                    # -----------------------------------------------------------
                    # combine samples
                    # sample_weights = np.ones(len(temp_dict))/len(temp_dict)
                    # for sample_n in range(len(temp_dict)):
                        # if sample_n == 0:
                            # if dist_type_dv[dv_i] == 'Uniform':
                                # self._DV_dict[dv_i][demand_k][epi_l] = \
                                    # temp_dict[sample_n].multiply(sample_weights[sample_n])
                            # elif dist_type_dv[dv_i] == 'Lognormal':
                                # self._DV_dict[dv_i][demand_k][epi_l] = \
                                    # temp_dict[sample_n].power(sample_weights[sample_n])
                        # else:
                            # if dist_type_dv[dv_i] == 'Uniform':
                                # self._DV_dict[dv_i][demand_k][epi_l] = \
                                    # self._DV_dict[dv_i][demand_k][epi_l] + \
                                    # temp_dict[sample_n].multiply(sample_weights[sample_n])
                            # elif dist_type_dv[dv_i] == 'Lognormal':
                                # self._DV_dict[dv_i][demand_k][epi_l] = \
                                    # self._DV_dict[dv_i][demand_k][epi_l].multiply(
                                    # temp_dict[sample_n].power(sample_weights[sample_n]))
                    # -----------------------------------------------------------
                    
                    # weight by rates and sum across events
                    # self._DV_dict[dv_i][demand_k][epi_l] = sparse.coo_matrix(
                        # np.sum(self._DV_dict[dv_i][demand_k][epi_l].multiply(rate).toarray(),axis=0))
                # -----------------------------------------------------------
                
                for sample_n in range(n_sample):
                    self._DV_dict[dv_i][demand_k][sample_n] = sparse.coo_matrix(
                        np.sum(self._DV_dict[dv_i][demand_k][sample_n].multiply(rate).toarray(),axis=0))
                    
                # combine samples and split into breaks and leaks if requesting "RepairRate"
                for sample_n in range(n_sample):
                    if sample_n == 0:
                        temp_sample = self._DV_dict[dv_i][demand_k][sample_n]
                    else:
                        if dist_type_dv[dv_i] == 'Uniform':
                            temp_sample = temp_sample + self._DV_dict[dv_i][demand_k][sample_n]
                        elif dist_type_dv[dv_i] == 'Lognormal':
                            temp_sample = temp_sample.multiply(self._DV_dict[dv_i][demand_k][sample_n])
                if dist_type_dv[dv_i] == 'Uniform':
                    temp_sample = temp_sample.multiply(1/n_sample)
                elif dist_type_dv[dv_i] == 'Lognormal':
                    temp_sample = temp_sample.power(1/n_sample)
                if 'RepairRate' in dv_i:
                    if 'PGV' in dv_i:
                        self._DV_dict[dv_i][demand_k] = {
                            'NumberOfBreaks': temp_sample.multiply(0.2),
                            'NumberOfLeaks': temp_sample.multiply(0.8)
                        }
                    elif 'PGD' in dv_i:
                        self._DV_dict[dv_i][demand_k] = {
                            'NumberOfBreaks': temp_sample.multiply(0.8),
                            'NumberOfLeaks': temp_sample.multiply(0.2)
                        }
                else:
                    self._DV_dict[dv_i][demand_k] = temp_sample
                
        #
        if len(method_param_for_assess['DV']) == 0:
            logging.info(f'No DV requested for this analysis\n')
        else:
            logging.info(f'Added DV results to "model._DV_dict\n')

    
    # -----------------------------------------------------------
    def export_DV(self, setup_config, other_config_param, site_data):
        """
        Exports calculated DVs to directory

        """
        
        if len(self._DV_dict) == 0:
            logging.info(f"No DV requested for this analysis - No results to export\n")
            
        else:
            logging.info(f"Export directory for DV:")
            logging.info(f"\t{other_config_param['Dir_DV']}")
            logging.info(f"Files:")
            
            # export to save_path
            if setup_config['General']['OutputFileType'] == 'txt':
                sep = ' '
            else:
                sep = ','
            
            # set up export file headers
            cols = []
            for i in range(len(other_config_param['ColumnsToKeepInFront'])):
                if not other_config_param['ColumnsToKeepInFront'][i] == 'SHAPE_LENGTH':
                    cols.append(other_config_param['ColumnsToKeepInFront'][i])
            
            # create a file with all of the results
            # make path
            save_path_all = os.path.join(other_config_param['Dir_DV'], 'AllResults.'+setup_config['General']['OutputFileType'])
            logging.info(f"\t{os.path.basename(save_path_all)}")
            # get data for initial column headers from site_data
            df_all_export = site_data[cols].copy()
            
            # track sum of dvs
            if 'RepairRatePGV' in self._DV_dict or 'RepairRatePGD' in self._DV_dict:
                total_repair_rate = np.zeros(site_data.shape[0])
            
            # loop through all DVs
            for dv_i in self._DV_dict:
            
                # make path
                save_path = os.path.join(other_config_param['Dir_DV'], dv_i+'.'+setup_config['General']['OutputFileType'])
                logging.info(f"\t{os.path.basename(save_path)}")
                # get data for initial column headers from site_data
                df_export = site_data[cols].copy()
            
                # loop through demands and epistemic branches to make more column headers and get data from _DV_dict
                for demand_j in self._DV_dict[dv_i]:
                    # print(f'4, {self._DV_dict[dv_i][demand_j].keys()}')
                    # -----------------------------------------------------------
                    # for without combining epistemic uncertainty
                    # for epi_i in self._DV_dict[dv_i][demand_j]:
                        # cols.append(demand_j+'_'+epi_i)
                        # df_export[demand_j+'_'+epi_i] = self._DV_dict[dv_i][demand_j][epi_i].toarray()[0]
                    # -----------------------------------------------------------
                    # for sample_n in range(len(self._DV_dict[dv_i][demand_j])):
                        # df_export[demand_j+'_IMsample'+str(sample_n+1)] = self._DV_dict[dv_i][demand_j][sample_n].toarray()[0]
                    if isinstance(self._DV_dict[dv_i][demand_j],dict):
                        for key in self._DV_dict[dv_i][demand_j].keys():
                            df_export[demand_j+'_'+key] = self._DV_dict[dv_i][demand_j][key].toarray()[0]
                            # file with all the results
                            df_all_export[dv_i+'_'+demand_j+'_'+key] = self._DV_dict[dv_i][demand_j][key].toarray()[0]
                            # track sum of dvs
                            if 'RepairRatePGV' in self._DV_dict or 'RepairRatePGD' in self._DV_dict:
                                total_repair_rate = total_repair_rate + self._DV_dict[dv_i][demand_j][key].toarray()[0]
                    else:
                        df_export[demand_j] = self._DV_dict[dv_i][demand_j].toarray()[0]
                        # file with all the results
                        df_all_export[dv_i+'_'+demand_j] = self._DV_dict[dv_i][demand_j].toarray()[0]
                        # track sum of dvs
                        if 'RepairRatePGV' in self._DV_dict or 'RepairRatePGD' in self._DV_dict:
                            total_repair_rate = total_repair_rate + self._DV_dict[dv_i][demand_j].toarray()[0]

                # export to file
                df_export.to_csv(save_path, index=False, sep=sep)
            
            # add column for total repair rate
            if 'RepairRatePGV' in self._DV_dict or 'RepairRatePGD' in self._DV_dict:
                df_all_export['TotalRepairRateForAllDemands'] = total_repair_rate
            
            # export to file
            df_all_export.to_csv(save_path_all, index=False, sep=sep)
            
            #
            logging.info(f'\n')