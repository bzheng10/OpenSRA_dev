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
import time
import importlib
import sys
import os
import logging
import numpy as np
from scipy import sparse

# OpenSRA modules and functions
from src import Fcn_Common
# from src import Fcn_InputOutput
from src.im import Fcn_IM
from lib.simcenter import OpenSHAInterface


# -----------------------------------------------------------
class assessment(object):
    """
    Assess the damages and decision values at target sites using ground motion predictions and demand parameters.
    
    .. autosummary::
    
        get_src_GM_site
        sim_IM
        assess_EDP
        assess_DM
        assess_DV
    
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
        event_keys = ['src','rup','mag','rate']

        # Check if files are present before interfacing with IM sources
        exist_ListOfScenarios = False
        exist_Predictions = True
        # Check if file with list of ruptures exists, if so, load and store into _EVENT_dict
        if os.path.exists(other_config_param['File_ListOfScenarios']):
            logging.info(f"File with list of rupture scenarios exist in:")
            logging.info(f"\t{other_config_param['Dir_IM']}")
            exist_ListOfScenarios = True
            rupture_list = OpenSHAInterface.get_src_rup_M_rate(
                erf=None, rupture_list_file=other_config_param['File_ListOfScenarios'], ind_range=['all'])
            for key in event_keys:
                self._EVENT_dict['Scenarios'][key] = rupture_list[key]
            self._EVENT_dict['Scenarios']['Num_Events'] = len(self._EVENT_dict['Scenarios']['src'])

        # Check if files with predictions already exist
        for im in other_config_param['IM']:
            if len(os.listdir(os.path.join(other_config_param['Dir_IM_GroundMotion_Prediction'],im))) < len(other_config_param['ListOfIMParams']):
                exist_Predictions = False
                break
        # Load predictions
        if exist_Predictions:
            logging.info(f"IM predictions already exist under:")
            logging.info(f"\t{other_config_param['Dir_IM_GroundMotion_Prediction']}")
            self._IM_dict.update({
                'Prediction': Fcn_IM.read_IM_means(
                    im_pred_dir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                    list_im = other_config_param['IM'],
                    list_param = other_config_param['ListOfIMParams'],
                    store_file_type = setup_config['General']['OutputFileType']
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
                    # Get full list
                    logging.info(f"Generating full list of rupture scenarios")
                    rupture_list = OpenSHAInterface.get_src_rup_M_rate(
                        erf=erf, rupture_list_file=None, ind_range=['all'], rate_cutoff=None)
                    # Filter list of ruptures
                    filters = setup_config['IntensityMeasure']['SourceParameters']['Filter']
                    filters_to_perform = {}
                    for item in filters.keys():
                        if filters[item]['ToInclude']:
                            filters_to_perform[item] = filters[item]
                    logging.info(f"... list of filters to perform: {filters_to_perform.keys()}")
                    rupture_list = OpenSHAInterface.filter_ruptures(
                        erf = erf,
                        locs = site_data[['Longitude','Latitude']].values,
                        filter_criteria = filters_to_perform,
                        rupture_list = rupture_list,
                        save_name = other_config_param['File_ListOfScenarios'],
                        rup_seg_file = other_config_param['Path_RuptureSegment'],
                        pt_src_file = other_config_param['Path_PointSource'])
                    # Store into _EVENT_dict
                    for key in event_keys:
                        if np.ndim(rupture_list[key]) < 1:
                            self._EVENT_dict['Scenarios'][key] = np.expand_dims(rupture_list[key],axis=0)
                        else:
                            self._EVENT_dict['Scenarios'][key] = rupture_list[key]
                
                # Get IM predictions
                if exist_Predictions is False:
                    logging.info(f"Looping through ruptures and get IM predictions")
                    self._IM_dict['Prediction'] = OpenSHAInterface.get_IM(
                        erf = erf,
                        imr = imr,
                        sites = sites,
                        src_list = rupture_list['src'],
                        rup_list = rupture_list['rup'],
                        list_im = other_config_param['IM'],
                        saveDir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                        store_file_type = setup_config['General']['OutputFileType'])
                    logging.info(f"... generated predictions for ruptures; results stored under:")
                    logging.info(f"\t{other_config_param['Dir_IM_GroundMotion_Prediction']}")
                    
                # Get fault crossings
                if setup_config['EngineeringDemandParameter']['Type']['SurfaceFaultRupture']['ToAssess'] and \
                    (len(os.listdir(other_config_param['Dir_FaultTrace'])) == 0 or len(os.listdir(other_config_param['Dir_Intersection'])) == 0):
                    
                    logging.info(f"Getting fault crossing function is outdated")
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
            
                logging.info(f"\n\n-----Interfacing with OpenSHA for GMs\n------------------------------------------------------")
                
            elif setup_config['IntensityMeasure']['SourceForIM'] == 'ShakeMap':
                logging.info(f"\n------------------------------------------------------\n-----Getting GMs from ShakeMap\n")
                #
                Fcn_IM.read_ShakeMap_data(
                    sm_dir = other_config_param['Dir_ShakeMap'],
                    event_names = other_config_param['ShakeMapEvents'],
                    sites = site_data[['Longitude','Latitude']].values,
                    IM_dir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                    trace_dir = other_config_param['Dir_FaultTrace'],
                    store_events_file = other_config_param['File_ListOfScenarios']
                )
                #
                self._IM_dict.update({
                    'Prediction': Fcn_IM.read_IM_means(
                        im_pred_dir = other_config_param['Dir_IM_GroundMotion_Prediction'],
                        list_im = other_config_param['IM'],
                        list_param = other_config_param['ListOfIMParams'],
                        store_file_type = setup_config['General']['OutputFileType']
                    )
                })
                #
                rupture_list = OpenSHAInterface.get_src_rup_M_rate(
                    erf=None, rupture_list_file=other_config_param['File_ListOfScenarios'], ind_range=['all'])
                for key in event_keys:
                    if np.ndim(rupture_list[key]) < 1:
                        self._EVENT_dict['Scenarios'][key] = np.expand_dims(rupture_list[key],axis=0)
                    else:
                        self._EVENT_dict['Scenarios'][key] = rupture_list[key]
                self._EVENT_dict['Scenarios']['Num_Events'] = len(self._EVENT_dict['Scenarios']['src'])
                logging.info(f"\n\n-----Getting GMs from ShakeMap\n------------------------------------------------------")
    
        logging.info(f'Added listOfRuptures to "model._EVENT_dict" and IM means and StdDev to "model._IM_dict"')
        
        
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
        self._IM_dict['Simulation'] = {'Correlation':{}, 'Samples':{}}
        n_site = other_config_param['Num_Sites']
        n_event = other_config_param['Num_Events']
        n_IM = other_config_param['Num_IM']
        sample_method = setup_config['UncertaintyQuantification']['Type']
        n_sample = setup_config['UncertaintyQuantification']['NumberOfSamples']
        seed_num = setup_config['UncertaintyQuantification']['Seed']
    
        # make period array for spectral correlations
        # approx_period_for_IM
        # approx_period_for_IM = []
        # for im_i in other_config_param['IM']:
            # approx_period_for_IM.append(other_config_param['ApproxPeriod'][im_i])
        
        # import correlation procedures to be applied
        proc_for_corr = {}
        for corr_type in ['Spatial', 'Spectral']:
            if setup_config['IntensityMeasure']['Correlation'][corr_type]['ToInclude']:
                proc_for_corr[corr_type] = getattr(
                    importlib.import_module('src.im.'+corr_type+'Correlation'),
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
        logging.info(f"Computing spatial and spectral correlations...")
        if other_config_param['Num_IM'] <= 1:
            logging.info(f"Spectral (cross) correlation is only applicable for 2 or more IMs:")
            self._IM_dict['Simulation']['Correlation']['Spectral']['Value'] = None
        else:
            if proc_for_corr['Spectral'] is None:
                logging.info(f"Spectral (cross) correlation not requested:")
                logging.info(f"... correlation value set to 0")
                self._IM_dict['Simulation']['Correlation']['Spectral']['Value'] = np.eye(other_config_param['Num_IM'])
            else:
                logging.info(f"Spectral (cross) correlation will be performed:")
                if self._IM_dict['Simulation']['Correlation']['Spectral']['Method'] == 'BakerJayaram2008':
                    logging.info(f"... method = {self._IM_dict['Simulation']['Correlation']['Spectral']['Method']}")
                    logging.info(f"... periods to assess with = {other_config_param['ApproxPeriod']}")
                    corr_val = proc_for_corr['Spectral'](
                            T1=other_config_param['ApproxPeriod']['PGA'],
                            T2=other_config_param['ApproxPeriod']['PGV'])
                    corr_mat = [[1,corr_val],[corr_val,1]]
                    self._IM_dict['Simulation']['Correlation']['Spectral']['Value'] = np.round(corr_mat,decimals=3)
                    logging.info(f"... correlation value = {self._IM_dict['Simulation']['Correlation']['Spectral']['Value']}")
                else:
                    logging.info(f"... invalid method requested; correlation value set to 0")
                    self._IM_dict['Simulation']['Correlation']['Spectral']['Value'] = np.eye(other_config_param['Num_IM'])

        # compute inter-event spatial correlations between sites
        if proc_for_corr['Spatial'] is None:
            logging.info(f"Spatial correlation not requested:")
            logging.info(f"... correlation value set to 0")
            self._IM_dict['Simulation']['Correlation']['Spatial']['Value'] = None
        else:
            logging.info(f"Spatial correlation will be performed:")
            if self._IM_dict['Simulation']['Correlation']['Spatial']['Method'] == 'JayaramBaker2009':
                logging.info(f"... method = {self._IM_dict['Simulation']['Correlation']['Spatial']['Method']}")
                # -----------------------------------------------------------
                # working procedure to calculate spatial correlation matrix
                site_lon = site_data['Longitude'].values
                site_lat = site_data['Latitude'].values
                # get distances between sites for upper triangle
                ind1,ind2 = np.triu_indices(n_site)
                d = Fcn_Common.get_haversine_dist(site_lon[ind1],site_lat[ind1],site_lon[ind2],site_lat[ind2])
                # compute intra-event correlations between sites for IMs
                self._IM_dict['Simulation']['Correlation']['Spatial']['Value'] = {}
                for i in range(other_config_param['Num_IM']):
                    corr_val = proc_for_corr['Spatial'](d=d, T=other_config_param['ApproxPeriod'][other_config_param['IM'][i]])
                    corr_mat = Fcn_Common.convert_triu_to_sym_mat(corr_val,n_site)
                    self._IM_dict['Simulation']['Correlation']['Spatial']['Value'].update({
                        other_config_param['IM'][i]: np.round(corr_mat,decimals=3)})
                # -----------------------------------------------------------
            else:
                logging.info(f"... invalid method requested; correlation value set to 0")
                self._IM_dict['Simulation']['Correlation']['Spatial']['Value'] = None
        
        # get flag for sampling with total sigma
        flag_SampleWithStDevTotal = other_config_param['Flag_SampleWithStDevTotal']
        
        # check if inter and intra event stdevs are available, if not, default to total stdev
        for im_i in other_config_param['IM']:
            if not 'stdev_inter' in self._IM_dict['Prediction'][im_i].keys() or \
                not 'stdev_intra' in self._IM_dict['Prediction'][im_i].keys():
                flag_SampleWithStDevTotal = True
                break
                
        # if sampling with total stdev, see if value is given, if not set to 1
        stdev_total_val = other_config_param['UniformSigmaAleatory']
        if stdev_total_val is not None:
            stdev_total_val = 1
        
        ###### perform sampling
        #####if flag_SampleWithStDevTotal:
        #####    pass
        #####else:
        #####    # get spatially-correlated residuals for each IM
        #####    self._IM_dict['Simulation']['Residuals'] = {'eps_intra':{},'eta_inter':{}}
        #####    eps_intra = {}
        #####    for im_i in other_config_param['IM']:
        #####        # np.random.seed(seed=seed_num) # set seed number
        #####        self._IM_dict['Simulation']['Residuals']['eps_intra'][im_i] = np.random.multivariate_normal(
        #####            mean = np.zeros(n_site),
        #####            cov = self._IM_dict['Simulation']['Correlation']['Spatial']['Value'][im_i],
        #####            size = n_sample)
        #####    # get spectrally-correlated residuals for each event
        #####    self._IM_dict['Simulation']['Residuals']['eta_inter'] = np.random.multivariate_normal(
        #####        mean = np.zeros(n_IM),
        #####        cov = self._IM_dict['Simulation']['Correlation']['Spectral']['Value'],
        #####        size = n_sample)
        #####    # get samples
        #####    self._IM_dict['Simulation']['Samples'] = {}
        #####    for event_j in range(n_event):
        #####        if event_j == 0:
        #####            for im_i in other_config_param['IM']:
        #####                mean = self._IM_dict['Prediction'][im_i]['mean'].toarray()
        #####                stdev_intra = self._IM_dict['Prediction'][im_i]['stdev_intra'].toarray()
        #####                stdev_inter = self._IM_dict['Prediction'][im_i]['stdev_inter'].toarray()
        #####                mean_j = mean[event_j,:]
        #####                stdev_intra_j = stdev_intra[event_j,:]
        #####                stdev_inter_j = stdev_inter[event_j,:]
        #####                print(mean_j, stdev_intra_j, stdev_inter_j)
        #####                # self._IM_dict['Simulation']['Samples'][im_i] = {}
        #####                for i in range(1):
        #####                    sample_i = mean_j*np.exp(
        #####                        self._IM_dict['Simulation']['Residuals']['eps_intra'][im_i][i]*np.diag(stdev_intra_j) + \
        #####                        self._IM_dict['Simulation']['Residuals']['eta_inter'][i]*np.tile(stdev_inter_j,(n_IM,1)))
        #####                    print(sample_i)
        #####                    # self._IM_dict['Simulation']['Samples'][im_i].update({
        #####                        # str(sample_i): sparse.coo_matrix(
        #####                            # self._IM_dict['Prediction'][im_i]['mean'] * np.exp(
        #####                            # self._IM_dict['Prediction'][im_i]['stdev_intra']*() +\
        #####                            # self._IM_dict['Prediction'][im_i]['stdev_inter']*()
        #####                            # )
        #####                        # )})

        # sys.exit()
                # epsilon_m = np.array([epsilon[:, i] for j in range(len(sa_data))])
                # ln_psa[:, :, i] = ln_sa + inter_sigma_sa * epsilon_m + intra_sigma_sa * eta[:, :, i]            
            
            
            # for sample_i in range(n_sample):
            # for sample_i in range(1):
                # get
                # eps_IM_1 = np.random.normal(size=(n_event,n_site))
                # samp = self._IM_pred_dict['rup']['pgv_mean']
        
        # eps = {}
        
        # eps_PGV
        

        ## check if spatial correlation is required
        #if flag_spatial_corr is False:
        #    # no correlations, random/LHS sampling
        #    
        #    self._IM_dict.update({'pgv':{}})
        #    self._IM_dict.update({'pga':{}})
        #
        #    # check if samples already exist
        #    if flag_im_sample_exist is True:   # load samples
        #    
        #        # loop through ims
        #        for im in ims:
        #            # loop through and import all samples
        #            for i in range(n_samp_im):
        #                file_name = os.path.join(sample_dir,im+'_samp_'+str(i)+'.'+store_file_type)
        #                self._IM_dict[im].update({i:file_io.read_im_samp(file_name, store_file_type, n_event, n_site)})
        #                    
        #    
        #    else:   # perform random sampling
        #
        #        # loop through number of samples
        #        for i in range(n_samp_im):
        #        
        #            # first sample pgv
        #            eps_pgv = np.random.normal(size=(n_event,n_site))
        #            samp = self._IM_pred_dict['rup']['pgv_mean']
        #            
        #            # use total sigma or separate into intra- and inter- event sigmas
        #            if flag_sample_with_sigma_total:
        #                # correct for sigma
        #                if sigma_aleatory is None: # if total sigma is not provided
        #                    sigma_total = self._IM_pred_dict['rup']['pgv_intra'].power(2) + self._IM_pred_dict['rup']['pgv_inter'].power(2)
        #                    sigma_total = sigma_total.power(0.5)
        #                    samp = samp.multiply(sigma_total.multiply(eps_pgv).expm1() + np.ones((n_event,n_site)))
        #                
        #                else:
        #                    # only supports singular inputs for sigma_aleatory, expand to matrix later
        #                    samp = samp.multiply(np.exp(sigma_aleatory*eps_pgv))
        #            
        #            else:
        #                # get residuals for intra (epsilon) and inter (eta) (norm dist with mean = 0 and sigma = 1)
        #                eta_pgv = np.random.normal(size=n_event)
        #                eta_pgv = np.repeat(eta_pgv[:,np.newaxis],n_site,axis=1) # eta is constant with site, varies only between rupture
        #                
        #                # correct for predicted mean and sigma
        #                samp = samp.multiply(self._IM_pred_dict['rup']['pgv_intra'].multiply(eps_pgv).expm1() + np.ones((n_event,n_site)))
        #                samp = samp.multiply(self._IM_pred_dict['rup']['pgv_inter'].multiply(eta_pgv).expm1() + np.ones((n_event,n_site)))
        #        
        #            # store samples internally
        #            self._IM_dict['pgv'].update({i:samp}) # update class
        #            
        #            # store samples locally                    
        #            save_name = os.path.join(sample_dir,'pgv_samp_'+str(i)+'.'+store_file_type)
        #            file_io.store_im_samp(save_name, samp, store_file_type, n_decimals)
        #            
        #            # see if 'pga' is needed
        #            if 'pga' in ims:
        #            
        #                # conditional sigma for pga
        #                sigma_cond_pga = np.sqrt(1-corr_T**2)
        #                
        #                # conditional mean of eps
        #                cond_mean_pga_eps = corr_T*eps_pgv
        #                eps_pga = np.random.normal(size=(n_event,n_site),loc=cond_mean_pga_eps,scale=sigma_cond_pga)
        #                samp = self._IM_pred_dict['rup']['pga_mean']
        #                                
        #                # use total sigma or separate into intra- and inter- event sigmas
        #                if flag_sample_with_sigma_total:
        #                    # correct for sigma
        #                    if sigma_aleatory is None: # if total sigma is not provided
        #                        sigma_total = self._IM_pred_dict['rup']['pga'+'_intra'].power(2) + self._IM_pred_dict['rup']['pga'+'_inter'].power(2)
        #                        sigma_total = sigma_total.power(0.5)
        #                        samp = samp.multiply(sigma_total.multiply(eps_pga).expm1() + np.ones((n_event,n_site)))
        #                    
        #                    else:
        #                        # only supports singular inputs for sigma_aleatory, expand to matrix later
        #                        samp = samp.multiply(np.exp(sigma_aleatory*eps_pga))
        #                        
        #                else:
        #                    # conditional sampling of eta
        #                    cond_mean_pga_eta = corr_T*eta_pgv
        #                    eta_pga = np.random.normal(size=cond_mean_pga_eta.shape,loc=cond_mean_pga_eta,scale=sigma_cond_pga)
        #                    
        #                    # correct for predicted mean and sigma
        #                    samp = samp.multiply(self._IM_pred_dict['rup']['pga'+'_intra'].multiply(eps_pga).expm1() + np.ones((n_event,n_site)))
        #                    samp = samp.multiply(self._IM_pred_dict['rup']['pga'+'_inter'].multiply(eta_pga).expm1() + np.ones((n_event,n_site)))
        #        
        #                # store samples internally
        #                self._IM_dict['pga'].update({i:samp}) # update class
        #            
        #                # store samples locally
        #                save_name = os.path.join(sample_dir,'pga_samp_'+str(i)+'.'+store_file_type)
        #                file_io.store_im_samp(save_name, samp, store_file_type, n_decimals)
        #
        #else:
        #    # get correlations between sites
        #    if flag_spatial_corr is True:
        #        
        #        # compute intra-event correlations between sites for pga and pgv
        #        if method_d == 'jayaram_baker_2009':
        #            geo_cond = kwargs.get('geo_cond',2)
        #            corr_d_intra_pga = proc(d=d, approx_period_IM=T_pga, geo_cond=geo_cond)
        #            corr_d_intra_pgv = proc(d=d, approx_period_IM=T_pgv, geo_cond=geo_cond)
        #        
        #        # inter-event sigma: perfect correlation (=1)
        #        corr_d_inter = np.ones(int(n_site*(n_site+1)/2)) # same correlations for pga and pgv
        #        
        #    else:
        #        # identity matrix
        #        corr_d_intra_pga = np.identity(int(n_site*(n_site+1)/2))
        #        corr_d_intra_pga = corr_d_intra_pga[np.triu_indices(n_site)]
        #        corr_d_intra_pgv = corr_d_intra_pga
        #
        #        # identity matrix
        #        corr_d_inter = corr_d_intra_pga # same correlations for pga and pgv
        #        
        #    # Get correlations between periods, same for intra- and inter- events
        #    if flag_cross_corr is True:
        #
        #        # check input for method to use
        #        method_T = kwargs.get('method_T','baker_jayaram_2008')
        #        proc = getattr(importlib.import_module('im.corr_spectral'),method_T)
        #
        #        # compute intra-event correlations between pga and pgv
        #        if method_T == 'baker_jayaram_2008':
        #            corr_T = np.asarray([proc(T1=approx_period_IM[i], T2=approx_period_IM[j])
        #                                for i in range(n_IM) for j in range(n_IM) if j >= i])
        #        
        #    else:
        #        # set correlations along diagonal to 1 and off-diagonal to 0 (i.e., identity matrix)
        #        corr_T = np.identity(int(n_IM*(n_IM+1)/2))
        #        corr_T = corr_T[np.triu_indices(n_IM)]
        #        
        #    # form correlation matrix for intra-event sigma
        #    cov_intra = addl_fcn_im.get_cov(corr_d_intra_pga, corr_d_intra_pgv, corr_T,
        #                                np.ones(n_site)*pga_intra, np.ones(n_site)*pgv_intra,
        #                                n_site,n_IM)
        #
        #    # form correlation matrix for inter-event 
        #    cov_inter = addl_fcn_im.get_cov(corr_d_inter, corr_d_inter, corr_T,
        #                                np.ones(n_site)*pga_inter, np.ones(n_site)*pgv_inter,
        #                                n_site,n_IM)
        #
    #   #    # calculate total covariance
        #    cov_total = np.asarray(cov_intra + cov_inter)
        #
        #    # store information
        #    self._IM_dict.update({'cov_intra': cov_intra,
        #                        'cov_inter': cov_inter,
        #                        'cov_total': cov_total})
    
    # -----------------------------------------------------------
    #def get_IM_pred2(self, phase_to_run, site_data, gm_tool, gm_pred_dir, ims, 
    #                    rup_meta_file, flag_clear_dict=False, store_file_type='npz', **kwargs):
    #    """
    #    Read and store IM and rupture data from files.
    #    
    #    Parameters
    #    ----------
    #    phase_to_run : int
    #        analysis phase for OpenSRA
    #    site_file : str
    #        file containing the site locations and data
    #    gm_tool : str
    #        tool used to generate intensity measures: currently **RegionalProcessor** or **EQHazard**
    #    gm_dir : str
    #        directory containing the GM files
    #    rup_meta_file : str
    #        full path for the file rupture metadata (mag, r)
    #    ims : str, list
    #        list of **IM** variables to create: **pga** and/or **pgv**
    #    flag_clear_dict : boolean, optional
    #        **True** to clear the calculated demands or **False** to keep them; default = **False**
    #    store_file_type : str, optional
    #        file type used to store the GM predictions (**npz** or **txt**), default = **npz**
    #    
    #    Returns
    #    -------
    #    _IM_pred_dict : dict
    #        stored IM moments, eq. rupture information, and site data, to be used for generating IMs
    #
    #    """
    #    
    #    # Initialize dictionary if it is none
    #    if self._IM_dict is None or flag_clear_dict is True:
    #        self._IM_pred_dict = {}
    #    
    #    if self._src_site_dict is None:
    #        self._src_site_dict = {}
    #        
    #        # read site locations and info
    #        if site_data is not None:
    #            # load and store site locations
    #            self._src_site_dict.update({'site_lon': site_data.get('Longitude').values,
    #                                'site_lat': site_data.get('Latitude').values})
    #            self._src_site_dict.update({'l_seg': site_data.get('l_seg (km)').values})
    #            logging.debug(f"\t\tLoaded site locations and pipe segment length into '_src_site_dict'")
    #        else:
    #            logging.debug(f"\t\tSite data not provided")
    #
    #        # Initialize dictionary for rupture information
    #        keys = ['src','rup','mag','rate']
    #        for key in keys:
    #            self._src_site_dict[key] = None
    #
    #        # load rupture information (source index, rupture index, Mw, rate)
    #        if rup_meta_file is None:
    #            logging.debug(f"\t\tRupture metafile not provided")
    #        
    #        else:
    #            data = file_io.read_rup_meta(rup_meta_file).copy()
    #            for key in keys:
    #                self._src_site_dict[key] = data[key]
    #            logging.debug(f"\t\tLoaded rupture meta data into '_src_site_dict'")
    #            
    #    src = self._src_site_dict['src']
    #    rup = self._src_site_dict['rup']
    #    mag = self._src_site_dict['mag']
    #    rate = self._src_site_dict['rate']
    #
    #    # get dimensions
    #    n_site = len(self._src_site_dict['site_lon'])
    #
    #    #
    #    # if phase_to_run <= 3:
    #    if 'regionalprocessor' in gm_tool.lower():
    #        rup_group = kwargs.get('rup_group',0)
    #        self._IM_pred_dict.update({'rup':file_io.read_gm_pred(gm_pred_dir, rup_group, 
    #                                                                ims, src, rup, mag, rate,
    #                                                                n_site, store_file_type,
    #                                                                phase_to_run).copy()})
    #        logging.debug(f"\t\tLoaded GM predictions for current rupture group into '_IM_pred_dict'")
    #                                                            
    #    elif 'eqhazard' in gm_tool.lower():
    #        # removed outdated code, may not implement again
    #        pass
    #    
    #    # store IM tool used
    #    self._IM_pred_dict.update({'gm_tool':gm_tool})
    #    
    #    # clear variables
    #    src = None
    #    rup = None
    #    r = None
    #    mag = None
    
    
    # -----------------------------------------------------------
    #def sim_IM2(n_samp_im, ims=['pga,pgv'], flag_spatial_corr=False, flag_cross_corr=True, T_pga=0.01, T_pgv=1.0, 
    #            method_d='jayaram_baker_2009', method_T='baker_jayaram_2008', flag_sample_with_sigma_total=False,
    #            sigma_aleatory=None, flag_clear_dict=False, flag_im_sample_exist=False, sample_dir=None, store_file_type='npz', n_decimals=None, **kwargs):
    #    """
    #    Perform multivariate random sampling of **PGA** and **PGV** using means and sigmas for all scenarios. Spatial and spectral orrelations can be applied.
    #
    #    Parameters
    #    ----------
    #    n_samp_im : float
    #        number of samples/realizations for intensity measures
    #    ims : str, list, optional
    #        list of **IM** variables to create: **pga** or **pgv** (default = both)
    #    flag_spatial_corr : boolean, optional
    #        decision on performing correlation between sites (distance); default = True
    #    flag_cross_corr : boolean, optional
    #        decision on performing correlation between periods; default = True (only if **ims** contains **pga** and **pgv**)
    #    T_pga : float, optional
    #        [sec] approximate period for **PGA** to be used in spectral correlation; default = 0.01 sec
    #    T_pgv : float, optional
    #        [sec] approximate period for **PGV** to be used in spectral correlation; default = 1.0 sec
    #    method_d = str, optional
    #        method to use for spatial correlation; default = jayaram_baker_2009
    #    method_T = str, optional
    #        method to use for spectral correlation; default = baker_jayaram_2008
    #    flag_clear_dict : boolean, optional
    #        **True** to clear the calculated demands or **False** to keep them; default = **False**
    #    flag_im_sample_exist : boolean, optional
    #        **True** if IM samples are available or **False** to import statistical parameters and create random variables; default = **False**
    #    sample_dir : str, optional
    #        directory with samples; default = None
    #    store_file_type : str, optional
    #        file type used to store the GM predictions (**npz** or **txt**), default = **npz**
    #    
    #    
    #    For additional parameters, see the respective methods under :func:`im.corr_spatial.py` and :func:`im.corr_spectral.py`
    #
    #    Returns
    #    -------
    #    _IM_dict : dict
    #        stored samples for all intensity measures and rupture scenarios
    #    
    #    """
    #    
    #    # Initialize dictionary if it is none
    #    if self._IM_dict is None or flag_clear_dict is True:
    #        self._IM_dict = {}
    #
    #    #
    #    gm_tool = self._IM_pred_dict.get('gm_tool',None)
    #
    #    # dimensions
    #    if 'event' in gm_tool.lower() or 'calc' in gm_tool.lower():
    #        n_site = len(self._IM_pred_dict['rup'][0]['pga_mean'])
    #        n_event = len(self._IM_pred_dict['rup'])
    #
    #    # make period array for spectral correlations
    #    approx_period_IM = [T_pga,T_pgv]
    #    
    #    # check input for method to use
    #    method_T = kwargs.get('method_T','baker_jayaram_2008')
    #    proc = getattr(importlib.import_module('src.im.corr_spectral'),method_T)
    #
    #    # compute intra-event correlations between pga and pgv
    #    if method_T == 'baker_jayaram_2008':
    #        corr_T = proc(T1=T_pga, T2=T_pgv)
    #
    #    # make list of variables
    #    param_names = ['mean', 'inter', 'intra']
    #    var_list = [i+'_'+j for i in ims for j in param_names]
    #    
    #    #
    #    n_site = len(self._src_site_dict['site_lon'])
    #    n_event = len(self._IM_pred_dict['rup']['src'])
    #    n_IM = len(ims) # number of periods of interest
    #
    #    # check if spatial correlation is required
    #    if flag_spatial_corr is False:
    #        # no correlations, random/LHS sampling
    #        
    #        self._IM_dict.update({'pgv':{}})
    #        self._IM_dict.update({'pga':{}})
    #
    #        # check if samples already exist
    #        if flag_im_sample_exist is True:   # load samples
    #        
    #            # loop through ims
    #            for im in ims:
    #                # loop through and import all samples
    #                for i in range(n_samp_im):
    #                    file_name = os.path.join(sample_dir,im+'_samp_'+str(i)+'.'+store_file_type)
    #                    self._IM_dict[im].update({i:file_io.read_im_samp(file_name, store_file_type, n_event, n_site)})
    #                        
    #        
    #        else:   # perform random sampling
    #
    #            # loop through number of samples
    #            for i in range(n_samp_im):
    #            
    #                # first sample pgv
    #                eps_pgv = np.random.normal(size=(n_event,n_site))
    #                samp = self._IM_pred_dict['rup']['pgv_mean']
    #                
    #                # use total sigma or separate into intra- and inter- event sigmas
    #                if flag_sample_with_sigma_total:
    #                    # correct for sigma
    #                    if sigma_aleatory is None: # if total sigma is not provided
    #                        sigma_total = self._IM_pred_dict['rup']['pgv_intra'].power(2) + self._IM_pred_dict['rup']['pgv_inter'].power(2)
    #                        sigma_total = sigma_total.power(0.5)
    #                        samp = samp.multiply(sigma_total.multiply(eps_pgv).expm1() + np.ones((n_event,n_site)))
    #                    
    #                    else:
    #                        # only supports singular inputs for sigma_aleatory, expand to matrix later
    #                        samp = samp.multiply(np.exp(sigma_aleatory*eps_pgv))
    #                
    #                else:
    #                    # get residuals for intra (epsilon) and inter (eta) (norm dist with mean = 0 and sigma = 1)
    #                    eta_pgv = np.random.normal(size=n_event)
    #                    eta_pgv = np.repeat(eta_pgv[:,np.newaxis],n_site,axis=1) # eta is constant with site, varies only between rupture
    #                    
    #                    # correct for predicted mean and sigma
    #                    samp = samp.multiply(self._IM_pred_dict['rup']['pgv_intra'].multiply(eps_pgv).expm1() + np.ones((n_event,n_site)))
    #                    samp = samp.multiply(self._IM_pred_dict['rup']['pgv_inter'].multiply(eta_pgv).expm1() + np.ones((n_event,n_site)))
    #            
    #                # store samples internally
    #                self._IM_dict['pgv'].update({i:samp}) # update class
    #                
    #                # store samples locally                    
    #                save_name = os.path.join(sample_dir,'pgv_samp_'+str(i)+'.'+store_file_type)
    #                file_io.store_im_samp(save_name, samp, store_file_type, n_decimals)
    #                
    #                # see if 'pga' is needed
    #                if 'pga' in ims:
    #                
    #                    # conditional sigma for pga
    #                    sigma_cond_pga = np.sqrt(1-corr_T**2)
    #                    
    #                    # conditional mean of eps
    #                    cond_mean_pga_eps = corr_T*eps_pgv
    #                    eps_pga = np.random.normal(size=(n_event,n_site),loc=cond_mean_pga_eps,scale=sigma_cond_pga)
    #                    samp = self._IM_pred_dict['rup']['pga_mean']
    #                                    
    #                    # use total sigma or separate into intra- and inter- event sigmas
    #                    if flag_sample_with_sigma_total:
    #                        # correct for sigma
    #                        if sigma_aleatory is None: # if total sigma is not provided
    #                            sigma_total = self._IM_pred_dict['rup']['pga'+'_intra'].power(2) + self._IM_pred_dict['rup']['pga'+'_inter'].power(2)
    #                            sigma_total = sigma_total.power(0.5)
    #                            samp = samp.multiply(sigma_total.multiply(eps_pga).expm1() + np.ones((n_event,n_site)))
    #                        
    #                        else:
    #                            # only supports singular inputs for sigma_aleatory, expand to matrix later
    #                            samp = samp.multiply(np.exp(sigma_aleatory*eps_pga))
    #                            
    #                    else:
    #                        # conditional sampling of eta
    #                        cond_mean_pga_eta = corr_T*eta_pgv
    #                        eta_pga = np.random.normal(size=cond_mean_pga_eta.shape,loc=cond_mean_pga_eta,scale=sigma_cond_pga)
    #                        
    #                        # correct for predicted mean and sigma
    #                        samp = samp.multiply(self._IM_pred_dict['rup']['pga'+'_intra'].multiply(eps_pga).expm1() + np.ones((n_event,n_site)))
    #                        samp = samp.multiply(self._IM_pred_dict['rup']['pga'+'_inter'].multiply(eta_pga).expm1() + np.ones((n_event,n_site)))
    #            
    #                    # store samples internally
    #                    self._IM_dict['pga'].update({i:samp}) # update class
    #                
    #                    # store samples locally
    #                    save_name = os.path.join(sample_dir,'pga_samp_'+str(i)+'.'+store_file_type)
    #                    file_io.store_im_samp(save_name, samp, store_file_type, n_decimals)
    #    
    #    else:
    #        # get correlations between sites
    #        if flag_spatial_corr is True:
    #            
    #            # check input for method to use
    #            method_d = kwargs.get('method_d','jayaram_baker_2009')
    #            proc = getattr(importlib.import_module('im.corr_spatial'),method_d)
    #    
    #            ind1,ind2 = np.triu_indices(n_site)
    #
    #            d = fcn_gen.get_haversine_dist(site_lon[ind1],site_lat[ind1],site_lon[ind2],site_lat[ind2])
    #            
    #            # compute intra-event correlations between sites for pga and pgv
    #            if method_d == 'jayaram_baker_2009':
    #                geo_cond = kwargs.get('geo_cond',2)
    #                corr_d_intra_pga = proc(d=d, approx_period_IM=T_pga, geo_cond=geo_cond)
    #                corr_d_intra_pgv = proc(d=d, approx_period_IM=T_pgv, geo_cond=geo_cond)
    #            
    #            # inter-event sigma: perfect correlation (=1)
    #            corr_d_inter = np.ones(int(n_site*(n_site+1)/2)) # same correlations for pga and pgv
    #            
    #        else:
    #            # identity matrix
    #            corr_d_intra_pga = np.identity(int(n_site*(n_site+1)/2))
    #            corr_d_intra_pga = corr_d_intra_pga[np.triu_indices(n_site)]
    #            corr_d_intra_pgv = corr_d_intra_pga
    #    
    #            # identity matrix
    #            corr_d_inter = corr_d_intra_pga # same correlations for pga and pgv
    #            
    #        # Get correlations between periods, same for intra- and inter- events
    #        if flag_cross_corr is True:
    #    
    #            # check input for method to use
    #            method_T = kwargs.get('method_T','baker_jayaram_2008')
    #            proc = getattr(importlib.import_module('im.corr_spectral'),method_T)
    #    
    #            # compute intra-event correlations between pga and pgv
    #            if method_T == 'baker_jayaram_2008':
    #                corr_T = np.asarray([proc(T1=approx_period_IM[i], T2=approx_period_IM[j])
    #                                    for i in range(n_IM) for j in range(n_IM) if j >= i])
    #            
    #        else:
    #            # set correlations along diagonal to 1 and off-diagonal to 0 (i.e., identity matrix)
    #            corr_T = np.identity(int(n_IM*(n_IM+1)/2))
    #            corr_T = corr_T[np.triu_indices(n_IM)]
    #            
    #        # form correlation matrix for intra-event sigma
    #        cov_intra = addl_fcn_im.get_cov(corr_d_intra_pga, corr_d_intra_pgv, corr_T,
    #                                    np.ones(n_site)*pga_intra, np.ones(n_site)*pgv_intra,
    #                                    n_site,n_IM)
    #    
    #        # form correlation matrix for inter-event 
    #        cov_inter = addl_fcn_im.get_cov(corr_d_inter, corr_d_inter, corr_T,
    #                                    np.ones(n_site)*pga_inter, np.ones(n_site)*pgv_inter,
    #                                    n_site,n_IM)
    #    
    ##       # calculate total covariance
    #        cov_total = np.asarray(cov_intra + cov_inter)
    #
    #        # store information
    #        self._IM_dict.update({'cov_intra': cov_intra,
    #                            'cov_inter': cov_inter,
    #                            'cov_total': cov_total})
        
    
    # -----------------------------------------------------------
    #def assess_EDP(self, edp_category, edp_procs_info, edp_other_params, n_samp_im=1, store_name=None, 
    #                flag_clear_dict=False, **kwargs):
    #    """
    #    Using the simulated intensity measures to calculate engineering demand parameters.
    #
    #    Parameters
    #    ----------
    #    edp_category : str
    #        demand category to calculate; options are **corr_spt**, **liq**, **ls**, **gs**, etc. (see :func:`edp` for all **EDP** categories)
    #    method : str
    #        method/procedure to use to calculate the demand; see :func:`edp` for available methods
    #    return_param : str, list
    #        single of a list of parameters to return, see the return variables under each function (:func:`edp`)
    #    store_name : str, list, optional
    #        names to store parameter as; default = **return_param**
    #    flag_clear_dict : boolean, optional
    #        **True** to clear the calculated demands or **False** to keep them; default = **False**
    #    flag_pga : boolean, optional
    #        **True** include simulated **PGA**; default = **False**
    #    flag_pgv : boolean, optional
    #        **True** include simulated **PGV**; default = **False**
    #    flag_mag : boolean, optional
    #        **True** include moment magnitude **mag** from rupture metadata; default = **False**
    #    flag_rup_depend : boolean, optional
    #        **True** if dependent on rupture scenario; default = **False**
    #    source_dict : str, list, optional
    #        dictionary that contains **source_param** and **source_method**; default = None
    #    source_param : str, list, optional
    #        parameter to get from **existing** stored parameters (e.g., **liq_susc**); default = None
    #    source_method : str, list, optional
    #        method used to obtain the source_param (e.g., **zhu_etal_2017**); default = None
    #    
    #    For input parameters to each method, refer to the method documentation under :func:`edp`.
    #    
    #    Returns
    #    -------
    #    output : varies
    #        [varies] output depends on the target demand and methods.
    #    
    #    """
    #    
    #    # pull other params
    #    method = edp_procs_info.get('method',None)[0]
    #    return_param = edp_procs_info.get('return_param',None)
    #    source_dict = edp_procs_info.get('source_dict',None)
    #    source_param = edp_procs_info.get('source_param',None)
    #    source_method = edp_procs_info.get('source_method',None)
    #    store_name = edp_procs_info.get('store_name',None)
    #    flag_pga = edp_procs_info.get('flag_pga',None)
    #    flag_pgv = edp_procs_info.get('flag_pgv',None)
    #    flag_mag = edp_procs_info.get('flag_mag',None)
    #    
    #    # add params under other_params to kwargs
    #    for key in edp_other_params.keys():
    #        kwargs[key] = edp_other_params.get(key)
    #    
    #    # add return_param into kwargs
    #    kwargs['return_param'] = return_param
    #    if store_name == None:
    #        store_name = return_param
    #        
    #    # add number of im samples to kwargs
    #    kwargs['n_samp_im'] = n_samp_im
    #    
    #    # Initialize dictionary if it is none or if user wishes to clear dict
    #    if self._EDP_dict is None or flag_clear_dict is True:
    #        self._EDP_dict = {}
    #    
    #    # create keys
    #    for i in return_param:
    #        if not i in self._EDP_dict.keys():
    #            self._EDP_dict[i] = {}
    #        
    #    # load method
    #    proc = getattr(importlib.import_module('src.edp.'+edp_category),method)
    #    
    #    # dimensions               
    #    n_site = len(self._src_site_dict['site_lon'])
    #    n_event = len(self._IM_pred_dict['rup']['src'])
    #    kwargs['n_site'] = n_site
    #    kwargs['n_event'] = n_event
    #    
    #    # set output storage
    #    output = {}
    #    
    #    # if source_param is not empty, then get parameter from method
    #    param_add = {}
    #    if source_param is not None:
    #        for i in range(len(source_param)):
    #            try:
    #                source_dict_i = getattr(self,source_dict[i])
    #            except:
    #                logging.info(f"\t\t{source_dict[i]} does not exist")
    #                break
    #            else:
    #                source_param_i = source_dict_i.get(source_param[i],None)
    #                if source_param_i is None:
    #                    logging.info(f"\t\t{source_param[i]} does not exist")
    #                    break
    #                else:
    #                    for j in source_param_i:
    #                        if source_param_i[j].get('method',None) == source_method[i]:
    #                            param_add.update({source_param[i]:source_param_i[j]['output'].copy()})
    #    
    #    # get mag from scenarios
    #    if flag_mag is True:
    #        kwargs['mag'] = self._IM_pred_dict['rup'].get('mag',None)
    #    
    #    # if pga nor pgv is required, only run procedure once
    #    if flag_pga is False and flag_pgv is False:
    #        # add additional parameters into kwargs
    #        if source_param is not None:
    #            for j in source_param:
    #                kwargs[j] = param_add[j].copy()
    #        # run method
    #        out = proc(**kwargs)
    #        output = {}
    #        output.update({'prob_dist':out.get('prob_dist',None)})
    #        for i in return_param:
    #            output.update({i:out[i]})
    #        out = None
    #        
    #    # if either pga or pgv is required, run procedure through all scenarios
    #    else:
    #        # add additional parameters into kwargs
    #        if source_param is not None:
    #            for j in source_param:
    #                kwargs[j] = param_add[j].copy()
    #        # get IM simulations for scenario
    #        if flag_pga is True:
    #            kwargs['pga'] = self._IM_dict['pga']
    #        if flag_pgv is True:
    #            kwargs['pgv'] = self._IM_dict['pgv']
    #
    #        # run method
    #        output = proc(**kwargs)
    #    
    #    # 
    #    eps_epistemic = edp_procs_info.get('eps_epistemic',[0])
    #    eps_aleatory = edp_procs_info.get('eps_aleatory',[0])
    #    wgt_aleatory = edp_procs_info.get('wgt_aleatory',[1])
    #    
    #    # store in dictionary
    #    for i in return_param:
    #        count = len(self._EDP_dict[i]) # see how many methods have been used
    #        
    #        if count == 0:
    #            name = 'method'+str(count+1)
    #            self._EDP_dict[i].update({name: {'method':method,
    #                                            'source_param':source_param,
    #                                            'source_method':source_method,
    #                                            'eps_epistemic':eps_epistemic,
    #                                            'eps_aleatory':eps_aleatory,
    #                                            'wgt_aleatory':wgt_aleatory}})
    #        
    #        else:
    #            name = None
    #            for key in self._EDP_dict[i].keys():
    #                if self._EDP_dict[i][key]['method'] == method and \
    #                    self._EDP_dict[i][key]['source_param'] == source_param and \
    #                    self._EDP_dict[i][key]['source_method'] == source_method and \
    #                    self._EDP_dict[i][key]['eps_epistemic'] == eps_epistemic and \
    #                    self._EDP_dict[i][key]['eps_aleatory'] == eps_aleatory and \
    #                    self._EDP_dict[i][key]['wgt_aleatory'] == wgt_aleatory:
    #                    
    #                    name = key
    #                    break
    #                    
    #            if name is None:
    #                name = 'method'+str(count+1)
    #                self._EDP_dict[i].update({name: {'method':method,
    #                                                'source_param':source_param,
    #                                                'source_method':source_method,
    #                                                'eps_epistemic':eps_epistemic,
    #                                                'eps_aleatory':eps_aleatory,
    #                                                'wgt_aleatory':wgt_aleatory}})
    #        
    #        # store distribution
    #        if 'pgd' in i or 'p_liq' in i: # added check for p_liq, to allow for uncertain in p_liq
    #            self._EDP_dict[i][name].update({'prob_dist': output['prob_dist']})
    #        
    #        # store output
    #        self._EDP_dict[i][name].update({'output': output[i]})
            
            
    # -----------------------------------------------------------
    #def assess_DM(self, dm_category, dm_procs_info, dm_other_params, n_samp_dm, store_name=None, 
    #                flag_clear_dict=False, **kwargs):
    #    """
    #    Using the simulated intensity measures and engineering demand parameters to calculate damage measures.
    #
    #    Parameters
    #    ----------
    #    
    #    For input parameters to each method, refer to the method documentation under :func:`dm`.
    #    
    #    Returns
    #    -------
    #    output : varies
    #        [varies] output depends on the target demand and methods.
    #    
    #    """
        
    # -----------------------------------------------------------
    #def assess_DV(self, dv_category, dv_method, dv_procs_info, dv_other_params, edp=None, n_samp_im=1, n_samp_edp=1,
    #                store_name=None, flag_clear_dict=False, **kwargs):
    #    """
    #    Using the simulated intensity measures, engineering demand parameters, and damage measures to calculate decision variables/values.
    #
    #    Parameters
    #    ----------
    #    dv_category : str
    #        decision variable category to calculate; options are **rr** (more to be added, see :func:`dv` for meaning of the options)
    #    dv_method : str
    #        method/procedure to use to calculate the damage; see :func:`dv` for available methods.
    #    return_param : str, list
    #        single of a list of parameters to return, see the return variables under each function (:func:`dv`)
    #    store_name : str, list, optional
    #        names to store parameter as; default = **return_param**
    #    ims : str, list, optional
    #        list of **IM** variables to create: **pga** or **pgv** (default = both)
    #    flag_clear_dict : boolean, optional
    #        **True** to clear the calculated damages or **False** to keep them; default = **False**
    #    flag_pga : boolean, optional
    #        **True** include simulated **PGA**; default = **False**
    #    flag_pgv : boolean, optional
    #        **True** include simulated **PGV**; default = **False**
    #    flag_mag : boolean, optional
    #        **True** include moment magnitude **mag** from rupture metadata; default = **False**
    #    flag_rup_depend : boolean, optional
    #        **True** if dependent on rupture scenario; default = **False**
    #    source_dict : str, list, optional
    #        dictionary that contains **source_param** and **source_method**; default = None
    #    source_param : str, list, optional
    #        parameter to get from **existing** stored parameters in (e.g., **liq_susc**); default = None
    #    source_method : str, list, optional
    #        method used to obtain the source_param (e.g., **zhu_etal_2017**); default = None
    #    
    #    For method parameters, refer to the method documentation under :func:`dv`.
    #    
    #    Returns
    #    -------
    #    output : varies
    #        [varies] output depends on the target decision variables and methods.
    #    
    #    """
    #    
    #    # pull other params
    #    return_param = dv_procs_info.get('return_param',None)
    #    source_dict = dv_procs_info.get('source_dict',None)
    #    source_param = dv_procs_info.get('source_param',None)
    #    source_method = dv_procs_info.get('source_method',None)
    #    if edp is not None:
    #        source_dict = source_dict.get(edp)
    #        source_param = source_param.get(edp)
    #        source_method = source_method.get(edp)
    #    store_name = dv_procs_info.get('store_name',None)
    #    flag_pga = dv_procs_info.get('flag_pga',False)
    #    flag_pgv = dv_procs_info.get('flag_pgv',False)
    #    flag_mag = dv_procs_info.get('flag_mag',False)
    #    
    #    #
    #    flag_rup_depend = dv_other_params.get('flag_rup_depend',False)
    #    
    #    # add params under other_params to kwargs
    #    for key in dv_other_params.keys():
    #        kwargs[key] = dv_other_params.get(key)
    #    
    #    # add return_param into kwargs
    #    kwargs['return_param'] = return_param
    #    if store_name == None:
    #        store_name = return_param
    #        
    #    # add number of im samples to kwargs
    #    kwargs['n_samp_im'] = n_samp_im
    #    # kwargs['n_samp_im'] = n_samp_im
    #    
    #    # Initialize dictionary if it is none or if user wishes to clear dict
    #    if self._DV_dict is None or flag_clear_dict is True:
    #        self._DV_dict = {}
    #    
    #    # create keys
    #    for i in return_param:
    #        if not i in self._DV_dict.keys():
    #            self._DV_dict[i] = {}
    #        
    #    # load method
    #    proc = getattr(importlib.import_module('src.dv.'+dv_category),dv_method)
    #    
    #    # dimensions
    #    n_site = len(self._src_site_dict['site_lon'])
    #    n_event = len(self._IM_pred_dict['rup']['src'])
    #    kwargs['n_site'] = n_site
    #    kwargs['n_event'] = n_event
    #
    #    # rate
    #    rate = self._IM_pred_dict['rup']['rate']
    #    rate = np.repeat(rate[:,np.newaxis],n_site,axis=1)
    #    
    #    # add l_seg to kwargs
    #    pgd_label = dv_other_params.get('pgd_label',None)
    #    if pgd_label is not None:
    #        if 'surf' in pgd_label and ('ala' in dv_method or 'orourke' in dv_method):
    #            kwargs['l_seg'] = self._src_site_dict['l_seg']
    #    
    #    # if source_param is not empty, then get parameter from method
    #    param_add = {}
    #    if source_param is not None:
    #        for i in range(len(source_param)):
    #            try:
    #                source_dict_i = getattr(self,source_dict[i])
    #            except:
    #                logging.info(f"\t\t{source_dict[i]} does not exist")
    #                break
    #            else:
    #                source_param_i = source_dict_i.get(source_param[i],None)
    #                if source_param_i is None:
    #                    logging.info(f"\t\t{source_param[i]} does not exist")
    #                    break
    #                else:
    #                    for j in source_param_i:
    #                        if source_param_i[j].get('method',None) == source_method[i]:
    #                            param_add.update({source_param[i]:source_param_i[j]['output'].copy()})
    #    
    #    # pull statisical epsilons for uncertainty
    #    eps_epistemic = [0]
    #    eps_aleatory = [0]
    #    wgt_aleatory = [1]
    #    if 'rr_pgd' in return_param:
    #        for param in source_param:
    #            if 'pgd' in param:
    #                eps_epistemic = self._EDP_dict[param]['method1']['eps_epistemic']
    #                eps_aleatory = self._EDP_dict[param]['method1']['eps_aleatory']
    #                wgt_aleatory = self._EDP_dict[param]['method1']['wgt_aleatory']
    #                prob_dist_pgd = self._EDP_dict[param]['method1']['prob_dist']
    #                break
    #    # print(eps_epistemic, eps_aleatory, wgt_aleatory)
    #                
    #    if 'rr_pgv' in return_param:
    #        sigma_epistemic = dv_procs_info.get('sigma_epistemic',None)
    #        eps_epistemic = dv_procs_info.get('eps_epistemic',None)
    #        
    #    # if pga nor pgv are required, only run procedure once
    #    if flag_pga is False and flag_pgv is False and flag_rup_depend is False:
    #        # add additional parameters into kwargs
    #        if source_param is not None:
    #            for param in source_param:
    #                if not 'pgd' in param:
    #                    kwargs[param] = param_add[param].copy()
    #        
    #        # loop through all cases of epistemic branches
    #        for epi_i in range(len(eps_epistemic)):
    #            
    #            # store in dictionary
    #            for i in return_param:
    #                
    #                count = len(self._DV_dict[i]) # see how many methods have been used
    #                if count == 0:
    #                    name = 'method'+str(count+1)
    #                    self._DV_dict[i].update({name: {'method':dv_method,
    #                                                    'source_param':source_param,
    #                                                    'source_method':source_method,
    #                                                    'eps_epistemic':eps_epistemic[epi_i],
    #                                                    'eps_aleatory':eps_aleatory,
    #                                                    'wgt_aleatory':wgt_aleatory}})
    #                
    #                else:
    #                    name = None
    #                    for key in self._DV_dict[i].keys():
    #                        if self._DV_dict[i][key]['method'] == dv_method and \
    #                            self._DV_dict[i][key]['source_param'] == source_param and \
    #                            self._DV_dict[i][key]['source_method'] == source_method and \
    #                            self._DV_dict[i][key]['eps_epistemic'] == eps_epistemic[epi_i]:
    #                        
    #                            name = key
    #                            break
    #                            
    #                    if name is None:
    #                        name = 'method'+str(count+1)
    #                        self._DV_dict[i].update({name: {'method':dv_method,
    #                                                        'source_param':source_param,
    #                                                        'source_method':source_method,
    #                                                        'eps_epistemic':eps_epistemic[epi_i],
    #                                                        'eps_aleatory':eps_aleatory,
    #                                                        'wgt_aleatory':wgt_aleatory}})
    #    
    #            # loop through all cases of aleatory branches
    #            for ale_j in range(len(eps_aleatory)):
    #            
    #                # add additional parameters into kwargs
    #                if source_param is not None:
    #                    for param in source_param:
    #                        if 'pgd' in param:
    #                            # temporarily store pgd mean
    #                            pgd_mean = param_add[param].copy()
    #                            
    #                            # operate on pgd to get realizations based on distribution
    #                            if prob_dist_pgd['type'] == 'lognormal':
    #                                if eps_epistemic[epi_i] == 999: # lumped uncertainty
    #                                    pgd_i_j = pgd_mean.muliply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total']))
    #                                else:
    #                                    pgd_i_j = pgd_mean.muliply(np.exp(
    #                                                    eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
    #                                                    eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic']))
    #                            
    #                            elif prob_dist_pgd['type'] == 'uniform':
    #                                samp_pgd = np.random.rand(len(pgd_mean.data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
    #                                            1/prob_dist_pgd['factor_aleatory']
    #                                samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_mean.row,pgd_mean.col)),shape=pgd_mean.shape)
    #                                pgd_i_j = pgd_mean.multiply(samp_pgd)*prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i]
    #                            
    #                            break
    #    
    #    # if either pga or pgv is required, run procedure through all scenarios
    #    else:
    #    
    #        # add additional parameters into kwargs
    #        if source_param is not None:
    #            for param in source_param:
    #                if not 'pgd' in param:
    #                    kwargs[param] = param_add[param].copy()            
    #        
    #        # loop through all cases of epistemic branches
    #        for epi_i in range(len(eps_epistemic)):
    #            
    #            # if running PGV fragilities
    #            if 'rr_pgv' in return_param:
    #                kwargs['n_samp'] = n_samp_im
    #                pgv_epi = {}
    #                for samp_i in range(n_samp_im):
    #                    pgv_epi.update({samp_i:self._IM_dict['pgv'][samp_i].multiply(np.exp(eps_epistemic[epi_i]*sigma_epistemic))})
    #                kwargs['pgv'] = pgv_epi
    #            
    #            # store in dictionary
    #            for i in return_param:
    #                
    #                count = len(self._DV_dict[i]) # see how many methods have been used
    #                if count == 0:
    #                    name = 'method'+str(count+1)
    #                    self._DV_dict[i].update({name: {'method':dv_method,
    #                                                    'source_param':source_param,
    #                                                    'source_method':source_method,
    #                                                    'eps_epistemic':eps_epistemic[epi_i],
    #                                                    'eps_aleatory':eps_aleatory,
    #                                                    'wgt_aleatory':wgt_aleatory}})
    #                else:
    #                    name = None
    #                    for key in self._DV_dict[i].keys():
    #                        if self._DV_dict[i][key]['method'] == dv_method and \
    #                            self._DV_dict[i][key]['source_param'] == source_param and \
    #                            self._DV_dict[i][key]['source_method'] == source_method and \
    #                            self._DV_dict[i][key]['eps_epistemic'] == eps_epistemic[epi_i]:
    #                        
    #                            name = key
    #                            break
    #                            
    #                    if name is None:
    #                        name = 'method'+str(count+1)
    #                        self._DV_dict[i].update({name: {'method':dv_method,
    #                                                        'source_param':source_param,
    #                                                        'source_method':source_method,
    #                                                        'eps_epistemic':eps_epistemic[epi_i],
    #                                                        'eps_aleatory':eps_aleatory,
    #                                                        'wgt_aleatory':wgt_aleatory}})
    #    
    #            # loop through all cases of aleatory branches
    #            for ale_j in range(len(eps_aleatory)):
    #            
    #                # add additional parameters into kwargs
    #                if source_param is not None:
    #                    for param in source_param:
    #                        if 'pgd' in param:
    #                            kwargs['n_samp'] = n_samp_im
    #                            # temporarily store pgd mean
    #                            pgd_mean = param_add[param].copy()
    #                            
    #                            # pgd_mean's type is dict if it has more than 1 IM sample
    #                            if type(pgd_mean) == dict:
    #                                pgd_i_j = {}
    #                                for k in range(n_samp_im):
    #                                    # operate on pgd to get realizations based on distribution
    #                                    if prob_dist_pgd['type'] == 'lognormal':
    #                                        if eps_epistemic[epi_i] == 999: # lumped uncertainty
    #                                            pgd_k = pgd_mean[k].multiply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])).tocoo()
    #                                            if 'bray_macedo_2019' in source_method:
    #                                                d0 = kwargs.get('d0',0.5)
    #                                                pgd_k.data[pgd_k.data<=d0] = 0
    #                                            pgd_i_j.update({k:pgd_k})
    #                                            # if k == 0:
    #                                                # print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])
    #
    #                                        else:
    #                                            pgd_k = pgd_mean[k].multiply(np.exp(
    #                                                                eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
    #                                                                eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])).tocoo()
    #                                            if 'bray_macedo_2019' in source_method:
    #                                                d0 = kwargs.get('d0',0.5)
    #                                                pgd_k.data[pgd_k.data<=d0] = 0
    #                                            pgd_i_j.update({k:pgd_k})
    #                                            # if k == 0:
    #                                                # print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
    #                                                    # eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])
    #                                        
    #                                    elif prob_dist_pgd['type'] == 'uniform':
    #                                        # samp_pgd = np.random.rand(len(pgd_mean[k].data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
    #                                                    # 1/prob_dist_pgd['factor_aleatory'] # interpolate in linear scale
    #                                        # print(2*np.random.rand(len(pgd_mean[k].data)) - 1)
    #                                        samp_pgd = prob_dist_pgd['factor_aleatory']**(2*np.random.rand(len(pgd_mean[k].data)) - 1) # interpolate in log scale
    #                                        samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_mean[k].row,pgd_mean[k].col)),shape=pgd_mean[k].shape)
    #                                        pgd_i_j.update({k:pgd_mean[k].multiply(samp_pgd).tocoo()*(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])})
    #                                        # if k == 0:                                            
    #                                            # print(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory'],1/prob_dist_pgd['factor_aleatory'])
    #                                            # print(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
    #                        
    #                            else:
    #                                # operate on pgd to get realizations based on distribution
    #                                if prob_dist_pgd['type'] == 'lognormal':
    #                                    if eps_epistemic[epi_i] == 999: # lumped uncertainty
    #                                        pgd_i_j = pgd_mean.multiply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total']))
    #                                        # print('a', ale_j, eps_aleatory[ale_j], prob_dist_pgd['sigma_total'], eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])
    #                                    else:
    #                                        pgd_i_j = pgd_mean.multiply(np.exp(
    #                                                    eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
    #                                                    eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic']))
    #                                        # print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
    #                                                    # eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])
    #                            
    #                                elif prob_dist_pgd['type'] == 'uniform':
    #                                    # samp_pgd = np.random.rand(len(pgd_mean.data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
    #                                                # 1/prob_dist_pgd['factor_aleatory'] # interpolate in linear scale
    #                                    samp_pgd = prob_dist_pgd['factor_aleatory']**(2*np.random.rand(len(pgd_mean.data)) - 1) # interpolate in log scale
    #                                    samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_mean.row,pgd_mean.col)),shape=pgd_mean.shape)
    #                                    pgd_i_j = pgd_mean.multiply(samp_pgd)*(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
    #                                    # print(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory'],1/prob_dist_pgd['factor_aleatory'])
    #                                    # print(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
    #                                
    #                                #
    #                                pgd_i_j = pgd_i_j.tocoo() # convert back to COO matrix
    #                
    #                        # add adjusted pgd to inputs for damage
    #                        kwargs[param] = pgd_i_j
    #                        
    #                        break
    #
    #                # run method
    #                output = proc(**kwargs)
    #                
    #                #
    #                if 'all' in output[i].keys():
    #                    result_curr = sparse.csc_matrix(np.sum(output[i]['all'].multiply(rate).toarray(),axis=0))
    #                else:
    #                    result_curr = sparse.csc_matrix([np.sum(output[i][j].multiply(rate).toarray(),axis=0) for j in range(n_samp_im)])
    #                
    #                #
    #                # print(result_curr.data[:10])
    #                # result_curr = result_curr.power(wgt_aleatory[ale_j]) # weight in log space
    #                # print(result_curr.data[:10])
    #                try:
    #                    result_updated = self._DV_dict[i][name]['output'] + result_curr*wgt_aleatory[ale_j] # linearly weight results
    #                    # result_updated = self._DV_dict[i][name]['output'].multiply(result_curr) # weight in log space
    #                    # print('b', ale_j, wgt_aleatory[ale_j])
    #                except:
    #                    result_updated = result_curr*wgt_aleatory[ale_j] # linearly weight results
    #                    # result_updated = result_curr # weight in log space
    #                    # print('c', ale_j, wgt_aleatory[ale_j])
    #                    
    #                # print(wgt_aleatory[ale_j])
    #        
    #                #
    #                self._DV_dict[i][name].update({'output':result_updated})

    
    # -----------------------------------------------------------
    #def get_probability(self, method_list, **kwargs):
    #
    #    """
    #    Text
    #    
    #    """
    #
    #    logging.info(f"Placeholder function; not working at this time")
    #
    #    return None