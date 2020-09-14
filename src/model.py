#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### Risk model class object
#####
##### Created: April 27, 2020
##### Updated: July 14, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Python modules
import numpy as np
import time, importlib, sys, h5py, os, logging
from scipy import sparse

##### OpenSRA modules and functions
from src import fcn_gen, file_io
from src.im import fcn_im


#####################################################################################################################
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
    

    #############################################################################################################
    ## base directories
    def __init__(self):
        """
        Initialize class object.
        
        Parameters
        ----------
        
        Returns
        -------
    
        """
        ## Source, GM, and site characterization
        self._src_site_dict = None # store rupture sources and site data/params
        self._GM_pred_dict = None # store GMM-predicted IM moments (mean, sigma)
        
        ## Random sampling for IM
        self._IM_dict = None # simulated intensity measures
        
        ## PBEE workflow
        self._EDP_dict = None # assessed engineering demand parameters
        self._DM_dict = None # assessed damage measures
        self._DV_dict = None # assessed decision values/variables
    
        ## results
        self._SUMMARY = None # summary of results
    
    
    #############################################################################################################
    ## create intensity measure random variables and store statistical moments
    def get_src_GM_site(self, phase_to_run, site_data, gm_tool, gm_pred_dir, ims, 
                        rup_meta_file, flag_clear_dict=False, store_file_type='npz', **kwargs):
        """
        Read and store IM and rupture data from files.
        
        Parameters
        ----------
        phase_to_run : int
            analysis phase for OpenSRA
        site_file : str
            file containing the site locations and data
        gm_tool : str
            tool used to generate intensity measures: currently **RegionalProcessor** or **EQHazard**
        gm_dir : str
            directory containing the GM files
        rup_meta_file : str
            full path for the file rupture metadata (M, r)
        ims : str, list
            list of **IM** variables to create: **pga** and/or **pgv**
        flag_im_sample_exist : boolean, optional
            **True** if IM samples are available or **False** to import statistical parameters and create random variables; default = **False**
        flag_clear_dict : boolean, optional
            **True** to clear the calculated demands or **False** to keep them; default = **False**
        store_file_type : str, optional
            file type used to store the GM predictions (**npz** or **txt**), default = **npz**
    
        Additional parameters for **EventCalc** tool
        rate_min : float, optional
            cutoff for mean annual rate; default to 1/1000
        rmax : float, optional
            cutoff distance for searching ruptures and calculating GMs
        flag_export : boolean, optional
            flag whether to simplify the matrix for sigma total and inter and export the reduced-size files
        
        Additional parameters for **RegionalProcessor** tool
        rup_group : str
            name of folder with **IM** to import (e.g., 0_99, 100_199)
        num_rups : int, optional
            first number of ruptures in the group to uploaded, default = None (run all ruptures in group)
        
        Returns
        -------
        _GM_pred_dict : dict
            stored IM moments, eq. rupture information, and site data, to be used for generating IMs
    
        """
        
        ## initialize dictionary if it is none
        if self._IM_dict is None or flag_clear_dict is True:
            self._GM_pred_dict = {}
        
        if self._src_site_dict is None:
            self._src_site_dict = {}
            
            ## read site locations and info
            if site_data is not None:
                ## load and store site locations
                self._src_site_dict.update({'site_lon': site_data.get('Longitude').values,
                                    'site_lat': site_data.get('Latitude').values})
                self._src_site_dict.update({'l_seg': site_data.get('l_seg (km)').values})
                logging.debug(f"\t\tLoaded site locations and pipe segment length into '_src_site_dict'")
            else:
                logging.debug(f"\t\tSite data not provided")

            ## initialize dictionary for rupture information
            keys = ['src','rup','M','rate']
            for key in keys:
                self._src_site_dict[key] = None
            # self._src_site_dict['src'] = None
            # self._src_site_dict['rup'] = None
            # self._src_site_dict['M'] = None
            # self._src_site_dict['rate'] = None

            ## load rupture information (source index, rupture index, Mw, rate)
            if rup_meta_file is None:
                logging.debug(f"\t\tRupture metafile not provided")
                # src = None
                # rup = None
                # M = None
                # rate = None
            
            else:
                data = file_io.read_rup_meta(rup_meta_file).copy()
                for key in keys:
                    self._src_site_dict[key] = data[key]
                # with h5py.File(rup_meta_file, 'r') as f:
                #     self._src_site_dict.update({'src':f.get('src')[:]})
                #     self._src_site_dict.update({'rup':f.get('rup')[:]})
                #     self._src_site_dict.update({'M':f.get('M')[:]})
                #     self._src_site_dict.update({'rate':f.get('rate')[:]})
                # f.close()
                logging.debug(f"\t\tLoaded rupture meta data into '_src_site_dict'")
                
        src = self._src_site_dict['src']
        rup = self._src_site_dict['rup']
        M = self._src_site_dict['M']
        rate = self._src_site_dict['rate']

        ## get dimensions
        n_site = len(self._src_site_dict['site_lon'])

        ##
        # if phase_to_run <= 3:
        if 'regionalprocessor' in gm_tool.lower():
            rup_group = kwargs.get('rup_group',0)
            self._GM_pred_dict.update({'rup':file_io.read_gm_pred(gm_pred_dir, rup_group, 
                                                                    ims, src, rup, M, rate,
                                                                    n_site, store_file_type,
                                                                    phase_to_run).copy()})
            logging.debug(f"\t\tLoaded GM predictions for current rupture group into '_GM_pred_dict'")
                                                                
        elif 'eqhazard' in gm_tool.lower():
            ## removed outdated code, may not implement again
            pass
        
        ## store IM tool used
        self._GM_pred_dict.update({'gm_tool':gm_tool})
        
        ## clear variables
        src = None
        rup = None
        r = None
        M = None
        
        
    #############################################################################################################
    def sim_IM(self, n_samp_im, ims=['pga,pgv'], flag_spatial_corr=False, flag_cross_corr=True, T_pga=0.01, T_pgv=1.0, 
                    method_d='jayaram_baker_2009', method_T='baker_jayaram_2008', flag_sample_with_sigma_total=False,
                    sigma_aleatory=None, flag_clear_dict=False, flag_im_sample_exist=False, sample_dir=None, store_file_type='npz', n_decimals=None, **kwargs):
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
        
        ## initialize dictionary if it is none
        if self._IM_dict is None or flag_clear_dict is True:
            self._IM_dict = {}
    
        ##
        gm_tool = self._GM_pred_dict.get('gm_tool',None)
    
        ## dimensions
        if 'event' in gm_tool.lower() or 'calc' in gm_tool.lower():
            n_site = len(self._GM_pred_dict['rup'][0]['pga_mean'])
            n_rup = len(self._GM_pred_dict['rup'])
    
        ## make period array for spectral correlations
        T_im = [T_pga,T_pgv]
        
        ## check input for method to use
        method_T = kwargs.get('method_T','baker_jayaram_2008')
        proc = getattr(importlib.import_module('src.im.corr_spectral'),method_T)

        ## compute intra-event correlations between pga and pgv
        if method_T == 'baker_jayaram_2008':
            corr_T = proc(T1=T_pga, T2=T_pgv)

        ## make list of variables
        param_names = ['mean', 'inter', 'intra']
        var_list = [i+'_'+j for i in ims for j in param_names]
        
        ##
        n_site = len(self._src_site_dict['site_lon'])
        n_rup = len(self._GM_pred_dict['rup']['src'])
        n_T = len(ims) # number of periods of interest

        ## check if spatial correlation is required
        if flag_spatial_corr is False:
            ## no correlations, random/LHS sampling
            
            self._IM_dict.update({'pgv':{}})
            self._IM_dict.update({'pga':{}})

            ## check if samples already exist
            if flag_im_sample_exist is True:   # load samples
            
                ## loop through ims
                for im in ims:
                    ## loop through and import all samples
                    for i in range(n_samp_im):
                        file_name = os.path.join(sample_dir,im+'_samp_'+str(i)+'.'+store_file_type)
                        self._IM_dict[im].update({i:file_io.read_im_samp(file_name, store_file_type, n_rup, n_site)})
                            
            
            else:   # perform random sampling

                ## loop through number of samples
                for i in range(n_samp_im):
                
                    ## first sample pgv
                    eps_pgv = np.random.normal(size=(n_rup,n_site))
                    samp = self._GM_pred_dict['rup']['pgv_mean']
                    
                    ## use total sigma or separate into intra- and inter- event sigmas
                    if flag_sample_with_sigma_total:
                        ## correct for sigma
                        if sigma_aleatory is None: ## if total sigma is not provided
                            sigma_total = self._GM_pred_dict['rup']['pgv_intra'].power(2) + self._GM_pred_dict['rup']['pgv_inter'].power(2)
                            sigma_total = sigma_total.power(0.5)
                            samp = samp.multiply(sigma_total.multiply(eps_pgv).expm1() + np.ones((n_rup,n_site)))
                        
                        else:
                            # only supports singular inputs for sigma_aleatory, expand to matrix later
                            samp = samp.multiply(np.exp(sigma_aleatory*eps_pgv))
                    
                    else:
                        ## get residuals for intra (epsilon) and inter (eta) (norm dist with mean = 0 and sigma = 1)
                        eta_pgv = np.random.normal(size=n_rup)
                        eta_pgv = np.repeat(eta_pgv[:,np.newaxis],n_site,axis=1) ## eta is constant with site, varies only between rupture
                        
                        ## correct for predicted mean and sigma
                        samp = samp.multiply(self._GM_pred_dict['rup']['pgv_intra'].multiply(eps_pgv).expm1() + np.ones((n_rup,n_site)))
                        samp = samp.multiply(self._GM_pred_dict['rup']['pgv_inter'].multiply(eta_pgv).expm1() + np.ones((n_rup,n_site)))
                
                    ## store samples internally
                    self._IM_dict['pgv'].update({i:samp}) # update class
                    
                    ## store samples locally                    
                    save_name = os.path.join(sample_dir,'pgv_samp_'+str(i)+'.'+store_file_type)
                    file_io.store_im_samp(save_name, samp, store_file_type, n_decimals)
                    
                    ## see if 'pga' is needed
                    if 'pga' in ims:
                    
                        ## conditional sigma for pga
                        sigma_cond_pga = np.sqrt(1-corr_T**2)
                        
                        ## conditional mean of eps
                        cond_mean_pga_eps = corr_T*eps_pgv
                        eps_pga = np.random.normal(size=(n_rup,n_site),loc=cond_mean_pga_eps,scale=sigma_cond_pga)
                        samp = self._GM_pred_dict['rup']['pga_mean']
                                        
                        ## use total sigma or separate into intra- and inter- event sigmas
                        if flag_sample_with_sigma_total:
                            ## correct for sigma
                            if sigma_aleatory is None: ## if total sigma is not provided
                                sigma_total = self._GM_pred_dict['rup']['pga'+'_intra'].power(2) + self._GM_pred_dict['rup']['pga'+'_inter'].power(2)
                                sigma_total = sigma_total.power(0.5)
                                samp = samp.multiply(sigma_total.multiply(eps_pga).expm1() + np.ones((n_rup,n_site)))
                            
                            else:
                                # only supports singular inputs for sigma_aleatory, expand to matrix later
                                samp = samp.multiply(np.exp(sigma_aleatory*eps_pga))
                                
                        else:
                            ## conditional sampling of eta
                            cond_mean_pga_eta = corr_T*eta_pgv
                            eta_pga = np.random.normal(size=cond_mean_pga_eta.shape,loc=cond_mean_pga_eta,scale=sigma_cond_pga)
                            
                            ## correct for predicted mean and sigma
                            samp = samp.multiply(self._GM_pred_dict['rup']['pga'+'_intra'].multiply(eps_pga).expm1() + np.ones((n_rup,n_site)))
                            samp = samp.multiply(self._GM_pred_dict['rup']['pga'+'_inter'].multiply(eta_pga).expm1() + np.ones((n_rup,n_site)))
                
                        ## store samples internally
                        self._IM_dict['pga'].update({i:samp}) # update class
                    
                        ## store samples locally
                        save_name = os.path.join(sample_dir,'pga_samp_'+str(i)+'.'+store_file_type)
                        file_io.store_im_samp(save_name, samp, store_file_type, n_decimals)
        
        else:
            ## get correlations between sites
            if flag_spatial_corr is True:
                
                ## check input for method to use
                method_d = kwargs.get('method_d','jayaram_baker_2009')
                proc = getattr(importlib.import_module('im.corr_spatial'),method_d)
        
                ind1,ind2 = np.triu_indices(n_site)

                d = fcn_gen.get_haversine_dist(site_lon[ind1],site_lat[ind1],site_lon[ind2],site_lat[ind2])
                
                ## compute intra-event correlations between sites for pga and pgv
                if method_d == 'jayaram_baker_2009':
                    geo_cond = kwargs.get('geo_cond',2)
                    corr_d_intra_pga = proc(d=d, T_im=T_pga, geo_cond=geo_cond)
                    corr_d_intra_pgv = proc(d=d, T_im=T_pgv, geo_cond=geo_cond)
                
                ## inter-event sigma: perfect correlation (=1)
                corr_d_inter = np.ones(int(n_site*(n_site+1)/2)) ## same correlations for pga and pgv
                
            else:
                ## identity matrix
                corr_d_intra_pga = np.identity(int(n_site*(n_site+1)/2))
                corr_d_intra_pga = corr_d_intra_pga[np.triu_indices(n_site)]
                corr_d_intra_pgv = corr_d_intra_pga
        
                ## identity matrix
                corr_d_inter = corr_d_intra_pga ## same correlations for pga and pgv
                
            # Get correlations between periods, same for intra- and inter- events
            if flag_cross_corr is True:
        
                ## check input for method to use
                method_T = kwargs.get('method_T','baker_jayaram_2008')
                proc = getattr(importlib.import_module('im.corr_spectral'),method_T)
        
                ## compute intra-event correlations between pga and pgv
                if method_T == 'baker_jayaram_2008':
                    corr_T = np.asarray([proc(T1=T_im[i], T2=T_im[j])
                                        for i in range(n_T) for j in range(n_T) if j >= i])
                
            else:
                ## set correlations along diagonal to 1 and off-diagonal to 0 (i.e., identity matrix)
                corr_T = np.identity(int(n_T*(n_T+1)/2))
                corr_T = corr_T[np.triu_indices(n_T)]
                
            ## form correlation matrix for intra-event sigma
            cov_intra = fcn_im.get_cov(corr_d_intra_pga, corr_d_intra_pgv, corr_T,
                                        np.ones(n_site)*pga_intra, np.ones(n_site)*pgv_intra,
                                        n_site,n_T)
        
            ## form correlation matrix for inter-event 
            cov_inter = fcn_im.get_cov(corr_d_inter, corr_d_inter, corr_T,
                                        np.ones(n_site)*pga_inter, np.ones(n_site)*pgv_inter,
                                        n_site,n_T)
        
    #       ## calculate total covariance
            cov_total = np.asarray(cov_intra + cov_inter)

            ## store information
            self._IM_dict.update({'cov_intra': cov_intra,
                                'cov_inter': cov_inter,
                                'cov_total': cov_total})
        
    
    #############################################################################################################
    def assess_EDP(self, edp_category, edp_procs_info, edp_other_params, n_samp_im=1, store_name=None, 
                    flag_clear_dict=False, **kwargs):
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
        flag_M : boolean, optional
            **True** include moment magnitude **M** from rupture metadata; default = **False**
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
        
        ## pull other params
        method = edp_procs_info.get('method',None)[0]
        return_param = edp_procs_info.get('return_param',None)
        source_dict = edp_procs_info.get('source_dict',None)
        source_param = edp_procs_info.get('source_param',None)
        source_method = edp_procs_info.get('source_method',None)
        store_name = edp_procs_info.get('store_name',None)
        flag_pga = edp_procs_info.get('flag_pga',None)
        flag_pgv = edp_procs_info.get('flag_pgv',None)
        flag_M = edp_procs_info.get('flag_M',None)
        
        ## add params under other_params to kwargs
        for key in edp_other_params.keys():
            kwargs[key] = edp_other_params.get(key)
        
        ## add return_param into kwargs
        kwargs['return_param'] = return_param
        if store_name == None:
            store_name = return_param
            
        ## add number of im samples to kwargs
        kwargs['n_samp_im'] = n_samp_im
        
        ## initialize dictionary if it is none or if user wishes to clear dict
        if self._EDP_dict is None or flag_clear_dict is True:
            self._EDP_dict = {}
        
        ## create keys
        for i in return_param:
            if not i in self._EDP_dict.keys():
                self._EDP_dict[i] = {}
            
        ## load method
        proc = getattr(importlib.import_module('src.edp.'+edp_category),method)
        
        ## dimensions               
        n_site = len(self._src_site_dict['site_lon'])
        n_rup = len(self._GM_pred_dict['rup']['src'])
        kwargs['n_site'] = n_site
        kwargs['n_rup'] = n_rup
        
        ## set output storage
        output = {}
        
        ## if source_param is not empty, then get parameter from method
        param_add = {}
        if source_param is not None:
            for i in range(len(source_param)):
                try:
                    source_dict_i = getattr(self,source_dict[i])
                except:
                    logging.info(f"\t\t{source_dict[i]} does not exist")
                    break
                else:
                    source_param_i = source_dict_i.get(source_param[i],None)
                    if source_param_i is None:
                        logging.info(f"\t\t{source_param[i]} does not exist")
                        break
                    else:
                        for j in source_param_i:
                            if source_param_i[j].get('method',None) == source_method[i]:
                                param_add.update({source_param[i]:source_param_i[j]['output'].copy()})
        
        ## get M from scenarios
        if flag_M is True:
            kwargs['M'] = self._GM_pred_dict['rup'].get('M',None)
        
        ## if pga nor pgv is required, only run procedure once
        if flag_pga is False and flag_pgv is False:
            ## add additional parameters into kwargs
            if source_param is not None:
                for j in source_param:
                    kwargs[j] = param_add[j].copy()
            ## run method
            out = proc(**kwargs)
            output = {}
            output.update({'prob_dist':out.get('prob_dist',None)})
            for i in return_param:
                output.update({i:out[i]})
            out = None
            
        ## if either pga or pgv is required, run procedure through all scenarios
        else:
            ## add additional parameters into kwargs
            if source_param is not None:
                for j in source_param:
                    kwargs[j] = param_add[j].copy()
            ## get IM simulations for scenario
            if flag_pga is True:
                kwargs['pga'] = self._IM_dict['pga']
            if flag_pgv is True:
                kwargs['pgv'] = self._IM_dict['pgv']

            ## run method
            output = proc(**kwargs)
        
        ## 
        eps_epistemic = edp_procs_info.get('eps_epistemic',[0])
        eps_aleatory = edp_procs_info.get('eps_aleatory',[0])
        wgt_aleatory = edp_procs_info.get('wgt_aleatory',[1])
        
        ## store in dictionary
        for i in return_param:
            count = len(self._EDP_dict[i]) # see how many methods have been used
            
            if count == 0:
                name = 'method'+str(count+1)
                self._EDP_dict[i].update({name: {'method':method,
                                                'source_param':source_param,
                                                'source_method':source_method,
                                                'eps_epistemic':eps_epistemic,
                                                'eps_aleatory':eps_aleatory,
                                                'wgt_aleatory':wgt_aleatory}})
            
            else:
                name = None
                for key in self._EDP_dict[i].keys():
                    if self._EDP_dict[i][key]['method'] == method and \
                        self._EDP_dict[i][key]['source_param'] == source_param and \
                        self._EDP_dict[i][key]['source_method'] == source_method and \
                        self._EDP_dict[i][key]['eps_epistemic'] == eps_epistemic and \
                        self._EDP_dict[i][key]['eps_aleatory'] == eps_aleatory and \
                        self._EDP_dict[i][key]['wgt_aleatory'] == wgt_aleatory:
                        
                        name = key
                        break
                        
                if name is None:
                    name = 'method'+str(count+1)
                    self._EDP_dict[i].update({name: {'method':method,
                                                    'source_param':source_param,
                                                    'source_method':source_method,
                                                    'eps_epistemic':eps_epistemic,
                                                    'eps_aleatory':eps_aleatory,
                                                    'wgt_aleatory':wgt_aleatory}})
            
            ## store distribution
            if 'pgd' in i:
                self._EDP_dict[i][name].update({'prob_dist': output['prob_dist']})
            
            ## store output
            self._EDP_dict[i][name].update({'output': output[i]})
            
            
    #############################################################################################################
    def assess_DM(self, dm_category, dm_procs_info, dm_other_params, n_samp_dm, store_name=None, 
                    flag_clear_dict=False, **kwargs):
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
        
    #############################################################################################################
    def assess_DV(self, dv_category, dv_method, dv_procs_info, dv_other_params, edp=None, n_samp_im=1, n_samp_edp=1,
                    store_name=None, flag_clear_dict=False, **kwargs):
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
        flag_M : boolean, optional
            **True** include moment magnitude **M** from rupture metadata; default = **False**
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
        
        ## pull other params
        return_param = dv_procs_info.get('return_param',None)
        source_dict = dv_procs_info.get('source_dict',None)
        source_param = dv_procs_info.get('source_param',None)
        source_method = dv_procs_info.get('source_method',None)
        if edp is not None:
            source_dict = source_dict.get(edp)
            source_param = source_param.get(edp)
            source_method = source_method.get(edp)
        store_name = dv_procs_info.get('store_name',None)
        flag_pga = dv_procs_info.get('flag_pga',False)
        flag_pgv = dv_procs_info.get('flag_pgv',False)
        flag_M = dv_procs_info.get('flag_M',False)
        
        ##
        flag_rup_depend = dv_other_params.get('flag_rup_depend',False)
        
        ## add params under other_params to kwargs
        for key in dv_other_params.keys():
            kwargs[key] = dv_other_params.get(key)
        
        ## add return_param into kwargs
        kwargs['return_param'] = return_param
        if store_name == None:
            store_name = return_param
            
        ## add number of im samples to kwargs
        kwargs['n_samp_im'] = n_samp_im
        # kwargs['n_samp_im'] = n_samp_im
        
        ## initialize dictionary if it is none or if user wishes to clear dict
        if self._DV_dict is None or flag_clear_dict is True:
            self._DV_dict = {}
        
        ## create keys
        for i in return_param:
            if not i in self._DV_dict.keys():
                self._DV_dict[i] = {}
            
        ## load method
        proc = getattr(importlib.import_module('src.dv.'+dv_category),dv_method)
        
        ## dimensions
        n_site = len(self._src_site_dict['site_lon'])
        n_rup = len(self._GM_pred_dict['rup']['src'])
        kwargs['n_site'] = n_site
        kwargs['n_rup'] = n_rup
    
        ## rate
        rate = self._GM_pred_dict['rup']['rate']
        rate = np.repeat(rate[:,np.newaxis],n_site,axis=1)
        
        ## add l_seg to kwargs
        pgd_label = dv_other_params.get('pgd_label',None)
        if pgd_label is not None:
            if 'surf' in pgd_label and ('ala' in dv_method or 'orourke' in dv_method):
                kwargs['l_seg'] = self._src_site_dict['l_seg']
        
        ## if source_param is not empty, then get parameter from method
        param_add = {}
        if source_param is not None:
            for i in range(len(source_param)):
                try:
                    source_dict_i = getattr(self,source_dict[i])
                except:
                    logging.info(f"\t\t{source_dict[i]} does not exist")
                    break
                else:
                    source_param_i = source_dict_i.get(source_param[i],None)
                    if source_param_i is None:
                        logging.info(f"\t\t{source_param[i]} does not exist")
                        break
                    else:
                        for j in source_param_i:
                            if source_param_i[j].get('method',None) == source_method[i]:
                                param_add.update({source_param[i]:source_param_i[j]['output'].copy()})
        
        ## pull statisical epsilons for uncertainty
        eps_epistemic = [0]
        eps_aleatory = [0]
        wgt_aleatory = [1]
        if 'rr_pgd' in return_param:
            for param in source_param:
                if 'pgd' in param:
                    eps_epistemic = self._EDP_dict[param]['method1']['eps_epistemic']
                    eps_aleatory = self._EDP_dict[param]['method1']['eps_aleatory']
                    wgt_aleatory = self._EDP_dict[param]['method1']['wgt_aleatory']
                    prob_dist_pgd = self._EDP_dict[param]['method1']['prob_dist']
                    break
        # print(eps_epistemic, eps_aleatory, wgt_aleatory)
                    
        if 'rr_pgv' in return_param:
            sigma_epistemic = dv_procs_info.get('sigma_epistemic',None)
            eps_epistemic = dv_procs_info.get('eps_epistemic',None)
            
        ## if pga nor pgv are required, only run procedure once
        if flag_pga is False and flag_pgv is False and flag_rup_depend is False:
            ## add additional parameters into kwargs
            if source_param is not None:
                for param in source_param:
                    if not 'pgd' in param:
                        kwargs[param] = param_add[param].copy()
            
            ## loop through all cases of epistemic branches
            for epi_i in range(len(eps_epistemic)):
                
                ## store in dictionary
                for i in return_param:
                    
                    count = len(self._DV_dict[i]) # see how many methods have been used
                    if count == 0:
                        name = 'method'+str(count+1)
                        self._DV_dict[i].update({name: {'method':dv_method,
                                                        'source_param':source_param,
                                                        'source_method':source_method,
                                                        'eps_epistemic':eps_epistemic[epi_i],
                                                        'eps_aleatory':eps_aleatory,
                                                        'wgt_aleatory':wgt_aleatory}})
                    
                    else:
                        name = None
                        for key in self._DV_dict[i].keys():
                            if self._DV_dict[i][key]['method'] == dv_method and \
                                self._DV_dict[i][key]['source_param'] == source_param and \
                                self._DV_dict[i][key]['source_method'] == source_method and \
                                self._DV_dict[i][key]['eps_epistemic'] == eps_epistemic[epi_i]:
                            
                                name = key
                                break
                                
                        if name is None:
                            name = 'method'+str(count+1)
                            self._DV_dict[i].update({name: {'method':dv_method,
                                                            'source_param':source_param,
                                                            'source_method':source_method,
                                                            'eps_epistemic':eps_epistemic[epi_i],
                                                            'eps_aleatory':eps_aleatory,
                                                            'wgt_aleatory':wgt_aleatory}})
        
                ## loop through all cases of aleatory branches
                for ale_j in range(len(eps_aleatory)):
                
                    ## add additional parameters into kwargs
                    if source_param is not None:
                        for param in source_param:
                            if 'pgd' in param:
                                ## temporarily store pgd mean
                                pgd_mean = param_add[param].copy()
                                
                                ## operate on pgd to get realizations based on distribution
                                if prob_dist_pgd['type'] == 'lognormal':
                                    if eps_epistemic[epi_i] == 999: # lumped uncertainty
                                        pgd_i_j = pgd_mean.muliply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total']))
                                    else:
                                        pgd_i_j = pgd_mean.muliply(np.exp(
                                                        eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
                                                        eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic']))
                                
                                elif prob_dist_pgd['type'] == 'uniform':
                                    samp_pgd = np.random.rand(len(pgd_mean.data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
                                                1/prob_dist_pgd['factor_aleatory']
                                    samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_mean.row,pgd_mean.col)),shape=pgd_mean.shape)
                                    pgd_i_j = pgd_mean.multiply(samp_pgd)*prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i]
                                
                                break
        
        ## if either pga or pgv is required, run procedure through all scenarios
        else:
        
            ## add additional parameters into kwargs
            if source_param is not None:
                for param in source_param:
                    if not 'pgd' in param:
                        kwargs[param] = param_add[param].copy()            
            
            ## loop through all cases of epistemic branches
            for epi_i in range(len(eps_epistemic)):
                
                ## if running PGV fragilities
                if 'rr_pgv' in return_param:
                    kwargs['n_samp'] = n_samp_im
                    pgv_epi = {}
                    for samp_i in range(n_samp_im):
                        pgv_epi.update({samp_i:self._IM_dict['pgv'][samp_i].multiply(np.exp(eps_epistemic[epi_i]*sigma_epistemic))})
                    kwargs['pgv'] = pgv_epi
                
                ## store in dictionary
                for i in return_param:
                    
                    count = len(self._DV_dict[i]) # see how many methods have been used
                    if count == 0:
                        name = 'method'+str(count+1)
                        self._DV_dict[i].update({name: {'method':dv_method,
                                                        'source_param':source_param,
                                                        'source_method':source_method,
                                                        'eps_epistemic':eps_epistemic[epi_i],
                                                        'eps_aleatory':eps_aleatory,
                                                        'wgt_aleatory':wgt_aleatory}})
                    else:
                        name = None
                        for key in self._DV_dict[i].keys():
                            if self._DV_dict[i][key]['method'] == dv_method and \
                                self._DV_dict[i][key]['source_param'] == source_param and \
                                self._DV_dict[i][key]['source_method'] == source_method and \
                                self._DV_dict[i][key]['eps_epistemic'] == eps_epistemic[epi_i]:
                            
                                name = key
                                break
                                
                        if name is None:
                            name = 'method'+str(count+1)
                            self._DV_dict[i].update({name: {'method':dv_method,
                                                            'source_param':source_param,
                                                            'source_method':source_method,
                                                            'eps_epistemic':eps_epistemic[epi_i],
                                                            'eps_aleatory':eps_aleatory,
                                                            'wgt_aleatory':wgt_aleatory}})
        
                ## loop through all cases of aleatory branches
                for ale_j in range(len(eps_aleatory)):
                
                    ## add additional parameters into kwargs
                    if source_param is not None:
                        for param in source_param:
                            if 'pgd' in param:
                                kwargs['n_samp'] = n_samp_im
                                ## temporarily store pgd mean
                                pgd_mean = param_add[param].copy()
                                
                                ## pgd_mean's type is dict if it has more than 1 IM sample
                                if type(pgd_mean) == dict:
                                    pgd_i_j = {}
                                    for k in range(n_samp_im):
                                        ## operate on pgd to get realizations based on distribution
                                        if prob_dist_pgd['type'] == 'lognormal':
                                            if eps_epistemic[epi_i] == 999: # lumped uncertainty
                                                pgd_k = pgd_mean[k].multiply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])).tocoo()
                                                if 'bray_macedo_2019' in source_method:
                                                    d0 = kwargs.get('d0',0.5)
                                                    pgd_k.data[pgd_k.data<=d0] = 0
                                                pgd_i_j.update({k:pgd_k})
                                                # if k == 0:
                                                    # print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])

                                            else:
                                                pgd_k = pgd_mean[k].multiply(np.exp(
                                                                    eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
                                                                    eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])).tocoo()
                                                if 'bray_macedo_2019' in source_method:
                                                    d0 = kwargs.get('d0',0.5)
                                                    pgd_k.data[pgd_k.data<=d0] = 0
                                                pgd_i_j.update({k:pgd_k})
                                                # if k == 0:
                                                    # print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
                                                        # eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])
                                            
                                        elif prob_dist_pgd['type'] == 'uniform':
                                            # samp_pgd = np.random.rand(len(pgd_mean[k].data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
                                                        # 1/prob_dist_pgd['factor_aleatory'] # interpolate in linear scale
                                            # print(2*np.random.rand(len(pgd_mean[k].data)) - 1)
                                            samp_pgd = prob_dist_pgd['factor_aleatory']**(2*np.random.rand(len(pgd_mean[k].data)) - 1) # interpolate in log scale
                                            samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_mean[k].row,pgd_mean[k].col)),shape=pgd_mean[k].shape)
                                            pgd_i_j.update({k:pgd_mean[k].multiply(samp_pgd).tocoo()*(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])})
                                            # if k == 0:                                            
                                                # print(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory'],1/prob_dist_pgd['factor_aleatory'])
                                                # print(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
                            
                                else:
                                    ## operate on pgd to get realizations based on distribution
                                    if prob_dist_pgd['type'] == 'lognormal':
                                        if eps_epistemic[epi_i] == 999: # lumped uncertainty
                                            pgd_i_j = pgd_mean.multiply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total']))
                                            # print('a', ale_j, eps_aleatory[ale_j], prob_dist_pgd['sigma_total'], eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])
                                        else:
                                            pgd_i_j = pgd_mean.multiply(np.exp(
                                                        eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
                                                        eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic']))
                                            # print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
                                                        # eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])
                                
                                    elif prob_dist_pgd['type'] == 'uniform':
                                        # samp_pgd = np.random.rand(len(pgd_mean.data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
                                                    # 1/prob_dist_pgd['factor_aleatory'] # interpolate in linear scale
                                        samp_pgd = prob_dist_pgd['factor_aleatory']**(2*np.random.rand(len(pgd_mean.data)) - 1) # interpolate in log scale
                                        samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_mean.row,pgd_mean.col)),shape=pgd_mean.shape)
                                        pgd_i_j = pgd_mean.multiply(samp_pgd)*(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
                                        # print(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory'],1/prob_dist_pgd['factor_aleatory'])
                                        # print(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
                                    
                                    ##
                                    pgd_i_j = pgd_i_j.tocoo() # convert back to COO matrix
                    
                            ## add adjusted pgd to inputs for damage
                            kwargs[param] = pgd_i_j
                            
                            break

                    ## run method
                    output = proc(**kwargs)
                    
                    ##
                    if 'all' in output[i].keys():
                        result_curr = sparse.csc_matrix(np.sum(output[i]['all'].multiply(rate).toarray(),axis=0))
                    else:
                        result_curr = sparse.csc_matrix([np.sum(output[i][j].multiply(rate).toarray(),axis=0) for j in range(n_samp_im)])
                    
                    ##
                    # print(result_curr.data[:10])
                    # result_curr = result_curr.power(wgt_aleatory[ale_j]) # weight in log space
                    # print(result_curr.data[:10])
                    try:
                        result_updated = self._DV_dict[i][name]['output'] + result_curr*wgt_aleatory[ale_j] # linearly weight results
                        # result_updated = self._DV_dict[i][name]['output'].multiply(result_curr) # weight in log space
                        # print('b', ale_j, wgt_aleatory[ale_j])
                    except:
                        result_updated = result_curr*wgt_aleatory[ale_j] # linearly weight results
                        # result_updated = result_curr # weight in log space
                        # print('c', ale_j, wgt_aleatory[ale_j])
                        
                    # print(wgt_aleatory[ale_j])
            
                    ##
                    self._DV_dict[i][name].update({'output':result_updated})

    
    #############################################################################################################
    def get_probability(self, method_list, **kwargs):
    
        return None
#        n_site = sum([1 for i in A._DM_dict.keys() if 'site_' in i])
#        
#        ## initialize arrays for getting probabilities
#        pga_arr_mid = [round((pga_arr[i]+pga_arr[i+1])/2,2) for i in range(len(pga_arr)-1)]
#        pgv_arr_mid = [round((pgv_arr[i]+pgv_arr[i+1])/2,2) for i in range(len(pgv_arr)-1)]
#        
##         pga_mean = np.mean(np.log(pga[i])) # get lognormal mean
##         pga_std = np.std(np.log(pga[i])) # get lognormal standard deviation
##         pga_width = 0.2 # g, width about target to search
##         pga_criteria = [np.exp(pga_mean),np.exp(pga_mean)-pga_width,np.exp(pga_mean)+pga_width]
#        pga_width = 0.03 # g, width about target to search
##         pga_criteria = [0.5,0.5-pga_width,0.5+pga_width]
#        pga_criteria = np.multiply(pga_cond,[1,0.85,1.15])
#        pgv_width = 5 # cm/s, width about target to search
##         pgv_criteria = [70,70-pgv_width,70+pgv_width]
#        pgv_criteria = np.multiply(pgv_cond,[1,0.85,1.15])
#        
#        rr_criteria = rr_cond
#        
#        self._FR_dict.update({'rr_arr': rr_arr,
#                              'pga_arr': pga_arr_mid,
#                              'pgv_arr': pgv_arr_mid,
#                              'rr_criteria': rr_criteria,
#                              'pga_criteria': pga_criteria[0],
#                              'pgv_criteria': pgv_criteria[0]})
#        
#        # Develop fragility curve for each site
#        for i in range(n_site):
#        
#            # pull previously calculated damage values
#            pga = self._DM_dict['site_'+str(i+1)]['pga']
#            pgv = self._DM_dict['site_'+str(i+1)]['pgv']
#            rr_leak = self._DM_dict['site_'+str(i+1)]['rr_leak']
#            rr_break = self._DM_dict['site_'+str(i+1)]['rr_break']
#            
#            # get probability of leaks and breaks
#            p_leak,_ = count_prob_exceed(rr_arr,rr_leak)
#            p_break,_ = count_prob_exceed(rr_arr,rr_break)
#            
##             # get probability of leaks and breaks given pga or pgv
##             ind4cond_g_pga = np.where((pga>pga_criteria[1]) & (pga<pga_criteria[2]))
##             ind4cond_g_pgv = np.where((pgv>pgv_criteria[1]) & (pgv<pgv_criteria[2]))
#            
##             p_leak_g_pga = get_cond_prob_exceed(np.exp(pgv_arr), pgv[ind4cond_g_pga], rr_leak[ind4cond_g_pga], rr_criteria[0])
##             p_leak_g_pgv = get_cond_prob_exceed(np.exp(pga_arr), pga[ind4cond_g_pgv], rr_leak[ind4cond_g_pgv], rr_criteria[0])
##             p_break_g_pga = get_cond_prob_exceed(np.exp(pgv_arr), pgv[ind4cond_g_pga], rr_break[ind4cond_g_pga], rr_criteria[0])
##             p_break_g_pgv = get_cond_prob_exceed(np.exp(pga_arr), pga[ind4cond_g_pgv], rr_break[ind4cond_g_pgv], rr_criteria[0])
#
#            p_leak_g_pga = np.zeros(len(pgv_arr)-1)
#            p_leak_g_pgv = np.zeros(len(pgv_arr)-1)
#            p_break_g_pga = np.zeros(len(pgv_arr)-1)
#            p_break_g_pgv = np.zeros(len(pgv_arr)-1)
#            
#            # store information
#            self._FR_dict.update({'site_'+str(i+1): {'p_leak': p_leak,
#                                                     'p_break': p_break,
#                                                     'p_leak_g_pgv': p_leak_g_pgv,
#                                                     'p_leak_g_pga': p_leak_g_pga,
#                                                     'p_break_g_pgv': p_break_g_pgv,
#                                                     'p_break_g_pga': p_break_g_pga}})
#            
#        # get probability of repair rates
#        rr_leak_avg = A._DM_dict['rr_leak_avg']
#        rr_break_avg = A._DM_dict['rr_break_avg']
#        
#        p_leak_avg,_ = count_prob_exceed(rr_arr,rr_leak_avg)
#        p_break_avg,_ = count_prob_exceed(rr_arr,rr_break_avg)
#        
#        # store information
#        self._FR_dict.update({'p_leak_avg': p_leak_avg,
#                              'p_break_avg': p_break_avg})
#                             
    