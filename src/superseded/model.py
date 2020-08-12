#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Risk model class object
##### 
##### Created: April 27, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Required packages
import numpy as np
import time, importlib, fcn_gen, file_io, sys
from im import fcn_im
#####################################################################################################################


#####################################################################################################################
##### Assessment class object that follows the PBEE framework (IM -> EDP -> DM -> DV)
#####################################################################################################################
class assessment(object):
	"""
	Assess the damages and decision values at target sites using ground motion predictions and demand parameters.
	
	.. autosummary::
	
		create_RV
		get_IM
		get_EDP
		get_DM
		get_DV
	
	"""
	
	#############################################################################################################
	## base directories
	def __init__(self):
	
		## Inputs and simulations
		self._RV_dict = None # dictionary for creating random variables (pga, pgv)
		self._IM_dict = None # create and randomly sample intensity measures
		
		## PBEE workflow
		self._EDP_dict = None # engineering demand parameters
		self._DM_dict = None # damage measures
		self._DV_dict = None # decision values/variables
	
		## results
		self._SUMMARY = None # summary of results
		
		# sys.path.append(os.path.dirname(os.getcwd())+'/data')
	
	#############################################################################################################
	## create intensity measure random variables and store statistical moments
	def create_RV(self, im_tool, site_loc_file, im_dir, **kwargs):
		"""
		Read and store IM and rupture data from files.
		
		Parameters
		----------
		im_tool : str
			tool used to generate intensity measures: currently **EventCalc** or **OpenSHAInterface**
		site_loc_file : str
			path of the file containing the site locations
		im_dir : str
			directory containing the IM files
	
		Additional parameters for **EventCalc** tool
		rate_min : float
			cutoff for mean annual rate; default to 1/1000
		flag_export : boolean, optional
			flag whether to simplify the matrix for sigma total and inter and export the reduced-size files
			
		Additional parameters for **OpenSHAInterface** tool
		rup_meta_file : boolean
			path of the file containing the vs30 at the site locations
		rup_meta_file : str
			full path for the file containing rupture metadata
		rup_num_start : int, optional
			starting rupture number (1 through nRups); default = 1
		rup_num_end : int, optional
			ending rupture nubmer (1 through nRups); default = 1e10 (e.g., all ruptures)
		
		Returns
		-------
		_RV_dict : dict
			stored IM moments, eq. rupture information, and site data, to be used for generating IMs
	
		"""
		
		## initialize dictionary if it is none
		if self._RV_dict is None:
			self._RV_dict = {}
		
		start_time = time.time()
		
		## read site coordinates
		print(site_loc_file)
		if site_loc_file is not None:
			site_loc = file_io.read_sitedata(site_loc_file)
		else:
			print('path for site locations is not provided')
			
		print("read site locations--- %10.6f seconds ---" % (time.time() - start_time))
		start_time = time.time()
		
		## list of IMs, typically PGA and PGV
		im_list=['pga','pgv']
		
		## read IM statistical moments and rupture scenario info
		if 'event' in im_tool.lower() or 'calc' in im_tool.lower():
			rate_min = kwargs.get('rate_min',1/1000)
			flag_export = kwargs.get('flag_export',False)
			im_in, rup_meta = file_io.read_EventCalc(im_dir, flag_export)
		
			print("read IM inputs--- %10.6f seconds ---" % (time.time() - start_time))
			start_time = time.time()
			
			## store sigmas
			for im in im_list:
				sig_intra = im_in[im+'_sig_intra']
				sig_inter = im_in[im+'_sig_inter']
				self._RV_dict.update({im+'_sigma': {'sig_intra': sig_intra,
													'sig_inter': sig_inter}})
		
			## loop through rupture scenarios and IMs and pull information to store
			keys = rup_meta.keys()
			ind = 0
			self._RV_dict.update({'rup':{}})
			for key in keys:
				r = rup_meta[key].get('r',None)
				## use rates above rate_min only
				if r >= rate_min:
					M = rup_meta[key].get('M',None)
					self._RV_dict['rup'].update({ind: {'id':key,'M':M,'r':r}})
					for im in im_list:
						self._RV_dict['rup'][ind].update({im+'_mean':im_in[im+'_mean'][ind]})
					ind += 1
						
		##
		elif 'sha' in im_tool.lower():
			rup_meta_file = kwargs.get('rup_meta_file',None)
			sha_meta_file = kwargs.get('sha_meta_file',None)
			# rup_num_start = kwargs.get('rup_num_start',1)
			# rup_num_end = kwargs.get('rup_num_end',1e10)
			group_num = kwargs.get('group_num',0)
			num_rups = kwargs.get('num_rups',1)
			im_in, rup_meta = file_io.read_OpenSHAInterface(im_dir, rup_meta_file, sha_meta_file, group_num, num_rups)
			
			print("read IM inputs--- %10.6f seconds ---" % (time.time() - start_time))
			start_time = time.time()
			
			## loop through rupture scenarios and IMs and pull information to store
			keys = rup_meta.keys()
			ind = 0
			self._RV_dict.update({'rup':{}})
			for key in keys:
				r = rup_meta[key].get('r',None)
				M = rup_meta[key].get('M',None)
				self._RV_dict['rup'].update({ind: {'id':key,'M':M,'r':r}})
				
				try:
					self._RV_dict['rup'][ind].update({'site_nonzero':im_in[key]['site_nonzero']})
					for im in im_list:
						self._RV_dict['rup'][ind].update({im+'_mean':im_in[key][im+'_mean'],
														im+'_sig_intra':im_in[key][im+'_sig_intra'],
                                                        im+'_sig_inter':im_in[key][im+'_sig_inter']})
				except:
					self._RV_dict['rup'][ind].update({'site_nonzero':None})
					for im in im_list:
						self._RV_dict['rup'][ind].update({im+'_mean':None,
														im+'_sig_intra':None,
                                                        im+'_sig_inter':None})
				ind += 1
		
		## store site locations
		self._RV_dict.update({'site_lon': site_loc.get('lon', None),
							'site_lat': site_loc.get('lat', None)})
		
		## clear variables
		im_in = None
		rup_meta = None
		r = None
		M = None
		site_loc = None
		
		print("store info--- %10.6f seconds ---" % (time.time() - start_time))
		
	#############################################################################################################
	def get_IM(self, nsamp, flag_corr_d=True, flag_corr_T=True, T_pga=0.01, T_pgv=1.0, 
					method_d='jayaram_baker_2009', method_T='baker_jayaram_2008', 
                                flag_clear_dict=False, **kwargs):
        """
        Perform multivariate random sampling of **PGA** and **PGV** using means and sigmas for all scenarios. Spatial and spectral orrelations can be applied.
        
        Parameters
        ----------
        nsamp : float
            number of samples/realizations
        flag_corr_d : boolean, optional
            decision on performing correlation between sites (distance); default = True
        flag_corr_T : boolean, optional
            decision on performing correlation between periods; default = True
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
        
        For additional parameters, see the respective methods under :func:`im.corr_spatial.py` and :func:`im.corr_spectral.py`
        
        Returns
        -------
        _IM_dict : dict
            stored samples for all intensity measures and rupture scenarios
        
        """
        
        ## initialize dictionary if it is none
        if self._IM_dict is None or flag_clear_dict is True:
            self._IM_dict = {}
        
        ## pull list of site coordinates
        site_lon = self._RV_dict.get('site_lon',None)
        site_lat = self._RV_dict.get('site_lat',None)
        
        ## dimensions
        dim_d = len(site_lat) # number of sites
        # dim_d = len(self._RV_dict['rup'][0]['pga_mean'])
        dim_T = 2 # number of periods of interest
        dim_rup = len(self._RV_dict['rup'])
        
        ## make period array for spectral correlations
        T = [T_pga,T_pgv]
        
        ##
        # for rup_i in range(dim_rup):
        for key in self._RV_dict['rup'].keys():
        
            ## pull sigmas
            id = self._RV_dict['rup'][key].get('id',None)
            M = self._RV_dict['rup'][key].get('M',None)
            rate = self._RV_dict['rup'][key].get('r',None)
            cols_NZ = self._RV_dict['rup'][key].get('site_nonzero',None)
            
            ## create arrays for vars
            var_dict = {}
            
            save_Var = ['pga_mean','pga_sig_inter','pga_sig_intra','pgv_mean','pgv_sig_inter','pgv_sig_intra']
            
            pga_mean = np.zeros(dim_d)
            pga_mean[
            pga_mean = self._RV_dict['rup'][key].get('pga_mean',None)
            pga_sig_intra = self._RV_dict['rup'][key].get('pga_sig_intra',None)
            pga_sig_inter = self._RV_dict['rup'][key].get('pga_sig_inter',None)
            pgv_mean = self._RV_dict['rup'][key].get('pgv_mean',None)
            pgv_sig_intra = self._RV_dict['rup'][key].get('pgv_sig_intra',None)
            pgv_sig_inter = self._RV_dict['rup'][key].get('pgv_sig_inter',None)
        
            ## get correlations between sites
            if flag_corr_d is True:
                
                ## check input for method to use
                method_d = kwargs.get('method_d','jayaram_baker_2009')
                proc = getattr(importlib.import_module('im.corr_spatial'),method_d)
        
                start_time = time.time()
                ## calculate distance between sites using Haversine function
                ind1,ind2 = np.triu_indices(dim_d)
                d = fcn_gen.get_haversine_dist(site_lon[ind1],site_lat[ind1],site_lon[ind2],site_lat[ind2])
                
                print("calculate distances for site combinations --- %10.6f seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
                ## compute intra-event correlations between sites for pga and pgv
                if method_d == 'jayaram_baker_2009':
                    geo_cond = kwargs.get('geo_cond',2)
                    corr_d_intra_pga = proc(d=d, T=T_pga, geo_cond=geo_cond)
                    corr_d_intra_pgv = proc(d=d, T=T_pgv, geo_cond=geo_cond)
                
                ## inter-event sigma: perfect correlation (=1)
                corr_d_inter = np.ones(int(dim_d*(dim_d+1)/2)) ## same correlations for pga and pgv
        
                print("create spatial correlation matrix --- %10.6f seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
            else:
                ## identity matrix
                corr_d_intra_pga = np.identity(int(dim_d*(dim_d+1)/2))
                corr_d_intra_pga = corr_d_intra_pga[np.triu_indices(dim_d)]
                corr_d_intra_pgv = corr_d_intra_pga
        
                ## identity matrix
                corr_d_inter = corr_d_intra_pga ## same correlations for pga and pgv
                
                print("create spatial correlation matrix --- %10.6f seconds ---" % (time.time() - start_time))
                start_time = time.time()
        
            # Get correlations between periods, same for intra- and inter- events
            if flag_corr_T is True:
        
                ## check input for method to use
                method_T = kwargs.get('method_T','baker_jayaram_2008')
                proc = getattr(importlib.import_module('im.corr_spectral'),method_T)
        
                ## compute intra-event correlations between pga and pgv
                if method_T == 'baker_jayaram_2008':
                    corr_T = np.asarray([proc(T1=T[i], T2=T[j])
                                        for i in range(dim_T) for j in range(dim_T) if j >= i])
                    
                print("create spectral correlation matrix --- %10.6f seconds ---" % (time.time() - start_time))
                start_time = time.time()
        
            else:
                ## set correlations along diagonal to 1 and off-diagonal to 0 (i.e., identity matrix)
                corr_T = np.identity(int(dim_T*(dim_T+1)/2))
                corr_T = corr_T[np.triu_indices(dim_T)]
                
                print("create spectral correlation matrix --- %10.6f seconds ---" % (time.time() - start_time))
                start_time = time.time()
        
            ## form correlation matrix for intra-event sigma
            corr_intra, cov_intra = fcn_im.get_cov(corr_d_intra_pga, corr_d_intra_pgv, corr_T,
                                            np.ones(dim_d)*pga_sigma['sig_intra'], np.ones(dim_d)*pgv_sigma['sig_intra'],
                                            dim_d,dim_T)
        
            print("create correlation and covariance matrices for intra-event sigma --- %10.6f seconds ---" % (time.time() - start_time))
            ## form correlation matrix for inter-event 
            corr_inter, cov_inter = fcn_im.get_cov(corr_d_inter, corr_d_inter, corr_T,
                                            np.ones(dim_d)*pga_sigma['sig_inter'], np.ones(dim_d)*pgv_sigma['sig_inter'],
                                            dim_d,dim_T)
        
            print("create correlation and covariance matrices for inter-event sigma --- %10.6f seconds ---" % (time.time() - start_time))
            
    # 		## calculate total covariance
            cov_total = np.asarray(cov_intra + cov_inter)
    # 		sig_total = np.sqrt(cov_total)

            ## store intermediate information
    # 		self._IM_dict.update({'corr_d_intra_pga': corr_d_intra_pga,
    # 							'corr_d_intra_pgv': corr_d_intra_pgv,
    # 							'corr_d_inter': corr_d_inter,
    # 							'corr_T': corr_T})
            ## store information
            self._IM_dict.update({'corr_intra': corr_intra,
                                'corr_inter': corr_inter,
                                'cov_intra': cov_intra,
                                'cov_inter': cov_inter,
                                'cov_total': cov_total})
            
            start_time = time.time()
    # 		## get means into an array
            self._IM_dict.update({'sim': {}})
            for i in range(dim_rup):
                if np.mod(i,100) == 0:
                    print("after rupture "+str(i)+": --- %10.6f seconds ---" % (time.time() - start_time))
                
                mean = np.hstack([self._RV_dict['rup'][i]['pga_mean'],self._RV_dict['rup'][i]['pgv_mean']])
        
                ## simulate intensity measures
                sample_dict = fcn_im.get_RV_sims(mean, cov_total, nsamp, dim_d, var_list=['pga','pgv'])
                pga = sample_dict['pga']
                pgv = sample_dict['pgv']
        
                self._IM_dict['sim'].update({i: {'pga': pga,'pgv': pgv}})
		
		print("after all ruptures ("+str(dim_rup)+"): --- %10.6f seconds ---" % (time.time() - start_time))
        
	
	#############################################################################################################
	def get_EDP(self, category, method, return_param, flag_clear_dict=False, 
					flag_pga=False, flag_pgv=False, flag_M=False,
					source_dict=None, source_param=None, source_method=None, **kwargs):
		"""
		Using the simulated intensity measures to calculate engineering demand parameters.
	
		Parameters
		----------
		category : str
			demand category to calculate; options are **corr_spt**, **liq**, **ls**, **gs**, etc. (see :func:`edp` for all **EDP** categories)
		method : str
			method/procedure to use to calculate the demand; see :func:`edp` for available methods
		return_param : str, list
			single of a list of parameters to return, see the return variables under each function (:func:`edp`)
		flag_clear_dict : boolean, optional
			**True** to clear the calculated demands or **False** to keep them; default = **False**
		flag_pga : boolean, optional
			**True** include simulated **PGA**; default = **False**
		flag_pgv : boolean, optional
			**True** include simulated **PGV**; default = **False**
		flag_M : boolean, optional
			**True** include moment magnitude **M** from rupture metadata; default = **False**
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
		
		## add return_param into kwargs
		kwargs['return_param'] = return_param
		
		## initialize dictionary if it is none or if user wishes to clear dict
		if self._EDP_dict is None or flag_clear_dict is True:
			self._EDP_dict = {}
		
		## create keys
		for i in return_param:
			if not i in self._EDP_dict.keys():
				self._EDP_dict[i] = {}
			
		## load method
		proc = getattr(importlib.import_module('edp.'+category),method)
		
		## dimensions
		dim_d = len(self._RV_dict['rup'][0]['pga_mean'])
		dim_T = 2 # number of periods of interest
		dim_rup = len(self._RV_dict['rup'])
		
		## number of samples
		nsamp = len(self._IM_dict['sim'][0]['pga'][0])
		kwargs['nsamp'] = nsamp
		
		## set output storage
		output = {}
		
		## if source_param is not empty, then get parameter from method
		param_add = {}
		if source_param is not None:
			for i in range(len(source_param)):
				try:
					source_dict_i = getattr(self,source_dict[i])
				except:
					print(source_dict[i]+' does not exist')
					break
				else:
					source_param_i = source_dict_i.get(source_param[i],None)
					if source_param_i is None:
						print(source_param[i]+' does not exist')
						break
					else:
						for j in source_param_i:
							if source_param_i[j].get('method',None) == source_method[i]:
								for k in source_param_i[j]['output']:
									param_add.update({source_param[i]:source_param_i[j]['output'][k].copy()})
		
		## get M from scenarios
		if flag_M is True:
			M = [self._RV_dict['rup'][i].get('M',None) for i in range(dim_rup)]
		
		## if pga nor pgv is required, only run procedure once
		if flag_pga is False and flag_pgv is False:
			## add additional parameters into kwargs
			if source_param is not None:
				for j in source_param:
					kwargs[j] = param_add[j].copy()
			## run method
			output.update({'all_rup':proc(**kwargs)})
			
		## if either pga or pgv is required, run procedure through all scenarios
		else:
			## loop through all scenarios
			for i in range(dim_rup):
			# for i in range(1):
				## add additional parameters into kwargs
				if source_param is not None:
					for j in source_param:
						kwargs[j] = param_add[j].copy()
				## get IM simulations for scenario
				if flag_pga is True:
					kwargs['pga'] = np.transpose(self._IM_dict['sim'][i]['pga'])
				if flag_pgv is True:
					kwargs['pgv'] = np.transpose(self._IM_dict['sim'][i]['pgv'])
				if flag_M is True:
					kwargs['M'] = M[i]

				## run method
				output.update({i:proc(**kwargs)})
		
		## store in dictionary
		for i in return_param:
			count = len(self._EDP_dict[i]) # see how many methods have been used
			name = 'method'+str(count+1)
			self._EDP_dict[i].update({name: {'method':method,
											'source_param':source_param,
											'source_method':source_method}})
			store_dict = {}
			store_dict.update({key: output[key][i] for key in output.keys()})
			self._EDP_dict[i][name].update({'output': store_dict.copy()})
	
	
	#############################################################################################################
	def get_DM(self, category, method, return_param, flag_clear_dict=False, 
					flag_pga=False, flag_pgv=False, flag_M=False, 
					source_dict=None, source_param=None, source_method=None, **kwargs):
		"""
		Using the simulated intensity measures and engineering demand parameters to calculate damage measures.
	
		Parameters
		----------
		category : str
			demand category to calculate; options are **rr** (more to be added, see :func:`dm` for meaning of the options)
		method : str
			method/procedure to use to calculate the damage; see :func:`dm` for available methods.
		return_param : str, list
			single of a list of parameters to return, see the return variables under each function (:func:`dm`)
		flag_clear_dict : boolean, optional
			**True** to clear the calculated damages or **False** to keep them; default = **False**
		flag_pga : boolean, optional
			**True** include simulated **PGA**; default = **False**
		flag_pgv : boolean, optional
			**True** include simulated **PGV**; default = **False**
		flag_M : boolean, optional
			**True** include moment magnitude **M** from rupture metadata; default = **False**
		source_dict : str, list, optional
			dictionary that contains **source_param** and **source_method**; default = None
		source_param : str, list, optional
			parameter to get from **existing** stored parameters in (e.g., **liq_susc**); default = None
		source_method : str, list, optional
			method used to obtain the source_param (e.g., **zhu_etal_2017**); default = None
		
		For method parameters, refer to the method documentation under :func:`dm`.
		
		Returns
		-------
		output : varies
			[varies] output depends on the target damage measure and methods.
		
		"""
		
		## add return_param into kwargs
		kwargs['return_param'] = return_param
		
		## initialize dictionary if it is none or if user wishes to clear dict
		if self._DM_dict is None or flag_clear_dict is True:
			self._DM_dict = {}
		
		## create keys
		for i in return_param:
			if not i in self._DM_dict.keys():
				self._DM_dict[i] = {}
			
		## load method
		proc = getattr(importlib.import_module('dm.'+category),method)
		
		## dimensions
		dim_d = len(self._RV_dict['rup'][0]['pga_mean'])
		dim_T = 2 # number of periods of interest
		dim_rup = len(self._RV_dict['rup'])
		
		## number of samples
		nsamp = len(self._IM_dict['sim'][0]['pga'][0])
		kwargs['nsamp'] = nsamp
		
		## set output storage
		output = {}
		
		## if source_param is not empty, then get parameter from method
		param_add = {}
		if source_param is not None:
			for i in range(len(source_param)):
				try:
					source_dict_i = getattr(self,source_dict[i])
				except:
					print(source_dict[i]+' does not exist')
					break
				else:
					source_param_i = source_dict_i.get(source_param[i],None)
					if source_param_i is None:
						print(source_param[i]+' does not exist')
						break
					else:
						for j in source_param_i:
							if source_param_i[j].get('method',None) == source_method[i]:
								for k in source_param_i[j]['output']:
									param_add.update({source_param[i]:source_param_i[j]['output'][k].copy()})
		
		## get M from scenarios
		if flag_M is True:
			M = [self._RV_dict['rup'][i].get('M',None) for i in range(dim_rup)]
		
		## if pga nor pgv is required, only run procedure once
		if flag_pga is False and flag_pgv is False:
			## add additional parameters into kwargs
			if source_param is not None:
				for j in source_param:
					kwargs[j] = param_add[j].copy()
			## run method
			output.update({'all_rup':proc(**kwargs)})
			
		## if either pga or pgv is required, run procedure through all scenarios
		else:
			## loop through all scenarios
			for i in range(dim_rup):
			# for i in range(1):
				## add additional parameters into kwargs
				if source_param is not None:
					for j in source_param:
						kwargs[j] = param_add[j].copy()
				## get IM simulations for scenario
				if flag_pga is True:
					kwargs['pga'] = np.transpose(self._IM_dict['sim'][i]['pga'])
				if flag_pgv is True:
					kwargs['pgv'] = np.transpose(self._IM_dict['sim'][i]['pgv'])
				if flag_M is True:
					kwargs['M'] = M[i]

				## run method
				output.update({i:proc(**kwargs)})
		
		## store in dictionary
		for i in return_param:
			count = len(self._DM_dict[i]) # see how many methods have been used
			name = 'method'+str(count+1)
			self._DM_dict[i].update({name: {'method':method,
											'source_param':source_param,
											'source_method':source_method}})
			store_dict = {}
			store_dict.update({key: output[key][i] for key in output.keys()})
			self._DM_dict[i][name].update({'output': store_dict.copy()})
		
		## get segment-weighted damages
		# n_leak_tot, rr_leak_avg = get_weighted_average([self._DM_dict['site_'+str(i+1)]['rr_leak'] for i in range(dim_d)], l_seg)
		# n_break_tot, rr_break_avg = get_weighted_average([self._DM_dict['site_'+str(i+1)]['rr_break'] for i in range(dim_d)], l_seg)
		
		# store information
		# self._DM_dict.update({'rr_leak_avg': rr_leak_avg,
							# 'rr_break_avg': rr_break_avg,
							# 'n_leak_tot': n_leak_tot,
							# 'n_break_tot': n_break_tot})
							
	#############################################################################################################
	def get_DV(self, param, method, flag_clear_dict, **kwargs):
		"""
		Using the damage measures to estimate decision variables/values.
	
		Parameters
		----------
		param : str
			damage parameter to calculate; options are **TBD** (more to be added, see :func:`dm` for meaning of the options)
		method : str
			method/procedure to use to calculate the damage; see :func:`dm` for available methods.
		flag_clear_dict : boolean, optional
			**True** to clear the calculated damages or **False** to keep them.
		
		For method parameters, refer to the method documentation under :func:`dm`.
		
		Returns
		-------
		output : varies
			[varies] output depends on the target damage measure and methods.
		
		"""
		
		## initialize dictionary if it is none or if user wishes to clear dict
		if self._DV_dict is None or flag_clear_dict is True:
			self._DV_dict = dict([(label, dict()) for label in [
				'rr'
			]])
			
		## load method
		proc = getattr(importlib.import_module('dm.'+param),method)
        
		## run method for every site 
		output = proc(kwargs)
		
		## store in dictionary
		count = len(self._DV_dict[param]) # see how many methods have been used
		name = 'method'+str(count+1)
		self._DV_dict[param].update({name: {'method':method,'output':output}})

	
	#############################################################################################################
	def get_probability(self, method_list, **kwargs):
	
		return None
#        dim_d = sum([1 for i in A._DM_dict.keys() if 'site_' in i])
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
#        for i in range(dim_d):
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
	
	
	#############################################################################################################
	## read inputs from EQHazard
	def _read_inputs_IM(self, path, flag_export=True):
		"""
		Read and process the input files from EQHazard to describe the assessment task.
	
		Parameters
		----------
		path: string
			Name of the intensity measure input file with relative path. The file
			is expected to be a JSON with data stored in a standard format described
			in detail in the Input section of the documentation.
		
		verbose: boolean, default: False
			If True, the method echoes the information read from the files.
			This can be useful to ensure that the information in the file is
			properly read by the method.
	
		"""
		## ## initialize dictionary if it is none
		if self._IM_in is None:
			self._IM_in = {}
		
		## read data
		# self._IM_in = file_io.read_EQHazard(path, verbose=verbose)
		self._IM_in = file_io.read_EventCalc(path, flag_export, verbose=verbose)
	
	
	#############################################################################################################
	## read other inputs on site data
	def _read_inputs_site_data(self, path, search_name, store_name=None):
		"""
		Read and process site data (e.g., map-based liquefaction suscep) not realted to IM.
	
		Parameters
		----------
		path: string
			Name of the site data input file with relative path. The file is
			expected to be a JSON with data stored in a standard format described
			in detail in the Input section of the documentation.
		
		verbose: boolean, default: False
			If True, the method echoes the information read from the files.
			This can be useful to ensure that the information in the file is
			properly read by the method.
	
		"""
		## ## initialize dictionary if it is none
		if self._Site_in is None:
			self._Site_in = {}
		
		## if store_name is not provided
		if store_name is None:
			store_name = search_name
			
		## read and store information
		self._Site_in[store_name].update({store_name: file_io.read_json_other(path,search_name)})
		# self._IM_in['site_data'].update({'l_seg': read_other_json(path_json,'LSegment')})
		# self._IM_in['site_data'].update({'liq_susc': read_other_json(path_json,'LiqSusc')})
		# self._IM_in['site_data'].update({'ls_susc': read_other_json(path_json,'LsSusc')})
		# self._IM_in['site_data'].update({'d_w': read_other_json(path_json,'Z2gw')})