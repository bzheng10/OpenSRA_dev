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
import time, importlib, fcn_gen, file_io, sys, h5py, os
from im import fcn_im
from scipy import sparse
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
		self._rup_site_dict = None # dictionary for rups and sites
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
	def create_RV(self, im_tool, gm_dir, im_list=['pga,pgv'], rup_meta_file=None, site_loc_file=None,
					vs30_file=None, l_seg_file=None, flag_clear_dict=False, flag_sample_exist=False, **kwargs):
		"""
		Read and store IM and rupture data from files.
		
		Parameters
		----------
		im_tool : str
			tool used to generate intensity measures: currently **EventCalc** or **OpenSHAInterface**
		site_loc_file : str
			path of the file containing the site locations
		gm_dir : str
			directory containing the GM files
		im_list : str, list, optional
			list of **IM** variables to create: **pga** or **pgv** (default = both)
		rup_meta_file : str, optional
			full path for the file containing rupture metadata (M, r) (default = None)
		vs30_file : str, optional
			path of the file containing the vs30 at the site locations (default = None)
		flag_clear_dict : boolean, optional
			**True** to clear the calculated demands or **False** to keep them; default = **False**
		flag_sample_exist : boolean, optional
			**True** if samples are available or **False** to import statistical parameters and create random variables; default = **False**
	
		Additional parameters for **EventCalc** tool
		rate_min : float
			cutoff for mean annual rate; default to 1/1000
		flag_export : boolean, optional
			flag whether to simplify the matrix for sigma total and inter and export the reduced-size files
		
		Additional parameters for **OpenSHAInterface** tool
		rup_group : str
			name of folder with **IM** to import (e.g., 0_99, 100_199)
		num_rups : int, optional
			first number of ruptures in the group to uploaded, default = None (run all ruptures in group)
		
		Returns
		-------
		_RV_dict : dict
			stored IM moments, eq. rupture information, and site data, to be used for generating IMs
	
		"""
		
		## initialize dictionary if it is none
		if self._IM_dict is None or flag_clear_dict is True:
			self._RV_dict = {}
		
		if self._rup_site_dict is None:
			self._rup_site_dict = {}
			
			## read site coordinates
			if site_loc_file is not None:
				## load and store site locations
				site_loc = file_io.read_sitedata(site_loc_file)
				self._rup_site_dict.update({'site_lon': site_loc.get('lon', None),
									'site_lat': site_loc.get('lat', None)})
			else:
				print('path for site locations is not provided')
				
			## read segment length
			if l_seg_file is not None:
				l_seg = np.loadtxt(l_seg_file)
				self._rup_site_dict.update({'l_seg': l_seg})
			else:
				print('path for segment lengths is not provided')
			
			## open rup_meta_file and load in rupture information
			if rup_meta_file is not None:
				with h5py.File(rup_meta_file, 'r') as f:
					self._rup_site_dict.update({'src':f.get('src')[:]})
					self._rup_site_dict.update({'rup':f.get('rup')[:]})
					self._rup_site_dict.update({'M':f.get('M')[:]})
					self._rup_site_dict.update({'rate':f.get('rate')[:]})
				f.close()
				# src_rup = [str(i)+'_'+str(j) for i in src for j in rup]
			else:
				src = None
				rup = None
				M = None
				rate = None
				
		src = self._rup_site_dict['src']
		rup = self._rup_site_dict['rup']
		M = self._rup_site_dict['M']
		rate = self._rup_site_dict['rate']
		
		## read IM statistical moments and rupture scenario info
		if 'event' in im_tool.lower() or 'calc' in im_tool.lower():
			rate_min = kwargs.get('rate_min',1/1000)
			flag_export = kwargs.get('flag_export',False)
			im_in, rup_meta = file_io.read_EventCalc(gm_dir, flag_export)
			
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
			rup_group = kwargs.get('rup_group',0)
			# num_rups = kwargs.get('num_rups',None)
			# im_in = file_io.read_sha_sparse(gm_dir, group_num, num_rups, im_list)
			self._RV_dict.update({'rup':file_io.read_sha_sparse(gm_dir, rup_group, 
																im_list, src, rup, M, rate, 
																flag_sample_exist).copy()})
		
		## store IM tool used
		self._RV_dict.update({'im_tool':im_tool})
		
		## clear variables
		src = None
		rup = None
		r = None
		M = None
		site_loc = None
		
		
	#############################################################################################################
	def get_IM(self, nsamp_im, im_list=['pga,pgv'], flag_corr_d=True, flag_corr_T=True, T_pga=0.01, T_pgv=1.0, 
					method_d='jayaram_baker_2009', method_T='baker_jayaram_2008', flag_sample_with_sigma_total=False,
                    sigma_aleatory=None,flag_clear_dict=False, flag_sample_exist=False, path_sample=None, **kwargs):
		"""
		Perform multivariate random sampling of **PGA** and **PGV** using means and sigmas for all scenarios. Spatial and spectral orrelations can be applied.
	
		Parameters
		----------
		nsamp_im : float
			number of samples/realizations for intensity measures
		im_list : str, list, optional
			list of **IM** variables to create: **pga** or **pgv** (default = both)
		flag_corr_d : boolean, optional
			decision on performing correlation between sites (distance); default = True
		flag_corr_T : boolean, optional
			decision on performing correlation between periods; default = True (only if **im_list** contains **pga** and **pgv**)
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
		flag_sample_exist : boolean, optional
			**True** if samples are available or **False** to import statistical parameters and create random variables; default = **False**
		path_sample : str, optional
			path to folder samples; default = None
		
		
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
		im_tool = self._RV_dict.get('im_tool',None)
	
		## dimensions
		if 'event' in im_tool.lower() or 'calc' in im_tool.lower():
			dim_d = len(self._RV_dict['rup'][0]['pga_mean'])
			dim_rup = len(self._RV_dict['rup'])
		dim_T = len(im_list) # number of periods of interest
	
		## make period array for spectral correlations
		T = [T_pga,T_pgv]
		
		## check input for method to use
		method_T = kwargs.get('method_T','baker_jayaram_2008')
		proc = getattr(importlib.import_module('im.corr_spectral'),method_T)

		## compute intra-event correlations between pga and pgv
		if method_T == 'baker_jayaram_2008':
			corr_T = proc(T1=T_pga, T2=T_pgv)

		## make list of variables
		param_names = ['mean', 'inter', 'intra']
		var_list = [i+'_'+j for i in im_list for j in param_names]
		
		##
		dim_d = len(self._rup_site_dict['site_lon'])
		dim_rup = len(self._RV_dict['rup']['src'])
	
		if flag_corr_d is False and flag_corr_T is False:
			## no correlations, random/LHS sampling
			
			self._IM_dict.update({'pgv':{}})
			self._IM_dict.update({'pga':{}})
			
			## check if samples already exist
			if flag_sample_exist is True:	# load samples
			
				## loop through ims
				for im in im_list:
					## loop through and import all samples
					for i in range(nsamp_im):
						file_name = im+'_samp_'+str(i)+'.npz'
						self._IM_dict[im].update({i:sparse.coo_matrix(sparse.load_npz(os.path.join(path_sample,file_name))).expm1()})
			
			else:	# perform multivariate random sampling

				## loop through number of samples
				for i in range(nsamp_im):
				
					## first sample pgv
					eps_pgv = np.random.normal(size=(dim_rup,dim_d))
					samp = self._RV_dict['rup']['pgv'+'_mean']
					
					## use total sigma or separate into intra- and inter- event sigmas
					if flag_sample_with_sigma_total:
						
						## correct for sigma
						if sigma_aleatory is None: ## if total sigma is not provided
							sigma_aleatory = self._RV_dict['rup']['pgv'+'_intra'].power(2) + self._RV_dict['rup']['pgv'+'_inter'].power(2)
							sigma_aleatory = sigma_total.power(0.5)
							samp = samp.multiply(sigma_aleatory.multiply(eps_pgv).expm1() + np.ones((dim_rup,dim_d)))
						
						else:
							# only supports singular inputs for sigma_aleatory, expand to matrix later
							samp = samp.multiply(np.exp(sigma_aleatory*eps_pgv))
					
					else:
						## get residuals for intra (epsilon) and inter (eta) (norm dist with mean = 0 and sigma = 1)
						eta_pgv = np.random.normal(size=dim_rup)
						eta_pgv = np.repeat(eta_pgv[:,np.newaxis],dim_d,axis=1) ## eta is constant with site, varies only between rupture
						
						## correct for predicted mean and sigma
						samp = samp.multiply(self._RV_dict['rup']['pgv'+'_intra'].multiply(eps_pgv).expm1() + np.ones((dim_rup,dim_d)))
						samp = samp.multiply(self._RV_dict['rup']['pgv'+'_inter'].multiply(eta_pgv).expm1() + np.ones((dim_rup,dim_d)))
				
					## store samples
					self._IM_dict['pgv'].update({i:samp})
					
					## see if 'pga' is needed
					if 'pga' in im_list:
					
						## conditional sigma for pga
						sigma_cond_pga = np.sqrt(1-corr_T**2)
						
						## conditional mean of eps
						cond_mean_pga_eps = corr_T*eps_pgv
						eps_pga = np.random.normal(size=(dim_rup,dim_d),loc=cond_mean_pga_eps,scale=sigma_cond_pga)
						samp = self._RV_dict['rup']['pga'+'_mean']
										
						## use total sigma or separate into intra- and inter- event sigmas
						if flag_sample_with_sigma_total:
							
							## correct for sigma
							if sigma_aleatory is None: ## if total sigma is not provided
								sigma_aleatory = self._RV_dict['rup']['pga'+'_intra'].power(2) + self._RV_dict['rup']['pga'+'_inter'].power(2)
								sigma_aleatory = sigma_total.power(0.5)
								samp = samp.multiply(sigma_aleatory.multiply(eps_pga).expm1() + np.ones((dim_rup,dim_d)))
							
							else:
								# only supports singular inputs for sigma_aleatory, expand to matrix later
								samp = samp.multiply(np.exp(sigma_aleatory*eps_pga))
								
						else:
							## conditional sampling of eta
							cond_mean_pga_eta = corr_T*eta_pgv
							eta_pga = np.random.normal(size=cond_mean_pga_eta.shape,loc=cond_mean_pga_eta,scale=sigma_cond_pga)
							
							## correct for predicted mean and sigma
							samp = samp.multiply(self._RV_dict['rup']['pga'+'_intra'].multiply(eps_pga).expm1() + np.ones((dim_rup,dim_d)))
							samp = samp.multiply(self._RV_dict['rup']['pga'+'_inter'].multiply(eta_pga).expm1() + np.ones((dim_rup,dim_d)))
				
						## store samples
						self._IM_dict['pga'].update({i:samp})
		
		else:
			## get correlations between sites
			if flag_corr_d is True:
				
				## check input for method to use
				method_d = kwargs.get('method_d','jayaram_baker_2009')
				proc = getattr(importlib.import_module('im.corr_spatial'),method_d)
		
				ind1,ind2 = np.triu_indices(dim_d)

				d = fcn_gen.get_haversine_dist(site_lon[ind1],site_lat[ind1],site_lon[ind2],site_lat[ind2])
				
				## compute intra-event correlations between sites for pga and pgv
				if method_d == 'jayaram_baker_2009':
					geo_cond = kwargs.get('geo_cond',2)
					corr_d_intra_pga = proc(d=d, T=T_pga, geo_cond=geo_cond)
					corr_d_intra_pgv = proc(d=d, T=T_pgv, geo_cond=geo_cond)
				
				## inter-event sigma: perfect correlation (=1)
				corr_d_inter = np.ones(int(dim_d*(dim_d+1)/2)) ## same correlations for pga and pgv
				
			else:
				## identity matrix
				corr_d_intra_pga = np.identity(int(dim_d*(dim_d+1)/2))
				corr_d_intra_pga = corr_d_intra_pga[np.triu_indices(dim_d)]
				corr_d_intra_pgv = corr_d_intra_pga
		
				## identity matrix
				corr_d_inter = corr_d_intra_pga ## same correlations for pga and pgv
				
			# Get correlations between periods, same for intra- and inter- events
			if flag_corr_T is True:
		
				## check input for method to use
				method_T = kwargs.get('method_T','baker_jayaram_2008')
				proc = getattr(importlib.import_module('im.corr_spectral'),method_T)
		
				## compute intra-event correlations between pga and pgv
				if method_T == 'baker_jayaram_2008':
					corr_T = np.asarray([proc(T1=T[i], T2=T[j])
										for i in range(dim_T) for j in range(dim_T) if j >= i])
				
			else:
				## set correlations along diagonal to 1 and off-diagonal to 0 (i.e., identity matrix)
				corr_T = np.identity(int(dim_T*(dim_T+1)/2))
				corr_T = corr_T[np.triu_indices(dim_T)]
				
			## form correlation matrix for intra-event sigma
			cov_intra = fcn_im.get_cov(corr_d_intra_pga, corr_d_intra_pgv, corr_T,
										np.ones(dim_d)*pga_intra, np.ones(dim_d)*pgv_intra,
										dim_d,dim_T)
		
			## form correlation matrix for inter-event 
			cov_inter = fcn_im.get_cov(corr_d_inter, corr_d_inter, corr_T,
										np.ones(dim_d)*pga_inter, np.ones(dim_d)*pgv_inter,
										dim_d,dim_T)
		
	# 		## calculate total covariance
			cov_total = np.asarray(cov_intra + cov_inter)

			## store information
			self._IM_dict.update({'cov_intra': cov_intra,
								'cov_inter': cov_inter,
								'cov_total': cov_total})
		
	
	#############################################################################################################
	def get_EDP(self, category, method, return_param, store_name=None, flag_clear_dict=False, 
					flag_pga=False, flag_pgv=False, flag_M=False, flag_rup_depend=False,
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
		
		## add return_param into kwargs
		kwargs['return_param'] = return_param
		if store_name == None:
			store_name = return_param
		
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
		dim_d = len(self._rup_site_dict['site_lon'])
		dim_rup = len(self._RV_dict['rup']['src'])
		kwargs['dim_d'] = dim_d
		kwargs['dim_rup'] = dim_rup
		
		## number of samples
		nsamp_im = kwargs.get('nsamp_im',1)
		
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
								param_add.update({source_param[i]:source_param_i[j]['output'].copy()})
		
		## get M from scenarios
		if flag_M is True:
			kwargs['M'] = self._RV_dict['rup'].get('M',None)
		
		## if pga nor pgv is required, only run procedure once
		if flag_pga is False and flag_pgv is False and flag_rup_depend is False:
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
		
		###
		eps_epistemic = kwargs.get('eps_epistemic',[0])
		eps_aleatory = kwargs.get('eps_aleatory',[0])
		wgt_aleatory = kwargs.get('wgt_aleatory',[1])
		
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
		
			if 'pgd' in i:
				self._EDP_dict[i][name].update({'prob_dist': output['prob_dist']})
			self._EDP_dict[i][name].update({'output': output[i]})
	
	#############################################################################################################
	def get_DM(self, category, method, return_param, store_name=None, im_list=['pga,pgv'], flag_clear_dict=False, 
					flag_pga=False, flag_pgv=False, flag_M=False, flag_rup_depend=False,
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
		store_name : str, list, optional
			names to store parameter as; default = **return_param**
		im_list : str, list, optional
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
		
		For method parameters, refer to the method documentation under :func:`dm`.
		
		Returns
		-------
		output : varies
			[varies] output depends on the target damage measure and methods.
		
		"""
		
		## add return_param into kwargs
		kwargs['return_param'] = return_param
		if store_name == None:
			store_name = return_param
		
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
		dim_T = len(im_list) # number of periods of interest
		dim_d = len(self._rup_site_dict['site_lon'])
		dim_rup = len(self._RV_dict['rup']['src'])
		kwargs['dim_d'] = dim_d
		kwargs['dim_rup'] = dim_rup
		
		## number of samples
		nsamp_im = kwargs.get('nsamp_im',1)
	
		## rate
		rate = self._RV_dict['rup']['rate']
		rate = np.repeat(rate[:,np.newaxis],dim_d,axis=1)
		
		## add l_seg to kwargs
		pgd_label = kwargs.get('pgd_label',None)
		if pgd_label is not None:
			if 'surf' in pgd_label and ('ala' in method or 'orourke' in method):
				kwargs['l_seg'] = self._rup_site_dict['l_seg']
		
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
					
					count = len(self._DM_dict[i]) # see how many methods have been used
					if count == 0:
						name = 'method'+str(count+1)
						self._DM_dict[i].update({method: {'method':method,
														'source_param':source_param,
														'source_method':source_method,
														'eps_epistemic':eps_epistemic[epi_i],
														'eps_aleatory':eps_aleatory,
														'wgt_aleatory':wgt_aleatory}})
					
					else:
						name = None
						for key in self._DM_dict[i].keys():
							if self._DM_dict[i][key]['method'] == method and \
								self._DM_dict[i][key]['source_param'] == source_param and \
								self._DM_dict[i][key]['source_method'] == source_method and \
								self._DM_dict[i][key]['eps_epistemic'] == eps_epistemic[epi_i]:
							
								name = key
								break
								
						if name is None:
							name = 'method'+str(count+1)
							self._DM_dict[i].update({name: {'method':method,
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
								## temporarily store pgd median
								pgd_median = param_add[param].copy()
								
								## operate on pgd to get realizations based on distribution
								if prob_dist_pgd['type'] == 'lognormal':
									if eps_epistemic[epi_i] == 999: # lumped uncertainty
										pgd_i_j = pgd_median.muliply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total']))
									else:
										pgd_i_j = pgd_median.muliply(np.exp(
														eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
														eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic']))
								
								elif prob_dist_pgd['type'] == 'uniform':
									samp_pgd = np.random.rand(len(pgd_median.data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
												1/prob_dist_pgd['factor_aleatory']
									samp_pgd = sparse.coo_marix((samp_pgd,(pgd_median.row,pgd_median.col)),shape=pgd_median.shape)
									pgd_i_j = pgd_median.multiply(samp_pgd)*prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i]
								
								break
								
					# add adjusted pgd to inputs for damage
					# kwargs[j] = pgd_i_j

					# run method
					# output = proc(**kwargs)
					
					#
					# if 'all' in output[i].keys():
						# result_curr = sparse.csc_matrix(np.sum(output[i]['all'].multiply(rate).toarray(),axis=0))
					# else:
						# result_curr = sparse.csc_matrix([np.sum(output[i][j].multiply(rate).toarray(),axis=0) for j in range(nsamp_im)])
					
					#
					# try:
						# result_updated = self._DM_dict[i][name]['output'] + result_curr
					# except:
						# result_updated = result_curr
				
					#
					# self._DM_dict[i][name].update({'output':result_updated})
				
				# result_updated = sparse.csc_matrix(np.sum(output[i]['all'].multiply(rate).toarray(),axis=0))
			
				# self._DM_dict[i][name].update({'output':result_updated})
		
		## if either pga or pgv is required, run procedure through all scenarios
		else:
		
			## add additional parameters into kwargs
			if source_param is not None:
				for param in source_param:
					if not 'pgd' in param:
						kwargs[param] = param_add[param].copy()
		
			# magnitude
			# if flag_M is True:
				# kwargs['M'] = self._RV_dict['rup']['M']
			
			## get IM simulations for scenario
			# if flag_pga is True:
				# kwargs['pga'] = self._IM_dict['pga']
			
			
			## loop through all cases of epistemic branches
			for epi_i in range(len(eps_epistemic)):
				
				## if running PGV fragilities
				if 'rr_pgv' in return_param:
					sigma_epistemic = kwargs.get('sigma_epistemic',None)
					pgv_epi = {}
					for samp_i in range(nsamp_im):
						pgv_epi.update({samp_i:self._IM_dict['pgv'][samp_i].multiply(sigma_epistemic**eps_epistemic[epi_i])})
					kwargs['pgv'] = pgv_epi
				
				## store in dictionary
				for i in return_param:
					
					count = len(self._DM_dict[i]) # see how many methods have been used
					if count == 0:
						name = 'method'+str(count+1)
						self._DM_dict[i].update({name: {'method':method,
														'source_param':source_param,
														'source_method':source_method,
														'eps_epistemic':eps_epistemic[epi_i],
														'eps_aleatory':eps_aleatory,
														'wgt_aleatory':wgt_aleatory}})
					else:
						name = None
						for key in self._DM_dict[i].keys():
							if self._DM_dict[i][key]['method'] == method and \
								self._DM_dict[i][key]['source_param'] == source_param and \
								self._DM_dict[i][key]['source_method'] == source_method and \
								self._DM_dict[i][key]['eps_epistemic'] == eps_epistemic[epi_i]:
							
								name = key
								break
								
						if name is None:
							name = 'method'+str(count+1)
							self._DM_dict[i].update({name: {'method':method,
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
								## temporarily store pgd median
								pgd_median = param_add[param].copy()
								
								## pgd_median's type is dict if it has more than 1 IM sample
								if type(pgd_median) == dict:
									pgd_i_j = {}
									for k in range(nsamp_im):
										## operate on pgd to get realizations based on distribution
										if prob_dist_pgd['type'] == 'lognormal':
											if eps_epistemic[epi_i] == 999: # lumped uncertainty
												pgd_k = pgd_median[k].multiply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])).tocoo()
												if 'bray_macedo_2019' in source_method:
													d0 = kwargs.get('d0',0.5)
													pgd_k.data[pgd_k.data<=d0] = 0
												pgd_i_j.update({k:pgd_k})
												# if k == 0:
													# print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])

											else:
												pgd_k = pgd_median[k].multiply(np.exp(
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
											# samp_pgd = np.random.rand(len(pgd_median[k].data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
														# 1/prob_dist_pgd['factor_aleatory'] # interpolate in linear scale
											samp_pgd = prob_dist_pgd['factor_aleatory']**(2*np.random.rand(len(pgd_median[k].data)) - 1) # interpolate in log scale
											samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_median[k].row,pgd_median[k].col)),shape=pgd_median[k].shape)
											pgd_i_j.update({k:pgd_median[k].multiply(samp_pgd).tocoo()*(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])})
											# if k == 0:											
												# print(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory'],1/prob_dist_pgd['factor_aleatory'])
												# print(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
							
								else:
									## operate on pgd to get realizations based on distribution
									if prob_dist_pgd['type'] == 'lognormal':
										if eps_epistemic[epi_i] == 999: # lumped uncertainty
											pgd_i_j = pgd_median.multiply(np.exp(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total']))
											# print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_total'])
										else:
											pgd_i_j = pgd_median.multiply(np.exp(
														eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
														eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic']))
											# print(eps_aleatory[ale_j]*prob_dist_pgd['sigma_aleatory'] + \
														# eps_epistemic[epi_i]*prob_dist_pgd['sigma_epistemic'])
								
									elif prob_dist_pgd['type'] == 'uniform':
										# samp_pgd = np.random.rand(len(pgd_median.data))*(prob_dist_pgd['factor_aleatory']-1/prob_dist_pgd['factor_aleatory']) + \
													# 1/prob_dist_pgd['factor_aleatory'] # interpolate in linear scale
										samp_pgd = prob_dist_pgd['factor_aleatory']**(2*np.random.rand(len(pgd_median.data)) - 1) # interpolate in log scale
										samp_pgd = sparse.coo_matrix((samp_pgd,(pgd_median.row,pgd_median.col)),shape=pgd_median.shape)
										pgd_i_j = pgd_median.multiply(samp_pgd)*(prob_dist_pgd['factor_epistemic']**eps_epistemic[epi_i])
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
						result_curr = sparse.csc_matrix([np.sum(output[i][j].multiply(rate).toarray(),axis=0) for j in range(nsamp_im)])
					
					##
					try:
						result_updated = self._DM_dict[i][name]['output'] + result_curr*wgt_aleatory[ale_j]
					except:
						result_updated = result_curr*wgt_aleatory[ale_j]
						
					# print(wgt_aleatory[ale_j])
			
					##
					self._DM_dict[i][name].update({'output':result_updated})


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