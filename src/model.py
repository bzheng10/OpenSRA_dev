#####################################################################################################################
##### Primary models (e.g., PBEE)
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Assessment class object that follows the PBEE framework (IM -> EDP -> DM -> DV)
#####################################################################################################################
class assessment(object):
	"""
	Order of operation for assessment class
	1) create class object:
		- var = asssessment()
	2) read input intensity measures from EQHazard:
		- 
	3) read other site data
		- 
	4) create random variables from EQHazard output:
		-
	5) randomly samply intensity measures
		-
	6) calculate engineering demand parameters from IM realizations
		-
	7) calculate damage measures
		-
	8) calculate decision values
		-
	9) sort and evaluate probabilities and recurrences
		-
	10) produce summary
		-
	
	"""
	#############################################################################################################
	## base directories
    def __init__(self):
		"""
        Clear dictionaries
		
		"""
        ## inputs
        self._Site_in = None # site data from maps (e.g. groundwater depth, mapped liquefaction susceptibility)
        self._IM_in = None # intensity measures from EQHazard (pga, pgv)
        self._EDP_in = None # engineering demand parameters (e.g., mapped deformation)
        # self._POP_in = None # population
        # self._FG_in = None # fragility

        ## random variables and loss model
        self._RV_dict = None # dictionary for creating random variables (pga, pgv)
        self._IM_dict = None # create and randomly sample intensity measures
        self._EDP_dict = None # engineering demand parameters
        self._DM_dict = None # damage measures
        # self._FR_dict = None # fragility estimates
        self._DV_dict = None # decision values/variables

        ## results
        # self._TIME = None
        # self._POP = None
        # self._COL = None
        # self._ID_dict = None
        # self._DMG = None
        # self._DV_dict = None
        self._SUMMARY = None

        # self._assessment_type = 'generic'
    
	
	#############################################################################################################
	## read inputs from EQHazard
    def read_inputs_IM(self, path, verbose=False):
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
        self._IM_in = read_json_EQHazard(path, verbose=verbose)
    
	
	#############################################################################################################
	## read other inputs on site data
	def read_inputs_site_data(self, path, search_name, store_name=None, verbose=False):
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
        self._Site_in[store_name].update({store_name: read_json_other(path,search_name)})
        # self._IM_in['site_data'].update({'l_seg': read_other_json(path_json,'LSegment')})
        # self._IM_in['site_data'].update({'liq_susc': read_other_json(path_json,'LiqSusc')})
        # self._IM_in['site_data'].update({'ls_susc': read_other_json(path_json,'LsSusc')})
        # self._IM_in['site_data'].update({'d_w': read_other_json(path_json,'Z2gw')})


	#############################################################################################################
	## create intensity measure random variables and store statistical moments
    def _create_RV(self, im_list=['pga','pgv']):
        """
        Searches through IM_in from EQHazard and gets the statstical moments.
		
		Parameters
        ----------
        im_list: array of strings (e.g., pga, pgv, sa)
			Define a list of intensity measures to pull and create random variables for

        """
		## initialize dictionary if it is none
		if self._RV_dict is None:
			self._RV_dict = {}

        ## get site coordinates from _IM_in
        site_loc = self._IM_in.get('site_loc', None)
		
		## pull lat and lon from site_loc to be stored in _IM_dict
        site_lat = site_loc.get('latitude', None)
        site_lon = site_loc.get('longitude', None)

		## get statistical moments and periods
        im_data = self._IM_in.get('im_data', None)
		
		## get periods for spectral acceleration
        period = im_data.get('period', None)
        
		## initialize data dictionary
        data = {}
		
		## loop through im_list and pull relevant information
        for var in var_list:
			##
            mean = im_data.get(var+'_mean', None) ## mean, natural log
            sig_tot = np.asarray(im_data.get(var+'_std_total', None)) ## total sigma
            sig_intra = np.asarray(im_data.get(var+'_std_intra', None)) ## sigma for intra event
            sig_inter = np.asarray(im_data.get(var+'_std_inter', None)) ## sigma for inter event
			
			## store the moments of each IM into data dictionary
            data.update({var: {'mean': mean,
                               'sig_total': sig_tot,
                               'sig_intra': sig_intra,
                               'sig_inter': sig_inter}})

		## transfer from data to _IM_dict
        self._RV_dict.update({'site_lat': site_lat,
                              'site_lon': site_lon})
		for var in var_list:
			self_.RV_dict.update({var: data[var]})

		
	#############################################################################################################
    def generate_IM(self, nsamp, flag_corr_d=True, flag_corr_T=True, T_pga=0.01, T_pgv=1.0, 
					method_d='jayaram_baker_2009', method_T='baker_jayaram_2008', **kwargs):
	    """
        Uses the statistcal moments pulled from EQHazard to generate random samples.
		Cross-correlations can be applied

        Parameters
        ----------
        nsamp: number of samples/realizations
		
		flag_corr_d: 'True' or 'False' for performing correlation between sites (distance)
			- default to 'True'

		flag_corr_T: 'True' or 'False' for performing correlation between periods
			- default to 'True'
        
		T_pga: approximate period for PGA to be used in period correlation
			- default to 0.01 seconds
		
		T_pgv: approximate period for PGV to be used in period correlation
			- default to 1.0 second
			
		method_d = method to use for spatial correlation
			- default = jayaram_baker_2009
			
		method_T = method to use for spectral correlation
			- default = baker_jayaram_2008
			
        kwargs: other parameters (method-dependent)

        """
		## initialize dictionary if it is none
		if self._IM_dict is None:
			self._IM_dict = {}
	
		## pull list of site coordinates
        site_lat = self._RV_dict.get('site_lat',None)
        site_lon = self._RV_dict.get('site_lon',None)
	
		## dimensions
        dim_d = len(site_lat) # number of sites
		dim_T = 2 # number of periods of interest
	
		## pull statistics
        # assess_ID = 'id_1'
        pga_moment = self._RV_dict.get('pga',None)
        pgv_moment = self._RV_dict.get('pgv',None)
        sa_moment = self._RV_dict.get('pga',None)
	
		## approximate periods to use for pga and pgv
        # T_pga = 0.01
		# T_pgv = 1.0
	
        ## get correlations between sites
        if flag_corr_d is True:
	
			## check input for method to use
			method_d = kwargs.get('method_d','jayaram_baker_2009')
	
			## calculate distance between sites using Haversine function
            d = [get_haversine_dist(site_lon[i],site_lat[i],
                                    site_lon[j],site_lat[j])
                 for i in range(dim_d) for j in range(dim_d) if j >= i]
	
            ## compute intra-event correlations between sites for pga and pgv
			if method_d == 'jayaram_baker_2009':
				corr_d_intra_pga = np.asarray([get_corr_spatial(method_d, d=i, T=T_pga, geo_cond=2) for i in d])
				corr_d_intra_pgv = np.asarray([get_corr_spatial(method_d, d=i, T=T_pgv, geo_cond=2) for i in d])
	
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
	
            ## compute intra-event correlations between pga and pgv
			if method_T == 'baker_jayaram_2008':
				corr_T = np.asarray([get_corr_spectral(method_T, T1=T_pga, T2=T_pgv)
				                     for i in range(dim_T) for j in range(dim_T) if j >= i])
	
        else:
			## set correlations along diagonal to 1 and off-diagonal to 0 (i.e., identity matrix)
            corr_T = np.identity(int(dim_T*(dim_T+1)/2))
            corr_T = corr_T[np.triu_indices(dim_T)]
	
        ## form correlation matrix for intra-event sigma
        corr_intra, _, cov_intra = get_cov(corr_d_intra_pga, corr_d_intra_pgv, corr_T,
                                           pga_moment['sig_intra'], pgv_moment['sig_intra'],
                                           dim_d,dim_T)
	
		## form correlation matrix for inter-event 
        corr_inter, _, cov_inter = get_cov(corr_d_inter, corr_d_inter, corr_T,
                                           pga_moment['sig_inter'], pgv_moment['sig_inter'],
                                           dim_d,dim_T)
	
        ## calculate total covariance
        cov_total = np.asarray(cov_intra + cov_inter)
        sig_total = np.sqrt(cov_total)
	
        ## get means into an array
        mean = np.hstack([pga_moment['mean'],pgv_moment['mean']])
	
		## simulate intensity measures
        sample_dict = get_IM_sims(mean, cov_total, nsamp, dim_d, var_list=['pga','pgv'])
		pga = sample_dict['pga']
		pgv = sample_dict['pgv']
	
		## meta-data
		meta = {}
		meta['dim_d'] = dim_d
		meta['dim_T'] = dim_T
		meta['T_pga'] = T_pga
		meta['T_pgv'] = T_pgv
		meta['method_d'] = 'jayaram_baker_2009'
		meta['method_T'] = 'baker_jayaram_2008'
		
        ## Store correlations and covariances
        self._IM_dict.update({'meta':meta,
							  'pga': pga,
							  'pgv': pgv,
							  'corr_intra': corr_intra,
                              'corr_inter': corr_inter,
                              'cov_intra': cov_intra,
                              'cov_inter': cov_inter,
                              'cov_total': cov_total})
	
	
	#############################################################################################################
    def generate_EDP(self, demand, method, **kwargs):
		"""
        Using the simulated IMs to calculate EDPs

        Parameters
        ----------
		demand: which demand parameter or procedure to run or solve for
			0) pen_corr: correction of penetration resistance for liquefaction triggering
				- SPT-based
					1) youd_etal_2001(N_type, N, fc, sv0_eff, patm, ERm, BH_diam, L_rod, 
									  flag_liner_room, flag_liner_absent, CN_method)
					2) cetin_etal_2004(N_type, N, fc, sv0_eff, patm, ERm, BH_diam, L_rod,
									    flag_liner_room, flag_liner_absent, CN_method)
					3) boulanger_idriss_2014(input_type, N_type, N, fc, sv0_eff, patm, ERm, BH_diam,
						 				     L_rod, flag_liner_room, flag_liner_absent, CN_method)
				- CPT-Based:
					4) moss_et_al_2006(q_type, qc, fs, patm, sv0_tot, sv0_eff, c)
					5) boulanger_idriss_2014(input_type, q_type, qc, u2, ar, patm, fc, sv0_tot, sv0_eff, fs, Cfc)
					
			1) liq: liquefaction triggering (run with pen_corr)
				1) youd_etal_2001(N1_60_cs, amax, sv0_tot, sv0_eff, rd, M, Dr, patm, tau_stat)
				2) cetin_etal_2004(N1_60, fc, amax, sv0_tot, sv0_eff, rd, M, patm, p_liq, flag_include_error)
				3) boulanger_idriss_2014(input_type, resistance, amax, sv0_tot, sv0_eff, rd, M, patm, tau_stat,
									     Q, K0, flag_det_prob, p_liq)
				4) moss_etal_2006(qc_1, Rf, c, amax, sv0_tot, sv0_eff, rd, M, patm, p_liq)
				5) zhu_etal_2017(pgv, vs30, precip, dc, dr, dw, wtd, M)
				6) hazus_2004(pga, M, dw, susc_liq)
						
			2) ls: lateral spreading
				1) grant_etal_2016(flag_failure, additional inputs depend on failure type)
				2) youd_etal_2002(M, R, W, S, T_15, F_15, D50_15)
				3) hazus_2004(susc_liq, pga, M, z, dr)
			
			3) gs: ground settlement
				1) cetin_etal_2009(N1_60_cs, amax, sv0_tot, sv0_eff, rd, Dr, patm)
				2) ishihara_yoshimine_1992(N1_60_cs, Dr, FS_liq)
				3) hazus_2004(susc_liq)
				4) tokimatsu_seed_1987(TBD)
			
			4) land: landslide/slope stability
				1) grant_etal_2016(flag_failure, susc_liq, pga, M, z, dr)
				2) jibson_2007(pga, M, Ia, ky, alpha, FS)
				3) saygili_2008(pga, pgv, M, Ia, Tm, ky, coh, phi, t, m, gamma, gamm_w, alpha)
				4) rathje_antonakos_2011(pga, pgv, M, Tm, Ts, ky, coh, phi, t, m, gamma, gamm_w, alpha)
				5) bray_travasarou_2007(M, T_arr, Sa_arr, slope, eps, pga, flag_topo, Ts, Sa_Ts_deg)
				6) bray_macedo_2019(gm_type, M, T_arr, sa_arr, slope, eps, pga, pgv, flag_topo, slope_loc, Ts, Sa_Ts_deg, ky, h, vs)
				7) hazus_2004(dir_makdisi_seed, pga, M, susc_land)
			
			5) tgs = transient ground strain
				1) newmark_1967(Vmax, Vs, gamma_s, Cr)
				2) shinozuka_koike_1979(eps_g, D, l, t, E, G, beta_0, gamma_0, gamma_cr)
				3) orourke_elhmadi_1988(eps_g, D, l, t, E, tu)
			
			6) sfr = surface fault rupture
				1) wells_coppersmith_1994(M, d_type, fault_type)
				2) hazus_2004(M)
			
			7) bfr = buried fault rupture
				1) TBD
		
        kwargs: other parameters (method-dependent)
		
        """
		## initialize dictionary if it is none
		if self._EDP_dict is None:
			self._EDP_dict = {}
		
		## get magnitude
		# M = self._IM_in['eq_rup']['Magnitude']
        # l_seg = self._IM_in['site_data']['l_seg']
        # liq_susc = self._IM_in['site_data']['liq_susc']
        # ls_susc = self._IM_in['site_data']['ls_susc']
        # d_w = self._IM_in['site_data']['ls_susc']
		
		## import method
		module = import_module('src.edp'+demand)
		proc = getatt(module,method)
		
		######################################################
		## run method for every site 
		output = proc(kwargs)
		
		## Note: need to figure out how to run it for every site efficiently
		## some methods can run all sites at once, while others require
		## site-by-site runs
		######################################################
		
		## store in dictionary
		self._EDP_dict.update({demand: {method: output}})
	
	#############################################################################################################
    def generate_DM(self, damage, method, dim_d, **kwargs):
		"""
        Using the simulated IMs and mapped/calculated EDPs to calculate DMs

        Parameters
        ----------
		damage: which damage measure to run
			1) rr: repair rate (number of repairs per distance)
				- 1) hazus_2004(pgv, pgd_ls, pgd_gs, pipe_type, l_seg)
				
			2) TBD
			
		dim_d = number of sites
		
        kwargs: other parameters (method-dependent)
		
        """	
		## initialize dictionary if it is none
		if self._DM_dict is None:
			self._DM_dict = {}
			
		## import method
		module = import_module('src.dm'+damage)
		proc = getatt(module,method)
		
		## run method for every site
		# for i in range(dim_d):
		output = proc(kwargs)
	
		## store in dictionary
		self._DM_dict.update({damage: {method: output}})
		
		## get segment-weighted damages
        # n_leak_tot, rr_leak_avg = get_weighted_average([self._DM_dict['site_'+str(i+1)]['rr_leak'] for i in range(dim_d)], l_seg)
        # n_break_tot, rr_break_avg = get_weighted_average([self._DM_dict['site_'+str(i+1)]['rr_break'] for i in range(dim_d)], l_seg)
        
        # store information
        # self._DM_dict.update({'rr_leak_avg': rr_leak_avg,
                              # 'rr_break_avg': rr_break_avg,
                              # 'n_leak_tot': n_leak_tot,
                              # 'n_break_tot': n_break_tot})
							  
    #############################################################################################################
    def generate_DV(self, method_list, **kwargs):
		"""
        Using the calculated DMs to assess DVs

        Parameters
        ----------
		method_list: list of methods to run
			1) TBD

        kwargs: other parameters (method-dependent)
		
        """	
		## initialize dictionary if it is none
		if self._DV_dict is None:
			self._DV_dict = {}
			
		print('Placeholder - still under development')
    
	#############################################################################################################
    def get_probability(self, method_list, **kwargs):
	
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
