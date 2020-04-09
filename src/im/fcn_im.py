#####################################################################################################################
##### Miscellaneous functions for intensity measures (e.g., source and ground motion characterization)
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### get spatial correlations (between sites)
##### Jayaram & Baker (2009) Correlation model for spatially distributed ground-motion intensities
#####################################################################################################################
def get_corr_spatial(method='jayaram_baker_2009', **kwargs):
	"""
	Compute correlations for spectral ordinates; available methods are:
	1) Jayaram & Baker (2009)
		- method = 'jayaram_baker_2009'
		- inputs:
			-- h = distance (km)
			-- T = period (sec)
			-- geo_cond = geologic condition: case 1 = variability within soil (default)
											  case 2 = homogenous/similar condition
	2) TBD

	"""
	if method == 'jayaram_baker_2009'
		## get inputs
		h = kwargs.get('h',None)
		T = kwargs.get('T',None)
		geo_cond= kwargs.get('geo_cond',2)
		
		## see if basic inputs are provided
		if h is None or T is None:
			print('Either h or T (or both) is not provided, defaulting to corr = 1')
			return 1
		else:
			if T < 1:
				if geo_cond == 1:
					b = 8.5 + 17.2*T
				if geo_cond == 2:
					b = 40.7 - 15.0*T
			elif T >= 1:
				b = 22.0 + 3.7*T

			##
			return np.exp(-3*h/b)
			
	else:
		print('Method is not available, defaulting to corr = 1')
		return 1
	
#####################################################################################################################
##### get spectral correlations (between periods)
##### Baker & Jayaram (2008) Correlation of Spectral Acceleration Values from NGA Ground Motion Models
#####################################################################################################################
def get_corr_spectral(method='jayaram_baker_2009',**kwargs):
	"""
	Compute correlations for spectral ordinates; available methods are:
	1) Baker & Jayaram (2008)
		- method = 'baker_jayaram_2008'
		- inputs = T1 (seconds), T2 (seconds)
	2) TBD

	"""
	if method == 'baker_jayaram_2008'
		## get inputs
		T1 = kwargs.get('T1',None)
		T2 = kwargs.get('T2',None)
		
		## see if basic inputs are provided
		if T1 is None or T2 is None:
			print('One or both of the periods are not provided, defaulting to corr = 1')
			return 1
		else:
			## figure out which is bigger
			Tmax = max(T1,T2)
			Tmin = min(T1,T2)
			
			## calculate correlations
			C1 = 1 - np.cos(np.pi/2 - 0.366*np.log(Tmax/max(Tmin,0.109)))
			if Tmax < 0.2:
				C2 = 1 - 0.105*(1 - 1/(1+np.exp(100*Tmax-5)))*((Tmax-Tmin)/(Tmax-0.0099))
			else:
				C2 = 0
			if Tmax < 0.109:
				C3 = C2
			else:
				C3 = C1
			C4 = C1 + 0.5*(np.sqrt(C3) - C3)*(1 + np.cos(np.pi*Tmin/0.109))
			
			## return the right correlation based on period amplitudes
			if Tmax < 0.109:
				return C2
			elif Tmin > 0.109:
				return C1
			elif Tmax < 0.2:
				return min(C2,C4)
			else:
				return C4
	
	else:
		print('Method is not available, defaulting to corr = 1')
		return 1
		
	
#####################################################################################################################
##### get correlation and covariance matrices
#####################################################################################################################
def get_cov(corr_h_pga, corr_h_pgv, corr_p, sig_pga, sig_pgv, dim_h, dim_p=2):
    """
	Compute correlation and covariance matrices for PGA and PGV and across different sites
	
	Parameters:
	----------
	Correlations:
		corr_h_pga: list of spatial correlations across sites for PGA
		corr_h_pgv: list of spatial correlations across sites for PGC
		corr_p: spectral correlation between PGA (T ~ 0.01sec) and PGV (T ~ 1sec)
	Covariances:
		sig_pga: sigma for PGA
		sig_pga: sigma for PGV
	Others:
		dim_h: number of sites
		dim_p: number of periods (fixed to 2 for now)
		
		Name of the site data input file with relative path. The file is
		expected to be a JSON with data stored in a standard format described
		in detail in the Input section of the documentation.

	"""
	## cross of spatial correlations between PGA and PGV
	corr_h_cross = np.sqrt(np.multiply(corr_h_pga,corr_h_pgv))
	
    # convert arrays to symmetric matrices
    corr_p_mat = convert_array_to_sym_mat(corr_p,dim_p) # make period correlation matrix symmetric
    corr_h_pga_mat = convert_array_to_sym_mat(corr_h_pga,dim_h) # make distance correlation matrix symmetric
    corr_h_pgv_mat = convert_array_to_sym_mat(corr_h_pgv,dim_h) # make distance correlation matrix symmetric
    corr_h_gm_mat = convert_array_to_sym_mat(corr_h_cross,dim_h) # make distance correlation matrix symmetric
    
    ## full correlation matrix
    corr_quad11 = corr_h_pga_mat*corr_p_mat[0][0]
    corr_quad12 = corr_h_cross_mat*corr_p_mat[0][1]
    corr_quad21 = np.transpose(corr_quad12)
    corr_quad22 = corr_h_pgv_mat*corr_p_mat[1][1]
    corr_mat = np.bmat([[corr_quad11, corr_quad12], [corr_quad21, corr_quad22]])
    
    ## joint uncorrelated covariance matrix
    cov_quad11 = np.outer(sig_pga,sig_pga)
    cov_quad12 = np.outer(sig_pga,sig_pgv)
    cov_quad21 = np.transpose(cov_quad12)
    cov_quad22 = np.outer(sig_pgv,sig_pgv)
    cov_mat_uncorr = np.bmat([[cov_quad11, cov_quad12], [cov_quad21, cov_quad22]])
    
    ## joint correlated covariance matrix
    cov_mat_corr = np.multiply(cov_mat_uncorr,corr_mat)
    
	## visualize correlation matrix
    # fig = plt.figure(figsize=(6,4))
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(corr_mat,cmap='coolwarm', vmin=-1, vmax=1)
    # fig.colorbar(cax)
    # ticks = np.arange(0,dim_h*2,1)
    # ax.set_xticks(ticks)
    # plt.xticks(rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_xlim([ticks[0]-0.5,ticks[len(ticks)-1]+0.5])
    # ax.set_ylim([ticks[len(ticks)-1]+0.5,ticks[0]-0.5])
    # labels_pga = ['pga-'+str(i+1) for i in range(dim_h)]
    # labels_pgv = ['pgv-'+str(i+1) for i in range(dim_h)]
    # ax.set_xticklabels(labels_pga+labels_pgv)
    # ax.set_yticklabels(labels_pga+labels_pgv)
    # plt.show()
    
	##
    return corr_mat, cov_mat_uncorr, cov_mat_corr
	
	
#####################################################################################################################
##### perform multivariate sampling on IM random variables
#####################################################################################################################
	def generate_IM_samples(mean, cov, nsamp, nsite=1, var_list=['pga','pgv']):
    """
	Perform multivariate sampling for normally distributed random variables
	
	Parameters:
	----------
	mean: mean of the random variable or a list for multiple variables
	cov: variance of the random variable or the covariance matrix for multiple variables
	nsamp: number of samples
	nsite: number of sites

	"""
    ## total number of cases = number of periods * number of sites
    ntot = len(mean)

    ## number of periods = total / number of sites
    nperiod = int(ntot/nsite)
    
    ## generate sample from mean and cov and transpose
    mrs_out = np.random.multivariate_normal(mean, cov, nsamp).T
    
    # partition into individual IMs
	sample_dict = {}
	for i in range(nperiod):
		sample_dict[var_list[i]] = np.exp(mrs_out[i*nsite:(i+1)*nsite])
    
	##
    return sample_dict