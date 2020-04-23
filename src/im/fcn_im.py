#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Miscellaneous functions for intensity measures
##### 
##### Created: April 13, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### get correlation and covariance matrices
#####################################################################################################################
def get_cov(corr_h_pga, corr_h_pgv, corr_p, sig_pga, sig_pgv, dim_h, dim_p=2):
	"""
	Compute correlation and covariance matrices for PGA and PGV and across different sites
	
	Parameters
	
	Correlations
	
		corr_h_pga: list of spatial correlations across sites for PGA
		corr_h_pgv: list of spatial correlations across sites for PGC
		corr_p: spectral correlation between PGA (T ~ 0.01sec) and PGV (T ~ 1sec)
		
	Covariances
	
		sig_pga: sigma for PGA
		sig_pga: sigma for PGV
		
	Others
	
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
	Perform multivariate sampling for normally distributed random variables and partition the results according to input variables. Currently works with two variables max, but as many spatial 
	
	Parameters
	----------
	mean : float, array
		[varies] mean of a random variable or a list for multiple variables
	cov : float, matrix
		variance of a random variable or the covariance matrix for multiple variables
	nsamp : float
		number of samples or realizations
	nsite : float, optional
		number of sites, default = 1
	var_list : str, array, optional
		list of variables, default = ['pga', 'pgv']
		
	Returns
	-------
	sample_dict : float, dictionary
		[varies] a dictionary of entries equal to number of variables. Each entry contains **nsamp** of realizations by **nsites**
	
	"""
	## total number of cases = number of periods * number of sites
	ntot = len(mean)
	
	## number of periods = total / number of sites
	try:
		nperiod = len(var_list)
	except:
		nperiod = int(ntot/nsite)
	
	## generate sample from mean and cov and transpose
	mrs_out = np.random.multivariate_normal(mean, cov, nsamp).T
	
	# partition into individual IMs
	sample_dict = {}
	for i in range(nperiod):
		sample_dict[var_list[i]] = np.exp(mrs_out[i*nsite:(i+1)*nsite])
	
	##
	return sample_dict