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
##### Python modules
import numpy as np
from scipy.linalg import cholesky
from numpy.random import standard_normal

##### OpenSRA modules and functions
from src.fcn_gen import convert_array_to_sym_mat


#####################################################################################################################
def get_cov(corr_d_pga, corr_d_pgv, corr_T, pga_sigma, pgv_sigma, dim_d, dim_T=2):
	"""
	Compute correlation and covariance matrices for PGA and PGV over a list of sites
	
	Parameters
	----------
	corr_d_pga : float, array
		list of spatial correlations across sites for **PGA**
	corr_d_pgv : float, array
		list of spatial correlations across sites for **PGV**
	corr_T : float, array
		list of spectral correlation between PGA (T ~ 0.01sec) and PGV (T ~ 1sec)
	pga_sigma : float, array
		sigmas for **PGA** for the list of sites
	pgv_sigma : float, array
		sigmas for **PGV** for the list of sites
	dim_d : int
		number of sites
	dim_T : int, optional
		number of periods (fixed to 2 for now)
		
	Returns
	-------
	cov_mat_corr : float, arry
		correlated covariance matrix
	
	"""
	
	## cross of spatial correlations between PGA and PGV
	corr_d_cross = np.sqrt(np.multiply(corr_d_pga,corr_d_pgv))
	
	# convert arrays to symmetric matrices
	corr_T_mat = convert_array_to_sym_mat(corr_T,dim_T) # convert spectral correlations to symmetric matrix
	corr_d_pga_mat = convert_array_to_sym_mat(corr_d_pga,dim_d) # convert PGA spatial correlations to symmetric matrix
	corr_d_pgv_mat = convert_array_to_sym_mat(corr_d_pgv,dim_d) # convert PGV spatial correlations to symmetric matrix
	corr_d_cross_mat = convert_array_to_sym_mat(corr_d_cross,dim_d) # convert PGAxPGV correlations to symmetric matrix
	
	## full correlation matrix
	corr_quad11 = corr_d_pga_mat*corr_T_mat[0][0] # upper-left quadrant, PGA x PGA
	corr_quad12 = corr_d_cross_mat*corr_T_mat[0][1] # upper-right quadrant, PGV x PGA
	corr_quad21 = np.transpose(corr_quad12) # lower-left quadrant, PGA x PGV
	corr_quad22 = corr_d_pgv_mat*corr_T_mat[1][1] # lower-right quadrant, PGV x PGV
	# corr_mat = np.bmat([[corr_quad11, corr_quad12], [corr_quad21, corr_quad22]])
	corr_mat = np.vstack([np.hstack([corr_quad11,corr_quad12]),np.hstack([corr_quad21,corr_quad22])])

	## joint uncorrelated covariance matrix
	cov_quad11 = np.outer(pga_sigma,pga_sigma) # upper-left quadrant, PGA x PGA
	cov_quad12 = np.outer(pga_sigma,pgv_sigma) # upper-right quadrant, PGV x PGA
	cov_quad21 = np.transpose(cov_quad12) # lower-left quadrant, PGA x PGV
	cov_quad22 = np.outer(pgv_sigma,pgv_sigma) # lower-right quadrant, PGV x PGV
	# cov_mat_uncorr = np.bmat([[cov_quad11, cov_quad12], [cov_quad21, cov_quad22]])
	cov_mat_uncorr = np.vstack([np.hstack([cov_quad11,cov_quad12]),np.hstack([cov_quad21,cov_quad22])])
	
	## joint correlated covariance matrix
	cov_mat_corr = np.multiply(cov_mat_uncorr,corr_mat)
	
	##
	return cov_mat_corr
	
	
#####################################################################################################################
def get_RV_sims(mean, cov, nsamp, nsite=1, var_list=['pga','pgv']):
	"""
	Perform multivariate sampling for normally distributed random variables and partition the results according to input variables. Currently works with two variables max, but as many sites as desired 
	
	Parameters
	----------
	mean : float, array
		[ln(g) or ln(cm/s)] mean of a random variable or a list for multiple variables
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
		[g or cm/s] a dictionary of entries equal to number of variables. Each entry contains **nsamp** of realizations by **nsites**
	
	"""
	
	## total number of cases = number of periods * number of sites
	ntot = len(mean)
	
	## number of periods = total / number of sites
	nperiod = len(var_list)
	
	## generate sample from mean and cov and transpose
	l = cholesky(cov, check_finite=False, overwrite_a=True)
	mrs_out = np.transpose(np.asarray([mean + l.dot(standard_normal(len(mean))) for i in range(nsamp)]))
	
	# mrs_out = np.random.multivariate_normal(mean, cov, nsamp).T
	
	# partition into individual IMs
	sample_dict = {}
	for i in range(nperiod):
		sample_dict[var_list[i]] = np.exp(mrs_out[i*nsite:(i+1)*nsite])
	
	##
	return sample_dict