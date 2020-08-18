#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### Input file for OpenSRA
#####
##### Created: August 5, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####
#####
##### Instructions
##### ------------
##### Currently, the program should be run in four phases:
##### 1. Get rupture scenarios and fault-crossing given target sites
##### 2. Get ground motion predictions at target sites given rupture scenarios
##### 3. Generate and store intensity measure realizations
##### 4. Assess EDPs, DMs, DVs using IM realizations
#####################################################################################################################


#####################################################################################################################
##### Python modules
import numpy as np

#####################################################################################################################
##### Setup and user inputs
## Define phase to run in OpenSRA; note that inputs should still be defined in order to setup the problem
## note: number corresponding to phase to run:
## 			1 = get rupture scenarios, and fault crossings and geologic units and properties given site
## 			2 = get GMM predictions of IMs
## 			3 = generate realizations of IMs (i.e., sampling)
## 			4 = assess EDPs, DMs, DVs
phase_to_run = 4
## phase display message
if phase_to_run == 1:
	phase_message = 'get rupture scenarios and fault crossings given site'
elif phase_to_run == 2:
    phase_message = 'get GMM predictions of IMs'
elif phase_to_run == 3:
    phase_message = 'generate realizations of IMs (i.e., sampling)'
elif phase_to_run == 4:
    phase_message = 'assess EDPs, DMs, DVs'

## Basic information
work_dir = r'C:\Users\barry\Desktop\results_for_chris' # working directory with input files and sub-directories
site_file_name = 'Balboa Blvd_OpenSRA Input File.csv' # file with site locations and site-specific inputs, should be placed under work_dir
geo_shp_file = r'lib\other\GDM_002_GMC_750k_v2_GIS\shapefiles\GMC_geo_poly.shp' # shapefile with geologic units, set path relative to OpenSRA base directory
geo_unit_param_file = r'lib\slate\Seismic Hazard_CGS_Unit Strengths.csv' # Micaela's file with properties for geologic units, set path relative to OpenSRA base directory

## Source and ground motion predictions
tr = 10000 # years, cutoff for return period
rmax = 100 # km, cutoff distance for calculation of ground motion predictions
gm_model = 'ngawest2_avg' # ground motion model to use (stick to ngawest2 for now)
src_model = 'ucerf3_1' # source model, available choices are ucerf2 and 3_1 or 2 use 'ucerf3_1' for now)
rup_per_group = 100 # number of ruptures to store per group
gm_tool = 'regionalprocessor' # options = 'regionalprocessor' or 'eqhazard' (use 'regionalprocessor' for now)
num_thread = 1 # number of threads for gm_tool
flag_include_point_source = False # include point source in selection of rupture scenarios

## Intensity measures
n_samp_im = 4 # number of samples for IM
flag_spatial_corr = False # True to apply spatial correlation
flag_cross_corr = True # True to apply spectral (cross) correlation
sample_algorithm = 'random' # choices = 'random' for random sampling, 'lhs' for latin-hypercube sampling
ims = ['pga','pgv'] # target IMs to get
flag_sample_with_sigma_total = True # True to sample from total sigma; False to sample intra- and inter-event sigmas separately then add together
sigma_aleatory = None # set value for uniform aleatory sigma; set to None to use GM predicted total sigma
n_decimals = 3 # number of decimals in log10 space for export
flag_force_resample_im = True # sample IM even if samples already exist

## Engineering demand parameters
## note: for now if EDP distribution is:
##       (1) lognormally distributed, samples are generated at [+1.65/0/-1.65] sigma with probabilities of 0.2,0.6,0.2
##       (2) uniformly distributed: samples are randomly generated, each with equal probability.
##       (3) other distribtions: not implemented
n_samp_edp = 4 # number of samples for EDP, for uniform distributions for now
## note: for list of EDPs, if either ls or gs is specified, must also include liq BEFORE BOTH ls and gs
edps = ['liq','ls','gs','land','surf'] # checked options = liq, ls, gs, land, surf; other options include subsurf
## note: (1) for each EDP demand, run one procedure at a time for now. The number branches may become too overwhelming
##       (2) keep procedures as lists
edp_procs = {
			'liq': {
					'method': ['zhu_etal_2017'],
					'return_param': [],
					'source_dict': ['_EDP_dict'],
					'source_param': [],
					'source_method': [],
					'flag_pga': False,
					'flag_pgv': True,
					'flag_M': True,
					'eps_aleatory': [],
					'wgt_aleatory': [],
					'eps_epistemic': []
					},
			'ls': {
					'method': ['hazus_2014_ls'],
					'return_param': ['pgd_ls'],
					'source_dict': ['_EDP_dict'],
					'source_param': ['liq_susc'],
					'source_method': ['zhu_etal_2017'],
					'flag_pga': True,
					'flag_pgv': False,
					'flag_M': True,
					'eps_aleatory': list(np.ones(n_samp_edp)),
					'wgt_aleatory': list(np.ones(n_samp_edp)/n_samp_edp),
					'eps_epistemic': [-1,0,1] # [0]
					},
			'gs': {
					'method': ['hazus_2014_gs'],
					'return_param': ['pgd_gs'],
					'source_dict': ['_EDP_dict'],
					'source_param': ['liq_susc'],
					'source_method': ['zhu_etal_2017'],
					'flag_pga': False,
					'flag_pgv': False,
					'flag_M': False,
					'eps_aleatory': list(np.ones(n_samp_edp)),
					'wgt_aleatory': list(np.ones(n_samp_edp)/n_samp_edp),
					'eps_epistemic': [-1,0,1] # [0]
					},
			'land': {
					'method': ['bray_macedo_2019'],
					'return_param': ['pgd_land'],
					'source_dict': ['_EDP_dict'],
					'source_param': [],
					'source_method': [],
					'flag_pga': True,
					'flag_pgv': True,
					'flag_M': True,
					'eps_aleatory': [-1.65,0,1.65],
					'wgt_aleatory': [0.2,0.6,0.2],
					'eps_epistemic': [-1.65,0,1.65] # [999]
					},
			'surf': {
					'method': ['wells_coppersmith_1994'],
					'return_param': ['pgd_surf'],
					'source_dict': [],
					'source_param': [],
					'source_method': [],
					'flag_pga': False,
					'flag_pgv': False,
					'flag_M': True,
					'eps_aleatory': [-1.65,0,1.65],
					'wgt_aleatory': [0.2,0.6,0.2],
					'eps_epistemic': [-1.65,0,1.65] # [999]
					}
			}
edp_other_params = {
					'dc_cutoff': 20, # for Zhu et al. (2017)
					'dw_cutoff': 50, # for lateral spreading
					'gm_type': 'general' # for Bray & Macedo (2019)
					}
			
########################################################################
################################### Damage section is under development - skip damages for now
## Damage measures
## note: for now if DM distribution is:
##       (1) lognormally distributed, samples are generated at [+1.65/0/-1.65] sigma with probabilities of 0.2,0.6,0.2
##       (2) uniformly distributed: samples are randomly generated, each with equal probability.
##       (3) other distribtions: not implemented
n_samp_dm = 4 # number of samples for DM, for uniform distributions for now
dms = [] # TBD
## note: (1) for each DM demand, run one procedure at a time for now. The number branches may become too overwhelming
##       (2) keep procedures as lists
dm_procs = {}
########################################################################

## Decision variables
## note: can specify multiple methods
flag_export_to_csv = True
dvs = ['rr_pgv','rr_pgd']
dv_procs = {
			'rr_pgv': {
					'method': [
								'orourke_2020_rr'
								], # base: orourke_2020_rr; other options: hazus_2014_rr, ala_2001_rr
					'return_param': ['rr_pgv'],
					'source_dict': ['_EDP_dict'],
                    'source_param': [],
					'source_method': [],
					'flag_pga': False,
					'flag_pgv': True,
					'flag_M': False,
					'eps_epistemic': [-1.65,0,1.65],
					'sigma_epistemic': 0.65
					},
			'rr_pgd': {
					'method': [
								'orourke_2020_rr'
								], # base: orourke_2020_rr; other options: hazus_2014_rr, ala_2001_rr
					'return_param': ['rr_pgd'],
					'source_dict': {				# specify sources for every type of deformation
									'ls': ['_EDP_dict', '_EDP_dict'],
									'gs': ['_EDP_dict', '_EDP_dict'],
									'land': ['_EDP_dict', '_EDP_dict'],
									'surf': ['_EDP_dict']
									},
					'source_param': {
									'ls': ['pgd_ls', 'p_liq'],
									'gs': ['pgd_gs', 'p_liq'],
									'land': ['pgd_land', 'p_land'],
									'surf': ['pgd_surf']
									},
					'source_method': {
									'ls': ['hazus_2014_ls', 'zhu_etal_2017'],
									'gs': ['hazus_2014_gs', 'zhu_etal_2017'],
									'land': ['bray_macedo_2019', 'bray_macedo_2019'],
									'surf': ['wells_coppersmith_1994']
									},
					'flag_pga': True,
					'flag_pgv': False,
					'flag_M': True,
					'eps_aleatory': list(np.ones(n_samp_edp)),
					'wgt_aleatory': list(np.ones(n_samp_edp)/n_samp_edp),
					'eps_epistemic': [-1,0,1] # [0]
					}
			}
dv_other_params = {
					'flag_rup_depend': True, # for rr_pgd, leave as is
					'pgd_cutoff': [4*2.54] # cm, for rr_pgd with orourke, leave as list but append 0s if using hazus or ala
					}