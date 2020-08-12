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
import os, time, logging
import numpy as np

##### SimCenter modules
from lib.simcenter.OpenSHAInterface import *


#####################################################################################################################
##### initialize processor
logging.info(f"return period = {tr} years.")

baseDir = r'C:\Users\barry\Desktop\CEC\OpenSRA_local'  ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
# baseDir = os.getcwd()  ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
# region = 'allsites'

logging.info(f"=============================================")
logging.info(f"baseDir = {baseDir}.")
logging.info(f"runDir = {runDir}.")
logging.info(f"saveDir = {saveDir}.")
logging.info(f"rupMetaFile = {rupMetaFile}.")
logging.info(f"siteFile = {siteFile}.")
logging.info(f"vs30File = {vs30File}.")
logging.info(f"=============================================")


##
if not os.path.isdir(saveDir):
	logging.info(f"save directory does not exist --- create it.")
	os.mkdir(saveDir)
try:
	var = processor
	logging.info(f"processor exists.") ## used this in Jupyter to avoid initializing every time
except NameError:
	logging.info(f"initialized processor.")
	processor = init_processor(siteFile, vs30File, numThreads=4)


##
logging.info(f"=============================================")
if tr == 1000:
	nScens = 54845
elif tr == 10000:
	nScens = 224259
elif tr == 100000:
	nScens = 437401
logging.info(f"number of scenarios = {nScens}.")


##
num_rup_per_group = 100 ## run and store every 100 ruptures, i.e., sparse matrix dimension = 100 x n_sites
groups = np.arange(0,nScens,num_rup_per_group)
if groups[len(groups)-1] < nScens:
	groups = np.hstack([groups,nScens])
logging.info(f'number of groups of {num_rup_per_group} scenarios = {len(groups)}; this is the number of folders/sets of stored files.')

listVar = ['pga_mean','pga_inter','pga_intra','pgv_mean','pgv_inter','pgv_intra']
logging.info(f"list of parameters to get = {listVar}.")


##
multi = 0 ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< update this number: groups_end varies from 0 to ~2240, where every range2 index = 100 ruptures
groups_start = multi*num_rup_per_group
groups_end = groups_start + num_rup_per_group
incr_groups = 10 ## <<<<<<<<<<<<<< I was using this to export messages after certain groups, just leave it as is.
sets_of_groups_to_run = np.arange(groups_start,groups_end,incr_groups)
logging.info(f"sets of groups to run = {sets_of_groups_to_run}.")


##
if flag_track_time:
	start_time = time.time()
for i in range(len(sets_of_groups_to_run)):
	logging.info(f"=============================================")
	for j in range(sets_of_groups_to_run[i],sets_of_groups_to_run[i]+incr_groups):
		##
		saveDir_group = os.path.join(saveDir,str(groups[j])+'_'+str(groups[j+1]-1))
		if not os.path.isdir(saveDir_group):
			os.mkdir(saveDir_group)
		##
		runHazardAnalysis(processor, rupMetaFile, [groups[j],groups[j+1]], saveDir_group, listVar, 1/tr)
		if flag_track_time:
			end_time = time.time()
			logging.info(f"j = {j}: scenarios {groups[j]} to {groups[j+1]} --- {np.round(end_time-start_time,decimals=2)}.")
			start_time = time.time()
		else:
			logging.info(f"j = {j}: scenarios {groups[j]} to {groups[j+1]}.")
logging.info(f"=============================================")
logging.info(f"...done with current set.")
