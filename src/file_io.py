#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Functions for input/output
##### 
##### Created: April 27, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Required packages
import numpy as np
import os, h5py, time
from scipy import sparse
#####################################################################################################################


#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_sha_sparse(gm_dir, rup_group, im_list, src, rup, M, rate):
    """
    Read outputs from OpenSHAInterface wrapper
    
    Parameters
    ----------
    gm_dir : str
        base directory of the **GM** outputs from OpenSHAInterface
    rup_group : str
        name of folder with **rups** to import (e.g., 0-99, 100:199)
        
    Returns
    -------
    gm_in : float, dict
        dictionary containing the nonzero site indices, and **PGA** and **PGV** predictions (i.e., means, sigmas)
    rup_meta : float, dict
        dictionary containing the rupture scenario information (UCERF3 index, magnitude, mean annual rate)
    
    """
    
    ## make list of variables
    param_names = ['mean', 'inter', 'intra']
    var_list = [i+'_'+j for i in im_list for j in param_names]
    
    ## number of rupture scenarios
    nRups = len(rup)
    
    ## starting and ending numbers for rupture scenarios
    # rup_start = int(group_num*100)
    # rup_end = int((group_num+1)*100-1)
    # folder_name = str(rup_start)+'_'+str(rup_end)
    rup_start = int(rup_group[0:next(i for i,val in enumerate(rup_group) if val=='_')])
    rup_end = int(rup_group[next(i for i,val in enumerate(rup_group) if val=='_')+1:])
    
    ## scenario parameters
    src4group = src[rup_start:rup_end+1]
    rup4group = rup[rup_start:rup_end+1]
    M4group = M[rup_start:rup_end+1]
    rate4group = rate[rup_start:rup_end+1]
    
    ## initialize dictionary
    gm_in = {}
    
    ## loop through variables
    for var in var_list:
    
        start_time = time.time()
        
        # if flag_GM_pred_exist is True:
            # mat = None
        # else:
        ## load file, store into im_data, then close
        mat = sparse.coo_matrix(sparse.load_npz(os.path.join(gm_dir,rup_group,var+'.npz'))) ## load npz
        # mat = sparse.load_npz(im_dir+'/'+folder_name+'/'+var+'.npz') ## load npz
            
        gm_in.update({var: mat})
        
        # if 'mean' in var:
            # rup_num_unique = np.unique(mat.row)+rup_start
            # sites_unique = np.unique(mat.col)
            # src_unique = src[rup_num_unique]
            # rup_unique = rup[rup_num_unique]
            # M_unique = M[rup_num_unique]
            # rate_unique = rate[rup_num_unique]
            
            # src_rup_unique = [str(src_unique[i])+'_'+str(rup_unique[i]) for i in range(len(rup_num_unique))]
            
        # print('imported '+var+'--- %10.6f seconds ---' % (time.time() - start_time))
    
    gm_in.update({'src': src4group,
                    'rup': rup4group,
                    'M': M4group,
                    'rate': rate4group})
    
    ##
    return gm_in
    
    
#####################################################################################################################
##### Reading other json files that are manually created
#####################################################################################################################
def read_json_other(path, var):
    ##
    with open(path, 'r') as f:
        jd = json.load(f)

    ##
    site = jd.get('Site',None)
    site_list = site.get('SiteList',None)
    
    ##
    data2get = []
    
    ##
    for i in range(len(site_list)):
        if site_list[i][var] is not None:
            data2get.append(site_list[i][var])
        else:
            data2get.append(np.nan)
    
    ##
    return data2get
    
#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_sitedata(path):
    """
    Read site locations
    """
    
    input = np.loadtxt(path,delimiter=',',unpack=True)
    site_loc = {'lon':input[0],'lat':input[1]}
    
    return site_loc
    

#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_EQHazard(path):
    
    ## load json file
    with open(EQHaz_file, 'r') as f:
        jd = json.load(f)

    ## set up empty labels for IM dictionary
    im_data = dict([(label, dict()) for label in [
        'site_data', 'im_data', 'im_type', 'site_loc', 'rup_meta'
    ]])

    ## partition json content into a few internal variables for convenience
    eq_info = jd.get('EqRupture',None)
    gm_input = jd.get('GroundMotions',None)

    ## items to get from EQHazard output
    im_list = ['SA','PGA','PGV']
    unit_list = ['g','g','cm_s']
    flag_log_list = ['ln','ln','ln']
    qnt_type_list = ['acc','acc','vel']
    
    ## initialize empty arrays
    site_lat = []
    site_long = []
    data_type = []
    data_val = []
    data_source = []
    im_qnt_type = []
    im_unit = []
    im_flag_log = []

    ## loop to store site and metadata information
    for i in range(len(gm_input)):
    
        ## store longitude and latitude
        site_loc = gm_input[i].get('Location',None)
        if site_loc is not None:
            site_lat.append(site_loc.get('Latitude', None))
            site_long.append(site_loc.get('Longitude', None))
        else:
            site_lat.append(None)
            site_long.append(None)

        ## store data type and source
        site_data = gm_input[i].get('SiteData',None)[0]
        if site_data is not None:
            data_type.append(site_data.get('Type', None))
            data_val.append(site_data.get('Value', None))
            data_source.append(site_data.get('Source', None))
        else:
            data_type.append(None)
            data_val.append(None)
            data_source.append(None)
    
    ## store IM measures
    for im in im_list:
    
        ## initialize empty arrays
        im_qnt = []
        im_mean = []
        im_sig_tot = []
        im_sig_intra = []
        im_sig_inter = []
        im_label = []
        
        ## loop through all sites
        for i in range(len(gm_input)):
        
            ## try to find target items specified in "im_list" and store them
            try:
                key_val = next(key_val for _, key_val in enumerate(gm_input[i].keys()) if im in key_val)
                im_qnt = gm_input[i][key_val]
                im_label = key_val
            except:
                im_qnt = None
                im_label = None

            ## if target IM measure is available, further partition its content into mean and sigma
            if im_qnt is not None:
                im_mean.append(im_qnt.get('Mean', None))
                im_sig_tot.append(im_qnt.get('TotalStdDev', None))
                im_sig_intra.append(im_qnt.get('IntraEvStdDev', None))
                im_sig_inter.append(im_qnt.get('InterEvStdDev', None))
            else:
                im_mean.append(None)
                im_sig_tot.append(None)
                im_sig_intra.append(None)
                im_sig_inter.append(None)

        ## if specral acceleration is specified, also store period
        if 'SA' in im and im_mean is not None:
            im_period = jd.get('Periods',None)
            im_data['im_data'].update({'period': im_period,
                                    im.lower()+'_mean': im_mean,
                                    im.lower()+'_sig_total': im_sig_tot,
                                    im.lower()+'_sig_intra': im_sig_intra,
                                    im.lower()+'_sig_inter': im_sig_inter})
        else:
            im_data['im_data'].update({im.lower()+'_mean': im_mean,
                                    im.lower()+'_sig_total': im_sig_tot,
                                    im.lower()+'_sig_intra': im_sig_intra,
                                    im.lower()+'_sig_inter': im_sig_inter})

        ## see if other information are provided (e.g., units, if values are in log)
        try:
            next(j for i, j in enumerate(gm_input[0].keys()) if im in j)
            im_qnt_type.append(qnt_type_list[i])
            im_unit.append(unit_list[i])
            im_flag_log.append(flag_log_list[i])
        except:
            pass

    ## update the IM data dictionary
    im_data['im_type'].update({'type': im_qnt_type,
                            'unit': im_unit,
                            'log_flag': im_flag_log})

    im_data['rup_meta'].update(eq_info)

    im_data['site_loc'].update({'latitude': site_lat,
                                'longitude': site_long})

    im_data['site_data'].update({'vs30': data_val,
                                'source': data_source})

    ##
    return im_data
    
    
#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_EventCalc(im_dir, site_loc_file, flag_export=False):
    """
    Read outputs from OpenSHA Event Calculator
    """
    
    ## get filenames in im_dir
    files = os.listdir(im_dir)
    print(files)
    
    ## GMMs used
    gm_label = [file[0:file.find('_PGA')] for file in files]
    gm_label = np.unique(gm_label)
    
    ## set up empty labels for IM dictionary
    im_in = {}
    rup_meta = {}
    
    ## get name for metadata file and mport and format data for storage
    metadata = [file for file in files if 'meta' in file.lower()][0]
    print(metadata)
    with open(im_dir + metadata) as f:
        lines = np.asarray([line.split()[0:4] for line in f])
    [rup_meta.update({str(i[0])+'_'+str(i[1]):{'M':float(i[3]),'r':float(i[2])}}) for i in lines]
    
    ## get filenames to import for PGA
    if flag_export is True:
        str_add = ''
    else:
        str_add = '_mean'
    
    files_pga = [file for file in files if 'pga'+str_add+'.txt' in file.lower()] # files for pga
    files_pgv = [file for file in files if 'pgv'+str_add+'.txt' in file.lower()] # files for pgv
    files_im = files_pga + files_pgv
    
    ## import and format data for storage
    for file in files_im:
        if 'pga' in file.lower():
            label = 'pga'
        elif 'pgv' in file.lower():
            label = 'pgv'
        
        if flag_export is True:
            mat = np.loadtxt(im_dir+file,unpack=True) # import file
            
            mat_red = mat[2:] # remove columns of source and rupture indices
            mean = np.asarray([mat_red[i] for i in range(len(mat_red)) if np.mod(i,3) == 0]) # retrieve ln_mean values
            sig_total = np.asarray([mat_red[i] for i in range(len(mat_red)) if np.mod(i,3) == 1])[0][0] # retrieve total sigma
            sig_inter = np.asarray([mat_red[i] for i in range(len(mat_red)) if np.mod(i,3) == 2])[0][0] # retrieve inter sigma
            sig_intra = (sig_total**2 - sig_inter**2)**0.5 # calculate intra sigma
            
            im_in.update({label+'_mean':mean,
                            label+'_sig_total':sig_total,
                            label+'_sig_intra':sig_inter,
                            label+'_sig_inter':sig_intra})
                            
            np.savetxt(im_dir+file[0:file.find('.txt')]+'_mean.txt',np.transpose(mean),fmt='%6.4f')
            np.savetxt(im_dir+file[0:file.find('.txt')]+'_sigma.txt',[sig_total,sig_inter,sig_intra],fmt='%1.3f')
            
        else:
            mean = np.loadtxt(im_dir+file)
            sig = np.loadtxt(im_dir+file[0:file.find('_mean.txt')]+'_sigma.txt')
            
            sig_total = sig[0]
            sig_inter = sig[1]
            sig_intra = sig[2]
            
            im_in.update({label+'_mean':mean,
                            label+'_sig_total':sig_total,
                            label+'_sig_intra':sig_inter,
                            label+'_sig_inter':sig_intra})
    
    if im_in is None:
        print('im_in is empty')
    
    ##
    return im_in, rup_meta
    
    
#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_OpenSHAInterface_old(im_dir, rup_meta_file, sha_meta_file, rup_num_start, rup_num_end):
    """
    Read outputs from OpenSHAInterface wrapper
    
    Parameters
    ----------
    im_dir : str
        directory of the **IM** outputs from OpenSHAInterface
    rup_meta_file : str
        full path for the file containing rupture metadata
    sha_meta_file : str
        full path for the file containing metadata from OpenSHAInterface
    rup_num_start : int
        starting rupture number (1 through nRups); default = 1
    rup_num_end : int
        ending rupture nubmer (1 through nRups); default = 1e10 (e.g., all ruptures)
        
    Returns
    -------
    im_in : float, dict
        dictionary containing the nonzero site indices, and **PGA** and **PGV** predictions (i.e., means, sigmas)
    rup_meta : float, dict
        dictionary containing the rupture scenario information (UCERF3 index, magnitude, mean annual rate)
    
    """
    
    ## open rup_meta_file and load in rupture information
    with h5py.File(rup_meta_file, 'r') as f:
        src = f.get('src')[:]
        rup = f.get('rup')[:]
        M = f.get('M')[:]
        rate = f.get('rate')[:]
    f.close()
        
    ## number of rupture scenarios
    nRups = len(rup)
    
    ## load sha_meta_file and extra information on nonzero IMs (sparse coordinates)
    sha_meta = np.loadtxt(sha_meta_file,delimiter=',',unpack=True)
    
    ## extract start and end rupture numbers for importing 
    rup_start = sha_meta[1].astype(np.int32)
    rup_end = sha_meta[2].astype(np.int32)
    nFiles = len(rup_start)
    i_start = 0 if rup_num_start == 0 else next(ind-1 for ind in range(nFiles) if rup_start[ind] >= rup_num_start)
    i_end = nFiles-1 if rup_num_end >= rup_end[nFiles-1]+1 else next(ind for ind in range(nFiles) if rup_end[ind]+1 >= rup_num_end)
    
    ## create list of ruptures to import based on input
    rup_list = np.arange(rup_num_start-1,min(rup_num_end,rup_end[nFiles-1]+1))
    
    ## loop to impot im_data
    for i in range(i_start, i_end+1):
        ## range of ruptures in file to be loaded
        rup_start_i = rup_start[i]
        rup_end_i = rup_end[i]
        
        ## load file, store into im_data, then close
        im_file = 'im_' + str(rup_start_i) + '_' + str(rup_end_i) + '.hdf5' ## im file name to load
        with h5py.File(im_dir+im_file, 'r') as f:
            im_data = f.get('im')[:]
        f.close()
        
        ## get the counting index for the rupture (not UCERF3 index)
        row = im_data[0].astype(np.int32)
        col = im_data[1].astype(np.int32) # load in nonzero cols now, easier to convert to integers
        
        ## store ruptures that are in rup_list
        im_in = {}
        for j in np.unique(row):
            if j in rup_list:
                im_in.update({str(src[j])+'_'+str(rup[j]):{'site_nonzero':col[row==i], 
                                                        'pga_mean':im_data[2][row==i], 
                                                        'pga_sig_inter':im_data[3][row==i], 
                                                        'pga_sig_intra':im_data[4][row==i], 
                                                        'pgv_mean':im_data[5][row==i], 
                                                        'pgv_sig_inter':im_data[6][row==i], 
                                                        'pgv_sig_intra':im_data[7][row==i]}})
    
    ## create rup_meta dictionary
    rup_meta = {}
    for i in rup_list:
        rup_meta.update({str(src[i])+'_'+str(rup[i]):{'M':np.round(M[i],decimals=2),
                                                    'r':np.round(rate[i],decimals=6)}})

    ##
    return im_in, rup_meta
    
    
#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_sha_hdf5(im_dir, rup_meta_file, sha_meta_file, group_num, num_rups):
    """
    Read outputs from OpenSHAInterface wrapper
    
    Parameters
    ----------
    im_dir : str
        base directory of the **IM** outputs from OpenSHAInterface
    rup_meta_file : str
        full path for the file containing rupture metadata hdf5 file
    sha_meta_file : str
        full path for the file containing metadata from OpenSHAInterface
    group_num : int
        number for folder to import (e.g., 0 = 0-99, 1 = 100:199)
    num_rups : int
        first n ruptures in the group to uploaded
        
    Returns
    -------
    im_in : float, dict
        dictionary containing the nonzero site indices, and **PGA** and **PGV** predictions (i.e., means, sigmas)
    rup_meta : float, dict
        dictionary containing the rupture scenario information (UCERF3 index, magnitude, mean annual rate)
    
    """
    
    ## default variable names
    in_var = ['pga_mean','pga_inter','pga_intra','pgv_mean','pgv_inter','pgv_intra']
    save_Var = ['pga_mean','pga_sig_inter','pga_sig_intra','pgv_mean','pgv_sig_inter','pgv_sig_intra']
    
    ## open rup_meta_file and load in rupture information
    with h5py.File(rup_meta_file, 'r') as f:
        src = f.get('src')[:]
        rup = f.get('rup')[:]
        M = f.get('M')[:]
        rate = f.get('rate')[:]
    f.close()
        
    ## number of rupture scenarios
    nRups = len(rup)
    
    ## load sha_meta_file and extra information on nonzero IMs (sparse coordinates)
    sha_meta = np.loadtxt(sha_meta_file,delimiter=',',unpack=True)
    
    ## extract start and end rupture numbers for importing 
    rup_start = sha_meta[0].astype(np.int32)
    rup_end = sha_meta[1].astype(np.int32)
    nFiles = len(rup_start)
    
    ## loop to import im_data
    im_in = {}
    ## loop through variables
    for i in range(len(in_var)):
    
        ## load file, store into im_data, then close
        mat = sparse.load_npz(im_dir+'/'+str(rup_start[group_num])+'_'+str(rup_end[group_num])+'/'+in_var[i]+'.npz') ## load npz
        mat = mat.toarray()
        
        ## search for nonzeros for first variable to importe: pga_mean
        if in_var[i] == 'pga_mean':
            rows,cols = np.where(mat > 0)
            [im_in.update({str(src[rup_start[group_num]+row])+'_'+str(rup[rup_start[group_num]+row]):{
                'site_nonzero':cols[rows==row],
                save_Var[i]:mat[rows[rows==row],cols[rows==row]]}}) for row in range(int(num_rups))]
        else:
            [im_in[str(src[rup_start[group_num]+row])+'_'+str(rup[rup_start[group_num]+row])].update({
                save_Var[i]:mat[rows[rows==row],cols[rows==row]]}) for row in range(int(num_rups))]
    
    ## create rup_meta dictionary
    rup_meta = {}
    for row in range(num_rups):
        rup_meta.update({str(src[rup_start[group_num]+row])+'_'+str(rup[rup_start[group_num]+row]):{
        'M':np.round(M[rup_start[group_num]+row],decimals=2),
        'r':np.round(rate[rup_start[group_num]+row],decimals=6)}})
    
    ##
    return im_in, rup_meta