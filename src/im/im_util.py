# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Additional functions for intensity measures
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import os
import json
# import sys
import logging
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy import sparse
import geopandas as gpd
from shapely.geometry import Polygon
# from scipy.linalg import cholesky
# from numpy.random import standard_normal
from scipy.interpolate import interp2d

# OpenSRA modules and functions
from src import util


# -----------------------------------------------------------
def store_IM_sample(file, sample, store_file_type, n_decimals=None):
    """
    Store generated IM sample
    """
    
    # round sample to reduce storage size
    # if n_decimals is not None:
        # sample.data = np.round(sample.data,decimals=n_decimals)
    # check file format to store
    if store_file_type == 'npz':
        sparse.save_npz(file,sample)
    #
    elif store_file_type == 'txt':
        np.savetxt(file,sample.toarray(),fmt='%.3e')
        # sample = sparse.coo_matrix(sample) # coo matrix is easier to understand and reconstruct
        # np.savetxt(file,np.transpose([sample.row,sample.col,sample.data]),fmt='%i %i %f')


# -----------------------------------------------------------
def read_IM_sample(file, store_file_type, n_event=None, n_site=None):
    """
    Read generated IM sample
    """
    
    # check file format to read
    if store_file_type == 'npz':
        data = sparse.coo_matrix(sparse.load_npz(file))
    #
    elif store_file_type == 'txt':
        if n_event is None or n_site is None:
            data = sparse.coo_matrix(np.loadtxt(file))
        else:
            data = sparse.coo_matrix(np.loadtxt(file),shape=(n_event,n_site))
        # data_import = np.loadtxt(file,unpack=True)
        # if len(data_import.shape) == 1:
            # data_import = data_import[:,np.newaxis]
        # data = sparse.coo_matrix((data_import[2],(data_import[0],data_import[1])),shape=(n_event,n_site))
    #
    return data
        

# -----------------------------------------------------------
def get_correlated_residuals(chol_mat, n_sample, n_event, dim3, 
    cross_corr=0, prev_residuals=None, algorithm='random', ):
    """
    Get n_sample of correlated residuals using cholesky matrix. **dim3** = **n_site** for spatial and **n_IM** for spectral.
    
    For conditioned residuals of second IM, provide the cross-correlation and residuals of the first IM
    
    Algorithms include: "Random" and "LHS".
    
    """
    
    # get uncorrelated residuals
    if 'random' in algorithm.lower():
        residuals = np.random.normal(size=(n_sample,n_event,dim3))
    elif 'lhs' in algorithm.lower() or 'latin' in algorithm.lower():
        residuals = np.zeros((n_sample,n_event,dim3))
        for i in range(n_event):
            residuals[:,i,:] = util.lhs(dim3, n_sample, 'normal')
    # get correlated residuals
    if cross_corr == 0:
        # loop through each sample
        for i in range(n_sample):
            residuals[i,:,:] = (chol_mat @ residuals[i,:,:].T).T
    else:
        for i in range(n_sample):
            # part 1/2 of eq.
            residuals[i,:,:] = cross_corr * \
                (chol_mat @ prev_residuals[i,:,:].T).T
            # part 2/2 of eq.
            residuals[i,:,:] = np.sqrt(1-cross_corr**2) * \
                (chol_mat @ residuals[i,:,:].T).T
    
    # initialize
    # residuals = np.zeros((n_sample, n_event, dim3))
    # sample for index in dim3
    #if 'random' in algorithm.lower():
    #    residuals[:,:,0] = np.random.normal(size=(n_sample,n_event))
    #elif 'lhs' in algorithm.lower() or 'latin' in algorithm.lower():
    #    # -----------------------------------------------------------
    #    # need to replace with LHS
    #    residuals[:,:,0] = np.random.normal(size=(n_sample,n_event))
    #    # -----------------------------------------------------------
    ## loop through rest of indices in dim3, get conditional distribution and sample
    #for i in range(dim3):
    #    cond_mean = corr_mat[i-1,i]*residuals[:,:,i-1]
    #    cond_sigma = np.sqrt((1-corr_mat[i-1,i]**2))
    #    if 'random' in algorithm.lower():
    #        residuals[:,:,i] = np.random.normal(size=(n_sample,n_event))*cond_sigma + cond_mean
    #    elif 'lhs' in algorithm.lower() or 'latin' in algorithm.lower():
    #        pass
    #
    return residuals


# -----------------------------------------------------------
def read_ShakeMap_data(sm_dir, event_names, sites, im_dir, store_events_file, trace_save_name=None, 
    im_list=['PGA','PGV'], interp_scheme='linear', out_of_bound_value=0,
    epistemic_default=0.25, aleatory_default=0.01, flag_export_metadata=True):
    """
    Reads ShakeMap data, gets IM values at sites, and exports. Option to export metadata to JSON and grid_data to txt file for easier access.
    
    Function looks for three files from ShakeMap directory: grid.xml (required), uncertainty.xml, rupture.json
    
    For a list of sites, interpolation will be performed to obtain im_list values at these locations using the specified interpolation scheme. Default extrapolated value outside of grid boundary is 0 unless specified.
    
    Parameters
    ----------
    sm_dir : str
        directory with ShakeMap events
    sites : float, list
        list of coordinate pairs
    im_list : str, list
        list of intensity measure types to retrieve from XML file; default = [**'PGA'**, **'PGV'**]
    interp_scheme : {‘linear’, ‘cubic’, ‘quintic’}, optional
        interpolation scheme for estimating im_list value at site; default = **'linear'** (see :xref:`scipy.interpolate.interp2d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html>`)
    out_of_bound_value : float, optional
        value to use when site is outside the gridden boundary; default = 0 (see :xref:`scipy.interpolate.interp2d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html>`)
    export_metadata : boolean, optional
        True to export parsed metadata into JSON; default = **True**

    Returns
    -------
    
    """
    
    # set up
    num_events = len(event_names)
    shape = (len(event_names),len(sites))
    export_IM = {}
    export_sigma = {}
    export_sigma_mu = {}
    index_for_ims_full = {}
    for im_i in im_list:
        export_IM[im_i] = np.zeros(shape)
        export_sigma[im_i] = np.zeros(shape)
        export_sigma_mu[im_i] = np.zeros(shape)
        index_for_ims_full[im_i] = []
    export_traces = {}
    export_metadata = {}
    export_listOfScenarios = np.zeros((len(event_names),4)) # stick to format used for OpenSHA: [source ID, rupture ID, mag, rate]
    export_listOfScenarios[:,3] = 1 # set rates to 1 for ShakeMap events
    output = {}
    trace_set = []
    
    # loop through all events
    counter = 0
    for event_i in event_names:
        counter += 1
        curr_sm_dir = os.path.join(sm_dir,event_i)
        export_listOfScenarios[counter-1,0] = counter
        export_metadata['Event_'+str(counter)] = {
            'Label': event_i,
            'Metadata': {},
            'Units': {}
        }
        
        # see what files are given
        files = os.listdir(curr_sm_dir)
        if not 'grid.xml' in files:
            logging.info(f'"grid.xml" is not available; mean IMs and StdDev set to 0')
            
        else:
            # if sites is not 2D, expand dimension
            while np.ndim(sites) < 1:
                sites = np.expand_dims(sites,axis=0)
            
            # -----------------------------------------------------------
            # first parse "grid.xml"
            root = ET.parse(os.path.join(curr_sm_dir,'grid.xml')).getroot()
            # list of unique tags in ShakeMap "grid.xml"
            grid_xml_unique_tag = ['root','event','grid_specification','event_specific_uncertainty','grid_field','grid_data']

            # create dictionary with ShakeMap tags
            gm_dict = dict.fromkeys(grid_xml_unique_tag)
            # update dictionary with root attributes
            gm_dict['root'] = root.attrib
            for key in list(gm_dict['root']):
                if '}' in key:
                    new_key = key[key.find('}')+1:]
                    gm_dict['root'][new_key] = gm_dict['root'][key]
                    gm_dict['root'].pop(key)

            # loop through all tags in the xml file
            for child in root:
                current_tag = child.tag[child.tag.find('}')+1:] # get current tag
                # if tag does not exist as a key, initialize it as a dictionary
                if not isinstance(gm_dict[current_tag],dict):
                    gm_dict[current_tag] = {}
                current_attribute = child.attrib # get list of attributes for current child

                # skip 'event_specific_uncertainty'; uncertainty information will be imported from "uncertainty.xml"
                if current_tag == 'event_specific_uncertainty':
                    pass
                # read 'grid_field' information
                elif current_tag == 'grid_field':
                    if not child.get('index') in gm_dict[current_tag].keys():
                        gm_dict[current_tag].update({child.get('index'):{}})
                        for sub_attrib in current_attribute:
                            if not sub_attrib == 'index':
                                try:
                                    gm_dict[current_tag][child.get('index')].update({sub_attrib:float(child.get(sub_attrib))})
                                except:
                                    gm_dict[current_tag][child.get('index')].update({sub_attrib:child.get(sub_attrib)})
                # parse 'grid_data' into matrix
                elif current_tag == 'grid_data':
                    gm_dict[current_tag] = []
                    for line in child.text.splitlines():
                        if len(line) > 0:
                            gm_dict[current_tag].append([float(val) for val in line.split()])
                    gm_dict[current_tag] = np.asarray(gm_dict[current_tag])
                # read other tags
                else:
                    for sub_attrib in current_attribute:
                        try:
                            gm_dict[current_tag].update({sub_attrib:float(child.get(sub_attrib))})
                        except:
                            gm_dict[current_tag].update({sub_attrib:child.get(sub_attrib)})

            # from 'grid_field' get positions for coordinates and target IMs
            index_for_coords = [0]*2 # order = [lon, lat]
            index_for_ims = [0]*len(im_list) # order = [im_list1, im_list2, ...]
            count_im_list = 0
            for field in gm_dict['grid_field']:
                # check for 'lon'
                if gm_dict['grid_field'][field]['name'].lower() in 'longitude':
                    index_for_coords[0] = int(field)-1
                # check for 'lat'
                if gm_dict['grid_field'][field]['name'].lower() in 'latitude':
                    index_for_coords[1] = int(field)-1
                # check for IMs
                for im_i in im_list:
                    if im_i.lower() in gm_dict['grid_field'][field]['name'].lower():
                        index_for_ims[count_im_list] = int(field)-1
                        count_im_list += 1
                        break
            # extract coordinates and target IMs from grid data
            xml_data_of_coords = gm_dict['grid_data'][:,index_for_coords]
            xml_data_of_ims = gm_dict['grid_data'][:,index_for_ims]

            # prepare for interpolation
            if interp_scheme == 'nearest': # nearest neighbor
                # use sindex in geopandas for nearest neighbor
                gdf_gm = gpd.GeoDataFrame(
                    # xml_data_of_ims,
                    # columns=im_list,
                    geometry=gpd.points_from_xy(xml_data_of_coords[:,0], xml_data_of_coords[:,1])
                )
                # make boundary to first filter points within ShakeMap extent
                bound = gpd.GeoDataFrame(geometry=[Polygon([
                    [gm_dict['grid_specification']['lon_min'], gm_dict['grid_specification']['lat_min']],
                    [gm_dict['grid_specification']['lon_min'], gm_dict['grid_specification']['lat_max']],
                    [gm_dict['grid_specification']['lon_max'], gm_dict['grid_specification']['lat_max']],
                    [gm_dict['grid_specification']['lon_max'], gm_dict['grid_specification']['lat_min']],
                ])])
                # turn sites into geoseries
                gs_sites = gpd.GeoSeries(gpd.points_from_xy(sites[:,0],sites[:,1]))
                # get points within boundary
                sites_in_bound = gs_sites.sindex.query_bulk(bound.geometry, predicate='intersects')[1]
                sites_not_in_bound = list(set(list(range(sites.shape[0]))).difference(set(sites_in_bound)))
                # for sites in bound, get nearest neighbor
                # first make new geoseries
                gs_sites_in_bound = gpd.GeoSeries(gpd.points_from_xy(sites[sites_in_bound,0],sites[sites_in_bound,1]))
                # get nearest neighbor for sites
                nearest_sm_node = gdf_gm.sindex.nearest(gs_sites_in_bound)[1]
            else:
                # get unique lon and lat array, only used for linear and higher level interpolation schemes
                grid_lon = np.unique(xml_data_of_coords[:,0])
                grid_lat = np.unique(xml_data_of_coords[:,1])
            # initialize array for interpolated means and uncertainty
            site_gm = np.zeros((sites.shape[0],len(im_list)))
            
            # interpolate for median IMs
            for i in range(len(im_list)):
                # based on scheme
                if interp_scheme == 'nearest':
                    # get im values
                    site_gm[sites_in_bound,i] = xml_data_of_ims[nearest_sm_node,i]
                    site_gm[sites_not_in_bound,i] = out_of_bound_value
                else:
                    # get the gridded IMs value
                    grid_gm = np.flipud(np.reshape(xml_data_of_ims[:,i],(len(grid_lat),len(grid_lon))))
                    # create interpolation function
                    interp_function = interp2d(grid_lon, grid_lat, grid_gm, kind=interp_scheme, fill_value=out_of_bound_value)
                    # interpolate at sites
                    site_gm[:,i] = np.transpose([interp_function(sites[j,0],sites[j,1]) for j in range(sites.shape[0])])        
                # multiply by 100 if IM values are in %
                for pct_str in ['%','pct']:
                    if pct_str in gm_dict['grid_field'][str(index_for_ims[i]+1)]['units']:
                        site_gm = site_gm/100
                        gm_dict['grid_field'][str(index_for_ims[i]+1)]['units'] = \
                            gm_dict['grid_field'][str(index_for_ims[i]+1)]['units'].replace(pct_str,'')
                        break
            index_for_ims_copy = index_for_ims
            
            # -----------------------------------------------------------
            # parse "uncertainty.xml"
            if not 'uncertainty.xml' in files:
                logging.info(f'"uncertainty.xml" is not available, set aleatory to {aleatory_default} and epistemic to {epistemic_default}')
                site_sigma = np.ones((sites.shape[0],len(im_list)))*aleatory_default
                site_sigma_mu = np.ones((sites.shape[0],len(im_list)))*epistemic_default
            else:
                root = ET.parse(os.path.join(curr_sm_dir,'uncertainty.xml')).getroot()
                # list of unique tags in ShakeMap "grid.xml"
                grid_xml_unique_tag = ['root','event','grid_specification','grid_field','grid_data']

                # create dictionary with ShakeMap tags
                sigma_dict = dict.fromkeys(grid_xml_unique_tag)

                # loop through all tags in the xml file
                for child in root:
                    current_tag = child.tag[child.tag.find('}')+1:] # get current tag
                    # if tag does not exist as a key, initialize it as a dictionary
                    if not isinstance(sigma_dict[current_tag],dict):
                        sigma_dict[current_tag] = {}
                    current_attribute = child.attrib # get list of attributes for current child

                    # read 'grid_field' information
                    if current_tag == 'grid_field':
                        if not child.get('index') in sigma_dict[current_tag].keys():
                            sigma_dict[current_tag].update({child.get('index'):{}})
                            for sub_attrib in current_attribute:
                                if not sub_attrib == 'index':
                                    try:
                                        sigma_dict[current_tag][child.get('index')].update({sub_attrib:float(child.get(sub_attrib))})
                                    except:
                                        sigma_dict[current_tag][child.get('index')].update({sub_attrib:child.get(sub_attrib)})
                    # parse 'grid_data' into matrix
                    elif current_tag == 'grid_data':
                        sigma_dict[current_tag] = []
                        for line in child.text.splitlines():
                            if len(line) > 0:
                                sigma_dict[current_tag].append([float(val) for val in line.split()])
                        sigma_dict[current_tag] = np.asarray(sigma_dict[current_tag])
                    # skip tags except 'grid_field' and 'grid_data'
                    else:
                        pass

                # from 'grid_field' get positions for coordinates and target IMs
                index_for_coords = [0]*2 # order = [lon, lat]
                index_for_ims = [0]*len(im_list) # order = [im_list1, im_list2, ...]
                count_im_list = 0
                for field in sigma_dict['grid_field']:
                    # check for 'lon'
                    if sigma_dict['grid_field'][field]['name'].lower() in 'longitude':
                        index_for_coords[0] = int(field)-1
                    # check for 'lat'
                    if sigma_dict['grid_field'][field]['name'].lower() in 'latitude':
                        index_for_coords[1] = int(field)-1
                    # check for IMs
                    for im_i in im_list:
                        if im_i.lower() in sigma_dict['grid_field'][field]['name'].lower():
                            index_for_ims[count_im_list] = int(field)-1
                            count_im_list += 1
                            break
                # extract coordinates and target IMs from grid data
                xml_data_of_coords = sigma_dict['grid_data'][:,index_for_coords]
                xml_data_of_ims = sigma_dict['grid_data'][:,index_for_ims]

                # prepare for interpolation
                if interp_scheme == 'nearest': # nearest neighbor
                    # already performed nearest neighbor search when getting means
                    pass
                else:
                    # get unique lon and lat array, only used for linear and higher level interpolation schemes
                    grid_lon = np.unique(xml_data_of_coords[:,0])
                    grid_lat = np.unique(xml_data_of_coords[:,1])
                # initialize array for interpolated means and uncertainty
                site_sigma = np.zeros((sites.shape[0],len(im_list)))
                site_aleatory = np.zeros((sites.shape[0],len(im_list)))
                # site_aleatory = np.ones((sites.shape[0],len(im_list)))*aleatory_default
                # site_aleatory = np.ones((sites.shape[0],len(im_list)))*aleatory_default
                site_epistemic = np.zeros((sites.shape[0],len(im_list)))

                # interpolate for median IMs
                for i in range(len(im_list)):
                    # based on scheme
                    if interp_scheme == 'nearest':
                        # get im values
                        site_sigma[sites_in_bound,i] = xml_data_of_ims[nearest_sm_node,i]
                        site_sigma[sites_not_in_bound,i] = out_of_bound_value
                    else:
                        # get the gridded IMs value
                        grid_sigma = np.flipud(np.reshape(xml_data_of_ims[:,i],(len(grid_lat),len(grid_lon))))
                        # create interpolation function
                        interp_function = interp2d(grid_lon, grid_lat, grid_sigma, kind=interp_scheme, fill_value=out_of_bound_value)
                        # interpolate at sites
                        site_sigma[:,i] = np.transpose([interp_function(sites[j,0],sites[j,1]) for j in range(sites.shape[0])])
                    print(out_of_bound_value)
                    print(site_sigma[:,i])
                    print(site_sigma[:,i]!=out_of_bound_value)
                    site_aleatory[site_sigma[:,i]!=out_of_bound_value,i] = aleatory_default
                    site_epistemic[site_sigma[:,i]!=out_of_bound_value,i] = np.sqrt(
                        site_sigma[site_sigma[:,i]!=out_of_bound_value,i]**2 - \
                        site_aleatory[site_sigma[:,i]!=out_of_bound_value,i]**2
                    )

            # -----------------------------------------------------------
            # compile rupture traces in a single file
            if not 'rupture.json' in files:
                logging.info(f'\t"rupture.json" is not available; default to event epicenter')
                trace_set.append([
                    gm_dict['event']['lon'],
                    gm_dict['event']['lat'],
                    gm_dict['event']['depth']
                ])
            else:
                logging.info(f'\t"rupture.json" exists; reading fault traces')
                with open(os.path.join(curr_sm_dir,'rupture.json'),'r') as f:
                    sm_rupture = json.load(f)
                coords_to_read = sm_rupture['features'][0]['geometry']['coordinates'][0][0]
                trace_set_i = []
                for coord_i in coords_to_read:
                    trace_set_i.append(coord_i[:2])
                trace_set.append(trace_set_i)
                # logging.info(f'\tcode on processing of rupture.json not ready')
                
            # -----------------------------------------------------------
            # update export params
            for i in range(len(im_list)):
                export_IM[im_list[i]][counter-1,:] = site_gm[:,i].T
                # export_sigma[im_list[i]][counter-1,:] = site_sigma[:,i].T
                export_sigma[im_list[i]][counter-1,:] = site_aleatory[:,i].T
                export_sigma_mu[im_list[i]][counter-1,:] = site_epistemic[:,i].T
                export_metadata['Event_'+str(counter)]['Units'][im_list[i]] = gm_dict['grid_field'][str(index_for_ims_copy[i]+1)]['units']
            export_listOfScenarios[counter-1,2] = gm_dict['event']['magnitude']
            export_metadata['Event_'+str(counter)]['Metadata'] = gm_dict['event']
            
    # -----------------------------------------------------------
    # export IMs and sigmas
    for im_i in im_list:
        file_IM = os.path.join(im_dir,im_i,'Median.txt')
        # file_sigma = os.path.join(im_dir,im_i,'TotalStdDev.txt')
        file_sigma = os.path.join(im_dir,im_i,'Aleatory.txt')
        file_sigma_mu = os.path.join(im_dir,im_i,'Epistemic.txt')
        np.savetxt(file_IM, export_IM[im_i], fmt='%5.3f')
        # np.savetxt(file_sigma, export_sigma[im_i], fmt='%5.3f')
        np.savetxt(file_sigma, export_sigma[im_i], fmt='%5.3f')
        np.savetxt(file_sigma_mu, export_sigma_mu[im_i], fmt='%5.3f')
        # np.savetxt(os.path.join(im_dir,im_i,'sigma_inter.txt'), np.zeros(export_sigma[im_i].shape))
        # np.savetxt(os.path.join(im_dir,im_i,'sigma_intra.txt'), np.zeros(export_sigma[im_i].shape))
    logging.info('\t... IMs interpolated from ShakeMap grids exported to:')
    logging.info(f"\t\t{im_dir}")
    # export list of events
    df_out = pd.DataFrame.from_dict({
        "SourceIndex": export_listOfScenarios[:,0].astype(int),
        "RuptureIndex": export_listOfScenarios[:,1].astype(int),
        "Magnitude": export_listOfScenarios[:,2].astype(float),
        "MeanAnnualRate": export_listOfScenarios[:,3].astype(float),
    })
    df_out.to_csv(store_events_file, index=False)
    # np.savetxt(store_events_file, export_listOfScenarios, fmt='%i %i %6.3f %6.3e')
    # export complete event metadata
    if flag_export_metadata:
        file_metadata = os.path.join(os.path.dirname(store_events_file),'ShakeMap_MetaData.json')
        with open(file_metadata, 'w') as outfile:
            json.dump(export_metadata, outfile, indent=4, separators=(',', ': '))
    logging.info('\t... ShakeMap event metadata exported to:')
    logging.info(f"\t\t{os.path.dirname(store_events_file)}")

    # export traces
    dict_out = {
        'SourceIndex': np.arange(num_events)+1,
        'ListOfTraces': trace_set
    }
    df = pd.DataFrame.from_dict(dict_out)
    df.to_csv(trace_save_name,index=False)
    logging.info(f"\t... list of traces for rupture scenarios exported to:")
    logging.info(f"\t\t{trace_save_name}")

    #
    return None

# -----------------------------------------------------------
def read_IM_means(im_pred_dir, im_list, list_param, store_file_type='txt', n_site=None, n_event=None):
    """
    Read outputs from OpenSHAInterface wrapper
    
    Parameters
    ----------
    im_pred_dir : str
        base directory of the **GM** outputs from OpenSHAInterface
    rup_group : str
        name of folder with **rups** to import (e.g., 0-99, 100:199)
        
    Returns
    -------
    im_pred : float, dict
        dictionary containing the nonzero site indices, and **PGA** and **PGV** predictions (i.e., means, sigmas)
    rup_meta : float, dict
        dictionary containing the rupture scenario information (UCERF3 index, magnitude, mean annual rate)
    
    """
    
    # initialize dictionary
    im_pred = {}
    
    # loop through variables
    for im_i in im_list:
        curr_dir = os.path.join(im_pred_dir,im_i)
        files_in_dir = os.listdir(curr_dir)
        im_pred[im_i] = {}
        for param_j in list_param:
            curr_file = param_j+'.'+store_file_type
            if curr_file in files_in_dir:
                # load file, store into im_data, then close
                file_name = os.path.join(im_pred_dir, im_i, param_j+'.'+store_file_type)
                # load txt and convert to COO matrix
                data = np.loadtxt(file_name)
                # assume dimension from file if not provided
                # if n_event is None:
                    # n_event = data.shape[0]
                # if n_site is None:
                    # n_site = data.shape[1]
                # update to dictionary
                im_pred[im_i].update({
                    # param_j:sparse.coo_matrix((data[2]),(data[0],data[1]),shape=(n_event, n_site))
                    param_j:sparse.coo_matrix(data)
                })
            else:
                im_pred[im_i].update({param_j:None})
    
    #
    return im_pred


# -----------------------------------------------------------
def get_cov(corr_spatial_pga, corr_spatial_pgv, corr_spectral, pga_sigma, pgv_sigma, n_site, n_period=2):
    """
    Compute correlation and covariance matrices for PGA and PGV over a list of sites
    
    Parameters
    ----------
    corr_spatial_pga : float, array
        list of spatial correlations across sites for **PGA**
    corr_spatial_pgv : float, array
        list of spatial correlations across sites for **PGV**
    corr_spectral : float, array
        list of spectral correlation between PGA (T ~ 0.01sec) and PGV (T ~ 1sec)
    pga_sigma : float, array
        sigmas for **PGA** for the list of sites
    pgv_sigma : float, array
        sigmas for **PGV** for the list of sites
    n_site : int
        number of sites
    n_period : int, optional
        number of periods (fixed to 2 for now)
        
    Returns
    -------
    cov_mat_corr : float, arry
        correlated covariance matrix
    
    """
    
    # initialize storage dictionary
    cov_dict = {}
    
    # cross of spatial correlations between PGA and PGV
    corr_spatial_spectral = np.sqrt(np.multiply(corr_spatial_pga,corr_spatial_pgv))
    
    # convert arrays to symmetric matrices
    # corr_spectral_mat = convert_triu_to_sym_mat(corr_spectral,n_period) # convert spectral correlations to symmetric matrix
    # corr_spatial_pga_mat = convert_triu_to_sym_mat(corr_spatial_pga,n_site) # convert PGA spatial correlations to symmetric matrix
    # corr_spatial_pgv_mat = convert_triu_to_sym_mat(corr_spatial_pgv,n_site) # convert PGV spatial correlations to symmetric matrix
    # corr_spatial_spectral_mat = convert_triu_to_sym_mat(corr_spatial_spectral,n_site) # convert PGAxPGV correlations to symmetric matrix
    corr_spectral_mat = corr_spectral # convert spectral correlations to symmetric matrix
    corr_spatial_pga_mat = corr_spatial_pga # convert PGA spatial correlations to symmetric matrix
    corr_spatial_pgv_mat =corr_spatial_pgv # convert PGV spatial correlations to symmetric matrix
    corr_spatial_spectral_mat = corr_spatial_spectral # convert PGAxPGV correlations to symmetric matrix
    
    # full correlation matrix
    corr_quad11 = corr_spatial_pga_mat*corr_spectral_mat[0][0] # upper-left quadrant, PGA x PGA
    corr_quad12 = corr_spatial_spectral_mat*corr_spectral_mat[0][1] # upper-right quadrant, PGV x PGA
    # corr_quad21 = np.transpose(corr_quad12) # lower-left quadrant, PGA x PGV
    corr_quad22 = corr_spatial_pgv_mat*corr_spectral_mat[1][1] # lower-right quadrant, PGV x PGV
    # corr_mat = np.bmat([[corr_quad11, corr_quad12], [corr_quad21, corr_quad22]])
    # corr_mat = np.vstack([np.hstack([corr_quad11,corr_quad12]),np.hstack([corr_quad21,corr_quad22])])
    
    # joint uncorrelated covariance matrix
    cov_quad11 = np.outer(pga_sigma,pga_sigma) # upper-left quadrant, PGA x PGA
    cov_quad12 = np.outer(pga_sigma,pgv_sigma) # upper-right quadrant, PGV x PGA
    # cov_quad21 = np.transpose(cov_quad12) # lower-left quadrant, PGA x PGV
    cov_quad22 = np.outer(pgv_sigma,pgv_sigma) # lower-right quadrant, PGV x PGV
    # cov_mat_uncorr = np.bmat([[cov_quad11, cov_quad12], [cov_quad21, cov_quad22]])
    # cov_mat_uncorr = np.vstack([np.hstack([cov_quad11,cov_quad12]),np.hstack([cov_quad21,cov_quad22])])
    
    # joint correlated covariance matrix
    cov_quad11_corr = np.multiply(corr_quad11,cov_quad11)
    cov_quad12_corr = np.multiply(corr_quad12,cov_quad12)
    cov_quad22_corr = np.multiply(corr_quad22,cov_quad22)
    # cov_mat_corr = np.multiply(cov_mat_uncorr,corr_mat)
    
    # store into dictionary for return
    cov_dict = {
        'cov_quad11_corr': cov_quad11_corr,
        'cov_quad12_corr': cov_quad12_corr,
        'cov_quad22_corr': cov_quad22_corr,
    }
    
    #
    return cov_dict
    
    
# -----------------------------------------------------------
def get_RV_sims(mean, cov, nsamp, nsite=1, var_list=['PGA','PGV']):
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
		list of variables, default = ['PGA', 'PGV']
		
	Returns
	-------
	sample_dict : float, dictionary
		[g or cm/s] a dictionary of entries equal to number of variables. Each entry contains **nsamp** of realizations by **nsites**
	
	"""
	
	# total number of cases = number of periods * number of sites
	ntot = len(mean)
	
	# number of periods = total / number of sites
	nperiod = len(var_list)
	
	# generate sample from mean and cov and transpose
	l = cholesky(cov, check_finite=False, overwrite_a=True)
	mrs_out = np.transpose(np.asarray([mean + l.dot(standard_normal(len(mean))) for i in range(nsamp)]))
	
	# mrs_out = np.random.multivariate_normal(mean, cov, nsamp).T
	
	# partition into individual IMs
	sample_dict = {}
	for i in range(nperiod):
		sample_dict[var_list[i]] = np.exp(mrs_out[i*nsite:(i+1)*nsite])
	
	#
	return sample_dict