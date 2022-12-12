# workflow functions used by PC

# Python base modules
import importlib
import copy
import os
import time

# scientific processing modules
import numpy as np
from pandas import DataFrame

# precompile
from numba_stats import norm
from numba import njit, float64

# OpenSRA modules and functionsx
from src.util import lhs
from src.pc_func.pc_util import res_to_samples, hermite_prob


@njit(
    float64[:,:](float64[:,:],float64[:,:]),
    fastmath=True,
    cache=True,
)
def sum_pc_terms(pc_coeffs, hermite_prob_table):
    """sum up pc terms, where pc_coeffs has dimensions of (n_site x n_pc_terms) and hermite_prob_table has dimensions of (n_pc_terms x n_samples)"""
    # get dims
    n_site = pc_coeffs.shape[0]
    n_pc_terms = pc_coeffs.shape[1]
    n_samples = hermite_prob_table.shape[1]
    # loop throuh pc terms
    pc_sum = np.zeros((n_samples,n_site))
    for i in range(n_pc_terms):
        pc_coeffs_i = pc_coeffs[:,i].repeat(n_samples).reshape((-1,n_samples)).T
        hermite_prob_table_i = hermite_prob_table[i].repeat(n_site).reshape((-1, n_site))
        pc_sum += pc_coeffs_i*hermite_prob_table_i
    # keep sum within 0 and 1
    pc_sum = np.maximum(np.minimum(pc_sum,1),0)
    # return
    return pc_sum

def sum_pc_terms_v2(pc_coeffs, hermite_prob_table):
    """sum up pc terms, where pc_coeffs has dimensions of (n_site x n_pc_terms) and hermite_prob_table has dimensions of (n_pc_terms x n_samples)"""
    # get dims
    n_site = pc_coeffs.shape[0]
    n_pc_terms = pc_coeffs.shape[1]
    n_samples = hermite_prob_table.shape[1]
    # loop through pc terms
    pc_sum = np.inner(pc_coeffs,hermite_prob_table)
    # keep sum within 0 and 1
    pc_sum = np.maximum(np.minimum(pc_sum,1),0)
    # return
    return pc_sum
    

def get_mean_kwargs(dictionary):
    """pull only the mean value from dictionary with distributions"""
    mean_dict = {}
    for param in dictionary:
        if dictionary[param]['dist_type'] == 'fixed':
            mean_dict[param] = np.tile(dictionary[param]['value'],(1,1)).T
        else:
            mean_dict[param] = np.tile(dictionary[param]['mean'],(1,1)).T
            # apply exp if lognormal
            if dictionary[param]['dist_type'] == 'lognormal':
                mean_dict[param] = np.exp(mean_dict[param])
    return mean_dict


def clean_up_input_params(input_dict):
    """add metrics if missing"""
    for each in input_dict:
        # set to normal if not given
        if not 'dist_type' in input_dict[each]:
            input_dict[each]['dist_type'] = 'normal'
        # if not fixed
        if input_dict[each]['dist_type'] != 'fixed':            
            if not 'low' in input_dict[each]:
                input_dict[each]['low'] = -np.inf*np.ones(n_site)
            if not 'high' in input_dict[each]:
                input_dict[each]['high'] = np.inf*np.ones(n_site)                
            # if sigma is 0, then set bounds to inf
            if 0 in input_dict[each]['sigma']:
                input_dict[each]['low'][input_dict[each]['sigma']==0] = -np.inf
                input_dict[each]['high'][input_dict[each]['sigma']==0] = np.inf
    # return
    return input_dict


def prepare_methods(workflow, n_site):
    """given workflow, construction dictionary with instances of methods"""
    # set up dictionary for methods and additional parameters loaded through setup_config
    mods_dict = {}
    additional_params = {}
    # order from DV to DM to EDP to IM
    cats_in_workflow_ordered = []
    for cat in ['DV','DM','EDP','IM']:
        if cat in list(workflow):
            cats_in_workflow_ordered.append(cat)
    # loop through PBEE categories
    print('processing workflow...')
    for cat in cats_in_workflow_ordered:
        # initialize dictionary
        mods_dict[cat.lower()] = {}
        # loop through hazards
        for haz in workflow[cat]:
            mods_dict[cat.lower()][haz] = {
                'module': importlib.import_module(f'src.{cat.lower()}.{haz}'),
                'method': {},
                'weight': [],
            }
            # reload for development
            importlib.reload(mods_dict[cat.lower()][haz]['module'])
            # loop through methods
            upstream_cat = []
            mods_dict[cat.lower()][haz]['upstream_params_by_method'] = {}
            mods_dict[cat.lower()][haz]['upstream_params'] = []
            mods_dict[cat.lower()][haz]['input_params_by_method'] = {}
            mods_dict[cat.lower()][haz]['input_params'] = []
            mods_dict[cat.lower()][haz]['return_params_by_method'] = {}
            mods_dict[cat.lower()][haz]['return_params'] = []
            if 'CPTBased' in workflow[cat][haz]:
                if haz == 'liquefaction':
                    for model in workflow[cat][haz]:
                        mods_dict[cat.lower()][haz]['method'][model] = "CPT_Based_Procedure"
                        mods_dict[cat.lower()][haz]['upstream_params_by_method'][model] = ['pga']
                        mods_dict[cat.lower()][haz]['return_params_by_method'][model] = ['fs_liq']
                    mods_dict[cat.lower()][haz]['upstream_params'] = ['pga']
                    mods_dict[cat.lower()][haz]['return_params'] = ['fs_liq']
                else:
                    for model in workflow[cat][haz]:
                        mods_dict[cat.lower()][haz]['method'][model] = "CPT_Based_Procedure"
                        mods_dict[cat.lower()][haz]['upstream_params_by_method'][model] = ['pga']
                        mods_dict[cat.lower()][haz]['return_params_by_method'][model] = ['pgdef']
                    mods_dict[cat.lower()][haz]['upstream_params'] = ['pga']
                    mods_dict[cat.lower()][haz]['return_params'] = ['pgdef']
                # print('\t - {cat.lower()}.{haz} detected')
            else:
                for model in workflow[cat][haz]:
                    # get instance
                    mods_dict[cat.lower()][haz]['method'][model] = copy.deepcopy(getattr(mods_dict[cat.lower()][haz]['module'], model))()
                    # set up analysis
                    mods_dict[cat.lower()][haz]['method'][model].set_analysis_size(n_site=n_site)
                    # get all upstream parameters
                    mods_dict[cat.lower()][haz]['upstream_params_by_method'][model] = \
                        list(mods_dict[cat.lower()][haz]['method'][model].input_pbee_dist['params'])
                    mods_dict[cat.lower()][haz]['upstream_params'] += \
                        list(mods_dict[cat.lower()][haz]['method'][model].input_pbee_dist['params'])
                    # get all input parameters
                    mods_dict[cat.lower()][haz]['input_params_by_method'][model] = \
                        mods_dict[cat.lower()][haz]['method'][model]._missing_inputs_all
                    mods_dict[cat.lower()][haz]['input_params'] += \
                        mods_dict[cat.lower()][haz]['method'][model]._missing_inputs_all
                    # get all return parameters
                    mods_dict[cat.lower()][haz]['return_params_by_method'][model] = \
                        list(mods_dict[cat.lower()][haz]['method'][model].return_pbee_dist['params'])
                    mods_dict[cat.lower()][haz]['return_params'] += \
                        list(mods_dict[cat.lower()][haz]['method'][model].return_pbee_dist['params'])
                    # additional parameters for model
                    if workflow[cat][haz][model] is None:
                        mods_dict[cat.lower()][haz]['weight'].append(1)
                    else:
                        items = list(workflow[cat][haz][model])
                        for item in items:
                            if item != 'ModelWeight':
                                additional_params[item] = workflow[cat][haz][model][item]
                    # get weights
                    if workflow[cat][haz][model] is None:
                        mods_dict[cat.lower()][haz]['weight'].append(1)
                    else:
                        mods_dict[cat.lower()][haz]['weight'].append(workflow[cat][haz][model]['ModelWeight'])
                    upstream_cat.append(mods_dict[cat.lower()][haz]['method'][model].input_pbee_dist['category'])
                # keep unique upstream params
                mods_dict[cat.lower()][haz]['upstream_params'] = list(set(mods_dict[cat.lower()][haz]['upstream_params']))
                # keep unique input params
                mods_dict[cat.lower()][haz]['input_params'] = list(set(mods_dict[cat.lower()][haz]['input_params']))
                # get unique upstream category
                upstream_cat = np.unique(upstream_cat)
                if len(upstream_cat) > 1:
                    raise ValueError(f'Detected multiple upstream dependencies for methods selected for {cat.lower()}.{haz}')
                elif len(upstream_cat) == 1:
                    if upstream_cat[0] == 'IM' and (
                        haz != 'settlement' and haz != 'lateral_spread'
                    ):
                        if 'pga' in mods_dict[cat.lower()][haz]['upstream_params'] or \
                        'pgv' in mods_dict[cat.lower()][haz]['upstream_params']:
                            print(f'\t- {cat.lower()}.{haz} depends on {upstream_cat[0].lower()}')
                            mods_dict[cat.lower()][haz]['upstream_category'] = upstream_cat[0]
                        else:
                            print(f'\t- {cat.lower()}.{haz} does not depend pga or pgv, do not include IM dependency')
                            mods_dict[cat.lower()][haz]['upstream_category'] = None
                    else:
                        if upstream_cat[0] is not None:
                            print(f'\t- {cat.lower()}.{haz} depends on {upstream_cat[0].lower()}')
                            mods_dict[cat.lower()][haz]['upstream_category'] = upstream_cat[0]
                elif len(upstream_cat) == 0:
                    print(f'\t- {cat.lower()}.{haz} does not have any upstream dependency')
                    mods_dict[cat.lower()][haz]['upstream_category'] = None
                # normalize weights
                mods_dict[cat.lower()][haz]['weight'] = np.divide(
                    mods_dict[cat.lower()][haz]['weight'],
                    sum(mods_dict[cat.lower()][haz]['weight'])
                )
    return mods_dict, additional_params


def get_workflow_order_list(methods_dict, infra_type='below_ground', verbose=True):
    """get PBEE workflow order list"""
    # if above-ground or well and caprocks, then used predefined path
    workflow_order_list = {}
    starting_cat = 'DV'
    count = 1
    if infra_type == 'above_ground':
        if 'wellhead_rupture' in methods_dict['dv']:
            workflow_order_list[f'case_{count}'] = {
                'cat_list': ['IM', 'EDP', 'DM', 'DV'],
                'haz_list': ['im', 'wellhead_rotation', 'wellhead_strain', 'wellhead_rupture'],
                'n_pbee_dim': 4,
            }
            count += 1
            workflow_order_list[f'case_{count}'] = {
                'cat_list': ['IM', 'EDP', 'DM', 'DV'],
                'haz_list': ['im', 'wellhead_rotation', 'wellhead_strain', 'wellhead_leakage'],
                'n_pbee_dim': 4,
            }
            count += 1
        if 'vessel_rupture' in methods_dict['dv']:
            workflow_order_list[f'case_{count}'] = {
                'cat_list': ['IM', 'DM', 'DV'],
                'haz_list': ['im', 'vessel_moment_ratio', 'vessel_rupture'],
                'n_pbee_dim': 3,
            }
            count += 1
    elif infra_type == 'wells_caprocks':
        if 'well_rupture_shear' in methods_dict['dv']:
            workflow_order_list[f'case_{count}'] = {
                'cat_list': ['EDP', 'DM', 'DV'],
                'haz_list': ['surface_fault_rupture', 'well_strain', 'well_rupture_shear'],
                'n_pbee_dim': 3,
            }
            count += 1
        if 'well_rupture_shaking' in methods_dict['dv']:
            workflow_order_list[f'case_{count}'] = {
                'cat_list': ['IM', 'DM', 'DV'],
                'haz_list': ['im', 'well_moment', 'well_rupture_shaking'],
                'n_pbee_dim': 3,
            }
            count += 1
        if 'caprock_leakage' in methods_dict['dv']:
            workflow_order_list[f'case_{count}'] = {
                # 'cat_list': ['IM', 'DV'],
                # 'haz_list': ['im', 'caprock_leakage'],
                # 'n_pbee_dim': 2,
                'cat_list': ['DV'],
                'haz_list': ['caprock_leakage'],
                'n_pbee_dim': 1,
            }
            count += 1
    elif infra_type == 'below_ground':
        # loop through to construct lists
        for haz in methods_dict[starting_cat.lower()]:
            if 'liquefaction' in haz.lower():
                pass
            else:
                if 'CPTBased' in methods_dict[starting_cat.lower()][haz]['method']:
                    upstream_cat = 'IM'
                else:
                    upstream_cat = methods_dict[starting_cat.lower()][haz]['upstream_category']
                if upstream_cat is not None:
                    # if IM
                    if upstream_cat == 'IM':
                        workflow_order_list[f'case_{count}'] = {
                            'cat_list': [starting_cat, 'IM'],
                            'haz_list': [haz, 'im'],
                            'n_pbee_dim': 2
                        }
                        count += 1
                    else:
                        # next category
                        for haz_1 in methods_dict[upstream_cat.lower()]:
                            if 'liquefaction' in haz_1.lower():
                                pass
                            else:
                                if 'CPTBased' in methods_dict[upstream_cat.lower()][haz_1]['method']:
                                    upstream_cat_1 = 'IM'
                                else:
                                    upstream_cat_1 = methods_dict[upstream_cat.lower()][haz_1]['upstream_category']
                                if upstream_cat_1 is not None:
                                    # if IM
                                    if upstream_cat_1 == 'IM':
                                        workflow_order_list[f'case_{count}'] = {
                                            'cat_list': [starting_cat, upstream_cat, 'IM'],
                                            'haz_list': [haz, haz_1, 'im'],
                                            'n_pbee_dim': 3
                                        }
                                        count += 1
                                    else:
                                        # next category
                                        for haz_2 in methods_dict[upstream_cat_1.lower()]:
                                            if 'liquefaction' in haz_2.lower():
                                                pass
                                            else:
                                                if 'CPTBased' in methods_dict[upstream_cat_1.lower()][haz_2]['method']:
                                                    upstream_cat_2 = 'IM'
                                                else:
                                                    upstream_cat_2 = methods_dict[upstream_cat_1.lower()][haz_2]['upstream_category']
                                                # has to be IM
                                                if upstream_cat_2 is not None:
                                                    workflow_order_list[f'case_{count}'] = {
                                                        'cat_list': [starting_cat, upstream_cat, upstream_cat_1, 'IM'],
                                                        'haz_list': [haz, haz_1, haz_2, 'im'],
                                                        'n_pbee_dim': 4
                                                    }
                                                    count += 1
                                                else:
                                                    workflow_order_list[f'case_{count}'] = {
                                                        'cat_list': [starting_cat, upstream_cat, upstream_cat_1],
                                                        'haz_list': [haz, haz_1, haz_2],
                                                        'n_pbee_dim': 3
                                                    }
                                                    count += 1
                                else:
                                    workflow_order_list[f'case_{count}'] = {
                                        'cat_list': [starting_cat, upstream_cat_1],
                                        'haz_list': [haz, haz_1],
                                        'n_pbee_dim': 2
                                    }
                                    count += 1
                else:
                    workflow_order_list[f'case_{count}'] = {
                        'cat_list': [starting_cat],
                        'haz_list': [haz],
                        'n_pbee_dim': 1
                    }
                    count += 1
        # reverse lists:
        for case in workflow_order_list:
            workflow_order_list[case]['cat_list'].reverse()
            workflow_order_list[case]['haz_list'].reverse()
            
    # display lists
    if verbose:
        print('list of integration cases:')
        for case in workflow_order_list:
            print(f'\t{case}:')
            print(f"\t\t- number of integrations = {workflow_order_list[case]['n_pbee_dim']}")
            print(f"\t\t- {', '.join(workflow_order_list[case]['cat_list'])}")
            print(f"\t\t- {', '.join(workflow_order_list[case]['haz_list'])}")
    # return
    return workflow_order_list


def get_samples_for_params(dist, n_sample, n_site):
    """get samples for list of parameters with distributions"""
    # setup
    samples = {}
    # loop through to find non-fixed params
    count = 0
    for each in dist:
        if dist[each]['dist_type'] != 'fixed':
            count += 1
    n_params = count
    param_names = list(dist)
    # lhs residuals
    res = lhs(
        n_site=n_site,
        n_var=n_params,
        n_samp=n_sample
    )
    # convert lhs residuals to samples
    count = 0
    for i, each in enumerate(dist):
        if dist[each]['dist_type'] != 'fixed':
            # skip param entirely if 'event_dependent' is flagged with any of the sites
            if isinstance(dist[each]['mean'][0],str) and (
                dist[each]['mean'][0] == 'event_dependent' or dist[each]['mean'][0] == 'sampling_dependent'
            ):
                pass
            else:
                # print(each, dist[each]['dist_type'])
                if dist[each]['dist_type'] == 'normal':
                    dist_type_int = 1
                elif dist[each]['dist_type'] == 'lognormal':
                    dist_type_int = 2
                elif dist[each]['dist_type'] == 'uniform':
                    dist_type_int = 3
                # get samples
                samples[each] = res_to_samples(
                    residuals=res[:,:,count],
                    mean=dist[each]['mean'],
                    sigma=dist[each]['sigma'],
                    low=dist[each]['low'],
                    high=dist[each]['high'],
                    dist_type=dist_type_int
                )
                count += 1
        else:
            samples[each] = np.tile(dist[each]['value'],(n_sample,1)).T
            
    # sys.exit()
    # return
    return samples


def process_methods_for_mean_and_sigma_of_mu(
    haz_dict, upstream_params, internal_params, input_samples=None, 
    n_sample=1, n_site=1, use_input_mean=False, input_dist=None):
    """preprocess methods to get mean of mu, sigma of mu, and sigma for inputs"""
    
    # time_start = time.time()
    
    # dictionary for storing results from each method
    haz_results_by_method={}

    # number of methods for hazard
    methods = haz_dict['method']
    weights = haz_dict['weight']
    n_methods = len(methods)
    
    # if using only mean of inputs for analysis
    if use_input_mean:
        input_samples = get_mean_kwargs(input_dist)
    
    # print(f'\t2a---1. time: {time.time()-time_start} seconds')
    # time_start = time.time()
    
    # loop through method
    for count,method in enumerate(methods):
        # get upstream params for current method
        upstream_params_for_method = {}
        for param in haz_dict['upstream_params_by_method'][method]:
            upstream_params_for_method[param] = upstream_params[param].copy()
        
        # print(f'\t2a---2a. time: {time.time()-time_start} seconds')
        # time_start = time.time()
        
        # initialize
        haz_results_by_method[method] = {}
        store_rvs = {}
        track_rvs_mean = {}
        store_sigma = {}
        store_sigma_mu = {}
        store_dist_type = {}
        return_params = list(methods[method].return_pbee_dist['params'])
        
        # run analysis
        out = methods[method]._model(
            **upstream_params_for_method,
            **internal_params,
            **input_samples
        )
        
        # if method == 'SasakiEtal2022' and 'strain_casing' in haz_dict['return_params']:
        #     if True in np.isnan(out['strain_casing']['mean']) or True in np.isnan(out['strain_tubing']['mean']):
        #         print(1)
        
        # print(f'\t2a---2b. time: {time.time()-time_start} seconds')
        # time_start = time.time()
        
        # print(f'\taa. time: {time.time()-time_start1} seconds')
        # time_start1 = time.time()
        
        # loop through and search for return var and sigma, some methods have multiple conditions
        for param in return_params:
            # see if output is a single value or an array
            if np.ndim(out[param]['mean']) == 0:
                # make ones array for results
                to_expand = True
                ones_arr = np.ones((n_site,n_sample))
            else:
                to_expand = False
                
            # get sigmas and dist_type
            store_sigma[param] = out[param]['sigma']
            store_sigma_mu[param] = out[param]['sigma_mu']
            if isinstance(out[param]['dist_type'],str):
                store_dist_type[param] = out[param]['dist_type']
            else:
                store_dist_type[param] = out[param]['dist_type'][:,0] # get site variation (same across samples)
            # repeat mat if output is a single value
            if to_expand:
                store_sigma[param] = ones_arr*store_sigma[param]
                store_sigma_mu[param] = ones_arr*store_sigma_mu[param]

            # store mean
            if isinstance(store_dist_type[param],str):
                if store_dist_type[param] == 'lognormal':
                    store_rvs[param] = np.log(out[param]['mean'])
                elif store_dist_type[param] == 'normal':
                    store_rvs[param] = out[param]['mean']
                # repeat mat if output is a single value
                if to_expand:
                    store_rvs[param] = ones_arr*store_rvs[param]
            else:
                ind_normal = np.where(store_dist_type[param]=='normal')
                store_rvs[param] = np.empty((n_site,n_sample))
                if len(ind_normal)>0:
                    store_rvs[param][ind_normal,:] = out[param]['mean'][ind_normal,:]
                ind_lognormal = np.where(store_dist_type[param]=='lognormal')
                if len(ind_lognormal)>0:
                    store_rvs[param][ind_lognormal,:] = np.log(out[param]['mean'][ind_lognormal,:])
           
            # get mean of mu
            track_rvs_mean[param] = store_rvs[param].mean(axis=1)
            
            # get average sigma over domain
            store_sigma[param] = np.sqrt(np.mean(store_sigma[param]**2,axis=1))

            # get average base sigma_mu over domain
            store_sigma_mu[param] = np.sqrt(np.mean(store_sigma_mu[param]**2,axis=1))
        
            # after getting mean of mu, get epistemic uncertainty
            track_rvs_mean_reshape = np.tile(track_rvs_mean[param].copy(),(n_sample,1)).T
            var_of_mu_input_vector_down = np.mean((store_rvs[param]-track_rvs_mean_reshape)**2,axis=1)
            store_sigma_mu[param] = np.sqrt(store_sigma_mu[param]**2 + var_of_mu_input_vector_down)
        
            # store for post-processing and checks
            haz_results_by_method[method][param] = {
                'mean_of_mu': track_rvs_mean[param],
                'sigma_of_mu': store_sigma_mu[param],
                'sigma': store_sigma[param],
                'dist_type': store_dist_type[param]
            }
            
        # print(f'\t2a---2c. time: {time.time()-time_start} seconds')
        # time_start = time.time()
            
        # print(f'\tbb. time: {time.time()-time_start1} seconds')
        # time_start1 = time.time()
    
    # print(f'\t2a---2. time: {time.time()-time_start} seconds')
    # time_start = time.time()
    
    # combine results from methods
    haz_results = {}
    # loop through return params to get mean, sigma, and sigma_mu between methods
    for param in return_params:

        # initialize hazard-level varibles for geting mean values
        mean_of_mu_vector_down = np.zeros(n_site)
        var_of_mu_down = 0
        var_down = 0
        
        # loop through methods
        for count,method in enumerate(methods):
            # accumulate to get mean values
            mean_of_mu_vector_down += haz_results_by_method[method][param]['mean_of_mu'] * weights[count]
            var_of_mu_down += haz_results_by_method[method][param]['sigma_of_mu']**2 * weights[count]
            var_down += haz_results_by_method[method][param]['sigma']**2 * weights[count]

        # get epistemic uncertainty with mean of method
        var_of_mu_btw_methods_vector_down = np.sum([
            (haz_results_by_method[method][param]['mean_of_mu']-mean_of_mu_vector_down)**2
            for each in methods
        ],axis=0)/n_methods

        # take square root
        sigma_mu_down = np.sqrt(var_of_mu_down + var_of_mu_btw_methods_vector_down)
        sigma_down = np.sqrt(var_down)
        
        # store to results
        haz_results[param] = {
            'mean_of_mu': mean_of_mu_vector_down,
            'sigma_of_mu': sigma_mu_down,
            'sigma': sigma_down,
            'dist_type': store_dist_type[param]
        }
    
    # print(f'\t2a---3. time: {time.time()-time_start} seconds')
    # time_start = time.time()
    
    # print(f'\tcc. time: {time.time()-time_start1} seconds')
    # time_start1 = time.time()
    # print('\n')

    # return
    return haz_results_by_method, haz_results


def process_methods_for_mean_and_sigma_of_mu_for_liq(
    haz_dict, upstream_params, internal_params, input_samples=None,
    n_sample=1, n_site=1, get_liq_susc=True, use_input_mean=False, input_dist=None,
    get_mean_over_samples=False):
    """preprocess methods to get mean of mu, sigma of mu, and sigma for inputs"""
    
    # dictionary for storing results from each method
    haz_results_by_method={}

    # number of methods for hazard
    methods = haz_dict['method']
    weights = haz_dict['weight']
    n_methods = len(methods)
    
    # if using only mean of inputs for analysis
    if use_input_mean:
        input_samples = get_mean_kwargs(input_dist)
    
    # also get liq_susc if available
    if get_liq_susc:
        liq_susc_val = {}

    # time_start1 = time.time()

    # loop through method
    for count,method in enumerate(methods):
        # get upstream params for current method
        upstream_params_for_method = {}
        for param in haz_dict['upstream_params_by_method'][method]:
            upstream_params_for_method[param] = upstream_params[param].copy()
        # if need liq susc
        if get_liq_susc:
            liq_susc_val[method] = None
        
        # allocate for storage and intermediate processing
        haz_results_by_method[method] = {}
        store_rvs = {}
        track_rvs_mean = {}
        store_sigma_mu = {}
        store_dist_type = {}
        return_params = list(methods[method].return_pbee_dist['params'])
                
        # run analysis
        out = methods[method]._model(
            **upstream_params_for_method,
            **internal_params,
            **input_samples
        )
        
        # print(f'\taa. time: {time.time()-time_start1} seconds')
        # time_start1 = time.time()
        
        # get liq susc
        if get_liq_susc:
            if 'liq_susc_val' in out:
                liq_susc_val[method] = out['liq_susc_val']
        
        # loop through and search for return var and sigma, some methods have multiple conditions
        for param in return_params:
            # if isinstance(out[param],dict):
            # see if output is a single value or an array
            if np.ndim(out[param]['mean']) == 0:
                # make ones array for results
                to_expand = True
                ones_arr = np.ones((n_site,n_sample))
            else:
                to_expand = False
                
            # get params
            store_sigma_mu[param] = out[param]['sigma_mu']
            store_dist_type[param] = out[param]['dist_type']
            # repeat mat if output is a single value
            if to_expand:
                store_sigma_mu[param] = ones_arr*store_sigma_mu[param]
                    
            # store mean
            if out[param]['dist_type'] == 'lognormal':
                store_rvs[param] = np.log(out[param]['mean'])
            elif out[param]['dist_type'] == 'normal':
                store_rvs[param] = out[param]['mean']
            # repeat mat if output is a single value
            if to_expand:
                store_rvs[param] = ones_arr*store_rvs[param]

            # get mean of mu
            track_rvs_mean[param] = store_rvs[param].mean(axis=1)
        
            # get average base sigma_mu over domain
            store_sigma_mu[param] = np.sqrt(np.mean(store_sigma_mu[param]**2,axis=1))
    
            # after getting mean, run loop again to get epistemic uncertainty
            track_rvs_mean_reshape = np.tile(track_rvs_mean[param].copy(),(n_sample,1)).T
            var_of_mu_input_vector_down = np.mean((store_rvs[param]-track_rvs_mean_reshape)**2,axis=1)
            sigma_of_mu_vector_method_down = np.sqrt(store_sigma_mu[param]**2 + var_of_mu_input_vector_down)

            # if tracking mean over samples
            if get_mean_over_samples:
                # store for post-processing and checks
                haz_results_by_method[method][param] = {
                    'mean_of_mu': track_rvs_mean[param],
                    'sigma_of_mu': sigma_of_mu_vector_method_down,
                    'dist_type': store_dist_type[param]
                }
            
            else:        
                # store for post-processing and checks
                haz_results_by_method[method][param] = {
                    'mean_of_mu': store_rvs[param],
                    'sigma_of_mu': sigma_of_mu_vector_method_down,
                    'dist_type': store_dist_type[param]
                }
        # print(f'\bb. time: {time.time()-time_start1} seconds')
        # time_start1 = time.time()
    
    # combine methods
    haz_results = {}
    # loop through return params to get mean, sigma, and sigma_mu between methods
    for param in return_params:

        # initialize hazard-level varibles for geting mean values
        if get_mean_over_samples:
            mean_of_mu_vector_down = np.zeros(n_site)
        else:
            mean_of_mu_vector_down = np.zeros((n_site,n_sample))
        var_of_mu_down = 0
        
        # loop through methods
        for count,method in enumerate(methods):
            # accumulate to get mean values
            mean_of_mu_vector_down += haz_results_by_method[method][param]['mean_of_mu'] * weights[count]
            var_of_mu_down += haz_results_by_method[method][param]['sigma_of_mu']**2 * weights[count]
        
        # get epistemic uncertainty with mean of method
        var_of_mu_btw_methods_vector_down = np.sum([
            (haz_results_by_method[method][param]['mean_of_mu']-mean_of_mu_vector_down)**2
            for each in methods
        ],axis=0)/n_methods
        # get mean over samples
        var_of_mu_btw_methods_vector_down = np.mean(var_of_mu_btw_methods_vector_down,axis=1)
        # take square root
        sigma_mu_down = np.sqrt(var_of_mu_down + var_of_mu_btw_methods_vector_down)
        
        # store to results
        haz_results[param] = {
            'mean_of_mu': mean_of_mu_vector_down,
            'sigma_of_mu': sigma_mu_down,
            'dist_type': store_dist_type[param]
        }
        
    # print(f'\tcc. time: {time.time()-time_start1} seconds')
    # time_start1 = time.time()
    
    # combine methods and samples of liq_susc
    if get_liq_susc:
        if get_mean_over_samples:
            liq_susc[method] = np.empty((n_site), dtype='<U10')
            liq_susc_val_mean = np.zeros(n_site)
        else:
            liq_susc[method] = np.empty((n_site,n_sample), dtype='<U10')
            liq_susc_val_mean = np.zeros((n_site,n_sample))
        # liq_susc = {}
        # liq_susc = None
        for count,method in enumerate(methods):
            # liq_susc[method] = None
            if liq_susc_val[method] is not None:
                if get_mean_over_samples:
                    liq_susc_val_mean += np.mean(liq_susc_val[method],axis=1)
                else:
                    liq_susc_val_mean += liq_susc_val[method]
                # liq_susc[method] = np.empty(liq_susc_val[method].shape, dtype='<U10')
        
        # get liq susc category
        liq_susc[liq_susc_val_mean[method]>-1.15] = 'very high'
        liq_susc[liq_susc_val_mean[method]<=-1.15] = 'high'
        liq_susc[liq_susc_val_mean[method]<=-1.95] = 'moderate'
        liq_susc[liq_susc_val_mean[method]<=-3.15] = 'low'
        liq_susc[liq_susc_val_mean[method]<=-3.20] = 'very low'
        liq_susc[liq_susc_val_mean[method]<=-38.1] = 'none'
    else:
        liq_susc = None

    # return
    return haz_results, liq_susc


# def get_fractiles(pc_samples, site_id, infra_type, fractiles=[5,16,50,84,95]):
def get_fractiles(pc_samples, fractiles=[5,16,50,84,95], n_sig_fig=None):
    """get fractiles and mean"""
    frac_return = np.vstack([
        np.percentile(pc_samples*100,fractiles,axis=0),
        np.mean(pc_samples*100,axis=0)
    ]).T
    # convert back to decimals
    frac_return = frac_return/100
    # print(frac_return)
    # round to N sig figs
    if n_sig_fig is not None:
        nonzero_frac_loc = frac_return>0
        nonzero_frac = frac_return[nonzero_frac_loc]
        decimals = n_sig_fig-np.floor(np.log10(np.abs(nonzero_frac))).astype(int)-1
        # print(nonzero_frac)
        # print(n_sig_fig-np.floor(np.log10(np.abs(nonzero_frac))).astype(int))
        frac_return[nonzero_frac_loc] = np.asarray([
            np.round(nonzero_frac[i],decimals[i])
            for i in range(len(nonzero_frac))
        ])
        # print(frac_return)
    
    # index = list(fractiles)+['mean']
    # headers = columns=['site_'+str(i+1) for i in range(pc_samples.shape[1])]
    headers = [f'{val}th' for val in list(fractiles)]+['mean']
    # index = ['site_'+str(i+1) for i in range(pc_samples.shape[1])]
    # if infra_type == 'below_ground':
    #     tag = 'segment'
    # elif infra_type == 'wells_caprocks':
    #     tag = 'well'
    # elif infra_type == 'above_ground':
    #     tag = 'component'
    # else:
    #     tag = 'site'
    # index = [f'{tag}_{each}' for each in site_id]
    # df_frac = DataFrame(frac_return,index=index,columns=headers).round(decimals=3)
    # df_frac = DataFrame(frac_return,index=index,columns=headers)
    df_frac = DataFrame(frac_return,columns=headers)
    return df_frac
    # return frac_return