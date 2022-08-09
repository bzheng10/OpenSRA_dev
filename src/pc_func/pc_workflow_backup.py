# workflow functions used by PC
import importlib
import copy
import os
import numpy as np
from pandas import DataFrame
from numba_stats import norm

from src.pc_func.pc_util import res_to_samples, hermite_prob
from src.util import lhs


def main(work_dir):
    """main script to run PC"""
    
    # define some directories
    input_dir = os.path.join(work_dir,'Input')
    im_fdir = os.path.join(work_dir,'IM')

    # import site data
    site_data = pd.read_csv(os.path.join(input_dir,'site_data.csv'))

    # import IM distributions
    im_import = {}
    for each in ['pga','pgv']:
        im_import[each] = {
            'mean_table': pd.read_csv(os.path.join(im_fdir,each.upper(),'MEAN.csv'),header=None),
            'sigma_table': pd.read_csv(os.path.join(im_fdir,each.upper(),'ALEATORY.csv'),header=None),
            'sigma_mu_table': pd.read_csv(os.path.join(im_fdir,each.upper(),'EPISTEMIC.csv'),header=None)
        }

    # import rupture information
    rupture_table = pd.read_csv(os.path.join(im_fdir,'RUPTURE_METADATA.csv'))

    # number of sites
    n_site = im_import[list(im_import)[0]]['mean_table'].shape[1]
    
    # pull workflow json file
    workflow_fpath = os.path.join(input_dir,'workflow.json')
    if os.path.exists(workflow_fpath):
        with open(workflow_fpath, 'r') as f:
            workflow = json.load(f)
    else:
        print("path to workflow does not exist (see below):")
        print(f"\t{workflow_fpath}")

    # preprocess workflow to use in PC
    methods_dict = prepare_methods(workflow, n_site)
    workflow_order_list = get_workflow_order_list(methods_dict)
    n_cases = len(workflow_order_list)
    
    # initialize some dictionaries for storage
    df_frac = {}
    last_cdf_integrated = {}
    
    # get parameters required by workflow
    cat_params = {}
    all_params = []
    for cat in methods_dict:
        cat_params[cat] = list(np.unique(np.hstack([
            methods_dict[cat.lower()][haz]['input_params']
            for haz in methods_dict[cat.lower()]
        ])))
        all_params += cat_params[cat]
    all_params = list(set(all_params))
    
    
    


def get_mean_kwargs(dictionary,num_pts=None):
    """pull only the mean value from dictionary with distributions"""
    mean_dict = {}
    for param in dictionary:
        if dictionary[param]['dist_type'] == 'fixed':
            if num_pts is None:
                mean_dict[param] = dictionary[param]['value']
            else:
                mean_dict[param] = np.tile(dictionary[param]['value'],(num_pts,1))
        else:
            if num_pts is None:
                mean_dict[param] = dictionary[param]['mean']
            else:
                mean_dict[param] = np.tile(dictionary[param]['mean'],(num_pts,1))
            # apply exp if lognormal
            if dictionary[param]['dist_type'] == 'lognormal':
                mean_dict[param] = np.exp(mean_dict[param])
    return mean_dict


def get_curr_epi_sample_and_reshape(samples_dict, i, num_pts_inte):
    """return specific epistemic sample"""
    kwargs_out = {}
    for each in samples_dict:
        kwargs_out[each] = np.tile(samples_dict[each][:,i],(num_pts_inte,1))
    return kwargs_out


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
    # set up dictionary for methods
    mods_dict = {}
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
                'weight': []
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
                    # get weights
                    if workflow[cat][haz][model] is None:
                        mods_dict[cat.lower()][haz]['weight'].append(1)
                    else:
                        mods_dict[cat.lower()][haz]['weight'].append(workflow[cat][haz][model])
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
    return mods_dict


def get_workflow_order_list(methods_dict, verbose=True):
    """get PBEE workflow order list"""
    workflow_order_list = {}
    starting_cat = 'DV'
    count = 1
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
    processed_dist = {}
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
            samples[each] = np.zeros((n_site,n_sample))
            for j in range(n_site):
                samples[each][j,:] = res_to_samples(
                    samples=res[j,:,count],
                    mean=dist[each]['mean'][j],
                    sigma=dist[each]['sigma'][j],
                    low=dist[each]['low'][j],
                    high=dist[each]['high'][j],
                    dist_type=dist[each]['dist_type']
                )
            count += 1
        else:
            samples[each] = np.tile(dist[each]['value'],(n_sample,1)).T
    # return
    return samples

def get_tangent_line(mean_of_mu_up, domain_vector_up, mean_of_mu_vector_down, sigma_mu_down):
    """get tangent lines for PC approximation"""
    # for each site, get index where domain vector exceeds mean
    index_mean_of_mu_up = [np.argmax(domain_vector_up[:,i]>=val) for i,val in enumerate(mean_of_mu_up)]
    # get intercept for downstream
    intercept_down = np.asarray([mean_of_mu_vector_down[val,i] for i,val in enumerate(index_mean_of_mu_up)])
    # get slope for downstream
    slope_down = np.asarray([
        (mean_of_mu_vector_down[val+1,i] - mean_of_mu_vector_down[val,i]) / \
        (domain_vector_up[val+1,i] - domain_vector_up[val,i]) \
        for i,val in enumerate(index_mean_of_mu_up)
    ])
    # get tangent line vector
    tangent_vector_down = slope_down * (domain_vector_up - np.expand_dims(mean_of_mu_up,axis=0)) + intercept_down
    # get sigma_mu at index
    sigma_mu_intercept_down = np.asarray([sigma_mu_down[val,i] for i,val in enumerate(index_mean_of_mu_up)])
    # return
    return index_mean_of_mu_up, intercept_down, slope_down, tangent_vector_down, sigma_mu_intercept_down


def preprocess_methods_for_mean_and_sigma_of_mu(
    haz_dict, upstream_params, internal_params, input_samples, 
    n_samples, n_site, n_pts_up):
    """preprocess methods to get mean of mu, sigma of mu, and sigma for inputs"""
    
    # dictionary for storing results from each method
    haz_method_results={}

    # number of methods for hazard
    methods = haz_dict['method']
    weights = haz_dict['weight']
    n_methods = len(methods)

    # loop through method
    for count,method in enumerate(methods):
        # get upstream params for current method
        upstream_params_for_method = {}
        for param in haz_dict['upstream_params_by_method'][method]:
            upstream_params_for_method[param] = upstream_params[param].copy()
        
        # allocate for storage and intermediate processing
        haz_method_results[method] = {}
        store_rvs = {}
        track_rvs_mean = {}
        store_sigma = {}
        store_sigma_mu = {}
        store_dist_type = {}
        return_params = list(methods[method].return_pbee_dist['params'])
        for param in return_params:
            store_sigma[param] = None
            store_sigma_mu[param] = None
            haz_method_results[method][param] = {}
            track_rvs_mean[param] = np.zeros((n_pts_up,n_site))
            store_rvs[param] = {}
            
        # loop through number of epistemic samples to get mean of mu and sigma aleatory
        for i in range(n_samples):            
            # print(i)
            out = methods[method]._model(
                **upstream_params_for_method,
                **internal_params,
                **get_curr_epi_sample_and_reshape(input_samples,i,n_pts_up)
            )
            # if i == 0:
            #     if 'Sasaki' in method:
            #         return out
            # print(out['dist_type'])
            # loop through and search for return var and sigma, some methods have multiple conditions
            for param in return_params:
                # initialize
                store_rvs[param][i] = np.zeros((n_pts_up,n_site))
                # run this for first sample and reuse for later
                if i == 0:
                    # see if output is a single value or an array
                    if np.ndim(out[param]['mean']) == 0:
                        # make ones array for results
                        to_expand = True
                        if n_pts_up is None:
                            ones_arr = np.ones(n_site)
                        else:
                            ones_arr = np.ones((n_pts_up,n_site))
                    else:
                        to_expand = False
                    # get sigmas and dist_type
                    store_sigma[param] = out[param]['sigma']
                    store_sigma_mu[param] = out[param]['sigma_mu']
                    if isinstance(out[param]['dist_type'],str):
                        store_dist_type[param] = out[param]['dist_type']
                    else:
                        store_dist_type[param] = out[param]['dist_type'][0,:] # same across domain
                    # repeat mat if output is a single value
                    if to_expand:
                        store_sigma[param] = ones_arr*store_sigma[param]
                        store_sigma_mu[param] = ones_arr*store_sigma_mu[param]
                # store mean
                if isinstance(store_dist_type[param],str):
                    if out[param]['dist_type'] == 'lognormal':
                        store_rvs[param][i] = np.log(out[param]['mean'])
                        # if True in np.isnan(np.log(out[param]['mean'])):
                        #     print(method)
                        #     sys.exit()
                        # track_rvs_mean[param] += np.log(out[param]['mean'])
                    elif out[param]['dist_type'] == 'normal':
                        store_rvs[param][i] = out[param]['mean']
                        # track_rvs_mean[param] += out[param]['mean']
                    # repeat mat if output is a single value
                    if to_expand:
                        store_rvs[param][i] = ones_arr*store_rvs[param][i]
                    # track sum of mean
                    # track_rvs_mean[param] += store_rvs[param][i]
                else:
                    ind_normal = np.where(store_dist_type[param]=='normal')[0]
                    # print(out[param]['mean'].shape, store_rvs[param][i].shape)
                    if len(ind_normal)>0:
                        store_rvs[param][i][:,ind_normal] = out[param]['mean'][:,ind_normal]
                    # store_rvs[param][i] = out[param]['mean']
                    # track_rvs_mean[param] += out[param]['mean']
                    ind_lognormal = np.where(store_dist_type[param]=='lognormal')[0]
                    if len(ind_lognormal)>0:
                        store_rvs[param][i][:,ind_lognormal] = np.log(out[param]['mean'][:,ind_lognormal])
                # track sum of mean
                track_rvs_mean[param] += store_rvs[param][i]
                # if 'Sasaki' in method and len(np.where(track_rvs_mean[param]<0)[0]) > 0:
                #     print(i, param)
                #     # print(upstream_params_for_method)
                #     print(store_rvs[param][i])
                #     # print(get_curr_epi_sample_and_reshape(input_samples,i,n_pts_up))
                #     return out, upstream_params_for_method, store_rvs[param][i], get_curr_epi_sample_and_reshape(input_samples,i,n_pts_up)
                #     # return out, upstream_params_for_method, store_rvs[param][i], get_curr_epi_sample_and_reshape(input_samples,i,n_pts_up)
        
        for param in return_params:
            # get mean of mu
            track_rvs_mean[param] = track_rvs_mean[param]/n_samples
        
            # get average sigma over domain
            store_sigma[param] = np.sqrt(np.mean(store_sigma[param]**2,axis=0))
        
            # after getting mean, run loop again to get epistemic uncertainty
            var_of_mu_input_vector_down = np.sum([
                (store_rvs[param][i]-track_rvs_mean[param])**2
                for i in range(n_samples)
            ],axis=0)/n_samples
            # sigma_of_mu_vector_method_down = np.sqrt(sigma_of_mu_base_down**2 + var_of_mu_input_vector_down)
            sigma_of_mu_vector_method_down = np.sqrt(store_sigma_mu[param]**2 + var_of_mu_input_vector_down)
        
            # store for post-processing and checks
            haz_method_results[method][param] = {
                'mean_of_mu': track_rvs_mean[param],
                'sigma_of_mu': sigma_of_mu_vector_method_down,
                'sigma': store_sigma[param],
                'dist_type': store_dist_type[param]
            }
    
    # combine methods
    haz_results = {}
    # loop through return params to get mean, sigma, and sigma_mu between methods
    for param in return_params:

        # initialize hazard-level varibles for geting mean values
        mean_of_mu_vector_down = np.zeros((n_pts_up,n_site))
        var_of_mu_down = 0
        var_down = 0
        
        # loop through methods
        for count,method in enumerate(methods):
            # accumulate to get mean values
            mean_of_mu_vector_down += haz_method_results[method][param]['mean_of_mu'] * weights[count]
            var_of_mu_down += haz_method_results[method][param]['sigma_of_mu']**2 * weights[count]
            var_down += haz_method_results[method][param]['sigma']**2 * weights[count]

        # get epistemic uncertainty with mean of method
        var_of_mu_btw_methods_vector_down = np.sum([
            (haz_method_results[method][param]['mean_of_mu']-mean_of_mu_vector_down)**2
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

    # return
    return haz_method_results, haz_results


def preprocess_methods_for_mean_of_mu_using_mean_inputs(
    haz_dict, upstream_params, internal_params, input_dist, n_site, n_pts_up=None):
    """preprocess methods to get mean of mu using mean inputs"""
    
    # dictionary for storing results from each method
    haz_method_results={}
    haz_dist_type = {}
    
    # number of methods for hazard
    methods = haz_dict['method']
    weights = haz_dict['weight']
    n_methods = len(methods)

    # loop through method
    for count,method in enumerate(methods):
        # get upstream params for current method
        upstream_params_for_method = {}
        for param in haz_dict['upstream_params_by_method'][method]:
            upstream_params_for_method[param] = upstream_params[param].copy()
        
        # allocate for storage and intermediate processing
        haz_method_results[method] = {}
        return_params = list(methods[method].return_pbee_dist['params'])
        
        # run model with mean inputs
        out = methods[method]._model(
            **internal_params,
            **upstream_params_for_method,
            **get_mean_kwargs(input_dist,num_pts=n_pts_up)
        )
        # store to dict
        for param in return_params:
            # see if output is a single value or an array
            if np.ndim(out[param]['mean']) == 0:
                # make ones array for results
                to_expand = True
                if n_pts_up is None:
                    ones_arr = np.ones(n_site)
                else:
                    ones_arr = np.ones((n_pts_up,n_site))
            else:
                to_expand = False
            # store dist type
            # if isinstance(out[param]['dist_type'],str):
            haz_dist_type[param] = out[param]['dist_type']
            # else:
            # haz_dist_type[param] = out[param]['dist_type'][0,:] # same across domain
            # store mean
            if isinstance(haz_dist_type[param],str):
                if out[param]['dist_type'] == 'lognormal':
                    haz_method_results[method][param] = np.log(out[param]['mean'])
                elif out[param]['dist_type'] == 'normal':
                    haz_method_results[method][param] = out[param]['mean']
                # repeat mat if output is a single value
                if to_expand:
                    haz_method_results[method][param] = ones_arr*haz_method_results[method][param]
                haz_dist_type[param] = out[param]['dist_type']
            else:
                haz_method_results[method][param] = out[param]['mean']
                ind_lognormal = np.where(haz_dist_type[param]=='lognormal')
                haz_method_results[method][param][ind_lognormal] = np.log(out[param]['mean'][ind_lognormal])
    
    # combine methods
    haz_results = {}
    # loop through return params to get mean between methods
    for param in return_params:

        # initialize hazard-level varibles for geting mean values
        if n_pts_up is None:
            mean_of_mu_vector_down = np.zeros(n_site)
        else:
            mean_of_mu_vector_down = np.zeros((n_pts_up,n_site))

        # loop through methods
        for count,method in enumerate(methods):
            # accumulate to get mean values
            mean_of_mu_vector_down += haz_method_results[method][param] * weights[count]

        # store to results
        haz_results[param] = mean_of_mu_vector_down

    # return
    return haz_results, haz_dist_type


def preprocess_methods_for_mean_and_sigma_of_mu_for_liq(
    haz_dict, upstream_params, internal_params, input_samples,
    n_samples, n_site, n_pts_up, get_liq_susc=True):
    """preprocess methods to get mean of mu, sigma of mu, and sigma for inputs"""
    
    # dictionary for storing results from each method
    haz_method_results={}

    # number of methods for hazard
    methods = haz_dict['method']
    weights = haz_dict['weight']
    n_methods = len(methods)
    
    # also get liq_susc if available
    if get_liq_susc:
        liq_susc_val = {}

    # loop through method
    for count,method in enumerate(methods):
        # get upstream params for current method
        upstream_params_for_method = {}
        for param in haz_dict['upstream_params_by_method'][method]:
            upstream_params_for_method[param] = upstream_params[param].copy()
        
        if get_liq_susc:
            liq_susc_val[method] = None
        
        # allocate for storage and intermediate processing
        haz_method_results[method] = {}
        store_rvs = {}
        track_rvs_mean = {}
        store_sigma_mu = {}
        store_dist_type = {}
        return_params = list(methods[method].return_pbee_dist['params'])
        # for param in return_params:
        #     store_sigma_mu[param] = None
        #     haz_method_results[method][param] = {}
        #     track_rvs_mean[param] = np.zeros((n_pts_up,n_site))
        #     store_rvs[param] = {}
        
        # loop through number of epistemic samples to get mean of mu and sigma aleatory
        for i in range(n_samples):
            out = methods[method]._model(
                **internal_params,
                **upstream_params_for_method,
                **get_curr_epi_sample_and_reshape(input_samples,i,n_pts_up)
            )
            # loop through and search for return var and sigma, some methods have multiple conditions
            for param in return_params:
                if isinstance(out[param],dict):
                    # run this for first sample and reuse for later
                    if i == 0:
                        # see if output is a single value or an array
                        if np.ndim(out[param]['mean']) == 0:
                            # make ones array for results
                            to_expand = True
                            if n_pts_up is None:
                                ones_arr = np.ones(n_site)
                            else:
                                ones_arr = np.ones((n_pts_up,n_site))
                        else:
                            to_expand = False
                        # initialize
                        haz_method_results[method][param] = {}
                        track_rvs_mean[param] = np.zeros((n_pts_up,n_site))
                        store_rvs[param] = {}
                        # get params
                        store_sigma_mu[param] = out[param]['sigma_mu']
                        store_dist_type[param] = out[param]['dist_type']
                        # repeat mat if output is a single value
                        if to_expand:
                            store_sigma_mu[param] = ones_arr*store_sigma_mu[param]
                    # store mean
                    if out[param]['dist_type'] == 'lognormal':
                        store_rvs[param][i] = np.log(out[param]['mean'])
                        # if True in np.isnan(np.log(out[param]['mean'])):
                        #     print(method)
                        #     sys.exit()
                        store_rvs[param][i] = np.log(out[param]['mean'])
                        track_rvs_mean[param] += np.log(out[param]['mean'])
                    elif out[param]['dist_type'] == 'normal':
                        store_rvs[param][i] = out[param]['mean']
                        track_rvs_mean[param] += out[param]['mean']
                    # repeat mat if output is a single value
                    if to_expand:
                        store_rvs[param][i] = ones_arr*store_rvs[param][i]
                        track_rvs_mean[param] = ones_arr*track_rvs_mean[param]
                    
            # get liq susc
            if get_liq_susc:
                if 'liq_susc_val' in out:
                    if i == 0:
                        liq_susc_val[method] = np.zeros((n_samples, n_pts_up, n_site))
                    liq_susc_val[method][i,:,:] = out['liq_susc_val']
        
        # for param in return_params:
        for param in track_rvs_mean:
            # get mean of mu
            track_rvs_mean[param] = track_rvs_mean[param]/n_samples
        
            # after getting mean, run loop again to get epistemic uncertainty
            var_of_mu_input_vector_down = np.sum([
                (store_rvs[param][i]-track_rvs_mean[param])**2
                for i in range(n_samples)
            ],axis=0)/n_samples
            # sigma_of_mu_vector_method_down = np.sqrt(sigma_of_mu_base_down**2 + var_of_mu_input_vector_down)
            sigma_of_mu_vector_method_down = np.sqrt(store_sigma_mu[param]**2 + var_of_mu_input_vector_down)
        
            # store for post-processing and checks
            haz_method_results[method][param] = {
                'mean_of_mu': track_rvs_mean[param],
                'sigma_of_mu': sigma_of_mu_vector_method_down,
                'dist_type': store_dist_type[param]
            }
    
    # combine methods
    haz_results = {}
    # loop through return params to get mean, sigma, and sigma_mu between methods
    # for param in return_params:
    for param in track_rvs_mean:

        # initialize hazard-level varibles for geting mean values
        mean_of_mu_vector_down = np.zeros((n_pts_up,n_site))
        var_of_mu_down = 0
        
        # loop through methods
        for count,method in enumerate(methods):
            # accumulate to get mean values
            mean_of_mu_vector_down += haz_method_results[method][param]['mean_of_mu'] * weights[count]
            var_of_mu_down += haz_method_results[method][param]['sigma_of_mu']**2 * weights[count]

        # get epistemic uncertainty with mean of method
        var_of_mu_btw_methods_vector_down = np.sum([
            (haz_method_results[method][param]['mean_of_mu']-mean_of_mu_vector_down)**2
            for each in methods
        ],axis=0)/n_methods

        # take square root
        sigma_mu_down = np.sqrt(var_of_mu_down + var_of_mu_btw_methods_vector_down)
        
        # store to results
        haz_results[param] = {
            'mean_of_mu': mean_of_mu_vector_down,
            'sigma_of_mu': sigma_mu_down,
            'dist_type': store_dist_type[param]
        }
        
    # combine samples of liq_susc
    if get_liq_susc:
        liq_susc = {}
        # liq_susc = None
        for count,method in enumerate(methods):
            liq_susc[method] = None
            if liq_susc_val[method] is not None:
                liq_susc_val[method] = np.mean(liq_susc_val[method],axis=0)
                liq_susc[method] = np.empty(liq_susc_val[method].shape, dtype='<U10')
        
            # get liq susc category
            liq_susc[method][liq_susc_val[method]>-1.15] = 'very high'
            liq_susc[method][liq_susc_val[method]<=-1.15] = 'high'
            liq_susc[method][liq_susc_val[method]<=-1.95] = 'moderate'
            liq_susc[method][liq_susc_val[method]<=-3.15] = 'low'
            liq_susc[method][liq_susc_val[method]<=-3.20] = 'very low'
            liq_susc[method][liq_susc_val[method]<=-38.1] = 'none'
    else:
        liq_susc = None

    # return
    return haz_results, liq_susc


def preprocess_methods_for_mean_of_mu_using_mean_inputs_for_liq(
    haz_dict, upstream_params, internal_params, input_dist, n_site, n_pts_up=None,
    get_liq_susc=True):
    """preprocess methods to get mean of mu using mean inputs"""
    
    # dictionary for storing results from each method
    haz_method_results={}
    haz_dist_type = {}

    # number of methods for hazard
    methods = haz_dict['method']
    weights = haz_dict['weight']
    n_methods = len(methods)

    # also get liq_susc if available
    if get_liq_susc:
        liq_susc_val = {}

    # loop through method
    for count,method in enumerate(methods):
        # get upstream params for current method
        upstream_params_for_method = {}
        for param in haz_dict['upstream_params_by_method'][method]:
            upstream_params_for_method[param] = upstream_params[param].copy()
        
        if get_liq_susc:
            liq_susc_val[method] = None
        
        # allocate for storage and intermediate processing
        haz_method_results[method] = {}
        return_params = list(methods[method].return_pbee_dist['params'])
        
        # run model with mean inputs
        out = methods[method]._model(
            **internal_params,
            **upstream_params_for_method,
            **get_mean_kwargs(input_dist,num_pts=n_pts_up)
        )
        # store to dict
        for param in return_params:
            if isinstance(out[param],dict):
                # see if output is a single value or an array
                if np.ndim(out[param]['mean']) == 0:
                    # make ones array for results
                    to_expand = True
                    if n_pts_up is None:
                        ones_arr = np.ones(n_site)
                    else:
                        ones_arr = np.ones((n_pts_up,n_site))
                else:
                    to_expand = False
                # store mean
                if out[param]['dist_type'] == 'lognormal':
                    haz_method_results[method][param] = np.log(out[param]['mean'])
                elif out[param]['dist_type'] == 'normal':
                    haz_method_results[method][param] = out[param]['mean']
                # repeat mat if output is a single value
                if to_expand:
                    haz_method_results[method][param] = ones_arr*haz_method_results[method][param]
                # store dist type
                haz_dist_type[param] = out[param]['dist_type']
                    
        # get liq susc
        if get_liq_susc:
            if 'liq_susc_val' in out:
                liq_susc_val[method] = out['liq_susc_val']
    
    # combine methods
    haz_results = {}
    # loop through return params to get mean between methods
    # for param in return_params:
    for param in haz_method_results[method]:

        # initialize hazard-level varibles for geting mean values
        if n_pts_up is None:
            mean_of_mu_vector_down = np.zeros(n_site)
        else:
            mean_of_mu_vector_down = np.zeros((n_pts_up,n_site))

        # loop through methods
        for count,method in enumerate(methods):
            # accumulate to get mean values
            mean_of_mu_vector_down += haz_method_results[method][param] * weights[count]

        # store to results
        haz_results[param] = mean_of_mu_vector_down

    # process liq susc
    if get_liq_susc:
        liq_susc = {}
        for method in methods:
            liq_susc[method] = None
            if liq_susc_val[method] is not None:
                liq_susc[method] = np.empty(liq_susc_val[method].shape, dtype='<U10')
        
            # get liq susc category
            liq_susc[method][liq_susc_val[method]>-1.15] = 'very high'
            liq_susc[method][liq_susc_val[method]<=-1.15] = 'high'
            liq_susc[method][liq_susc_val[method]<=-1.95] = 'moderate'
            liq_susc[method][liq_susc_val[method]<=-3.15] = 'low'
            liq_susc[method][liq_susc_val[method]<=-3.20] = 'very low'
            liq_susc[method][liq_susc_val[method]<=-38.1] = 'none'
    else:
        liq_susc = None
        
    # return
    return haz_results, liq_susc, haz_dist_type


def pc_integral_for_fragility_func(
    mean_of_mu_down, sigma_of_mu_down, sigma_down,
    domain_vector_up, num_epi_samples
):
    """hard calc pc single integral"""
    # get dims
    n_site = len(mean_of_mu_down)
    n_domain = len(domain_vector_up)
    # n_domain = len(domain_vector_up)
    
    # match dimensions
    mean_of_mu_down = np.tile(mean_of_mu_down,(n_domain,1))
    # sigma_of_mu_down = np.tile(sigma_of_mu_down,(num_pts_dm,1))
    sigma_down = np.tile(sigma_down,(n_domain,1))
    # domain_vector_up = np.tile(domain_vector_up,(n_site,1)).T
    
    # const
    a = -sigma_of_mu_down**2/(2*sigma_down**2) - 1/2
    # b =  (-np.log(domain_vector_up) - mean_of_mu_down) * sigma_of_mu_down/(sigma_down**2)
    # c = - (-np.log(domain_vector_up) - mean_of_mu_down)**2/(2*sigma_down**2)
    b =  (np.log(domain_vector_up) - mean_of_mu_down) * sigma_of_mu_down/(sigma_down**2)
    c = - (np.log(domain_vector_up) - mean_of_mu_down)**2/(2*sigma_down**2)
    alpha = sigma_of_mu_down/(sigma_down*2*np.sqrt(np.pi)) * np.exp(c - b**2/(4*a))

    # pc terms 4th order
    # pc_term_0 = 1/1 * (1 - norm.cdf((-np.log(domain_vector_up) - mean_of_mu_down)/np.sqrt(sigma_of_mu_down**2 + sigma_down**2)))
    pc_term_0 = 1/1 * (1-norm.cdf(
        (np.log(domain_vector_up) - mean_of_mu_down)/np.sqrt(sigma_of_mu_down**2 + sigma_down**2)
        ,loc=0,scale=1))
    pc_term_1 = 1/1 * (1/np.sqrt(-a)) * alpha
    pc_term_2 = 1/2 * alpha * b/(2*(-a)**(3/2))
    pc_term_3 = 1/6 * alpha * (-2*a*(1 + 2*a) + b**2)/(4*(-a)**(5/2))
    pc_term_4 = 1/24 * alpha * (-b)*(6*a*(1 + 2*a) - b**2)/(8*(-a)**(7/2))
    # allocate space
    cdf_down_pc_samples = np.zeros((num_epi_samples,n_domain,n_site))

    # get epistemic samples
    epi_samples = np.random.normal(size=(1,num_epi_samples))
    
    # get hermite_prob and reshape, 4th order
    hermite_prob_1 = hermite_prob(epi_samples, 1)
    hermite_prob_2 = hermite_prob(epi_samples, 2)
    hermite_prob_3 = hermite_prob(epi_samples, 3)
    hermite_prob_4 = hermite_prob(epi_samples, 4)
    hermite_prob_1 = np.tile(hermite_prob_1,(n_site,1)).T
    hermite_prob_2 = np.tile(hermite_prob_2,(n_site,1)).T
    hermite_prob_3 = np.tile(hermite_prob_3,(n_site,1)).T
    hermite_prob_4 = np.tile(hermite_prob_4,(n_site,1)).T

    # loop through each domain value
    for i in range(domain_vector_up.shape[0]):
        # for shape
        pc_term_0_i = np.tile(pc_term_0[i,:],(num_epi_samples,1))
        pc_term_1_i = np.tile(pc_term_1[i,:],(num_epi_samples,1))
        pc_term_2_i = np.tile(pc_term_2[i,:],(num_epi_samples,1))
        pc_term_3_i = np.tile(pc_term_3[i,:],(num_epi_samples,1))
        pc_term_4_i = np.tile(pc_term_4[i,:],(num_epi_samples,1))
        #
        cdf_down_pc_samples[:,i,:] = \
            pc_term_0_i + \
            pc_term_1_i * hermite_prob_1 + \
            pc_term_2_i * hermite_prob_2 + \
            pc_term_3_i * hermite_prob_3 + \
            pc_term_4_i * hermite_prob_4

    # return
    return cdf_down_pc_samples


def get_fractiles(pc_samples, fractiles=[5,16,50,84,95]):
    """get fractiles and mean"""
    frac_return = np.vstack([
        np.percentile(pc_samples*100,fractiles,axis=0),
        np.mean(pc_samples*100,axis=0)
    ]).T
    # convert back to decimals
    frac_return = frac_return/100
    # index = list(fractiles)+['mean']
    # headers = columns=['site_'+str(i+1) for i in range(pc_samples.shape[1])]
    headers = [f'{val}th' for val in list(fractiles)]+['mean']
    index = ['site_'+str(i+1) for i in range(pc_samples.shape[1])]
    df_frac = DataFrame(frac_return,index=index,columns=headers).round(decimals=3)
    return df_frac
    # return frac_return