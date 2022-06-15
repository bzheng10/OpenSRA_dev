# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for transient pipe strain
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import importlib
import copy
import os

# data manipulation modules
import numpy as np
# from numpy import tan, radians, where
# from numba import jit
# from numba import njit
import pandas as pd


# def get_rvs_and_fix_by_level(rv_input, fix_input, methods_to_use, infra_class={}):
def get_rvs_and_fix_by_level(methods_to_use, infra_class={}):
    """gets all the required RVs and fixed variables sorted by level"""
    
    all_rvs = [] # list of all RVs needed for selected methods
    req_rvs_by_level = {} # RVs needed by levels
    req_fixed_by_level = {} # fixed params needed by levels

    # for each category
    for cat in methods_to_use:
        curr_cat = methods_to_use[cat]
        # for each hazard
        for haz_type in curr_cat:
            curr_haz_type = curr_cat[haz_type]
            # load python file
            _file = importlib.import_module('.'.join(['src',cat.lower(),haz_type.lower()]))
            # _file = importlib.import_module('.'.join([cat.lower(),haz_type.lower()]))
            # importlib.reload(_file)
            # for each method
            for method in curr_haz_type:
                # create instance
                _inst1 = copy.deepcopy(getattr(_file, method)())
                # get all RVs for method
                # all_rvs += _inst1._missing_inputs_all
                all_rvs += _inst1._missing_inputs_rvs
                rvs_by_level, fix_by_level = _inst1.get_req_rv_and_fix_params(infra_class)
                # rvs_by_level, _ = _inst1.get_req_rv_and_fix_params(infra_class)
                # gather required model parameters for each level
                for i in range(3):
                    # if method varies with level
                    if _inst.input_dist_vary_with_level:
                        if f'level{i+1}' in req_rvs_by_level:
                            req_rvs_by_level[f'level{i+1}'] += rvs_by_level[f'level{i+1}']
                            req_fixed_by_level[f'level{i+1}'] += fix_by_level[f'level{i+1}']
                        else:
                            req_rvs_by_level[f'level{i+1}'] = rvs_by_level[f'level{i+1}']
                            req_fixed_by_level[f'level{i+1}'] = fix_by_level[f'level{i+1}']
                    # if method doesn't with level
                    else:
                        if f'level{i+1}' in req_rvs_by_level:
                            req_rvs_by_level[f'level{i+1}'] += rvs_by_level
                            req_fixed_by_level[f'level{i+1}'] += fix_by_level
                        else:
                            req_rvs_by_level[f'level{i+1}'] = rvs_by_level
                            req_fixed_by_level[f'level{i+1}'] = fix_by_level

    # get unique required params for each level
    for i in range(3):
        req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
        req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))

    # sort all rvs
    all_rvs = sorted(list(set(all_rvs)))
    
    # return
    return all_rvs, req_rvs_by_level, req_fixed_by_level
    # return all_rvs, req_rvs_by_level


def import_param_dist_table(infra_type='below_ground'):
    """loads table with param distributions, choose from 'below_ground', 'above_ground', and 'wells_caprocks'"""
    n_levels = 3
    pref_param_dist_path = os.path.join('param_dist',f'{infra_type}.xlsx')
    pref_param_dist = {}
    for i in range(n_levels):
        curr_level = f'level{i+1}'
        pref_param_dist[curr_level] = pd.read_excel(
            pref_param_dist_path,
            sheet_name=curr_level
        )
    return pref_param_dist


def get_pref_dist_for_params(
    site_data,
    methods_to_use,
    rv_input=pd.DataFrame(None,columns=['rv_label']),
    infra_class={},
    infra_type='below_ground'
):
    """get preferred distribution metrics from internal tables"""
    
    # number of sites
    n_site = site_data.shape[0]
    
    # first get table with distributions
    pref_param_dist = import_param_dist_table(infra_type=infra_type)
    
    # determine RVs needed by level
    # all_rvs, req_rvs_by_level = get_rvs_and_fix_by_level(rv_input, methods_to_use, infra_class)
    all_rvs, req_rvs_by_level, req_fixed_by_level = get_rvs_and_fix_by_level(methods_to_use, infra_class)

    
    # list of dist metrics to get
    dist_info_to_get = ['mean','sigma','cov','low','high','dist_type']

    # first go through list of required rvs, expand dims, then see which param requires preferred distributions
    input_kwargs_df = {}

    # print(all_rvs)
    # print(req_fixed_by_level)
    
    # first read table as is
    for info in dist_info_to_get:
        input_kwargs_df[info] = pd.DataFrame(index=list(range(n_site)))
        for req_rv in all_rvs:
            if req_rv in rv_input.rv_label.values:
                row_for_rv = np.where(rv_input.rv_label.values==req_rv)[0][0]
                if rv_input[info].iloc[row_for_rv] == 'preferred':
                    input_kwargs_df[info][req_rv] = np.repeat(rv_input[info].iloc[row_for_rv],n_site)
                else:
                    if isinstance(rv_input[info].iloc[row_for_rv],str):
                        try:
                            to_num = float(rv_input[info].iloc[row_for_rv])
                            input_kwargs_df[info][req_rv] = np.repeat(to_num,n_site)
                        except:
                            input_kwargs_df[info][req_rv] = site_data[rv_input[info].iloc[row_for_rv]].values
                    else:
                        input_kwargs_df[info][req_rv] = np.repeat(rv_input[info].iloc[row_for_rv],n_site)
            else:
                input_kwargs_df[info][req_rv] = ['preferred']*n_site
    
    # for each site, determine level to run
    level_to_run = np.zeros(n_site)
    # loop through levels
    for i in range(3):
        # parameters required for current level
        params_for_curr_level = req_rvs_by_level[f'level{i+1}']
        # parameters required, see if they are preferred/null/nan
        inds = np.where(
            np.logical_or(
                input_kwargs_df['mean'][params_for_curr_level]=='preferred',
                input_kwargs_df['mean'][params_for_curr_level].isnull(),
            )
        )
        # level to run for indices wheres there are no invalid entries
        level_to_run[list(set(list(range(n_site))).difference(set(inds[0])))] = i+1
    # store levels to run
    site_data['level_to_run'] = level_to_run

    # now to get the metrics from preferred distributions
    for info in dist_info_to_get:
        # loop through each level to run
        for level in np.unique(level_to_run).astype(int):
            # current table with preferred parameter values for current level
            curr_pref_param_dist = pref_param_dist[f'level{level}']
            # get rows for current level to run
            rows_for_level = np.where(level_to_run==level)[0]
            # get indices in rows with invalid entries
            inds = np.where(
                np.logical_or(
                    input_kwargs_df[info].iloc[rows_for_level]=='preferred',
                    input_kwargs_df[info].iloc[rows_for_level].isnull(),
                )
            )
            # get list of params with invalid entries
            unique_col_inds = np.unique(inds[1])
            params_to_get = list(input_kwargs_df[info].columns[unique_col_inds])
            # print(params_to_get)

            # loop through each remaining param
            for i, param in enumerate(params_to_get):
                row_inds_for_unique_col = inds[0][inds[1]==unique_col_inds[i]]
                row_for_rv = np.where(curr_pref_param_dist.rv_label.values==param)[0][0]
                val = curr_pref_param_dist[info].iloc[row_for_rv]
                # print(level, unique_col_inds[i], param, val)
                if val == 'depends':
                    # pipe diameter
                    if param == 'd_pipe':
                        if info == 'low' or info == 'high':
                            # get welded locations
                            if 'weld_flag' in infra_class:
                                rows_welded = rows_for_level[row_inds_for_unique_col][infra_class['weld_flag'][rows_for_level[row_inds_for_unique_col]]==True]
                                rows_not_welded = list(set(rows_for_level[row_inds_for_unique_col]).difference(set(rows_welded)))
                            else:
                                rows_not_welded = rows_for_level[row_inds_for_unique_col]
                            # get d_pipe ranges
                            d_pipe_range1 = rows_for_level[row_inds_for_unique_col][input_kwargs_df['mean'][param][rows_for_level[row_inds_for_unique_col]]<=168] # <= 168 mm
                            d_pipe_range3 = rows_for_level[row_inds_for_unique_col][input_kwargs_df['mean'][param][rows_for_level[row_inds_for_unique_col]]>=610] # >= 610 mm
                            d_pipe_range2 = list(set(rows_for_level[row_inds_for_unique_col]).difference(set(d_pipe_range1).union(set(d_pipe_range3))))
                            # combinations
                            comb1 = list(set(rows_welded).intersection(set(d_pipe_range1)))
                            comb2 = list(set(rows_welded).intersection(set(d_pipe_range2)))
                            comb3 = list(set(rows_welded).intersection(set(d_pipe_range3)))
                            comb4 = list(set(rows_not_welded).intersection(set(d_pipe_range1)))
                            comb5 = list(set(rows_not_welded).intersection(set(d_pipe_range2)))
                            comb6 = list(set(rows_not_welded).intersection(set(d_pipe_range3)))
                            # get tolerances, in mm
                            tol1 = input_kwargs_df['mean'][param].loc[comb1] * 0.0075
                            tol2 = np.maximum(input_kwargs_df['mean'][param].loc[comb2] * 0.0075, 3.2)
                            tol3 = np.maximum(input_kwargs_df['mean'][param].loc[comb3] * 0.005 , 4.1)
                            tol4 = input_kwargs_df['mean'][param].loc[comb4] * 0.0075
                            tol5 = input_kwargs_df['mean'][param].loc[comb5] * 0.0075
                            tol6 = input_kwargs_df['mean'][param].loc[comb6] * 0.01
                            # for each combination
                            if info == 'low':
                                input_kwargs_df[info].loc[comb1, param] = input_kwargs_df['mean'][param].loc[comb1] - tol1
                                input_kwargs_df[info].loc[comb2, param] = input_kwargs_df['mean'][param].loc[comb2] - tol2
                                input_kwargs_df[info].loc[comb3, param] = input_kwargs_df['mean'][param].loc[comb3] - tol3
                                input_kwargs_df[info].loc[comb4, param] = input_kwargs_df['mean'][param].loc[comb4] - tol4
                                input_kwargs_df[info].loc[comb5, param] = input_kwargs_df['mean'][param].loc[comb5] - tol5
                                input_kwargs_df[info].loc[comb6, param] = input_kwargs_df['mean'][param].loc[comb6] - tol6
                            elif info == 'high':
                                input_kwargs_df[info].loc[comb1, param] = input_kwargs_df['mean'][param].loc[comb1] + tol1
                                input_kwargs_df[info].loc[comb2, param] = input_kwargs_df['mean'][param].loc[comb2] + tol2
                                input_kwargs_df[info].loc[comb3, param] = input_kwargs_df['mean'][param].loc[comb3] + tol3
                                input_kwargs_df[info].loc[comb4, param] = input_kwargs_df['mean'][param].loc[comb4] + tol4
                                input_kwargs_df[info].loc[comb5, param] = input_kwargs_df['mean'][param].loc[comb5] + tol5
                                input_kwargs_df[info].loc[comb6, param] = input_kwargs_df['mean'][param].loc[comb6] + tol6
                    # wall thickness
                    if param == 't_pipe':
                        if info == 'low' or info == 'high':
                            # get welded locations
                            if 'weld_flag' in infra_class:
                                rows_welded = rows_for_level[row_inds_for_unique_col][infra_class['weld_flag'][rows_for_level[row_inds_for_unique_col]]==True]
                                rows_not_welded = list(set(rows_for_level[row_inds_for_unique_col]).difference(set(rows_welded)))
                            else:
                                rows_not_welded = rows_for_level[row_inds_for_unique_col]
                            # get d_pipe ranges
                            t_pipe_range1 = rows_for_level[row_inds_for_unique_col][input_kwargs_df['mean'][param][rows_for_level[row_inds_for_unique_col]]<=4] # <= 4mm
                            t_pipe_range3 = rows_for_level[row_inds_for_unique_col][input_kwargs_df['mean'][param][rows_for_level[row_inds_for_unique_col]]>=24] # >= 24mm
                            t_pipe_range2 = list(set(rows_for_level[row_inds_for_unique_col]).difference(set(t_pipe_range1).union(set(t_pipe_range3))))
                            t_pipe_range4 = rows_for_level[row_inds_for_unique_col][input_kwargs_df['mean'][param][rows_for_level[row_inds_for_unique_col]]<=5] # <= 5mm
                            t_pipe_range6 = rows_for_level[row_inds_for_unique_col][input_kwargs_df['mean'][param][rows_for_level[row_inds_for_unique_col]]>=15] # >= 15mm
                            t_pipe_range5 = list(set(rows_for_level[row_inds_for_unique_col]).difference(set(t_pipe_range4).union(set(t_pipe_range6))))
                            # combinations
                            comb1 = list(set(rows_welded).intersection(set(t_pipe_range1)))
                            comb2 = list(set(rows_welded).intersection(set(t_pipe_range2)))
                            comb3 = list(set(rows_welded).intersection(set(t_pipe_range3)))
                            comb4 = list(set(rows_not_welded).intersection(set(t_pipe_range1)))
                            comb5 = list(set(rows_not_welded).intersection(set(t_pipe_range2)))
                            comb6 = list(set(rows_not_welded).intersection(set(t_pipe_range3)))
                            # get tolerances, in mm
                            tol1 = 0.5
                            tol2 = input_kwargs_df['mean'][param].loc[comb2] * 0.1
                            tol3 = 1.5
                            tol4low = -0.5
                            tol4high = 0.6
                            tol5low = input_kwargs_df['mean'][param].loc[comb5] * 0.125
                            tol5high = input_kwargs_df['mean'][param].loc[comb5] * 0.15
                            tol6low = np.maximum(input_kwargs_df['mean'][param].loc[comb6] * 0.1, 3)
                            tol6high = np.maximum(input_kwargs_df['mean'][param].loc[comb6] * 0.1, 3.7)
                            # for each combination
                            if info == 'low':
                                input_kwargs_df[info].loc[comb1, param] = input_kwargs_df['mean'][param].loc[comb1] - tol1
                                input_kwargs_df[info].loc[comb2, param] = input_kwargs_df['mean'][param].loc[comb2] - tol2
                                input_kwargs_df[info].loc[comb3, param] = input_kwargs_df['mean'][param].loc[comb3] - tol3
                                input_kwargs_df[info].loc[comb4, param] = input_kwargs_df['mean'][param].loc[comb4] - tol4low
                                input_kwargs_df[info].loc[comb5, param] = input_kwargs_df['mean'][param].loc[comb5] - tol5low
                                input_kwargs_df[info].loc[comb6, param] = input_kwargs_df['mean'][param].loc[comb6] - tol6low
                            elif info == 'high':
                                input_kwargs_df[info].loc[comb1, param] = input_kwargs_df['mean'][param].loc[comb1] + tol1
                                input_kwargs_df[info].loc[comb2, param] = input_kwargs_df['mean'][param].loc[comb2] + tol2
                                input_kwargs_df[info].loc[comb3, param] = input_kwargs_df['mean'][param].loc[comb3] + tol3
                                input_kwargs_df[info].loc[comb4, param] = input_kwargs_df['mean'][param].loc[comb4] + tol4high
                                input_kwargs_df[info].loc[comb5, param] = input_kwargs_df['mean'][param].loc[comb5] + tol5high
                                input_kwargs_df[info].loc[comb6, param] = input_kwargs_df['mean'][param].loc[comb6] + tol6high
                    # yield stress
                    if param == 'sigma_y':
                        if info == 'low' or info == 'high':
                            # get locations of various grades locations
                            if 'steel_grade' in infra_class:
                                rows_gradeb = rows_for_level[row_inds_for_unique_col][infra_class['steel_grade'][rows_for_level[row_inds_for_unique_col]]=='Grade-B']
                                rows_x42 = rows_for_level[row_inds_for_unique_col][infra_class['steel_grade'][rows_for_level[row_inds_for_unique_col]]=='X-42']
                                rows_x52 = rows_for_level[row_inds_for_unique_col][infra_class['steel_grade'][rows_for_level[row_inds_for_unique_col]]=='X-52']
                                rows_x60 = rows_for_level[row_inds_for_unique_col][infra_class['steel_grade'][rows_for_level[row_inds_for_unique_col]]=='X-60']
                                rows_x70 = rows_for_level[row_inds_for_unique_col][infra_class['steel_grade'][rows_for_level[row_inds_for_unique_col]]=='X-70']
                                rows_x80 = rows_for_level[row_inds_for_unique_col][infra_class['steel_grade'][rows_for_level[row_inds_for_unique_col]]=='X-80']
                            else:
                                rows_gradeb = []
                                rows_x42 = rows_for_level[row_inds_for_unique_col]
                                rows_x52 = []
                                rows_x60 = []
                                rows_x70 = []
                                rows_x80 = []
                            # set low, kPa
                            if info == 'low':
                                input_kwargs_df[info].loc[rows_gradeb, param] = 241000
                                input_kwargs_df[info].loc[rows_x42, param] = 290000
                                input_kwargs_df[info].loc[rows_x52, param] = 359000
                                input_kwargs_df[info].loc[rows_x60, param] = 414000
                                input_kwargs_df[info].loc[rows_x70, param] = 483000
                                input_kwargs_df[info].loc[rows_x80, param] = 552000
                            if info == 'high':
                                input_kwargs_df[info].loc[rows_for_level[row_inds_for_unique_col], param] = input_kwargs_df['mean'][param].loc[rows_for_level[row_inds_for_unique_col]] + 75000
                else:
                    input_kwargs_df[info].loc[rows_for_level[row_inds_for_unique_col], param] = val
                    
    # resolve cov into sigma
    for col in input_kwargs_df['sigma'].columns:
        # find where sigma is NaN
        check = np.where(input_kwargs_df['sigma'][col].isnull())[0]
        if len(check) > 0:
            input_kwargs_df['sigma'].loc[check, col] = \
                np.log(input_kwargs_df['mean'].loc[check, col].values.astype(float)) * \
                    input_kwargs_df['cov'].loc[check, col].values/100
    
    # convert each column to numerics if possible
    for info in input_kwargs_df:
        for col in input_kwargs_df[info].columns:
            input_kwargs_df[info][col] = pd.to_numeric(input_kwargs_df[info][col], errors='ignore')
    
    # drop cov
    input_kwargs_df.pop('cov')
    
    #
    return site_data, input_kwargs_df