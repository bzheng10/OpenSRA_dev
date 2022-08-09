# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Base classes used in OpenSRA
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
import sys

# data manipulation modules
import numpy as np
from numpy.testing import assert_allclose, assert_string_equal, assert_array_equal
import pandas as pd
from scipy.stats import truncnorm

# a collection of name variants for random variables used in existing models
# rv_name_variants = {
#     'pga': ['pga', 'PGA'],
#     'pgdef': ['pgd', 'PGD', 'PGDef', 'PGDEF'],
#     'vs30': ['vs30', 'Vs30'],
#     'T': ['T', 'period'],
# }

# OpenSRA modules and classes
from src.util import get_basename_without_extension


# -----------------------------------------------------------
# class BaseClass(object):
class BaseModel(object):
    """Model template"""

    # class definitions
    _NAME = None  # Name of the model
    _ABBREV = None            # Abbreviated name of the model
    _REF = None                    # Reference for the model
    # _RETURN_PBEE_META = {
    #     'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'landslide',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'pgdef'
    #     ] # Return variable for PBEE category, e.g., pgdef, eps_p
    # }
    _RETURN_PBEE_DIST = {                            # Distribution information
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            # 'pgdef': {
            #     'desc': 'permanent ground deformation',
            #     'unit': 'm',
            #     # 'mean': None,
            #     # 'aleatory': None,
            #     # 'epistemic': {
            #     #     'coeff': None, # base uncertainty, based on coeffcients
            #     #     'input': None, # sigma_mu uncertainty from input parameters
            #     #     'total': None # SRSS of coeff and input sigma_mu uncertainty
            #     # },
            #     # 'dist_type': 'lognormal',
            # }
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': 'pga'        # Input variable for PBEE category, e.g., pgdef, eps_p
    # }
    _INPUT_PBEE_DIST = {
        "desc": 'PBEE upstream random variables:', # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, eps_p
        'params': {
            # 'pga': {
            #     'desc': 'peak ground acceleration (g)',
            #     'unit': 'g',
            #     # 'mean': None,
            #     # 'aleatory': None,
            #     # 'epistemic': None,
            #     # 'dist_type': 'lognormal'
            # }
        }
    }
    _MODEL_INPUT_INFRA = {'desc': None, 'params': None}
    _MODEL_INPUT_GEO = {'desc': None, 'params': None}
    _MODEL_INPUT_OTHER = {'desc': None, 'params': None}
    _MODEL_INPUT_FIXED = {'desc': None, 'params': None}
    # random inputs with means, sigmas, and distributions (normal or lognormal)
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 1
    _MODEL_INPUT_RV = {
        'rv1': {'mean':None, 'sigma': None, 'cov': None, 'min': -np.inf, 'max': np.inf, 'dist': 'lognormal', 'unit': None, 'desc': 'text'},
        'rv2': {'mean':None, 'sigma': None, 'cov': None, 'min': -np.inf, 'max': np.inf, 'dist': 'normal',    'unit': None, 'desc': 'text'},
        'rv3': {'mean':None, 'sigma': None, 'cov': None, 'min': -np.inf, 'max': np.inf, 'dist': 'normal',    'unit': None, 'desc': 'text'},
    }
    # Model inputs   (random and fixed variables)
    # List of terms, required information include:
    # 1. coefficient mean value
    # 2. coefficient epistemic uncertainty
    # 3. reference to random variable name under MODEL_RV
    # 4. apply natural log (no, the same RV cannot subject to a mix of ln and linear scale in the same model)
    # 5. raise term by power
    # Note that terms in higher level is expected to work in deeper levels (e.g., level 1 terms will show up in level 3 analysis)
    _MODEL_FORM_DETAIL = {
        'level1': {
            'term0': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': None, 'apply_ln': True, 'power': 1} # e.g. constant
            },
            'term1': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': 'pga', 'apply_ln': True, 'power': 1}
            },
            'term2': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': 'pga', 'apply_ln': True, 'power': 2}
            },
        },
        'level2': {
            'term3': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': 'rv1', 'apply_ln': False, 'power': 1}
            },
            'term4': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': 'rv1', 'apply_ln': False, 'power': 2}
            },
        },
        'level3': {
            'term5': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': 'rv2', 'apply_ln': True, 'power': 2}
            },
            'term6': {
                'coeff': {'mean': 1, 'sigma': 0},
                'var': {'label': 'rv3', 'apply_ln': False, 'power': 2}
            },
        },
    }
    _SUB_CLASS = None
    #  # List of other internal random variables
    # _MODEL_INTERAL_RV = {
    #     # 'return_param': ['all'], # "List of output parameters to return; default = ['all']"
    #     'int_rv1': {'mean': None, 'sigma': None, 'cov': None, 'min': None, 'max': None, 'dist': 'lognormal', 'unit': None},
    #     'int_rv2': {'mean': None, 'sigma': None, 'cov': None, 'min': None, 'max': None, 'dist': 'lognormal', 'unit': None},
    #     'int_rv3': {'mean': None, 'sigma': None, 'cov': None, 'min': None, 'max': None, 'dist': 'lognormal', 'unit': None},
    #     # 'opt1': 0,
    #     # 'opt2': False,
    #     # 'opt3': 'string'
    # }


    # instantiation
    def __init__(self):
        """Create an instance of the class"""        
        # initialize instance variables
        self._set_instance_var()
        
        # gather required inputs
        self._gather_all_inputs()
        # self._gather_all_rvs()
        # self._gather_all_fixed()
        # if self.input_dist_vary_with_level:
            # self._gather_inputs_by_level()
        
        # other instance variables
        self.model_form = None
        self.model_form_lambda = None
        
        # clear instance dictionaries
        self._inputs = {}
        self._internals = {}
        self._intermediates = {}
        self._outputs = {}


    def _set_instance_var(self):
        """Store class variables to instance"""
        class_var_to_set = [
            attr for attr in dir(self) \
                if attr.startswith("_") and \
                not attr.startswith("__") and \
                attr[1].isupper()
        ]
        # print(dir(self))
        for var in class_var_to_set:
            if isinstance(getattr(self, var),dict):
                setattr(self, var.lower()[1:], getattr(self, var).copy())
            else:
                setattr(self, var.lower()[1:], getattr(self, var))
    

    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """get required rv and fixed params"""
        req_rvs_by_level = {}
        req_fixed_by_level = {}
        for i in range(3):
            if f'level{i+1}' in req_rvs_by_level:
                if cls._INPUT_DIST_VARY_WITH_LEVEL:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL[f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL[f'level{i+1}']
                else:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL
            else:
                if cls._INPUT_DIST_VARY_WITH_LEVEL:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL[f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL[f'level{i+1}']
                else:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL
            req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
            req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))
        return req_rvs_by_level, req_fixed_by_level
    
    
    def _gather_all_inputs(self):
        """Gathers all required inputs"""
        # self._missing_inputs_all = []
        # first get RVs
        self._missing_inputs_rvs = []
        for each in ['infra','geo','other']:
        # for each in ['infra','geo','other','fixed']:
            if hasattr(self,f'model_input_{each}'):
                if getattr(self,f'model_input_{each}')['params'] is None:
                    setattr(self,f'_missing_inputs_{each}',[])
                else:
                    setattr(self,f'_missing_inputs_{each}',list(getattr(self,f'model_input_{each}')['params']))
                # self._missing_inputs_all += getattr(self,f'_missing_inputs_{each}')
                self._missing_inputs_rvs += getattr(self,f'_missing_inputs_{each}')
        # next get fixed
        if self.model_input_fixed['params'] is None:
            self._missing_inputs_fixed = []
        else:
            self._missing_inputs_fixed = list(self.model_input_fixed['params'])
        # make a full list
        self._missing_inputs_all = self._missing_inputs_rvs + self._missing_inputs_fixed
    
    # def _gather_all_fixed(self):
    #     """Gathers all required inputs"""
    #     # self._missing_inputs_fixed = []
    #     # if hasattr(self,'model_input_fixed'):
    #     if self.model_input_fixed['params'] is None:
    #         self._missing_inputs_fixed = []
    #     else:
    #         self._missing_inputs_fixed = self.model_input_fixed['params']
    #         # setattr(self,f'_missing_inputs_fixed',list(getattr(self,f'model_input_fixed')['params']))
    #         # self._missing_inputs_fixed += getattr(self,f'_missing_inputs_{each}')
    
    
    # def _gather_inputs_by_level(self):
    #     """Gathers needed inputs by levels"""
    #     self._req_inputs_by_level = {}
    #     for i in range(self.n_level):
    #         self._req_inputs_by_level[f'level{i+1}'] = []
    #         for each in ['infra','geo','other']:
    #             if hasattr(self,f'model_input_{each}'):
    #                 if getattr(self,f'model_input_{each}')['params'] is not None:
    #                     params = getattr(self,f'model_input_{each}')['params']
    #                     for param_name in params:
    #                         # if mean is a dictionary, then there are levels embedded
    #                         if isinstance(params[param_name]['mean'],dict):
    #                             # if 'mean' for level is not 'user provided', then distribution exists for current level
    #                             if params[param_name]['mean'][f'level{i+1}'] == 'user provided':
    #                                 self._req_inputs_by_level[f'level{i+1}'].append(param_name)
    #                         else:
    #                             # if 'mean' is not 'user provided', then distribution exists
    #                             if params[param_name]['mean'] == 'user provided':
    #                                 self._req_inputs_by_level[f'level{i+1}'].append(param_name)
                                
    
    @classmethod
    def print_inputs(cls, disp_dist=True):
        """Print inputs"""
        dict_list = ['_INPUT_PBEE_DIST', '_MODEL_INPUT_INFRA', '_MODEL_INPUT_GEO', '_MODEL_INPUT_OTHER', '_MODEL_INPUT_FIXED']
        # dist_info = ['mean','sigma','cov','low','high','dist_type']
        for key in dict_list:
            if hasattr(cls,key):
                if getattr(cls,key)['desc'] is not None:
                    count = 0
                    meta = getattr(cls,key).copy()
                    logging.info(f"\n{meta['desc']}")
                    params = meta['params'].copy()
                    if len(params)>0:
                        for param in params:
                            count += 1
                            # logging.info(f'\t{count}) {param}: {params[param]}')
                            # print description
                            if 'desc' in params[param]:
                                param_desc = params[param]["desc"]
                            else:
                                param_desc = params[param]
                            if isinstance(param_desc,str):
                                logging.info(f'\t{count}) {param}: {param_desc}')
                            else:
                                for i, item in enumerate(param_desc):
                                    if i == 0:
                                        logging.info(f'\t{count}) {param}: {param_desc[i]}')
                                    else:
                                        logging.info(f'\t\t{param_desc[i]}')
                            # print distribution information
                            # if disp_dist:
                            #     for each in dist_info:
                            #         if each in params[param]:
                            #             if isinstance(params[param][each],dict):
                            #                 logging.info(f'\t\t{each}:')
                            #                 for level in params[param][each]:
                            #                     logging.info(f'\t\t\t{level}: {params[param][each][level]}')
                            #             else:
                            #                 logging.info(f'\t\t{each}: {params[param][each]}')
                    else:
                        logging.info(f"\tNone")
        
                
    @property
    def n_model_terms(self):
        """Count number of terms in model, including coeff"""
        return sum(len(self.model_form_detail[level]) for level in self.model_form_detail)


    # call function
    def __call__(self):
        """Actions to perform when called after instantiation"""
        pass
        # self._get_inputs(**kwargs)
        # self._perform_calc()
        # self._get_results()
        # return self._get_results()
    
    
    def clear_inputs(self):
        """Clear inputs dictionary"""
        self._inputs = {}
    
        
    # def set_input_pbee_rv_dist(self, mean, aleatory, epistemic, dist_type='lognormal'):
    def set_input_pbee_dist(self, kwargs):
        """sets distribution of upstream PBEE variable"""
        for param_name in self.input_pbee_dist['params']:
            if param_name in kwargs:
                if isinstance(kwargs[param_name], dict):
                    for item in kwargs[param_name]:
                        if item in self.input_pbee_dist['params'][param_name]:
                            self.input_pbee_dist['params'][param_name][item] = kwargs[param_name][item]
                else:
                    self.input_pbee_dist['params'][param_name]['mean'] = kwargs[param_name]
        
    def set_analysis_size(self, n_site=1, n_sample=1):
        """set size to use in analysis, n_site and n_sample"""
        self._n_site = n_site
        self._n_sample = n_sample
    
    
    def set_all_inputs(self, kwargs):
        """Set all parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting all inputs')
        self._set_general_inputs(kwargs, param_type='infra')
        self._set_general_inputs(kwargs, param_type='geo')
        self._set_general_inputs(kwargs, param_type='other')
        self._set_general_inputs(kwargs, param_type='fixed')
    
            
    def set_infra_inputs(self, kwargs):
        """Set site parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting site inputs')
        self._set_general_inputs(kwargs, param_type='infra')
            
            
    def set_geo_inputs(self, kwargs):
        """Set eq parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting earthquake inputs')
        self._set_general_inputs(kwargs, param_type='geo')
    
    
    def set_other_inputs(self, kwargs):
        """Set other parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting other inputs')
        self._set_general_inputs(kwargs, param_type='other')

    
    def set_fixed_inputs(self, kwargs):
        """Set other parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting other inputs')
        self._set_general_inputs(kwargs, param_type='fixed')
        
    
    def _set_general_inputs(self, kwargs, param_type='geo'):
        """Set parameters: 'infra', 'geo', 'other', or 'fixed'"""
        # inputs
        params = getattr(self,f'model_input_{param_type}')['params']
        if params is None:
            params = {}
        missing_inputs = getattr(self,f'_missing_inputs_{param_type}')
        # loop through and set inputs:
        for param_name in params:
            if param_name in kwargs:
                self._inputs[param_name] = kwargs.get(param_name)
                if param_name in missing_inputs:
                    missing_inputs.remove(param_name) # remove from missing inputs list
            else:
                if not param_name in missing_inputs:
                    missing_inputs.append(param_name) # add to list of missing inputs to report
        # print missing inputs
        if len(missing_inputs) > 0:
            missing_param_str = '", "'.join(missing_inputs)
            logging.info(f'missing "{missing_param_str}" for "{param_type}" inputs')
        # store missing inputs
        setattr(self,f'_missing_inputs_{param_type}',missing_inputs)
        # expand dims
        self._expand_input_dims()
        
    
    # def _clean_up_sigma_or_cov(self):
    #     """if both sigma and cov are present, remove one"""
    #     for param_name in self._inputs:
    #         if 'sigma' in self._inputs[param_name]:
    #             if np.isnan(self._inputs[param_name]['sigma']):
    #                 # check for cov
    #                 if 'cov' in self._inputs[param_name]:
    #                     if 
                    
                    
                
    #             if self._inputs[param_name]['sigma'] == 'preferred' or \
    #                 self._inputs[param_name]['sigma'] is None or \
    #                     np.isnan(self._inputs[param_name]['sigma']):
    #                 # check for cov
    #                 if 'cov' in self._inputs[param_name]:
    #                     if self._inputs[param_name]['cov'] == 'preferred' or \
    #                         self._inputs[param_name]['cov'] is None or \
    #                             np.isnan(self._inputs[param_name]['cov']):
    #                         # get value from internal
    #                         # see whether internal dist uses cov or sigma
    #                         if 'sigma' in internal_param_dist:
    #                             to_get = 'sigma'
    #                         elif 'cov' in internal_param_dist:
    #                             to_get = 'cov'
    #                         # if mean is a dictionary, then there are levels embedded
    #                         if isinstance(internal_param_dist[to_get],dict):
    #                             self._inputs[param_name][to_get] = internal_param_dist[to_get][f'level{self._level_to_run}']
    #                         else:
    #                             self._inputs[param_name][to_get] = internal_param_dist[to_get]
    #                 # remove cov is sigma value is valid
    #                 if 'cov' in self._inputs[param_name]:
    #                     self._inputs[param_name].pop('cov')


    def _expand_input_dims(self):
        """For all input parameters, expand dims to be 2d (1 x n_site)"""
        for param_name in self._inputs:
            if isinstance(self._inputs[param_name],dict):
                for item in self._inputs[param_name]:
                    if np.ndim(self._inputs[param_name][item]) == 0:
                        self._inputs[param_name][item] = np.repeat(self._inputs[param_name][item],self._n_site)
            else:
                if np.ndim(self._inputs[param_name]) == 0:
                    self._inputs[param_name] = np.repeat(self._inputs[param_name],self._n_site)
    
    
    def perform_calc(self, run_with_mean=True, return_inter_params=False):
        """Performs calculations"""        
        # prepare inputs
        inputs = {}
        if run_with_mean:
            for param_name in self._inputs:
                if isinstance(self._inputs[param_name],dict):
                    inputs[param_name] = self._inputs[param_name]['mean']
                else:
                    inputs[param_name] = self._inputs[param_name]
            # get upstream PBEE RV
            for param_name in self.input_pbee_dist['params']:
                inputs[param_name] = self.input_pbee_dist['params'][param_name]['mean']
        
        # run model
        output = self._model(**inputs, return_inter_params=return_inter_params)
        
        # store final params
        for param_name in self.return_pbee_dist['params']:
            self._outputs.update({
                param_name: output[param_name]
            })
            output.pop(param_name)
        # store remaining as intermediate params
        for param_name in output:
            self._intermediates.update({
                param_name: output[param_name]
            })
    
    
    @staticmethod
    def _get_kwargs_for_lambda_func(kwargs, lambda_func, inds=None):
        """returns dictionary with only arguments for lambda function"""
        if inds is None:
            return {key:val for key, val in kwargs.items() if key in lambda_func.__code__.co_varnames}
        else:
            return {key:val[inds] for key, val in kwargs.items() if key in lambda_func.__code__.co_varnames}
    
    
    @staticmethod
    def _get_mean_coeff_for_lambda_func(coeff_dict, lambda_func):
        """returns dictionary with only arguments for lambda function"""
        return {key:val['mean'] for key, val in coeff_dict.items()}

    
    # get inputs
    def set_inputs(self, kwargs):
        """Checks kwargs for input parameters and stores in class"""
        # get general params
        if self.req_pbee_rv in kwargs:
            self._inputs['n_site'] = kwargs.get('n_site')
        else:
            return ValueError(f'missing "{n_site}" in inputs; must provide it to proceed')
        if self.req_pbee_rv in kwargs:
            self._inputs[self.req_pbee_rv] = kwargs.get(self.req_pbee_rv)
        else:
            return ValueError(f'missing "{self.req_pbee_rv}" in inputs; must provide it to proceed')
        # self._inputs['n_sample'] = kwargs.get('n_sample')
        # self._inputs['n_event'] = kwargs.get('n_event')
        # get required inputs
        missing_inputs = []
        for param in self.model_input_rv:
            self._inputs[param] = kwargs.get(param, None)
            if not param in kwargs:
                missing_inputs.append(param) # add to list of missing inputs to report
                self.sample_rv(param)
        if len(missing_inputs) > 0:
            logging.info(f'missing "{", ".join(missing_inputs)}" in inputs; will sample from default distribution')
        # store missing inputs
        self._missing_inputs = missing_inputs
            
            # if param in kwargs:
            # try:
                # self._inputs[param] = kwargs.get(param)
            # else:
            # except ValueError:
                # raise MissingParameterError(f'Missing "{param}" in inputs; will sample from distribution')

                # raise ValueError(f"Missing model input: {param}; cannot proceed with method.")
        # get optional inputs
        # for param in self.INPUT['OPTIONAL']:
        #     self._inputs[param] = kwargs.get(param, self.INPUT['OPTIONAL'][param])


    def sample_rv(self, param):
        """sample from default distribution"""
        if param in self.model_input_rv:
            dist_type = self.model_input_rv[param]['dist']
            if self.model_input_rv[param]['mean'] is None:
                return ValueError(f"need mean for {param} to perform sampling")
            else:
                # get mean
                mean = self.model_input_rv[param]['mean']
                if dist_type == 'lognormal':
                    mean = np.log(mean)
                # get sigma
                if self.model_input_rv[param]['sigma'] is None:
                    if self.model_input_rv[param]['cov'] is None:
                        raise ValueError(f"need either sigma or CoV for {param} to perform sampling")
                    else:
                        sigma = mean*self.model_input_rv[param]['cov']
                else:
                    sigma = self.model_input_rv[param]['sigma']
                # get min and max
                low = self.model_input_rv[param]['min']
                high = self.model_input_rv[param]['max']
                if dist_type == 'lognormal':
                    low = np.log(low)
                    if np.isnan(low):
                        low = -np.inf
                    high = np.log(high)
                # get samples
                self._inputs[param] = self.rvs(
                    mean=mean, sigma=sigma, dist_type=dist_type,
                    low=low, high=high,
                    n_sample=self._inputs['n_site']
                )
        else:
            return ValueError(f'default distribution for "{param}" does not exist')


    @staticmethod
    def rvs(mean, sigma, dist_type, low=None, high=None, n_sample=1):
        """set boundaries to infinite if not given"""
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        # get samples
        samples = truncnorm.rvs(
            a=(low-mean)/sigma,
            b=(high-mean)/sigma,
            loc=mean, scale=sigma, size=n_sample)
        # if lognormal, convert to linear scale
        if dist_type == 'lognormal':
            samples = np.exp(samples)
        return samples


    # get results
    # def _get_results(self):
    #     """Return outputs based on self.OUTPUT definition"""
    #     if 'return_param' in self._inputs:
    #         if 'all' in self._inputs['return_param']:
    #             for param in self.OUTPUT:
    #                 self._outputs[param] = self._inters[param]
    #         else:
    #             for param in self._inputs.get('return_param'):
    #                 if param in self._inters:
    #                     self._outputs[param] = self._inters[param]
    #                 else:
    #                     raise ValueError(f"Requested return parameter {param} is not a valid output.")
    #     else:
    #         for param in self.OUTPUT:
    #             self._outputs[param] = self._inters[param]
    #     return self._outputs

    # get model distribution
    # def get_dist(self):
    #     """Return model distribution"""
    #     # self._outputs['prob_dist'] = {
    #     #     'type': self.DIST['TYPE'],
    #     #     'sigma_total': self.DIST['TOTAL'],
    #     #     'sigma_aleatory': self.DIST['ALEATORY'],
    #     #     'sigma_epistemic': self.DIST['sigma']
    #     # }
    #     return self.MODEL_DIST


    @classmethod
    def run_check(cls, rtol=5e-3, atol=0, verbose=True, detailed_verbose=False, opensra_path=None):
        """Check result from function against known results"""
        # tolerance to use
        tol_dict = {}
        if rtol is None and atol is None:
            tol_dict['rtol'] = 1e-5
        else:
            if rtol is not None:
                tol_dict['rtol'] = rtol
            if atol is not None:
                tol_dict['atol'] = atol
        if opensra_path is None:
            opensra_path = os.getcwd()
        test_file_dir = os.path.join(
            opensra_path,
            'test',
            # cls._RETURN_PBEE_META['category'].lower(),
            cls._RETURN_PBEE_DIST['category'].lower(),
            get_basename_without_extension(sys.modules[cls.__module__].__file__)
        )
        method_name = cls.__name__
        test_file = os.path.join(test_file_dir, f"{method_name}.csv")
        # read test file
        test_data = pd.read_csv(test_file)
        # convert test_data DataFrame to dictionary
        test_data_dict = {
            key: test_data[key].values
            for key in test_data.columns
        }
        
        # initialize method instance
        inst = cls()
        # set analysis size
        inst.set_analysis_size()
        # set input pbee distribution
        inst.set_input_pbee_dist(test_data_dict)
        # set input parameters
        inst.set_all_inputs(test_data_dict)
        # perform calculations
        inst.perform_calc(return_inter_params=True)
        
        # distribution metric to check, some do not contain
        # if 'liquefaction' in inst.__module__:
            # dist_metric = ['mean','sigma','sigma_mu']
        # else:
            # dist_metric = ['mean']
        dist_metric = ['mean','sigma','sigma_mu']
        
        # perform check
        print(f"Running tests for {cls.__module__}.{method_name}:")
        if verbose and detailed_verbose:
            print(f"\tChecking...")
        for group in ['_intermediates', '_outputs']:
            # first check outputs
            list_params = getattr(inst,group)
            if verbose and detailed_verbose:
                if len(list_params) == 0:
                    print(f"\t\t- {group}: none")
                else:
                    print(f"\t\t- {group}: {', '.join(list_params)}")
            for param_name in list_params:
                if verbose and detailed_verbose:
                    print(f"\t\t\t- {param_name}")
                # if output is a dictionary, then it is a distribution - check metrics
                param_out = getattr(inst,group)[param_name]
                if isinstance(param_out,dict):
                    for met in dist_metric:
                        if param_out[met] is not None:
                            if verbose and detailed_verbose:
                                print(f"\t\t\t\t- {met}")
                            # metric name in table
                            if met != 'mean':
                                table_param_name = met + '_' + param_name
                            else:
                                table_param_name = param_name
                            # run check
                            cls._numpy_check(
                                param_out[met], # calculation
                                test_data_dict[table_param_name], # true
                                rtol=rtol, atol=atol,
                            )
                else:
                    # run check
                    cls._numpy_check(
                        param_out, # calculation
                        test_data_dict[param_name], # true
                        rtol=rtol, atol=atol,
                    )                
                    
        # # print message if able to check through all cases
        if verbose:
            if detailed_verbose:
                print(f"\t...tests passed for {cls.__module__}.{method_name}\n")
            else:
                print(f"\t...passed\n")
                # print(f"Running tests for {cls.__module__}.{method_name}... passed\n")


    @staticmethod
    def _numpy_check(arr_calc, arr_ref, rtol, atol):
        try:
            assert_allclose(
                arr_calc, # calculation
                arr_ref, # true
                rtol=rtol, atol=atol, equal_nan=True,
            )
        # if returning TypeError, likely because elements are strings, then check string equality
        except TypeError:
            assert_array_equal(
                arr_calc, # calculation
                arr_ref, # true
            )
        # if can't assert, try higher rtol
        except AssertionError:
            assert_allclose(
                arr_calc, # calculation
                arr_ref, # true
                rtol=rtol*10, atol=atol, equal_nan=True,
            )


    @staticmethod
    def get_sigma_from_cov(mean, cov):
        """Calculates sigma from COV [decimals]"""
        return mean*cov


    @classmethod
    def _update_param(cls):
        return NotImplementedError("To be added")


    @staticmethod
    def _convert_to_ndarray(arr, length=1):
        """convert to array; if float/int, then add dimension to convert to array"""
        if isinstance(arr, np.ndarray):
            return arr
        elif isinstance(arr, list):
            return np.asarray(arr)
        else:
            return np.asarray([arr]*length)
        # return arr if isinstance(arr, np.ndarray) else np.asarray([arr]*length)


    # addtional templates
    def instance_method_placeholder_1(self):
        pass

    @classmethod
    def class_method_placeholder_1(cls):
        pass

    @staticmethod
    def static_method_placeholder_1():
        pass


# # -----------------------------------------------------------
class GenericModel(BaseModel):
    """
    Compute output using generic model, e.g.:
    ln(Y) = c0 + c1*RV1 + c2*RV1^2 + c3ln(RV2) + ...
    
    Parameters
    ----------
    coeffs: float, np.ndarray or list
       coefficients starting with constant
    params: float, np.ndarray or list
       [mm] pipe outside diameter

    Returns
    -------
    TBD : TBD
        [TBD] TBD

    References
    ----------
    .. [1] Authors, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.

    """

    # class definitions
    _NAME = 'GenericModel_2022'  # Name of the model
    _ABBREV = 'GM22'            # Abbreviated name of the model
    _MODEL_PBEE_CAT = None        # Return category in PBEE framework, e.g., IM, EDP, DM
    _MODEL_RETURN_RV = None        # Return variable for PBEE category, e.g., pgdef, pipe_strain
    _MODEL_TYPE = None               # Type of model (e.g., liquefaction, landslide)
    _MODEL_DIST = {          # Distribution information for model
        'type': 'normal',
        'aleatory': 1,
        'sigma': 0,
    }
    _REQ_PBEE_CAT = None     # Upstream PBEE variable required by model, e.g, IM, EDP, DM
    _REQ_PBEE_RV = None     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
    _MODEL_INPUT_RV = {} # random inputs with means, sigmas, and distributions (normal or lognormal)
    _MODEL_FORM_DETAIL = {
        'level1': {},
        'level2': {},
        'level3': {}
    }
    # _FIXED = {}
    # _OUTPUT = []

    # instantiation
    def __init__(self):
        super().__init__()    
    
    # get inputs
    def set_inputs(self, kwargs):
        """Checks kwargs for input parameters and stores in class"""
        # get general params
        if self.req_pbee_rv in kwargs:
            self._inputs['n_site'] = kwargs.get('n_site')
        else:
            return ValueError(f'missing "{n_site}" in inputs; must provide it to proceed')
        if self.req_pbee_rv in kwargs:
            self._inputs[self.req_pbee_rv] = kwargs.get(self.req_pbee_rv)
        else:
            return ValueError(f'missing "{self.req_pbee_rv}" in inputs; must provide it to proceed')
        # self._inputs['n_sample'] = kwargs.get('n_sample')
        # self._inputs['n_event'] = kwargs.get('n_event')
        # get required inputs
        missing_inputs = []
        for param in self.model_input_rv:
            self._inputs[param] = kwargs.get(param, None)
            if not param in kwargs:
                missing_inputs.append(param) # add to list of missing inputs to report
                self.sample_rv(param)
        if len(missing_inputs) > 0:
            logging.info(f'missing "{", ".join(missing_inputs)}" in inputs; will sample from default distribution')
        # store missing inputs
        self._missing_inputs = missing_inputs
            
            # if param in kwargs:
            # try:
                # self._inputs[param] = kwargs.get(param)
            # else:
            # except ValueError:
                # raise MissingParameterError(f'Missing "{param}" in inputs; will sample from distribution')

                # raise ValueError(f"Missing model input: {param}; cannot proceed with method.")
        # get optional inputs
        # for param in self.INPUT['OPTIONAL']:
        #     self._inputs[param] = kwargs.get(param, self.INPUT['OPTIONAL'][param])
    
    
    def define_upstream_pbee_info(self, req_pbee_cat, req_pbee_rv):
        self.req_pbee_cat = req_pbee_cat
        self.req_pbee_rv = req_pbee_rv
        
    def define_return_pbee_info(self, model_pbee_cat, model_type, model_return_rv):
        self.model_pbee_cat = model_pbee_cat
        self.model_type = model_type
        self.model_return_rv = model_return_rv
        
    def define_model_dist(self, dist_type='lognormal', aleatory=1, epistemic=0):
        self.model_dist = {
            'type': dist_type,
            'aleatory': aleatory,
            'epistemic': epistemic,
        }

    def add_model_rv(self, rv_label,
                     mean=None, sigma=None, cov=None, 
                     dist_min=-np.inf, dist_max=np.inf,
                     dist_type='normal', unit=None):
        """"Add to or overwrite a dictionary of random variables"""
        if cov is None and sigma is None:
            return ValueError('Must provide either "cov" or "sigma"')
        self.model_input_rv[rv_label] = {
            'mean': mean,
            'sigma': sigma,
            'cov': cov,
            'min': dist_min,
            'max': dist_max,
            'dist': dist_type
        }

    # add model term
    def add_model_term(self, level=1, coeff_mean=1, coeff_sigma=0, rv_label=None, apply_ln=False, power=1):
        """Add terms to model form; for coefficient, set rv_label to 'None' or 'coeff'"""
        # make term label
        if rv_label is None or isinstance(rv_label, float):
            term_id = 'term0'
        elif isinstance(rv_label, str):
            if 'const' in rv_label:
                term_id = 'term0'
            else:
                if( 'term0' in self.model_form_detail['level1'] or 
                    'term0' in self.model_form_detail['level2'] or
                    'term0' in self.model_form_detail['level3']):
                    term_id = f'term{self.n_model_terms}'
                else:
                    term_id = f'term{self.n_model_terms+1}'
        # if term0 (for constant), set rv_label to 'None' if it is not already 'None'
        if term_id == 'term0':
            rv_label = None
        # set values
        self.model_form_detail[f"level{level}"][term_id] = {
            'coeff': {
                'mean': coeff_mean,
                'sigma': coeff_sigma
            },
            'var': {
                'label': rv_label,
                'apply_ln': apply_ln,
                'power': power
            }
        }
    
    

    # make string of model form
    def construct_model_form(self):
        """Constructs strings of model form"""
        self.model_form = {}
        list_of_rvs = []
        lambda_arg_string = 'lambda' # for creating argument string
        lambda_func_string = ':' # for creating argument string
        # return string
        if self.model_return_rv is None:
            output_string = None
        else:
            if self.model_dist['type'] == 'lognormal':
                output_string = f'ln({self.model_return_rv}) ='
            else:
                output_string = f'{self.model_return_rv} ='
        term_counter = 0
        # make string for every level
        for level in self.model_form_detail:
            self.model_form[level] = {}
            # preset for LEVEL1; string gets appended for subsequent levels
            if level == 'level1':
                model_string_for_level = output_string
            level_info = self.model_form_detail[level] # level info
            # for every term in level
            for term in level_info:
                term_info = level_info[term]
                coeff_str = term.replace('term','c') # coefficient string
                # coeff_val_str = str(term_info['coeff']['mean'])
                if term_info['var']['label'] is None:
                    term_string = coeff_str # no additional string from RV
                    # lambda_func_string += f" {term_info['coeff']['mean']}" # for creating model
                else:
                    if term_info['var']['label'] not in list_of_rvs:
                        list_of_rvs.append(term_info['var']['label'])
                        # add RV to arguments
                        if len(list_of_rvs) == 1:
                            lambda_arg_string += f" {term_info['var']['label']}"
                        else:
                            lambda_arg_string += f", {term_info['var']['label']}"
                    # create term for function
                    if term_info['var']['apply_ln']:
                        rv_str = f"ln({(term_info['var']['label'])})" # add ln label
                    else:
                        rv_str = term_info['var']['label']
                    if term_info['var']['power'] > 1:
                        rv_str = f"({rv_str})**{term_info['var']['power']}" # add power term
                    term_string = f"{coeff_str}*{rv_str}" # no additional string from RV
                # append to model string
                if term_counter == 0:
                    model_string_for_level += f" {term_string}"
                    lambda_func_string += f" {term_string.replace(coeff_str,str(term_info['coeff']['mean']))}"
                else:
                    model_string_for_level += f" + {term_string}"
                    lambda_func_string += f" + {term_string.replace(coeff_str,str(term_info['coeff']['mean']))}"
                # lambda_func_string = model_string_for_level.replace(term_string, str(term_info['coeff']['mean']))
                term_counter += 1
            # store model string under dictionary for current level
            self.model_form[level]['string'] = model_string_for_level
            # self.model_form[level]['func'] = eval(model_string_for_level.replace(output_string, lambda_arg_string+' :'))
            self.model_form[level]['func_string'] = lambda_arg_string + lambda_func_string
            self.model_form[level]['func'] = eval(lambda_arg_string + lambda_func_string)
    
    # update calculation method
    def perform_calc(self):
        """Performs calculations"""
        pass
        # pull inputs locally
        # n_sample = self._inputs['n_sample']
        # n_site = self._inputs['n_site']



    # generic model
    # @staticmethod
    # # @jit
    # # @jit(nopython=True)
    # def _model():
    #     pass