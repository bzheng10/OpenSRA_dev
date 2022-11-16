# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Base classes used in OpenSRA
#
# Created: April 1, 2022
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python base modules
import os
import logging
import sys

# data manipulation modules
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import interp1d

# geospatial processing modules


# efficient processing modules
# import numba as nb
from numba import njit, float64, int64, typeof
# from numba.core import types
# from numba.typed import Dict
from numba.types import Tuple

# plotting modules
# if importlib.util.find_spec('matplotlib') is not None:
    # import matplotlib.pyplot as plt
    # from matplotlib.collections import LineCollection
# if importlib.util.find_spec('contextily') is not None:
    # import contextily as ctx

# OpenSRA modules
from src.util import get_basename_without_extension
# sys.append('..')


# disable some numpy warnings
np.seterr(divide='ignore', invalid='ignore')


# -----------------------------------------------------------
class GMPE(object):
    """
    Base class for GMPEs
    
    Parameters
    ----------
    
    Returns
    -------
        
    References
    ----------
    .. [1] Author, Year, Title, Publication, Issue, Pages.
    
    """
    
    # class definitions
    _NAME = None   # Name of the model
    _ABBREV = None                 # Abbreviated name of the model
    _REF = "".join([                 # Reference for the model
        'Authors, Year, ',
        'Title, ',
        'Publication, ',
        'Issue, Pages.'
    ])
    _MODEL_PBEE_CAT = 'IM'           # Return category in PBEE framework, e.g., IM, EDP, DM
    _MODEL_RETURN_RV = ['PGA', 'PGV', 'Sa_T'] # Return variable for PBEE category, e.g., pgdef, pipe_strain
    _MODEL_TYPE = 'seismic_intensity'               # Type of model (e.g., liquefaction, landslide)
    _MODEL_DIST = {                          # Distribution information for model
        'type': 'lognormal',
        'mean': None,
        'aleatory': {
            'phi': None,
            'tau': None,
            'sigma': None
        },
        'epistemic': None,
        'period': None,
        'dims': None,
        'dims_desc': 'n_period x n_site'
    }
    _REQ_PBEE_CAT = None     # Upstream PBEE variable required by model, e.g, IM, EDP, DM
    _REQ_PBEE_RV = None     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
    
    # Model inputs
    # Site variables
    # _MODEL_INPUT_SITE = {}
    
    # Earthquake scenario variables
    # _MODEL_INPUT_EQ = {}
    
    # Other parameters
    # _MODEL_INPUT_OTHER = {}
    
    
    # TEMP
    _FIXED = {}
    _OUTPUT = []
    
    # Other backend parameters
    _OPENSRA_DIR = os.path.realpath(__file__)
    count = 0
    while not _OPENSRA_DIR.endswith('OpenSRA'):
        _OPENSRA_DIR = os.path.abspath(os.path.dirname(_OPENSRA_DIR))
        # in case can't locate OpenSRA dir and goes into infinite loop
        if count>5:
            print('Cannot locate OpenSRA directory - contact dev.')
        count += 1
    # if not os.path.basename(os.getcwd()) == 'OpenSRA' and not os.path.basename(os.getcwd()) == 'OpenSRABackEnd':
    #     os.chdir('..')
    # _OPENSRA_DIR = os.getcwd()
    # _GMPE_COEFF_DIR = os.path.join('..','..','OpenSRA','lib','NGAW2_Supplement_Data')
    _GMPE_COEFF_DIR = os.path.join(_OPENSRA_DIR,'lib','NGAW2_Supplement_Data')
    _PGA_PERIOD = 0
    _PGV_PERIOD = -1
    
    
    # instantiation
    def __init__(self, opensra_dir=None):
        """Create an instance of the class"""
        
        # initialize instance variables
        self._set_instance_var()
        
        # gather required inputs
        self._gather_all_req_inputs()
        
        # read coefficients
        self.coeffs = self._read_coeffs(opensra_dir).copy()
        
        # initialize params
        self._inputs = {}
        self._calc_params = {}
        self._n_site = 0
        self._n_period = 0
    
    
    def _set_instance_var(self):
        """Store class variables to instance"""
        class_var_to_set = [
            attr for attr in dir(self) \
                if attr.startswith("_") and \
                not attr.startswith("__") and \
                attr[1].isupper()
        ]
        for var in class_var_to_set:
            if isinstance(getattr(self, var),dict):
                setattr(self, var.lower()[1:], getattr(self, var).copy())
            else:
                setattr(self, var.lower()[1:], getattr(self, var))
    
    
    @classmethod
    def _read_coeffs(cls, opensra_dir=None):
        """Read table of coefficients"""
        # check if OpenSRA path is given explicitly, use if so
        if opensra_dir is None:
            coeffs_path = os.path.join(cls._GMPE_COEFF_DIR, f"{cls.__name__.lower()}.csv") # file
        else:
            coeffs_path = os.path.join(opensra_dir,'lib','NGAW2_Supplement_Data',f"{cls.__name__.lower()}.csv")
        coeffs = pd.read_csv(coeffs_path) # read from table
        return coeffs

    
    @classmethod
    def print_inputs(cls):
        """Print inputs for the different parts to the GMPE"""
        dict_list = ['_MODEL_INPUT_SITE', '_MODEL_INPUT_EQ', '_MODEL_INPUT_OTHER']
        for key in dict_list:
            if hasattr(cls,key):
                req_count = 0
                opt_count = 0
                meta = getattr(cls,key).copy()
                logging.info(f"\n{meta['desc']}")
                req_params = meta['required']
                opt_params = meta['optional']
                logging.info(f'Required:')
                if len(req_params)>0:
                    for param in req_params:
                        req_count += 1
                        # print description
                        param_desc = req_params[param]["desc"]
                        if isinstance(param_desc,str):
                            logging.info(f'\t{req_count}) {param}: {param_desc}')
                        else:
                            for i, item in enumerate(param_desc):
                                if i == 0:
                                    logging.info(f'\t{req_count}) {param}: {param_desc[i]}')
                                else:
                                    logging.info(f'\t\t{param_desc[i]}')
                        # print note
                        if req_params[param]['note'] is not None:
                            logging.info(f'\t->Note: {req_params[param]["note"]}')
                else:
                    logging.info(f"\tNone")
                logging.info(f'Optional:')
                if len(opt_params)>0:
                    for param in opt_params:
                        opt_count += 1
                        # print description
                        param_desc = opt_params[param]["desc"]
                        if isinstance(param_desc,str):
                            logging.info(f'\t{opt_count}) {param}: {param_desc}')
                        else:
                            for i, item in enumerate(param_desc):
                                if i == 0:
                                    logging.info(f'\t{opt_count}) {param}: {param_desc[i]}')
                                else:
                                    logging.info(f'\t\t{param_desc[i]}')
                        # print default value
                        logging.info(f'\t\t-> Default: {opt_params[param]["default"]}')
                        # print note
                        if opt_params[param]['note'] is not None:
                            logging.info(f'\t\t-> Note: {opt_params[param]["note"]}')
                else:
                    logging.info(f"\tNone")
    
    
    def clear_outputs(self):
        """Clear output dictionary"""
        self.model_dist['mean'] = None
        self.model_dist['aleatory']['phi'] = None
        self.model_dist['aleatory']['tau'] = None
        self.model_dist['aleatory']['sigma'] = None
        self.model_dist['dims'] = None
        self.model_dist['period'] = None
    

    def run_model(self, njit_on=False):
        """Some preprocesssing, then runs model"""
        
        #-------------------------------------------
        # convert period_out to array of pure numerics and reduce coeff matrix to only periods needed
        coeffs, period_out_num = self._reduce_coeffs_to_required_periods(
            self.coeffs.copy(), self._inputs['period_out'].copy()
        )
        
        # take out 'period_out' from inputs
        inputs = self._inputs.copy()
        inputs.pop('period_out', None)        
        
        # organize coefficient into by periods
        coeffs_dict = {}
        for col in coeffs.columns:
            coeffs_dict[col] = coeffs[col].values

        # loop through number of periods
        if njit_on:
            mean, phi, tau = \
                self._model(**inputs, **coeffs_dict)
        else:
            mean, phi, tau = \
                self._model.py_func(**inputs, **coeffs_dict)

        # convert to numpy array
        mean = np.asarray(mean)
        phi = np.asarray(phi)
        tau = np.asarray(tau)
        
        # calculate total aleatory variability, sigma
        sigma = np.sqrt(phi**2 + tau**2)
        
        #-------------------------------------------
        # setup output dictionary
        output = {
            'mean': mean,
            'phi': phi,
            'tau': tau,
            'sigma': sigma,
            # 'dist': self.model_dist['type'],
            'period': period_out_num,
            'dims': mean.shape,
            # 'dims_desc': 'n_period x n_site'
        }
        
        #-------------------------------------------
        # interpolate spectra at target periods if periods outside PGA and PGV are requested
        if max(period_out_num) > 0:
            output = self._interp_output_at_period_out(output, coeffs.Period.values)
        
        #-------------------------------------------
        # store output into distribution
        self.model_dist['mean'] = output['mean']
        self.model_dist['aleatory']['phi'] = output['phi']
        self.model_dist['aleatory']['tau'] = output['tau']
        self.model_dist['aleatory']['sigma'] = output['sigma']
        self.model_dist['dims'] = output['mean'].shape
        # self.model_dist['dims_desc'] = 'n_period x n_site'
        self.model_dist['period'] = period_out_num
    
    
    # @classmethod
    # def _setup_coeff_mat_for_numba(cls, coeffs, cols_to_skip):
    #     """convert coeffs to numba type dictionary to prepare for numba operation"""
    #     # prepare for numba operation
    #     # convert coeffs to numba type dictionary
    #     coeffs_nb = Dict.empty(
    #         key_type=types.unicode_type,
    #         # value_type=types.float64[:,:],
    #         value_type=types.float64[:,:],
    #     )
    #     # add dictionary entires
    #     for col in coeffs.columns:
    #         if not col in cols_to_skip:
    #             # coeffs_nb[col] = np.asarray(self._get_coeff(coeffs,col,n_site),dtype=float)
    #             coeffs_nb[col] = np.asarray(cls._get_coeff(coeffs,col,1),dtype=float)
    #     # self.coeffs_nb = coeffs_nb
    #     return coeffs_nb
    
    
    def _gather_all_req_inputs(self):
        """Gathers all required inputs for GMPE"""
        for each in ['site','eq','other']:
            if hasattr(self,f'model_input_{each}'):
                setattr(self,f'_missing_req_inputs_{each}',list(getattr(self,f'model_input_{each}')['required']))
    
        
    def set_all_inputs(self, kwargs):
        """Set all parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting all inputs')
        self._set_general_inputs(kwargs, param_type='eq')
        self._set_general_inputs(kwargs, param_type='site')
        self._set_general_inputs(kwargs, param_type='other')
    
            
    def set_site_inputs(self, kwargs):
        """Set site parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting site inputs')
        self._set_general_inputs(kwargs, param_type='site')
            
            
    def set_eq_inputs(self, kwargs):
        """Set eq parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting earthquake inputs')
        self._set_general_inputs(kwargs, param_type='eq')
            
            
    def set_other_inputs(self, kwargs):
        """Set other parameters"""
        # initalize lists for tracking missing variables
        # logging.info(f'Setting other inputs')
        self._set_general_inputs(kwargs, param_type='other')
    
    
    def _set_general_inputs(self, kwargs, param_type='eq'):
        """Set parameters: 'site', 'eq', or 'other'"""
        # inputs
        req_params = getattr(self,f'model_input_{param_type}')['required']
        opt_params = getattr(self,f'model_input_{param_type}')['optional']
        missing_req_inputs = getattr(self,f'_missing_req_inputs_{param_type}')
        # loop through and set required inputs:
        for param_name in req_params:
            if param_name in kwargs:
                self._inputs[param_name] = kwargs.get(param_name)
                if param_name in missing_req_inputs:
                    missing_req_inputs.remove(param_name) # remove from missing inputs list
            else:
                if not param_name in missing_req_inputs:
                    missing_req_inputs.append(param_name) # add to list of missing inputs to report
        # optional inputs:
        for param_name in opt_params:
            param_val = kwargs.get(param_name, opt_params[param_name]['default']) # if can't get, use default
            # if None, convert to np.nan
            if param_val is None:
                param_val = np.nan
            # convert periods that are in str to numerics (e.g., PGA)
            if param_name == 'period_out':
                param_val = self._convert_periods_to_numerics(param_val)
                self._n_period = len(param_val)
            else:
                if np.ndim(param_val) == 0:
                    # print(self.__class__, param_name)
                    # catch incorrect assign of NaN
                    if np.isnan(param_val) and opt_params[param_name]['default'] is not None:
                        param_val = opt_params[param_name]['default']
            # store
            self._inputs[param_name] = param_val
        # print missing inputs
        if len(missing_req_inputs) > 0:
            missing_param_str = '", "'.join(missing_req_inputs)
            logging.info(f'missing "{missing_param_str}" for required inputs')
        # store missing inputs
        setattr(self,f'_missing_req_inputs_{param_type}',missing_req_inputs)
        # expand dims if all required inputs are available
        total_num_missing_req = 0
        for each in ['site','eq','other']:
            total_num_missing_req += len(getattr(self,f'_missing_req_inputs_{each}'))
        if total_num_missing_req == 0:
            self._expand_input_dims()
    

    def _expand_input_dims(self):
        """For all input parameters (except f_region and period_out), expand dims to be 2d (1 x n_site)"""
        if np.ndim(self._inputs['mag']) == 0:
            self._n_site = 1
        else:
            self._n_site = len(self._inputs['mag'])
        for param_name in self._inputs:
            if param_name != 'f_region' and param_name != 'period_out':
                if np.ndim(self._inputs[param_name]) == 0:
                    self._inputs[param_name] = np.repeat(self._inputs[param_name],self._n_site)
                
    
    def _match_dims(self, param_names, dims=None):
        """For input parameters that are using default values, update dimension to match user-defined"""
        if dims is None:
            ndim = np.ndim(self._inputs['mag'])
        if ndim == 0:
            pass
        else:
            # length = len(self._inputs['mag'])
            for param_name in param_names:
                if param_name != 'period_out':
                    self._inputs[param_name] = np.ones(self._inputs['mag'].shape)*self._inputs[param_name]
    
    
    def clear_inputs(self):
        """empties self._inputs"""
        self._inputs = {}
    
        
    def update_inputs(self, param_name=None, param_val=None, kwargs=None):
        """Update inputs"""
        # update function
        def _update(store, name, val):
            store[name] = val # store
        # update by param name directly
        if param_name is not None and param_val is not None:
            # convert to ndarray
            if param_name == 'period_out':
                param_val = self._convert_periods_to_numerics(param_val)
            else:
                param_val = self._convert_to_ndarray(param_val).astype(float)
            self._inputs[param_name] = param_val # store
        # update by dictionary of parameters
        if kwargs is not None:
            for param_name in kwargs:
                if param_name in self._inputs:
                    param_val = kwargs.get(param_name)
                    # convert to ndarray
                    if param_name == 'period_out':
                        param_val = self._convert_periods_to_numerics(param_val)
                    else:
                        param_val = self._convert_to_ndarray(param_val).astype(float)
                    self._inputs[param_name] = param_val # store
        # rerun preprocess with new inputs
    
    
    @staticmethod
    def interp(x_source, y_source, x_out, scale='linear', kind='linear', fill_value=np.nan, bounds_error=False):
        """
        Interpolate source data at target x values
        Options: linear, semilogx, semilogy, loglog
        """
        if scale == 'linear':
            interp_func = interp1d(
                x_source, y_source,
                axis=0, kind=kind,
                fill_value=fill_value, bounds_error=bounds_error)
            return interp_func(x_out)
        elif scale == 'semilogx':
            interp_func = interp1d(
                np.log(x_source), y_source,
                axis=0, kind=kind,
                fill_value=fill_value, bounds_error=bounds_error)
            return interp_func(np.log(x_out))
        elif scale == 'semilogy':
            interp_func = interp1d(
                x_source, np.log(y_source),
                axis=0, kind=kind,
                fill_value=fill_value, bounds_error=bounds_error)
            return np.exp(interp_func(x_out))
        elif scale == 'loglog':
            interp_func = interp1d(
                np.log(x_source), np.log(y_source),
                axis=0, kind=kind,
                fill_value=fill_value, bounds_error=bounds_error)
            return np.exp(interp_func(np.log(x_out)))
        
        
    def get_coeff_for_period(self, period):
        """Get coeffients for a specific period"""
        # pull inputs locally
        coeffs = self.coeffs.copy()
        pga_period = self.pga_period
        pgv_period = self.pgv_period
        
        # action
        if isinstance(period,str):
            if period.lower() == 'pga':
                return coeffs.loc[coeffs.Period==pga_period,:].copy()
            elif period.lower() == 'pgv':
                return coeffs.loc[coeffs.Period==pgv_period,:].copy()
        else:
            if isinstance(period,list):
                ind = [np.where(coeffs.Period==val)[0][0] for val in period if len(np.where(coeffs.Period==val)[0])>0]
                return coeffs.loc[ind].copy()
            else:
                return coeffs.loc[coeffs.Period==period,:].copy()
        
    
    def print_result(self):
        """Print result from analysis"""
        logging.info(f"dist_type: {self.model_dist['type']}")
        logging.info(f"dims ({self.model_dist['dims_desc']}): {self.model_dist['dims']}")
        logging.info(f"mean: {self.model_dist['mean']}")
        logging.info(f"aleatory-sigma: {self.model_dist['aleatory']['sigma']}")
        logging.info(f"aleatory-tau: {self.model_dist['aleatory']['tau']}")
        logging.info(f"aleatory-phi: {self.model_dist['aleatory']['phi']}")
        logging.info(f"epistemic: {self.model_dist['epistemic']}")
    
    
    @classmethod
    def _convert_periods_to_numerics(cls, periods):
        """Convert periods to an array of numerics"""
        period_num = []
        # if string, then either PGA or PGV
        if isinstance(periods,str):
            if periods.lower() == 'pga':
                period_num.append(cls._PGA_PERIOD) # replace with period proxy
            elif periods.lower() == 'pgv':
                period_num.append(cls._PGV_PERIOD) # replace with period proxy
            else:
                raise ValueError(f'Acceptable string inputs: "PGA", "PGV"')
        # if single value
        elif isinstance(periods,float) or isinstance(periods,int):
            period_num.append(periods)
        else:
            # lists or arrays
            for val in periods:
                if isinstance(val,str):
                    if val.lower() == 'pga':
                        period_num.append(cls._PGA_PERIOD) # replace with period proxy
                    elif val.lower() == 'pgv':
                        period_num.append(cls._PGV_PERIOD) # replace with period proxy
                else:
                    period_num.append(val)
        return np.asarray(period_num)
    
    
    @classmethod
    def _reduce_coeffs_to_required_periods(cls, coeffs, period_out):
        """Convert period_out to array of pure numerics and reduce coeff matrix to only periods needed"""
        # convert period_out to array of pure numerics
        period_out_num = cls._convert_periods_to_numerics(period_out).astype(float)
        
        # see if periods in period_out all coincide with periods in coeffs matrix
        has_all_period_out = True
        coeff_period_array = coeffs.Period.to_numpy()
        ind_to_get = []
        for per in period_out_num:
            if per in coeff_period_array:
                ind_to_get.append(np.where(coeff_period_array==per)[0][0])
            else:
                has_all_period_out = False
                break
        
        # keep PGA (T=0s) in coeffs
        if not 0 in ind_to_get:
            ind_to_get.append(0)
            ind_to_get.sort()
        
        # if coeffs contain all periods in period_out, then just get those rows
        if has_all_period_out:
            coeffs = coeffs.loc[ind_to_get].reset_index(drop=True)

        else:
            # return periods not pga, pgv
            not_pga_pgv_ind = np.where(np.logical_and(period_out_num!=cls._PGA_PERIOD,period_out_num!=cls._PGV_PERIOD))
            period_out_num_no_pga_pgv = period_out_num[not_pga_pgv_ind]
            
            # check if target periods are outside the range for coefficients
            coeff_period_not_pga_pgv = coeffs.Period[np.where(np.logical_and(coeffs.Period!=cls._PGA_PERIOD,coeffs.Period!=cls._PGV_PERIOD))[0]]
            coeff_period_min = min(coeff_period_not_pga_pgv)
            coeff_period_max = max(coeff_period_not_pga_pgv)
            if min(period_out_num_no_pga_pgv) < coeff_period_min or max(period_out_num_no_pga_pgv) > coeff_period_max:
                raise ValueError(
                    f"Target periods cannot be outside the range of {coeff_period_min} and {coeff_period_max} for {cls.__name__}!" + \
                    f"; except for T_PGA={cls._PGA_PERIOD} and T_PGV={cls._PGV_PERIOD}")
            
            # remove coeffs outside limits of period_out_num
            if len(period_out_num_no_pga_pgv) > 0:
                ind_under_period_out_min = np.where(coeffs.Period<min(period_out_num_no_pga_pgv))[0]
                ind_above_period_out_max = np.where(coeffs.Period>max(period_out_num_no_pga_pgv))[0]
                # lower_bound
                if len(ind_under_period_out_min) > 0:
                    one_ind_up_from_min = min(ind_under_period_out_min[-1]+1,len(coeffs.Period)-1)
                    if coeffs.Period[one_ind_up_from_min] == min(period_out_num_no_pga_pgv):
                        ind_lower_cutoff = one_ind_up_from_min
                    else:
                        ind_lower_cutoff = one_ind_up_from_min-1
                else:
                    ind_lower_cutoff = 0
                # upper_bound
                if len(ind_above_period_out_max)>0:
                    one_ind_down_from_max = max(ind_above_period_out_max[0]-1,0)
                    if coeffs.Period[one_ind_down_from_max] == max(period_out_num_no_pga_pgv):
                        ind_upper_cutoff = one_ind_down_from_max
                    else:
                        ind_upper_cutoff = one_ind_down_from_max+1
                else:
                    ind_upper_cutoff = coeffs.shape[0]-1
                # get periods
                coeffs = pd.concat([
                    coeffs.loc[np.where(np.logical_or(coeffs.Period==cls._PGA_PERIOD,coeffs.Period==cls._PGV_PERIOD))[0]],
                    coeffs.loc[ind_lower_cutoff:ind_upper_cutoff],
                ],axis=0).reset_index(drop=True)
            else:
                coeffs = coeffs.loc[np.where(np.logical_or(coeffs.Period==cls._PGA_PERIOD,coeffs.Period==cls._PGV_PERIOD))[0]].reset_index(drop=True)
            # check to keep PGA and PGV coeffs
            if not cls._PGA_PERIOD in period_out_num:
                coeffs = coeffs.drop(np.where(coeffs.Period==cls._PGA_PERIOD)[0]).reset_index(drop=True)
            if not cls._PGV_PERIOD in period_out_num:
                coeffs = coeffs.drop(np.where(coeffs.Period==cls._PGV_PERIOD)[0]).reset_index(drop=True)
        #
        return coeffs, period_out_num
    
    
    @classmethod
    def _interp_output_at_period_out(cls, output, coeffs_period):
        """interpolate spectra at target periods"""
        period_out = output['period']
        # see if periods = 0 (PGA) and -1 (PGV) are in source and target periods
        periods_pga_ind = np.where(coeffs_period==cls._PGA_PERIOD)[0]
        periods_pgv_ind = np.where(coeffs_period==cls._PGV_PERIOD)[0]
        period_out_pga_ind = np.where(period_out==cls._PGA_PERIOD)[0]
        period_out_pgv_ind = np.where(period_out==cls._PGV_PERIOD)[0]
        # get indices of periods that are not PGA, PGV
        periods_not_pga_pgv = np.where(period_out>0)[0]
        # go through output items
        for item in output:
            if item != 'dist' and item != 'period' and item != 'dims' and item != 'dims_desc':
                # initialize interp array (n_period_out x n_site)
                interp_vals = np.zeros((len(period_out),output['dims'][1]))
                # first get values <= 0 (PGA, PGV):
                if len(period_out_pga_ind) > 0:
                    interp_vals[period_out_pga_ind[0]] = output[item][periods_pga_ind[0]]
                if len(period_out_pgv_ind) > 0:
                    interp_vals[period_out_pgv_ind[0]] = output[item][periods_pgv_ind[0]]
                # perform interpolation on rest of target periods
                if len(periods_not_pga_pgv) > 0:
                    if item == 'mean':
                        # for mean, just interpolate with semilogx
                        interp_vals[periods_not_pga_pgv] = \
                            cls.interp(
                                x_source=coeffs_period[coeffs_period>0],
                                y_source=output['mean'][coeffs_period>0],
                                x_out=period_out[periods_not_pga_pgv],
                                scale='semilogx',
                            )
                    else:
                        # for sigmas, square, interpolate, then sqrt
                        interp_vals[periods_not_pga_pgv] = np.sqrt(cls.interp(
                            x_source=coeffs_period[coeffs_period>0],
                            y_source=(output[item][coeffs_period>0])**2,
                            x_out=period_out[periods_not_pga_pgv],
                            scale='semilogx',
                        ))
                # update model_output dictionary
                output[item] = interp_vals
        output['dims'] = output['mean'].shape
        return output
    

    @staticmethod
    def _convert_to_ndarray(arr, length=1, expand_dim=False):
        """Convert to array; if float/int, then add dimension to convert to array"""
        if isinstance(arr, np.ndarray):
            return arr
        elif isinstance(arr, list):
            return np.asarray(arr)
        else:
            if expand_dim:
                return np.asarray([arr]*length)
            else:
                return np.asarray(arr)
        # return arr if isinstance(arr, np.ndarray) else np.asarray([arr]*length)
    
    
    @staticmethod
    def _get_coeff(coeffs,name,n=1):
        """Return coeff values for ID=name and repeat by n"""
        if n == 1:
            # return coeffs[name]
            return np.expand_dims(coeffs[name],axis=1)
        else:
            return np.repeat(np.expand_dims(coeffs[name],axis=1),repeats=n,axis=1)
    
    
    @classmethod
    def run_check(cls, rtol=0, atol=1e-3):
        """Check result from function against known results"""
        # tolerance to use
        tol_dict = {}
        if rtol is None and atol is None:
            tol_dict['atol'] = 1e-3
        else:
            if rtol is not None:
                tol_dict['rtol'] = rtol
            if atol is not None:
                tol_dict['atol'] = atol
        test_file_dir = os.path.join(
            'test',
            cls._MODEL_PBEE_CAT.lower(),
            get_basename_without_extension(sys.modules[cls.__module__].__file__)
        )
        gmpe_name = cls.__name__
        test_file = os.path.join(test_file_dir, f"{gmpe_name}.csv")
        # read test file
        test_data = pd.read_csv(test_file)
        # get inputs
        # note - NGAW2 spreadsheet gives ground motions for 23 periods (including PGA, PGV)
        n_period = 23
        # last 23 columns are sigmas
        test_sigma = test_data.iloc[:,-n_period:].copy()
        # next 23 columns from last are median values
        test_median = test_data.iloc[:,-2*n_period:-n_period].copy()
        # rest are input values
        test_inputs = test_data.iloc[:,:-2*n_period].copy() #
        # set sigma columns to median columns (pandas appends numbers to repeating column names)
        test_sigma.columns = test_median.columns
        # get periods from column names
        test_periods = np.asarray(test_median.columns)
        test_periods[test_periods=='pga'] = cls._PGA_PERIOD # replace PGA text with period value
        test_periods[test_periods=='pgv'] = cls._PGV_PERIOD # replace PGA text with period value
        test_periods = pd.to_numeric(test_periods) # convert to numeric

        # initialize gmpe instance
        inst = cls()
        # run each case and assert allclose
        for i in range(test_inputs.shape[0]):
            kwargs = {} # put inputs into dictionary
            for col in test_inputs.columns:
                kwargs[col] = test_inputs[col][i]
            kwargs['period_out'] = test_periods
            # set inputs
            inst.set_all_inputs(kwargs)
            inst.run_model(njit_on=False)
            # check relative error on mean
            assert_allclose(
                inst.model_dist['mean'][:,0], # calculation
                np.log(test_median.loc[i].values), # true
                rtol=rtol, atol=atol, equal_nan=True,
            )
            # check relative error on sigma
            assert_allclose(
                inst.model_dist['aleatory']['sigma'][:,0], # calculation
                test_sigma.loc[i].values, # true
                rtol=rtol, atol=atol, equal_nan=True,
            )
        # print message if able to check through all cases
        print(f"Tests pass for {cls.__module__}.{gmpe_name}")
     
     
# ----------------------------------------------------------- 
class WeightedGMPE(GMPE):
    """
    Weighted average of the NGA West 2 relationships, except for Idriss
    
    Parameters
    ----------
    
    Returns
    -------
        
    References
    ----------
    
    """
    
    # class definitions
    _NAME = 'NGA West 2 (2014)'   # Name of the model
    _ABBREV = 'NGAW2'                 # Abbreviated name of the model
    _REF = None                 # Reference for the model
    _MODEL_PBEE_CAT = 'IM'           # Return category in PBEE framework, e.g., IM, EDP, DM
    _MODEL_RETURN_RV = ['PGA', 'PGV', 'Sa_T'] # Return variable for PBEE category, e.g., pgdef, pipe_strain
    _MODEL_TYPE = 'seismic_intensity'               # Type of model (e.g., liquefaction, landslide)

    # Other parameters
    _MODEL_INPUT_OTHER = {
        "desc": 'Other inputs for model (see each GMPE for specific inputs):',
        'required': {
        },
        "optional": {
            'models': {
                'desc': f'A list of GMPEs to run',
                'default': ['ASK14','BSSA14','CB14','CY14'], 'unit': None, 'note': None
            },
            'weights': {
                'desc': [f'A list of weights for each specified GMPE',
                         '- if unspecified, weigh GMPEs equally',
                         '- if weights do not add up to one, then normalize weights to one'],
                'default': None, 'unit': None, 'note': None
            }
        },
    }
    
    
    # instantiation
    def __init__(self):
        """Create an instance of the class"""
        
        # initialize instance variables
        self._set_instance_var()
        
        # gather required inputs
        self._gather_all_req_inputs()
        
        # initialize params
        # self._gmpes_and_weights = {}
        self.instance = {}

    
    def set_gmpes_and_weights(self, gmpe_names=None, weights=None):
        """Set models and weights to run"""
        # clean stored values
        self._gmpes_and_weights = {}
        # if gmpes are not provided
        if gmpe_names is None:
            gmpe_names = self.model_input_other['optional']['models']['default']
        # get default weight vaLues if not provided
        if weights is None:
            weights = np.ones(len(gmpe_names)) / len(gmpe_names)
        else:
            # make sure length of weights match length of gmpes
            if len(gmpe_names) != len(weights):
                raise ValueError("Number of models and weights do not match.")
        # make sure weights add up to 1
        if sum(weights) != 1:
            weights = np.asarray(weights)/sum(weights)
        # initialize gmpe instance and set weight
        for i, each_gmpe in enumerate(gmpe_names):
            self.instance[each_gmpe] = {
                "inst": globals()[each_gmpe](),
                'weight': weights[i]
            }
            
        
    def run_model(self, inputs_dict, njit_on=False):
        """Some preprocesssing, then runs model"""
        # see if models and weights are assigned, if not, use default
        if len(self.instance) == 0:
            self.set_gmpes_and_weights()
        
        # also store mean by gmpe in order to calculate epistemic uncertainty
        mean_separate = {}
        
        # initialize instances of GMPEs
        for i, each_gmpe in enumerate(self.instance):
            # clear and set inputs
            self.instance[each_gmpe]['inst'].clear_inputs()
            self.instance[each_gmpe]['inst'].set_all_inputs(inputs_dict)
            # run model
            self.instance[each_gmpe]['inst'].run_model(njit_on=njit_on)
            # get means
            mean_separate[each_gmpe] = self.instance[each_gmpe]['inst'].model_dist['mean']
            # get outputs
            if i == 0:
                mean = \
                    self.instance[each_gmpe]['inst'].model_dist['mean'] * \
                    self.instance[each_gmpe]['weight']
                phi_2 = \
                    self.instance[each_gmpe]['inst'].model_dist['aleatory']['phi']**2 * \
                    self.instance[each_gmpe]['weight']
                tau_2 = \
                    self.instance[each_gmpe]['inst'].model_dist['aleatory']['tau']**2 * \
                    self.instance[each_gmpe]['weight']
                dims = self.instance[each_gmpe]['inst'].model_dist['dims']
                period = self.instance[each_gmpe]['inst'].model_dist['period']
            else:
                mean += \
                    self.instance[each_gmpe]['inst'].model_dist['mean'] * \
                    self.instance[each_gmpe]['weight']
                phi_2 += \
                    self.instance[each_gmpe]['inst'].model_dist['aleatory']['phi']**2 * \
                    self.instance[each_gmpe]['weight']
                tau_2 += \
                    self.instance[each_gmpe]['inst'].model_dist['aleatory']['tau']**2 * \
                    self.instance[each_gmpe]['weight']
        # compute base epistemic uncertainty
        # eq. 2.1 in Al Atik and Youngs (2013)
        epistemic_base = 0
        total_weight = 0
        for i, each_gmpe in enumerate(self.instance):
            epistemic_base += \
                self.instance[each_gmpe]['weight'] * (mean_separate[each_gmpe] - mean)**2
            total_weight += self.instance[each_gmpe]['weight']
        epistemic_base = np.sqrt(epistemic_base/total_weight)
        # compute added epistemic uncertainty
        any_gmpe = list(self.instance)[0]
        epistemic_add = self.get_additional_epistemic(
            period,
            self.instance[any_gmpe]['inst']._inputs['mag'],
            self.instance[any_gmpe]['inst']._inputs['rake'])
        # store to model_dist
        self.model_dist['mean'] = mean
        self.model_dist['aleatory']['phi'] = np.sqrt(phi_2)
        self.model_dist['aleatory']['tau'] = np.sqrt(tau_2)
        self.model_dist['aleatory']['sigma'] = np.sqrt(phi_2 + tau_2)
        self.model_dist['epistemic'] = np.sqrt(epistemic_base**2 + epistemic_add**2)
        self.model_dist['dims'] = dims
        # self.model_dist['dims_desc'] = 'n_period x n_site'
        self.model_dist['period'] = period

    
    @staticmethod
    def get_additional_epistemic(periods, mags, rakes):
        """Calculates additional epistemic uncertainty using Al Atik and Youngs (2013)"""
        # dimensions = n_period x n_site
        sigma_mu_add = np.zeros((len(periods),len(mags)))
        # precalculate sigma_mu for T < 1 sec
        # eq 4.1 in paper
        sigma_mu_lt_1 = 0.0665 * (mags-7) + 0.072
        sigma_mu_lt_1[mags<7] = 0.072
        # loop through periods
        for i in range(len(periods)):
            if periods[i] < 1:
                sigma_mu_add[i,:] = sigma_mu_lt_1
            else:
                # eq 4.2
                sigma_mu_add[i,:] = sigma_mu_lt_1 + 0.0217*np.log(periods[i])
        # determine positions where rake -> normal faulting (between -150 and -30 degrees)
        ind_rake_nm = np.where(np.logical_and(rakes>=-150,rakes<=-30))[0]
        # for normal faults, add additional sigma_mu
        # eq 4.3
        sigma_mu_add[:,ind_rake_nm] += 0.034
        return sigma_mu_add
        
        
# ----------------------------------------------------------- 
class ASK14(GMPE):
    """
    Abrahamson, Silva, and Kamai (2014)
    
    Parameters
    ----------
    
    Returns
    -------
        
    References
    ----------
    .. [1] Abrahamson, N.A., Silva, W.J., and Kamai, R., 2014, 
    Summary of the ASK14 Ground Motion Relation for Active Crustal Regions, 
    Earthquake Spectra, vol. 30, no. 3, pp. 1025-1055.
    
    """
    
    # class definitions
    _NAME = 'Abrahamson, Silva, and Kamai (2014)'   # Name of the model
    _ABBREV = 'ASK14'                 # Abbreviated name of the model
    _REF = "".join([                 # Reference for the model
        'Abrahamson, N.A., Silva, W.J., and Kamai, R., 2014, ',
        'NGA-Summary of the ASK14 Ground Motion Relation for Active Crustal Regions, ',
        'Earthquake Spectra, ',
        'vol. 30, no. 3, pp. 1025-1055.'
    ])
    # Model inputs
    # Site variables
    _MODEL_INPUT_SITE = {
        "desc": 'Site inputs for model:',
        'required': {
            'vs30': {
                'desc': 'Vs30 (m/s)',
                'default': None, 'unit': 'm/s', 'note': None},
        },
        "optional": {
            'z1p0': {
                'desc': 'Depth to Vs=1km/s (km)',
                'default': None, 'unit': 'km', 'note': None},
            'vs30_source': {
                'desc': 'Source for Vs30 (estimated (0) or measured (1))',
                'default': 0, 'unit': None, 'note': None},
        },
    }
    
    # Earthquake scenario variables
    _MODEL_INPUT_EQ = {
        "desc": 'EQ inputs for model:',
        'required': {
            'mag': {
                'desc': 'Moment magnitude',
                'default': None, 'unit': None, 'note': None},
            'dip': {
                'desc': 'Dip angle (deg)',
                'default': None, 'unit': 'degree', 'note': None},
            'z_tor': {
                'desc': 'Depth to top of rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'z_bor': {
                'desc': 'Depth to bottom of rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_rup': {
                'desc': 'Closest distance to coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_jb': {
                'desc': 'Closest distance to surface projection of coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_x': {
                'desc': 'Horizontal distance from top of rupture measured perpendicular to fault strike (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_y0': {
                'desc': 'Horizontal distance off the end of the rupture measured parallel to strike (km)',
                'default': None, 'unit': 'km', 'note': None},
            'rake': {
                'desc': ['Rake angle (deg)',
                        '-150 < rake < -30: Normal (f_nm=1)',
                        '  30 < rake < 150: Reverse and Rev/Obl (f_rv=1)',
                        '     otherwise:    Strike-Slip'],
                'default': None, 'unit': 'degree',
                'note': 'to get fault toggles, provide "rake" or both "f_nm", "f_rv", "f_ss", and "f_u" directly'},
        },
        "optional": {
            'f_hw': {
                'desc': ['Hanging wall toggle',
                        '1: hanging wall (r_x >= 0)',
                        '0: foot wall (r_x < 0)'],
                'default': None, 'unit': 'km', 'note': 'determined from "r_x" if not provided'},
            'f_nm': {
                'desc': ['Normal fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_rv': {
                'desc': ['Reverse fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_region': {
                'desc': ['Region to analyze',
                        '0: Global',
                        '1: Japan/Italy',
                        '2: Wenchuan (only applicable for M7.9 event)'],
                'default': 0, 'unit': None, 'note': None},
        },
    }
    
    # Other parameters
    _MODEL_INPUT_OTHER = {
        "desc": 'Other inputs for model:',
        'required': {
        },
        "optional": {
            'period_out': {
                'desc': ['Periods to return (list or single str, int, float), e.g.',
                        'PGA: provide "PGA", "pga", or "0"',
                        'PGV: provide "PGV", "pgv", or "-1"',
                        'Sa(T): [-1, 0, 0.2, 0.4, 0.6, 2.0, 6.0]'],
                'default': [
                    0, -1,
                    0.01, 0.02, 0.03, 0.05, 0.075,
                    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
                    1, 1.5, 2, 3, 4, 5, 7.5, 10
                ], 'unit': None, 'note': None},
        },
    }
    

    @staticmethod
    @njit(
        # Tuple((float64[:,:],float64[:,:],float64[:,:]))(
        #     float64[:],float64[:],float64[:],int64,
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],
        #     float64[:],float64[:],int64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:]
        # ),
        fastmath=True,
        cache=True
    )
    def _model(
        # site params
        vs30, vs30_source, z1p0, f_region, 
        # eq params
        mag, dip, rake, f_rv, f_nm, f_hw, r_rup, r_jb, r_x, r_y0, z_tor, z_bor, 
        # coeffs
        Period, M1, Vlin, b, c, c4, a1, a2, a3, a4, a5, a6, 
        a8, a10, a11, a12, a13, a14, a15, a17, a43, a44, a45, 
        a46, a25, a28, a29, a31, a36, a37, a38, a39, a40, a41, 
        a42, s1e, s2e, s3, s4, s1m, s2m, s5, s6
    ):
        
        # print(typeof(vs30))
        # print(typeof(vs30_source))
        # print(typeof(z1p0))
        # print(typeof(f_region))
        
        # print(typeof(mag))
        # print(typeof(dip))
        # print(typeof(rake))
        # print(typeof(f_rv))
        # print(typeof(f_nm))
        # print(typeof(f_hw))
        # print(typeof(r_rup))
        # print(typeof(r_jb))
        # print(typeof(r_x))
        # print(typeof(r_y0))
        # print(typeof(z_tor))
        # print(typeof(z_bor))
        
        
        # print(typeof(M1))
        # print(typeof(Vlin))
        # print(typeof(b))
        # print(typeof(c))
        # print(typeof(c4))
        # print(typeof(a1))
        # print(typeof(a2))
        # print(typeof(a3))
        # print(typeof(a4))
        # print(typeof(a5))
        # print(typeof(a6))
        
        
        #-------------------------------------------
        # constants
        M2 = 5
        n = 1.5
        h1 = 0.25
        h2 = 1.5
        h3 = -0.75
        a2hw = 0.2
        phi_amp = 0.4
        a7 = 0
        vs30_1180 = 1180 # m/s
        
        #-------------------------------------------
        # term: aftershock - eq. 20
        # not coded in here
        f11 = 0
        
        #-------------------------------------------
        # dimensions
        n_site = len(mag)
        n_period = len(Period)
        shape = (n_period, n_site)
        
        #-------------------------------------------
        # preset output matrices
        # ln_y = np.zeros(shape)
        # tau = np.zeros(shape)
        # phi = np.zeros(shape)
        ln_y = np.empty(shape)
        tau = np.empty(shape)
        phi = np.empty(shape)
        
        #-------------------------------------------
        # precompute some terms
        dip_rad = np.radians(dip)
        cos_dip = np.cos(dip_rad)
        sin_dip = np.sin(dip_rad)
        
        #-------------------------------------------
        # determine fault type toggles
        # find where f_nm and f_rv are not given
        ind_f_nm_is_nan = np.where(np.isnan(f_nm))[0]
        ind_f_rv_is_nan = np.where(np.isnan(f_rv))[0]
        
        # find fault types based on rake angles
        # -150 < rake < -30: Normal
        #   30 < rake < 150: Reverse and Rev/Ob
        #      otherwise:    Strike-Slip and Nml/Obl
        full_ind_list = np.arange(n_site)
        ind_rake_nm = np.where(np.logical_and(rake>=-150,rake<=-30))[0] # normal
        ind_rake_rv = np.where(np.logical_and(rake>=30,rake<=150))[0] # reverse
        # set f_nm = 1 for cases inferred to be normal from rake
        if len(ind_f_nm_is_nan)>0:
            f_nm[ind_f_nm_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_nm).intersection(set(ind_f_nm_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_nm[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # set f_rv = 1 for cases inferred to be reverse from rake
        if len(ind_f_rv_is_nan)>0:
            f_rv[ind_f_rv_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_rv).intersection(set(ind_f_rv_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_rv[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # convert to integers
        f_nm = f_nm.astype(np.int64)
        f_rv = f_rv.astype(np.int64)
        
        #-------------------------------------------
        # determine hanging wall toggle
        # find where f_jw is not given
        ind_f_hw_is_nan = np.where(np.isnan(f_hw))[0]

        # get hanging wall toggle based on r_x
        # 1 (hanging wall): r_x >= 0
        # 0 (foot wall): r_x < 0
        full_ind_list = np.arange(n_site)
        ind_hanging = np.where(r_x>0)[0] # hanging
        # set f_hw = 1 for cases inferred to be normal from rake
        if len(ind_f_hw_is_nan)>0:
            f_hw[ind_f_hw_is_nan] = 0
            ind = list(set(ind_hanging).intersection(set(ind_f_hw_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_hw[ind] = 1
        # convert to integers
        f_hw = f_hw.astype(np.int64)
        
        #-------------------------------------------
        # calculate expected value for z1.0 - eq 18-19
        # Japan only, eq. 19
        if f_region == 1:
            numer = vs30**2 + 412**2
            denom = 1360**2 + 412**2
            z1ref = np.exp(-5.23/2 * np.log(numer/denom))/1000 # km
        # California - eq. 18 (for now expanding to other regions too)
        else:
            numer = vs30**4 + 610**4
            denom = 1360**4 + 610**4
            z1ref = np.exp(-7.67/4 * np.log(numer/denom))/1000 # km
        # also find where z1p0 = nan (not given)
        # set z1p0 to z1p0_mean if z1p0 is nan
        z1p0[np.isnan(z1p0)] = z1ref[np.isnan(z1p0)] # default input unit=km
        
        #-------------------------------------------
        # precompute terms for: hanging wall
        # eq 11
        T1 = (90-dip)/45
        T1[dip<=30] = 60/45
        # eq 12
        T2 = 1+a2hw*(mag-6.5) - (1-a2hw)*(mag-6.5)**2
        T2[mag<=5.5] = 0
        T2[mag>=6.5] = 1 + a2hw*(mag[mag>=6.5]-6.5)
        # eq 13
        width = (z_bor-z_tor)/sin_dip
        R1 = width*cos_dip
        R2 = 3*R1
        Ry1 = r_x*np.tan(np.radians(20))
        T3 = 1 - ((r_x - R1)/(R2 - R1))
        T3[r_x>R2] = 0
        T3[r_x<R1] = h1 + h2*(r_x[r_x<R1]/R1[r_x<R1]) + h3*(r_x[r_x<R1]/R1[r_x<R1])**2
        # eq 14
        T4 = 1 - z_tor**2 / 100
        T4[z_tor>10] = 0
        # to simplify checks for where r_y0 is available, first calculate T15 with eq 15b
        T5 = 1 - r_jb/30
        T5[r_jb>=30] = 0
        T5[r_jb==0] = 1
        # now continue with eq 15a to update T15 array where r_y0 is available
        full_ind_list = np.arange(n_site)
        ind_r_y0_is_nan = np.where(np.isnan(r_y0))[0] # where r_y0 is Nan
        ind_r_y0_is_avail =  list(set(full_ind_list).difference(set(ind_r_y0_is_nan))) # r_y0 given
        ind_r_y0_is_avail = np.asarray(ind_r_y0_is_avail, dtype=np.int64)
        if len(ind_r_y0_is_avail) > 0:
            # first calculate middle branch with 2 logical conditions
            T5[ind_r_y0_is_avail] = 1 - (r_y0[ind_r_y0_is_avail]-Ry1[ind_r_y0_is_avail])/5
            T5[(r_y0-Ry1)>=5] = 0
            T5[(r_y0-Ry1)<=0] = 1
        
        
        # print('vs30', vs30)
        # print('z1p0', z1p0)
        # print('mag', mag)
        # print('dip', dip)
        # print('rake', rake)
        # print('z_tor', z_tor)
        # print('z_bor', z_bor)
        # print('f_rv', f_rv)
        # print('f_nm', f_nm)
        # print('f_hw', f_hw)
        # print('r_rup', r_rup)
        # print('r_jb', r_jb)
        # print('r_x', r_x)
        # print('r_y0', r_y0)
        # print('width ', width)
        # print('vs30_source ', vs30_source)
        
        
        #-------------------------------------------
        # loop through periods
        for i in range(n_period):
            #-------------------------------------------
            # term: magnitude - eq. 2-4
            # eq 4
            c4M = c4[i] - (c4[i]-1)*(5-mag)
            c4M[mag<=4] = 1
            c4M[mag>5] = c4[i]
            # eq 3
            R = np.sqrt(r_rup**2 + c4M**2)
            # eq 2
            f1 = a1[i] + a4[i]*(mag-M1[i]) + a8[i]*(8.5-mag)**2 + \
                (a2[i]+a3[i]*(mag-M1[i]))*np.log(R) + a17[i]*r_rup
            ind = np.where(mag<M2)[0].astype(np.int64)
            f1[ind] = a1[i] + a4[i]*(M2-M1[i]) + a8[i]*(8.5-M2)**2 + a6[i]*(mag[ind]-M2) + a7*(mag[ind]-M2)**2 + \
                (a2[i]+a3[i]*(M2-M1[i]))*np.log(R[ind]) + a17[i]*r_rup[ind]
            ind = np.where(mag>M1[i])[0].astype(np.int64)
            f1[ind] = a1[i] + a5[i]*(mag[ind]-M1[i]) + a8[i]*(8.5-mag[ind])**2 + \
                (a2[i]+a3[i]*(mag[ind]-M1[i]))*np.log(R[ind]) + a17[i]*r_rup[ind]
            
            #-------------------------------------------
            # term: magnitude - eq. 5-6
            # eq 5
            f7 = a11[i] * (mag-4)
            f7[mag<4] = 0
            f7[mag>5] = a11[i]
            # eq 6
            f8 = a12[i] * (mag-4)
            f8[mag<4] = 0
            f8[mag>5] = a12[i]
            
            #-------------------------------------------
            # term: site response - eq. 7-9
            # eq 9
            if Period[i]<=0.5:
                v1 = 1500
            elif Period[i]>0.5 and Period[i]<3:
                v1 = np.exp(-0.35*np.log(Period[i]/0.5) + np.log(1500))
            else:
                v1 = 800
            # first compute for Sa at vs=1180
            # eq 8
            if vs30_1180 >= v1:
                vs30_star_1180 = v1
            else:
                vs30_star_1180 = vs30_1180
            # vs30_star_1180[vs30_1180>=v1] = v1
            # eq 7 - only evaluate first case where vs30_1180 > Vlin, as max(Vlin) = 960
            f5_1180 = (a10[i] + b[i]*n) * np.log(vs30_star_1180/Vlin[i])
                
            #-------------------------------------------
            # term: hanging wall - eq. 10-15
            # other terms are not period dependent and have been computed outside the loop
            # eq 10
            f4 = a13[i] * T1 * T2 * T3 * T4 * T5
            
            #-------------------------------------------
            # term: depth-to-top of rupture - eq. 16
            f6 = a15[i] * z_tor/20
            f6[z_tor>=20] = a15[i]
            
            #-------------------------------------------
            # term: soil depth - eq. 17
            # first compute for Sa1180
            f10_1180 = np.zeros(n_site) # z1 = z1ref, and ln(1) = 0
            
            #-------------------------------------------
            # term: regionalization - eq. 21-23
            # first compute for Sa1180
            if f_region == 0: # global
                regional_1180 = np.zeros(n_site)
            else:
                if f_region == 1: # Japan
                    f13_1180 = a42[i] # vs30_1180 > 1000 m/s
                    regional_1180 = f13_1180 + a29[i]*r_rup
                elif f_region == 2: # China
                    regional_1180 = a28[i]*r_rup
                elif f_region == 3: # Taiwan
                    f12_1180 = a31[i] * np.log(vs30_star_1180/Vlin[i])
                    regional_1180 = f12_1180 + a25[i]*r_rup

            #-------------------------------------------
            # compute median PGA on rock - eq 1
            sa_1180 = np.exp(f1 + f_rv*f7 + f_nm*f8 + f5_1180 + f_hw*f4 + f6 + f10_1180 + regional_1180)
            
            #-------------------------------------------
            # term: site response - eq. 7-9
            # adjust for site Vs30
            # eq 8
            vs30_star = vs30
            vs30_star[vs30>=v1] = v1
            # eq 7
            f5 = a10[i]*np.log(vs30_star/Vlin[i]) - b[i]*np.log(sa_1180+c[i]) + b[i]*np.log(sa_1180+c[i]*(vs30_star/Vlin[i])**n)
            f5[vs30>=Vlin[i]] = (a10[i] + b[i]*n) * np.log(vs30_star[vs30>=Vlin[i]]/Vlin[i])
            
            #-------------------------------------------
            # term: soil depth - eq. 17
            # adjust for site Vs30
            f10 = a43[i] * np.log((z1p0+0.01) / (z1ref+0.01))
            ind = np.where(np.logical_and(vs30>200, vs30<=300))[0].astype(np.int64)
            f10[ind] = a44[i] * np.log((z1p0[ind]+0.01) / (z1ref[ind]+0.01))
            ind = np.where(np.logical_and(vs30>300, vs30<=500))[0].astype(np.int64)
            f10[ind] = a45[i] * np.log((z1p0[ind]+0.01) / (z1ref[ind]+0.01))
            f10[vs30>500] = a46[i] * np.log((z1p0[vs30>500]+0.01) / (z1ref[vs30>500]+0.01))
            
            #-------------------------------------------
            # term: regionalization - eq. 21-23
            # adjust for site Vs30
            if f_region == 0: # global
                regional = np.zeros(n_site)
            else:
                if f_region == 1: # Japan
                    f13 = np.ones(n_site)*a36[i]
                    ind = np.where(np.logical_and(vs30>=200,vs30<300))[0].astype(np.int64)
                    f13[ind] = a37[i]
                    ind = np.where(np.logical_and(vs30>=300,vs30<400))[0].astype(np.int64)
                    f13[ind] = a38[i]
                    ind = np.where(np.logical_and(vs30>=400,vs30<500))[0].astype(np.int64)
                    f13[ind] = a39[i]
                    ind = np.where(np.logical_and(vs30>=500,vs30<700))[0].astype(np.int64)
                    f13[ind] = a40[i]
                    ind = np.where(np.logical_and(vs30>=700,vs30<1000))[0].astype(np.int64)
                    f13[ind] = a41[i]
                    f13[vs30>=1000] = a42[i]
                    regional = f13 + a29[i]*r_rup 
                elif f_region == 2: # China
                    regional = a28[i]*r_rup
                elif f_region == 3: # Taiwan
                    f12 = a31[i] * np.log(vs30_star/Vlin[i])
                    regional = f12 + a25[i]*r_rup
            
            #-------------------------------------------
            # compute median PGA for site - eq 1
            ln_y[i] = f1 + f_rv*f7 + f_nm*f8 + f5 + f_hw*f4 + f6 + f10 + regional
            
            #-------------------------------------------
            # aleatory variability - eq. 24-30
            # first get linear standard deviations
            # eq 25
            tau_A_L = s3[i] + (s4[i]-s3[i])/2 * (mag-5)
            tau_A_L[mag<5] = s3[i]
            tau_A_L[mag>7] = s4[i]
            # if Japan
            if f_region == 1:
                # eq 26
                phi_A_L = s5[i] + (s6[i]-s5[i])/50 * (r_rup-30)
                phi_A_L[r_rup<30] = s5[i]
                phi_A_L[r_rup>80] = s6[i]
            else:
                # get s1 and s2 to use
                s1 = s1e[i] * np.ones(vs30_source.shape) # default
                s2 = s2e[i] * np.ones(vs30_source.shape) # default
                s1[vs30_source==1] = s1m[i] # use measured if vs30_source == 1
                s2[vs30_source==1] = s2m[i] # use measured if vs30_source == 1
                # eq 24
                phi_A_L = s1 + (s2-s1)/2 * (mag-4)
                phi_A_L[mag<4] = s1[mag<4]
                phi_A_L[mag>6] = s2[mag>6]
            # calculate amplification
            amp_rate = -b[i]*sa_1180/(sa_1180+c[i]) + b[i]*sa_1180/(sa_1180+c[i]*(vs30/Vlin[i])**n)
            amp_rate[vs30>=Vlin[i]] = 0
            # next get nonlinear standard deviations
            phi_B = np.sqrt(phi_A_L**2 - phi_amp**2)
            tau_B = tau_A_L
            phi[i] = np.sqrt(phi_B**2*(1+amp_rate)**2 + phi_amp**2)
            tau[i] = tau_B*(1+amp_rate)
        
        # return
        return ln_y, phi, tau
   
       
# ----------------------------------------------------------- 
class BSSA14(GMPE):
    """
    Boore, Stewart, Seyhan, and Atkinson (2014)
    
    Parameters
    ----------
    
    Returns
    -------
        
    References
    ----------
    .. [1] Boore, D.M., Stewart, J.P., Seyhan, E., and Atkinson, G.M., 2014, 
    NGA-West2 Equations for Predicting PGA, PGV, and 5% Damped PSA for Shallow
    Crustal Earthquakes, Earthquake Spectra, vol. 30, no. 3, pp. 1057-1085.
    
    """
    
    # class definitions
    _NAME = 'Boore, Stewart, Seyhan, and Atkinson (2014)'   # Name of the model
    _ABBREV = 'BSSA14'                 # Abbreviated name of the model
    _REF = "".join([                 # Reference for the model
        'Boore, D.M., Stewart, J.P., Seyhan, E., and Atkinson, G.M., 2014, ',
        'NGA-West2 Equations for Predicting PGA, PGV, and 5% Damped PSA for Shallow Crustal Earthquakes, ',
        'Earthquake Spectra, ',
        'vol. 30, no. 3, pp. 1057-1085.'
    ])
    # Model inputs
    # Site variables
    _MODEL_INPUT_SITE = {
        "desc": 'Site inputs for model (single values only, i.e., one site per run):',
        'required': {
            'vs30': {
                'desc': 'Vs30 (m/s)',
                'default': None, 'unit': 'm/s', 'note': None},
        },
        "optional": {
            'z1p0': {
                'desc': 'Depth to Vs=1km/s (km)',
                'default': None, 'unit': 'km', 'note': None},
        },
    }
    
    # Earthquake scenario variables
    _MODEL_INPUT_EQ = {
        "desc": 'EQ inputs for model (lists allowed for multiple events):',
        'required': {
            'mag': {
                'desc': 'Moment magnitude',
                'default': None, 'unit': None, 'note': None},
            'r_jb': {
                'desc': 'Closest distance to surface projection of coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'rake': {
                'desc': ['Rake angle (deg)',
                        '-150 < rake < -30: Normal and Nml/Obl (f_nm=1)',
                        '  30 < rake < 150: Reverse and Rev/Obl (f_rv=1)',
                        '     otherwise:    Strike-Slip (f_ss=1)',
                        '    unspecified:   Unspecified (f_u=1)'],
                'default': None, 'unit': 'degree',
                'note': 'to get fault toggles, provide "rake" or both "f_nm", "f_rv", "f_ss", and "f_u" directly'},
        },
        "optional": {
            'f_nm': {
                'desc': ['Normal fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_rv': {
                'desc': ['Reverse fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_ss': {
                'desc': ['Strike-slip fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_u': {
                'desc': ['Unspecified fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_region': {
                'desc': ['Region to analyze',
                        '0: Global',
                        '1: Japan/Italy',
                        '2: Wenchuan (only applicable for M7.9 event)'],
                'default': 0, 'unit': None, 'note': None},
        },
    }
    
    # Other parameters
    _MODEL_INPUT_OTHER = {
        "desc": 'Other inputs for model:',
        'required': {
        },
        "optional": {
            'period_out': {
                'desc': ['Periods to return (list or single str, int, float), e.g.',
                        'PGA: provide "PGA", "pga", or "0"',
                        'PGV: provide "PGV", "pgv", or "-1"',
                        'Sa(T): [-1, 0, 0.2, 0.4, 0.6, 2.0, 6.0]'],
                'default': [
                    0, -1,
                    0.01, 0.02, 0.03, 0.05, 0.075,
                    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
                    1, 1.5, 2, 3, 4, 5, 7.5, 10
                ], 'unit': None, 'note': None},
        },
    }
    
    
    @staticmethod
    @njit(
        # Tuple((float64[:,:],float64[:,:],float64[:,:]))(
        #     float64[:],float64[:],int64,
        #     float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:],float64[:],float64[:],float64[:],
        #     float64[:],float64[:]
        # ),
        fastmath=True,
        cache=True
    )
    def _model(
        # site params
        vs30, z1p0, f_region, 
        # eq params
        mag, rake, f_rv, f_nm, f_ss, f_u, r_jb, 
        # coeffs
        Period, e0, e1, e2, e3, e4, e5, e6, Mh, c1, c2, c3, Mref, Rref, h, 
        Dc3Global, Dc3ChinaTrk, Dc3ItalyJapan, c, Vc, Vref, f1, f3, f4, f5, 
        f6, f7, R1, R2, DfR, DfV, V1, V2, l1, l2, t1, t2
    ):
        
        #-------------------------------------------
        # dimensions
        n_site = len(mag)
        n_period = len(Period)
        shape = (n_period, n_site)
        
        #-------------------------------------------
        # preset output matrices
        # ln_y = np.zeros(shape)
        # tau = np.zeros(shape)
        # phi = np.zeros(shape)
        ln_y = np.empty(shape)
        tau = np.empty(shape)
        phi = np.empty(shape)
        
        #-------------------------------------------
        # adjust coefficients by region
        if f_region == 0:
            Dc3 = Dc3Global
        elif f_region == 1:
            Dc3 = Dc3ItalyJapan
        elif f_region == 2:
            Dc3 = Dc3ChinaTrk
        
        #-------------------------------------------
        # determine fault type toggles
        # find where f_nm and f_rv are not given
        ind_f_nm_is_nan = np.where(np.isnan(f_nm))[0]
        ind_f_rv_is_nan = np.where(np.isnan(f_rv))[0]
        ind_f_ss_is_nan = np.where(np.isnan(f_ss))[0]
        ind_f_u_is_nan = np.where(np.isnan(f_u))[0]
        
        # find fault types based on rake angles
        # -150 < rake < -30: Normal and Nml/Obl
        #   30 < rake < 150: Reverse and Rev/Ob
        #      otherwise:    Strike-Slip
        full_ind_list = np.arange(n_site)
        ind_rake_unspecified = np.where(np.isnan(rake))[0] # unspecified
        ind_rake_specified =  list(set(full_ind_list).difference(set(ind_rake_unspecified))) # specified
        # for specified rake angles, get fault type
        ind_rake_nm = np.where(np.logical_and(rake>=-150,rake<=-30))[0] # normal
        ind_rake_rv = np.where(np.logical_and(rake>=30,rake<=150))[0] # reverse
        ind_rake_ss = list(set(ind_rake_specified).difference(set(ind_rake_nm).union(set(ind_rake_rv)))) # strike-slip
        # set f_nm = 1 for cases inferred to be normal from rake
        if len(ind_f_nm_is_nan)>0:
            f_nm[ind_f_nm_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_nm).intersection(set(ind_f_nm_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_nm[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # set f_rv = 1 for cases inferred to be reverse from rake
        if len(ind_f_rv_is_nan)>0:
            f_rv[ind_f_rv_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_rv).intersection(set(ind_f_rv_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_rv[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # set f_ss = 1 for cases inferred to be strike-slip from rake
        if len(ind_f_ss_is_nan)>0:
            f_ss[ind_f_ss_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_ss).intersection(set(ind_f_ss_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_ss[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        if len(ind_f_u_is_nan)>0:
            f_u[ind_f_u_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_unspecified).intersection(set(ind_f_u_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_u[ind] = 1
        # convert to integers
        f_nm = f_nm.astype(np.int64)
        f_rv = f_rv.astype(np.int64)
        f_ss = f_ss.astype(np.int64)
        f_u = f_u.astype(np.int64)
        
        #-------------------------------------------
        # calculate expected value for z1.0 - eq 10-12
        # Japan only, eq. 12
        if f_region == 1:
            numer = vs30**2 + 412.39**2
            denom = 1360**2 + 412.39**2
            z1p0_mean = np.exp(-5.23/2 * np.log(numer/denom) - np.log(1000)) # km
        # California - eq. 11 (for now expanding to other regions too)
        else:
            numer = vs30**4 + 571.94**4
            denom = 1360**4 + 571.94**4
            z1p0_mean = np.exp(-7.15/4 * np.log(numer/denom) - np.log(1000)) # km
        # also find where z1p0 = nan (not given)
        # set z1p0 to z1p0_mean if z1p0 is nan
        z1p0[np.isnan(z1p0)] = z1p0_mean[np.isnan(z1p0)] # default input unit=km
        # delta z1p0 - eq 10
        delta_z1p0 = z1p0 - z1p0_mean # km
        
        
        #-------------------------------------------
        # loop through periods
        for i in range(n_period):
            #-------------------------------------------
            # term: source (event) - eq. 2
            f_E = e0[i]*f_u + e1[i]*f_ss + e2[i]*f_nm + e3[i]*f_rv + e4[i]*(mag-Mh[i]) + e5[i]*(mag-Mh[i])**2
            ind = np.where(mag>Mh[i])[0]
            f_E[ind] = e0[i]*f_u[ind] + e1[i]*f_ss[ind] + e2[i]*f_nm[ind] + e3[i]*f_rv[ind] + e6[i]*(mag[ind]-Mh[i])

            #-------------------------------------------
            # term: path - eq. 3-4
            # eq 4
            R = np.sqrt(r_jb**2 + h[i]**2)
            # eq 3
            f_P = (c1[i]+c2[i]*(mag-Mref[i]))*np.log(R/Rref[i]) + (c3[i]+Dc3[i])*(R-Rref[i])
            
            #-------------------------------------------
            # compute median PGA on rock - eq 1
            if Period[i] == 0: # should happen when i == 0
                pga_rock = np.exp(f_E + f_P)
            
            #-------------------------------------------
            # term: site - eq. 5-12
            # eq 6
            ln_f_lin = c[i]*np.log(vs30/Vref[i])
            ln_f_lin[vs30>Vc[i]] = c[i]*np.log(Vc[i]/Vref[i])
            # eq 8
            f2 = f4[i]*(np.exp(f5[i]*(np.minimum(vs30,760)-360)) - np.exp(f5[i]*(760-360)))
            # eq 7
            ln_f_nl = f1[i] + f2*np.log((pga_rock + f3[i]) / f3[i])
            # eq 9
            if Period[i] < 0.65:
                f_dz1 = np.zeros(n_site)
            else:
                f_dz1 = f6[i]*delta_z1p0
                f_dz1[delta_z1p0>(f7[i]/f6[i])] = f7[i]
            # eq 5
            f_S = ln_f_lin + ln_f_nl + f_dz1
            
            #-------------------------------------------
            # compute median PGA for site - eq 1
            ln_y[i] = f_E + f_P + f_S
            
            #-------------------------------------------
            # aleatory variability - eq. 13-17
            # eq 14
            tau_i = t1[i] + (t2[i]-t1[i]) * (mag-4.5)
            tau_i[mag<=4.5] = t1[i]
            tau_i[mag>=5.5] = t2[i]
            # eq 17
            phi_i_M = l1[i] + (l2[i]-l1[i]) * (mag-4.5)
            phi_i_M[mag<=4.5] = l1[i]
            phi_i_M[mag>=5.5] = l2[i]
            # eq 16
            phi_i_M_Rjb = phi_i_M + DfR[i]
            ind = np.where(np.logical_and(r_jb>R1[i],r_jb<R2[i]))[0]
            phi_i_M_Rjb[ind] = phi_i_M[ind] + DfR[i]*(np.log(r_jb[ind]/R1[i])/np.log(R2[i]/R1[i]))
            phi_i_M_Rjb[r_jb<=R1[i]] = phi_i_M[r_jb<=R1[i]]
            # phi_i_M_Rjb[r_jb>R2[i]] = phi_i_M[r_jb>R2[i]] + DfR[i]
            # eq 15
            phi_i_M_Rjb_Vs30 = phi_i_M_Rjb - DfV[i]*(np.log(V2[i]/vs30)/np.log(R2[i]/R1[i]))
            phi_i_M_Rjb_Vs30[vs30>=V2[i]] = phi_i_M_Rjb[vs30>=V2[i]]
            phi_i_M_Rjb_Vs30[vs30<=V1[i]] = phi_i_M_Rjb[vs30<=V1[i]] - DfV[i]
            # store to output
            tau[i] = tau_i
            phi[i] = phi_i_M_Rjb_Vs30
        
        # return
        return ln_y, phi, tau
    
      
# ----------------------------------------------------------- 
class CB14(GMPE):
    """
    Campbell and Bozorgnia (2014)
    
    Parameters
    ----------
    
    Returns
    -------
        
    References
    ----------
    .. [1] Campbell, K.W., and Bozorgnia, Y., 2014, NGA-West2 Ground Motion Model
    for the Average Horizontal Components of PGA, PGV, and 5% Damped Linear
    Acceleration Response Spectra, Earthquake Spectra, vol. 30, no. 3, pp. 1087-1115.
    
    """
    
    # class definitions
    _NAME = 'Campbell and Bozorgnia (2014)'   # Name of the model
    _ABBREV = 'CB14'                 # Abbreviated name of the model
    _REF = "".join([                 # Reference for the model
        'Campbell, K.W., and Bozorgnia, Y., 2014, ',
        'NGA-West2 Ground Motion Model for the Average Horizontal Components of PGA, PGV, and 5% Damped Linear Acceleration Response Spectra, ',
        'Earthquake Spectra, ',
        'vol. 30, no. 3, pp. 1087-1115.'
    ])
    # Model inputs
    # Site variables
    _MODEL_INPUT_SITE = {
        "desc": 'Site inputs for model (single values only, i.e., one site per run):',
        'required': {
            'vs30': {
                'desc': 'Vs30 (m/s)',
                'default': None, 'unit': 'm/s', 'note': None}
        },
        "optional": {
            'z2p5': {
                'desc': 'Depth to Vs=2.5km/s (km)',
                'default': None, 'unit': 'km', 'note': None},
        },
    }
    
    # Earthquake scenario variables
    _MODEL_INPUT_EQ = {
        "desc": 'EQ inputs for model (lists allowed for multiple events):',
        'required': {
            'mag': {
                'desc': 'Moment magnitude',
                'default': None, 'unit': None, 'note': None},
            'dip': {
                'desc': 'Dip angle (deg)',
                'default': None, 'unit': 'degree', 'note': None},
            'z_tor': {
                'desc': 'Depth to top of rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'z_bor': {
                'desc': 'Depth to top of rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_rup': {
                'desc': 'Closest distance to coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_jb': {
                'desc': 'Closest distance to surface projection of coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_x': {
                'desc': 'Horizontal distance from top of rupture measured perpendicular to fault strike (km)',
                'default': None, 'unit': 'km', 'note': None},
            'rake': {
                'desc': ['Rake angle (deg)',
                        '-150 < rake < -30: Normal (f_nm=1)',
                        '  30 < rake < 150: Reverse and Rev/Obl (f_rv=1)',
                        '     otherwise:    Strike-Slip and Nml/Obl'],
                'default': None, 'unit': 'degree',
                'note': 'to get fault toggles, provide "rake" or both "f_nm" and "f_rv" directly'},
        },
        "optional": {
            'z_hypo': {
                'desc': 'Depth to hypocenter (km)',
                'default': None, 'unit': 'km', 'note': None},
            'z_bot': {
                'desc': 'Depth to bottom of seismogenic crust (km)',
                'default': 15, 'unit': 'km', 'note': None},
            'f_nm': {
                'desc': ['Normal fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_rv': {
                'desc': ['Reverse fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_region': {
                'desc': ['Region to analyze',
                        '0: California',
                        '1: Japan',
                        '2: China',
                        '3: Italy'],
                'default': 0, 'unit': None, 'note': None},
        },
    }
    
    # Other parameters
    _MODEL_INPUT_OTHER = {
        "desc": 'Other inputs for model:',
        'required': {
        },
        "optional": {
            'period_out': {
                'desc': ['Periods to return (list or single str, int, float), e.g.',
                        'PGA: provide "PGA", "pga", or "0"',
                        'PGV: provide "PGV", "pgv", or "-1"',
                        'Sa(T): [-1, 0, 0.2, 0.4, 0.6, 2.0, 6.0]'],
                'default': [
                    0, -1,
                    0.01, 0.02, 0.03, 0.05, 0.075,
                    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
                    1, 1.5, 2, 3, 4, 5, 7.5, 10
                ], 'unit': None, 'note': None},
        },
    }
    
    
    @staticmethod
    @njit(
        fastmath=True,
        cache=True
    )
    def _model(
        # site params
        vs30, z2p5, f_region, 
        # eq params
        mag, dip, rake, z_tor, z_bor, z_hypo, f_rv, f_nm, r_rup, r_jb, r_x, z_bot, 
        # coeffs
        Period, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, 
        c14, c15, c16, c17, c18, c19, k1, k2, k3, a2, h1, h2, h3, h4, h5, h6, 
        c20, Dc20CA, Dc20JP, Dc20CH, t1, t2, phi1, phi2, flnaf, phic, rho
        ):        

        
        #-------------------------------------------
        # constants
        n = 1.18
        c = 1.88
        
        #-------------------------------------------
        # dimensions
        n_site = len(mag)
        n_period = len(Period)
        shape = (n_period, n_site)
        
        #-------------------------------------------
        # preset output matrices
        ln_y = np.zeros(shape)
        tau = np.zeros(shape)
        phi = np.zeros(shape)
        
        #-------------------------------------------
        # precompute some terms
        dip_rad = np.radians(dip)
        cos_dip = np.cos(dip_rad)
        sin_dip = np.sin(dip_rad)
        
        #-------------------------------------------
        # adjust coefficients by region
        if f_region == 0:
            Dc20 = Dc20CA
        elif f_region == 1 or f_region == 3:
            Dc20 = Dc20JP
        elif f_region == 2:
            Dc20 = Dc20CH
        
        #-------------------------------------------
        # determine fault type toggles
        # find where f_nm and f_rv are not given
        ind_f_nm_is_nan = np.where(np.isnan(f_nm))[0]
        ind_f_rv_is_nan = np.where(np.isnan(f_rv))[0]
        
        # find fault types based on rake angles
        # -150 < rake < -30: Normal and Nml/Obl
        #   30 < rake < 150: Reverse and Rev/Obl
        #      otherwise:    Strike-Slip
        full_ind_list = np.arange(n_site)
        ind_rake_nm = np.where(np.logical_and(rake>=-150,rake<=-30))[0] # normal
        ind_rake_rv = np.where(np.logical_and(rake>=30,rake<=150))[0] # reverse
        # set f_nm = 1 for cases inferred to be normal from rake
        if len(ind_f_nm_is_nan)>0:
            f_nm[ind_f_nm_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_nm).intersection(set(ind_f_nm_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_nm[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # set f_rv = 1 for cases inferred to be reverse from rake
        if len(ind_f_rv_is_nan)>0:
            f_rv[ind_f_rv_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_rv).intersection(set(ind_f_rv_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_rv[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # convert to integers
        f_nm = f_nm.astype(np.int64)
        f_rv = f_rv.astype(np.int64)
        
        #-------------------------------------------
        # determine z_tor from CY14 if not given (eq. 11 in CY14)
        # mean z_tor from model
        # normal and strike-slip
        z_tor_mean = np.maximum(2.673-1.136*np.maximum(mag-4.970,0),0)**2 # km, eq. 5
        # reverse and reverse/oblique
        z_tor_mean[f_rv==1] = np.maximum(2.704-1.226*np.maximum(mag[f_rv==1]-5.849,0),0)**2 # km, eq. 4
        # set z_tor to mean from model if z_tor is nan (not given)
        z_tor[np.isnan(z_tor)] = z_tor_mean[np.isnan(z_tor)] # km
        
        #-------------------------------------------
        # estimate width from eq 39
        # find where z_bor is not given
        width = np.zeros(mag.shape)
        ind_z_bor_is_nan = np.where(np.isnan(z_bor))[0].astype(np.int64)
        ind_z_bor_is_avail = np.asarray(list(set(np.arange(n_site)).difference(set(ind_z_bor_is_nan))), dtype=np.int64)
        # for where z_bor is NaN, estimate with mag and z_bot
        width[ind_z_bor_is_nan] = np.minimum(
            np.sqrt(10**((mag[ind_z_bor_is_nan]-4.07)/0.98)),
            (z_bot[ind_z_bor_is_nan] - z_tor[ind_z_bor_is_nan])/sin_dip[ind_z_bor_is_nan]
        )
        # else compute from known geometry
        width[ind_z_bor_is_avail] = (z_bor[ind_z_bor_is_avail] - z_tor[ind_z_bor_is_avail])/sin_dip[ind_z_bor_is_avail]
        
        #-------------------------------------------
        # estimate z_bor now that z_tor and width are fully defined
        z_bor[ind_z_bor_is_nan] = z_tor[ind_z_bor_is_nan] + width[ind_z_bor_is_nan]*sin_dip[ind_z_bor_is_nan]
        
        #-------------------------------------------
        # estimate z_hypo depth if unknown
        ind_z_hypo_is_nan = np.where(np.isnan(z_hypo))[0].astype(np.int64)
        if len(ind_z_hypo_is_nan) > 0:
            # eq 36
            f_dz_M = -4.317 + 0.984*mag
            f_dz_M[mag>=6.75] = 2.325
            # eq 37
            f_dz_dip = 0.0445*(dip-40)
            f_dz_dip[dip>40] = 0
            # eq 35
            ln_dz = np.minimum(f_dz_M+f_dz_dip, np.log(0.9*(z_bor-z_tor)))
            # estimate hypocenter depth
            z_hypo[ind_z_hypo_is_nan] = z_tor[ind_z_hypo_is_nan] + np.exp(ln_dz[ind_z_hypo_is_nan])
        
        #-------------------------------------------
        # estimate z2p5 if unknown
        ind_z2p5_is_nan = np.where(np.isnan(z2p5))[0]
        if len(ind_z_hypo_is_nan) > 0:
            if f_region == 1:
                z2p5[ind_z_hypo_is_nan] = np.exp(5.359 - 1.102*np.log(vs30[ind_z_hypo_is_nan]))
            else:
                z2p5[ind_z_hypo_is_nan] = np.exp(7.089 - 1.144*np.log(vs30[ind_z_hypo_is_nan]))

        #-------------------------------------------
        # precompute terms for: style of fault
        # eq 6
        f_flt_M = mag - 4.5
        f_flt_M[mag<=4.5] = 0
        f_flt_M[mag>5.5] = 1
        
        #-------------------------------------------
        # precompute terms for: hanging wall
        # eq 11
        R1 = width*cos_dip
        # eq 12
        R2 = 62*mag - 350
        # for eq 9 and 10
        r_x_over_R1 = r_x/R1
        r_x_over_R1_2 = r_x_over_R1**2
        norm_r_x = (r_x-R1)/(R2-R1)
        norm_r_x_2 = norm_r_x**2
        # eq 13
        f_hng_r_rup = (r_rup - r_jb)/r_rup
        f_hng_r_rup[r_rup==0] = 1
        # eq 15
        f_hng_Z = 1 - 0.06*z_tor
        f_hng_Z[z_tor>16.66] = 0
        # eq 16
        f_hng_dip = (90 - dip) / 45
        
        #-------------------------------------------
        # precompute terms for: hypocentral depth
        # eq 22
        f_hyp_H = z_hypo - 7
        f_hyp_H[z_hypo<=7] = 0
        f_hyp_H[z_hypo>20] = 13
        
        
        # print('vs30', vs30)
        # print('z2p5', z2p5)
        # print('mag', mag)
        # print('dip', dip)
        # print('rake', rake)
        # print('z_tor', z_tor)
        # print('z_bor', z_bor)
        # print('z_hypo', z_hypo)
        # print('f_rv', f_rv)
        # print('f_nm', f_nm)
        # print('r_rup', r_rup)
        # print('r_jb', r_jb)
        # print('r_x', r_x)
        # print('z_bot', z_bot)
        # print('width ', width )
        
        
        #-------------------------------------------
        # loop through periods
        for i in range(n_period):
            #-------------------------------------------
            # term: magnitude - eq. 2
            f_mag = c0[i] + c1[i]*mag
            ind = np.where(np.logical_and(mag>4.5,mag<=5.5))[0]
            f_mag[ind] = c0[i] + c1[i]*mag[ind] + c2[i]*(mag[ind]-4.5)
            ind = np.where(np.logical_and(mag>5.5,mag<=6.5))[0]
            f_mag[ind] = c0[i] + c1[i]*mag[ind] + c2[i]*(mag[ind]-4.5) + c3[i]*(mag[ind]-5.5)
            ind = np.where(mag>6.5)[0]
            f_mag[ind] = c0[i] + c1[i]*mag[ind] + c2[i]*(mag[ind]-4.5) + c3[i]*(mag[ind]-5.5) + c4[i]*(mag[ind]-6.5)
            
            #-------------------------------------------
            # term: geometric attenuation - eq. 3
            f_dis = (c5[i] + c6[i]*mag) * np.log(np.sqrt(r_rup**2 + c7[i]**2))
            
            #-------------------------------------------
            # term: style of fault - eq. 4-6
            # eq 5
            f_flt_F = c8[i]*f_rv + c9[i]*f_nm
            # eq 4
            f_flt = f_flt_F*f_flt_M
            
            #-------------------------------------------
            # term: hanging wall - eq.7-16
            # eq 9
            f_1 = h1[i] + h2[i]*r_x_over_R1 + h3[i]*r_x_over_R1_2
            # eq 10
            f_2 = h4[i] + h5[i]*norm_r_x + h6[i]*norm_r_x_2
            # eq 8
            f_hng_r_x = f_1
            f_hng_r_x[r_x<0] = 0
            f_hng_r_x[r_x>=R1] = np.maximum(f_2[r_x>=R1],0)
            # eq 14
            f_hng_M = (mag-5.5)*(1+a2[i]*(mag-6.5))
            f_hng_M[mag<=5.5] = 0
            f_hng_M[mag>6.5] = 1 + a2[i]*(mag[mag>6.5]-6.5)
            # eq 7
            f_hng = c10[i] * f_hng_r_x * f_hng_r_rup * f_hng_M * f_hng_Z * f_hng_dip
            
            #-------------------------------------------
            # term: shallow site response - eq.17-19
            # for rock, Vs30 = 1100 m/s > all k1 values
            # eq 17 and 18
            if Period[i] == 0: # should happen when i == 0
                f_site_rock = (c11[i] + k2[i]*n) * np.log(1100/k1[i])
                if f_region == 1:
                    # eq 17 and 19
                    f_site_rock += (c13[i] + k2[i]*n) * np.log(1100/k1[i])
            
            #-------------------------------------------
            # term: basin response - eq.20
            ###
            # should z2p5 be deteremined for rock?
            ###
            # # for rock, Vs30 = 1100 m/s > all k1 values
            # if Period[i] == 0: # should happen when i == 0
            #     # first determine z2p5 to use for rock
            #     if f_region == 1:
            #         z2p5_rock = np.exp(5.359 - 1.102*np.log(1100))
            #     else:
            #         z2p5_rock = np.exp(7.089 - 1.144*np.log(1100))
            #     # estimate f_sed
            #     if z2p5_rock <= 1:
            #         f_sed_rock = (c14[i] + c15[i]*f_region) * (z2p5_rock-1)
            #     elif z2p5_rock > 3:
            #         f_sed_rock = c16[i] * k3[i] * np.exp(-0.75) * (1 - np.exp(-0.25*(z2p5_rock-3)))
            #     else:
            #         f_sed_rock = 0
            f_sed = np.zeros(z2p5.shape)
            f_sed[z2p5<=1] = (c14[i] + c15[i]*f_region) * (z2p5[z2p5<=1]-1)
            f_sed[z2p5>3] = c16[i] * k3[i] * np.exp(-0.75) * (1 - np.exp(-0.25*(z2p5[z2p5>3]-3)))
            
            #-------------------------------------------
            # term: hypocentral depth - eq.21-23
            # eq 23
            f_hyp_M = c17[i] + (c18[i]-c17[i])*(mag-5.5)
            f_hyp_M[mag<=5.5] = c17[i]
            f_hyp_M[mag>6.5] = c18[i]
            # eq 21
            f_hyp = f_hyp_H * f_hyp_M
            
            #-------------------------------------------
            # term: fault dip - eq.24
            # eq 23
            f_dip = c19[i] * (5.5-mag)*dip
            f_dip[mag<=4.5] = c19[i]*dip[mag<=4.5]
            f_dip[mag>5.5] = 0
            
            #-------------------------------------------
            # term: anelastic attenuation - eq.25
            f_atn = (c20[i] + Dc20[i]) * (r_rup - 80)
            f_atn[r_rup<=80] = 0
            
            # print('f_mag', f_mag)
            # print('f_dis', f_dis)
            # print('f_flt', f_flt)
            # print('f_hng', f_hng)
            # print('f_site_rock', f_site_rock)
            # print('f_sed_rock', f_sed_rock)
            # print('f_hyp', f_hyp)
            # print('f_dip', f_dip)
            # print('f_atn', f_atn)
            
            #-------------------------------------------
            # compute median PGA on rock - eq 1
            if Period[i] == 0: # should happen when i == 0
                pga_rock = np.exp(
                    # f_mag + f_dis + f_flt + f_hng + f_site_rock + f_sed_rock + f_hyp + f_dip + f_atn
                    f_mag + f_dis + f_flt + f_hng + f_site_rock + f_sed + f_hyp + f_dip + f_atn
                )
            
            #-------------------------------------------
            # adjust for site Vs30
            # term: shallow site response - eq.17-19
            # eq 17 and 18
            f_site = c11[i]*np.log(vs30/k1[i]) + k2[i]*(np.log(pga_rock + c*(vs30/k1[i])**n) - np.log(pga_rock + c))
            f_site[vs30>k1[i]] = (c11[i] + k2[i]*n) * np.log(vs30[vs30>k1[i]]/k1[i])
            if f_region == 1:
                # eq 17 and 19
                f_site_J = (c13[i] + k2[i]*n) * np.log(vs30/k1[i])
                f_site_J[vs30<=200] = (c12[i] + k2[i]*n) * (np.log(vs30[vs30<=200]/k1[i]) - np.log(200/k1[i]))
                f_site += f_site_J
            
            #-------------------------------------------
            # adjust for site Vs30
            # term: basin response - eq.20
            ###
            # should z2p5 be deteremined for rock?
            ###
            # f_sed = np.zeros(z2p5.shape)
            # f_sed[z2p5<=1] = (c14[i] + c15[i]*f_region) * (z2p5[z2p5<=1]-1)
            # f_sed[z2p5>3] = c16[i] * k3[i] * np.exp(-0.75) * (1 - np.exp(-0.25*(z2p5[z2p5>3]-3)))
            
            #-------------------------------------------
            # compute median PGA for site - eq 1
            ln_y_i = f_mag + f_dis + f_flt + f_hng + f_site + f_sed + f_hyp + f_dip + f_atn
            if Period[i] < 0.25:
                ln_y_i[ln_y_i<np.log(pga_rock)] = np.log(pga_rock)[ln_y_i<np.log(pga_rock)] 
            # if Period[i] < 0.25:
            #     ln_y[i] = np.log(pga_rock)
            # else:
            #     ln_y[i] = f_mag + f_dis + f_flt + f_hng + f_site + f_sed + f_hyp + f_dip + f_atn
            ln_y[i] = ln_y_i

            #-------------------------------------------
            # aleatory variability - eq. 26-32
            # pga tau and phi
            if Period[i] == 0: # should happen when i == 0
                # eq 27
                tau_ln_pga = t2[i] + (t1[i]-t2[i]) * (5.5-mag)
                tau_ln_pga[mag<=4.5] = t1[i]
                tau_ln_pga[mag>=5.5] = t2[i]
                # eq 28
                phi_ln_pga = phi2[i] + (phi1[i]-phi2[i]) * (5.5-mag)
                phi_ln_pga[mag<=4.5] = phi1[i]
                phi_ln_pga[mag>=5.5] = phi2[i]
            # eq 27
            tau_ln_y = t2[i] + (t1[i]-t2[i]) * (5.5-mag)
            tau_ln_y[mag<=4.5] = t1[i]
            tau_ln_y[mag>=5.5] = t2[i]
            # eq 28
            phi_ln_y = phi2[i] + (phi1[i]-phi2[i]) * (5.5-mag)
            phi_ln_y[mag<=4.5] = phi1[i]
            phi_ln_y[mag>=5.5] = phi2[i]
            # eq 31
            alpha = k2[i]*pga_rock*(1/(pga_rock + c*(vs30/k1[i])**n) - 1/(pga_rock + c))
            alpha[vs30>=k1[i]] = 0
            # get correlated tau and phi for current period
            # eq 29
            tau[i] = np.sqrt(
                tau_ln_y**2 + alpha**2*tau_ln_pga**2 + 2*alpha*rho[i]*tau_ln_y*tau_ln_pga
            )
            # eq 30
            phi_ln_y_B = np.sqrt(phi_ln_y**2 - flnaf[i]**2)
            phi_ln_pga_B = np.sqrt(phi_ln_pga**2 - flnaf[i]**2)
            phi[i] = np.sqrt(
                phi_ln_y_B**2 + flnaf[i]**2 + alpha**2*phi_ln_pga_B**2 + 2*alpha*rho[i]*phi_ln_y_B*phi_ln_pga_B
            )
            
        # return
        return ln_y, phi, tau
    
    
# ----------------------------------------------------------- 
class CY14(GMPE):
    """
    Chiou and Youngs (2014)
    
    Parameters
    ----------
    
    Returns
    -------
        
    References
    ----------
    .. [1] Chiou, B.S.-J., and Youngs, R.R., 2014, Update of the Chiou and Youngs
    NGA Model for the Average Horizontal Component of Peak Ground Motion and
    Response Spectra, Earthquake Spectra, vol. 30, no. 3, pp. 1117-1153.
    
    """
    
    # class definitions
    _NAME = 'Chiou and Youngs (2022)'   # Name of the model
    _ABBREV = 'CY14'                 # Abbreviated name of the model
    _REF = "".join([                 # Reference for the model
        'Chiou, B.S.-J., and Youngs, R.R., 2014, ',
        'Update of the Chiou and Youngs NGA Model for the Average Horizontal Component of Peak Ground Motion and Response Spectra, ',
        'Earthquake Spectra, ',
        'vol. 30, no. 3, pp. 1117-1153.'
    ])
    # Model inputs
    # Site variables
    _MODEL_INPUT_SITE = {
        "desc": 'Site inputs for model (single values only, i.e., one site per run):',
        'required': {
            'vs30': {
                'desc': 'Vs30 (m/s)',
                'default': None, 'unit': 'm/s', 'note': None},
        },
        "optional": {
            'z1p0': {
                'desc': 'Depth to Vs=1km/s (km)',
                'default': None, 'unit': 'km', 'note': None},
            'vs30_source': {
                'desc': 'Source for Vs30 (inferred (0) or measured (1))',
                'default': 0, 'unit': None, 'note': None},
        },
    }
    
    # Earthquake scenario variables
    _MODEL_INPUT_EQ = {
        "desc": 'EQ inputs for model (lists allowed for multiple events):',
        'required': {
            'mag': {
                'desc': 'Moment magnitude',
                'default': None, 'unit': None, 'note': None},
            'dip': {
                'desc': 'Dip angle (deg)',
                'default': None, 'unit': 'degree', 'note': None},
            'z_tor': {
                'desc': 'Depth to top of rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_rup': {
                'desc': 'Closest distance to coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_jb': {
                'desc': 'Closest distance to surface projection of coseismic rupture (km)',
                'default': None, 'unit': 'km', 'note': None},
            'r_x': {
                'desc': 'Horizontal distance from top of rupture measured perpendicular to fault strike (km)',
                'default': None, 'unit': 'km', 'note': None},
            'rake': {
                'desc': ['Rake angle (deg)',
                        '-120 < rake < -60: Normal (f_nm=1)',
                        '  30 < rake < 150: Reverse and Rev/Obl (f_rv=1)',
                        '     otherwise:    Strike-Slip and Nml/Obl'],
                'default': None, 'unit': 'degree',
                'note': 'to get fault toggles, provide "rake" or both "f_nm" and "f_rv" directly'},
        },
        "optional": {
            'f_hw': {
                'desc': ['Hanging wall toggle',
                        '1: hanging wall (r_x >= 0)',
                        '0: foot wall (r_x < 0)'],
                'default': None, 'unit': 'km', 'note': 'determined from "r_x" if not provided'},
            'f_nm': {
                'desc': ['Normal fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_rv': {
                'desc': ['Reverse fault toggle',
                        '1: True',
                        '0: False'],
                'default': None, 'unit': None, 'note': 'see note under "rake"'},
            'f_region': {
                'desc': ['Region to analyze',
                        '0: Global',
                        '1: Japan/Italy',
                        '2: Wenchuan (only applicable for M7.9 event)'],
                'default': 0, 'unit': None, 'note': None},
            'delta_dpp': {
                'desc': 'Directivity term, direct point parameter (use 0 for median predictions)',
                'default': 0, 'unit': None, 'note': None},
        },
    }
    
    # Other parameters
    _MODEL_INPUT_OTHER = {
        "desc": 'Other inputs for model:',
        'required': {
        },
        "optional": {
            'period_out': {
                'desc': ['Periods to return (list or single str, int, float), e.g.',
                        'PGA: provide "PGA", "pga", or "0"',
                        'PGV: provide "PGV", "pgv", or "-1"',
                        'Sa(T): [-1, 0, 0.2, 0.4, 0.6, 2.0, 6.0]'],
                'default': [
                    0, -1,
                    0.01, 0.02, 0.03, 0.05, 0.075,
                    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
                    1, 1.5, 2, 3, 4, 5, 7.5, 10
                ], 'unit': None, 'note': None},
        },
    }
    
    
    @staticmethod
    @njit(
        fastmath=True,
        cache=True
    )
    def _model(
        # site params
        vs30, vs30_source, z1p0, f_region, 
        # eq params
        mag, dip, rake, z_tor, f_rv, f_nm, f_hw, r_rup, r_jb, r_x, delta_dpp, 
        # coeffs
        Period, c1, c1a, c1b, c1c, c1d, cn, cM, c2, c3, c4, c4a, cRB,
        c5, cHM, c6, c7, c7b, c8, c8a, c8b, c9, c9a, c9b, c11, c11b, 
        gamma1, gamma2, gm, phi1, phi2, phi3, phi4, phi5, phi6,
        tau1, tau2, sigma1, sigma2, sigma3, gscaleJp, gscaleWen,
        phi1Jp, phi5Jp, phi6Jp, sigma2Jp
        ):        
        
        
        #-------------------------------------------
        # dimensions
        n_site = len(mag)
        n_period = len(Period)
        shape = (n_period, n_site)
        
        #-------------------------------------------
        # preset output matrices
        ln_y = np.zeros(shape)
        tau = np.zeros(shape)
        phi = np.zeros(shape)
            
        #-------------------------------------------
        # precompute some terms
        cos_dip = np.cos(np.radians(dip))
        
        #-------------------------------------------
        # adjust coefficients for Japan region
        if f_region == 1:
            phi1 = phi1Jp
            phi5 = phi5Jp
            phi6 = phi6Jp
            sigma2 = sigma2Jp
        
        #-------------------------------------------
        # determine fault type toggles
        # find where f_nm and f_rv are not given
        ind_f_nm_is_nan = np.where(np.isnan(f_nm))[0]
        ind_f_rv_is_nan = np.where(np.isnan(f_rv))[0]
        
        # find fault types based on rake angles
        # -120 < rake < -60: Normal
        #   30 < rake < 150: Reverse and Rev/Ob
        #      otherwise:    Strike-Slip and Nml/Obl
        full_ind_list = np.arange(n_site)
        ind_rake_nm = np.where(np.logical_and(rake>=-120,rake<=-60))[0] # normal
        ind_rake_rv = np.where(np.logical_and(rake>=30,rake<=150))[0] # reverse
        # set f_nm = 1 for cases inferred to be normal from rake
        if len(ind_f_nm_is_nan)>0:
            f_nm[ind_f_nm_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_nm).intersection(set(ind_f_nm_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_nm[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # set f_rv = 1 for cases inferred to be reverse from rake
        if len(ind_f_rv_is_nan)>0:
            f_rv[ind_f_rv_is_nan] = 0 # default to zero
            ind = list(set(ind_rake_rv).intersection(set(ind_f_rv_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_rv[ind] = 1 # set to 1 only for cases inferred from rake and is nan
        # convert to integers
        f_nm = f_nm.astype(np.int64)
        f_rv = f_rv.astype(np.int64)
        
        #-------------------------------------------
        # determine hanging wall toggle
        # find where f_jw is not given
        ind_f_hw_is_nan = np.where(np.isnan(f_hw))[0]

        # get hanging wall toggle based on r_x
        # 1 (hanging wall): r_x >= 0
        # 0 (foot wall): r_x < 0
        full_ind_list = np.arange(n_site)
        ind_hanging = np.where(r_x>0)[0] # hanging
        # set f_hw = 1 for cases inferred to be normal from rake
        if len(ind_f_hw_is_nan)>0:
            f_hw[ind_f_hw_is_nan] = 0
            ind = list(set(ind_hanging).intersection(set(ind_f_hw_is_nan)))
            ind = np.asarray(ind, dtype=np.int64)
            f_hw[ind] = 1
        # convert to integers
        f_hw = f_hw.astype(np.int64)
        
        #-------------------------------------------
        # calculate expected value for z1.0
        # Japan only, eq. 2
        if f_region == 1:
            numer = vs30**2 + 412**2
            denom = 1360**2 + 412**2
            z1p0_mean = np.exp(-5.23/2 * np.log(numer/denom)) # m
        # California and non-Japan regions - eq. 1
        else:
            numer = vs30**4 + 571**4
            denom = 1360**4 + 571**4
            z1p0_mean = np.exp(-7.15/4 * np.log(numer/denom)) # m
        # also find where z1p0 = nan (not given)
        # set z1p0 to z1p0_mean if z1p0 is nan
        z1p0[np.isnan(z1p0)] = z1p0_mean[np.isnan(z1p0)]/1000 # default input unit=km
        # delta z1p0
        delta_z1p0 = z1p0*1000 - z1p0_mean # m
        
        #-------------------------------------------
        # term: top of rupture - eq. 11
        # mean z_tor from model
        # normal and strike-slip
        z_tor_mean = np.maximum(2.673-1.136*np.maximum(mag-4.970,0),0)**2 # km, eq. 5
        # reverse and reverse/oblique
        z_tor_mean[f_rv==1] = np.maximum(2.704-1.226*np.maximum(mag[f_rv==1]-5.849,0),0)**2 # km, eq. 4
        # set z_tor to mean from model if z_tor is nan (not given)
        z_tor[np.isnan(z_tor)] = z_tor_mean[np.isnan(z_tor)] # km
        # delta z_tor
        delta_z_tor = z_tor - z_tor_mean
        
        #-------------------------------------------
        # inferred vs measured vs30
        f_measured = np.zeros(vs30_source.shape)
        f_measured[vs30_source==1] = 1
        f_inferred = np.ones(f_measured.shape) - f_measured
        
        #-------------------------------------------
        # loop through periods
        for i in range(n_period):
            #-------------------------------------------
            # term: fault type - eq. 11
            # reverse
            term_fault_rv = c1a[i] + c1c[i]/np.cosh(2*np.maximum(mag-4.5,0))
            term_fault_rv = term_fault_rv*f_rv
            # normal
            term_fault_nm = c1b[i] + c1d[i]/np.cosh(2*np.maximum(mag-4.5,0))
            term_fault_nm = term_fault_nm*f_nm

            #-------------------------------------------
            # term: top of rupture - eq. 11
            term_z_tor = c7[i] + c7b[i]/np.cosh(2*np.maximum(mag-4.5,0))
            term_z_tor = term_z_tor*delta_z_tor
            
            #-------------------------------------------
            # term: dip - eq. 11
            term_dip = c11[i] + c11b[i]/np.cosh(2*np.maximum(mag-4.5,0))
            term_dip = term_dip * cos_dip**2
            
            #-------------------------------------------
            # term: magnitude scaling - eq. 11
            term_mag_1 = c1[i]
            term_mag_2 = c2[i]*(mag-6)
            term_mag_3 = (c2[i]-c3[i])/cn[i]*np.log(1 + np.exp(cn[i]*(cM[i]-mag)))
            term_mag = term_mag_1 + term_mag_2 + term_mag_3
            
            #-------------------------------------------
            # term: near-field magnitude and distance scaling - eq. 11
            cc = c5[i] * np.cosh(c6[i]*np.maximum(mag-cHM[i],0))
            term_near_field = c4[i] * np.log(r_rup + cc)
            
            #-------------------------------------------
            # term: large distance scaling - eq. 11
            gamma = gamma1[i] + gamma2[i]/np.cosh(np.maximum(mag-gm[i],0))
            # # correct for Japan and Wenchuan
            if f_region == 1: # Japan
                # if mag > 6 and mag < 6.9: # correct for this range of magnitudes only
                ind = np.where(np.logical_and(mag>6,mag<6.9))[0]
                gamma[ind] = gamma[ind] * gscaleJp[i]
            elif f_region == 2: # Wenchuan, single event
                gamma = gamma * gscaleWen[i]
            term_large_dist_1 = (c4a[i]-c4[i]) * np.log(np.sqrt(r_rup**2 + cRB[i]**2))
            term_large_dist_2 = gamma*r_rup
            term_large_dist = term_large_dist_1 + term_large_dist_2
            
            #-------------------------------------------
            # term: directivity - eq. 11
            term_direct_1 = c8[i]*np.maximum(1-np.maximum(r_rup-40,0)/30,0)
            term_direct_2 = \
                np.minimum(np.maximum(mag-5.5,0)/0.8,1) * \
                np.exp(-c8a[i]*(mag-c8b[i])**2)*delta_dpp
            term_direct = term_direct_1 * term_direct_2
            
            #-------------------------------------------
            # term: hanging wall - eq. 11
            term_hw = \
                c9[i]*f_hw*cos_dip * \
                (c9a[i]+(1-c9a[i])*np.tanh(r_x/c9b[i])) * \
                (1-np.sqrt(r_jb**2+z_tor**2)/(r_rup+1))
            
            #-------------------------------------------
            # population mean for Vs30 = 1130 m/s (Class B: rock) - eq. 11
            ln_y_ref = \
                term_fault_rv + term_fault_nm + term_z_tor + term_dip + \
                term_mag + term_near_field + term_large_dist + term_direct + term_hw
            
            #-------------------------------------------
            # term: site amplification, linear - eq. 12
            term_amp_lin = phi1[i] * np.minimum(np.log(vs30/1130),0)
            
            #-------------------------------------------
            # term: site amplification, nonlinear - eq. 12
            term_amp_nonlin = \
                phi2[i] * \
                (np.exp(phi3[i]*(np.minimum(vs30,1130)-360)) - np.exp(phi3[i]*(1130-360))) * \
                np.log((np.exp(ln_y_ref) + phi4[i])/phi4[i]) # NOTE: scaling with between-event residual

            #-------------------------------------------
            # term: bedrock depth - eq. 12
            term_z1p0 = phi5[i]*(1-np.exp(-delta_z1p0/phi6[i]))
            
            #-------------------------------------------
            # site-specific mean (site Vs30 dependent) - eq. 12
            ln_y[i] = ln_y_ref + term_amp_lin + term_amp_nonlin + term_z1p0
            
            #-------------------------------------------
            # aleatory variability - eq. 13
            # nonlinear adjustment
            nl0 = \
                phi2[i] * \
                (np.exp(phi3[i]*(np.minimum(vs30,1130)-360)) - np.exp(phi3[i]*(1130-360))) * \
                np.exp(ln_y_ref)/(np.exp(ln_y_ref) + phi4[i])
            # within-event phi
            phi[i] = \
                (sigma1[i] + (sigma2[i]-sigma1[i])/1.5*(np.minimum(np.maximum(mag,5),6.5)-5)) * \
                np.sqrt(sigma3[i]*f_inferred + 0.7*f_measured + (1+nl0)**2)
            # between-event tau
            tau_i = \
                (tau1[i] + (tau2[i]-tau1[i])/1.5*(np.minimum(np.maximum(mag,5),6.5)-5))
            tau[i] = tau_i*(1+nl0) # correct tau with nonlinear term
        
        # return
        return ln_y, phi, tau