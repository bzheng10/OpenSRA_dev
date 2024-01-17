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
# import logging

# data manipulation modules
import numpy as np
# from numpy import tan, radians, where
# from numba import jit
# from numba import njit

# OpenSRA modules and classes
from src.base_class import BaseModel

# -----------------------------------------------------------
class RepairRate(BaseModel):
    "Inherited class specfic to repair rates"

    def __init__(self):
        super().__init__()
    
    
# -----------------------------------------------------------
class Hazus2020(RepairRate):
    """
    Model for repair rates using Hazus (FEMA, 2020).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation
    pgv: float, np.ndarray or list
        [cm/s] peak ground velocity
        
    Geotechnical/geologic:
    
    Fixed:

    Returns
    -------
    repair_rate : float, np.ndarray
        [repairs/km] number of repairs per km of segment length
    
    References
    ----------
    .. [1] TBD.
    
    """

    _NAME = 'Hazus (FEMA, 2020)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Authors, Year, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'repair_rate': {
                'desc': 'number of repairs per km of segment length',
                'unit': 'repairs/km',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
            },
            'pgv': {
                'desc': 'peak ground velocity (cm/s)',
                'unit': 'cm/s',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
        

    @staticmethod
    # @njit
    def _model(
        pgdef, pgv, # upstream PBEE RV
        return_inter_params=False # to get intermediate params
    ):
        # convert pgdef to units for repair rate models
        pgdef = pgdef*100/2.54 # m to inch
        
        # calculate repair rates
        repair_rate_pgv = 0.3 * 0.0001 * pgv**2.25 # pgv in cm/s, repair rate in repairs/km
        repair_rate_pgdef = 0.3 * pgdef**0.56 # pgdef in inch, repair rate in repairs/km
        
        # 20% of repairs by shaking (pgv) contribute to breakage
        # 80% of repairs by deformation (pgdef) contribute to breakage
        repair_rate = repair_rate_pgv * 0.2 + repair_rate_pgdef * 0.8
        
        # avoid 0 repair_rates
        repair_rate = np.maximum(repair_rate,1e-10)
        
        # prepare outputs
        output = {
            'repair_rate': {
                'mean': repair_rate,
                'sigma': np.ones(repair_rate.shape)*0.3,
                'sigma_mu': np.ones(repair_rate.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': ''
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['repair_rate_pgv'] = repair_rate_pgv
            output['repair_rate_pgdef'] = repair_rate_pgdef
            # pass
        
        # return
        return output
    
    
# -----------------------------------------------------------
class ALA2001(RepairRate):
    """
    Model for repair rates using ALA (2001).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation
    pgv: float, np.ndarray or list
        [cm/s] peak ground velocity
        
    Geotechnical/geologic:
    
    Fixed:

    Returns
    -------
    repair_rate : float, np.ndarray
        [repairs/km] number of repairs per km of segment length
    
    References
    ----------
    .. [1] TBD.
    
    """

    _NAME = 'Hazus (FEMA, 2020)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Authors, Year, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'repair_rate': {
                'desc': 'number of repairs per km of segment length',
                'unit': 'repairs/km',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
            },
            'pgv': {
                'desc': 'peak ground velocity (cm/s)',
                'unit': 'cm/s',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'd_pipe',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'corrosivity', 'install_year',
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
        

    @staticmethod
    # @njit
    def _model(
        pgdef, pgv, # upstream PBEE RV
        d_pipe, # infrastructure
        pipe_mat, joint_type, soil_corr, # fixed
        return_inter_params=False # to get intermediate params
    ):
        # initialize arrays for k1 and k2
        k1 = np.ones(pgdef.shape)
        k2 = np.ones(pgdef.shape)
        
        cond = np.logical_and(pipe_mat == 'steel',joint_type=='riveted')
        k1[cond] = 0.7
        
        # convert parameters to units for repair rate models
        pgv = pgv/2.54 # from cm/s to inch/s
        pgdef = pgdef*100/2.54 # from m to inch
        
        # conditions for k1 and k2
        # --- to be implemented
        
        # calculate repair rates
        repair_rate_pgv = k1 * 0.00187 * pgv # pgv in inch/s, repair rate in repairs/1000 feet
        repair_rate_pgdef = k2 * 1.06 * pgdef**0.319 # pgdef in inch, repair rate in repairs/1000 feet
        
        # 20% of repairs by shaking (pgv) contribute to breakage
        # 80% of repairs by deformation (pgdef) contribute to breakage
        repair_rate = repair_rate_pgv * 0.2 + repair_rate_pgdef * 0.8
        
        # avoid 0 repair_rates
        repair_rate = np.maximum(repair_rate,1e-10)
        
        # prepare outputs
        output = {
            'repair_rate': {
                'mean': repair_rate,
                'sigma': np.ones(repair_rate.shape)*0.3,
                'sigma_mu': np.ones(repair_rate.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': ''
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['repair_rate_pgv'] = repair_rate_pgv
            output['repair_rate_pgdef'] = repair_rate_pgdef
            # pass
        
        # return
        return output