# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for buckling
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import numpy as np
from scipy.stats import norm, lognorm
# from numpy import tan, radians, where
from numba import jit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class CompressionFailure(BaseModel):
    "Inherited class specfic to compressive buckling"

    _RETURN_PBEE_META = {
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        'type': 'compressive rupture',       # Type of model (e.g., liquefaction, landslide, pipe strain)
        'variable': 'rupture'        # Return variable for PBEE category, e.g., pgdef, eps_p
    }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class BainEtAl2022(CompressionFailure):
    """
    Compute probability of compressive rupture using Bain and Bray (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    eps_p: float, np.ndarray or list
       [%] compressive pipe strain
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
       [mm] pipe outside pipe_diameter
    t_pipe: float, np.ndarray or list
       [mm] ]pipe wall pipe_thickness
    sigma_y: float, np.ndarray or list
       [kPa] pipe yield stress
    op_press: float, np.ndarray or list
       [kPa] pipe internal operating pressure
    
    Fixed:
    weld_flag: boolean, np.ndarray or list
        welded (True/False)
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80
        
    Returns
    -------
    eps_crit_rupture : np.ndarray or list
        [%] mean critical compressive pipe strain for rupture
        
    References
    ----------
    .. [1] Bain, C., and Bray, J.D., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    # class definitions
    NAME = 'Bain et al. (2022)'   # Name of the model
    ABBREV = None                 # Abbreviated name of the model
    REF = "".join([                 # Reference for the model
        'Bain, C., and Bray, J.D., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_crit_rupture': {
                'desc': 'critical compressive pipe strain for rupture (%)',
                'unit': '%',
                'mean': None,
                'aleatory': 0.5,
                'epistemic': {
                    'coeff': 0.25, # base uncertainty, based on coeffcients
                    'input': None, # sigma_mu uncertainty from input parameters
                    'total': None # SRSS of coeff and input sigma_mu uncertainty
                },
                'dist_type': 'lognormal',
            }
        }
    }
    _INPUT_PBEE_META = {
        'category': 'DM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        'variable': 'eps_p'        # Input variable for PBEE category, e.g., pgdef, eps_p
    }
    _INPUT_PBEE_RV = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        "desc": 'PBEE upstream random variables:',
        'params': {
            'eps_p': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
                'mean': None,
                'aleatory': None,
                'epistemic': None,
                'dist_type': 'lognormal'
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': {
                'desc': 'pipe diameter (mm)',
                'unit': 'mm',
                'mean': {'level1': 610, 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 25, 'level2': 1, 'level3': 1},
                'low': {'level1': 102, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'high': {'level1': 1067, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'dist_type': 'normal'
            },
            't_pipe': {
                'desc': 'pipe wall thickness (mm)',
                'unit': 'mm',
                'mean': {'level1': 10.2, 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 40, 'level2': 5, 'level3': 5},
                'low': {'level1': 2.5, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'high': {'level1': 20.3, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'dist_type': 'normal'
            },
            'sigma_y': {
                'desc': 'pipe yield stress (kPa)',
                'unit': 'kPa',
                'mean': {'level1': 359000, 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 15, 'level2': 7.5, 'level3': 7.5},
                'low': {'level1': 240000, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'high': {'level1': 600000, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'dist_type': 'normal'
            },
            'op_press': {
                'desc': 'pipe internal operating pressure (kPa)',
                'unit': 'kPa',
                'mean': {'level1': 'user provided', 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 10, 'level2': 10, 'level3': 10},
                'low': {'level1': 0.1, 'level2': 0.1, 'level3': 0.1},
                'high': {'level1': 13800, 'level2': 13800, 'level3': 13800},
                'dist_type': 'normal'
            },
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'weld_flag': {
                'desc': 'welded (True/False)',
                'unit': 'unitless',
            },
            'steel_grade': {
                'desc': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
                'unit': 'unitless',
            }
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': ['op_press'],
        'level2': ['d_pipe', 't_pipe', 'sigma_y', 'op_press'],
        'level3': ['d_pipe', 't_pipe', 'sigma_y', 'op_press'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': [],
        'level2': ['weld_flag', 'steel_grade'],
        'level3': ['weld_flag', 'steel_grade'],
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    # _SUB_CLASS = None
    # OUTPUT = [                      # List of available outputs
    #     'eps_crit_rupture',
    # ]


    # instantiation
    def __init__(self):
        """Create an instance of the class"""
        super().__init__()


    # update calculation method
    def perform_calc(self):
        """Performs calculations"""
        # pull inputs locally
        # n_sample = self._inputs['n_sample']
        # n_site = self._inputs['n_site']

        # eps_p = self._check_and_convert_to_ndarray(self._inputs['eps_p'], n_sample)
        # d_pipe = self._check_and_convert_to_ndarray(self._inputs['d_pipe'], n_sample)
        # t_pipe = self._check_and_convert_to_ndarray(self._inputs['t_pipe'], n_sample)
        # sigma_y = self._check_and_convert_to_ndarray(self._inputs['sigma_y'], n_sample)
        # op_press = self._check_and_convert_to_ndarray(self._inputs['op_press'], n_sample)

        # calculations
        eps_crit_rupture = self._model(eps_p, d_pipe, t_pipe, sigma_y, op_press)
        
        # store intermediate params
        self._outputs.update({
            'eps_crit_rupture': eps_crit_rupture
        })


    @classmethod
    def _model(cls, eps_p, d_pipe, t_pipe, sigma_y, op_press):
        """Model"""
        # model coefficients
        # c0 =  1.709     # constant
        # c1 = -1.000     # ln(eps_p)
        # c2 = -1.617     # ln(d_pipe/t_pipe)

        # calculations
        sigma_h = op_press * d_pipe/t_pipe / 2 # kPa, pipe hoop stress
        # eps_p_eq = eps_p / (1 + sigma_h/sigma_y) # %, zero internal pressure equivalent compressive pipe strain
        # prob = norm.sf((c0 + c1*np.log(eps_p_eq/100) + c2*np.log(d_pipe/t_pipe)) / cls.DIST['ALEATORY']) # survival function
        eps_crit_rupture = np.log(1 + sigma_h/sigma_y) - 1.617*np.log(d_pipe/t_pipe) + 2.130
        
        # return
        # return prob
        return eps_crit_rupture