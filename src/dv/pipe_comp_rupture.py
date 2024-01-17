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

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class PipeCompressiveRupture(BaseModel):
    "Inherited class specfic to compressive pipe rupture"

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class BainEtal2022(PipeCompressiveRupture):
    """
    Compute probability of compressive pipe rupture using Bain et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    # eps_pipe: float, np.ndarray or list
    #    [%] compressive pipe strain
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
       [mm] pipe outside pipe_diameter
    t_pipe: float, np.ndarray or list
       [mm] pipe wall pipe_thickness
    sigma_y: float, np.ndarray or list
       [kPa] pipe yield stress
    op_press: float, np.ndarray or list
       [kPa] pipe internal operating pressure
    
    Fixed:
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80
        
    Returns
    -------
    eps_crit_rup : np.ndarray or list
        [%] mean critical compressive pipe strain for rupture
    sigma_eps_crit_rup : float, np.ndarray
        aleatory variability for ln(sigma_eps_crit_rup)
        
    References
    ----------
    .. [1] Bain, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    # class definitions
    NAME = 'Bain et al. (2022)'   # Name of the model
    ABBREV = None                 # Abbreviated name of the model
    REF = "".join([                 # Reference for the model
        'Bain, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_comp_crit_rup': {
                'desc': 'critical compressive pipe strain for rupture (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_META = {
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {}
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': 'pipe diameter (mm)',
            't_pipe': 'pipe wall thickness (mm)',
            'sigma_y': 'pipe yield stress (kPa)',
            'op_press': 'pipe internal operating pressure (kPa)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'steel_grade': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': ['d_pipe', 't_pipe', 'op_press'],
        'level3': ['d_pipe', 't_pipe', 'op_press'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': [],
        'level2': [],
        'level3': [],
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()


    @staticmethod
    # @njit
    def _model(
        d_pipe, t_pipe, sigma_y, op_press, # infrastructure
        steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # first get sigma_y, kPa
        # Grade-B
        grade = 'Grade-B'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 241*1000 # kPa
        # Grade X-42
        grade = 'X-42'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 290*1000 # kPa
        # Grade X-52
        grade = 'X-52'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 359*1000 # kPa
        # Grade X-60
        grade = 'X-60'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 414*1000 # kPa
        # Grade X-70
        grade = 'X-70'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 483*1000 # kPa
        # Grade X-80
        grade = 'X-80'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 552*1000 # kPa
        # specifically for above ground case
        grade = 'above_ground_model'
        sigma_y[np.logical_and(steel_grade!='NA',steel_grade==grade)] = 360.4*1000 # kPa
        # if any of the params are still missing, use default grade of X-52
        sigma_y[np.isnan(sigma_y)] = 359*1000 # kPa

        # calculations
        sigma_h = op_press * d_pipe/t_pipe / 2 # kPa, pipe hoop stress
        # eps_pipe_eq = eps_pipe / (1 + sigma_h/sigma_y) # %, zero internal pressure equivalent compressive pipe strain
        # prob = norm.sf((c0 + c1*np.log(eps_pipe_eq/100) + c2*np.log(d_pipe/t_pipe)) / cls.DIST['ALEATORY']) # survival function
        eps_comp_crit_rup = np.exp(-(np.log(1 + sigma_h/sigma_y) + 1.617*np.log(d_pipe/t_pipe) - 2.130))
        
        # prepare outputs
        output = {
            'eps_comp_crit_rup': {
                'mean': eps_comp_crit_rup * 100, # convert to %
                'sigma': np.ones(eps_comp_crit_rup.shape)*0.5,
                'sigma_mu': np.ones(eps_comp_crit_rup.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            # 'eps_crit_rup': eps_crit_rup,
            # 'sigma_eps_crit_rup': sigma_eps_crit_rup,
        }
        # get intermediate values if requested
        if return_inter_params:
            output['sigma_h'] = sigma_h
        
        # return
        return output
