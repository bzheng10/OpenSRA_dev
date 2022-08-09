# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for rupture of wells
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import numpy as np
from scipy.stats import norm, lognorm
# from numpy import tan, radians, where
# from numba import jit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class ShakingInducedWellRupture(BaseModel):
    "Inherited class specfic to shaking-induced well rupture"

    # _RETURN_PBEE_META = {
    #     'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'shaking-induced well rupture',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'moment_crit_rup_conductor',
    #         'moment_crit_rup_surface',
    #         'moment_crit_rup_production',
    #         'moment_crit_rup_tubing',
    #     ] # Return variable for PBEE category, e.g., pgdef, eps_pipe
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class LuuEtal2022(ShakingInducedWellRupture):
    """
    Compute probability of shaking-induced well rupture using Luu et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Fixed:
    mode: int, np.ndarray or list
        well mode type: 1, 2, 4
    
    Returns
    -------
    moment_crit_rup_conductor : float, np.ndarray
        [N-m] mean critical moment for rupture for conductor casing
    moment_crit_rup_surface : float, np.ndarray
        [N-m] mean critical moment for rupture for surface casing
    moment_crit_rup_production : float, np.ndarray
        [N-m] mean critical moment for rupture for production casing
    moment_crit_rup_tubing : float, np.ndarray
        [N-m] mean critical moment for rupture for tubing
    
    References
    ----------
    .. [1] Luu, K. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    NAME = 'Luu et al. (2022)'              # Name of the model
    ABBREV = None                      # Abbreviated name of the model
    REF = "".join([                     # Reference for the model
        'Luu, K. et al., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'moment_crit_rup_conductor': {
                'desc': 'mean critical moment for rupture for conductor casing (N-m)',
                'unit': 'N-m',
            },
            'moment_crit_rup_surface': {
                'desc': 'mean critical moment for rupture for surface casing (N-m)',
                'unit': 'N-m',
            },
            'moment_crit_rup_production': {
                'desc': 'mean critical moment for rupture for production casing (N-m)',
                'unit': 'N-m',
            },
            'moment_crit_rup_tubing': {
                'desc': 'mean critical moment for rupture for tubing (N-m)',
                'unit': 'N-m',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'mode': 'well mode type: 1, 2, 4',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': [],
        'level3': [],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': ['mode'],
        'level2': ['mode'],
        'level3': ['mode'],
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
        mode, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # mean
        moment_crit_rup_conductor = np.empty_like(mode)
        moment_crit_rup_surface = np.empty_like(mode)
        moment_crit_rup_production = np.empty_like(mode)
        moment_crit_rup_tubing = np.empty_like(mode)
        
        # by mode
        moment_crit_rup_conductor[mode==1] = 1311917.834 # N-m
        moment_crit_rup_conductor[mode==2] = 1311917.834 # N-m
        moment_crit_rup_conductor[mode==4] = 1066185.86 # N-m
        moment_crit_rup_surface[mode==1] = 564045.0387 # N-m
        moment_crit_rup_surface[mode==2] = 300336.8095 # N-m
        moment_crit_rup_surface[mode==4] = 435757.8627 # N-m
        moment_crit_rup_production[mode==1] = 211194.8578 # N-m
        moment_crit_rup_production[mode==2] = 109907.6238 # N-m
        moment_crit_rup_production[mode==4] = 162947.3442 # N-m
        moment_crit_rup_tubing[mode==1] = 27913.05453 # N-m
        moment_crit_rup_tubing[mode==2] = 21532.92778 # N-m
        moment_crit_rup_tubing[mode==4] = 12302.58423 # N-m
        
        # prepare outputs
        output = {
            'moment_crit_rup_conductor': {
                'mean': moment_crit_rup_conductor,
                'sigma': np.ones(mode.shape)*0.2,
                'sigma_mu': np.ones(mode.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            'moment_crit_rup_surface': {
                'mean': moment_crit_rup_surface,
                'sigma': np.ones(mode.shape)*0.2,
                'sigma_mu': np.ones(mode.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            'moment_crit_rup_production': {
                'mean': moment_crit_rup_production,
                'sigma': np.ones(mode.shape)*0.2,
                'sigma_mu': np.ones(mode.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            'moment_crit_rup_tubing': {
                'mean': moment_crit_rup_tubing,
                'sigma': np.ones(mode.shape)*0.2,
                'sigma_mu': np.ones(mode.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
            # output['sigma_h'] = sigma_h
        
        # return
        return output
