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
# from numba import jit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class PipeTensileLeakage(BaseModel):
    "Inherited class specfic to tensile pipe leakage"

    # _RETURN_PBEE_META = {
    #     'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'rupture',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'eps_crit_rup'
    #     ] # Return variable for PBEE category, e.g., pgdef, eps_pipe
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class BainEtal2022(PipeTensileLeakage):
    """
    Compute probability of tensile pipe leakage using Bain et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    # eps_pipe: float, np.ndarray or list
    #    [%] tensile pipe strain
    
    Infrastructure:
    
    Fixed:
        
    Returns
    -------
    eps_crit_rup : np.ndarray or list
        [%] mean critical tensile pipe strain for leakage
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
            'eps_tens_crit_leak': {
                'desc': 'critical tensile pipe strain for leakage (%)',
                'unit': '%',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': 0.25, # base model uncertainty, does not include input uncertainty
                # 'dist_type': 'lognormal',
            },
            # 'sigma_eps_crit_rup': {
            #     'desc': 'aleatory variability for ln(eps_crit_rup)',
            #     'unit': '',
            #     'mean': None,
            # },
        }
    }
    _INPUT_PBEE_META = {
        # 'category': 'DM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        # 'variable': 'eps_pipe'        # Input variable for PBEE category, e.g., pgdef, eps_pipe
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            # 'eps_pipe': {
            #     'desc': 'longitudinal pipe strain (%)',
            #     'unit': '%',
            #     'mean': None,
            #     'aleatory': None,
            #     'epistemic': None,
            #     'dist_type': 'lognormal'
            # }
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
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        # 'level1': [],
        # 'level2': [],
        # 'level3': [],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        # 'level1': [],
        # 'level2': [],
        # 'level3': [],
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
        return_inter_params=False # to get intermediate params
        ):
        """Model"""
        # model coefficients

        # calculations
        eps_tens_crit_leak = 2.34 # %
        
        # prepare outputs
        output = {
            'eps_tens_crit_leak': {
                'mean': eps_tens_crit_leak,
                'sigma': 0.3,
                'sigma_mu': 0.25,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output
