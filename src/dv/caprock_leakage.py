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
import numpy as np

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class CaprockLeakage(BaseModel):
    "Inherited class specfic to caprock leakage"

    def __init__(self):
        super().__init__()
        

# -----------------------------------------------------------
class ZhangEtal2022(CaprockLeakage):
    """
    Compute total gas leakage from caprocks using Sasaki et al. (2022). This model is not dependent on any inputs
    
    Parameters
    ----------

    Returns
    -------
    leakage : float
        [kg] total gas leakage out of caprocks
    
    References
    ----------
    .. [1] Zhang, Y.-Q. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    NAME = 'Zhang et al. (2022)'              # Name of the model
    ABBREV = None                      # Abbreviated name of the model
    REF = "".join([                     # Reference for the model
        'Sasaki,  Y.-Q., et al., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'prob_leak': {
                'desc': 'probability of leakage',
                'unit': 'decimal',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': None,        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {}
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 1
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {}
    }
    _REQ_MODEL_RV_FOR_LEVEL = {}
    _REQ_MODEL_FIXED_FOR_LEVEL = {}
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': {},
        'aleatory': {},
        'func_string': {},
        'string': {}
    }
    

    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()


    @classmethod
    # @njit
    def _model(cls, 
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""
        # initialize intermediate and output arrays
        prob_leak = 0.089
        ln_leak = 5.7886219
        expected_ln_leak = ln_leak + prob_leak
        leak = np.exp(expected_ln_leak)

        # prepare outputs
        output = {
            'prob_leak': {
                # 'mean': prob_leak * 100, # convert to %
                'mean': prob_leak,
                'sigma': 1e-5,
                # 'sigma_mu': 0.0085993 * 100, # convert to %
                'sigma_mu': 0.0085993,
                'dist_type': 'normal',
                'unit': 'decimal'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            # pass
            output['leak'] = leak
            output['ln_leak'] = ln_leak
            output['expected_ln_leak'] = expected_ln_leak
        
        # return
        return output