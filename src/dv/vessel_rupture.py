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
class VesselRupture(BaseModel):
    "Inherited class specfic to shaking-induced rupture of pressure vessels"

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PantoliEtal2022(VesselRupture):
    """
    Compute probability of shaking-induced rupture of pressure vessels using Pantoli et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Infrastructure:
    
    Fixed:
        
    Returns
    -------
    moment_ratio_crit_rup : np.ndarray or list
        mean critical moment ratio for rupture of pressure vessels
        
    References
    ----------
    .. [1] Pantoli, E. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    # class definitions
    NAME = 'Pantoli et al. (2022)'              # Name of the model
    ABBREV = None                      # Abbreviated name of the model
    REF = "".join([                     # Reference for the model
        'Pantoli, E. et al., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'moment_ratio_crit_rup': {
                'desc': 'mean critical moment ratio for rupture of pressure vessels',
                'unit': '',
            },
        }
    }
    _INPUT_PBEE_META = {
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
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
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
        # mean
        moment_ratio_crit_rup = 1
        sigma_moment_ratio_crit_rup = 0.45
        
        # prepare outputs
        output = {
            'moment_ratio_crit_rup': {
                'mean': moment_ratio_crit_rup,
                'sigma': sigma_moment_ratio_crit_rup,
                'sigma_mu': 0.25,
                'dist_type': 'normal',
                'unit': ''
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output
