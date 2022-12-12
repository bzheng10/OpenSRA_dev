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
class ShearInducedWellRupture(BaseModel):
    "Inherited class specfic to shear-induced well rupture"

    # _RETURN_PBEE_META = {
    #     'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'shaking-induced well rupture',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'strain_crit_rup_casing',
    #         'strain_crit_rup_tubing',
    #     ] # Return variable for PBEE category, e.g., pgdef, strain_pipe
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class SasakiEtal2022(ShearInducedWellRupture):
    """
    Compute probability of shear-induced well rupture using Luu et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Returns
    -------
    strain_crit_rup_casing : float, np.ndarray
        [N-m] mean critical shear strain for rupture for production casing
    strain_crit_rup_tubing : float, np.ndarray
        [N-m] mean critical shear strain for rupture for tubing
    sigma_strain_crit_rup_casing : float, np.ndarray
        aleatory variability for ln(strain_crit_rup_casing)
    sigma_strain_crit_rup_tubing : float, np.ndarray
        aleatory variability for ln(strain_crit_rup_tubing)
    
    References
    ----------
    .. [1] Sasaki, T. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
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
            'strain_crit_rup_casing': {
                'desc': 'mean critical shear strain for rupture for production casing (%)',
                'unit': '%',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': {
                #     'coeff': 0.25, # base uncertainty, based on coeffcients
                #     'input': None, # sigma_mu uncertainty from input parameters
                #     'total': None # SRSS of coeff and input sigma_mu uncertainty
                # },
                # 'dist_type': 'lognormal',
            },
            'strain_crit_rup_tubing': {
                'desc': 'mean critical shear strain for rupture for tubing (%)',
                'unit': '%',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': {
                #     'coeff': 0.25, # base uncertainty, based on coeffcients
                #     'input': None, # sigma_mu uncertainty from input parameters
                #     'total': None # SRSS of coeff and input sigma_mu uncertainty
                # },
                # 'dist_type': 'lognormal',
            },
            # 'sigma_strain_crit_rup_casing': {
            #     'desc': 'aleatory variability for ln(strain_crit_rup_casing)',
            #     'unit': '',
            #     'mean': None,
            # },
            # 'sigma_strain_crit_rup_tubing': {
            #     'desc': 'aleatory variability for ln(strain_crit_rup_tubing)',
            #     'unit': '',
            #     'mean': None,
            # },
        }
    }
    # _INPUT_PBEE_META = {
    #     # 'category': 'DM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     # 'variable': 'strain_pipe'        # Input variable for PBEE category, e.g., pgdef, strain_pipe
    # }
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
    # _SUB_CLASS = None
    # OUTPUT = [                      # List of available outputs
    #     'strain_crit_rupture',
    # ]


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
        # strain_crit_rup_casing = np.empty_like(mode)
        # strain_crit_rup_tubing = np.empty_like(mode)
        
        # by mode
        # strain_crit_rup_casing = 4.073354318 # %
        # strain_crit_rup_tubing = 4.005356698 # %
        # strain_crit_rup_casing = np.ones(1)*4.073354318 # %
        # strain_crit_rup_tubing = np.ones(1)*4.005356698 # %
        
        # sigma
        # sigma_strain_crit_rup_casing = np.ones(1)*0.185657455
        # sigma_strain_crit_rup_tubing = np.ones(1)*0.392014819
        
        # prepare outputs
        output = {
            'strain_crit_rup_casing': {
                'mean': np.exp(4.073354318),
                'sigma': 0.185657455,
                'sigma_mu': 0.102984227,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            'strain_crit_rup_tubing': {
                'mean': np.exp(4.005356698),
                'sigma': 0.392014819,
                'sigma_mu': 0.261343213,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            # 'strain_crit_rup_casing': strain_crit_rup_casing,
            # 'strain_crit_rup_tubing': strain_crit_rup_tubing,
            # 'sigma_strain_crit_rup_casing': sigma_strain_crit_rup_casing,
            # 'sigma_strain_crit_rup_tubing': sigma_strain_crit_rup_tubing,
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
            # output['sigma_h'] = sigma_h
        
        # return
        return output
