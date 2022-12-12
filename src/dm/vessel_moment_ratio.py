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
class VesselMomentRatio(BaseModel):
    "Inherited class specfic to moment ratios of pressure vessels"

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PantoliEtal2022(VesselMomentRatio):
    """
    Compute ground-shaking-induced moment ratios for pressure vessels using Pantoli et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration

    Infrastructure:
    h_vessel: float, np.ndarray or list
        [m] height of pressure vessels
    h_d_ratio_vessel: float, np.ndarray or list
        height-to-diameter ratio for pressure vessels
    p_vessel: float, np.ndarray or list
        [MPa] design pressure for pressure vessels
    d_anchor: float, np.ndarray or list
        [mm] diameter for anchors

    Fixed:
    stretch_length_flag: int, np.ndarray or list
        presence of stretch length for anchors: True/False

    Returns
    -------
    moment_ratio : np.ndarray or list
        moment ratio for pressure vessel (analogous to factor of safety)
    
    References
    ----------
    .. [1] Pantoli, E. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    NAME = 'Pantoli et al. (2022)'              # Name of the model
    ABBREV = None                      # Abbreviated name of the model
    REF = "".join([                     # Reference for the model
        'Pantoli, E. et al., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'moment_ratio': {
                'desc': 'moment ratio for pressure vessel (analogous to factor of safety)',
                'unit': '',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 1
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'h_vessel': 'height of pressure vessels [m]',
            'h_d_ratio_vessel': 'height-to-diameter ratio for pressure vessels',
            'p_vessel': 'design pressure for pressure vessels [MPa]',
            'd_anchor': 'diameter for anchors [mm]',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'stretch_length_flag': 'presence of stretch length for anchors: True/False',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'h_vessel', 'h_d_ratio_vessel', 'p_vessel', 'd_anchor'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'stretch_length_flag'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {
        'no_stretch_length_model': {
            'b0':  {'mean': -13.43384  , 'sigma': 0},
            'b1':  {'mean':   1.0365665, 'sigma': 0},
            'b2':  {'mean':   5.7411092, 'sigma': 0},
            'b3':  {'mean':  -1.16954  , 'sigma': 0},
            'b4':  {'mean':   1.2319458, 'sigma': 0},
            'b5':  {'mean':  -1.512576 , 'sigma': 0},
            'b6':  {'mean':   0.0171137, 'sigma': 0},
            'b7':  {'mean':  -0.383869 , 'sigma': 0},
            'b8':  {'mean':  -0.094814 , 'sigma': 0},
            'b9':  {'mean':   0.0955814, 'sigma': 0},
            'b10': {'mean':  -0.278842 , 'sigma': 0},
        },
        'with_stretch_length_model': {
            'b0':  {'mean': -17.70944  , 'sigma': 0},
            'b1':  {'mean':   0.5571467, 'sigma': 0},
            'b2':  {'mean':   9.2071341, 'sigma': 0},
            'b3':  {'mean':  -1.296184 , 'sigma': 0},
            'b4':  {'mean':   0.90376  , 'sigma': 0},
            'b5':  {'mean':  -1.189906 , 'sigma': 0},
            'b6':  {'mean':  -0.084697 , 'sigma': 0},
            'b7':  {'mean':  -1.04548  , 'sigma': 0},
            'b8':  {'mean':   0.0978417, 'sigma': 0},
            'b9':  {'mean':  -0.031504 , 'sigma': 0},
            'b10': {'mean':  -0.254759 , 'sigma': 0},
        },
    }
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': \
            lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, \
                h_vessel, d_vessel, ln_t_vessel, d_anchor, pga: \
            np.exp(b0 + b1*np.log(pga) + b2*np.log(h_vessel) + \
            b3*np.log(d_vessel) + b4*ln_t_vessel + \
            b5*np.log(d_anchor) + b6*np.log(pga)**2 + \
            b7*np.log(h_vessel)**2 + b8*np.log(d_vessel)**2 + \
            b9*ln_t_vessel**2 + b8*np.log(d_anchor)**2)
        ,
        'sigma': {
            'no_stretch_length_model':   0.331,
            'with_stretch_length_model': 0.387,
        },
        'sigma_mu': {
            'no_stretch_length_model':   0.25,
            'with_stretch_length_model': 0.25,
        },
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
        pga, # upstream PBEE RV
        h_vessel, h_d_ratio_vessel, p_vessel, d_anchor, # infrastructure
        stretch_length_flag, # fixed/toggles
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""
        # initialize intermediate and output arrays
        moment_ratio = np.zeros(pga.shape)
        sigma_moment_ratio = np.zeros(pga.shape)
        sigma_mu_moment_ratio = np.zeros(pga.shape)

        # find indices for cases with cement
        ind_stretch_length = {
            key: stretch_length_flag==val
            for key,val in {
                'with_stretch_length_model': True,
                'no_stretch_length_model': False
            }.items() if True in (stretch_length_flag==val)
        }

        # # convert units from SI to imperial (model developed in imperial units)
        h_vessel = h_vessel/0.3048 # meter to feet
        p_vessel = p_vessel*1000/101.3*14.6959 # MPa to psi
        d_anchor = d_anchor/25.4 # mm to feet
        
        # intermediate calculations
        d_vessel = h_vessel / h_d_ratio_vessel # ft, vessel diameter
        ln_t_vessel = -7.95 + 0.934*np.log(p_vessel) + 0.968*np.log(d_vessel) # ln(inch), vessel thickness
        
        # initialize intermediate and output arrays
        cases_to_run = []
        output_mean = {}
        output_sigma = {}
        output_sigma_mu = {}
        for case in ind_stretch_length:
            # get indices for current stretch length flag
            ind = ind_stretch_length[case]
            # get sigma and sigma mu
            sigma_moment_ratio[ind] = cls._MODEL_FORM['sigma'][case]
            sigma_mu_moment_ratio[ind] = cls._MODEL_FORM['sigma_mu'][case]
            # run calc using lambda function
            moment_ratio[ind] = \
                cls._MODEL_FORM['func'](
                    # mean coefficients
                    **cls._get_mean_coeff_for_lambda_func(
                        cls._MODEL_FORM_DETAIL[case],
                        cls._MODEL_FORM['func']
                    ),
                    # inputs
                    **cls._get_kwargs_for_lambda_func(
                        locals(),
                        cls._MODEL_FORM['func'],
                        inds=ind
                    )
                )
                
        # prepare outputs
        output = {
            'moment_ratio': {
                'mean': moment_ratio,
                'sigma': sigma_moment_ratio,
                'sigma_mu': sigma_mu_moment_ratio,
                'dist_type': 'lognormal',
                'unit': ''
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['d_vessel'] = d_vessel
            output['t_vessel'] = np.exp(ln_t_vessel)
        
        # return
        return output
    
    # @staticmethod
    # def _get_kwargs_for_lambda_func(kwargs, lambda_func, inds=None):
    #     """returns dictionary with only arguments for lambda function"""
    #     if inds is None:
    #         return {key:val for key, val in kwargs.items() if key in lambda_func.__code__.co_varnames}
    #     else:
    #         out = {}
    #         print(lambda_func.__code__.co_varnames)
    #         for key, val in kwargs.items():
    #             print(key, val)
    #             if key in lambda_func.__code__.co_varnames:
    #                 out[key] = val[inds]
    #         # return {key:val[inds] for key, val in kwargs.items() if key in lambda_func.__code__.co_varnames}