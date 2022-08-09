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
class WellheadCompressiveRupture(BaseModel):
    "Inherited class specfic to compressive rupture for wellhead subsystem components"

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
class BainEtal2022(WellheadCompressiveRupture):
    """
    Compute probability of compressive rupture of wellhead subsystem components using Bain et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Infrastructure:
    d_sys: float, np.ndarray or list
       [mm] outside pipe_diameter for subsystem components
    t_sys: float, np.ndarray or list
       [mm] wall pipe_thickness for subsystem components
    sigma_y: float, np.ndarray or list
       [kPa] yield stress for subsystem components
    op_press: float, np.ndarray or list
       [kPa] internal operating pressure for subsystem components
    
    Fixed:
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80
        
    Returns
    -------
    eps_crit_rup_sys2_4E90_jointA_xdir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint A, x-dir, pos(open)
    eps_crit_rup_sys2_4E90_jointA_xdir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint A, x-dir, neg(closed)
    eps_crit_rup_sys2_4E90_jointA_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint A, y-dir, pos(open)
    eps_crit_rup_sys2_4E90_jointA_ydir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint A, y-dir, neg(closed)
    eps_crit_rup_sys2_4TIP_jointA_xdir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, pos(open)
    eps_crit_rup_sys2_4TIP_jointA_xdir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, neg(closed)
    eps_crit_rup_sys2_4TIP_jointA_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, pos(open)
    eps_crit_rup_sys2_4TIP_jointA_ydir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, neg(closed)
    eps_crit_rup_sys3_4E90_jointB_xdir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint B, x-dir, pos(open)
    eps_crit_rup_sys3_4E90_jointB_xdir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint B, x-dir, neg(closed)
    eps_crit_rup_sys3_4E90_jointA_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint A, y-dir, pos(open)
    eps_crit_rup_sys3_4E90_jointA_ydir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 2, elbow, joint A, y-dir, neg(closed)
    eps_crit_rup_sys3_4TOP_jointA_xdir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 3, tee out-of-plane, joint A, x-dir, pos(open)
    eps_crit_rup_sys3_4TIP_jointB_xdir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, pos(open)
    eps_crit_rup_sys3_4TIP_jointB_xdir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, neg(close)
    eps_crit_rup_sys3_4TIP_jointA_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, pos(open)
    eps_crit_rup_sys3_4TIP_jointA_ydir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, neg(close)
    eps_crit_rup_sys4_4E90_jointA_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 4, elbow, joint A, y-dir, pos(open)
    eps_crit_rup_sys4_4E90_jointA_ydir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 4, elbow, joint A, y-dir, neg(close)
    eps_crit_rup_sys4_4TIP_jointA_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, pos(open)
    eps_crit_rup_sys4_4TIP_jointA_ydir_neg_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, neg(close)
    eps_crit_rup_sys4_4TOP_jointC_ydir_pos_rot : np.ndarray or list
        [%] critical strain for wellhead subsystem 4, tee out-of-plane, joint C, y-dir, pos(open)
        
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
            ##########################
            'eps_crit_rup_sys2_4E90_jointA_xdir_pos': {
                'desc': 'critical strain for wellhead subsystem 2, elbow, joint A, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys2_4E90_jointA_xdir_neg': {
                'desc': 'critical strain for wellhead subsystem 2, elbow, joint A, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys2_4E90_jointA_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 2, elbow, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys2_4E90_jointA_ydir_neg': {
                'desc': 'critical strain for wellhead subsystem 2, elbow, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_crit_rup_sys2_4TIP_jointA_xdir_pos': {
                'desc': 'critical strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys2_4TIP_jointA_xdir_neg': {
                'desc': 'critical strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys2_4TIP_jointA_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys2_4TIP_jointA_ydir_neg': {
                'desc': 'critical strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_crit_rup_sys3_4E90_jointB_xdir_pos': {
                'desc': 'critical strain for wellhead subsystem 3, elbow, joint B, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4E90_jointB_xdir_neg': {
                'desc': 'critical strain for wellhead subsystem 3, elbow, joint B, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4E90_jointA_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 3, elbow, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4E90_jointA_ydir_neg': {
                'desc': 'critical strain for wellhead subsystem 3, elbow, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_crit_rup_sys3_4TOP_jointA_xdir_pos': {
                'desc': 'critical strain for wellhead subsystem 3, tee out-of-plane, joint A, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4TIP_jointB_xdir_pos': {
                'desc': 'critical strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4TIP_jointB_xdir_neg': {
                'desc': 'critical strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4TIP_jointA_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys3_4TIP_jointA_ydir_neg': {
                'desc': 'critical strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_crit_rup_sys4_4E90_jointA_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 4, elbow, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys4_4E90_jointA_ydir_neg': {
                'desc': 'critical strain for wellhead subsystem 4, elbow, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_crit_rup_sys4_4TIP_jointA_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys4_4TIP_jointA_ydir_neg': {
                'desc': 'critical strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_crit_rup_sys4_4TOP_jointC_ydir_pos': {
                'desc': 'critical strain for wellhead subsystem 4, tee out-of-plane, joint C, y-dir, pos(open) (%)',
                'unit': '%',
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
    _N_LEVEL = 1
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_sys': 'diameter for wellhead subsystem components (mm)',
            't_sys': 'wall thickness for wellhead subsystem components (mm)',
            'sigma_y': 'yield stress for wellhead subsystem components (kPa)',
            'op_press': 'internal operating pressure for wellhead subsystem components (kPa)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'd_sys', 't_sys', 'sigma_y', 'op_press'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        # 'steel_grade'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()


    @staticmethod
    def _model(
        d_sys, t_sys, sigma_y, op_press, # infrastructure
        # steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # calculations
        sigma_h = op_press * d_sys/t_sys / 2 # kPa, pipe hoop stress
        eps_crit_rup = np.exp(-(np.log(1 + sigma_h/sigma_y) + 1.617*np.log(d_sys/t_sys) - 2.130))
        # convert to percent
        eps_crit_rup = eps_crit_rup * 100
        
        # make mean dictionary
        output_mean = {
            ##########################
            'eps_crit_rup_sys2_4E90_jointA_xdir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys2_4E90_jointA_xdir_neg': eps_crit_rup.copy(),
            'eps_crit_rup_sys2_4E90_jointA_ydir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys2_4E90_jointA_ydir_neg': eps_crit_rup.copy(),
            ##########################
            'eps_crit_rup_sys2_4TIP_jointA_xdir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys2_4TIP_jointA_xdir_neg': eps_crit_rup.copy(),
            'eps_crit_rup_sys2_4TIP_jointA_ydir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys2_4TIP_jointA_ydir_neg': eps_crit_rup.copy(),
            ##########################
            'eps_crit_rup_sys3_4E90_jointB_xdir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4E90_jointB_xdir_neg': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4E90_jointA_ydir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4E90_jointA_ydir_neg': eps_crit_rup.copy(),
            ##########################
            'eps_crit_rup_sys3_4TOP_jointA_xdir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4TIP_jointB_xdir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4TIP_jointB_xdir_neg': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4TIP_jointA_ydir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys3_4TIP_jointA_ydir_neg': eps_crit_rup.copy(),
            ##########################
            'eps_crit_rup_sys4_4E90_jointA_ydir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys4_4E90_jointA_ydir_neg': eps_crit_rup.copy(),
            ##########################
            'eps_crit_rup_sys4_4TIP_jointA_ydir_pos': eps_crit_rup.copy(),
            'eps_crit_rup_sys4_4TIP_jointA_ydir_neg': eps_crit_rup.copy(),
            'eps_crit_rup_sys4_4TOP_jointC_ydir_pos': eps_crit_rup.copy(),
        }
        
        # dimension
        ones = np.ones(d_sys.shape)
        
        # sigma
        output_sigma = {}
        output_sigma_mu = {}
        for each in output_mean:
            output_sigma[each] = ones*0.5
            output_sigma_mu[each] = ones*0.25
        
        # prepare outputs
        output = {}
        for case in output_mean:
            output[case] = {
                'mean': output_mean[case],
                'sigma': output_sigma[case],
                'sigma_mu': output_sigma_mu[case],
                'dist_type': 'lognormal',
                'unit': '%'
            }
        # get intermediate values if requested
        if return_inter_params:
            output['sigma_h'] = sigma_h
        
        # return
        return output
