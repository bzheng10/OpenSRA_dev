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
# from numpy import tan, radians, where
# from numba import jit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class WellheadStrain(BaseModel):
    "Inherited class specfic to strain of wellhead subsystems"

    # _RETURN_PBEE_META = {
    #     'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'subsystem rotation',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'rot_in', # in-plane rotation of tee
    #         'rot_out', # in-plane rotation of tee
    #     ]        # Return variable for PBEE category, e.g., pgdef, eps_p
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PantoliEtal2022(WellheadStrain):
    """
    Compute rotation-induced strain on tees and elbows of wellhead subystems using Pantoli et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    rot_rup_sys2_4E90_jointA_xdir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint A, x-dir, pos(open)
    rot_rup_sys2_4E90_jointA_xdir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint A, x-dir, neg(closed)
    rot_rup_sys2_4E90_jointA_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint A, y-dir, pos(open)
    rot_rup_sys2_4E90_jointA_ydir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint A, y-dir, neg(closed)
    rot_rup_sys2_4TIP_jointA_xdir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 2, tee in-plane, joint A, x-dir, pos(open)
    rot_rup_sys2_4TIP_jointA_xdir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 2, tee in-plane, joint A, x-dir, neg(closed)
    rot_rup_sys2_4TIP_jointA_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 2, tee in-plane, joint A, y-dir, pos(open)
    rot_rup_sys2_4TIP_jointA_ydir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 2, tee in-plane, joint A, y-dir, neg(closed)
    rot_rup_sys3_4E90_jointB_xdir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint B, x-dir, pos(open)
    rot_rup_sys3_4E90_jointB_xdir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint B, x-dir, neg(closed)
    rot_rup_sys3_4E90_jointA_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint A, y-dir, pos(open)
    rot_rup_sys3_4E90_jointA_ydir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 2, elbow, joint A, y-dir, neg(closed)
    rot_rup_sys3_4TOP_jointA_xdir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 3, tee out-of-plane, joint A, x-dir, pos(open)
    rot_rup_sys3_4TIP_jointB_xdir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 3, tee in-plane, joint B, x-dir, pos(open)
    rot_rup_sys3_4TIP_jointB_xdir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 3, tee in-plane, joint B, x-dir, neg(close)
    rot_rup_sys3_4TIP_jointA_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 3, tee in-plane, joint A, y-dir, pos(open)
    rot_rup_sys3_4TIP_jointA_ydir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 3, tee in-plane, joint A, y-dir, neg(close)
    rot_rup_sys4_4E90_jointA_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 4, elbow, joint A, y-dir, pos(open)
    rot_rup_sys4_4E90_jointA_ydir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 4, elbow, joint A, y-dir, neg(close)
    rot_rup_sys4_4TIP_jointA_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 4, tee in-plane, joint A, y-dir, pos(open)
    rot_rup_sys4_4TIP_jointA_ydir_neg : np.ndarray or list
        [%] rotation for wellhead subsystem 4, tee in-plane, joint A, y-dir, neg(close)
    rot_rup_sys4_4TOP_jointC_ydir_pos : np.ndarray or list
        [%] rotation for wellhead subsystem 4, tee out-of-plane, joint C, y-dir, pos(open)

    Infrastructure:

    Fixed:
    sys_type: int, np.ndarray or list
        wellhead subsystem type: 2 for p2, 3 for p3, 4 for p4
    tee_flag: boolean, np.ndarray or list
        flag for tees in wellhead subsystem: True/False
    elbow_flag: boolean, np.ndarray or list
        flag for elbows in wellhead subsystem: True/False
    high_pressure_weight: float, np.ndarray or list
        weight to apply to high pressure model; for low pressure model, weight = 1 - high_pressure_weight

    Returns
    -------
    eps_rup_sys2_4E90_jointA_xdir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint A, x-dir, pos(open)
    eps_rup_sys2_4E90_jointA_xdir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint A, x-dir, neg(closed)
    eps_rup_sys2_4E90_jointA_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint A, y-dir, pos(open)
    eps_rup_sys2_4E90_jointA_ydir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint A, y-dir, neg(closed)
    eps_rup_sys2_4TIP_jointA_xdir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, pos(open)
    eps_rup_sys2_4TIP_jointA_xdir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, neg(closed)
    eps_rup_sys2_4TIP_jointA_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, pos(open)
    eps_rup_sys2_4TIP_jointA_ydir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, neg(closed)
    eps_rup_sys3_4E90_jointB_xdir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint B, x-dir, pos(open)
    eps_rup_sys3_4E90_jointB_xdir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint B, x-dir, neg(closed)
    eps_rup_sys3_4E90_jointA_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint A, y-dir, pos(open)
    eps_rup_sys3_4E90_jointA_ydir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 2, elbow, joint A, y-dir, neg(closed)
    eps_rup_sys3_4TOP_jointA_xdir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 3, tee out-of-plane, joint A, x-dir, pos(open)
    eps_rup_sys3_4TIP_jointB_xdir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, pos(open)
    eps_rup_sys3_4TIP_jointB_xdir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, neg(close)
    eps_rup_sys3_4TIP_jointA_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, pos(open)
    eps_rup_sys3_4TIP_jointA_ydir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, neg(close)
    eps_rup_sys4_4E90_jointA_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 4, elbow, joint A, y-dir, pos(open)
    eps_rup_sys4_4E90_jointA_ydir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 4, elbow, joint A, y-dir, neg(close)
    eps_rup_sys4_4TIP_jointA_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, pos(open)
    eps_rup_sys4_4TIP_jointA_ydir_neg : np.ndarray or list
        [%] strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, neg(close)
    eps_rup_sys4_4TOP_jointC_ydir_pos : np.ndarray or list
        [%] strain for wellhead subsystem 4, tee out-of-plane, joint C, y-dir, pos(open)
    
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
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            ##########################
            'eps_sys2_4E90_jointA_xdir_pos': {
                'desc': 'strain for wellhead subsystem 2, elbow, joint A, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys2_4E90_jointA_xdir_neg': {
                'desc': 'strain for wellhead subsystem 2, elbow, joint A, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_sys2_4E90_jointA_ydir_pos': {
                'desc': 'strain for wellhead subsystem 2, elbow, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys2_4E90_jointA_ydir_neg': {
                'desc': 'strain for wellhead subsystem 2, elbow, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_sys2_4TIP_jointA_xdir_pos': {
                'desc': 'strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys2_4TIP_jointA_xdir_neg': {
                'desc': 'strain for wellhead subsystem 2, tee in-plane, joint A, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_sys2_4TIP_jointA_ydir_pos': {
                'desc': 'strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys2_4TIP_jointA_ydir_neg': {
                'desc': 'strain for wellhead subsystem 2, tee in-plane, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_sys3_4E90_jointB_xdir_pos': {
                'desc': 'strain for wellhead subsystem 3, elbow, joint B, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys3_4E90_jointB_xdir_neg': {
                'desc': 'strain for wellhead subsystem 3, elbow, joint B, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_sys3_4E90_jointA_ydir_pos': {
                'desc': 'strain for wellhead subsystem 3, elbow, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys3_4E90_jointA_ydir_neg': {
                'desc': 'strain for wellhead subsystem 3, elbow, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_sys3_4TOP_jointA_xdir_pos': {
                'desc': 'strain for wellhead subsystem 3, tee out-of-plane, joint A, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys3_4TIP_jointB_xdir_pos': {
                'desc': 'strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys3_4TIP_jointB_xdir_neg': {
                'desc': 'strain for wellhead subsystem 3, tee in-plane, joint B, x-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_sys3_4TIP_jointA_ydir_pos': {
                'desc': 'strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys3_4TIP_jointA_ydir_neg': {
                'desc': 'strain for wellhead subsystem 3, tee in-plane, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_sys4_4E90_jointA_ydir_pos': {
                'desc': 'strain for wellhead subsystem 4, elbow, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys4_4E90_jointA_ydir_neg': {
                'desc': 'strain for wellhead subsystem 4, elbow, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            ##########################
            'eps_sys4_4TIP_jointA_ydir_pos': {
                'desc': 'strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, pos(open) (%)',
                'unit': '%',
            },
            'eps_sys4_4TIP_jointA_ydir_neg': {
                'desc': 'strain for wellhead subsystem 4, tee in-plane, joint A, y-dir, neg(close) (%)',
                'unit': '%',
            },
            'eps_sys4_4TOP_jointC_ydir_pos': {
                'desc': 'strain for wellhead subsystem 4, tee out-of-plane, joint C, y-dir, pos(open) (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            ##########################
            'rot_sys2_4E90_jointA_xdir_pos': {
                'desc': 'rotation for wellhead subsystem 2, elbow, joint A, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys2_4E90_jointA_xdir_neg': {
                'desc': 'rotation for wellhead subsystem 2, elbow, joint A, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'rot_sys2_4E90_jointA_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 2, elbow, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys2_4E90_jointA_ydir_neg': {
                'desc': 'rotation for wellhead subsystem 2, elbow, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'rot_sys2_4TIP_jointA_xdir_pos': {
                'desc': 'rotation for wellhead subsystem 2, tee in-plane, joint A, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys2_4TIP_jointA_xdir_neg': {
                'desc': 'rotation for wellhead subsystem 2, tee in-plane, joint A, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'rot_sys2_4TIP_jointA_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 2, tee in-plane, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys2_4TIP_jointA_ydir_neg': {
                'desc': 'rotation for wellhead subsystem 2, tee in-plane, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'rot_sys3_4E90_jointB_xdir_pos': {
                'desc': 'rotation for wellhead subsystem 3, elbow, joint B, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4E90_jointB_xdir_neg': {
                'desc': 'rotation for wellhead subsystem 3, elbow, joint B, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4E90_jointA_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 3, elbow, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4E90_jointA_ydir_neg': {
                'desc': 'rotation for wellhead subsystem 3, elbow, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'rot_sys3_4TOP_jointA_xdir_pos': {
                'desc': 'rotation for wellhead subsystem 3, tee out-of-plane, joint A, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4TIP_jointB_xdir_pos': {
                'desc': 'rotation for wellhead subsystem 3, tee in-plane, joint B, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4TIP_jointB_xdir_neg': {
                'desc': 'rotation for wellhead subsystem 3, tee in-plane, joint B, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4TIP_jointA_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 3, tee in-plane, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys3_4TIP_jointA_ydir_neg': {
                'desc': 'rotation for wellhead subsystem 3, tee in-plane, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'rot_sys4_4E90_jointA_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 4, elbow, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys4_4E90_jointA_ydir_neg': {
                'desc': 'rotation for wellhead subsystem 4, elbow, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'rot_sys4_4TIP_jointA_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 4, tee in-plane, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'rot_sys4_4TIP_jointA_ydir_neg': {
                'desc': 'rotation for wellhead subsystem 4, tee in-plane, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'rot_sys4_4TOP_jointC_ydir_pos': {
                'desc': 'rotation for wellhead subsystem 4, tee out-of-plane, joint C, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
        }
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
        'params': {
            'sys_type': 'wellhead subsystem type: 2 for p2, 3 for p3, 4 for p4',
            'tee_flag': 'flag for tee in wellhead subsystem: True/False',
            'elbow_flag': 'flag for elbow in wellhead subsystem: True/False',
            'high_pressure_weight': 'weight to apply to high pressure model; for low pressure model, weight = 1 - high_pressure_weight'
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'sys_type', 'elbow_flag', 'tee_flag', 'high_pressure_weight'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {
        ##########################
        'eps_4TIP_neg_lowOP': {
            'b0': {'mean': -4.698614, 'sigma': 0},
            'b1': {'mean': 0.9171584, 'sigma': 0},
        },
        'eps_4TIP_pos_lowOP': {
            'b0': {'mean': -4.846407, 'sigma': 0},
            'b1': {'mean': 1.6389036, 'sigma': 0},
        },
        'eps_4TIP_neg_highOP': {
            'b0': {'mean': -4.901993, 'sigma': 0},
            'b1': {'mean': 1.6269746, 'sigma': 0},
        },
        'eps_4TIP_pos_highOP': {
            'b0': {'mean': -4.196789, 'sigma': 0},
            'b1': {'mean': 1.8360272, 'sigma': 0},
        },
        ##########################
        'eps_4TOP_neg_lowOP': {
            'b0': {'mean': -5.672304, 'sigma': 0},
            'b1': {'mean': 1.1390712, 'sigma': 0},
        },
        'eps_4TOP_pos_lowOP': {
            'b0': {'mean': -5.672304, 'sigma': 0},
            'b1': {'mean': 1.1390712, 'sigma': 0},
        },
        'eps_4TOP_neg_highOP': {
            'b0': {'mean': -4.780363, 'sigma': 0},
            'b1': {'mean': 1.1869063, 'sigma': 0},
        },
        'eps_4TOP_pos_highOP': {
            'b0': {'mean': -4.780363, 'sigma': 0},
            'b1': {'mean': 1.1869063, 'sigma': 0},
        },
        ##########################
        'eps_4E90_neg_lowOP': {
            'b0': {'mean': -5.650503, 'sigma': 0},
            'b1': {'mean': 1.5118598, 'sigma': 0},
        },
        'eps_4E90_pos_lowOP': {
            'b0': {'mean': -5.538686, 'sigma': 0},
            'b1': {'mean': 1.0648018, 'sigma': 0},
        },
        'eps_4E90_neg_highOP': {
            'b0': {'mean': -5.675531, 'sigma': 0},
            'b1': {'mean': 1.456687, 'sigma': 0},
        },
        'eps_4E90_pos_highOP': {
            'b0': {'mean': -5.293366, 'sigma': 0},
            'b1': {'mean': 1.2205598, 'sigma': 0},
        },
    }
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': lambda b0, b1, rot: \
                np.exp(b0 + b1*np.log(rot)),
        'sigma': {
            ##########################
            'eps_4TIP_neg_lowOP':  0.085,
            'eps_4TIP_pos_lowOP':  0.051,
            'eps_4TIP_neg_highOP': 0.163,
            'eps_4TIP_pos_highOP': 0.095,
            ##########################
            'eps_4TOP_neg_lowOP':  0.093,
            'eps_4TOP_pos_lowOP':  0.093,
            'eps_4TOP_neg_highOP': 0.222,
            'eps_4TOP_pos_highOP': 0.222,
            ##########################
            'eps_4E90_neg_lowOP':  0.160,
            'eps_4E90_pos_lowOP':  0.062,
            'eps_4E90_neg_highOP': 0.137,
            'eps_4E90_pos_highOP': 0.090,
        },
        'sigma_mu': {
            'eps_4TIP_neg_lowOP':  0.25,
            'eps_4TIP_pos_lowOP':  0.25,
            'eps_4TIP_neg_highOP': 0.25,
            'eps_4TIP_pos_highOP': 0.25,
            ##########################
            'eps_4TOP_neg_lowOP':  0.25,
            'eps_4TOP_pos_lowOP':  0.25,
            'eps_4TOP_neg_highOP': 0.25,
            'eps_4TOP_pos_highOP': 0.25,
            ##########################
            'eps_4E90_neg_lowOP':  0.25,
            'eps_4E90_pos_lowOP':  0.25,
            'eps_4E90_neg_highOP': 0.25,
            'eps_4E90_pos_highOP': 0.25,
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
        ##########################
        rot_sys2_4E90_jointA_xdir_pos,
        rot_sys2_4E90_jointA_xdir_neg,
        rot_sys2_4E90_jointA_ydir_pos,
        rot_sys2_4E90_jointA_ydir_neg,
        ##########################
        rot_sys2_4TIP_jointA_xdir_pos,
        rot_sys2_4TIP_jointA_xdir_neg,
        rot_sys2_4TIP_jointA_ydir_pos,
        rot_sys2_4TIP_jointA_ydir_neg,
        ##########################
        rot_sys3_4E90_jointB_xdir_pos,
        rot_sys3_4E90_jointB_xdir_neg,
        rot_sys3_4E90_jointA_ydir_pos,
        rot_sys3_4E90_jointA_ydir_neg,
        ##########################
        rot_sys3_4TOP_jointA_xdir_pos,
        rot_sys3_4TIP_jointB_xdir_pos,
        rot_sys3_4TIP_jointB_xdir_neg,
        rot_sys3_4TIP_jointA_ydir_pos,
        rot_sys3_4TIP_jointA_ydir_neg,
        ##########################
        rot_sys4_4E90_jointA_ydir_pos,
        rot_sys4_4E90_jointA_ydir_neg,
        ##########################
        rot_sys4_4TIP_jointA_ydir_pos,
        rot_sys4_4TIP_jointA_ydir_neg,
        rot_sys4_4TOP_jointC_ydir_pos, # upstream PBEE RV
        ##########################
        sys_type, elbow_flag, tee_flag,
        high_pressure_weight=0.7, # fixed/toggles
        ##########################
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""
        # get dim
        length = len(sys_type)

        # cases
        joints = {
            'sys2': ['A'],
            'sys3': ['A','B'],
            'sys4': ['A','B','C'],
        }
        rot_type = {
            'elbow': ['4E90'],
            # 'tee': ['4TIP','4TOP'],
            'tee': ['4TOP','4TIP'],
        }
        
        # determine cases
        ind_sys_type = {
            num: sys_type==num
            for num in [2, 3, 4]
        }
        ind_elbow = elbow_flag==True
        ind_tee = tee_flag==True
        
        # initialize intermediate and output arrays
        output_mean = {}
        output_sigma = {}
        output_sigma_mu = {}
        
        # loop through possible branches
        for num in [2,3,4]:
            for comp in ['elbow','tee']:
                ind_zero_strain = np.logical_or(sys_type!=num,locals()[f'{comp}_flag']==False)
                # initialize dictionary for current system + component
                base_str = f"eps_sys{num}_{comp}"
                # additional loops for rotation model
                for rot in rot_type[comp]:
                    # see if num is in 
                    for joint in joints[f'sys{num}']:
                        for d in ['x','y']:
                            for p in ['neg','pos']:
                                case = f"sys{num}_{rot}_joint{joint}_{d}dir_{p}"
                                rot_case = f'rot_{case}'
                                # case exists as an input, run rotation through both high and low OP models
                                if rot_case in locals():
                                    # append rotation case name
                                    # first compute weighted high operating pressure case
                                    eps_model_case = f"eps_{rot}_{p}_highOP"
                                    output_mean['eps_'+case] = \
                                        cls._MODEL_FORM['func'](
                                            # mean coefficients
                                            **cls._get_mean_coeff_for_lambda_func(
                                                cls._MODEL_FORM_DETAIL[eps_model_case],
                                                cls._MODEL_FORM['func']
                                            ),
                                            rot=locals()[rot_case]
                                        ) ** high_pressure_weight
                                    # next add weighted low operating pressure case
                                    eps_model_case = f"eps_{rot}_{p}_lowOP"
                                    output_mean['eps_'+case] = output_mean['eps_'+case] * \
                                        cls._MODEL_FORM['func'](
                                            # mean coefficients
                                            **cls._get_mean_coeff_for_lambda_func(
                                                cls._MODEL_FORM_DETAIL[eps_model_case],
                                                cls._MODEL_FORM['func']
                                            ),
                                            rot=locals()[rot_case]
                                        ) ** (1-high_pressure_weight)
                                    # convert to percent
                                    output_mean['eps_'+case] = output_mean['eps_'+case] * 100
                                    # for rotation cases that do not match the site conditions, set to np.exp(-10)
                                    output_mean['eps_'+case][ind_zero_strain] = 1e-10
                                    # get shape for sigmas
                                    _shape = output_mean['eps_'+case].shape
                                    
                                    ###
                                    # get sigma
                                    eps_model_case = f"eps_{rot}_{p}_highOP"
                                    output_sigma['eps_'+case] = \
                                        cls._MODEL_FORM['sigma'][eps_model_case]**2 * high_pressure_weight
                                    eps_model_case = f"eps_{rot}_{p}_lowOP"
                                    output_sigma['eps_'+case] = output_sigma['eps_'+case] + \
                                        cls._MODEL_FORM['sigma'][eps_model_case]**2 * (1-high_pressure_weight)
                                    output_sigma['eps_'+case] = np.sqrt(output_sigma['eps_'+case]) * np.ones(_shape)
                                    
                                    # get sigma_mu
                                    output_sigma_mu['eps_'+case] = np.ones(_shape)*0.25
        
        # prepare output
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
            pass
        
        # return
        return output