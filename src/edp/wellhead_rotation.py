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
from numba import njit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class WellheadRotation(BaseModel):
    "Inherited class specfic to rotations of wellhead subsystems"

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PantoliEtal2022(WellheadRotation):
    """
    Compute ground-shaking-induced rotations on tees and elbows of wellhead subystems using Pantoli et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration

    Infrastructure:
    h_tree: float, np.ndarray or list, optional
        [m] tree height, for all subsystem types
    l_p2: float, np.ndarray or list, optional
        [m] length of pipe segment S2, for subsystem types 2 and 3
    l_p6_sys23: float, np.ndarray or list, optional
        [m] length of pipe segment S6, for subsystem types 2 and 3
    l_p6_sys4: float, np.ndarray or list, optional
        [m] length of pipe segment S6, for subsystem types 4
    w_valve: float, np.ndarray or list, optional
        [kN] valve weight, for subsystem type 4

    Fixed:
    sys_type: int, np.ndarray or list
        subsystem type: 2 for p2, 3 for p3, 4 for p4
    tee_flag: boolean, np.ndarray or list
        flag for tees in subsystem: True/False
    elbow_flag: boolean, np.ndarray or list
        flag for elbows in subsystem: True/False

    Returns
    -------
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
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
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
            'h_tree': 'tree height (m), for all wellhead subsystem types',
            'l_p2': 'length of pipe segment S2 (m), for wellhead subsystem types 2 and 3',
            'l_p6_sys23': 'length of pipe segment S6 (m), for wellhead subsystem types 2 and 3',
            'l_p6_sys4': 'length of pipe segment S6 (m), for wellhead subsystem type 4',
            'w_valve': 'valve weight (kN), for wellhead subsystem type 4',
        }
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
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'h_tree', 'l_p2', 'l_p6_sys23', 'l_p6_sys4', 'w_valve'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'sys_type', 'tee_flag', 'elbow_flag'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {
        ##########################
        'rot_sys2_4E90_jointA_xdir_neg': {
            'b0': {'mean': -4.492392, 'sigma': 0},
            'b1': {'mean': -3.551781, 'sigma': 0},
            'b2': {'mean': -0.468746, 'sigma': 0},
            'b3': {'mean': -0.056979, 'sigma': 0},
            'b4': {'mean': 1.0163933, 'sigma': 0},
            'b5': {'mean': 1.5459539, 'sigma': 0},
            'b6': {'mean': 0.121702 , 'sigma': 0},
            'b7': {'mean': -0.10712 , 'sigma': 0},
            'b8': {'mean': 0.0066523, 'sigma': 0},
        },
        'rot_sys2_4E90_jointA_xdir_pos': {
            'b0': {'mean': -4.595058, 'sigma': 0},
            'b1': {'mean': -3.412715, 'sigma': 0},
            'b2': {'mean': -0.477662, 'sigma': 0},
            'b3': {'mean': -0.077827, 'sigma': 0},
            'b4': {'mean': 1.0204682, 'sigma': 0},
            'b5': {'mean': 1.5071445, 'sigma': 0},
            'b6': {'mean': 0.1226335, 'sigma': 0},
            'b7': {'mean': -0.10177 , 'sigma': 0},
            'b8': {'mean': 0.0125435, 'sigma': 0},
        },
        'rot_sys2_4E90_jointA_ydir_neg': {
            'b0': {'mean': -4.449407, 'sigma': 0},
            'b1': {'mean': -1.027858, 'sigma': 0},
            'b2': {'mean': 0.7419337, 'sigma': 0},
            'b3': {'mean': -4.04115 , 'sigma': 0},
            'b4': {'mean': 1.0210042, 'sigma': 0},
            'b5': {'mean': 0.5579471, 'sigma': 0},
            'b6': {'mean': -0.400375, 'sigma': 0},
            'b7': {'mean': 1.3897573, 'sigma': 0},
            'b8': {'mean': 0.0197555, 'sigma': 0},
        },
        'rot_sys2_4E90_jointA_ydir_pos': {
            'b0': {'mean': -4.561067, 'sigma': 0},
            'b1': {'mean': -0.925346, 'sigma': 0},
            'b2': {'mean': 0.7506968, 'sigma': 0},
            'b3': {'mean': -4.002458, 'sigma': 0},
            'b4': {'mean': 1.0196173, 'sigma': 0},
            'b5': {'mean': 0.5373218, 'sigma': 0},
            'b6': {'mean': -0.401999, 'sigma': 0},
            'b7': {'mean': 1.3701711, 'sigma': 0},
            'b8': {'mean': 0.016749 , 'sigma': 0},
        },
        ##########################
        'rot_sys2_4TIP_jointA_xdir_neg': {
            'b0': {'mean': -4.177576, 'sigma': 0},
            'b1': {'mean': -3.821444, 'sigma': 0},
            'b2': {'mean': -0.561183, 'sigma': 0},
            'b3': {'mean': -0.064491, 'sigma': 0},
            'b4': {'mean': 1.017156 , 'sigma': 0},
            'b5': {'mean': 1.6261647, 'sigma': 0},
            'b6': {'mean': 0.150721 , 'sigma': 0},
            'b7': {'mean': -0.102599, 'sigma': 0},
            'b8': {'mean': 0.0111488, 'sigma': 0},
        },
        'rot_sys2_4TIP_jointA_xdir_pos': {
            'b0': {'mean': -4.518731, 'sigma': 0},
            'b1': {'mean': -3.36422 , 'sigma': 0},
            'b2': {'mean': -0.579427, 'sigma': 0},
            'b3': {'mean': -0.082421, 'sigma': 0},
            'b4': {'mean': 1.0131557, 'sigma': 0},
            'b5': {'mean': 1.4949155, 'sigma': 0},
            'b6': {'mean': 0.1566652, 'sigma': 0},
            'b7': {'mean': -0.098604, 'sigma': 0},
            'b8': {'mean': 0.0109415, 'sigma': 0},
        },
        'rot_sys2_4TIP_jointA_ydir_neg': {
            'b0': {'mean': -4.462963, 'sigma': 0},
            'b1': {'mean': -0.963695, 'sigma': 0},
            'b2': {'mean': 0.7376861, 'sigma': 0},
            'b3': {'mean': -4.126469, 'sigma': 0},
            'b4': {'mean': 1.0306181, 'sigma': 0},
            'b5': {'mean': 0.5475509, 'sigma': 0},
            'b6': {'mean': -0.397127, 'sigma': 0},
            'b7': {'mean': 1.4222763, 'sigma': 0},
            'b8': {'mean': 0.0235704, 'sigma': 0},
        },
        'rot_sys2_4TIP_jointA_ydir_pos': {
            'b0': {'mean': -4.558921, 'sigma': 0},
            'b1': {'mean': -0.8824  , 'sigma': 0},
            'b2': {'mean': 0.7598913, 'sigma': 0},
            'b3': {'mean': -3.975664, 'sigma': 0},
            'b4': {'mean': 1.0152461, 'sigma': 0},
            'b5': {'mean': 0.5250301, 'sigma': 0},
            'b6': {'mean': -0.404852, 'sigma': 0},
            'b7': {'mean': 1.3611456, 'sigma': 0},
            'b8': {'mean': 0.0154418, 'sigma': 0},
        },
        ##########################
        'rot_sys3_4E90_jointB_xdir_neg': {
            'b0': {'mean': -14.61194, 'sigma': 0},
            'b1': {'mean': 4.5151701, 'sigma': 0},
            'b2': {'mean': -0.420656, 'sigma': 0},
            'b3': {'mean': 1.0096009, 'sigma': 0},
            'b4': {'mean': 1.0183309, 'sigma': 0},
            'b5': {'mean': -0.512237, 'sigma': 0},
            'b6': {'mean': 0.401096 , 'sigma': 0},
            'b7': {'mean': -0.205076, 'sigma': 0},
            'b8': {'mean': 0.0071062, 'sigma': 0},
        },
        'rot_sys3_4E90_jointB_xdir_pos': {
            'b0': {'mean': -14.71895, 'sigma': 0},
            'b1': {'mean': 4.7016769, 'sigma': 0},
            'b2': {'mean': -0.42377 , 'sigma': 0},
            'b3': {'mean': 1.022949 , 'sigma': 0},
            'b4': {'mean': 1.0221844, 'sigma': 0},
            'b5': {'mean': -0.570144, 'sigma': 0},
            'b6': {'mean': 0.3969054, 'sigma': 0},
            'b7': {'mean': -0.213641, 'sigma': 0},
            'b8': {'mean': 0.0115539, 'sigma': 0},
        },
        'rot_sys3_4E90_jointA_ydir_neg': {
            'b0': {'mean': -11.9348 , 'sigma': 0},
            'b1': {'mean': 2.6223667, 'sigma': 0},
            'b2': {'mean': 0.1510967, 'sigma': 0},
            'b3': {'mean': 0.3164401, 'sigma': 0},
            'b4': {'mean': 1.052507 , 'sigma': 0},
            'b5': {'mean': 0.0224929, 'sigma': 0},
            'b6': {'mean': -0.05415 , 'sigma': 0},
            'b7': {'mean': 0.2143374, 'sigma': 0},
            'b8': {'mean': 0.0414339, 'sigma': 0},  
        },
        'rot_sys3_4E90_jointA_ydir_pos': {
            'b0': {'mean': -11.78664, 'sigma': 0},
            'b1': {'mean': 2.6286328, 'sigma': 0},
            'b2': {'mean': 0.1982484, 'sigma': 0},
            'b3': {'mean': 0.3240902, 'sigma': 0},
            'b4': {'mean': 1.034169 , 'sigma': 0},
            'b5': {'mean': 0.0030352, 'sigma': 0},
            'b6': {'mean': -0.08186 , 'sigma': 0},
            'b7': {'mean': 0.2012005, 'sigma': 0},
            'b8': {'mean': 0.0252604, 'sigma': 0},  
        },
        ##########################
        'rot_sys3_4TOP_jointA_xdir_pos': {
            'b0': {'mean': -12.41726, 'sigma': 0},
            'b1': {'mean': 3.5396056, 'sigma': 0},
            'b2': {'mean': -1.004937, 'sigma': 0},
            'b3': {'mean': 0.0617563, 'sigma': 0},
            'b4': {'mean': 1.0198677, 'sigma': 0},
            'b5': {'mean': -0.518125, 'sigma': 0},
            'b6': {'mean': 0.7775454, 'sigma': 0},
            'b7': {'mean': -0.000747, 'sigma': 0},
            'b8': {'mean': 0.0087592, 'sigma': 0},
        },
        'rot_sys3_4TIP_jointB_xdir_neg': {
            'b0': {'mean': -14.51882, 'sigma': 0},
            'b1': {'mean': 4.4162116, 'sigma': 0},
            'b2': {'mean': -0.415758, 'sigma': 0},
            'b3': {'mean': 0.9985423, 'sigma': 0},
            'b4': {'mean': 1.0250871, 'sigma': 0},
            'b5': {'mean': -0.484312, 'sigma': 0},
            'b6': {'mean': 0.4102262, 'sigma': 0},
            'b7': {'mean': -0.197498, 'sigma': 0},
            'b8': {'mean': 0.0112979, 'sigma': 0},
        },
        'rot_sys3_4TIP_jointB_xdir_pos': {
            'b0': {'mean': -14.56424, 'sigma': 0},
            'b1': {'mean': 4.6306891, 'sigma': 0},
            'b2': {'mean': -0.384809, 'sigma': 0},
            'b3': {'mean': 1.0291383, 'sigma': 0},
            'b4': {'mean': 1.0187997, 'sigma': 0},
            'b5': {'mean': -0.561522, 'sigma': 0},
            'b6': {'mean': 0.383851 , 'sigma': 0},
            'b7': {'mean': -0.216541, 'sigma': 0},
            'b8': {'mean': 0.0096327, 'sigma': 0},
        },
        'rot_sys3_4TIP_jointA_ydir_neg': {
            'b0': {'mean': -12.3567 , 'sigma': 0},
            'b1': {'mean': 2.9112064, 'sigma': 0},
            'b2': {'mean': 0.1671651, 'sigma': 0},
            'b3': {'mean': 0.3527012, 'sigma': 0},
            'b4': {'mean': 1.079375 , 'sigma': 0},
            'b5': {'mean': -0.023617, 'sigma': 0},
            'b6': {'mean': -0.060284, 'sigma': 0},
            'b7': {'mean': 0.2172012, 'sigma': 0},
            'b8': {'mean': 0.0501512, 'sigma': 0},
        },
        'rot_sys3_4TIP_jointA_ydir_pos': {
            'b0': {'mean': -11.41485, 'sigma': 0},
            'b1': {'mean': 2.4161603, 'sigma': 0},
            'b2': {'mean': 0.1855476, 'sigma': 0},
            'b3': {'mean': 0.3304397, 'sigma': 0},
            'b4': {'mean': 1.0189061, 'sigma': 0},
            'b5': {'mean': 0.0421813, 'sigma': 0},
            'b6': {'mean': -0.07951 , 'sigma': 0},
            'b7': {'mean': 0.1933541, 'sigma': 0},
            'b8': {'mean': 0.0183808, 'sigma': 0},
        },
        ##########################
        'rot_sys4_4E90_jointA_ydir_neg': {
            'b0': {'mean': -11.44195, 'sigma': 0},
            'b1': {'mean': 4.611668 , 'sigma': 0},
            'b2': {'mean': 1.7558679, 'sigma': 0},
            'b3': {'mean': 1.125698 , 'sigma': 0},
            'b4': {'mean': 1.0653531, 'sigma': 0},
            'b5': {'mean': -0.664844, 'sigma': 0},
            'b6': {'mean': -0.200098, 'sigma': 0},
            'b7': {'mean': 0.1039246, 'sigma': 0},
            'b8': {'mean': 0.0421932, 'sigma': 0},  
        },
        'rot_sys4_4E90_jointA_ydir_pos': {
            'b0': {'mean': -11.77564, 'sigma': 0},
            'b1': {'mean': 4.654279 , 'sigma': 0},
            'b2': {'mean': 1.9257181, 'sigma': 0},
            'b3': {'mean': 0.8767135, 'sigma': 0},
            'b4': {'mean': 0.9975371, 'sigma': 0},
            'b5': {'mean': -0.688403, 'sigma': 0},
            'b6': {'mean': -0.239633, 'sigma': 0},
            'b7': {'mean': 0.0295991, 'sigma': 0},
            'b8': {'mean': 0.0168513, 'sigma': 0},  
        },
        ##########################
        'rot_sys4_4TIP_jointA_ydir_neg': {
            'b0': {'mean': -12.18058, 'sigma': 0},
            'b1': {'mean': 5.4998008, 'sigma': 0},
            'b2': {'mean': 1.6729466, 'sigma': 0},
            'b3': {'mean': 1.3561716, 'sigma': 0},
            'b4': {'mean': 1.1713781, 'sigma': 0},
            'b5': {'mean': -0.827624, 'sigma': 0},
            'b6': {'mean': -0.17294 , 'sigma': 0},
            'b7': {'mean': 0.1535901, 'sigma': 0},
            'b8': {'mean': 0.072764 , 'sigma': 0},  
        },
        'rot_sys4_4TIP_jointA_ydir_pos': {
            'b0': {'mean': -11.65122, 'sigma': 0},
            'b1': {'mean': 4.6981409, 'sigma': 0},
            'b2': {'mean': 1.8763506, 'sigma': 0},
            'b3': {'mean': 0.890829 , 'sigma': 0},
            'b4': {'mean': 1.0068938, 'sigma': 0},
            'b5': {'mean': -0.695686, 'sigma': 0},
            'b6': {'mean': -0.236414, 'sigma': 0},
            'b7': {'mean': 0.029141 , 'sigma': 0},
            'b8': {'mean': 0.0336236, 'sigma': 0},  
        },
        'rot_sys4_4TOP_jointC_ydir_pos': {
            'b0': {'mean': -7.222566, 'sigma': 0},
            'b1': {'mean': 0.176772 , 'sigma': 0},
            'b2': {'mean': 1.9895425, 'sigma': 0},
            'b3': {'mean': 0.4757355, 'sigma': 0},
            'b4': {'mean': 1.0779995, 'sigma': 0},
            'b5': {'mean': 0.1533041, 'sigma': 0},
            'b6': {'mean': -0.149774, 'sigma': 0},
            'b7': {'mean': -0.142017, 'sigma': 0},
            'b8': {'mean': 0.0419737, 'sigma': 0},  
        },
    }
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': {
            'sys2': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p2, l_p6_sys23, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(l_p2) + \
                b3*np.log(l_p6_sys23) + b4*np.log(pga) + \
                b5*np.log(h_tree)**2 + b6*np.log(l_p2)**2 + \
                b7*np.log(l_p6_sys23)**2 + b8*np.log(pga)**2),
            'sys3': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p2, l_p6_sys23, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(l_p2) + \
                b3*np.log(l_p6_sys23) + b4*np.log(pga) + \
                b5*np.log(h_tree)**2 + b6*np.log(l_p2)**2 + \
                b7*np.log(l_p6_sys23)**2 + b8*np.log(pga)**2),
            'sys4': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p6_sys4, w_valve, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(l_p6_sys4) + \
                b3*np.log(w_valve) + b4*np.log(pga) + \
                b5*np.log(h_tree)**2 + b6*np.log(l_p6_sys4)**2 + \
                b7*np.log(w_valve)**2 + b8*np.log(pga)**2),
        },
        'sigma': {
            ##########################
            'rot_sys2_4E90_jointA_xdir_neg': 0.447,
            'rot_sys2_4E90_jointA_xdir_pos': 0.441,
            'rot_sys2_4E90_jointA_ydir_neg': 0.706,
            'rot_sys2_4E90_jointA_ydir_pos': 0.703,
            ##########################
            'rot_sys2_4TIP_jointA_xdir_neg': 0.456,
            'rot_sys2_4TIP_jointA_xdir_pos': 0.433,
            'rot_sys2_4TIP_jointA_ydir_neg': 0.721,
            'rot_sys2_4TIP_jointA_ydir_pos': 0.698,
            ##########################
            'rot_sys3_4E90_jointB_xdir_neg': 0.424,
            'rot_sys3_4E90_jointB_xdir_pos': 0.418,
            'rot_sys3_4E90_jointA_ydir_neg': 0.494,
            'rot_sys3_4E90_jointA_ydir_pos': 0.448,
            ##########################
            'rot_sys3_4TOP_jointA_xdir_pos': 0.368,
            'rot_sys3_4TIP_jointB_xdir_neg': 0.443,
            'rot_sys3_4TIP_jointB_xdir_pos': 0.41 ,
            'rot_sys3_4TIP_jointA_ydir_neg': 0.512,
            'rot_sys3_4TIP_jointA_ydir_pos': 0.445,
            ##########################
            'rot_sys4_4E90_jointA_ydir_neg': 0.421,
            'rot_sys4_4E90_jointA_ydir_pos': 0.395,
            ##########################
            'rot_sys4_4TIP_jointA_ydir_neg': 0.5  ,
            'rot_sys4_4TIP_jointA_ydir_pos': 0.45 ,
            'rot_sys4_4TOP_jointC_ydir_pos': 0.431,
        },
        'sigma_mu': {
            ##########################
            'rot_sys2_4E90_jointA_xdir_neg': 0.25,
            'rot_sys2_4E90_jointA_xdir_pos': 0.25,
            'rot_sys2_4E90_jointA_ydir_neg': 0.25,
            'rot_sys2_4E90_jointA_ydir_pos': 0.25,
            ##########################
            'rot_sys2_4TIP_jointA_xdir_neg': 0.25,
            'rot_sys2_4TIP_jointA_xdir_pos': 0.25,
            'rot_sys2_4TIP_jointA_ydir_neg': 0.25,
            'rot_sys2_4TIP_jointA_ydir_pos': 0.25,
            ##########################
            'rot_sys3_4E90_jointB_xdir_neg': 0.25,
            'rot_sys3_4E90_jointB_xdir_pos': 0.25,
            'rot_sys3_4E90_jointA_ydir_neg': 0.25,
            'rot_sys3_4E90_jointA_ydir_pos': 0.25,
            ##########################
            'rot_sys3_4TOP_jointA_xdir_pos': 0.25,
            'rot_sys3_4TIP_jointB_xdir_neg': 0.25,
            'rot_sys3_4TIP_jointB_xdir_pos': 0.25,
            'rot_sys3_4TIP_jointA_ydir_neg': 0.25,
            'rot_sys3_4TIP_jointA_ydir_pos': 0.25,
            ##########################
            'rot_sys4_4E90_jointA_ydir_neg': 0.25,
            'rot_sys4_4E90_jointA_ydir_pos': 0.25,
            ##########################
            'rot_sys4_4TIP_jointA_ydir_neg': 0.25,
            'rot_sys4_4TIP_jointA_ydir_pos': 0.25,
            'rot_sys4_4TOP_jointC_ydir_pos': 0.25,
        },
        'func_string': {},
        'string': {}
    }
    
    
    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """uses sys_type to determine what model parameters to use"""
        sys_type = kwargs.get('sys_type')
        # subsystem types present
        sys_types_present = [num for num in [2, 3, 4] if num in sys_type]
        # initialize
        # req_rvs_by_level = []
        # req_fixed_by_level = []
        req_rvs_by_level = {
            'level1': [],
            'level2': [],
            'level3': [],
        }
        req_fixed_by_level = {
            'level1': list(cls._REQ_MODEL_FIXED_FOR_LEVEL),
            'level2': list(cls._REQ_MODEL_FIXED_FOR_LEVEL),
            'level3': list(cls._REQ_MODEL_FIXED_FOR_LEVEL),
        }
        # make list of coefficients to get track
        coeffs = [f"b{i}" for i in range(10)]
        # loop through subsystem types
        for num in sys_types_present:
            case = f'sys{num}'
            temp_params = [
                param for param in cls._MODEL_FORM['func'][case].__code__.co_varnames
                if not param in coeffs
            ]
            for level in range(1, 4):
                req_rvs_by_level[f'level{level}'] += temp_params
        for level in range(1, 4):
            req_rvs_by_level[f'level{level}'] = sorted(list(set(req_rvs_by_level)))
        # req_fixed_by_level = list(cls._REQ_MODEL_FIXED_FOR_LEVEL)
        return req_rvs_by_level, req_fixed_by_level
    
    
    @classmethod
    # @njit
    def _model(cls, 
        pga, # upstream PBEE RV
        h_tree, l_p2, l_p6_sys23, l_p6_sys4, w_valve, # infrastructure
        sys_type, tee_flag, elbow_flag, # fixed/toggles
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""

        # cases
        joints = {
            'sys2': ['A'],
            'sys3': ['A','B'],
            'sys4': ['A','B','C'],
        }
        rot_type = {
            'elbow': ['4E90'],
            'tee': ['4TOP','4TIP'],
        }       
                 
        # # convert units from SI to imperial (model developed in imperial units)
        h_tree = h_tree/0.3048 # meter to feet
        l_p2 = l_p2/0.3048 # meter to feet
        l_p6_sys23 = l_p6_sys23/0.3048 # meter to feet
        l_p6_sys4 = l_p6_sys4/0.3048 # meter to feet
        w_valve = w_valve*224.809/1000 # kN to kips
        
        # prevent 0 values for log(val)
        zero_val = 1e-5
        h_tree[h_tree==0] = zero_val # ft
        l_p2[l_p2==0] = zero_val # ft
        l_p6_sys23[l_p6_sys23==0] = zero_val # ft
        l_p6_sys4[l_p6_sys4==0] = zero_val # ft
        w_valve[w_valve==0] = zero_val # kips, or 0.01 lbs
        
        # initialize intermediate and output arrays
        cases_to_run = []
        output_mean = {}
        output_sigma = {}
        output_sigma_mu = {}
        for num in [2,3,4]:
            sys_case = f'sys{num}'
            # see if sys type exists
            for comp in ['elbow','tee']:
                ind_zero = np.logical_or(sys_type!=num,locals()[f'{comp}_flag']==False)
                for rot in rot_type[comp]:
                    # see if num is in 
                    for joint in joints[f'sys{num}']:
                        for d in ['x','y']:
                            for p in ['neg','pos']:
                                rot_case = f"rot_sys{num}_{rot}_joint{joint}_{d}dir_{p}"
                                if rot_case in cls._MODEL_FORM_DETAIL:
                                    # get sigma and sigma mu
                                    output_sigma[rot_case] = np.ones(pga.shape)*cls._MODEL_FORM['sigma'][rot_case]
                                    output_sigma_mu[rot_case] = np.ones(pga.shape)*cls._MODEL_FORM['sigma_mu'][rot_case]        
                                    output_mean[rot_case] = \
                                        cls._MODEL_FORM['func'][sys_case](
                                            # mean coefficients
                                            **cls._get_mean_coeff_for_lambda_func(
                                                cls._MODEL_FORM_DETAIL[rot_case],
                                                cls._MODEL_FORM['func'][sys_case]
                                            ),
                                            # inputs
                                            **cls._get_kwargs_for_lambda_func(
                                                locals(),
                                                cls._MODEL_FORM['func'][sys_case],
                                            )
                                        )
                                    # for rotation cases that do not match the site conditions, set to np.exp(-10)
                                    output_mean[rot_case][ind_zero] = np.exp(-10)
        
        # prepare outputs
        output = {}
        for case in output_mean:
            output[case] = {
                'mean': output_mean[case],
                'sigma': output_sigma[case],
                'sigma_mu': output_sigma_mu[case],
                'dist_type': 'lognormal',
                'unit': 'deg'
            }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output