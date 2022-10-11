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
class SubsystemStrain(BaseModel):
    "Inherited class specfic to strain of subsystems"

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
class PantoliEtal2022(SubsystemStrain):
    """
    Compute rotation-induced strain on tees and elbows of subystems using Pantoli et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
        rotations from "edp.subsystem_strain.PantoliEtal2022"

    Infrastructure:

    Fixed:
    sys_type: int, np.ndarray or list
        subsystem type: 2 for p2, 3 for p3, 4 for p4
    tee_flag: boolean, np.ndarray or list
        flag for tees in subsystem: True/False
    elbow_flag: boolean, np.ndarray or list
        flag for elbows in subsystem: True/False
    high_pressure_weight: float, np.ndarray or list
        weight to apply to high pressure model; for low pressure model, weight = 1 - high_pressure_weight

    Returns
    -------
    eps_elbow : float
        [deg] tensile strain for elbows
    eps_tee : float
        [deg] tensile strain for tee-joints
    
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
            'eps_elbow': {
                'desc': 'tensile strain for elbows (%)',
                'unit': '%',
            },
            'eps_tee': {
                'desc': 'tensile strain for elbows (%)',
                'unit': '%',
            },
            'rot_elbow_control_case': {
                'desc': 'subsystem rotation case that controls tensile strains for elbows',
                'unit': '',
            },
            'rot_tee_control_case': {
                'desc': 'subsystem rotation case that controls tensile strains for tees',
                'unit': '',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            ##########################
            'sys2_4E90_jointA_xdir_pos_rot': {
                'desc': 'subsystem 2, elbow, joint A, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys2_4E90_jointA_xdir_neg_rot': {
                'desc': 'subsystem 2, elbow, joint A, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'sys2_4E90_jointA_ydir_pos_rot': {
                'desc': 'subsystem 2, elbow, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys2_4E90_jointA_ydir_neg_rot': {
                'desc': 'subsystem 2, elbow, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'sys2_4TIP_jointA_xdir_pos_rot': {
                'desc': 'subsystem 2, tee in-plane, joint A, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys2_4TIP_jointA_xdir_neg_rot': {
                'desc': 'subsystem 2, tee in-plane, joint A, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'sys2_4TIP_jointA_ydir_pos_rot': {
                'desc': 'subsystem 2, tee in-plane, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys2_4TIP_jointA_ydir_neg_rot': {
                'desc': 'subsystem 2, tee in-plane, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'sys3_4E90_jointB_xdir_pos_rot': {
                'desc': 'subsystem 3, elbow, joint B, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys3_4E90_jointB_xdir_neg_rot': {
                'desc': 'subsystem 3, elbow, joint B, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'sys3_4E90_jointA_ydir_pos_rot': {
                'desc': 'subsystem 3, elbow, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys3_4E90_jointA_ydir_neg_rot': {
                'desc': 'subsystem 3, elbow, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'sys3_4TOP_jointA_xdir_pos_rot': {
                'desc': 'subsystem 3, tee out-of-plane, joint A, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys3_4TIP_jointB_xdir_pos_rot': {
                'desc': 'subsystem 3, tee in-plane, joint B, x-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys3_4TIP_jointB_xdir_neg_rot': {
                'desc': 'subsystem 3, tee in-plane, joint B, x-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'sys3_4TIP_jointA_ydir_pos_rot': {
                'desc': 'subsystem 3, tee in-plane, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys3_4TIP_jointA_ydir_neg_rot': {
                'desc': 'subsystem 3, tee in-plane, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'sys4_4E90_jointA_ydir_pos_rot': {
                'desc': 'subsystem 4, elbow, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys4_4E90_jointA_ydir_neg_rot': {
                'desc': 'subsystem 4, elbow, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            ##########################
            'sys4_4TIP_jointA_ydir_pos_rot': {
                'desc': 'subsystem 4, tee in-plane, joint A, y-dir, pos(open) (deg)',
                'unit': 'deg',
            },
            'sys4_4TIP_jointA_ydir_neg_rot': {
                'desc': 'subsystem 4, tee in-plane, joint A, y-dir, neg(close) (deg)',
                'unit': 'deg',
            },
            'sys4_4TOP_jointC_ydir_pos_rot': {
                'desc': 'subsystem 4, tee out-of-plane, joint C, y-dir, pos(open) (deg)',
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
            'sys_type': 'subsystem type: 2 for p2, 3 for p3, 4 for p4',
            # 'tee_flag': 'flag for tee in subsystem: True/False',
            # 'elbow_flag': 'flag for elbow in subsystem: True/False',
            'high_pressure_weight': 'weight to apply to high pressure model; for low pressure model, weight = 1 - high_pressure_weight'
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'sys_type', 'high_pressure_weight'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {
        ##########################
        '4TIP_neg_lowOP_eps': {
            'b0': {'mean': -4.698614, 'sigma': 0},
            'b1': {'mean': 0.9171584, 'sigma': 0},
        },
        '4TIP_pos_lowOP_eps': {
            'b0': {'mean': -4.846407, 'sigma': 0},
            'b1': {'mean': 1.6389036, 'sigma': 0},
        },
        '4TIP_neg_highOP_eps': {
            'b0': {'mean': -4.901993, 'sigma': 0},
            'b1': {'mean': 1.6269746, 'sigma': 0},
        },
        '4TIP_pos_highOP_eps': {
            'b0': {'mean': -4.196789, 'sigma': 0},
            'b1': {'mean': 1.8360272, 'sigma': 0},
        },
        ##########################
        '4TOP_neg_lowOP_eps': {
            'b0': {'mean': -5.672304, 'sigma': 0},
            'b1': {'mean': 1.1390712, 'sigma': 0},
        },
        '4TOP_pos_lowOP_eps': {
            'b0': {'mean': -5.672304, 'sigma': 0},
            'b1': {'mean': 1.1390712, 'sigma': 0},
        },
        '4TOP_neg_highOP_eps': {
            'b0': {'mean': -4.780363, 'sigma': 0},
            'b1': {'mean': 1.1869063, 'sigma': 0},
        },
        '4TOP_pos_highOP_eps': {
            'b0': {'mean': -4.780363, 'sigma': 0},
            'b1': {'mean': 1.1869063, 'sigma': 0},
        },
        ##########################
        '4E90_neg_lowOP_eps': {
            'b0': {'mean': -5.650503, 'sigma': 0},
            'b1': {'mean': 1.5118598, 'sigma': 0},
        },
        '4E90_pos_lowOP_eps': {
            'b0': {'mean': -5.538686, 'sigma': 0},
            'b1': {'mean': 1.0648018, 'sigma': 0},
        },
        '4E90_neg_highOP_eps': {
            'b0': {'mean': -5.675531, 'sigma': 0},
            'b1': {'mean': 1.456687, 'sigma': 0},
        },
        '4E90_pos_highOP_eps': {
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
            '4TIP_neg_lowOP_eps':  0.085,
            '4TIP_pos_lowOP_eps':  0.051,
            '4TIP_neg_highOP_eps': 0.163,
            '4TIP_pos_highOP_eps': 0.095,
            ##########################
            '4TOP_neg_lowOP_eps':  0.093,
            '4TOP_pos_lowOP_eps':  0.093,
            '4TOP_neg_highOP_eps': 0.222,
            '4TOP_pos_highOP_eps': 0.222,
            ##########################
            '4E90_neg_lowOP_eps':  0.160,
            '4E90_pos_lowOP_eps':  0.062,
            '4E90_neg_highOP_eps': 0.137,
            '4E90_pos_highOP_eps': 0.090,
        },
        'sigma_mu': {
            '4TIP_neg_lowOP_eps':  0.25,
            '4TIP_pos_lowOP_eps':  0.25,
            '4TIP_neg_highOP_eps': 0.25,
            '4TIP_pos_highOP_eps': 0.25,
            ##########################
            '4TOP_neg_lowOP_eps':  0.25,
            '4TOP_pos_lowOP_eps':  0.25,
            '4TOP_neg_highOP_eps': 0.25,
            '4TOP_pos_highOP_eps': 0.25,
            ##########################
            '4E90_neg_lowOP_eps':  0.25,
            '4E90_pos_lowOP_eps':  0.25,
            '4E90_neg_highOP_eps': 0.25,
            '4E90_pos_highOP_eps': 0.25,
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
        sys2_4E90_jointA_xdir_pos_rot,
        sys2_4E90_jointA_xdir_neg_rot,
        sys2_4E90_jointA_ydir_pos_rot,
        sys2_4E90_jointA_ydir_neg_rot,
        ##########################
        sys2_4TIP_jointA_xdir_pos_rot,
        sys2_4TIP_jointA_xdir_neg_rot,
        sys2_4TIP_jointA_ydir_pos_rot,
        sys2_4TIP_jointA_ydir_neg_rot,
        ##########################
        sys3_4E90_jointB_xdir_pos_rot,
        sys3_4E90_jointB_xdir_neg_rot,
        sys3_4E90_jointA_ydir_pos_rot,
        sys3_4E90_jointA_ydir_neg_rot,
        ##########################
        sys3_4TOP_jointA_xdir_pos_rot,
        sys3_4TIP_jointB_xdir_pos_rot,
        sys3_4TIP_jointB_xdir_neg_rot,
        sys3_4TIP_jointA_ydir_pos_rot,
        sys3_4TIP_jointA_ydir_neg_rot,
        ##########################
        sys4_4E90_jointA_ydir_pos_rot,
        sys4_4E90_jointA_ydir_neg_rot,
        ##########################
        sys4_4TIP_jointA_ydir_pos_rot,
        sys4_4TIP_jointA_ydir_neg_rot,
        sys4_4TOP_jointC_ydir_pos_rot, # upstream PBEE RV
        ##########################
        sys_type, high_pressure_weight=0.7, # fixed/toggles
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
        
        # initialize intermediate and output arrays
        controlling_eps_ind_for_all_sys_and_rot = {}
        controlling_eps_case_for_all_sys_and_rot = {}
        eps_for_all_sys_and_rot = {}
        sigma_for_all_sys_and_rot = {}
        all_eps_for_all_sys_and_rot = {}
        # loop through possible branches
        for num in [2,3,4]:
            for comp in ['elbow','tee']:
                # initialize dictionary for current system + component
                all_eps_for_curr_sys_and_rot = {}
                base_str = f"eps_sys{num}_{comp}"
                track_rot_type_and_config_for_sigma = []
                track_full_rot_case_name = []
                # additional loops for rotation model
                for rot in rot_type[comp]:
                    # see if num is in 
                    for joint in joints[f'sys{num}']:
                        for d in ['x','y']:
                            for p in ['neg','pos']:
                                rot_case = f"sys{num}_{rot}_joint{joint}_{d}dir_{p}_rot"
                                # case exists as an input, run rotation through both high and low OP models
                                if rot_case in locals():
                                    # append rotation case name
                                    track_rot_type_and_config_for_sigma.append(f"{rot}_{p}")
                                    track_full_rot_case_name.append('eps_'+rot_case)
                                    # first compute weighted high operating pressure case
                                    eps_model_case = f"{rot}_{p}_highOP_eps"
                                    all_eps_for_curr_sys_and_rot['eps_'+rot_case] = \
                                        cls._MODEL_FORM['func'](
                                            # mean coefficients
                                            **cls._get_mean_coeff_for_lambda_func(
                                                cls._MODEL_FORM_DETAIL[eps_model_case],
                                                cls._MODEL_FORM['func']
                                            ),
                                            rot=locals()[rot_case]
                                        ) ** high_pressure_weight
                                    # next add weighted low operating pressure case
                                    eps_model_case = f"{rot}_{p}_lowOP_eps"
                                    all_eps_for_curr_sys_and_rot['eps_'+rot_case] = all_eps_for_curr_sys_and_rot['eps_'+rot_case] * \
                                        cls._MODEL_FORM['func'](
                                            # mean coefficients
                                            **cls._get_mean_coeff_for_lambda_func(
                                                cls._MODEL_FORM_DETAIL[eps_model_case],
                                                cls._MODEL_FORM['func']
                                            ),
                                            rot=locals()[rot_case]
                                        ) ** (1-high_pressure_weight)
                # store results
                for each in all_eps_for_curr_sys_and_rot:
                    all_eps_for_all_sys_and_rot[each] = all_eps_for_curr_sys_and_rot[each]
                # once all the strains for current system and component case has been computed, check for case with highest strain
                # first combine into in num
                all_eps_for_curr_sys_and_rot_in_arr = np.asarray(
                    [all_eps_for_curr_sys_and_rot[each] for each in all_eps_for_curr_sys_and_rot]
                )
                # store to dictionary
                controlling_eps_ind_for_all_sys_and_rot[base_str+'_control_ind'] = \
                    np.argmax(all_eps_for_curr_sys_and_rot_in_arr,axis=0)
                controlling_eps_case_for_all_sys_and_rot[base_str+'_control_case'] = np.asarray(
                    [track_full_rot_case_name[ind] for ind in controlling_eps_ind_for_all_sys_and_rot[base_str+'_control_ind']]
                )
                eps_for_all_sys_and_rot[base_str] = np.max(all_eps_for_curr_sys_and_rot_in_arr,axis=0)
                controlling_lowOP_sigma = np.asarray([
                    cls._MODEL_FORM['sigma'][track_rot_type_and_config_for_sigma[ind]+'_lowOP_eps']
                 for ind in controlling_eps_ind_for_all_sys_and_rot[base_str+'_control_ind']])
                controlling_highOP_sigma = np.asarray([
                    cls._MODEL_FORM['sigma'][track_rot_type_and_config_for_sigma[ind]+'_highOP_eps']
                 for ind in controlling_eps_ind_for_all_sys_and_rot[base_str+'_control_ind']])
                sigma_for_all_sys_and_rot['sigma_'+base_str] = np.sqrt(
                    controlling_highOP_sigma**2*high_pressure_weight + controlling_lowOP_sigma**2*(1-high_pressure_weight)
                )
                # print(sigma_for_all_sys_and_rot['sigma_'+base_str].shape)
        
        # now pick the case to output based on subsystem type
        # first run elbow
        eps_elbow = np.asarray([eps_for_all_sys_and_rot[f"eps_sys{num}_elbow"][i] for i,num in enumerate(sys_type)])
        sigma_eps_elbow = np.asarray([sigma_for_all_sys_and_rot[f"sigma_eps_sys{num}_elbow"][i] for i,num in enumerate(sys_type)])
        # next check tees
        eps_tee = np.asarray([eps_for_all_sys_and_rot[f"eps_sys{num}_tee"][i] for i,num in enumerate(sys_type)])
        sigma_eps_tee = np.asarray([sigma_for_all_sys_and_rot[f"sigma_eps_sys{num}_tee"][i] for i,num in enumerate(sys_type)])
        
        # determine subsystem rotation case that controls strains, to be used in PC
        # first run elbow
        eps_elbow = np.asarray([eps_for_all_sys_and_rot[f"eps_sys{num}_elbow"][i] for i,num in enumerate(sys_type)])
        sigma_eps_elbow = np.asarray([sigma_for_all_sys_and_rot[f"sigma_eps_sys{num}_elbow"][i] for i,num in enumerate(sys_type)])
        # next check tees
        eps_tee = np.asarray([eps_for_all_sys_and_rot[f"eps_sys{num}_tee"][i] for i,num in enumerate(sys_type)])
        sigma_eps_tee = np.asarray([sigma_for_all_sys_and_rot[f"sigma_eps_sys{num}_tee"][i] for i,num in enumerate(sys_type)])
                    
        # prepare outputs
        output = {
            'eps_elbow': {
                'mean': eps_elbow * 100, # convert to %
                'sigma': sigma_eps_elbow,
                'sigma_mu': np.ones(length)*0.25,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            'eps_tee': {
                'mean': eps_tee * 100, # convert to %
                'sigma': sigma_eps_tee,
                'sigma_mu': np.ones(length)*0.25,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            'rot_elbow_control_case': rot_elbow_control_case,
            'rot_tee_control_case': rot_tee_control_case

        }
        # get intermediate values if requested
        if return_inter_params:
            # add intermediate results to output
            for each in all_eps_for_all_sys_and_rot:
                output[each] = all_eps_for_all_sys_and_rot[each]*100
            for each in eps_for_all_sys_and_rot:
                output[each] = eps_for_all_sys_and_rot[each]*100
            for each in sigma_for_all_sys_and_rot:
                output[each] = sigma_for_all_sys_and_rot[each]
        
        # return
        return output