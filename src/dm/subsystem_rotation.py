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
class SubsystemRotation(BaseModel):
    "Inherited class specfic to rotations of subsystems"

    _RETURN_PBEE_META = {
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        'type': 'subsystem rotation',       # Type of model (e.g., liquefaction, landslide, pipe strain)
        'variable': [
            'rot_in', # in-plane rotation of tee
            'rot_out' # out-of-plane rotation of tee
        ]        # Return variable for PBEE category, e.g., pgdef, eps_p
    }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PantoliEtal2022(SubsystemRotation):
    """
    Compute ground-shaking-induced rotations on tees of subystems using Pantoli et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration

    Infrastructure:
    h_tree: float, np.ndarray or list, optional
        [ft] tree height, for all subsystem types
    l_p2: float, np.ndarray or list, optional
        [ft] length of pipe segment S2, for subsystem types 2 and 3
    l_p6: float, np.ndarray or list, optional
        [ft] length of pipe segment S6, for subsystem types 2 and 3
    l_p: float, np.ndarray or list, optional
        [ft] length of pipe segment S6, for subsystem type 4
    w_valve: float, np.ndarray or list, optional
        [lbf] valve weight, for subsystem type 4

    Fixed:
    sys_type: int, np.ndarray or list
        subsystem type: 2, 3, 4

    Returns
    -------
    rot_in : float
        [deg] in-plane rotation of tee
    rot_out : float
        [deg] out-of-plane rotation of tee (non-zero for subsystem 4 only)
    
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
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'rot_in': {
                'desc': 'in-plane rotation of tee (deg)',
                'unit': 'deg',
                'mean': None,
                'aleatory': None,
                'epistemic': {
                    'coeff': None, # base uncertainty, based on coeffcients
                    'input': None, # sigma_mu uncertainty from input parameters
                    'total': None # SRSS of coeff and input sigma_mu uncertainty
                },
                'dist_type': 'lognormal',
            },
            'rot_out': {
                'desc': 'out-of-plane rotation of tee (deg)',
                'unit': 'deg',
                'mean': None,
                'aleatory': None,
                'epistemic': {
                    'coeff': None, # base uncertainty, based on coeffcients
                    'input': None, # sigma_mu uncertainty from input parameters
                    'total': None # SRSS of coeff and input sigma_mu uncertainty
                },
                'dist_type': 'lognormal',
            }
        }
    }
    _INPUT_PBEE_META = {
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        'variable': 'pga'        # Input variable for PBEE category, e.g., pgdef, eps_pipe
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
                'mean': None,
                'aleatory': None,
                'epistemic': None,
                'dist_type': 'lognormal'
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 1
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'h_tree': 'tree height (ft), for all subsystem types',
            'l_p2': 'length of pipe segment S2 (ft), for subsystem types 2 and 3',
            'l_p6': 'length of pipe segment S6 (ft), for subsystem types 2 and 3',
            'l_p': 'length of pipe segment S6 (ft), for subsystem type 4',
            'w_valve': 'valve weight (lbf), for subsystem type 4',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'sys_type': 'subsystem type: 2, 3, 4',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'sys_type',
    }
    _MODEL_INTERNAL = {
        'n_sample': 1,
        'n_site': 1,
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {
        'subsystem_2_rot_in': {
            'b0': {'mean':  2.294935754 , 'sigma': 0.37986942},
            'b1': {'mean': -6.175623962 , 'sigma': 0.37434652},
            'b2': {'mean':  1.6671207   , 'sigma': 0.09154472},
            'b3': {'mean': -0.087025336 , 'sigma': 0.0155941 },
            'b4': {'mean':  0           , 'sigma': 0.        },
            'b5': {'mean': -4.329023733 , 'sigma': 0.04774954},
            'b6': {'mean':  1.389144471 , 'sigma': 0.01547973},
            'b7': {'mean':  1.039350826 , 'sigma': 0.00720858},
            'b8': {'mean':  0.032322755 , 'sigma': 0.00326133},  
        },
        'subsystem_2_rot_out': {
        },
        'subsystem_3_rot_in': {
            'b0': {'mean': -11.750823   , 'sigma': 0.06698283},
            'b1': {'mean':   2.679232199, 'sigma': 0.0226412 },
            'b2': {'mean':   0          , 'sigma': 0.        },
            'b3': {'mean':   0.147545651, 'sigma': 0.0386119 },
            'b4': {'mean':  -0.062645666, 'sigma': 0.01669676},
            'b5': {'mean':   0.325638661, 'sigma': 0.03276234},
            'b6': {'mean':   0.208363281, 'sigma': 0.01022789},
            'b7': {'mean':   1.076572574, 'sigma': 0.00507652},
            'b8': {'mean':   0.045263547, 'sigma': 0.00284189},
        },
        'subsystem_3_rot_out': {
        },
        'subsystem_4_rot_in': {
            'b0': {'mean': -15.81097096 , 'sigma': 0.41615274},
            'b1': {'mean':   5.064247787, 'sigma': 0.25754404},
            'b2': {'mean':  -0.73379727 , 'sigma': 0.0627999 },
            'b3': {'mean':   1.727102193, 'sigma': 0.18643054},
            'b4': {'mean':  -0.18475923 , 'sigma': 0.03698633},
            'b5': {'mean':   0          , 'sigma': 0.        },
            'b6': {'mean':   0.082570279, 'sigma': 0.00153002},
            'b7': {'mean':   1.155981637, 'sigma': 0.00610085},
            'b8': {'mean':   0.074983723, 'sigma': 0.0037632 },
        },
        'subsystem_4_rot_out': {
            'b0': {'mean': -17.37804336 , 'sigma': 0.97574403},
            'b1': {'mean':   0.795123525, 'sigma': 0.02162045},
            'b2': {'mean':   0          , 'sigma': 0.        },
            'b3': {'mean':   2.176109205, 'sigma': 0.16939652},
            'b4': {'mean':  -0.181175614, 'sigma': 0.03388567},
            'b5': {'mean':   2.084769343, 'sigma': 0.32169171},
            'b6': {'mean':  -0.109545254, 'sigma': 0.02880312},
            'b7': {'mean':   1.082451825, 'sigma': 0.00565291},
            'b8': {'mean':   0.052457305, 'sigma': 0.00348689},
        },
    }
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': {
            'subsystem_2_rot_in': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p2, l_p6, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(h_tree)**2 + \
                b3*np.log(l_p2) + b4*np.log(l_p2)**2 + \
                b5*np.log(l_p6) + b6*np.log(l_p6)**2 + \
                b7*np.log(pga) + b8*np.log(pga)**2),
            'subsystem_2_rot_out': \
                lambda : 0,
            'subsystem_3_rot_in': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p2, l_p6, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(h_tree)**2 + \
                b3*np.log(l_p2) + b4*np.log(l_p2)**2 + \
                b5*np.log(l_p6) + b6*np.log(l_p6)**2 + \
                b7*np.log(pga) + b8*np.log(pga)**2),
            'subsystem_3_rot_out': \
                lambda : 0,
            'subsystem_4_rot_in': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p, w_valve, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(h_tree)**2 + \
                b3*np.log(l_p) + b4*np.log(l_p)**2 + \
                b5*np.log(w_valve) + b6*np.log(w_valve)**2 + \
                b7*np.log(pga) + b8*np.log(pga)**2),
            'subsystem_4_rot_out': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8, h_tree, l_p, w_valve, pga: \
                np.exp(b0 + b1*np.log(h_tree) + b2*np.log(h_tree)**2 + \
                b3*np.log(l_p) + b4*np.log(l_p)**2 + \
                b5*np.log(w_valve) + b6*np.log(w_valve)**2 + \
                b7*np.log(pga) + b8*np.log(pga)**2),
        },
        'aleatory': {
            'subsystem_2_rot_in': 0.554448,
            'subsystem_2_rot_out': 0,
            'subsystem_3_rot_in': 0.0758868,
            'subsystem_3_rot_out': 0,
            'subsystem_4_rot_in': 0.0358501,
            'subsystem_4_rot_out': 0.0366267,
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
        """uses soil_type to determine what model parameters to use"""
        sys_type = kwargs.get('sys_type')
        # modes present
        sys_types = [num for num in [2, 3, 4] if num in sys_type]
        # loop through modes, then tubing/casing, then cementation condition
        req_rvs_by_level = []
        req_fixed_by_level = []
        # make list of coefficients to get track
        coeffs = [f"b{i}" for i in range(10)]
        # 
        for num in modes:
            for rot in ['rot_in', 'rot_out']:
                case = f'subsystem_{num}_{rot}'
                req_rvs_by_level += [
                    param for param in cls._MODEL_FORM['func'][case].__code__.co_varnames
                    if not param in coeffs
                ]
        req_rvs_by_level = sorted(list(set(req_rvs_by_level)))
        req_fixed_by_level = cls._REQ_MODEL_FIXED_FOR_LEVEL
        return req_rvs_by_level, req_fixed_by_level
    

    @classmethod
    # @njit
    def _model(cls, 
        pga, # upstream PBEE RV
        h_tree, l_p2, l_p6, l_p, w_valve, # infrastructure
        sys_type, # fixed/toggles
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""
        # initialize intermediate and output arrays
        rot_in = np.zeros(pga.shape)
        rot_out = np.zeros(pga.shape)
        
        # determine cases
        # find indices for each mode
        ind_sys_type = {
            num: np.where(sys_type==num)[0]
            for num in [2, 3, 4]
        }
        
        # loop through modes, then tubing/casing, then cementation condition
        for num in [2, 3, 4]:
            if len(ind_sys_type[num]) > 0:
                for rot in ['rot_in', 'rot_out']:
                    # case name
                    case = f'subsystem_{num}_{rot}'
                    # run calc using lambda function
                    locals()[rot][ind_sys_type[num]] = cls._MODEL_FORM['func'][case](
                        # mean coefficients
                        **cls._get_mean_coeff_for_lambda_func(
                            cls._MODEL_FORM_DETAIL[case],
                            cls._MODEL_FORM['func'][case]
                        ),
                        # inputs
                        **cls._get_kwargs_for_lambda_func(
                            locals(),
                            cls._MODEL_FORM['func'][case],
                            inds=ind_sys_type[num]
                        )
                    )

        # prepare outputs
        output = {
            'rot_in': rot_in,
            'rot_out': rot_out,
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output