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
class WellStrain(BaseModel):
    "Inherited class specfic to well strain"

    # _RETURN_PBEE_META = {
    #     'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'well strain',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'strain_tubing',
    #         'strain_casing',
    #     ]        # Return variable for PBEE category, e.g., pgdef, strain_p
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class SasakiEtal2022(WellStrain):
    """
    Compute deformation-induced tubing and casing strain using Sasaki et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation

    Infrastructure:
    phi_cmt: float, np.ndarray or list, optional
        [mm] internal friction angle of cement; for mode 3 + cemented tubing only
    ucs_cmt: float, np.ndarray or list, optional
        [MPa] uniaxial compressive strength of cement; for mode 4 + cemented tubing only
    
    Geotechnical/geologic:
    theta: float, np.ndarray or list
        [deg] fault angle; for all modes + (un)cemented tubing and casing
    w_fc: float, np.ndarray or list, optional
        [m] fault core width; for all modes + cemented tubing only
    w_dz: float, np.ndarray or list, optional
        [m] damage zone width; for all modes + cemented tubing only
    e_rock: float, np.ndarray or list, optional
        [GPa] Young's modulus of rock; for mode 1 + cemented tubing only

    Fixed:
    mode: int, np.ndarray or list
        well mode type: 1, 2, 4
    tubing_cement_flag: boolean, np.ndarray or list
        toggle for cemented tubing; default = **False**
    casing_cement_flag: boolean, np.ndarray or list
        toggle for cemented casing; default = **False**

    Returns
    -------
    strain_tubing : float, np.ndarray
        [%] tubing shear strain
    strain_casing : float, np.ndarray
        [%] casing shear strain
    sigma_strain_tubing : float, np.ndarray
        aleatory variability for ln(strain_tubing)
    sigma_strain_casing : float, np.ndarray
        aleatory variability for ln(strain_casing)
    
    References
    ----------
    .. [1] Sasaki, T. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    NAME = 'Sasaki et al. (2022)'              # Name of the model
    ABBREV = None                      # Abbreviated name of the model
    REF = "".join([                     # Reference for the model
        'Sasaki, T. et al., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'strain_casing': {
                'desc': 'casing shear strain (%)',
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
            'strain_tubing': {
                'desc': 'tubing shear strain (%)',
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
            # 'sigma_strain_tubing': {
            #     'desc': 'aleatory variability for ln(strain_casing)',
            #     'unit': '',
            #     'mean': None,
            # },
            # 'sigma_strain_casing': {
            #     'desc': 'aleatory variability for ln(strain_tubing)',
            #     'unit': '',
            #     'mean': None,
            # },
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'EDP',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': 'pgdef'        # Input variable for PBEE category, e.g., pgdef, strain_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': None,
                # 'dist_type': 'lognormal'
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 1
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'phi_cmt': 'friction angle of cement (deg) - for mode 2 + cemented tubing only',
            'ucs_cmt': 'uniaxial compressive strength of cement (MPa) - for mode 4 + cemented tubing only',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'theta': 'fault angle (deg) - for all cases (modes + cementation)',
            'w_fc': 'fault core width (m) - for all modes + cemented tubing only',
            'w_dz': 'damage zone width (m) - for all modes + cemented tubing only',
            'z_crossing': 'depth to fault crossing (m)',
            'e_rock': "Young's modulus of rock (GPa) - for mode 1 + cemented tubing only",
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'z_top_of_cement': 'depth to top of cement in well (m)',
            'd_production_boring': 'diameter of production boring (m)',
            'd_production_casing': 'outer diameter of production casing (m)',
            'd_tubing': 'outer diameter of tubing (m)',
            'casing_flow': 'flag for whether well is configured for casing flow (True/False)',
            # 'mode': 'well mode type: 1, 2, 4',
            # 'cement_flag': 'cemented casing/tubing (True/False)',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'phi_cmt', 'ucs_cmt', 'theta', 'w_fc', 'w_dz', 'z_crossing', 'e_rock'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        # 'mode', 'cement_flag',
        'z_top_of_cement', 'd_production_boring', 'd_production_casing', 'd_tubing', 'casing_flow'
    }
    # _MODEL_INTERNAL = {
    #     'n_sample': 1,
    #     'n_site': 1,
    # }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {
        'mode_1_uncemented_tubing': {
            'b0': {'mean': -22.79563382 , 'sigma': 0.50913421},
            'b1': {'mean':   0.139049469, 'sigma': 0.01620919},
            'b2': {'mean':  -0.004700783, 'sigma': 0.00020684},
            'b3': {'mean': 112.297895   , 'sigma': 3.60563693},
            'b4': {'mean': -90.95872023 , 'sigma': 4.95632217},
        },
        'mode_1_cemented_tubing': {
            'b0': {'mean':  0.068742036, 'sigma': 0.04182211},
            'b1': {'mean':  0.007101637, 'sigma': 0.00073071},
            'b2': {'mean': -0.0001272  , 'sigma': 8.38E-06  },
            'b3': {'mean': -5.468137779, 'sigma': 0.20376311},
            'b4': {'mean': -0.070454446, 'sigma': 0.00475759},
            'b5': {'mean': -0.015280032, 'sigma': 0.00629218},
            'b6': {'mean':  0.000655275, 'sigma': 0.00027682},
            'b7': {'mean':  7.982903586, 'sigma': 0.04889402},
        },
        'mode_1_uncemented_casing': {
            'b0': {'mean':  1.277194854, 'sigma': 0.10106733},
            'b1': {'mean':  0.003094863, 'sigma': 0.0010972 },
            'b2': {'mean': -0.0001421  , 'sigma': 0.00001312},
            'b3': {'mean':  0.326505755, 'sigma': 0.12206713},
            'b4': {'mean': -0.103797413, 'sigma': 0.03340548},
        },
        'mode_1_cemented_casing': {
            'b0': {'mean': -8.27553775 , 'sigma': 2.58743956},
            'b1': {'mean':  1.806290126, 'sigma': 0.11228552},
            'b2': {'mean': -0.02291701 , 'sigma': 0.00116379},
        },
        'mode_2_uncemented_tubing': {
            'b0': {'mean':  -17.19529023 , 'sigma': 0.45425985},
            'b1': {'mean':    0.131966584, 'sigma': 0.01508432},
            'b2': {'mean':   -0.003913075, 'sigma': 0.00018845},
            'b3': {'mean':  105.0307884  , 'sigma': 4.16549102},
            'b4': {'mean': -104.0647275  , 'sigma': 7.69119745},
        },
        'mode_2_cemented_tubing': {
            'b0': {'mean': -0.014463229, 'sigma': 0.03120066},
            'b1': {'mean':  0.001322172, 'sigma': 0.00102841},
            'b2': {'mean': -0.000037928, 'sigma': 1.18E-05  },
            'b3': {'mean': -7.125630426, 'sigma': 0.27013422},
            'b4': {'mean':  0.025936262, 'sigma': 0.00601251},
            'b5': {'mean':  0.004351959, 'sigma': 0.00082907},
            'b6': {'mean':  7.985906205, 'sigma': 0.06276379},
        },
        'mode_2_uncemented_casing': {
            'b0': {'mean':  1.063469129, 'sigma': 0.12874534},
            'b1': {'mean':  0.00325468 , 'sigma': 0.00108402},
            'b2': {'mean': -0.000126915, 'sigma': 1.29E-05  },
            'b3': {'mean':  0.150758586, 'sigma': 0.13782272},
            'b4': {'mean': -0.11222946 , 'sigma': 0.0339996 },
        },
        'mode_2_cemented_casing': {
            'b0': {'mean': 11.61794753 , 'sigma': 2.80661122},
            'b1': {'mean':  0.768008296, 'sigma': 0.12012021},
            'b2': {'mean': -0.011210508, 'sigma': 0.00124526},
        },
        'mode_4_uncemented_tubing': {
            'b0': {'mean': -20.95932681 , 'sigma': 0.6306581 },
            'b1': {'mean':   0.193476649, 'sigma': 0.01936461},
            'b2': {'mean':  -0.00514883 , 'sigma': 0.00024557},
            'b3': {'mean': 109.6198155  , 'sigma': 5.09033271},
            'b4': {'mean': -93.05359314 , 'sigma': 7.90741713},
        },
        'mode_4_cemented_tubing': {
            'b0': {'mean': -0.131910173, 'sigma': 0.03100048},
            'b1': {'mean':  0.006023247, 'sigma': 0.00111964},
            'b2': {'mean': -0.000057215, 'sigma': 1.28E-05  },
            'b3': {'mean': -6.623803917, 'sigma': 0.27934754},
            'b4': {'mean': -0.013661116, 'sigma': 0.00624288},
            'b5': {'mean':  0.00223905 , 'sigma': 0.00031597},
            'b6': {'mean':  9.01981912 , 'sigma': 0.08368002},
        },
        'mode_4_uncemented_casing': {
            'b0': {'mean':  1.41667918 , 'sigma': 0.10794909},
            'b1': {'mean':  0.0049175  , 'sigma': 0.00077548},
            'b2': {'mean': -0.000135388, 'sigma': 9.79E-06  },
            'b3': {'mean':  0.750017405, 'sigma': 0.12695711},
            'b4': {'mean':  0.044507392, 'sigma': 0.03452467},
        },
        'mode_4_cemented_casing': {
            'b0': {'mean': 21.99907037 , 'sigma': 0.59234276},
            'b1': {'mean': -0.175879321, 'sigma': 0.01201327},
        },
    }
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': {
            'mode_1_uncemented_tubing': \
                lambda b0, b1, b2, b3, b4, theta, pgdef: \
                np.exp(b0 + b1*theta + b2*theta**2 + \
                    b3*np.minimum(pgdef,-112.297895/(2*-90.95872023)) + \
                    b4*np.minimum(pgdef,-112.297895/(2*-90.95872023))**2),
            'mode_1_cemented_tubing': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, theta, w_fc, w_dz, e_rock, pgdef: \
                b0 + b1*theta + b2*theta**2 + b3*w_fc + b4*w_dz + b5*e_rock + b6*e_rock**2 + \
                b7*(pgdef - (0.1605 - 0.0040*theta + 7e-5*theta**2)),
            'mode_1_uncemented_casing': \
                lambda b0, b1, b2, b3, b4, theta, pgdef: \
                b0 + b1*theta + b2*theta**2 + \
                b3*np.log(np.minimum(pgdef,np.exp(-0.326505755/(2*-0.103797413)))) + \
                b4*np.log(np.minimum(pgdef,np.exp(-0.326505755/(2*-0.103797413))))**2,
            'mode_1_cemented_casing': \
                lambda b0, b1, b2, theta, pgdef: \
                (b0 + b1*theta + b2*theta**2) * pgdef,
            'mode_2_uncemented_tubing': \
                lambda b0, b1, b2, b3, b4, theta, pgdef: \
                # np.exp(b0 + b1*theta + b2*theta**2 + b3*pgdef + b4*pgdef**2),
                np.exp(b0 + b1*theta + b2*theta**2 + \
                    b3*np.minimum(pgdef,-105.0307884/(2*-104.0647275)) + \
                    b4*np.minimum(pgdef,-105.0307884/(2*-104.0647275))**2),
            'mode_2_cemented_tubing': \
                lambda b0, b1, b2, b3, b4, b5, b6, theta, w_fc, w_dz, phi_cmt, pgdef: \
                np.exp(b0 + b1*theta + b2*theta**2 + b3*w_fc + b4*w_dz + b5*phi_cmt + \
                b6*(pgdef - (0.1051 - 0.0026*theta + 5e-5*theta**2))),
            'mode_2_uncemented_casing': \
                lambda b0, b1, b2, b3, b4, theta, pgdef: \
                b0 + b1*theta + b2*theta**2 + \
                b3*np.log(np.minimum(pgdef,-0.150758586/(2*-0.11222946))) + \
                b4*np.log(np.minimum(pgdef,-0.150758586/(2*-0.11222946)))**2,
            'mode_2_cemented_casing': \
                lambda b0, b1, b2, theta, pgdef: \
                (b0 + b1*theta + b2*theta**2) * pgdef,
            'mode_4_uncemented_tubing': \
                lambda b0, b1, b2, b3, b4, theta, pgdef: \
                np.exp(b0 + b1*theta + b2*theta**2 + \
                    b3*np.minimum(pgdef,-109.6198155/(2*-93.05359314)) + \
                    b4*np.minimum(pgdef,-109.6198155/(2*-93.05359314))**2),
            'mode_4_cemented_tubing': \
                lambda b0, b1, b2, b3, b4, b5, b6, theta, w_fc, w_dz, ucs_cmt, pgdef: \
                b0 + b1*theta + b2*theta**2 + b3*w_fc + b4*w_dz + b5*ucs_cmt + \
                b6*(pgdef - (0.0982 - 0.0023*theta + 5e-5*theta**2)),
            'mode_4_uncemented_casing': \
                lambda b0, b1, b2, b3, b4, theta, pgdef: \
                b0 + b1*theta + b2*theta**2 + \
                b3*np.log(pgdef) + \
                b4*np.log(pgdef)**2,
            'mode_4_cemented_casing': \
                lambda b0, b1, theta, pgdef: \
                (b0 + b1*theta) * pgdef,
        },
        'sigma': {
            'mode_1_uncemented_tubing': 0.554448,
            'mode_1_cemented_tubing': 0.0758868,
            'mode_1_uncemented_casing': 0.0358501,
            'mode_1_cemented_casing': 0.0366267,
            'mode_2_uncemented_tubing': 0.5203696,
            'mode_2_cemented_tubing': 0.1015194,
            'mode_2_uncemented_casing': 0.0306518,
            'mode_2_cemented_casing': 0.0399986,
            'mode_4_uncemented_tubing': 0.6253589,
            'mode_4_cemented_tubing': 0.092901,
            'mode_4_uncemented_casing': 0.0184673,
            'mode_4_cemented_casing': 0.0290133,
        },
        'sigma_mu': {
            'mode_1_uncemented_tubing': 0.25,
            'mode_1_cemented_tubing': 0.25,
            'mode_1_uncemented_casing': 0.25,
            'mode_1_cemented_casing': 0.25,
            'mode_2_uncemented_tubing': 0.25,
            'mode_2_cemented_tubing': 0.25,
            'mode_2_uncemented_casing': 0.25,
            'mode_2_cemented_casing': 0.25,
            'mode_4_uncemented_tubing': 0.25,
            'mode_4_cemented_tubing': 0.25,
            'mode_4_uncemented_casing': 0.25,
            'mode_4_cemented_casing': 0.25,
        },
        'dist_type': {
            'mode_1_uncemented_tubing': 'lognormal',
            'mode_1_cemented_tubing': 'normal',
            'mode_1_uncemented_casing': 'normal',
            'mode_1_cemented_casing': 'normal',
            'mode_2_uncemented_tubing': 'lognormal',
            'mode_2_cemented_tubing': 'lognormal',
            'mode_2_uncemented_casing': 'normal',
            'mode_2_cemented_casing': 'normal',
            'mode_4_uncemented_tubing': 'lognormal',
            'mode_4_cemented_tubing': 'normal',
            'mode_4_uncemented_casing': 'normal',
            'mode_4_cemented_casing': 'normal',
        },
        'func_string': {},
        'string': {}
    }
    

    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    # @classmethod
    # def get_req_rv_and_fix_params(cls, kwargs):
    #     """determine what model parameters to use"""
    #     # mode = kwargs.get('mode')
    #     # tubing_cement_flag = kwargs.get('tubing_cement_flag')
    #     # casing_cement_flag = kwargs.get('casing_cement_flag')
    #     # cement_flag = kwargs.get('cement_flag')
    #     # get inputs
    #     z_top_of_cement = kwargs.get('z_top_of_cement',None)
    #     z_crossing = kwargs.get('z_crossing',None)
    #     d_production_boring = kwargs.get('d_production_boring',None)
    #     d_production_casing = kwargs.get('d_production_casing',None)
    #     d_tubing = kwargs.get('d_tubing',None)
    #     casing_flow = kwargs.get('casing_flow',None)
    #     # get well modes and cement flag
    #     mode, cement_flag = cls.get_well_mode(
    #         z_top_of_cement, d_production_boring, d_production_casing, d_tubing, casing_flow
    #     )
    #     # modes present
    #     modes = [num for num in [1, 2, 4] if num in mode]
    #     # find indices for cases with cement
    #     ind_cement = {
    #         # key: np.where(cement_flag==val)[0]
    #         key: cement_flag==val
    #         for key,val in {'cemented': True, 'uncemented': False}.items()
    #         # if len(np.where(cement_flag==val)[0]) > 0
    #         if True in (cement_flag==val)
    #     }
    #     # # tubing cementation cases
    #     # tubing_cement = [
    #     #     key for key,val in {'cemented': True, 'uncemented': False}.items()
    #     #     if val in tubing_cement_flag
    #     # ]
    #     # # casing cementation cases
    #     # casing_cement = [
    #     #     key for key,val in {'cemented': True, 'uncemented': False}.items()
    #     #     if val in casing_cement_flag
    #     # ]
    #     # loop through modes, then tubing/casing, then cementation condition
    #     req_rvs_by_level = []
    #     req_fixed_by_level = []
    #     # make list of coefficients to get track
    #     coeffs = [f"b{i}" for i in range(10)]
    #     # for different modes
    #     for num in modes:
    #         for key in ind_cement:
    #             # if len(ind_cement[key]) > 0:
    #             for part in ['tubing', 'casing']:
    #                 case = f'mode_{num}_{key}_{part}'
    #                 req_rvs_by_level += [
    #                     param for param in cls._MODEL_FORM['func'][case].__code__.co_varnames
    #                     if not param in coeffs
    #                 ]
    #     # for num in modes:
    #     #     for part in ['tubing', 'casing']:
    #     #         for key in ['uncemented', 'cemented']:
    #     #             if key in locals()[f'{part}_cement']:
    #     #                 case = f'mode_{num}_{key}_{part}'
    #     #                 req_rvs_by_level += [
    #     #                     param for param in cls._MODEL_FORM['func'][case].__code__.co_varnames
    #     #                     if not param in coeffs
    #     #                 ]
    #     req_rvs_by_level = sorted(list(set(req_rvs_by_level)))
    #     req_fixed_by_level = cls._REQ_MODEL_FIXED_FOR_LEVEL
    #     return req_rvs_by_level, req_fixed_by_level
    

    @classmethod
    def get_well_mode(
        cls,
        z_top_of_cement,
        z_crossing,
        d_production_boring,
        d_production_casing,
        d_tubing,
        casing_flow
    ):
        """determine the well mode based on diameters and flow configuration"""
        # intermediate calculation
        d_production_boring_inch = d_production_boring *100/2.54 # meter to inch
        d_production_casing_inch = d_production_casing *100/2.54 # meter to inch
        d_tubing_inch = d_tubing *100/2.54 # meter to inch
        # get diameter differences
        boring_casing = d_production_boring_inch - d_production_casing_inch
        boring_tubing = d_production_boring_inch - d_tubing_inch
        casing_tubing = d_production_casing_inch - d_tubing_inch
        # boring-casing checks
        boring_casing_closest_mode_1 = np.abs(boring_casing-(3+5/8))
        boring_casing_closest_mode_2 = np.abs(boring_casing-(2+7/8))
        boring_casing_closest_mode_4 = np.abs(boring_casing-(4+0/8))
        
        boring_casing_argmin_stack = np.empty((3,*casing_flow.shape))
        boring_casing_argmin_stack[0] = boring_casing_closest_mode_1
        boring_casing_argmin_stack[1] = boring_casing_closest_mode_2
        boring_casing_argmin_stack[2] = boring_casing_closest_mode_4
        boring_casing_argmin = np.argmin(boring_casing_argmin_stack, axis=0)
        # boring_casing_argmin = np.argmin(np.vstack((
        #         boring_casing_closest_mode_1,
        #         boring_casing_closest_mode_2,
        #         boring_casing_closest_mode_4,
        #     )), axis=0)
        
        boring_casing_control_mode = np.zeros(casing_flow.shape)
        boring_casing_control_mode[boring_casing_argmin==0] = 1
        boring_casing_control_mode[boring_casing_argmin==1] = 2
        boring_casing_control_mode[boring_casing_argmin==2] = 4
        # boring-tubing checks
        boring_tubing_closest_mode_1 = np.abs(boring_tubing-(8+3/4))
        boring_tubing_closest_mode_2 = np.abs(boring_tubing-(6+3/8))
        boring_tubing_closest_mode_4 = np.abs(boring_tubing-(7+3/4))
        
        boring_tubing_argmin_stack = np.empty((3,*casing_flow.shape))
        boring_tubing_argmin_stack[0] = boring_tubing_closest_mode_1
        boring_tubing_argmin_stack[1] = boring_tubing_closest_mode_2
        boring_tubing_argmin_stack[2] = boring_tubing_closest_mode_4
        boring_tubing_argmin = np.argmin(boring_tubing_argmin_stack, axis=0)
        # boring_tubing_argmin = np.argmin(np.vstack((
        #         boring_tubing_closest_mode_1,
        #         boring_tubing_closest_mode_2,
        #         boring_tubing_closest_mode_4,
        #     )), axis=0)
        
        boring_tubing_control_mode = np.zeros(casing_flow.shape)
        boring_tubing_control_mode[boring_tubing_argmin==0] = 1
        boring_tubing_control_mode[boring_tubing_argmin==1] = 2
        boring_tubing_control_mode[boring_tubing_argmin==2] = 4
        # casing-tubing checks
        casing_tubing_closest_mode_1 = np.abs(casing_tubing-(5+1/8))
        casing_tubing_closest_mode_2 = np.abs(casing_tubing-(3+1/2))
        casing_tubing_closest_mode_4 = np.abs(casing_tubing-(3+3/4))
        
        casing_tubing_argmin_stack = np.empty((3,*casing_flow.shape))
        casing_tubing_argmin_stack[0] = casing_tubing_closest_mode_1
        casing_tubing_argmin_stack[1] = casing_tubing_closest_mode_2
        casing_tubing_argmin_stack[2] = casing_tubing_closest_mode_4
        casing_tubing_argmin = np.argmin(casing_tubing_argmin_stack, axis=0)
        # casing_tubing_argmin = np.argmin(np.vstack((
        #         casing_tubing_closest_mode_1,
        #         casing_tubing_closest_mode_2,
        #         casing_tubing_closest_mode_4,
        #     )), axis=0)
        
        casing_tubing_control_mode = np.zeros(casing_flow.shape)
        casing_tubing_control_mode[casing_tubing_argmin==0] = 1
        casing_tubing_control_mode[casing_tubing_argmin==1] = 2
        casing_tubing_control_mode[casing_tubing_argmin==2] = 4
        # get cement flag for cementation at fault depth
        cement_flag = cls.get_cement_flag(z_top_of_cement,z_crossing)
        cement_flag_true = cement_flag==True
        cement_flag_false = ~cement_flag_true
        # condtions for whether well is configured for casing flow
        casing_flow_true = casing_flow==True
        casing_flow_false = ~casing_flow_true
        # determine well mode
        mode = np.ones(d_tubing.shape)*2 # default to 2 to simplify check
        # -> if cemented at fault depth
        if True in cement_flag_true:
            # ---> if well is configured for casing flow
            joint_cond = np.logical_and(cement_flag_true,casing_flow_true)
            if True in joint_cond:
                mode[np.logical_and(joint_cond,d_production_casing_inch>=8+5/8)] = 1
                mode[np.logical_and(joint_cond,d_production_casing_inch<=6+5/8)] = 4
            # ---> if well is not configured for casing flow
            joint_cond = np.logical_and(cement_flag_true,casing_flow_false)
            if True in joint_cond:
                mode[joint_cond] = casing_tubing_control_mode[joint_cond]
        # -> if not cemented at fault depth
        if True in cement_flag_false:
            # ---> if well is configured for casing flow
            joint_cond = np.logical_and(cement_flag_false,casing_flow_true)
            if True in joint_cond:
                mode[joint_cond] = boring_casing_control_mode[joint_cond]
            # ---> if well is not configured for casing flow
            joint_cond = np.logical_and(cement_flag_false,casing_flow_false)
            if True in joint_cond:
                mode[joint_cond] = boring_tubing_control_mode[joint_cond]
        # return
        return mode, cement_flag        
    
    
    @staticmethod
    def get_cement_flag(
        z_top_of_cement,
        z_crossing
    ):
        """determine the cementation based on depth to cementation and fault crossing depth"""
        cement_flag = np.empty_like(z_top_of_cement,dtype=bool)
        cement_flag[z_top_of_cement>z_crossing] = False
        cement_flag[z_top_of_cement<=z_crossing] = True
        return cement_flag
    

    @classmethod
    # @njit
    def _model(cls, 
        pgdef, # upstream PBEE RV
        phi_cmt, ucs_cmt, # infrastructure
        theta, w_fc, w_dz, z_crossing, e_rock, # geotechnical/geologic
        # mode, cement_flag, # fixed/toggles
        z_top_of_cement, d_production_boring, d_production_casing, d_tubing, casing_flow, # fixed/toggles
        # mode, tubing_cement_flag, casing_cement_flag, # fixed/toggles
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""
        # initialize intermediate and output arrays
        # inflect = np.empty_like(pgdef)
        # strain_tubing = np.empty_like(pgdef)
        # strain_casing = np.empty_like(pgdef)
        strain_tubing = np.zeros(pgdef.shape)
        strain_casing = np.zeros(pgdef.shape)
        sigma_strain_tubing = np.zeros(pgdef.shape)
        sigma_strain_casing = np.zeros(pgdef.shape)
        sigma_mu_strain_tubing = np.zeros(pgdef.shape)
        sigma_mu_strain_casing = np.zeros(pgdef.shape)
        dist_type_strain_tubing = np.empty(pgdef.shape,dtype="<U10")
        dist_type_strain_casing = np.empty(pgdef.shape,dtype="<U10")
        
        # apply a pgdef cap of 0.25m to prevent model from blowing up
        # pgdef = np.minimum(pgdef,0.25)
        
        # get well modes and cement flag
        mode, cement_flag = cls.get_well_mode(
            z_top_of_cement, z_crossing, d_production_boring, d_production_casing, d_tubing, casing_flow
        )
        # cement_flag = cls.get_cement_flag(z_top_of_cement, z_crossing)
        
        # other params
        modes = [1,2,4]
        
        # determine cases
        # find indices for each mode
        ind_mode = {
            # num: np.where(mode==num)[0]
            num: mode==num
            for num in modes
        }
        # find indices for cases with cement
        ind_cement = {
            # key: np.where(cement_flag==val)[0]
            key: cement_flag==val
            for key,val in {'cemented': True, 'uncemented': False}.items()
            # if len(np.where(cement_flag==val)[0]) > 0
            if True in (cement_flag==val)
        }
        # ind_tubing = {
        #     key: np.where(tubing_cement_flag==val)[0]
        #     for key,val in {'cemented': True, 'uncemented': False}.items()
        # }
        # # find indices for cases with cemented casing
        # ind_casing = {
        #     key: np.where(casing_cement_flag==val)[0]
        #     for key,val in {'cemented': True, 'uncemented': False}.items()
        # }
        
        # loop through modes, then tubing/casing, then cementation condition
        for num in modes:
            # set_ind_1 = set(ind_mode[num])
            set_ind_1 = ind_mode[num]
            # run if current mode exist
            # if len(set_ind_1) > 0:
            if True in set_ind_1:
                for key in ind_cement:
                    # get indices for current mode and cementation flag
                    # set_ind_2 = set(locals()[f'ind_cement'][key])
                    # set_ind_2 = locals()[f'ind_cement'][key]
                    set_ind_2 = ind_cement[key]
                    # ind_joint = list(set_ind_1.intersection(set_ind_2))
                    # continue if length of intersection list is at least 1
                    set_ind_3 = theta>0 # only if fault angle > 0
                    ind_joint = (set_ind_1 & set_ind_2 & set_ind_3)
                    # if len(ind_joint) > 0:
                    if True in ind_joint:
                        # loop through well parts
                        for part in ['tubing', 'casing']:
                            # case name
                            case = f'mode_{num}_{key}_{part}'
                            # run calc using lambda function
                            locals()[f'strain_{part}'][ind_joint] = cls._MODEL_FORM['func'][case](
                                # mean coefficients
                                **cls._get_mean_coeff_for_lambda_func(
                                    cls._MODEL_FORM_DETAIL[case],
                                    cls._MODEL_FORM['func'][case]
                                ),
                                # inputs
                                **cls._get_kwargs_for_lambda_func(
                                    locals(),
                                    cls._MODEL_FORM['func'][case],
                                    inds=ind_joint
                                )
                            )
                            # get sigma
                            locals()[f'sigma_strain_{part}'][ind_joint] = cls._MODEL_FORM['sigma'][case]
                            # get sigma_mu
                            locals()[f'sigma_mu_strain_{part}'][ind_joint] = cls._MODEL_FORM['sigma_mu'][case]
                            # get dist_type
                            locals()[f'dist_type_strain_{part}'][ind_joint] = cls._MODEL_FORM['dist_type'][case]
        
        
        # some contraints on mean
        strain_tubing = np.minimum(np.maximum(strain_tubing * 100,1e-5),200) # convert to %, limit to 0 to 200%
        strain_casing = np.minimum(np.maximum(strain_casing * 100,1e-5),200) # convert to %, limit to 0 to 200%

        # if dist_type is normal, then convert sigma to %
        sigma_strain_tubing[dist_type_strain_tubing=='normal'] = \
            sigma_strain_tubing[dist_type_strain_tubing=='normal']*100
        sigma_strain_casing[dist_type_strain_casing=='normal'] = \
            sigma_strain_casing[dist_type_strain_casing=='normal']*100
        # if dist_type is normal, then sigma_mu = exp(0.25), geometric sigma
        sigma_mu_strain_tubing[dist_type_strain_tubing=='normal'] = \
            np.exp(sigma_mu_strain_tubing[dist_type_strain_tubing=='normal'])
            # np.exp(sigma_mu_strain_tubing[dist_type_strain_tubing=='normal']) * strain_tubing[dist_type_strain_tubing=='normal']
        sigma_mu_strain_casing[dist_type_strain_casing=='normal'] = \
            np.exp(sigma_mu_strain_casing[dist_type_strain_casing=='normal'])
            # np.exp(sigma_mu_strain_casing[dist_type_strain_casing=='normal']) * strain_casing[dist_type_strain_casing=='normal']

        # limit sigma to be above 0 to avoid PC error
        sigma_strain_tubing[theta<=0] = 0.001
        sigma_strain_casing[theta<=0] = 0.001
        # limit sigma to be above 0 to avoid PC error
        sigma_mu_strain_tubing[theta<=0] = 0
        sigma_mu_strain_casing[theta<=0] = 0
            
            # run if current mode exist
            # if len(set_ind_1) > 0:
            #     for part in ['tubing', 'casing']:
            #         for key in ['uncemented', 'cemented']:
            #             # case name
            #             case = f'mode_{num}_{key}_{part}'
            #             # get indices for current mode, well part, and cementation flag
            #             set_ind_2 = set(locals()[f'ind_{part}'][key])
            #             ind_joint = list(set_ind_1.intersection(set_ind_2))
            #             # continue if length of intersection list is at least 1
            #             if len(ind_joint) > 0:
            #                 # run calc using lambda function
            #                 locals()[f'strain_{part}'][ind_joint] = cls._MODEL_FORM['func'][case](
            #                     # mean coefficients
            #                     **cls._get_mean_coeff_for_lambda_func(
            #                         cls._MODEL_FORM_DETAIL[case],
            #                         cls._MODEL_FORM['func'][case]
            #                     ),
            #                     # inputs
            #                     **cls._get_kwargs_for_lambda_func(
            #                         locals(),
            #                         cls._MODEL_FORM['func'][case],
            #                         inds=ind_joint
            #                     )
            #                 )

        # prepare outputs
        output = {
            'strain_tubing': {
                'mean': strain_tubing,
                # 'mean': np.minimum(np.maximum(strain_tubing * 100,0),200),
                # 'mean': np.maximum(strain_tubing * 100,0), # convert to %, limit to 0 to 500%
                'sigma': sigma_strain_tubing,
                'sigma_mu': sigma_mu_strain_tubing,
                # 'dist_type': 'lognormal',
                'dist_type': dist_type_strain_tubing,
                'unit': '%'
            },
            'strain_casing': {
                'mean': strain_casing, # convert to %, limit to 0 to 500%
                # 'mean': np.minimum(np.maximum(strain_casing * 100,0),200), # convert to %, limit to 0 to 500%
                # 'mean': np.maximum(strain_casing * 100,0), # convert to %, limit to 0 to 500%
                'sigma': sigma_strain_casing,
                'sigma_mu': sigma_mu_strain_casing,
                # 'dist_type': 'lognormal',
                'dist_type': dist_type_strain_casing,
                'unit': '%'
            },
            # 'strain_tubing': strain_tubing*100, # convert to %
            # 'strain_casing': strain_casing*100, # convert to %
            # 'sigma_strain_tubing': sigma_strain_tubing,
            # 'sigma_strain_casing': sigma_strain_casing,
        }
        # get intermediate values if requested
        if return_inter_params:
            output['mode'] = mode
            output['cement_flag'] = cement_flag
            output['dist_type_strain_tubing'] = dist_type_strain_tubing
            output['dist_type_strain_casing'] = dist_type_strain_casing
        
        # return
        return output