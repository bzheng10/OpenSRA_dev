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
class WellMoment(BaseModel):
    "Inherited class specfic to well strain"

    # _RETURN_PBEE_META = {
    #     'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'well moment',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'moment_conductor',
    #         'moment_surface',
    #         'moment_production',
    #         'moment_tubing',
    #     ] # Return variable for PBEE category, e.g., pgdef, eps_p
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class LuuEtal2022(WellMoment):
    """
    Compute ground-shaking-induced casing and tubing moments using Luu et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration

    Infrastructure:
    h_wh: float, np.ndarray or list
        [m] wellhead height, for all modes
    mpl_wh: float, np.ndarray or list, optional
        [kg/m] wellhead mass per length, for all modes
    
    Geotechnical/geologic:
    phi_soil: float, np.ndarray or list, optional
        [deg] soil friction angle, modes 1 and 2 only

    Fixed:
    mode: int, np.ndarray or list
        well mode type: 1, 2, 4
    # conductor_flag: boolean, np.ndarray or list
    #     if conductor casing is present (True/False)
    # surface_flag: boolean, np.ndarray or list
    #     if surface casing is present (True/False)
    # production_flag: boolean, np.ndarray or list
    #     if production casing is present (True/False)

    Returns
    -------
    moment_conductor : float, np.ndarray
        [N-m] moment for conductor casing
    moment_surface : float, np.ndarray
        [N-m] moment for surface casing
    moment_production : float, np.ndarray
        [N-m] moment for production casing
    moment_tubing : float, np.ndarray
        [N-m] moment for tubing
    sigma_moment_conductor : float, np.ndarray
        aleatory variability for ln(moment_conductor)
    sigma_moment_surface : float, np.ndarray
        aleatory variability for ln(moment_surface)
    sigma_moment_production : float, np.ndarray
        aleatory variability for ln(moment_production)
    sigma_moment_tubing : float, np.ndarray
        aleatory variability for ln(moment_tubing)
    
    References
    ----------
    .. [1] Luu, K. and others, 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
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
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'moment_conductor': {
                'desc': 'moment for conductor casing (N-m)',
                'unit': 'N-m',
            },
            'moment_surface': {
                'desc': 'moment for surface casing (N-m)',
                'unit': 'N-m',
            },
            'moment_production': {
                'desc': 'moment for production casing (N-m)',
                'unit': 'N-m',
            },
            'moment_tubing': {
                'desc': 'moment for tubing (N-m)',
                'unit': 'N-m',
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
            'h_wh': 'wellhead height (m), for all modes',
            'mpl_wh': 'wellhead mass per length (kg/m), for all modes',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'phi_soil': 'soil friction angle (deg), modes 1 and 2 only',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'mode': 'well mode type: 1, 2, 4',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'h_wh', 'mpl_wh', 'phi_soil'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'mode'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {
        'mode_1_conductor': {
            'b0': {'mean': 10.76168    , 'sigma': 0.123135 },
            'b1': {'mean':  0.658693   , 'sigma': 0.073107 },
            'b2': {'mean': -0.066265051, 'sigma': 0.011974 },
            'b3': {'mean':  0.001758   , 'sigma': 0.001685 },
            'b4': {'mean':  0          , 'sigma': 0.       },
            'b5': {'mean':  0.001374   , 'sigma': 4.25E-05 },
            'b6': {'mean': -3.74E-07   , 'sigma': 1.95E-08 },
            'b7': {'mean':  0.632604   , 'sigma': 0.003829 },
            'b8': {'mean': -0.019615648, 'sigma': 0.002022 },
        },
        'mode_1_surface': {
            'b0': {'mean':  9.527197057, 'sigma': 0.1250119 },
            'b1': {'mean':  0.665763982, 'sigma': 0.08460407},
            'b2': {'mean': -0.063815828, 'sigma': 0.013857  },
            'b3': {'mean':  0          , 'sigma': 0.        },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.001426671, 'sigma': 4.92E-05  },
            'b6': {'mean': -3.83E-07   , 'sigma': 2.26E-08  },
            'b7': {'mean':  0.696715699, 'sigma': 0.00443176},
            'b8': {'mean':  0.008165592, 'sigma': 0.00234013},
        },
        'mode_1_production': {
            'b0': {'mean':  8.026536772, 'sigma': 0.13269488},
            'b1': {'mean':  0.68340017 , 'sigma': 0.08980367},
            'b2': {'mean': -0.064411022, 'sigma': 0.01470862},
            'b3': {'mean':  0          , 'sigma': 0.        },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.001475629, 'sigma': 5.22E-05  },
            'b6': {'mean': -3.91E-07   , 'sigma': 2.40E-08  },
            'b7': {'mean':  0.753005854, 'sigma': 0.00470413},
            'b8': {'mean':  0.027774763, 'sigma': 0.00248395},
        },
        'mode_1_tubing': {
            'b0': {'mean': 10.7548858  , 'sigma': 0.08930272},
            'b1': {'mean': -0.658186485, 'sigma': 0.06043723},
            'b2': {'mean':  0.106031029, 'sigma': 0.0098988 },
            'b3': {'mean':  0          , 'sigma': 0.        },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.000653493, 'sigma': 3.51E-05  },
            'b6': {'mean': -1.96E-07   , 'sigma': 1.61E-08  },
            'b7': {'mean':  0.263827344, 'sigma': 0.00316585},
            'b8': {'mean': -0.136420913, 'sigma': 0.00167168},
        },
        'mode_2_conductor': {
            'b0': {'mean': 15.360203887, 'sigma': 1.36737252},
            'b1': {'mean':  1.080489976, 'sigma': 0.06984432},
            'b2': {'mean': -0.136677368, 'sigma': 0.01141489},
            'b3': {'mean': -0.262816399, 'sigma': 0.07646939},
            'b4': {'mean':  0.003674085, 'sigma': 0.00106167},
            'b5': {'mean':  0.000551472, 'sigma': 6.87E-06  },
            'b6': {'mean':  0          , 'sigma': 0.        },
            'b7': {'mean':  0.611786781, 'sigma': 0.00370157},
            'b8': {'mean': -0.025841966, 'sigma': 0.00195457},
        },
        'mode_2_surface': {
            'b0': {'mean':  8.480328407, 'sigma': 0.12720121},
            'b1': {'mean':  1.119292226, 'sigma': 0.08464293},
            'b2': {'mean': -0.13830766 , 'sigma': 0.01384422},
            'b3': {'mean':  0          , 'sigma': 0.        },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.0006089  , 'sigma': 8.35E-06  },
            'b6': {'mean':  0          , 'sigma': 0.        },
            'b7': {'mean':  0.722273817, 'sigma': 0.00450938},
            'b8': {'mean':  0.015596479, 'sigma': 0.00238112},
        },
        'mode_2_production': {
            'b0': {'mean':  6.964497847, 'sigma': 0.13602472},
            'b1': {'mean':  1.147248993, 'sigma': 0.09051432},
            'b2': {'mean': -0.140938628, 'sigma': 0.01480455},
            'b3': {'mean':  0          , 'sigma': 0.        },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.000640423, 'sigma': 8.93E-06  },
            'b6': {'mean':  0          , 'sigma': 0.        },
            'b7': {'mean':  0.778036032, 'sigma': 0.00482219},
            'b8': {'mean':  0.035062354, 'sigma': 0.00254629},
        },
        'mode_2_tubing': {
            'b0': {'mean': 10.435500516, 'sigma': 0.08033476},
            'b1': {'mean': -0.519328984, 'sigma': 0.054368  },
            'b2': {'mean':  0.081967163, 'sigma': 0.00890474},
            'b3': {'mean':  0          , 'sigma': 0.        },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.000582685, 'sigma': 0.00003161},
            'b6': {'mean': -1.74334E-07, 'sigma': 1.45E-08  },
            'b7': {'mean':  0.195671799, 'sigma': 0.00284793},
            'b8': {'mean': -0.138078864, 'sigma': 0.00150381},
        },
        'mode_4_conductor': {
            'b0': {'mean': 10.54768231 , 'sigma': 0.1056165 },
            'b1': {'mean':  1.065652806, 'sigma': 0.07027991},
            'b2': {'mean': -0.139158606, 'sigma': 0.011495  },
            'b3': {'mean':  0.000559665, 'sigma': 6.94E-06  },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.606700068, 'sigma': 0.00374419},
            'b6': {'mean': -0.027203298, 'sigma': 0.00197707},
        },
        'mode_4_surface': {
            'b0': {'mean':  9.083362489, 'sigma': 0.12269242},
            'b1': {'mean':  0.8919632  , 'sigma': 0.08303432},
            'b2': {'mean': -0.105895366, 'sigma': 0.0135999 },
            'b3': {'mean':  0.001398292, 'sigma': 4.83E-05  },
            'b4': {'mean': -3.73E-07   , 'sigma': 2.22E-08  },
            'b5': {'mean':  0.688092449, 'sigma': 0.00434953},
            'b6': {'mean':  0.004312053, 'sigma': 0.00229671},
        },
        'mode_4_production': {
            'b0': {'mean':  7.027724127, 'sigma': 0.14233827},
            'b1': {'mean':  1.208893516, 'sigma': 0.09471551},
            'b2': {'mean': -0.155941745, 'sigma': 0.0154917 },
            'b3': {'mean':  0.00066118 , 'sigma': 9.35E-06  },
            'b4': {'mean':  0          , 'sigma': 0.        },
            'b5': {'mean':  0.808520392, 'sigma': 0.00504601},
            'b6': {'mean':  0.046310218, 'sigma': 0.00266447},
        },
        'mode_4_tubing': {
            'b0': {'mean':  9.398604091, 'sigma': 0.07410191},
            'b1': {'mean': -0.244062198, 'sigma': 0.05014981},
            'b2': {'mean':  0.051431037, 'sigma': 0.00821386},
            'b3': {'mean':  0.000556108, 'sigma': 2.92E-05  },
            'b4': {'mean': -1.61E-07   , 'sigma': 1.34E-08  },
            'b5': {'mean':  0.139840062, 'sigma': 0.00262697},
            'b6': {'mean': -0.119300337, 'sigma': 0.00138713},
        },
    }
    _MODEL_INPUT_RV = {}
    _MODEL_FORM = {
        'func': {
            'mode_1': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8,
                pga, h_wh, mpl_wh, phi_soil: \
                np.exp(b0 + b1*h_wh + b2*h_wh**2 + b3*phi_soil + b4*phi_soil**2 + \
                b5*mpl_wh + b6*mpl_wh**2 + b7*np.log(pga) + b8*np.log(pga)**2),
            'mode_2': \
                lambda b0, b1, b2, b3, b4, b5, b6, b7, b8,
                pga, h_wh, mpl_wh, phi_soil: \
                np.exp(b0 + b1*h_wh + b2*h_wh**2 + b3*phi_soil + b4*phi_soil**2 + \
                b5*mpl_wh + b6*mpl_wh**2 + b7*np.log(pga) + b8*np.log(pga)**2),
            'mode_4': \
                lambda b0, b1, b2, b3, b4, b5, b6,
                pga, h_wh, mpl_wh: \
                np.exp(b0 + b1*h_wh + b2*h_wh**2 + b3*mpl_wh + b4*mpl_wh**2 + \
                b5*np.log(pga) + b6*np.log(pga)**2),
        },
        'sigma': {
            'mode_1_conductor': 0.3351434,
            'mode_1_surface': 0.3878656,
            'mode_1_production': 0.4117031,
            'mode_1_tubing': 0.2770733,
            'mode_2_conductor': 0.3239599,
            'mode_2_surface': 0.3946591,
            'mode_2_production': 0.4220353,
            'mode_2_tubing': 0.2770733,
            'mode_4_conductor': 0.3276896,
            'mode_4_surface': 0.3806691,
            'mode_4_production': 0.4416239,
            'mode_4_tubing': 0.2299108,
        },
        'sigma_mu': {
            'mode_1_conductor': 0.25,
            'mode_1_surface': 0.25,
            'mode_1_production': 0.25,
            'mode_1_tubing': 0.25,
            'mode_2_conductor': 0.25,
            'mode_2_surface': 0.25,
            'mode_2_production': 0.25,
            'mode_2_tubing': 0.25,
            'mode_4_conductor': 0.25,
            'mode_4_surface': 0.25,
            'mode_4_production': 0.25,
            'mode_4_tubing': 0.25,
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
        mode = kwargs.get('mode')
        # modes present
        modes = [num for num in [1, 2, 4] if num in mode]
        # loop through modes, then tubing/casing, then cementation condition
        req_rvs_by_level = []
        req_fixed_by_level = []
        # make list of coefficients to get track
        coeffs = [f"b{i}" for i in range(10)]
        # 
        for num in modes:
            case = f'mode_{num}'
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
        h_wh, mpl_wh, # infrastructure
        phi_soil, # geotechnical/geologic
        mode, # fixed/toggles
        # mode, conductor_flag, surface_flag, production_flag, # fixed/toggles
        return_inter_params=False # to get intermediate params    
    ):
        """Model"""
        # initialize intermediate and output arrays
        # moment_conductor = np.empty_like(pga)
        # moment_surface = np.empty_like(pga)
        # moment_production = np.empty_like(pga)
        # moment_tubing = np.empty_like(pga)
        moment_conductor = np.zeros(pga.shape)
        moment_surface = np.zeros(pga.shape)
        moment_production = np.zeros(pga.shape)
        moment_tubing = np.zeros(pga.shape)
        sigma_moment_conductor = np.zeros(pga.shape)
        sigma_moment_surface = np.zeros(pga.shape)
        sigma_moment_production = np.zeros(pga.shape)
        sigma_moment_tubing = np.zeros(pga.shape)
        sigma_mu_moment_conductor = np.zeros(pga.shape)
        sigma_mu_moment_surface = np.zeros(pga.shape)
        sigma_mu_moment_production = np.zeros(pga.shape)
        sigma_mu_moment_tubing = np.zeros(pga.shape)
        
        # other params
        modes = [1,2,4]
        
        # determine cases
        # find indices for each mode
        ind_mode = {
            # num: np.where(mode==num)[0]
            num: mode==num
            for num in modes
        }
        # find indices for cases where casings are present
        # ind_part_present = {}
        # for part in ['conductor', 'surface', 'production']:
        #     ind_part_present[part] = np.where(locals()[f'{part}_flag']==True)[0]
            # print(part, np.where(locals()[f'{part}_flag']!=True)[0])
        
        # loop through modes, then tubing/casing, then cementation condition
        for num in modes:
            # set_ind_1 = set(ind_mode[num])
            # run if current mode exist
            # if len(set_ind_1) > 0:
            if len(ind_mode[num]) > 0:
                # case name
                model_case = f'mode_{num}'
                # the casings
                for part in ['conductor', 'surface', 'production', 'tubing']:
                # for part in ['surface', 'production', 'tubing']:
                    # continue if at least 1 well has part
                    # if len(ind_part_present[part]) > 0:
                    # model coeffs
                    coeff_case = f'mode_{num}_{part}'
                    # get indices for current mode, well part, and cementation flag
                    # set_ind_2 = set(ind_part_present[part])
                    # ind_joint = list(set_ind_1.intersection(set_ind_2))
                    # continue if length of intersection list is at least 1
                    # if len(ind_joint) > 0:
                    # run calc using lambda function
                    # locals()[f'moment_{part}'][ind_joint] = cls._MODEL_FORM['func'][model_case](
                    locals()[f'moment_{part}'][ind_mode[num]] = cls._MODEL_FORM['func'][model_case](
                        # mean coefficients
                        **cls._get_mean_coeff_for_lambda_func(
                            cls._MODEL_FORM_DETAIL[coeff_case],
                            cls._MODEL_FORM['func'][model_case]
                        ),
                        # inputs
                        **cls._get_kwargs_for_lambda_func(
                            locals(),
                            cls._MODEL_FORM['func'][model_case],
                            # inds=ind_joint
                            inds=ind_mode[num]
                        )
                    )
                    # get sigma
                    locals()[f'sigma_moment_{part}'][ind_mode[num]] = cls._MODEL_FORM['sigma'][coeff_case]
                    # get sigma_mu
                    locals()[f'sigma_mu_moment_{part}'][ind_mode[num]] = cls._MODEL_FORM['sigma_mu'][coeff_case]
                
                # # always run tubing
                # # model coeffs
                # coeff_case = f'mode_{num}_tubing'
                # ind_joint = list(set_ind_1)
                # # run calc using lambda function
                # moment_tubing[ind_joint] = cls._MODEL_FORM['func'][model_case](
                #     # mean coefficients
                #     **cls._get_mean_coeff_for_lambda_func(
                #         cls._MODEL_FORM_DETAIL[coeff_case],
                #         cls._MODEL_FORM['func'][model_case]
                #     ),
                #     # inputs
                #     **cls._get_kwargs_for_lambda_func(
                #         locals(),
                #         cls._MODEL_FORM['func'][model_case],
                #         inds=ind_joint
                #     )
                # )

        # prepare outputs
        output = {
            'moment_conductor': {
                'mean': moment_conductor,
                'sigma': sigma_moment_conductor,
                'sigma_mu': sigma_mu_moment_conductor,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            'moment_surface': {
                'mean': moment_surface,
                'sigma': sigma_moment_surface,
                'sigma_mu': sigma_mu_moment_surface,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            'moment_production': {
                'mean': moment_production,
                'sigma': sigma_moment_production,
                'sigma_mu': sigma_mu_moment_production,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            'moment_tubing': {
                'mean': moment_tubing,
                'sigma': sigma_moment_tubing,
                'sigma_mu': sigma_mu_moment_tubing,
                'dist_type': 'lognormal',
                'unit': 'N-m'
            },
            # 'moment_conductor': moment_conductor,
            # 'moment_surface': moment_surface,
            # 'moment_production': moment_production,
            # 'moment_tubing': moment_tubing,
            # 'sigma_moment_conductor': sigma_moment_conductor,
            # 'sigma_moment_surface': sigma_moment_surface,
            # 'sigma_moment_production': sigma_moment_production,
            # 'sigma_moment_tubing': sigma_moment_tubing,
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output