# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for landslide deformation
#
# Created: April 13, 2020
# Updated: July 14, 2020
# --- improved documentation
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import numpy as np
from scipy import sparse
from scipy.stats import norm

# OpenSRA modules
# import src.EDP.Fcn_EDP
from src.base_class import BaseModel


# -----------------------------------------------------------
class Landslide(BaseModel):
    "Inherited class specfic to landslide deformation"
    
    _RETURN_PBEE_META = {
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        'type': 'landslide',       # Type of model (e.g., liquefaction, landslide, pipe strain)
        'variable': [
            'pgdef',
        ] # Return variable for PBEE category, e.g., pgdef, eps_pipe
    }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class BrayMacedo2019(Landslide):
    """
    Compute landslide deformation at a given location using the Bray and Macedo (2007) probabilistic model.
    Regression models based on three sets of ground motions are provided:

    1. **Ordinary**: **d** = f(ky, Sa(T), Ts, M)
    2. **Near-fault**: **d** = f(ky, Sa(T), Ts, M, pgv) - unavailable for this version of OpenSRA
    3. **General** (default): **d** = f(ky, Sa(T), Ts, M, pgv) - unavailable for this version of OpenSRA

    The default relationship for **ky** uses **coh_soil**, **phi_soil**, **gamma_soil**, **t_slope**, **slope**

    **PGA** is used in place of **Sa(T)** (i.e., Ts=0)

    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
        
    Geotechnical/geologic:
    slope: float, np.ndarray or list
        [deg] slope angle
    t_slope: float, np.ndarray or list
        [m] slope thickness (infinite-slope problem)
    gamma_soil: float, np.ndarray or list
        [kN/m^3] unit weight of soil
    phi_soil: float, np.ndarray or list
        [deg] friction angle of soil
    coh_soil: float, np.ndarray or list
        [kPa] cohesion of soil
        
    Fixed:
    mag: float, np.ndarray or list
        moment magnitude

    Returns
    -------
    pgdef : float, np.ndarray or list
        [cm] permanent ground deformation
    
    References
    ----------
    .. [1] Bray, J.D., and Macedo, J., 2019, Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 145, pp. 12, 04019106.
    
    """

    _NAME = 'Bray and Macedo (2007)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Bray, J.D., and Macedo, J., 2019, ',
        'Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, ',
        'Journal of Geotechnical and Geoenvironmental Engineering, ',
        'vol. 145, pp. 12, 04019106.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (cm)',
                'unit': 'cm',
                'mean': None,
                'aleatory': 0.72,
                'epistemic': {
                    'coeff': 0.25, # base uncertainty, based on coeffcients
                    'input': None, # sigma_mu uncertainty from input parameters
                    'total': None # SRSS of coeff and input sigma_mu uncertainty
                },
                'dist_type': 'lognormal',
            }
        }
    }
    _INPUT_PBEE_META = {
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        'variable': [
            'pga'
        ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
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
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        "params": {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'slope': 'slope angle (deg)',
            't_slope': 'slope thickness (m)',
            'gamma_soil': 'unit weight of soil (kN/m^3)',
            'phi_soil': 'friction angle of soil (deg)',
            'coh_soil': 'cohesion of soil (kPa)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'mag': 'moment magnitude',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'beta', 't_beta', 'gamma_soil', 'phi_soil', 'coh_soil',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'mag',
    }
    _MODEL_INTERNAL = {
        'n_sample': 1,
        'n_site': 1,
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    @staticmethod
    # @njit
    def _model(
        pga, # upstream PBEE RV
        slope, t_slope, gamma_soil, phi_soil, coh_soil, # geotechnical/geologic
        mag, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # initialize arrays
        prob_d_eq_0 = np.empty(pga.shape)
        pgdef = np.empty(pga.shape)
        
        # convert from deg to rad
        slope_rad = slope*np.pi/180
        phi_soil_rad = phi_soil*np.pi/180
        
        # yield acceleration
        ky = np.tan(phi_soil_rad-slope_rad) + \
            coh_soil/(
                gamma_soil * t_slope * np.cos(slope_rad)**2 * \
                (1+np.tan(phi_soil_rad)*np.tan(slope_rad)))
        ky = np.maximum(ky,0.001) # to avoid ky = 0
        
        # probability of zero displacement, eq. 2 with Ts=0
        prob_d_eq_0 = 1 - norm.cdf(
            -2.480 + \
            -2.970*np.log(ky) + \
            -0.120*(np.log(ky))**2 + \
            2.780*np.log(pga)
        )
       
        # deformation, eq 3b
        ln_pgdef_trunc = \
            -4.684 + \
            -2.482*np.log(ky) + \
            -0.244*(np.log(ky))**2 + \
            0.344*np.log(ky)*np.log(pga) + \
            2.649*np.log(pga) + \
            -0.090*(np.log(pga))**2 + \
            0.603*mag # cm
        # sigma = 0.72
        
        # apply non-zero displacement correction/condition, eq 11
        prob_d_gt_0 = 1 - prob_d_eq_0
        nonzero_median_cdf = np.maximum(1 - .5/prob_d_gt_0,0) # limit to 0 and bove
        sigma = 0.72
        nonzero_ln_pgdef = ln_pgdef_trunc + sigma*norm.ppf(nonzero_median_cdf)
        pgdef = np.exp(nonzero_ln_pgdef)
        # sigma = np.ones(pga.shape)*sigma
        
        # prepare outputs
        output = {
            'pgdef': pgdef,
            # 'sigma': sigma,
        }
        # get intermediate values if requested
        if return_inter_params:
            output['ky'] = ky
            output['prob_d_eq_0'] = prob_d_eq_0
            output['ln_pgdef_trunc'] = ln_pgdef_trunc
            output['nonzero_median_cdf'] = nonzero_median_cdf
        
        # return
        return output


# -----------------------------------------------------------
class Jibson2007(Landslide):
    """
    Compute landslide displacement at a given location using the Jibson (2007) probabilistic models. The four regression models are:

    1. **pgdef** = f(**ky**, **pga**)
    2. **pgdef** = f(**ky**, **pga**, **M**)
    3. **pgdef** = f(**ky**, **pga**, **Ia**) - unavailable for this version of OpenSRA
    4. **pgdef** = f(**ky**, **Ia**) - unavailable for this version of OpenSRA
    
    The default relationship for **ky** uses **coh_soil**, **phi_soil**, **gamma_soil**, **t_slope**, **slope**

    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
        
    Geotechnical/geologic:
    slope: float, np.ndarray or list
        [deg] slope angle
    t_slope: float, np.ndarray or list
        [m] slope thickness (infinite-slope problem)
    gamma_soil: float, np.ndarray or list
        [kN/m^3] unit weight of soil
    phi_soil: float, np.ndarray or list
        [deg] friction angle of soil
    coh_soil: float, np.ndarray or list
        [kPa] cohesion of soil
        
    Fixed:
    mag: float, np.ndarray or list
        moment magnitude

    Returns
    -------
    pgdef : float, np.ndarray or list
        [cm] permanent ground deformation
    
    References
    ----------
#     .. [1] Jibson, R.W., 2007. Regression Models for Estimating Coseismic Landslide Displacement, Engineering Geology, vol. 91, no. 2, pp. 209–218.
    
    """

    _NAME = 'Jibson (2007)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Jibson, R.W., 2007. ',
        'Regression Models for Estimating Coseismic Landslide Displacement, ',
        'Engineering Geology, ',
        'vol. 91, no. 2, pp. 209–218.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (cm)',
                'unit': 'cm',
                'mean': None,
                'aleatory': None,
                'epistemic': {
                    'coeff': 0.25, # base uncertainty, based on coeffcients
                    'input': None, # sigma_mu uncertainty from input parameters
                    'total': None # SRSS of coeff and input sigma_mu uncertainty
                },
                'dist_type': 'lognormal',
            },
            'sigma': {
                'desc': 'aleatory variability for ln(pgdef)',
                'unit': '',
                'mean': None,
            },
        }
    }
    _INPUT_PBEE_META = {
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        'variable': [
            'pga'
        ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
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
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        "params": {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'slope': 'slope angle (deg)',
            't_slope': 'slope thickness (m)',
            'gamma_soil': 'unit weight of soil (kN/m^3)',
            'phi_soil': 'friction angle of soil (deg)',
            'coh_soil': 'cohesion of soil (kPa)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'mag': 'moment magnitude',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'beta', 't_beta', 'gamma_soil', 'phi_soil', 'coh_soil',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'mag',
    }
    _MODEL_INTERNAL = {
        'n_sample': 1,
        'n_site': 1,
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    @staticmethod
    # @njit
    def _model(
        pga, # upstream PBEE RV
        slope, t_slope, gamma_soil, phi_soil, coh_soil, # geotechnical/geologic
        mag, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # initialize arrays
        pgdef = np.empty(pga.shape)
        sigma = np.empty(pga.shape)
        
        # convert from deg to rad
        slope_rad = slope*np.pi/180
        phi_soil_rad = phi_soil*np.pi/180
        
        # factor of safety
        fs = coh_soil/(gamma_soil*t_slope*np.sin(slope_rad)) + \
            np.tan(phi_soil_rad)/np.tan(slope_rad)
        
        # yield acceleration
        ky = np.maximum((fs-1) * np.sin(slope_rad),0.01)

        # for magnitudes between 5.3 and 7.6
        ind_mag_btw_5p3_7p6 = np.where(np.logical_and(mag>=5.3, mag<=7.6))[0]
        if len(ind_mag_btw_5p3_7p6) > 0:
            pgdef[ind_mag_btw_5p3_7p6] = 10**(
                -2.710 + np.log10(
                    (1-ky[ind_mag_btw_5p3_7p6]/pga[ind_mag_btw_5p3_7p6])**2.335 * \
                    (ky[ind_mag_btw_5p3_7p6]/pga[ind_mag_btw_5p3_7p6])**-1.478
                ) + 0.424*mag[ind_mag_btw_5p3_7p6]
            )
            sigma[ind_mag_btw_5p3_7p6] = 0.454 * np.log(10)
        
        # for other magnitudes
        ind_mag_other = list(set(list(range(pga.shape[0]))).difference(set(ind_mag_btw_5p3_7p6)))
        pgdef[ind_mag_other] = 10**(
            0.215 + np.log10(
                (1-ky[ind_mag_other]/pga[ind_mag_other])**2.341 * \
                (ky[ind_mag_other]/pga[ind_mag_other])**-1.438
            )
        )
        sigma[ind_mag_other] = 0.510 * np.log(10)
        
        # prepare outputs
        output = {
            'pgdef': pgdef,
            'sigma': sigma,
        }
        # get intermediate values if requested
        if return_inter_params:
            output['fs'] = fs
            output['ky'] = ky
        
        # return
        return output



# -----------------------------------------------------------
# class BrayMacedo2019(Landslide):
#     """
#     Compute slope displacement at a given location using the Bray and Macedo (2007) probabilistic model. Based on the site period, **Ts**, the model decides whether to use the PGA (for Ts = 0 sec) or the Sa(Ts_deg) correlation. Regression models based on three sets of ground motions are available:

#     1. **Ordinary**: **d** = f(ky, Sa(T), Ts, M)
#     2. **Near-fault**: **d** = f(ky, Sa(T), Ts, M, pgv)
#     3. **General** (default): **d** = f(ky, Sa(T), Ts, M, pgv)

#     Note that:\n
#     - The site period can either be specified or calculated using site conditions; see :func:`_edp.fcn_liq_land.get_Ts`
#     - If **Ts** < 0.05 sec, PGA must be entered. For all other **Ts**, either **Sa_Ts_deg** or a list of of periods (**T_arr**) and spectral accelerations (**Sa_arr**) are required.

#     Parameters
#     ----------
#     gm_type : str
#         type of ground motions used to develop the regressional model: **ordinary**, **near-fault**, or **general**; default = **general**
#     ky : float
#         [g] yield acceleration
#     M : float
#         moment magnitude
#     Ts : float
#         [sec] site period (see notes above); default = 0.15 sec if site/geolgic data is not sufficient for calculation
#     pga : float
#         [g] peak ground acceleration
#     pgv : float
#         [cm/s] peak ground velocity, required for **near-fault** and **general** models

#     Parameters for Ts
#     H : float
#         [m] slope/structure height
#     vs : float
#         [m/sec] shear wave velocity
#     nDim : int
#         **1** (trapezoidal) or **2** (triangular) dimension for calculating **Ts**; default = 2

#     Parameters for Ts_deg
#     T_arr : float, array
#         [sec] array of periods for interpolation of Ts_deg
#     Sa_arr : float, array
#         [g] array of spectral accelerations for interpolation of Ts_deg

#     Returns
#     -------
#     d : float
#         [cm] slope displacement
        
#     References
#     ----------
#     .. [1] Bray, J.D., and Macedo, J., 2019, Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 145, pp. 12, 04019106.
#     .. [2] Bray, J.D., and Travasarou, T., 2007, Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 133, no. 4, pp. 381-392.
#     .. [3] Rathje, E.M., and Bray, J.D., 2001, One- and Two-Dimensional Seismic Analysis of Solid-Waste Landfills, Canadian Geotechnical Journal, vol. 38, no. 4, pp. 850–862.
#     .. [4] Ashford, S.A., and Sitar, N., 2002, Simplified Method for Evaluating Seismic Stability of Steep Slopes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 128, no. 2, pp. 119-128.
    
#     """

#     NAME = 'Bray and Macedo (2022)' # Name of the model
#     ABBREV = 'BM19'                 # Abbreviated name of the model
#     REF = "".join([                 # Reference for the model
#         'Bray, J.D., and Macedo, J., 2019, ',
#         'Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, ',
#         'Journal of Geotechnical and Geoenvironmental Engineering, ',
#         'vol. 145, pp. 12, 04019106.'
#     ])
#     LEVEL = 1                       # Level of analysis: 1 = statewide, 2 = regional, 3 = site-specific
#     DIST = {                        # Distribution information
#         'TYPE': 'lognormal',
#         'ALEATORY': 0.4,
#         'EPISTEMIC': (0.74**2 - 0.4**2)**0.5,
#         'TOTAL': 0.74
#     }
#     INPUT = {                       # Model inputs (required and optional)
#         'REQUIRED': [               # List of required inputs
#             'eps_p',
#             'd',
#             't',
#             'sigma_y',
#             'pressure_op'
#         ],
#         'OPTIONAL': {               # Optional inputs with default values
#             'gm_type': 'general',
#             'd0': 0.5
#         }
#     }
#     OUTPUT = [                      # List of available outputs
#         'pgd',
#         'prob'
#     ]


#     # instantiation
#     def __init__(self):
#         super().__init__()


#     # update calculation method
#     def _perform_calc(self):
#         """Performs calculations"""
#         # pull inputs locally
#         n_sample = self._inputs['n_sample']
#         n_site = self._inputs['n_site']

#         pga = self._convert_to_ndarray(self._inputs['pga'], n_sample)



#         eps_p = self._convert_to_ndarray(self._inputs['eps_p'], n_sample)
#         d = self._convert_to_ndarray(self._inputs['d'], n_sample)
#         t = self._convert_to_ndarray(self._inputs['t'], n_sample)
#         sigma_y = self._convert_to_ndarray(self._inputs['sigma_y'], n_sample)
#         pressure_op = self._convert_to_ndarray(self._inputs['pressure_op'], n_sample)

#         # calculations
#         prob = self._model(eps_p, d, t, sigma_y, pressure_op)
        
#         # store intermediate params
#         self._inters.update({
#             'prob': prob
#         })


#     @classmethod
#     def _model(cls, eps_p, d, t, sigma_y, pressure_op):
#         """Model"""
#         # model coefficients
#         c0 = np.zeros(pgv_k.shape)  # constant
#         c1 = np.zeros(pgv_k.shape)  # Ts
#         c2 = np.zeros(pgv_k.shape)  # Ts^2
#         c3 = np.zeros(pgv_k.shape)  # ln(pgv)
#         c4 = np.zeros(pgv_k.shape)  # 
#         c5 = np.zeros(pgv_k.shape)  # 
#         c6 = np.zeros(pgv_k.shape)  # 
#         c7 = np.zeros(pgv_k.shape)  # 
#         c8 = np.zeros(pgv_k.shape)  # 
#         c9 = np.zeros(pgv_k.shape)  # 

#         c0 =  1.709     # constant
#         c1 = -1.000     # ln(eps_p)
#         c2 = -1.617     # ln(d/t)

#         # calculations
#         sigma_h = pressure_op * d/t / 2 # kPa, pipe hoop stress
#         eps_p_eq = eps_p / (1 + sigma_h/sigma_y) # %, zero internal pressure equivalent compressive pipe strain
#         prob = norm.sf((c0 + c1*np.log(eps_p_eq/100) + c2*np.log(d/t)) / cls.DIST['ALEATORY']) # survival function

#         # return
#         return prob


#     @staticmethod
#     def _coeff_ordinary(Ts=0):
#         if 
#         c0 = # constant
#         # 



# -----------------------------------------------------------
def bm19(**kwargs):
    """
    Compute slope displacement at a given location using the Bray and Macedo (2007) probabilistic model. Based on the site period, **Ts**, the model decides whether to use the PGA (for Ts = 0 sec) or the Sa(Ts_deg) correlation. Regression models based on three sets of ground motions are available:

    1. **Ordinary**: **d** = f(ky, Sa(T), Ts, M)
    2. **Near-fault**: **d** = f(ky, Sa(T), Ts, M, pgv)
    3. **General** (default): **d** = f(ky, Sa(T), Ts, M, pgv)

    Note that:\n
    - The site period can either be specified or calculated using site conditions; see :func:`_edp.fcn_liq_land.get_Ts`
    - If **Ts** < 0.05 sec, PGA must be entered. For all other **Ts**, either **Sa_Ts_deg** or a list of of periods (**T_arr**) and spectral accelerations (**Sa_arr**) are required.

    Parameters
    ----------
    gm_type : str
        type of ground motions used to develop the regressional model: **ordinary**, **near-fault**, or **general**; default = **general**
    ky : float
        [g] yield acceleration
    M : float
        moment magnitude
    Ts : float
        [sec] site period (see notes above); default = 0.15 sec if insufficient site/geolgic data for calculation
    pga : float
        [g] peak ground acceleration
    pgv : float
        [cm/s] peak ground velocity, required for **near-fault** and **general** models

    Parameters for Ts
    H : float
        [m] slope/structure height
    vs : float
        [m/sec] shear wave velocity
    nDim : int
        **1** (trapezoidal) or **2** (triangular) dimension for calculating **Ts**; default = 2

    Parameters for Ts_deg
    T_arr : float, array
        [sec] array of periods for interpolation of Ts_deg
    Sa_arr : float, array
        [g] array of spectral accelerations for interpolation of Ts_deg

    Returns
    -------
    d : float
        [cm] slope displacement

    References
    ----------
    .. [1] Bray, J.D., and Macedo, J., 2019, Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 145, pp. 12, 04019106.
    .. [2] Bray, J.D., and Travasarou, T., 2007, Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 133, no. 4, pp. 381-392.
    .. [3] Rathje, E.M., and Bray, J.D., 2001, One- and Two-Dimensional Seismic Analysis of Solid-Waste Landfills, Canadian Geotechnical Journal, vol. 38, no. 4, pp. 850–862.
    .. [4] Ashford, S.A., and Sitar, N., 2002, Simplified Method for Evaluating Seismic Stability of Steep Slopes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 128, no. 2, pp. 119-128.

    """

    # Inputs
    # Demand
    pga = kwargs.get('pga',None) # g, peak ground acceleration
    pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
    n_site = kwargs.get('n_site',pga[0].shape[1]) # number of sites
    n_event = kwargs.get('n_event',pga[0].shape[0]) # number of ruptures
    return_param = kwargs.get('return_param',['pgd_land','p_land']) # default to liq susc
    # gm_type = kwargs.get('gm_type','gen') # takes ord = ordinary GMs, nf = near-fault GMs, gen/full = general or full (ordinary + near-fault) GMs
    gm_type = kwargs.get('GroundMotionType','gen') # takes ord = ordinary GMs, nf = near-fault GMs, gen/full = general or full (ordinary + near-fault) GMs
    ky = kwargs.get('ky',None) # yield accelerations
    # Ts = kwargs.get('Ts',0.15) # sec, site period = 4H/Vs, use 0.15 sec if site/geologic data is not available
    M = kwargs.get('M',None) # moment magnitude
    # beta = kwargs.get('beta',None) # deg, shallow = <30, moderately steep = 30 to 60, steep = >60, Ashford & Sitar (2002)
    d0 = kwargs.get('d0',0.5) # cm, displacement threshold below which is considered "zero"
    # flag_topo = kwargs.get('flag_topo','local') # localized versus long potential sliding mass
    slope_loc = kwargs.get('slope_loc',90) # deg, slope location with respect to fault-normal for near-fault GMs
                                        # if <= 45, output D100, else, output D50
                                        # default to 90 deg (fault parallel)
    n_sample = kwargs.get('n_sample',1) # number of samples, default to 1

    # if pga is given, approximate using Ts = 0.01 and Sa(Ts_deg) ~= pga
    if pga is not None:
        Ts = 0.01
        Sa_Ts_deg = pga

    # get pgv- and period-dependent coefficients and sigmas
    if 'ord' in gm_type.lower():
        # total sigma
        sigma_total = 0.72 # for ln(D)

        # initialize matrix for coeffcients
        a1 = np.zeros([n_event,n_site])
        a2 = np.zeros([n_event,n_site])
        a3 = np.zeros([n_event,n_site])

        a1[Ts<=0.01] = -4.684
        a1[np.logical_and(Ts>0.01,Ts<0.1)] = -4.684
        a1[Ts>=0.1] = -5.981

        a2[Ts<=0.01] = 0.0
        a2[np.logical_and(Ts>0.01,Ts<0.1)] = -9.471
        a2[Ts>=0.1] = 3.223

        a3[Ts<=0.01] = 0.0
        a3[np.logical_and(Ts>0.01,Ts<0.1)] = 0.0
        a3[Ts>=0.1] = -0.945

    elif 'nf' in gm_type.lower() or 'near' in gm_type.lower() or 'fault' in gm_type.lower():
        # check if slope is oriented within 45 degree of fault-normal direction
        # if slope_loc <= 45:
            # total sigma
            # sigma_total = 0.56 # for ln(D)
        # elif slope_loc > 45:
            # total sigma
            # sigma_total = 0.54 # for ln(D)
            
        # use site-independent sigma of 0.55 for now
            sigma_total = 0.55

    elif 'full' in gm_type.lower() or 'gen' in gm_type.lower():
        # total sigma
        sigma_total = 0.74 # for ln(D)

    # distribution based on case:
    prob_dist_type = 'Lognormal'
    # aleatory
    sigma_aleatory = 0.4 # for ln(D)
    # estimate epistemic from total and aleatory
    sigma_epistemic = (sigma_total**2-sigma_aleatory**2)**0.5 # for ln(D)

    # initialize dictionaries for outputs
    d = {}
    p_land = {}

    # loop through samples
    for k in range(n_sample):

        # get current sampled inputs
        Sa_Ts_deg_k = Sa_Ts_deg[k].data
        pgv_k = pgv[k].data
        row_k = pgv[k].row
        col_k = pgv[k].col

        # get other params for current sample
        ky_k = ky[col_k]
        M_k = M[row_k]
        Ts_k = np.ones(pgv_k.shape)*Ts

        # see which case is picked:
        if 'ord' in gm_type.lower():

            # seismic displacement, cm (eq. 3)
            ln_d_k = a1 - 2.482*np.log(ky_k) - 0.244*np.multiply(np.log(ky_k),np.log(ky_k)) +\
                        0.344*np.multiply(np.log(ky_k),np.log(Sa_Ts_deg_k)) + 2.649*np.log(Sa_Ts_deg_k) -\
                        0.090*np.multiply(np.log(Sa_Ts_deg_k),np.log(Sa_Ts_deg_k)) + a2*Ts_k +\
                        a3*np.multiply(Ts_k,Ts_k) + 0.603*M_k

            # probably of displacement = 0 (threshold = 0.5 cm)
            if 'p_land' in return_param:
                p_land_k = np.zeros(pgv_k.shape)
                p_land_k[Ts_k<=0.7] = norm.cdf(-2.48 - 2.97*np.log(ky_k[Ts_k<=0.7]) -\
                                        0.12*np.multiply(np.log(ky_k[Ts_k<=0.7]),np.log(ky_k[Ts_k<=0.7])) -\
                                        0.72*np.multiply(Ts_k[Ts_k<=0.7],np.log(ky_k[Ts_k<=0.7])) +\
                                        1.70*Ts_k[Ts_k<=0.7] + 2.78*np.log(Sa_Ts_deg_k[Ts_k<=0.7]))# eq. 2(a)
                p_land_k[Ts_k>0.7] = norm.cdf(-3.42 - 4.93*np.log(ky_k[Ts_k>0.7]) -\
                                        0.30*np.multiply(np.log(ky_k[Ts_k>0.7]),np.log(ky_k[Ts_k>0.7])) -\
                                        0.35*np.multiply(Ts_k[Ts_k>0.7],np.log(ky_k[Ts_k>0.7])) -\
                                        0.62*Ts_k[Ts_k>0.7] + 2.86*np.log(Sa_Ts_deg_k[Ts_k>0.7])) # eq. 2(b)
                # p_land.update({k:sparse.coo_matrix(1-p_D_eq_0_k)})


        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # need to update definition of coefficients for near-fault case
        elif 'nf' in gm_type.lower() or 'near' in gm_type.lower() or 'fault' in gm_type.lower():

            # check if slope is oriented within 45 degree of fault-normal direction
            if slope_loc <= 45:

                # pgv- and period-dependent coefficients and parameters for D100
                if pgv <= 150:
                    if Ts < 0.1:
                        c1 = -6.724
                        c2 = -2.744
                        c3 = 0.0
                        c4 = 1.547
                    elif Ts >= 0.1:
                        c1 = -6.951
                        c2 = 1.069
                        c3 = -0.498
                        c4 = 1.547
                if pgv > 150:
                    if Ts < 0.1:
                        c1 = 1.764
                        c2 = -2.744
                        c3 = 0.0
                        c4 = -0.097
                    elif Ts >= 0.1:
                        c1 = 1.991
                        c2 = 1.069
                        c3 = -0.498
                        c4 = -0.097

                # maximum seismic displacement, D100 (eq. 5)
                ln_d = c1 - 2.632*np.log(ky) - 0.278*(np.log(ky))**2 + 0.527*np.log(ky)*np.log(Sa) +\
                    1.978*np.log(Sa) - 0.233*(np.log(Sa))**2 + c2*Ts + c3*Ts**2 + 0.01*M +\
                    c4*np.log(pgv)

                # probability of maximum seismic displacement = 0
                if Ts <= 0.7:
                    # eq. 4(a)
                    p_D_eq_0 = 1/(1 + np.exp(-10.787 - 8.717*np.log(ky) + 1.660*np.log(pgv) + 3.150*Ts +
                                            7.560*np.log(Sa)))
                elif Ts > 0.7:
                    # eq. 4(b)
                    p_D_eq_0 = 1/(1 + np.exp(-12.771 - 9.979*np.log(ky) + 2.286*np.log(pgv) - 4.965*Ts +
                                            4.817*np.log(Sa)))

            elif slope_loc > 45:

                # pgv- and period-dependent coefficients and parameters for D50
                if pgv <= 150:
                    if Ts < 0.1:
                        c1 = -7.497
                        c2 = -2.731
                        c3 = 0.0
                        c4 = 1.458
                    elif Ts >= 0.1:
                        c1 = -7.718
                        c2 = 1.031
                        c3 = -0.480
                        c4 = 1.458
                if pgv > 150:
                    if Ts < 0.1:
                        c1 = -0.148
                        c2 = -2.731
                        c3 = 0.0
                        c4 = 0.025
                    elif Ts >= 0.1:
                        c1 = -0.369
                        c2 = 1.031
                        c3 = -0.480
                        c4 = 0.025

                # median seismic displacement, D50 (eq. 7)
                ln_d = c1 - 2.931*np.log(ky) - 0.319*(np.log(ky))**2 + 0.584*np.log(ky)*np.log(Sa) +\
                    2.261*np.log(Sa) - 0.241*(np.log(Sa))**2 + c2*Ts + c3*Ts**2 + 0.05*M +\
                    c4*np.log(pgv)

                # probability of maximum seismic displacement = 0
                if Ts <= 0.7:
                    # eq. 5(a)
                    p_D_eq_0 = 1/(1 + np.exp(-14.930 - 10.383*np.log(ky) + 1.971*np.log(pgv) + 3.763*Ts +
                                            8.812*np.log(Sa)))
                elif Ts > 0.7:
                    # eq. 6(b)
                    p_D_eq_0 = 1/(1 + np.exp(-14.671 - 10.489*np.log(ky) + 2.222*np.log(pgv) - 4.759*Ts +
                                            5.549*np.log(Sa)))
        # -----------------------------------------------------------
        # -----------------------------------------------------------


        elif 'full' in gm_type.lower() or 'gen' in gm_type.lower():

            # initialize matrix for coefficients
            a1 = np.zeros(pgv_k.shape)
            a2 = np.zeros(pgv_k.shape)
            a3 = np.zeros(pgv_k.shape)
            a4 = np.zeros(pgv_k.shape)
            a5 = np.zeros(pgv_k.shape)

            # set values for coefficients
            a1[Ts_k<0.1] = -4.551
            a1[Ts_k>=0.1] = -5.894

            a2[Ts_k<0.1] = -9.690
            a2[Ts_k>=0.1] = 3.152

            a3[Ts_k>=0.1] = -0.910

            a4[pgv_k>115] = 1.0

            a5[pgv_k>115] = -4.75

            # seismic displacement, cm (eq. 9)
            ln_d_k = a1 - 2.491*np.log(ky_k) - 0.245*np.multiply(np.log(ky_k),np.log(ky_k)) +\
                    0.344*np.multiply(np.log(ky_k),np.log(Sa_Ts_deg_k)) + 2.703*np.log(Sa_Ts_deg_k) -\
                    0.089*np.multiply(np.log(Sa_Ts_deg_k),np.log(Sa_Ts_deg_k)) + a2*Ts_k +\
                    a3*np.multiply(Ts_k,Ts_k) + 0.607*M_k + a4*np.log(pgv_k) + a5

            # probably of displacement = 0 (threshold = 0.5 cm)
            if 'p_land' in return_param:
                p_land_k = np.zeros(pgv_k.shape)
                p_land_k[Ts_k<=0.7] = norm.cdf(-2.46 - 2.98*np.log(ky_k[Ts_k<=0.7]) -\
                                        0.12*np.multiply(np.log(ky_k[Ts_k<=0.7]),np.log(ky_k[Ts_k<=0.7])) -\
                                        0.71*np.multiply(Ts_k[Ts_k<=0.7],np.log(ky_k[Ts_k<=0.7])) +\
                                        1.69*Ts_k[Ts_k<=0.7] + 2.76*np.log(Sa_Ts_deg_k[Ts_k<=0.7])) *100 # %, eq. 8(a)
                p_land_k[Ts_k>0.7] = norm.cdf(-3.40 - 4.95*np.log(ky_k[Ts_k>0.7]) -\
                                        0.30*np.multiply(np.log(ky_k[Ts_k>0.7]),np.log(ky_k[Ts_k>0.7])) -\
                                        0.33*np.multiply(Ts_k[Ts_k>0.7],np.log(ky_k[Ts_k>0.7])) -\
                                        0.62*Ts_k[Ts_k>0.7] + 2.85*np.log(Sa_Ts_deg_k[Ts_k>0.7])) *100 # %, eq. 8(b)

        # exp() and update dictionary
        d_k = np.exp(ln_d_k)
        d_k[np.isnan(d_k)] = 0
        d_k[d_k<=d0] = 0
        ind_red = np.where(d_k>0)[0]
        d_k_red = d_k[ind_red]
        row_k_red = row_k[ind_red]
        col_k_red = col_k[ind_red]
        d.update({k:sparse.coo_matrix((d_k_red,(row_k_red,col_k_red)),shape=(n_event,n_site))})

        # update dictionary if p_land is inquired
        if 'p_land' in return_param:
            p_land_k_red = p_land_k[ind_red]
            p_land.update({k:sparse.coo_matrix((p_land_k_red,(row_k_red,col_k_red)),shape=(n_event,n_site))})

    # store outputs
    output = {}
    #
    if 'pgd_land' in return_param:
        output.update({'pgd_land': d.copy()})
    if 'p_land' in return_param:
        output.update({'p_land': p_land.copy()})
    # store probability distribution parameters
    output.update({'prob_dist': {
        'type': prob_dist_type,
        'sigma_total': sigma_total,
        'sigma_aleatory': sigma_aleatory,
        'sigma_epistemic': sigma_epistemic}})

    #
    return output



# # -----------------------------------------------------------
# def GrantEtal2016(slope_type, pga, **kwargs):
#     """
#     Compute rock and soil landslide displacement using following the Grant et al. (2016) deterministic procedure. Of the 40 coseismic landslide datasets described in Keefer (1984), Grant et al. (2016) selected the four most fundamental modes of failures (see Table 1 in Grant et al., 2016, for detailed descriptions):

#     1. **Rock**-slope failures: wedge geometry, slope range = 35-90 degrees, typically tens of meters to kilometers
#     2. Disrupted soil slides: **infinite slope**, slope range = 15-50 degrees, typically meters to a hundred meters
#     3. Coherent **rotational** slides: circular-rotation, slope range = 20-35 degrees, typically under 2 meters
#     4. Lateral spreads: empirirically-developed geometry, slope range = 0-6 degrees, typically under 2 meters

#     Note that:\n
#     - The regression model for lateral spreads is coded separately; see :func:`edp.ls.grant_etal_2016_ls`.\n
#     - The yield acceleration, **ky**, can either be specfied or or calculated using material and site properties; see :func:`edp.fcn_liq_land.get_ky` for methods used to estimate yield accelerations for landslide.

#     Parameters
#     ----------
#     slope_type : str
#         type of slope to assess; enter any of the bolded words shown in the above list of failure modes (i.e., **rock**, **infinite slope**, or **rotational**)
#     pga : float
#         [g] peak ground acceleration
#     ky : float, optional
#         [g] yield acceleration (see notes above)

#     General parameters required ky
#     phi : float
#         [degree] friction angle
#     c : float
#         [kPa] cohesion
#     beta : float
#         [degree] slope angle
#     gamma : float
#         [kN/m^3] unit weight

#     Additional parameters specific to rock-slope failures:
#     H : float
#         [m] height of slope (local relief)

#     Additional parameters specific to disrupted soil slides:
#     cr : float
#         [kPa] root cohesion
#     t : float
#         [m] thickness of failure mass

#     Parameters for Coherent Rotational Slides (failure_mode == 3):
#     H : float
#         [m] height of slope (local relief)
#     y : float
#         [m] width of cylindrical pixel

#     Returns
#     -------
#     d : float
#         [cm] permanent ground deformation (see *return_param* under "Parameters")

#     References
#     ----------
#     .. [1] Grant, A., Wartman, J., and Abou-Jaoude, G., 2016, Multimodal Method for Coseismic Landslide Hazard Assessment, Engineering Geology, vol. 212, pp. 146-160.
#     .. [2] Keefer, D.K., 1984., Landslides Caused by Earthquakes, Geological Society of America Bulletin, vol. 95, no. 4, pp. 406-421.
#     .. [3] Wills, C.J., Perez, F.G., and Gutierrez, C.I., 2011, Susceptibility to Deep-Seated Landslides in California. California Geological Survey Map Sheet no. 58.

#     """

#     # determine yield acceleration based on input params and mode of failure
#     ky = fcn_liq_land.get_ky(slope_type,method='grant',kwargs=kwargs) # g

#     # get pga for calculation
#     pga = kwargs.get('pga',None) # g

#     # 50% reduction applied to pga for rotational failures
#     if 'rotational' in slope_type.lower:
#         pga = 0.5*pga

#     # calculation
#     if pga > ky:
#         d = np.exp(0.215 + np.log((1-ky/pga)**2.341 * (ky/pga)**-1.438)) # cm, coseismic displacement
#     else:
#         d = 0 # pga under ky, no movement

#     #
#     return d


# # -----------------------------------------------------------
# def Jibson2007(**kwargs):
#     """
#     Compute landslide displacement at a given location using the Jibson (2007) probabilistic models. Four regression models are provided:

#     1. **d** = f(**ky**, **pga**)
#     2. **d** = f(**ky**, **pga**, **M**)
#     3. **d** = f(**ky**, **pga**, **Ia**)
#     4. **d** = f(**ky**, **Ia**)

#     Note that:\n
#     - If all parameters are given, the model with the lowest **sigma** is returned\n
#     - The yield acceleration, **ky**, can either be specfied or calculated using the methods from Newmark (1965). In the simplest form, **ky** can be calculate as follows, with **FS** as the factor of safety against landslide and **alpha** as the slope angle (see :func:`edp.fcn_liq_land.get_ky` for additional methods used to estimate yield accelerations for landslide):\n

#     **ky** = (**FS** - 1) * sin(**alpha**)

#     Parameters
#     ----------
#     pga : float
#         [g] peak ground acceleration, primary model parameter
#     ky : float, optional
#         [g] yield acceleration, additional model parameter (see notes above)
#     M : float, optional
#         moment magnitude, additional model parameter
#     Ia : float, optional
#         [g] peak ground acceleration, additional model parameter

#     Additional parameters for ky
#     alpha : float
#         [degree] slope angle, required if **ky** needs to be calculated (see description above)
#     FS : float
#         factor of safety against slope stability, required if **ky** needs to be calculated (see description above)

#     Returns
#     -------
#     d : float
#         [cm] landslide displacement for the regression model with the lowest sigma

#     References
#     ----------
#     .. [1] Jibson, R.W., 2007. Regression Models for Estimating Coseismic Landslide Displacement, Engineering Geology, vol. 91, no. 2, pp. 209–218.
#     .. [2] Newmark, N.M., 1965, Effects of Earthquakes on Dams and Embankments, Geotechnique, vol. 15, pp. 139–159.

#     """

#     # Get inputs
#     pga = kwargs.get('pga',None) # g, peak ground acceleration = amax in paper
#     ky = kwargs.get('ky',None) # g, yield acceleration = ac in paper
#     M = kwargs.get('M',None) # moment magnitude
#     Ia = kwargs.get('Ia',None) # m/s, arias intensity
#     # slope_type = kwargs.get('slope_type',None)
#     n_sample = kwargs.get('n_sample',1) # number of samples, default to 1
#     n_site = kwargs.get('n_site',pga[0].shape[1]) # number of sites
#     n_event = kwargs.get('n_event',pga[0].shape[0]) # number of ruptures
#     return_param = kwargs.get('return_param',['pgd_land']) # default to liq susc
#     model_type = kwargs.get('ModelType',None) # number of ruptures

#     # initialize diciontary for displacement
#     d = {}

#     # if ky is None and (pga is None or Ia is None):
#     if pga is None and Ia is None:
#         # logging.info('Requires at the minimum ky and either pga or Ia; cannot proceed with procedure')
#         logging.info('Requires either pga or Ia; cannot proceed with procedure')
#         return None

#     else:
#         # see if model type is not specified
#         if model_type is None:
#             # if model is not specified, check what input parameters are available
#             if pga is not None:
#                 # Model as a function of ky and pga
#                 if M is None:
#                     case = 1
#                     model_type = 'ky_pga'
#                     sigma_total = 0.510*np.log(10) # eq. 6 in Jibson (2007)
                    
#                 # Model as a function of ky, pga, and M
#                 else:
#                     case = 2
#                     model_type = 'ky_pga_M'
#                     sigma_total = 0.454*np.log(10) # eq. 7 in Jibson (2007)

#             if Ia is not None:
#                 # Model as a function of ky and Ia
#                 if pga is None:
#                     case = 3
#                     model_type = 'ky_Ia'
#                     sigma_total = 0.656*np.log(10) # eq. 9 in Jibson (2007)

#                 # Model as a function of ky, Ia, and pga
#                 else:
#                     case = 4
#                     model_type = 'ky_pga_Ia'
#                     sigma_total = 0.616*np.log(10) # eq. 10 in Jibson (2007)
#         # see if model type is specified
#         else:
#             if model_type == 'ky_pga':
#                 case = 1
#                 sigma_total = 0.510*np.log(10) # eq. 6 in Jibson (2007)
#             elif model_type == 'ky_pga_M':
#                 case = 2
#                 sigma_total = 0.454*np.log(10) # eq. 7 in Jibson (2007)
#             elif model_type == 'ky_Ia':
#                 case = 3
#                 sigma_total = 0.656*np.log(10) # eq. 9 in Jibson (2007)
#             elif model_type == 'ky_pga_Ia':
#                 case = 4
#                 sigma_total = 0.616*np.log(10) # eq. 10 in Jibson (2007)
#             else:            
#                 # default case
#                 case = 2
#                 model_type = 'ky_pga_M'
#                 sigma_total = 0.454*np.log(10) # eq. 7 in Jibson (2007)
        
        
#         # -----------------------------------------------------------
#         # -----------------------------------------------------------
#         # distribution
#         # borrewed from BrayMacedo2019 for now
#         prob_dist_type = 'Lognormal'
#         sigma_aleatory = 0.4 # for ln(D)
#         sigma_epistemic = (sigma_total**2-sigma_aleatory**2)**0.5 # for ln(D)
#         # -----------------------------------------------------------
#         # -----------------------------------------------------------
        

#         # loop through samples
#         for k in range(n_sample):
        
#             # Model as a function of ky and pga
#             if case == 1:
#                 # get current sampled inputs
#                 pga_k = pga[k].data
#                 row_k = pga[k].row
#                 col_k = pga[k].col
#                 # get other params for current sample
#                 ky_k = ky[col_k]
#                 # initialize zero array
#                 d_k = np.zeros(len(pga_k))
#                 # displacement, cm
#                 cond = np.where(pga_k>ky_k)[0] # indices where condition is true
#                 d_k[cond] = 10**(0.215 + np.log10((1 - ky_k[cond]/pga_k[cond])**2.341 * \
#                     (ky_k[cond]/pga_k[cond])**-1.438)) # eq. 6 in Jibson (2007)
                
#             # Model as a function of ky, pga, and M
#             elif case == 2:
#                 # get current sampled inputs
#                 pga_k = pga[k].data
#                 row_k = pga[k].row
#                 col_k = pga[k].col
#                 # get other params for current sample
#                 ky_k = ky[col_k]
#                 M_k = M[row_k]
#                 M_k = np.maximum(np.minimum(M_k,7.6),5.3) # limit assessment to 5.3 <= Mw <= 7.6
#                 # initialize zero array
#                 d_k = np.zeros(len(pga_k))
#                 # displacement, cm
#                 cond = np.where(pga_k>ky_k)[0] # indices where condition is true
#                 d_k[cond] = 10**(-2.710 + np.log10((1 - ky_k[cond]/pga_k[cond])**2.335 * \
#                     (ky_k[cond]/pga_k[cond])**-1.478) + 0.424*M_k[cond]) # eq. 7 in Jibson (2007)
            
#             # Model as a function of ky and Ia
#             elif case == 3:
#                 # get current sampled inputs
#                 Ia_k = Ia[k].data
#                 row_k = Ia[k].row
#                 col_k = Ia[k].col
#                 # get other params for current sample
#                 ky_k = ky[col_k]
#                 # displacement, cm
#                 d_k = 10**(2.401*np.log10(Ia_k) - 3.481*np.log10(ky_k) - 3.230) # eq. 9 in Jibson (2007)
                
#             # Model as a function of ky, Ia, and pga
#             elif case == 4:
#                 # get current sampled inputs
#                 pga_k = pga[k].data
#                 row_k = pga[k].row
#                 col_k = pga[k].col
#                 Ia_k = Ia[k].data
#                 # get other params for current sample
#                 ky_k = ky[col_k]
#                 # initialize zero array
#                 d_k = np.zeros(len(pga_k))
#                 # displacement, cm
#                 cond = np.where(pga_k>ky_k)[0] # indices where condition is true
#                 d_k[cond] = 10**(0.561*np.log10(Ia[cond]) - \
#                     3.833*np.log10(ky[cond]/pga[cond]) - 1.474) # eq. 10 in Jibson (2007)

#             # update dictionary
#             d.update({k:sparse.coo_matrix((d_k,(row_k,col_k)),shape=(n_event,n_site))})

#         # store outputs
#         output = {}
#         #
#         if 'pgd_land' in return_param:
#             output.update({'pgd_land': d.copy()})
#         # store probability distribution parameters
#         output.update({'prob_dist': {
#             'type': prob_dist_type,
#             'sigma_total': sigma_total,
#             'sigma_aleatory': sigma_aleatory,
#             'sigma_epistemic': sigma_epistemic}})

#         #
#         return output


# # -----------------------------------------------------------
# def Saygili2008(pga, **kwargs):
#     """
#     Compute landslide displacement at a given location using the Saygili (2008) probabilistic models. Seven regression
#     models are provided:

#     1. **d** = f(**ky**, **pga**)
#     2. **d** = f(**ky**, **pga**, **M**)
#     3. **d** = f(**ky**, **pga**, **pgv**)
#     4. **d** = f(**ky**, **pga**, **pgv**, **Ia**)
#     5. **d** = f(**ky**, **pga**, **Ia**)
#     6. **d** = f(**ky**, **pga**, **Tm**)
#     7. **d** = f(**ky**, **pga**, **Tm**, **Ia**)

#     Note that:\n
#     - If all parameters are given, the model with the lowest **sigma** is returned\n
#     - The yield acceleration, **ky**, can either be specfied or calculated using material and site propertie; see the method for **infinite slope** in :func:`edp.fcn_liq_land.get_ky` 

#     Parameters
#     ----------
#     pga : float
#         [g] peak ground acceleration, primary model parameter
#     ky : float, optional
#         [g] yield acceleration (see notes above)
#     M : float, optional
#         moment magnitude, additional model parameter
#     Ia : float, optional
#         [g] peak ground acceleration, additional model parameter

#     Additional parameters for ky
#     phi : float
#         [degree] friction angle
#     c : float
#         [kPa] cohesion
#     beta : float
#         [degree] slope angle
#     gamma : float
#         [kN/m^3] total unit weight of material
#     t : float
#         [m] thickness of failure surface
#     m : float
#         [%] percent of failure thickness that is saturated
#     gamma_w : float
#         [kN/m^3] unit weight of water, default = 9.81 kN/m^3

#     Returns
#     -------
#     d : float
#         [cm] landslide displacement calculated using the regression model that maximizes the mumber of supplied parameters (pga, ky, M, Ia, Tm)

#     References
#     ----------
#     .. [1] Newmark, N.M., 1965, Effects of Earthquakes on Dams and Embankments, Geotechnique, vol. 15, pp. 139–159.
#     .. [2] Rathje, E.M., and Saygili, G., 2009, Probabilistic Assessment of Earthquake-Induced Sliding Displacements of Natural Slopes, Bulletin of the New Zealand Society for Earthquake Engineering, vol. 42, no. 1, pp. 18-27.
#     .. [3] Saygili, G., 2008, A Probabilistic Approach for Evaluating Earthquake-Induced Landslides, PhD Thesis, Universtiy of Texas at Austin.
#     .. [4] Saygili, G., and Rathje, E.M., 2008, Empirical Predictive Models for Earthquake-Induced Sliding Displacements of Slopes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 134, no. 6, pp. 790-803.

#     """

#     # Get inputs
#     pga = kwargs.get('pga',None) # g, peak ground acceleration
#     pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
#     M = kwargs.get('M',None) # moment magnitude
#     Ia = kwargs.get('Ia',None) # m/s, arias intensity
#     Tm = kwargs.get('Tm',None) # sec, mean period
#     ky = kwargs.get('ky',None) # g, yield acceleration, either provided or computed below

#     # Check if ky is provided, if not then compute it
#     if ky is None:
#         try:
#             # determine yield acceleration based on input params and mode of failure
#             ky = fcn_liq_land.get_ky(slope_type='infinite slope',method='rathje',kwargs=kwargs) # g
#         except:
#             logging.info('Not enough inputs to calculate ky - see Rathje & Saygili (2011) for all required inputs')
#             ky = None

#     # initialize diciontary for displacement
#     d_dict = {}

#     if pga is None or ky is None:
#         logging.info('Requires at the minimum PGA and ky as inputs; cannot proceed with procedure')
#         d = np.nan
#         sigma = np.nan

#     else:

#         # scalar model: f(pga)
#         # Table 4.2 in Saygili (2008) dissertation
#         a1 = 5.52
#         a2 = -4.43
#         a3 = -20.39
#         a4 = 42.61
#         a5 = -28.74
#         a6 = 0.72
#         a7 = 0

#         # displacement, cm
#         if pga > ky:
#             d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
#                         a6*np.log(pga)) # eq. 4.5 in Saygili (2008) dissertation
#         else:
#             d = 0

#         # sigma for ln(D)
#         sigma = 1.13 # Table 4.2 in Saygili (2008) dissertation

#         d_dict.update({'ky_pga': [d,sigma]})

#         # modified scalar model: f(pga,M)
#         if M is not None:
#             # Table 4.2 in Saygili (2008) dissertation
#             a1 = 4.89
#             a2 = -4.85
#             a3 = -19.64
#             a4 = 42.49
#             a5 = -29.06
#             a6 = 0.72
#             a7 = 0.89

#             # displacement, cm
#             if pga > ky:
#                 d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
#                     a6*np.log(pga) + a7*(M-6)) # eq. 4.6 in Saygili (2008) dissertation
#             else:
#                 d = 0

#             # sigma for ln(D)
#             sigma = 0.73 + 0.789*(ky/pga) - 0.539*(ky/pga)**2 # eq. 4.7 in Saygili (2008) dissertation
#                                                             # and eq. 9 in Rathje and Saygili (2009)

#             d_dict.update({'ky_pga_M': [d,sigma]})

#         # two-parameter vector model: f(pga,pgv)
#         if pgv is not None:
#             # Table 4.3 in Saygili (2008) dissertation
#             a1 = -1.56
#             a2 = -4.58
#             a3 = -20.84
#             a4 = 44.75
#             a5 = -30.50
#             a6 = -0.64
#             a7 = 1.55

#             # displacement, cm
#             if pga > ky:
#                 d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 +\
#                     a6*np.log(pga) + a7*np.log(pgv)) # eq. 4.8 in Saygili (2008) dissertation
#             else:
#                 d = 0

#             # sigma for ln(D)
#             sigma = 0.405 + 0.524*(ky/pga) # Table 4.4 in Saygili (2008) dissertation
#                                         # and eq. 11 in Rathje and Saygili (2009)

#             d_dict.update({'ky_pga_pgv': [d,sigma]})

#             # three-parameter vector model: f(pga,pgv,Ia)
#             if Ia is not None:
#                 # Table 4.3 in Saygili (2008) dissertation
#                 a1 = -0.74
#                 a2 = -4.93
#                 a3 = -19.91
#                 a4 = 43.75
#                 a5 = -30.12
#                 a6 = -1.30
#                 a7 = 1.04
#                 a8 = 0.67

#                 # displacement, cm
#                 if pga > ky:
#                     d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
#                             a6*np.log(pga) + a7*np.log(pgv) + a8*np.log(Ia)) # eq. 4.8 in Saygili (2008) dissertation
#                 else:
#                     d = 0

#                 # sigma for ln(D)
#                 sigma = 0.20 + 0.79*(ky/pga) # Table 4.4 in Saygili (2008) dissertation

#                 d_dict.update({'ky_pga_pgv_Ia': [d,sigma]})

#         # two-parameter vector model: f(pga,Tm)
#         if Tm is not None:
#             # Table 4.3 in Saygili (2008) dissertation
#             a1 = 6.62
#             a2 = -3.93
#             a3 = -23.71
#             a4 = 49.37
#             a5 = -32.94
#             a6 = -0.93
#             a7 = 1.79

#             # displacement, cm
#             if pga > ky:
#                 d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 +\
#                         a6*np.log(pga) + a7*np.log(Tm)) # eq. 4.8 in Saygili (2008) dissertation
#             else:
#                 d = 0

#             # sigma for ln(D)
#             sigma = 0.60 + 0.26*(ky/pga) # Table 4.4 in Saygili (2008) dissertation

#             d_dict.update({'ky_pga_Tm': [d,sigma]})

#             # three-parameter vector model: f(pga,Tm,Ia)
#             if Ia is not None:
#                 # Table 4.3 in Saygili (2008) dissertation
#                 a1 = 4.27
#                 a2 = -4.62
#                 a3 = -21.49
#                 a4 = 46.53
#                 a5 = -31.66
#                 a6 = -0.57
#                 a7 = 1.14
#                 a8 = 0.86

#                 # displacement, cm
#                 if pga > ky:
#                     d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 + \
#                             a6*np.log(pga) + a7*np.log(Tm) + a8*np.log(Ia)) # eq. 4.8 in Saygili (2008) dissertation
#                 else:
#                     d = 0

#                 # sigma for ln(D)
#                 sigma = 0.19 + 0.75*(ky/pga) # Table 4.4 in Saygili (2008) dissertation

#                 d_dict.update({'ky_pga_Tm_Ia': [d,sigma]})

#         # two-parameter vector model: f(pga,Ia)
#         if Ia is not None:
#             # Table 4.3 in Saygili (2008) dissertation
#             a1 = 2.39
#             a2 = -5.24
#             a3 = -18.78
#             a4 = 42.01
#             a5 = -29.15
#             a6 = -1.56
#             a7 = 1.38

#             # displacement, cm
#             if pga > ky:
#                 d = np.exp(a1 + a2*(ky/pga) + a3*(ky/pga)**2 + a4*(ky/pga)**3 + a5*(ky/pga)**4 +\
#                         a6*np.log(pga) + a7*np.log(Ia)) # eq. 4.8 in Saygili (2008) dissertation
#             else:
#                 d = 0

#             # sigma for ln(D)
#             sigma = 0.46 + 0.56*(ky/pga) # Table 4.4 in Saygili (2008) dissertation

#             d_dict.update({'ky_pga_Ia': [d,sigma]})

#     # loop to find the model with the lowest sigma
#     sigma = 999 # set initial check to very high number
#     keys = d_dict.keys() # get all the keys in dictionary
#     if len(d_dict.keys()) > 0 and d is not np.nan: # see if there are any entries in d_dict and if d is NaN
#         for i in keys:
#             if d_dict[i][1] < sigma:
#                 d = d_dict[i][0]
#                 sigma = d_dict[i][1]

#     #
#     return d


# # -----------------------------------------------------------
# def RathjeAntonakos2011(pga, **kwargs):
#     """
#     Compute landslide displacement at a given location using the Saygili (2008) probabilistic models. Two regression models are provided, with and without correction for flexibility (**Ts**/**Tm**):

#     1. **d** = f(**ky**, **pga**, **M**, **Tm**, **Ts**)
#     2. **d** = f(**ky**, **pga**, **pgv**, **Tm**, **Ts**)

#     Note that:\n
#     - If all parameters are given, the model with the lowest **sigma** is returned\n
#     - The yield acceleration, **ky**, can either be specfied or calculated using material and site properties; see the method for **infinite slope** in :func:`_edp.fcn_liq_land.get_ky`

#     Parameters
#     ----------
#     pga : float
#         [g] peak ground acceleration, primary model parameter
#     Tm : float, optional
#         [sec] mean period
#     Ts : float, optional
#         [sec] peak ground acceleration, additional model parameter
#     ky : float, optional
#         [g] yield acceleration (see notes above)
#     M : float, optional
#         moment magnitude, additional model parameter
#     pgv : float, optional
#         moment magnitude, additional model parameter

#     Additional parameters for ky
#     phi : float
#         [degree] friction angle
#     c : float
#         [kPa] cohesion
#     beta : float
#         [degree] slope angle
#     gamma : float
#         [kN/m^3] total unit weight of material
#     t : float
#         [m] thickness of failure surface
#     m : float
#         [%] percent of failure thickness that is saturated
#     gamma_w : float
#         [kN/m^3] unit weight of water, default = 9.81 kN/m^3

#     Returns
#     -------
#     d : float
#         [cm] landslide displacement calculated using the regression model that maximizes the mumber of supplied parameters (pga, ky, M, Tm, Ts)

#     References
#     ----------
#     .. [1] Rathje, E.M., and Antonakos, G., 2011, A Unified Model for Predicting Earthquake-Induced Sliding Displacements of Rigid and Flexible Slopes. Engineering Geology, vol. 122, no. 1-2, pp. 51-60.
#     .. [2] Rathje, E.M., and Saygili, G., 2009, Probabilistic Assessment of Earthquake-Induced Sliding Displacements of Natural Slopes, Bulletin of the New Zealand Society for Earthquake Engineering, vol. 42, no. 1, pp. 18-27.

#     """

#     # Inputs
#     pga = kwargs.get('pga',None) # g, peak ground acceleration
#     pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
#     M = kwargs.get('M',None) # moment magnitude
#     Tm = kwargs.get('Tm',None) # sec, mean period
#     Ts = kwargs.get('Ts',None) # sec, site period
#     ky = kwargs.get('ky',None) # g, yield acceleration, either provided or computed below
#     n_sample = kwargs.get('n_sample',1) # number of samples, default to 1
#     n_site = kwargs.get('n_site',pga[0].shape[1]) # number of sites
#     n_event = kwargs.get('n_event',pga[0].shape[0]) # number of ruptures
#     return_param = kwargs.get('return_param','pgd_land') # default to liq susc

#     # initialize diciontary for displacement
#     d = {}

#     if pga is None or ky is None:
#         logging.info('Not enough inputs: cannot proceed with evaluation')
#         return None

#     else:
#         # compute Kmax to be used in place of pga
#         try:
#             if Ts/Tm >= 0.1:
#                 Kmax = pga * np.exp((0.459 - 0.702*pga) * np.log((Ts/Tm)/0.1) + \
#                                     (-0.228 + 0.076*pga) * np.log((Ts/Tm)/0.1)**2)
#             else:
#                 Kmax = pga * np.exp(0)
#         except:
#             logging.info('Not enough inputs for Kmax; setting Kmax = pga')
#             Kmax = pga

#         # modified scalar model: f(pga,M)
#         if M is not None:

#             # Table 4.2 in Saygili (2008) dissertation
#             a1 = 4.89
#             a2 = -4.85
#             a3 = -19.64
#             a4 = 42.49
#             a5 = -29.06
#             a6 = 0.72
#             a7 = 0.89

#             # cm, displacement of rigid sliding mass
#             if Kmax > ky:
#                 d = np.exp(a1 + a2*(ky/Kmax) + a3*(ky/Kmax)**2 + a4*(ky/Kmax)**3 + a5*(ky/Kmax)**4 + \
#                         a6*np.log(Kmax) + a7*(M-6)) # eq. 4.6 in Saygili (2008) dissertation

#                 # cm, correct displacement for rigid mass for site flexibility, eq. 3 in Rathje and Antonakos (2011)
#                 if Ts is not None:
#                     if Ts <= 1.5:
#                         d = np.exp(np.log(d) + 3.69*Ts - 1.22*Ts**2)
#                     else:
#                         d = np.exp(np.log(d) + 2.78)
#             else:
#                 d = 0

#             # sigma for ln(D_flex)
#             sigma = 0.694 + 0.32*(ky/Kmax) # eq. 5 in Rathje and Antonakos (2011)

#             d_dict.update({'ky_pga_M': [d,sigma]})

#         # two-parameter vector model: f(pga,pgv)
#         if pgv is not None:

#             # compute K_velmax to be used in place of pgv
#             try:
#                 if Ts/Tm >= 0.2:
#                     K_velmax = pgv * np.exp(0.240 * np.log((Ts/Tm)/0.2) + \
#                                             (-0.091 - 0.171*pga) * np.log((Ts/Tm)/0.2)**2)
#                 else:
#                     K_velmax = pgv * np.exp(0)

#             except:
#                 logging.info('Not enough inputs for K_velmax; setting K_velmax = pgv')
#                 K_velmax = pgv

#             # Table 4.3 in Saygili (2008) dissertation
#             a1 = -1.56
#             a2 = -4.58
#             a3 = -20.84
#             a4 = 44.75
#             a5 = -30.50
#             a6 = -0.64
#             a7 = 1.55

#             # cm, displacement of rigid sliding mass
#             if Kmax > ky:
#                 d = np.exp(a1 + a2*(ky/Kmax) + a3*(ky/Kmax)**2 + a4*(ky/Kmax)**3 + a5*(ky/Kmax)**4 +\
#                         a6*np.log(Kmax) + a7*np.log(K_velmax)) # eq. 4.8 in Saygili (2008) dissertation

#                 # cm, correct displacement for rigid mass for site flexibility, eq. 3 in Rathje and Antonakos (2011)
#                 if Ts is not None:
#                     if Ts <= 0.5:
#                         d = np.exp(np.log(d) + 1.42*Ts)
#                     else:
#                         d = np.exp(np.log(d) + 0.71)
#             else:
#                 d = 0

#             # sigma for ln(D_flex)
#             sigma = 0.400 + 0.284*(ky/Kmax) # eq. 5 in Rathje and Antonakos (2011)

#             d_dict.update({'ky_pga_pgv': [d,sigma]})

#         # store outputs
#         output = {}
#         #
#         if 'pgd_land' in return_param:
#             output.update({'pgd_land': d.copy()})
#         # store probability distribution parameters
#         output.update({'prob_dist': {
#             'type': prob_dist_type,
#             'sigma_total': sigma_total,
#             'sigma_aleatory': sigma_aleatory,
#             'sigma_epistemic': sigma_epistemic}})

#         #
#         return output


# # -----------------------------------------------------------
# def BrayTravasarou2007(gm_type, ky, M, **kwargs):
#     """
#     Compute slope displacement at a given location using the Bray and Travasarou (2007) probabilistic model. Based on the site period, **Ts**, the model decides whether to use the PGA (for Ts < 0.05 sec) or the Sa(Ts_deg) correlation.

#     Note that:\n
#     - The site period can either be specified or calculated using site conditions; see :func:`_edp.fcn_liq_land.get_Ts`
#     - If **Ts** < 0.05 sec, PGA must be entered. For all other **Ts**, either **Sa_Ts_deg** or a list of of periods (**T_arr**) and spectral accelerations (**Sa_arr**) are required.

#     Parameters
#     ----------
#     ky : float
#         [g] yield acceleration
#     M : float
#         moment magnitude
#     Ts : float, optional
#         [sec] site period (see notes above)
#     pga : float, optional
#         [g] peak ground acceleration

#     Parameters for PGA correction (if PGA is specified)
#     beta : float
#         [degree] slope angle, for PGA correction (see Rathje & Bray, 2001; Ashford & Sitar, 2002)
#     flag_topo : str
#         **local** or **long** potential sliding mass, for PGA correction

#     Parameters for Ts
#     H : float
#         [m] slope/structure height
#     vs : float
#         [m/sec] shear wave velocity
#     nDim : int
#         **1** (trapezoidal) or **2** (triangular) dimension for calculating **Ts**; default = 2

#     Parameters for Ts_deg
#     T_arr : float, array
#         [sec] array of periods for interpolation of Ts_deg
#     Sa_arr : float, array
#         [g] array of spectral accelerations for interpolation of Ts_deg

#     Returns
#     -------
#     d : float
#         [cm] slope displacement

#     References
#     ----------
#     .. [1] Bray, J.D., and Travasarou, T., 2007, Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 133, no. 4, pp. 381-392.
#     .. [2] Bray, J.D., Macedo, J., and Travasarou, T., 2018, Simplified Procedure for Estimating Seismic Slope Displacements for Subduction Zone Earthquakes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 144, no. 3, 04017124.
#     .. [3] Rathje, E.M., and Bray, J.D., 2001, One- and Two-Dimensional Seismic Analysis of Solid-Waste Landfills, Canadian Geotechnical Journal, vol. 38, no. 4, pp. 850–862.
#     .. [4] Ashford, S.A., and Sitar, N., 2002, Simplified Method for Evaluating Seismic Stability of Steep Slopes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 128, no. 2, pp. 119-128.

#     """

#     # Inputs
#     # Demand
#     pga = kwargs.get('pga',None) # g, peak ground acceleration
#     M = kwargs.get('M',None) # moment magnitude
#     beta = kwargs.get('beta',None) # degree, shallow = <30, moderately steep = 30 to 60, steep = > 60, Ashford & Sitar (2002)
#     flag_topo = kwargs.get('flag_topo','local') # localized versus long potential sliding mass

#     # Fundamental period of structure
#     Ts = kwargs.get('Ts',None) # sec, allow user-defined Ts = 0

#     # Strength
#     ky = kwargs.get('ky',None) # yield coefficient, from pseudostatic analysis or Grant et al. (2016) correlations 

#     # Calculations
#     # Correct pga for steepness of slope and localization of failure
#     if pga is not None:
#         if 'long' in flag_topo:
#             f_pga = 0.65 # long potential sliding mass
#         elif 'local' in flag_topo: # localized potential sliding mass
#             # factor for increase in pga given steepness of slope
#             if beta is None:
#                 f_pga = 1 # default
#             else:
#                 if beta > 30 and beta <= 60:
#                     f_pga = 1.3 # Rathje & Bray (2001)
#                 elif beta > 60:
#                     f_pga = 1.5 # Ashford & Sitar (2002)
#                 else:
#                     f_pga = 1 # default
#         pga_corr = pga*f_pga # slope-corrected pga
#     else:
#         f_pga = None
#         pga_corr = None

#     # If user defines Ts < 0.05 sec, then does not go in to interpolate Sa(Ts_deg)
#     if Ts < 0.05:
#         Ts_deg = None
#         Sa_Ts_deg = None
#     else:
#         Sa_Ts_deg = kwargs.get('Sa_Ts_deg',None) # g, spectral acceleration at the degraded period, Ts_deg

#         # if Sa_Ts_deg is not given, then interpolate
#         if Sa_Ts_deg is None:

#             # check if Ts is given
#             if Ts is None:
#                 H = kwargs.get('H',None) # m, thickness/height of structure/slope
#                 vs = kwargs.get('vs',None) # m/sec, shear wave velocity
#                 nDim = kwargs.get('nDim',None) # 1D versus 2D case for fundamental period (trapezoidal-shaped or triangular-shaped)
#                 Ts = fcn_liq_land.get_Ts(H,vs,nDim) # sec, fundamental period of site/structure

#             # calculate degraded period and use it to interpolate Sa(Ts_deg)
#             Ts_deg = 1.3*Ts # sec, degraded period
#             T_arr = kwargs.get('T_arr',None) # sec, periods for the input spectral accelerations, used to interpolate Sa(Ts_deg)
#             Sa_arr = kwargs.get('Sa_arr',None) # g, spectral accelerations used to interpolate Sa(Ts_deg)
#             Sa_Ts_deg = np.exp(np.interp(Ts_deg, T_arr, np.log(Sa_arr))) # g, spectral acceleration at degraded period

#     # standard deviation
#     sigma = 0.66

#     # period-dependent coefficients and parameters
#     if Ts < 0.05:
#         a1 = -0.22
#         a2 = 0.0
#         Sa = pga_corr
#     else:
#         a1 = -1.10
#         a2 = 1.50
#         Sa = Sa_Ts_deg

#     # seismic displacement, cm (eq. 5)
#     ln_d = a1 - 2.83*np.log(ky) - 0.333*(np.log(ky))**2 + 0.566*np.log(ky)*np.log(Sa) +\
#         3.04*np.log(Sa) - 0.244*(np.log(Sa))**2 + a2*Ts + 0.278*(M-7)

#     # probably of displacement = 0 (threshold = 1 cm), eq. 3
#     p_d_eq_0 = 1-norm.cdf(-1.76 - 3.22*np.log(ky) - 0.484*Ts*np.log(ky) + 3.52*np.log(Sa))

#     # other percentiles
#     ln_d_84 = ln_d - sigma
#     ln_d_16 = ln_d + sigma

#     # convert from ln_d to d
#     d_arr = np.exp([ln_d_84, ln_d, ln_d_16]) # cm
#     d = d_arr[1]

#     # calculate p(d > D) for given D array
#     # d_arr = kwargs.get('d_arr',None)
#     # if d_arr is not None:
#         # for i in d_arr:
#             # p_d_gt_d = (1-p_D_eq_0) * (1-norm.sf(np.log(i), loc=lnD, scale=sigma)) # eq. 10 and 11
#     # else:
#         # p_d_gt_d = None

#     #
#     return d


# # -----------------------------------------------------------
# def BrayMacedo2019(**kwargs):
#     """
#     Compute slope displacement at a given location using the Bray and Macedo (2007) probabilistic model. Based on the site period, **Ts**, the model decides whether to use the PGA (for Ts = 0 sec) or the Sa(Ts_deg) correlation. Regression models based on three sets of ground motions are available:

#     1. **Ordinary**: **d** = f(ky, Sa(T), Ts, M)
#     2. **Near-fault**: **d** = f(ky, Sa(T), Ts, M, pgv)
#     3. **General** (default): **d** = f(ky, Sa(T), Ts, M, pgv)

#     Note that:\n
#     - The site period can either be specified or calculated using site conditions; see :func:`_edp.fcn_liq_land.get_Ts`
#     - If **Ts** < 0.05 sec, PGA must be entered. For all other **Ts**, either **Sa_Ts_deg** or a list of of periods (**T_arr**) and spectral accelerations (**Sa_arr**) are required.

#     Parameters
#     ----------
#     gm_type : str
#         type of ground motions used to develop the regressional model: **ordinary**, **near-fault**, or **general** (ordinary + near-fault)
#     ky : float
#         [g] yield acceleration
#     M : float
#         moment magnitude
#     Ts : float
#         [sec] site period (see notes above); default = 0.15 sec if site/geolgic data is not sufficient for calculation
#     pga : float
#         [g] peak ground acceleration
#     pgv : float
#         [cm/s] peak ground velocity, required for **near-fault** and **general** models

#     Parameters for PGA correction (if PGA is specified)
#     beta : float, optional
#         [degree] slope angle, for PGA correction (see Rathje & Bray, 2001; Ashford & Sitar, 2002)
#     flag_topo : str, optional
#         **local** or **long** potential sliding mass, for PGA correction

#     Parameters for Ts
#     H : float
#         [m] slope/structure height
#     vs : float
#         [m/sec] shear wave velocity
#     nDim : int
#         **1** (trapezoidal) or **2** (triangular) dimension for calculating **Ts**; default = 2

#     Parameters for Ts_deg
#     T_arr : float, array
#         [sec] array of periods for interpolation of Ts_deg
#     Sa_arr : float, array
#         [g] array of spectral accelerations for interpolation of Ts_deg

#     Returns
#     -------
#     d : float
#         [cm] slope displacement

#     References
#     ----------
#     .. [1] Bray, J.D., and Macedo, J., 2019, Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 145, pp. 12, 04019106.
#     .. [2] Bray, J.D., and Travasarou, T., 2007, Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 133, no. 4, pp. 381-392.
#     .. [3] Rathje, E.M., and Bray, J.D., 2001, One- and Two-Dimensional Seismic Analysis of Solid-Waste Landfills, Canadian Geotechnical Journal, vol. 38, no. 4, pp. 850–862.
#     .. [4] Ashford, S.A., and Sitar, N., 2002, Simplified Method for Evaluating Seismic Stability of Steep Slopes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 128, no. 2, pp. 119-128.

#     """

#     # Inputs
#     # Demand
#     pga = kwargs.get('pga',None) # g, peak ground acceleration
#     pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
#     n_site = kwargs.get('n_site',pga[0].shape[1]) # number of sites
#     n_event = kwargs.get('n_event',pga[0].shape[0]) # number of ruptures
#     return_param = kwargs.get('return_param',['pgd_land','p_land']) # default to liq susc
#     # gm_type = kwargs.get('gm_type','gen') # takes ord = ordinary GMs, nf = near-fault GMs, gen/full = general or full (ordinary + near-fault) GMs
#     gm_type = kwargs.get('GroundMotionType','gen') # takes ord = ordinary GMs, nf = near-fault GMs, gen/full = general or full (ordinary + near-fault) GMs
#     ky = kwargs.get('ky',None) # yield accelerations
#     # Ts = kwargs.get('Ts',0.15) # sec, site period = 4H/Vs, use 0.15 sec if site/geologic data is not available
#     M = kwargs.get('M',None) # moment magnitude
#     # beta = kwargs.get('beta',None) # deg, shallow = <30, moderately steep = 30 to 60, steep = >60, Ashford & Sitar (2002)
#     d0 = kwargs.get('d0',0.5) # cm, displacement threshold below which is considered "zero"
#     # flag_topo = kwargs.get('flag_topo','local') # localized versus long potential sliding mass
#     slope_loc = kwargs.get('slope_loc',90) # deg, slope location with respect to fault-normal for near-fault GMs
#                                         # if <= 45, output D100, else, output D50
#                                         # default to 90 deg (fault parallel)
#     n_sample = kwargs.get('n_sample',1) # number of samples, default to 1

#     # if pga is given, approximate using Ts = 0.01 and Sa(Ts_deg) ~= pga
#     if pga is not None:
#         Ts = 0.01
#         Sa_Ts_deg = pga

#     # get pgv- and period-dependent coefficients and sigmas
#     if 'ord' in gm_type.lower():
#         # total sigma
#         sigma_total = 0.72 # for ln(D)

#         # initialize matrix for coeffcients
#         a1 = np.zeros([n_event,n_site])
#         a2 = np.zeros([n_event,n_site])
#         a3 = np.zeros([n_event,n_site])

#         a1[Ts<=0.01] = -4.684
#         a1[np.logical_and(Ts>0.01,Ts<0.1)] = -4.684
#         a1[Ts>=0.1] = -5.981

#         a2[Ts<=0.01] = 0.0
#         a2[np.logical_and(Ts>0.01,Ts<0.1)] = -9.471
#         a2[Ts>=0.1] = 3.223

#         a3[Ts<=0.01] = 0.0
#         a3[np.logical_and(Ts>0.01,Ts<0.1)] = 0.0
#         a3[Ts>=0.1] = -0.945

#     elif 'nf' in gm_type.lower() or 'near' in gm_type.lower() or 'fault' in gm_type.lower():
#         # check if slope is oriented within 45 degree of fault-normal direction
#         # if slope_loc <= 45:
#             # total sigma
#             # sigma_total = 0.56 # for ln(D)
#         # elif slope_loc > 45:
#             # total sigma
#             # sigma_total = 0.54 # for ln(D)
            
#         # use site-independent sigma of 0.55 for now
#             sigma_total = 0.55

#     elif 'full' in gm_type.lower() or 'gen' in gm_type.lower():
#         # total sigma
#         sigma_total = 0.74 # for ln(D)

#     # distribution based on case:
#     prob_dist_type = 'Lognormal'
#     # aleatory
#     sigma_aleatory = 0.4 # for ln(D)
#     # estimate epistemic from total and aleatory
#     sigma_epistemic = (sigma_total**2-sigma_aleatory**2)**0.5 # for ln(D)

#     # initialize dictionaries for outputs
#     d = {}
#     p_land = {}

#     # loop through samples
#     for k in range(n_sample):

#         # get current sampled inputs
#         Sa_Ts_deg_k = Sa_Ts_deg[k].data
#         pgv_k = pgv[k].data
#         row_k = pgv[k].row
#         col_k = pgv[k].col

#         # get other params for current sample
#         ky_k = ky[col_k]
#         M_k = M[row_k]
#         Ts_k = np.ones(pgv_k.shape)*Ts

#         # see which case is picked:
#         if 'ord' in gm_type.lower():

#             # seismic displacement, cm (eq. 3)
#             ln_d_k = a1 - 2.482*np.log(ky_k) - 0.244*np.multiply(np.log(ky_k),np.log(ky_k)) +\
#                         0.344*np.multiply(np.log(ky_k),np.log(Sa_Ts_deg_k)) + 2.649*np.log(Sa_Ts_deg_k) -\
#                         0.090*np.multiply(np.log(Sa_Ts_deg_k),np.log(Sa_Ts_deg_k)) + a2*Ts_k +\
#                         a3*np.multiply(Ts_k,Ts_k) + 0.603*M_k

#             # probably of displacement = 0 (threshold = 0.5 cm)
#             if 'p_land' in return_param:
#                 p_land_k = np.zeros(pgv_k.shape)
#                 p_land_k[Ts_k<=0.7] = norm.cdf(-2.48 - 2.97*np.log(ky_k[Ts_k<=0.7]) -\
#                                         0.12*np.multiply(np.log(ky_k[Ts_k<=0.7]),np.log(ky_k[Ts_k<=0.7])) -\
#                                         0.72*np.multiply(Ts_k[Ts_k<=0.7],np.log(ky_k[Ts_k<=0.7])) +\
#                                         1.70*Ts_k[Ts_k<=0.7] + 2.78*np.log(Sa_Ts_deg_k[Ts_k<=0.7]))# eq. 2(a)
#                 p_land_k[Ts_k>0.7] = norm.cdf(-3.42 - 4.93*np.log(ky_k[Ts_k>0.7]) -\
#                                         0.30*np.multiply(np.log(ky_k[Ts_k>0.7]),np.log(ky_k[Ts_k>0.7])) -\
#                                         0.35*np.multiply(Ts_k[Ts_k>0.7],np.log(ky_k[Ts_k>0.7])) -\
#                                         0.62*Ts_k[Ts_k>0.7] + 2.86*np.log(Sa_Ts_deg_k[Ts_k>0.7])) # eq. 2(b)
#                 # p_land.update({k:sparse.coo_matrix(1-p_D_eq_0_k)})


#         # -----------------------------------------------------------
#         # -----------------------------------------------------------
#         # need to update definition of coefficients for near-fault case
#         elif 'nf' in gm_type.lower() or 'near' in gm_type.lower() or 'fault' in gm_type.lower():

#             # check if slope is oriented within 45 degree of fault-normal direction
#             if slope_loc <= 45:

#                 # pgv- and period-dependent coefficients and parameters for D100
#                 if pgv <= 150:
#                     if Ts < 0.1:
#                         c1 = -6.724
#                         c2 = -2.744
#                         c3 = 0.0
#                         c4 = 1.547
#                     elif Ts >= 0.1:
#                         c1 = -6.951
#                         c2 = 1.069
#                         c3 = -0.498
#                         c4 = 1.547
#                 if pgv > 150:
#                     if Ts < 0.1:
#                         c1 = 1.764
#                         c2 = -2.744
#                         c3 = 0.0
#                         c4 = -0.097
#                     elif Ts >= 0.1:
#                         c1 = 1.991
#                         c2 = 1.069
#                         c3 = -0.498
#                         c4 = -0.097

#                 # maximum seismic displacement, D100 (eq. 5)
#                 ln_d = c1 - 2.632*np.log(ky) - 0.278*(np.log(ky))**2 + 0.527*np.log(ky)*np.log(Sa) +\
#                     1.978*np.log(Sa) - 0.233*(np.log(Sa))**2 + c2*Ts + c3*Ts**2 + 0.01*M +\
#                     c4*np.log(pgv)

#                 # probability of maximum seismic displacement = 0
#                 if Ts <= 0.7:
#                     # eq. 4(a)
#                     p_D_eq_0 = 1/(1 + np.exp(-10.787 - 8.717*np.log(ky) + 1.660*np.log(pgv) + 3.150*Ts +
#                                             7.560*np.log(Sa)))
#                 elif Ts > 0.7:
#                     # eq. 4(b)
#                     p_D_eq_0 = 1/(1 + np.exp(-12.771 - 9.979*np.log(ky) + 2.286*np.log(pgv) - 4.965*Ts +
#                                             4.817*np.log(Sa)))

#             elif slope_loc > 45:

#                 # pgv- and period-dependent coefficients and parameters for D50
#                 if pgv <= 150:
#                     if Ts < 0.1:
#                         c1 = -7.497
#                         c2 = -2.731
#                         c3 = 0.0
#                         c4 = 1.458
#                     elif Ts >= 0.1:
#                         c1 = -7.718
#                         c2 = 1.031
#                         c3 = -0.480
#                         c4 = 1.458
#                 if pgv > 150:
#                     if Ts < 0.1:
#                         c1 = -0.148
#                         c2 = -2.731
#                         c3 = 0.0
#                         c4 = 0.025
#                     elif Ts >= 0.1:
#                         c1 = -0.369
#                         c2 = 1.031
#                         c3 = -0.480
#                         c4 = 0.025

#                 # median seismic displacement, D50 (eq. 7)
#                 ln_d = c1 - 2.931*np.log(ky) - 0.319*(np.log(ky))**2 + 0.584*np.log(ky)*np.log(Sa) +\
#                     2.261*np.log(Sa) - 0.241*(np.log(Sa))**2 + c2*Ts + c3*Ts**2 + 0.05*M +\
#                     c4*np.log(pgv)

#                 # probability of maximum seismic displacement = 0
#                 if Ts <= 0.7:
#                     # eq. 5(a)
#                     p_D_eq_0 = 1/(1 + np.exp(-14.930 - 10.383*np.log(ky) + 1.971*np.log(pgv) + 3.763*Ts +
#                                             8.812*np.log(Sa)))
#                 elif Ts > 0.7:
#                     # eq. 6(b)
#                     p_D_eq_0 = 1/(1 + np.exp(-14.671 - 10.489*np.log(ky) + 2.222*np.log(pgv) - 4.759*Ts +
#                                             5.549*np.log(Sa)))
#         # -----------------------------------------------------------
#         # -----------------------------------------------------------


#         elif 'full' in gm_type.lower() or 'gen' in gm_type.lower():

#             # initialize matrix for coefficients
#             a1 = np.zeros(pgv_k.shape)
#             a2 = np.zeros(pgv_k.shape)
#             a3 = np.zeros(pgv_k.shape)
#             a4 = np.zeros(pgv_k.shape)
#             a5 = np.zeros(pgv_k.shape)

#             # set values for coefficients
#             a1[Ts_k<0.1] = -4.551
#             a1[Ts_k>=0.1] = -5.894

#             a2[Ts_k<0.1] = -9.690
#             a2[Ts_k>=0.1] = 3.152

#             a3[Ts_k>=0.1] = -0.910

#             a4[pgv_k>115] = 1.0

#             a5[pgv_k>115] = -4.75

#             # seismic displacement, cm (eq. 9)
#             ln_d_k = a1 - 2.491*np.log(ky_k) - 0.245*np.multiply(np.log(ky_k),np.log(ky_k)) +\
#                     0.344*np.multiply(np.log(ky_k),np.log(Sa_Ts_deg_k)) + 2.703*np.log(Sa_Ts_deg_k) -\
#                     0.089*np.multiply(np.log(Sa_Ts_deg_k),np.log(Sa_Ts_deg_k)) + a2*Ts_k +\
#                     a3*np.multiply(Ts_k,Ts_k) + 0.607*M_k + a4*np.log(pgv_k) + a5

#             # probably of displacement = 0 (threshold = 0.5 cm)
#             if 'p_land' in return_param:
#                 p_land_k = np.zeros(pgv_k.shape)
#                 p_land_k[Ts_k<=0.7] = norm.cdf(-2.46 - 2.98*np.log(ky_k[Ts_k<=0.7]) -\
#                                         0.12*np.multiply(np.log(ky_k[Ts_k<=0.7]),np.log(ky_k[Ts_k<=0.7])) -\
#                                         0.71*np.multiply(Ts_k[Ts_k<=0.7],np.log(ky_k[Ts_k<=0.7])) +\
#                                         1.69*Ts_k[Ts_k<=0.7] + 2.76*np.log(Sa_Ts_deg_k[Ts_k<=0.7])) *100 # %, eq. 8(a)
#                 p_land_k[Ts_k>0.7] = norm.cdf(-3.40 - 4.95*np.log(ky_k[Ts_k>0.7]) -\
#                                         0.30*np.multiply(np.log(ky_k[Ts_k>0.7]),np.log(ky_k[Ts_k>0.7])) -\
#                                         0.33*np.multiply(Ts_k[Ts_k>0.7],np.log(ky_k[Ts_k>0.7])) -\
#                                         0.62*Ts_k[Ts_k>0.7] + 2.85*np.log(Sa_Ts_deg_k[Ts_k>0.7])) *100 # %, eq. 8(b)

#         # exp() and update dictionary
#         d_k = np.exp(ln_d_k)
#         d_k[np.isnan(d_k)] = 0
#         d_k[d_k<=d0] = 0
#         ind_red = np.where(d_k>0)[0]
#         d_k_red = d_k[ind_red]
#         row_k_red = row_k[ind_red]
#         col_k_red = col_k[ind_red]
#         d.update({k:sparse.coo_matrix((d_k_red,(row_k_red,col_k_red)),shape=(n_event,n_site))})

#         # update dictionary if p_land is inquired
#         if 'p_land' in return_param:
#             p_land_k_red = p_land_k[ind_red]
#             p_land.update({k:sparse.coo_matrix((p_land_k_red,(row_k_red,col_k_red)),shape=(n_event,n_site))})

#     # store outputs
#     output = {}
#     #
#     if 'pgd_land' in return_param:
#         output.update({'pgd_land': d.copy()})
#     if 'p_land' in return_param:
#         output.update({'p_land': p_land.copy()})
#     # store probability distribution parameters
#     output.update({'prob_dist': {
#         'type': prob_dist_type,
#         'sigma_total': sigma_total,
#         'sigma_aleatory': sigma_aleatory,
#         'sigma_epistemic': sigma_epistemic}})

#     #
#     return output


# # -----------------------------------------------------------
# def Hazus2014(land_susc, pga, M, pga_is):
#     """
#     Compute landslide displacement at a given location using a simplified deterministic approach (after Makdisi & Seed, 1978).

#     Parameters
#     ----------
#     land_susc : int
#         susceptibility category to deep-seated landslide (1 to 10, see Wills et al., 2011)
#     M : float
#         moment magnitude
#     pga : float
#         [g] peak ground acceleration
#     pga_is : float, optional
#         [g] average (induced) acceleration within entire slide mass; default = PGA

#     Returns
#     -------
#     d : float
#         [cm] permanent ground displacement

#     References
#     ----------
#     .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
#     .. [2] Makdisi, F.I., and Seed, H.B., 1978, Simplified Procedure for Estimating Dam and Embankment Earthquake-Induced Deformations, Journal of Geotechnical and Geoenvironmental Engineering, vol. 104, no. 7, pp. 849-867.
#     .. [3] Seed, H.B., and Idriss, I.M., 1982, Ground Motions and Soil Liquefaction During Earthquakes, Earthquake Engineering Research Institute, Oakland, California, Monograph Series.
#     .. [4] Wills, C.J., Perez, F.G., and Gutierrez, C.I., 2011, Susceptibility to Deep-Seated Landslides in California. California Geological Survey Map Sheet no. 58.

#     """

#     # Load Makdisi & Seed (1978) digitized data for the displacement factor, d/pga_is
#     #     where d is the displacement, and pga_is is the induced acceleration
#     # Note: the data file is composed of two sets of curves (uppper- and lower-bound)
#     dir_makdisi_seed = os.path.join(os.path.dirname(os.getcwd()),'data','makdisi_seed.txt')
#     makdisi_seed = pd.read_csv(dir_makdisi_seed, sep='\t')
#     makdisi_seed_keys = makdisi_seed.keys()

#     # Critical PGA based on landslide susceptibility (Wills et al., 2011)
#     pga_c = [0.60 if land_susc == 1 else
#             0.50 if land_susc == 2 else
#             0.40 if land_susc == 3 else
#             0.35 if land_susc == 4 else
#             0.30 if land_susc == 5 else
#             0.25 if land_susc == 6 else
#             0.20 if land_susc == 7 else
#             0.15 if land_susc == 8 else
#             0.10 if land_susc == 9 else
#             0.05 if land_susc == 10 else 999]
#     pga_c = pga_c[0]

#     # Magnitude and duration correction
#     n_cyc = 0.3419 * M**3 - 5.5214 * M**2 + 33.6154 * M - 70.7692 # number of cycles, after Seed & Idriss (1982)

#     # average (induced) acceleration within entire slide mass
#     pga_is = pga # g, default: pga_is = pga

#     # upper-bound for displacement factor, d/pga_is
#     d_pgais_n_upper = np.interp(pga_c/pga_is,makdisi_seed[makdisi_seed_keys[0]],makdisi_seed[makdisi_seed_keys[1]])

#     # lower-bound for displacement factor, d/pga_is
#     d_pgais_n_lower = np.interp(pga_c/pga_is,makdisi_seed[makdisi_seed_keys[2]],makdisi_seed[makdisi_seed_keys[3]])

#     # Estimate median displacement factor (assuming lognormal distribution for displacements)
#     d_pgais_n = np.exp(np.log(d_pgais_n_upper) + np.log(d_pgais_n_lower))/2

#     # Calculate displacement and correct for number of cycles
#     d = d_pgais_n * pga_is * n_cyc # cm

#     #
#     return d