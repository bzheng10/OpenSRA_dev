# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for liquefaction triggering: CRR, CSR, FS, and/or pLiq
#
# Created: April 13, 2020
# Updated: July 14, 2020
# --- improved documentation
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging

# data manipulation modules
import numpy as np
# from numpy import tan, radians, where
# from numba import jit
# from numba import njit
from scipy import sparse
from scipy.stats import norm

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class Liquefaction(BaseModel):
    "Inherited class specfic to liquefaction triggering"
    
    # _RETURN_PBEE_META = {
    #     'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'liquefaction',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'prob_liq'
    #     ] # Return variable for PBEE category, e.g., pgdef, eps_pipe
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PGE2022(Liquefaction):
    """
    A procedure to quantify probability of liquefaction at a given location using a model by PG&E (2022):

    1. For distance to coast < cutoff, **prob_liq** = f(**pgv**, **vs30**, **precip**, **dist_coast**, **dist_river**)
    2. For distance to coast >= cutoff, **prob_liq** = f(**pgv**, **vs30**, **precip**, **dist_coast**, **dist_river**, **gw_depth**)
    
    Parameters
    ----------
    From upstream PBEE:
    mag: float, np.ndarray or list
        moment magnitude
    pga: float, np.ndarray or list
        [g] peak ground acceleration
        
    Geotechnical/geologic:
    pge_a: float, np.ndarray or list
        [probability (fraction)] coefficent a from pge rasters
    pge_b: float, np.ndarray or list
        [g] coefficent b from pge rasters
    pge_c: float, np.ndarray or list
        [dimensionless] coefficent c from pge rasters
    msf_max: float, np.ndarray or list
        [dimensionless] maximum magnitude scale factor 

    Returns
    -------
    prob_liq : float, np.ndarray
        [%] probability for liquefaciton
    
    References
    ----------
    .. [1] Pacific Gas & Electric (PG&E), 2022, San Francisco Bay Area Liquefaction and Lateral Spread Study, Pacific Gas & Electric (PG&E) Revised Report.
    .. [2] Greenfield, M. W., Guckenheimer, M., Wade, A., Wilson, J. M., Hitchcock, C., Kottke, A. R., & Boone, M. (2022). A Tool To Evaluate Liquefaction and Resulting Permanent Ground Deformation In The San Francisco Bay Area. In Geo-Risk 2023 (pp. 285-295).
    
    """

    _NAME = 'PG&E (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Pacific Gas & Electric (PG&E), 2022, San Francisco Bay Area Liquefaction and Lateral Spread Study, Pacific Gas & Electric (PG&E) Revised Report.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'prob_liq': {
                'desc': 'probability of liquefaction',
                'unit': 'decimal',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': 0.25, # base model uncertainty, does not include input uncertainty,
                # 'dist_type': 'lognormal',
            }
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': [
    #         'pgv', 'pga'
    #     ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': None,
                # 'dist_type': 'lognormal'
            },
            'mag': {
                'desc': 'moment magnitude',
                'unit': '',
                # 'mean': None,
                # 'dist_type': 'fixed'
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
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'pge_a': 'coefficent a from pge rasters',
            'pge_b': 'coefficent b from pge rasters (g)',
            'pge_c': 'coefficent c from pge rasters',
            'msf_max': 'maximum magnitude scale factor',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        # 'vs30', 'precip', 'dist_coast', 'dist_river', 'dist_water', 'gw_depth',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'pge_a', 'pge_b', 'pge_c', 'msf_max'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        pga, mag, # upstream PBEE RV
        pge_a, pge_b, pge_c, msf_max, # geotechnical/geologic
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # distance cutoff for model
        # model_transition = 20 # km
        
        # hard PGA
        
        # initialize arrays
        msf = np.empty(pga.shape)
        prob_liq = np.empty(pga.shape)
        
        # # magnitude correction, from Baise & Rashidian (2020) and Allstadt et al. (2022)
        # pgv_mag = pgv/(1+np.exp(-2*(mag-6)))
        # pga_mag = pga/(10**2.24/mag**2.56)
        
        # index for all values
        
        # magnitude scaling factor value
        msf = 1 + (msf_max - 1)*(8.64 * np.exp(-1 * mag / 4) - 1.325)

        # probability of liquefaction
        prob_liq = pge_a / (1 + ((pga / msf) / pge_b) ** pge_c)

        # calculate sigma_mu (not sure if this should be here)
        sigma_mu = (np.exp(0.25)-1) * prob_liq
        
        # apply lower range limit
        prob_liq[np.isnan(prob_liq)] = zero_prob_liq
        prob_liq = np.maximum(prob_liq,zero_prob_liq)
        
        # prepare outputs
        output = {
            'prob_liq': {
                'mean': prob_liq,
                'sigma': None,
                # 'sigma_mu': np.ones(prob_liq.shape)*0.25,
                'sigma_mu': sigma_mu,
                'dist_type': 'normal',
                'unit': 'decimal'
            },
            # # 'liq_susc': liq_susc,
            # 'liq_susc_val': liq_susc_val
        }
        # get intermediate values if requested
        if return_inter_params:
            output['msf'] = msf
            # output['x_logistic'] = x_logistic
            # output['pgv_mag'] = pgv_mag
            # output['pga_mag'] = pga_mag
            # output['liq_susc'] = liq_susc
        
        # return
        return output
        

# -----------------------------------------------------------
class Hazus2020_with_ZhuEtal2017(Liquefaction):
    """
    Compute probability of liquefaction using Hazus (FEMA, 2020), with liq. susc. category from Zhu et al. (2017).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    vs30: float, np.ndarray or list
        [m/s] time-averaged shear wave velocity in the upper 30-meters
    precip: float, np.ndarray or list
        [mm] mean annual precipitation
    dist_coast: float, np.ndarray or list
        [km] distance to nearest coast
    dist_river: float, np.ndarray or list
        [km] distance to nearest river
    dist_water: float, np.ndarray or list
        [km] distance to nearest river, lake, or coast
    gw_depth: float, np.ndarray or list
        [m] groundwater table depth
        
    Fixed:
    # liq_susc: str, np.ndarray or list
    #     susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Liao, S.S., Veneziano, D., and Whitman, R.V., 1988, Regression Models for Evaluating Liquefaction Probability, Journal of Geotechnical Engineering, vol. 114, no. 4, pp. 389-411.
    .. [3] Zhu, J., Baise, L.G., and Thompson, E.M., 2017, An Updated Geospatial Liquefaction Model for Global Application, Bulletin of the Seismological Society of America, vol. 107, no. 3, pp. 1365-1385.
    
    """

    _NAME = 'Hazus (FEMA, 2020)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Federal Emergency Management Agency (FEMA), 2020, ',
        'Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, ',
        '436 pp.',
        'https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.',
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'prob_liq': {
                'desc': 'probability of liquefaction',
                'unit': 'decimal',
            },
            'liq_susc_val': {
                'desc': 'liquefaction susceptibility category value',
                'unit': '',
            }
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
            },
            'mag': {
                'desc': 'moment magnitude',
                'unit': '',
            },
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
            'vs30': 'time-averaged shear wave velocity in the upper 30-meters (m/s)',
            'precip': 'mean annual precipitation (mm)',
            'dist_coast': 'distance to nearest coast (km)',
            'dist_river': 'distance to nearest river (km)',
            'dist_water': 'distance to nearest river, lake, or coast (km)',
            'gw_depth': 'groundwater table depth (m)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            # 'liq_susc': 'liquefaction susceptibility category',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'vs30', 'precip', 'dist_coast', 'dist_river', 'dist_water', 'gw_depth',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        # 'liq_susc'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        pga, mag, # upstream PBEE RV
        vs30, precip, dist_coast, dist_river, dist_water, gw_depth, # geotechnical/geologic
        model_transition=20, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # run ZhuEtal2017 to get liquefaction susceptibility
        zhu_return = ZhuEtal2017._model(
            pgv=np.ones(pga.shape), # not used for liquefaction susceptibility category,
            pga=pga, mag=mag, # upstream PBEE RV
            vs30=vs30, precip=precip, dist_coast=dist_coast, # geotechnical/geologic
            dist_river=dist_river, dist_water=dist_water, gw_depth=gw_depth, # geotechnical/geologic
            return_inter_params=True
        )
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # get liquefaction susceptility category and value
        liq_susc_val = zhu_return['liq_susc_val']
        liq_susc = zhu_return['liq_susc']
        pga_mag = zhu_return['pga_mag']
            
        # initialize arrays
        prob_liq_pga = np.zeros(pga.shape)
        p_ml = np.zeros(pga.shape)
        
        # correction factor for moment magnitudes other than M=7.5, eq. 4-21
        k_mag = 0.0027*mag**3 - 0.0267*mag**2 - 0.2055*mag + 2.9188
        # correction for groudnwater depths other than 5 feet, eq. 4-22
        k_gw_depth = 0.022 * gw_depth*3.28 + 0.93
        
        # get uncorrected p_liq given pga
        prob_liq_pga[liq_susc=='very high'] = np.maximum(np.minimum(9.09*pga[liq_susc=='very high']-0.82,1),0)
        prob_liq_pga[liq_susc=='high'] = np.maximum(np.minimum(7.67*pga[liq_susc=='high']-0.92,1),0)
        prob_liq_pga[liq_susc=='moderate'] = np.maximum(np.minimum(6.67*pga[liq_susc=='moderate']-1.00,1),0)
        prob_liq_pga[liq_susc=='low'] = np.maximum(np.minimum(5.57*pga[liq_susc=='low']-1.18,1),0)
        prob_liq_pga[liq_susc=='very low'] = np.maximum(np.minimum(4.16*pga[liq_susc=='very low']-1.08,1),0)
        prob_liq_pga[liq_susc=='none'] = 0
        
        # get portion of map unit susceptible to liquefaction
        p_ml[liq_susc=='very high'] = 0.25
        p_ml[liq_susc=='high'] = 0.20
        p_ml[liq_susc=='moderate'] = 0.10
        p_ml[liq_susc=='low'] = 0.05
        p_ml[liq_susc=='very low'] = 0.02
        p_ml[liq_susc=='none'] = 0.00
        
        # liquefaction likelihood, p_liq
        prob_liq = prob_liq_pga / k_mag / k_gw_depth * p_ml # decimal, eq. 4-20
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log
        
        # Zhu et al. (2017) boundary constraints
        # for pga_mag < 0.1 g, set prob to "0"
        prob_liq[pga_mag<0.1] = zero_prob_liq
        # for vs30 > 620 m/s, set prob to "0"
        prob_liq[vs30>620] = zero_prob_liq
        # for precip > 1700 mm, set prob to "0"
        prob_liq[precip>1700] = zero_prob_liq

        # calculate sigma_mu
        sigma_mu = (np.exp(0.25)-1) * prob_liq
        
        # prepare outputs
        output = {
            # 'prob_liq': prob_liq,
            'prob_liq': {
                'mean': prob_liq,
                'sigma': None,
                # 'sigma_mu': np.ones(prob_liq.shape)*np.exp(0.25),
                'sigma_mu': sigma_mu,
                'dist_type': 'normal',
                'unit': 'decimal'
            },
            'liq_susc_val': liq_susc_val
        }
        # get intermediate values if requested
        if return_inter_params:
            output['pga_mag'] = pga_mag
            output['liq_susc'] = liq_susc
            output['k_mag'] = k_mag
            output['k_gw_depth'] = k_gw_depth
            output['prob_liq_pga'] = prob_liq_pga
            output['p_ml'] = p_ml
        
        # return
        return output


# -----------------------------------------------------------
class Hazus2020(Liquefaction):
    """
    Compute probability of liquefaction at a given location using a simplified method after Liao et al. (1988).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    gw_depth: float, np.ndarray or list
        [m] groundwater table depth
        
    Fixed:
    liq_susc: str, np.ndarray or list
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Liao, S.S., Veneziano, D., and Whitman, R.V., 1988, Regression Models for Evaluating Liquefaction Probability, Journal of Geotechnical Engineering, vol. 114, no. 4, pp. 389-411.
    
    """

    _NAME = 'Hazus (FEMA, 2020)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Federal Emergency Management Agency (FEMA), 2020, ',
        'Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, ',
        '436 pp.',
        'https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.',
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'prob_liq': {
                'desc': 'probability of liquefaction',
                'unit': 'decimal',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
            },
            'mag': {
                'desc': 'moment magnitude',
                'unit': '',
            }
        }
    }
    # _INPUT_DIST_VARY_WITH_LEVEL = False
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        "params": {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'gw_depth': 'groundwater table depth (m)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'liq_susc': 'liquefaction susceptibility category',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        # 'gw_depth',
        'level1': [],
        'level2': ['gw_depth'],
        'level3': ['gw_depth'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': ['liq_susc'],
        'level2': ['liq_susc'],
        'level3': ['liq_susc'],
        # 'liq_susc'
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        pga, mag, # upstream PBEE RV
        gw_depth, # geotechnical/geologic
        liq_susc, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # initialize arrays
        prob_liq_pga = np.zeros(pga.shape)
        p_ml = np.zeros(pga.shape)
        
        # if gw_depth is nan
        gw_depth[np.isnan(gw_depth)] = 999        
        
        # correction factor for moment magnitudes other than M=7.5, eq. 4-21
        k_mag = 0.0027*mag**3 - 0.0267*mag**2 - 0.2055*mag + 2.9188
        # correction for groudnwater depths other than 5 feet, eq. 4-22
        k_gw_depth = 0.022 * gw_depth*3.28 + 0.93
        
        # get uncorrected p_liq given pga
        prob_liq_pga[liq_susc=='very high'] = np.maximum(np.minimum(9.09*pga[liq_susc=='very high']-0.82,1),0)
        prob_liq_pga[liq_susc=='high'] = np.maximum(np.minimum(7.67*pga[liq_susc=='high']-0.92,1),0)
        prob_liq_pga[liq_susc=='moderate'] = np.maximum(np.minimum(6.67*pga[liq_susc=='moderate']-1.00,1),0)
        prob_liq_pga[liq_susc=='low'] = np.maximum(np.minimum(5.57*pga[liq_susc=='low']-1.18,1),0)
        prob_liq_pga[liq_susc=='very low'] = np.maximum(np.minimum(4.16*pga[liq_susc=='very low']-1.08,1),0)
        prob_liq_pga[liq_susc=='none'] = 0
        
        # get portion of map unit susceptible to liquefaction
        p_ml[liq_susc=='very high'] = 0.25
        p_ml[liq_susc=='high'] = 0.20
        p_ml[liq_susc=='moderate'] = 0.10
        p_ml[liq_susc=='low'] = 0.05
        p_ml[liq_susc=='very low'       ] = 0.02
        p_ml[liq_susc=='none'] = 0.00
        
        # liquefaction likelihood, p_liq
        prob_liq = prob_liq_pga / k_mag / k_gw_depth * p_ml # eq. 4-20
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log

        # calculate sigma_mu
        sigma_mu = (np.exp(0.25)-1) * prob_liq

        # prepare outputs
        output = {
            # 'prob_liq': prob_liq,
            'prob_liq': {
                'mean': prob_liq,
                'sigma': None,
                # 'sigma_mu': np.ones(prob_liq.shape)*0.25,
                'sigma_mu': sigma_mu,
                'dist_type': 'normal',
                'unit': 'decimal'
            }
        }
        # get intermediate values if requested
        if return_inter_params:
            output['k_mag'] = k_mag
            output['k_gw_depth'] = k_gw_depth
            output['prob_liq_pga'] = prob_liq_pga
            output['p_ml'] = p_ml
        
        # return
        return output


# -----------------------------------------------------------
class ZhuEtal2017(Liquefaction):
    """
    A map-based procedure to quantify liquefaction at a given location using logistic models by Zhu et al. (2017). Two models are provided:

    1. For distance to coast < cutoff, **prob_liq** = f(**pgv**, **vs30**, **precip**, **dist_coast**, **dist_river**)
    2. For distance to coast >= cutoff, **prob_liq** = f(**pgv**, **vs30**, **precip**, **dist_coast**, **dist_river**, **gw_depth**)
    
    Parameters
    ----------
    From upstream PBEE:
    pgv: float, np.ndarray or list
        [cm/s] peak ground velocity
    mag: float, np.ndarray or list
        moment magnitude
    pga: float, np.ndarray or list
        [g] peak ground acceleration, only to check threshold where prob_liq(pga<0.1g)=0
        
    Geotechnical/geologic:
    vs30: float, np.ndarray or list
        [m/s] time-averaged shear wave velocity in the upper 30-meters
    precip: float, np.ndarray or list
        [mm] mean annual precipitation
    dist_coast: float, np.ndarray or list
        [km] distance to nearest coast
    dist_river: float, np.ndarray or list
        [km] distance to nearest river
    dist_water: float, np.ndarray or list
        [km] distance to nearest river, lake, or coast
    gw_depth: float, np.ndarray or list
        [m] groundwater table depth
        
    Fixed:
    # dist_water_cutoff: float, optional
    #     [km] distance to water cutoff for switching between global and coastal model, default = 20 km

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    liq_susc_val : str, np.ndarray
        liquefaction susceptibility category value
    
    References
    ----------
    .. [1] Zhu, J., Baise, L.G., and Thompson, E.M., 2017, An Updated Geospatial Liquefaction Model for Global Application, Bulletin of the Seismological Society of America, vol. 107, no. 3, pp. 1365-1385.
    
    """

    _NAME = 'Zhu et al. (2017)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Zhu, J., Baise, L.G., and Thompson, E.M., 2017, ',
        'An Updated Geospatial Liquefaction Model for Global Application, ',
        'Bulletin of the Seismological Society of America, ',
        'vol. 107, no. 3, pp. 1365-1385.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'prob_liq': {
                'desc': 'probability of liquefaction',
                'unit': 'decimal',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': 0.25, # base model uncertainty, does not include input uncertainty,
                # 'dist_type': 'lognormal',
            },
            # 'liq_susc': {
            'liq_susc_val': {
                'desc': 'liquefaction susceptibility category value',
                'unit': '',
                # 'mean': None,
                # 'dist_type': 'fixed',
            }
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': [
    #         'pgv', 'pga'
    #     ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgv': {
                'desc': 'peak ground velocity (cm/s)',
                'unit': 'cm/s',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': None,
                # 'dist_type': 'lognormal'
            },
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': None,
                # 'dist_type': 'lognormal'
            },
            'mag': {
                'desc': 'moment magnitude',
                'unit': '',
                # 'mean': None,
                # 'dist_type': 'fixed'
            }
        }
    }
    # _INPUT_DIST_VARY_WITH_LEVEL = False
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        "params": {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'vs30': 'time-averaged shear wave velocity in the upper 30-meters (m/s)',
            'precip': 'mean annual precipitation (mm)',
            'dist_coast': 'distance to nearest coast (km)',
            'dist_river': 'distance to nearest river (km)',
            'dist_water': 'distance to nearest river, lake, or coast (km)',
            'gw_depth': 'groundwater table depth (m)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            # 'coast_model_cutoff': 'cutoff distance to switch to coastal model (default: 20 km)',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': ['vs30', 'precip', 'dist_coast', 'dist_river', 'dist_water', 'gw_depth'],
        'level3': ['vs30', 'precip', 'dist_coast', 'dist_river', 'dist_water', 'gw_depth'],
        # 'vs30', 'precip', 'dist_coast', 'dist_river', 'dist_water', 'gw_depth',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': [],
        'level2': [],
        'level3': [],
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        pgv, pga, mag, # upstream PBEE RV
        vs30, precip, dist_coast, dist_river, dist_water, gw_depth, # geotechnical/geologic
        model_transition=20, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # distance cutoff for model
        # model_transition = 20 # km
        
        # hard PGA
        
        # initialize arrays
        x_logistic = np.empty(pgv.shape)
        prob_liq = np.empty(pgv.shape)
        liq_susc_val = np.ones(pgv.shape)*-99
        liq_susc = np.empty(pgv.shape, dtype='<U10')
        
        # magnitude correction, from Baise & Rashidian (2020) and Allstadt et al. (2022)
        pgv_mag = pgv/(1+np.exp(-2*(mag-6)))
        pga_mag = pga/(10**2.24/mag**2.56)
        
        # find where dist_water <= cutoff for model of 20 km
        # coastal model
        # ind_coastal = np.where(dist_water<=model_transition)[0]
        ind_coastal = dist_water<=model_transition
        # global model
        # ind_global = list(set(list(range(pgv.shape[0]))).difference(set(ind_coastal)))
        ind_global = ~(dist_water<=model_transition)
        
        # set cap of precip to 1700 mm
        precip[precip>1700] = 1700
        
        # x = b0 + b1*var1 + ...
        # if len(ind_global) > 0:
        # liquefaction susceptbility value, disregard pgv term
        liq_susc_val[ind_global] = \
            8.801 + \
            -1.918   * np.log(vs30[ind_global]) + \
            5.408e-4 * precip[ind_global] + \
            -0.2054  * dist_water[ind_global] + \
            -0.0333  * gw_depth[ind_global]
        # liquefaction susceptbility value, disregard pgv term
        liq_susc_val[ind_coastal] = \
            12.435 + \
            -2.615   * np.log(vs30[ind_coastal]) + \
            5.556e-4 * precip[ind_coastal] + \
            -0.0287  * np.sqrt(dist_coast[ind_coastal]) + \
            0.0666   * dist_river[ind_coastal] + \
            -0.0369  * dist_river[ind_coastal]*np.sqrt(dist_coast[ind_coastal])
        # catch nan values
        liq_susc_val[np.isnan(liq_susc_val)] = -99.
        # x-term for logistic model = liq susc val + pgv term
        x_logistic[ind_global] = liq_susc_val[ind_global] + 0.334*np.log(pgv_mag[ind_global])
        # x-term for logistic model = liq susc val + pgv term
        x_logistic[ind_coastal] = liq_susc_val[ind_coastal] + 0.301*np.log(pgv_mag[ind_coastal])

        # probability of liquefaction
        prob_liq = 1/(1+np.exp(-x_logistic)) # decimal
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log
        
        # for pgv_mag < 3 cm/s, set prob to "0"
        prob_liq[pgv_mag<3] = zero_prob_liq
        # for pga_mag < 0.1 g, set prob to "0"
        prob_liq[pga_mag<0.1] = zero_prob_liq
        # for vs30 > 620 m/s, set prob to "0"
        prob_liq[vs30>620] = zero_prob_liq

        # calculate sigma_mu
        sigma_mu = (np.exp(0.25)-1) * prob_liq

        # determine liquefaction susceptibility category
        liq_susc[liq_susc_val>-1.15] = 'very high'
        liq_susc[liq_susc_val<=-1.15] = 'high'
        liq_susc[liq_susc_val<=-1.95] = 'moderate'
        liq_susc[liq_susc_val<=-3.15] = 'low'
        liq_susc[liq_susc_val<=-3.20] = 'very low'
        liq_susc[liq_susc_val<=-38.1] = 'none'
        
        # prepare outputs
        output = {
            'prob_liq': {
                'mean': prob_liq,
                'sigma': None,
                # 'sigma_mu': np.ones(prob_liq.shape)*0.25,
                'sigma_mu': sigma_mu,
                'dist_type': 'normal',
                'unit': 'decimal'
            },
            # 'liq_susc': liq_susc,
            'liq_susc_val': liq_susc_val
        }
        # get intermediate values if requested
        if return_inter_params:
            # output['liq_susc_val'] = liq_susc_val
            output['x_logistic'] = x_logistic
            output['pgv_mag'] = pgv_mag
            output['pga_mag'] = pga_mag
            output['liq_susc'] = liq_susc
        
        # return
        return output


# -----------------------------------------------------------
class ZhuEtal2015(Liquefaction):
    """
    A map-based procedure to quantify liquefaction at a given location using logistic models by Zhu et al. (2015).

    **prob_liq** = f(**pga**, **mag**, **vs30**, **cti**)
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
    pgv: float, np.ndarray or list
        [cm/s] peak ground velocity, only to check threshold where prob_liq(pga<0.1g)=0
        
    Geotechnical/geologic:
    vs30: float, np.ndarray or list
        [m/s] time-averaged shear wave velocity in the upper 30-meters
    cti: float, np.ndarray or list
        compound topographic index, also known as wetness index
        
    Fixed:

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    
    References
    ----------
    .. [1] Zhu, J., Daley, D., Baise, L.G., Thompson, E.M., Wald, D.J., and Knudsen, K.L., 2017, A Geospatial Liquefaction Model for Rapid Response and Loss Estimation, Earthquake Spectra, vol. 31, no. 3, pp. 1813-1837.
    
    """

    _NAME = 'Zhu et al. (2015)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Zhu, J., Daley, D., Baise, L.G., Thompson, E.M., Wald, D.J., and Knudsen, K.L., 2017, ',
        'A Geospatial Liquefaction Model for Rapid Response and Loss Estimation, ',
        'Earthquake Spectra, ',
        'vol. 31, no. 3, pp. 1813-1837.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'prob_liq': {
                'desc': 'probability of liquefaction',
                'unit': 'decimal',
                # 'dist_type': 'lognormal',
            },
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': [
    #         'pga', 'pgv', 'mag'
    #     ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pga': {
                'desc': 'peak ground acceleration (g)',
                'unit': 'g',
            },
            'pgv': {
                'desc': 'peak ground velocity (cm/s)',
                'unit': 'cm/s',
            },
            'mag': {
                'desc': 'moment magnitude',
                'unit': '',
            },
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        "params": {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'vs30': 'time-averaged shear wave velocity in the upper 30-meters (m/s)',
            'cti': 'compound topographic index',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': ['vs30', 'cti'],
        'level3': ['vs30', 'cti'],
        # 'vs30', 'cti',
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': [],
        'level2': [],
        'level3': [],
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        pgv, pga, mag, # upstream PBEE RV
        vs30, cti, # geotechnical/geologic
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # initialize arrays
        x_logistic = np.empty(pga.shape)
        prob_liq = np.empty(pga.shape)
        
        # magnitude correction, from Baise & Rashidian (2020) and Allstadt et al. (2022)
        pgv_mag = pgv/(1+np.exp(-2*(mag-6)))
        pga_mag = pga/(10**2.24/mag**2.56)
        
        # x-term for logistic model = b0 + b1*var1 + ...
        x_logistic = 24.10 + 2.067*np.log(pga_mag) - 4.784*np.log(vs30) + 0.355*cti 

        # probability of liquefaction
        prob_liq = 1/(1+np.exp(-x_logistic)) # decimal
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log
        
        # for pgv_mag < 3 cm/s, set prob to "0"
        prob_liq[pgv_mag<3] = zero_prob_liq
        # for pga_mag < 0.1 g, set prob to "0"
        prob_liq[pga_mag<0.1] = zero_prob_liq
        # for vs30 > 620 m/s, set prob to "0"
        prob_liq[vs30>620] = zero_prob_liq
        
        # calculate sigma_mu
        sigma_mu = (np.exp(0.25)-1) * prob_liq

        # prepare outputs
        output = {
            'prob_liq': {
                'mean': prob_liq,
                'sigma': None,
                # 'sigma_mu': np.ones(prob_liq.shape)*0.25,
                'sigma_mu': sigma_mu,
                'dist_type': 'normal',
                'unit': 'decimal'
            }
        }
        # get intermediate values if requested
        if return_inter_params:
            output['x_logistic'] = x_logistic
            output['pgv_mag'] = pgv_mag
            output['pga_mag'] = pga_mag
        
        # return
        return output


# -----------------------------------------------------------
def HolzerEtal2011(**kwargs):
    """
    A simplified procedure to quantify probability of liquefaction using **PGA** and **MSF**.

    Parameters
    ----------


    Returns
    -------

    References
    ----------
    .. [1] Holzer, T.L., Noce, T.E., and Bennett, M.J., 2011, Liquefaction Probability Curves for Surficial Geologic Deposits, Environmental and Engineering Geoscience, vol. 17, no. 1, pp. 1-21. doi:10.2113/gseegeosci.17.1.1
    .. [2] Holzer, T.L., Noce, T.E., and Bennett, M.J., 2008, Liquefaction Hazard Maps for Three Earthquake Scenarios for the Communities of San Jose, Campbell, Cupertino, Los Altos, Los Gatos, Milpitas, Mountain View, Palo Alto, Santa Clara, Saratoga, and Sunnyvale, Northern Santa Clara County, California. Open-File Report, 2008, 1270 pp. doi:10.3133/ofr20081270
    .. [3] Youd, T.L., Idriss, I.M., Andrus, R.D., Arango, I., Castro, G., Christian, J.T., Dobry, R., Finn, W.D.L., Harder, L.F., Jr., Hynes, M.E., Ishihara, K., Koester, J.P., Liao, S.S.C., Marcuson, W.F., III, Martin, G.R., Mitchell, J.K., Moriwaki, Y., Power, M.S., Robertson, P.K., Seed, R.B., and Stokoe, K.H., II, 2001, Liquefaction resistance of soils: Summary report from the 1996 NCEER and 1998 NCEER/NSF workshops on evaluation of liquefaction resistance of soils, Journal of Geotechnical and Geoenvironmental Engineering, v. 127, no. 10, pp. 817-833. 

    """

    logging.info("Placeholder - not implemented")
    
    return None


# -----------------------------------------------------------
def YoudEtal2001(**kwargs):
    """
    An SPT-based procedure to quantify liquefaction (**crr** and **fs**) at a given location using the deterministic model by Youd et al. (2001). This function is to be used in conjuction with :func:`edp.corr_spt.youd_etal_2001_corr`.

    Parameters
    ----------
    N1_60_cs : float
        N-value corrected for energy, overburden, and fines content, see :func:`edp.corr_spt.youd_etal_2001_corr`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    Dr : float
        [%] relative density
    tau_stat : float
        [kPa] static shear stress
    patm : float, optional
        [kPa] atmospheric pressure, default = 101.3 kPa


    Returns
    -------
    crr : float
        equivalent cyclic resistance ratio
    csr : float
        cyclic stress ratio, corrected for magnitude/duration, overburden, and static bias
    fs : float
        factor of safety

    References
    ----------
    .. [1] Youd, T.L., et al., 2001, Liquefaction Resistance of Soils; Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops on Evaluation of Liquefaction Resistance of Soils, Journal of Geotechnical and Geoenvironmental Engineering, vol. 127, no. 10, pp. 817–833.

    """

    # Inputs
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    Dr = kwargs.get('Dr',None) # %, relative density
    patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 101.3 kPa
    tau_stat = kwargs.get('tau_stat',None) # kPa, static shear stress

    # magnitude scaling factor
    if M is None:
        msf = 1.0 # eq. 31
    else:
        msf = 10**2.24/M**2.56 # Idriss (1999) factors, recommended by Youd et al. (2001), eq. 24

    # overburden correction factor
    if Dr is None:
        K_sigma = 1.0 # eq. 31
    else:
        f = min(0.8,max(0.6,(0.8-0.6)/(80-40)*(Dr-40)+0.6)) # factor for K_sigma, Figure 15
        K_sigma = (sv0_eff/patm)**(f-1) # eq. 31

    # sloping ground correction factor
    if tau_stat is None:
        K_alpha = 1.0 # eq. 31
    else:
        K_alpha = tau_stat/sv0_eff # eq. 32

    if N1_60_cs is None:
        crr = None
    else:
        # SPT base CRR curve (cyclic resistance ratio)
        crr = 1/(34-min(N1_60_cs,30)) + min(N1_60_cs,30)/135 + 50/(10*min(N1_60_cs,30) + 45)**2 - 1/200 # eq. 4

    # see if enough inputs are given for csr
    if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
        csr = None
        print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
    else:
        # Cyclic stress ratio (demand)
        csr_m_sigv = 0.65 * amax * sv0_tot/sv0_eff * rd # Seed and Idriss (1971)
        csr = csr_m_sigv/msf/K_sigma/K_alpha # CSR for M=7.5 and corrected to 1 atm and static bias

    # determine if FS can be calculated
    if crr is None or csr is None:
        fs = None
    else:
        # Factor of safety
        fs = crr/csr

    #
    return crr, csr, fs


# -----------------------------------------------------------
def CetinEtal2004(**kwargs):
    """
    An SPT-Based procedure to quantify liquefaction at a given location using probabilistic model by Cetin et al. (2004). This function produces **p_liq** or equivalent **crr** if **p_liq** is specified. This function is to be used in conjuction with :func:`edp.corr_spt.cetin_etal_2004_corr`.

    Parameters
    ----------
    N1_60 : float
        N-value corrected for energy and overburden, see :func:`edp.corr_spt.cetin_etal_2004_corr`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    fc : float
        [%] relative density
    patm : float, optional
        [kPa] atmospheric pressure, default = 101.3 kPa
    p_liq : float, optional
        [%] probability of liquefaction, if looking to get **crr**


    Returns
    -------
    crr : float
        equivalent cyclic resistance ratio if p_liq is specified, otherwise NaN
    csr : float
        cyclic stress ratio, corrected for magnitude/duration and overburden
    p_liq : float
        [%] probability of liquefaction

    References
    ----------
    .. [1] Cetin, K.O., Seed, R.B., Der Kiureghian, A., Tokimatsu, K., Harder Jr, L.F., Kayen, R.E., and Moss, R.E., 2004, Standard Penetration Test-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential, Journal of Geotechnical and Geoenvironmental Engineering, vol. 130, no. 12, pp. 1314-1340.

    """

    # Inputs
    N1_60 = kwargs.get('N1_60',None) # corrected SPT blow count
    fc = kwargs.get('fc',0.0) # percent, fines content
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    p_liq = kwargs.get('p_liq',None) # %, probability of liquefaction, if want to determine equivalent CRR

    # flag for cases 1 and 2: measurement/estimation errors included/removed, default to remove (Cetin et al., 2004)
    flag_include_error = kwargs.get('flag_include_error',False)

    # model parameters with measurement/estimation errors included/removed (Kramer and Mayfield, 2007)
    if flag_include_error is True:
        t1 = 0.004
        t2 = 13.79
        t3 = 29.06
        t4 = 3.82
        t5 = 0.06
        t6 = 15.25
        sigma = 4.21
    else:
        t1 = 0.004
        t2 = 13.32
        t3 = 29.53
        t4 = 3.70
        t5 = 0.05
        t6 = 16.85
        sigma = 2.70

    # overburden correction factor, K_sigma
    # data from Figure 12, form provided in Idriss and Boulanger (2012, Spectra)
    K_sigma = (sv0_eff/patm)**-0.278 #

    # duration weighting factor (or magnitude scaling factor)
    # data from Figure 10, form provided in Idriss and Boulanger (2012, Spectra)
    dwf = (M/7.5)**-2.217

    # check what inputs are given
    if p_liq is None:

        # cyclic stress ratio
        csr_eq = 0.65*amax*sv0_tot/sv0_eff*rd # eq. 10, uncorrected for magnitude and duration
        print('csr_eq = ', str(csr_eq))
        csr = csr_eq/dwf/K_sigma # correction for duration and to 1 atm

        # probability of liquefaction (Cetin et al., 2004)
        p_liq = norm.cdf(-(N1_60*(1+t1*fc) - t2*np.log(csr_eq) - t3*np.log(M) - t4*np.log(sv0_eff/patm) + t5*fc + t6)/sigma) * 100 # eq. 19

        # cyclic resistance ratio
        crr = np.nan # not used

    else:

        # cyclic stress ratio
        csr = None # not used

        # inverse analysis of CRR given pLiq, (Cetin et al., 2004)
        crr = np.exp((N1_60*(1+t1*fc) - t3*np.log(M) - t4*np.log(sv0_eff/patm) + t5*fc + t6 + sigma*norm.ppf(p_liq/100))/t2) # eq. 20

    #
    return crr, csr, p_liq


# -----------------------------------------------------------
def BoulangerIdriss2014(**kwargs):
    """
    A site-specific procedure to quantify liquefaction at a given location using the deterministic and probabilistic procedures by Idriss and Boulanger (2008) and Boulanger and Idriss (2012; 2014; 2016). This function works with both SPT or CPT inputs, and should be used in conjuction with either :func:`edp.corr_spt.boulanger_idriss_2014_corr_spt` or :func:`edp.corr_cpt.boulanger_idriss_2014_corr_cpt`. A flag is required for deciding whether to follow the deterministic or the probabilistic approach. For the probabilistic approach, this funtion produces **p_liq** or equivalent **crr** if **p_liq** is specified

    Parameters
    ----------
    flag_det_prob : str
        procedure to run: **deterministic** or **probabilistic**
    pen_type : str
        type of penetrometer: specify **spt** or **cpt**
    resistance : float
        [varies] corrected penetration resistance, **N1_60_cs** or **qc_1N_cs**, see :func:`edp.corr_spt.boulanger_idriss_2014_corr_spt` or :func:`edp.corr_cpt.boulanger_idriss_2014_corr_cpt`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    tau_stat : float
        [kPa] static shear stress
    patm : float, optional
        [kPa] atmospheric pressure, default = 101.3 kPa
    Q : float, optional
        fitting parameter for the critical state line used to calculate the relative state parameter; default = 10
    K0 : float, optional
        coefficient of lateral earth pressure; default = 0.5
    p_liq : float, optional
        [%] probability of liquefaction, if looking to get CRR

    Returns
    -------
    crr : float
        [%] equivalent cyclic resistance ratio if p_liq is specified, otherwise NaN
    csr : float
        [%] cyclic stress ratio, corrected for magnitude/duration and overburden
    fs : float, optional
        factor of safety, depends on **flag_det_prob**
    p_liq : float, optional
        [%] probability of liquefaction, depends on **flag_det_prob**

    References
    ----------
    .. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.
    .. [2] Boulanger, R.W., and Idriss, I.M., 2014, CPT and SPT Based Liquefaction Triggering procedures, Report UCD/CGM-14/01, Department of Civil and Environmental Engineering, University of California, Davis, CA, 134 pp.
    .. [3] Boulanger, R.W., and Idriss, I.M., 2012, Probabilistic Standard Penetration Test–Based Liquefaction–Triggering Procedure, Journal of Geotechnical and Geoenvironmental Engineering, vol. 138, no. 10, pp. 1185-1195.
    .. [4] Boulanger, R.W., and Idriss, I.M., 2016, CPT-Based Liquefaction Triggering Procedure, Journal of Geotechnical and Geoenvironmental Engineering, vol. 142, no. 2, 04015065.

    """

    # # flag for deterministic or probabilistic approach (FS vs prob for result)
    flag_det_prob = kwargs.get('flag_det_prob',None)
    if flag_det_prob is None:
        print('Must specify whether to run "deterministic" or "probabilistic" procedure; exiting function')
        return None, None, None

    else:
        # pen_type:
        # - spt (resistance = N1_60_cs)
        # - cpt (resistance = qc_1N_cs)
        pen_type = kwargs.get('pen_type',None)
        resistance = kwargs.get('resistance',None)
        # N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
        # qc_1N_cs = kwargs.get('qc_1N_cs',None) # corrected CPT tip measurement

        # Inputs
        amax = kwargs.get('amax',None) # g, peak surface acceleration
        sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
        sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
        rd = kwargs.get('rd',None) # stress reduction factor with depth
        M = kwargs.get('M',None) # moment magnitude
        patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
        tau_stat = kwargs.get('tau_stat',None) # kPa, static shear stress
        Q = kwargs.get('Q',10) # fitting parameter for critical state line for calculating the relative state parameter
        K0 = kwargs.get('K0',0.5) # coefficient of lateral earth pressure
        p_liq = kwargs.get('p_liq',None) # %, probability of liquefaction, if want to determine equivalent CRR=

        # magnitude scaling factor
        if M is None:
            msf = 1.0 # eq. 31
        else:
        #     msf = min(1.8,6.9*np.exp(-M/4)-0.058) # Eq. 2.17, Idriss and Boulanger (2008)
            if 's' in pen_type.lower():
                msf_c1 = 31.5
                msf_c2 = 2

            elif 'c' in pen_type.lower():
                msf_c1 = 180
                msf_c2 = 3

            msf_max = min(2.2,1.09+(resistance/msf_c1)**msf_c2) # eq. 2.20 and 2.21
            msf = 1 + (msf_max-1) * (8.64*np.exp(-M/4)-1.325) # eq. 2.19, Boulanger and Idriss (2014)

        # overburden correction factor
        if 's' in pen_type.lower():
            C_sigma_c1 = 18.9
            C_sigma_c2 = 2.55
            C_sigma_c3 = 0.5
            if resistance > 37:
                print('N1_60_cs > 37, over the limit described in Boulanger and Idriss (2014)')

        elif 'c' in pen_type.lower():
            C_sigma_c1 = 37.3
            C_sigma_c2 = 8.27
            C_sigma_c3 = 0.264
            if resistance > 211:
                print('qc_1N_cs > 211, over the limit described in Boulanger and Idriss (2014)')

        C_sigma = min(0.3,1/(C_sigma_c1-C_sigma_c2*resistance**C_sigma_c3)) # eq. 2.16b and 2.16c
        K_sigma = min(1.1,1 - C_sigma*np.log(sv0_eff/patm)) # eq. 2.16a

        # static shear stress correction factor
        if tau_stat is None:
            K_alpha = 1.0
        else:
            if 's' in pen_type.lower():
                xi_r_c1 = (resistance/46)**0.5 # eq. 64

            elif 'c' in pen_type.lower():
                xi_r_c1 = 0.478*resistance**0.264 - 1.063 # eq. 65
                if resistance < 21:
                    print('qc_1N_cs < 21, below the limit recommended in Idriss and Boulanger (2008)')

            xi_r = 1/(Q-np.log(100*(1+2*K0)/3*sv0_eff/patm)) - xi_r_c1 # eq. 64 and 65
            xi_r = min(0.1,max(-0.6,xi_r)) # eq. 67
            alpha = min(0.35, tau_stat/sv0_eff) # eq. 63, 66
            a = 1267 + 636*alpha**2 - 634*np.exp(alpha) - 632*np.exp(-alpha) # eq. 60
            b = np.exp(-1.11 + 12.3*alpha**2 + 1.31*np.log(alpha + 0.0001)) # eq. 61
            c = 0.138 + 0.126*alpha + 2.52*alpha**3 # eq. 62
            K_alpha = a + b*np.exp(-xi_r/c) # eq. 59

        # see if enough inputs are given for csr
        if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
            csr = None
            print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
        else:
            # cyclic stress ratio (demand)
            csr_m_sigv = 0.65 * amax * sv0_tot/sv0_eff *rd # Seed and Idriss (1971), uncorrected
            csr = csr_m_sigv/msf/K_sigma/K_alpha # CSR for M=7.5 and corrected to 1 atm and static bias

        # see if enough inputs are given for crr or p_liq
        if resistance is None:
            print('N1_60_cs or qc_1N_cs is not provided as an input, exiting procedure')
            return None

        else:
            # overburden correction factor
            if 's' in pen_type.lower():
                crr_c1 = 14.1
                crr_c2 = 126
                crr_c3 = 23.6
                crr_c4 = 25.4

            elif 'c' in pen_type.lower():
                crr_c1 = 113
                crr_c2 = 1000
                crr_c3 = 140
                crr_c4 = 137

            # if deterministic, calculate factor of safety
            if flag_det_prob[0] == 'd':
                Co = 2.8 # value for deterministic analysis, pp. 1197 in BI12

                # SPT base CRR curve (cyclic resistance ratio)
                crr = np.exp(resistance/crr_c1 + (resistance/crr_c2)**2 - \
                            (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - Co) # eq. 2.24 and 2.25 in BI14

                if crr is None or csr is None:
                    fs = None
                else:
                    # Factor of safety
                    fs = crr/csr

                #
                return crr, csr, fs

            # if probabilistic, calculate p_liq or equivalent crr
            if flag_det_prob[0] == 'p':

                # check what inputs are given
                if p_liq is None:
                    Co = 2.67 # recommended on pp. 1192 in BI12
                    sigma_R = 0.13 # recommended on pp. 1192 in BI12

                    # Probability of liquefaction
                    p_liq = norm.cdf(-(resistance/crr_c1 + (resistance/crr_c2)**2 - \
                                    (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - \
                                    Co - np.log(csr))/sigma_R) * 100 # eq. 31 in BI12 and 36 in BI16

                    # Cyclic resistance ratio
                    crr = np.nan # not used

                else:
                    # Inverse analysis of CRR given pLiq, (Cetin et al., 2004)
                    crr = np.exp(resistance/crr_c1 + (resistance/crr_c2)**2 - \
                                (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - \
                                Co + sigma_R*norm.ppf(p_liq/100)) # eq. 30 in BI12 and 34 in BI16

                #
                return crr, csr, p_liq


# -----------------------------------------------------------
def MossEtal2006(**kwargs):
    """
    A CPT-based procedure to quantify liquefaction at a given location using probabilistic model by Moss et al. (2006). This function produces **p_liq** or equivalent **crr** if **p_liq** is specified. This function is to be used in conjuction with :func:`edp.corr_cpt.moss_etal_2006_corr`.

    Parameters
    ----------
    qc_1 : float
        normalized tip resistance corrected for overburden, see :func:`edp.corr_cpt.moss_etal_2006_corr`
    Rf : float
        [%] friction ratio, see :func:`edp.corr_cpt.moss_etal_2006_corr`
    c : float
        overburden normalization exponent, see :func:`edp.corr_cpt.moss_etal_2006_corr`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    p_liq : float, optional
        [%] probability of liquefaction, if looking to get **crr**

    Returns
    -------
    crr : float
        equivalent cyclic resistance ratio if p_liq is specified, otherwise NaN
    csr : float
        cyclic stress ratio, corrected for magnitude/duration
    p_liq : float
        [%] probability of liquefaction

    References
    ----------
    .. [1] Moss, R.E.S., Seed, R.B., Kayen, R.E., Stewart, J.P., Der Kiureghian, A., and Cetin, K.O., 2006a, CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential, Journal of Geotechnical and Geoenvironmental Engineering, vol. 132, no. 8, pp. 1032-1051.

    """

    # Inputs
    qc_1 = kwargs.get('qc_1',None) # qc corrected for overburden
    Rf = kwargs.get('Rf',None) # %, friction ratio = fs/qc
    c = kwargs.get('Rf',None) # %, overburden normalization exponent
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    # patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    p_liq = kwargs.get('p_liq',None) # %, probability of liquefaction, if want to determine equivalent CRR

    # check if enough base inputs are given
    if qc_1 is None or Rf is None or c is None or M is None or sv0_eff is None:

        #
        csr = None
        crr = None
        p_liq = None
        print('Not enough inputs to proceed with procedure')

    #
    else:

        # Duration weighting factor (or magnitude scaling factor)
        dwf = 17.84*M**-1.43
        if M < 5.5 or M > 8.5:
            print('Magnitude is outside the valid range of 5.5-8.5')

        # Cyclic stress ratio
        csr_eq = 0.65*amax*sv0_tot/sv0_eff*rd # eq. 1, uncorrected for magnitude and duration
        csr = csr_eq/dwf # correction for duration, eq. 6

        # Increase in qc_1 and friction ratio due to frictional effects of apparent "fines" and character
        if Rf <= 0.5:
            d_qc = 0
        #
        else:
            x1 = 0.38*(min(Rf,5)) - 0.19
            x2 = 1.46*(min(Rf,5)) - 0.73
            d_qc = x1*np.log(csr) + x2

        # apply correction
        qc_1_mod = qc_1 + d_qc

        # Model parameters
        t1 = 1.045
        t2 = 0.110
        t3 = 0.001
        t4 = 0.850
        t5 = 7.177
        t6 = 0.848
        t7 = 0.002
        t8 = 20.923
        sigma = 1.632

        # check what inputs are given
        if p_liq is None:

            # Probability of liquefaction
            p_liq = norm.cdf(-(qc_1_mod**t1 + qc_1_mod*(t2*Rf) + t3*Rf + \
                            c*(1+t4*Rf) - t5*np.log(csr) - t6*np.log(M) - \
                            t7*np.log(sv0_eff) - t8)/sigma) * 100 # eq. 20

            # Cyclic resistance ratio
            crr = np.nan # not used

        #
        else:

            # Inverse analysis of CRR given pLiq
            crr = np.exp((qc_1_mod**t1 + qc_1_mod*(t2*Rf)  + t3*Rf + \
                        c*(1+t4*Rf) - t6*np.log(M) - t6*np.log(M) - \
                        t7 + sigma*norm.ppf(p_liq/100))/t5) # eq. 21

    #
    return crr, csr, p_liq



# -----------------------------------------------------------
# class Robertson2009(Liquefaction):
#     """
#     Compute factor of safety against liquefaction for CPTs using Robertson (2009).
    
#     Parameters
#     ----------
#     From upstream PBEE:
#     pga: float, np.ndarray or list
#         [g] peak ground acceleration
#     mag: float, np.ndarray or list
#         moment magnitude
        
#     Geotechnical/geologic:
#     gw_depth: float, np.ndarray or list
#         [m] groundwater table depth
        
#     Fixed:
#     qc: float, np.ndarray or list
#         [MPa] measured tip resistance
#     fs: float, np.ndarray or list
#         [MPa] measured sleeve resistance
#     z: float, np.ndarray or list
#         [m] depth of measurements
#     tot_sig_v0: float, np.ndarray or list
#         [kPa] initial vertical total stress
#     eff_sig_v0: float, np.ndarray or list
#         [kPa] initial vertical effective stress

#     Returns
#     -------
#     fs_liq : float, np.ndarray
#         [] factor of safety against liquefaction
#     qc1ncs : float, np.ndarray
#         [] corrected tip resistance
    
#     References
#     ----------
#     .. [1] Reference...
    
#     """

#     _NAME = 'Robertson (2009)'       # Name of the model
#     _ABBREV = None                     # Abbreviated name of the model
#     _REF = "".join([                     # Reference for the model
#         'Robertson, P.K., 2009, ',
#         'TBD',
#         'TBD,
#         'TBD',
#     ])
#     _RETURN_PBEE_DIST = {                            # Distribution information
#         'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
#         "desc": 'returned PBEE upstream random variables:',
#         'params': {
#             'fs_liq': {
#                 'desc': 'factor of safety against liquefactino',
#                 'unit': '',
#             },
#             'qc1ncs': {
#                 'desc': 'corrected tip resistance',
#                 'unit': '',
#             }
#         }
#     }
#     _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
#         'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
#         "desc": 'PBEE upstream random variables:',
#         'params': {
#             'pga': {
#                 'desc': 'peak ground acceleration (g)',
#                 'unit': 'g',
#             },
#             'mag': {
#                 'desc': 'moment magnitude',
#                 'unit': '',
#             },
#         }
#     }
#     _INPUT_DIST_VARY_WITH_LEVEL = True
#     _N_LEVEL = 3
#     _MODEL_INPUT_INFRA = {
#         "desc": 'Infrastructure random variables:',
#         "params": {}
#     }
#     _MODEL_INPUT_GEO = {
#         "desc": 'Geotechnical/geologic random variables:',
#         'params': {
#             'gw_depth': 'groundwater table depth (m)',
#         }
#     }
#     _MODEL_INPUT_FIXED = {
#         'desc': 'Fixed input variables:',
#         'params': {
#             'qc': 'measured tip resistance (MPa)',
#             'fs': 'measured sleeve resistance (MPa)',
#             'z': 'depth of measurements (m)',
#             'tot_sig_v0': 'initial vertical total stress (kPa)',
#             'eff_sig_v0': 'initial vertical effective stress (kPa)',
#         }
#     }
#     _REQ_MODEL_RV_FOR_LEVEL = {
#         'level1': ['gw_depth'],
#         'level2': ['gw_depth'],
#         'level3': ['gw_depth'],
#     }
#     _REQ_MODEL_FIXED_FOR_LEVEL = {
#         # 'liq_susc'
#     }
#     _REQ_PARAMS_VARY_WITH_CONDITIONS = False
#     _MODEL_FORM_DETAIL = {}
#     _MODEL_INPUT_RV = {}
    
    
#     @staticmethod
#     # @njit
#     def _model(
#         pga, mag, # upstream PBEE RV
#         gw_depth, # geotechnical/geologic
#         qc, fs, z, tot_sig_v0, eff_sig_v0, # fixed/toggles
#         return_inter_params=False # to get intermediate params
#     ):
#         """Model"""
        
#         # run ZhuEtal2017 to get liquefaction susceptibility
#         zhu_return = ZhuEtal2017._model(
#             pgv=np.ones(pga.shape), # not used for liquefaction susceptibility category,
#             pga=pga, mag=mag, # upstream PBEE RV
#             vs30=vs30, precip=precip, dist_coast=dist_coast, # geotechnical/geologic
#             dist_river=dist_river, dist_water=dist_water, gw_depth=gw_depth, # geotechnical/geologic
#             return_inter_params=True
#         )
#         # get liquefaction susceptility category and value
#         liq_susc_val = zhu_return['liq_susc_val']
#         liq_susc = zhu_return['liq_susc']
#         pga_mag = zhu_return['pga_mag']
            
#         # initialize arrays
#         prob_liq_pga = np.zeros(pga.shape)
#         p_ml = np.zeros(pga.shape)
        
#         # correction factor for moment magnitudes other than M=7.5, eq. 4-21
#         k_mag = 0.0027*mag**3 - 0.0267*mag**2 - 0.2055*mag + 2.9188
#         # correction for groudnwater depths other than 5 feet, eq. 4-22
#         k_gw_depth = 0.022 * gw_depth*3.28 + 0.93
        
#         # get uncorrected p_liq given pga
#         prob_liq_pga[liq_susc=='very high'] = np.maximum(np.minimum(9.09*pga[liq_susc=='very high']-0.82,1),0)
#         prob_liq_pga[liq_susc=='high'] = np.maximum(np.minimum(7.67*pga[liq_susc=='high']-0.92,1),0)
#         prob_liq_pga[liq_susc=='moderate'] = np.maximum(np.minimum(6.67*pga[liq_susc=='moderate']-1.00,1),0)
#         prob_liq_pga[liq_susc=='low'] = np.maximum(np.minimum(5.57*pga[liq_susc=='low']-1.18,1),0)
#         prob_liq_pga[liq_susc=='very low'] = np.maximum(np.minimum(4.16*pga[liq_susc=='very low']-1.08,1),0)
#         prob_liq_pga[liq_susc=='none'] = 0
        
#         # get portion of map unit susceptible to liquefaction
#         p_ml[liq_susc=='very high'] = 0.25
#         p_ml[liq_susc=='high'] = 0.20
#         p_ml[liq_susc=='moderate'] = 0.10
#         p_ml[liq_susc=='low'] = 0.05
#         p_ml[liq_susc=='very low'] = 0.02
#         p_ml[liq_susc=='none'] = 0.00
        
#         # liquefaction likelihood, p_liq
#         prob_liq = prob_liq_pga / k_mag / k_gw_depth * p_ml * 100 # %, eq. 4-20
#         prob_liq = np.maximum(prob_liq,1e-3) # set prob to >= 1e-3 to avoid 0% in log
        
#         # Zhu et al. (2017) boundary constraints
#         # for pga_mag < 0.1 g, set prob to 0
#         prob_liq[pga_mag<0.1] = 1e-3
#         # for vs30 > 620 m/s, set prob to 0
#         prob_liq[vs30>620] = 1e-3
#         # for precip > 1700 mm, set prob to 0
#         prob_liq[precip>1700] = 1e-3

#         # calculate sigma_mu
#         sigma_mu = (np.exp(0.25)-1) * prob_liq
        
#         # prepare outputs
#         output = {
#             # 'prob_liq': prob_liq,
#             'fs_liq': {
#                 'mean': prob_liq,
#                 'sigma': None,
#                 'sigma_mu': sigma_mu,
#                 'dist_type': 'normal',
#                 'unit': '%'
#             },
#             'liq_susc_val': liq_susc_val
#         }
#         # get intermediate values if requested
#         if return_inter_params:
#             output['pga_mag'] = pga_mag
#             output['liq_susc'] = liq_susc
#             output['k_mag'] = k_mag
#             output['k_gw_depth'] = k_gw_depth
#             output['prob_liq_pga'] = prob_liq_pga
#             output['p_ml'] = p_ml
        
#         # return
#         return output
        