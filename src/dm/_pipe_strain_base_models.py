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
# import logging

# data manipulation modules
import numpy as np
# from numpy import tan, radians, where
# from numba import jit
# from numba import njit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class _PipeStrainBase(BaseModel):
    "Inherited class specfic to pipe strain"

    def __init__(self):
        super().__init__()
    
    
# -----------------------------------------------------------
class BainEtal2022(_PipeStrainBase):
    """
    Model for lateral-spread-induced pipe strain using Bain et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - lateral-spread-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
        
    Geotechnical/geologic:
    Clay:
    def_length: float, np.ndarray or list
        [m] length of ground deformation zone
    s_u_backfill: float, np.ndarray or list
        [kPa] undrained shear strength
    alpha_backfill: float, np.ndarray or list
        adhesion factor
    
    Sand:
    def_length: float, np.ndarray or list
        [m] length of ground deformation zon
    gamma_backfill: float, np.ndarray or list
        [kN/m^3] total unit weight of backfill soil
    h_pipe: float, np.ndarray or list
        [m] burial depth to pipe centerline
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
    delta_backfill: float, np.ndarray or list
        sand-pipe interface friction angle ratio
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Bain, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Bain et al. (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Bain, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation - lateral-spread-induced (m)',
                'unit': 'm',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': 'pipe diameter (mm)',
            't_pipe': 'pipe wall thickness (mm)',
            'sigma_y': 'pipe yield stress (kPa)',
            'n_param': 'Ramberg-Osgood parameter',
            'r_param': 'Ramberg-Osgood parameter',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'def_length': 'length of ground deformation zone (m) - for clay and sand',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'h_pipe': 'burial depth to pipe centerline (m) - for sand',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': 'soil type (sand/clay) for model',
            'steel_grade': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'clay': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'def_length', 'alpha_backfill', 's_u_backfill'],
        },
        'sand': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'def_length', 'h_pipe', 'gamma_backfill', 'phi_backfill', 'delta_backfill'],
        }
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'clay': {
            'level1': ['soil_type'],
            'level2': ['soil_type'],
            'level3': ['soil_type', 'steel_grade'],
        },
        'sand': {
            'level1': ['soil_type'],
            'level2': ['soil_type'],
            'level3': ['soil_type', 'steel_grade'],
        }
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """uses soil_type to determine what model parameters to use"""
        soil_type = kwargs.get('soil_type', None)
        if soil_type is None:
            soil_type = ['clay','sand']
        req_rvs_by_level = {}
        req_fixed_by_level = {}
        soils = []
        if len(soil_type) == 0:
            soils = ['clay'] # if soil_type is empty, just use clay as default
        else:
            if 'sand' in soil_type:
                soils.append('sand')
            if 'clay' in soil_type:
                soils.append('clay')
        for i in range(3):
            for each in soils:
                if f'level{i+1}' in req_rvs_by_level:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
                else:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
            req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
            req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))
        return req_rvs_by_level, req_fixed_by_level


    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, sigma_y, n_param, r_param, # infrastructure
        def_length, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        h_pipe, gamma_backfill, phi_backfill, delta_backfill, # sand
        soil_type, steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # initialize arrays
        t_u = np.empty_like(pgdef)
        l_e = np.empty_like(pgdef)
        
        # model coefficients
        # sand
        c0_s =  0.188     # constant
        c1_s =  0.853     # ln(t_pipe)
        c2_s =  0.018     # ln(d_pipe)
        c3_s =  0.751     # ln(sigma_y)
        c4_s = -0.862     # ln(h_pipe)
        c5_s = -0.863     # ln(gamma_backfill)
        c6_s = -1.005     # ln(phi_backfill)
        c7_s = -1.000     # ln(delta_backfill)
        c8_s =  0.136     # ln(pgdef)
        # clay
        c0_c = -4.019     # constant
        c1_c =  0.876     # ln(t_pipe)
        c2_c =  0.787     # ln(sigma_y)
        c3_c = -0.886     # ln(s_u_backfill)
        c4_c = -0.889     # ln(alpha_backfill)
        c5_c =  0.114     # ln(pgdef)

        # setup
        # --------------------
        young_mod = 2e8 # kpa, hard-coded now
        # --------------------
        phi_rad = np.radians(phi_backfill)
        pi = np.pi
        d_in = d_pipe - 2*t_pipe # mm
        circum = pi * d_pipe/1000 # m
        area = pi * ((d_pipe/1000)**2 - (d_in/1000)**2) / 4 # m^2
        
        # find where sigma_y, n_param, and r_param are nan
        sigma_y_isnan = np.isnan(sigma_y)
        n_param_isnan = np.isnan(n_param)
        r_param_isnan = np.isnan(r_param)
        
        # if steel-grade is provided, use properties informed by steel grade
        # for steel_grade = "NA", set to "x-52"
        steel_grade[steel_grade=='NA'] = 'X-52'
        # Grade-B
        grade = 'Grade-B'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 241*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 8
        # Grade X-42
        grade = 'X-42'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 290*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 9
        # Grade X-52
        grade = 'X-52'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 359*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 10
        # Grade X-60
        grade = 'X-60'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 414*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 12
        # Grade X-70
        grade = 'X-70'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 483*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 14
        r_param[np.logical_and(cond,r_param_isnan)] = 15
        # Grade X-80
        grade = 'X-80'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 552*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 15
        r_param[np.logical_and(cond,r_param_isnan)] = 20
        # if any of the params are still missing, use default grade of X-52
        sigma_y[sigma_y_isnan] = 359*1000 # kPa
        n_param[n_param_isnan] = 8
        r_param[r_param_isnan] = 10
        
        # calculations
        # find indices with sand and clay
        ind_sand = soil_type=='sand'
        ind_clay = soil_type=='clay'
        # for sand
        if True in ind_sand:
            t_u[ind_sand] = gamma_backfill[ind_sand] * h_pipe[ind_sand] \
                            * np.tan(phi_rad[ind_sand]*delta_backfill[ind_sand]) \
                            * circum[ind_sand]
            l_e[ind_sand] = np.exp(
                c0_s    +   c1_s*np.log(t_pipe[ind_sand])           +   c2_s*np.log(d_pipe[ind_sand])
                        +   c3_s*np.log(sigma_y[ind_sand])          +   c4_s*np.log(h_pipe[ind_sand])
                        +   c5_s*np.log(gamma_backfill[ind_sand])   +   c6_s*np.log(phi_backfill[ind_sand])
                        +   c7_s*np.log(delta_backfill[ind_sand])   +   c8_s*np.log(pgdef[ind_sand])
            )
        # for clay
        if True in ind_clay:
            t_u[ind_clay] = alpha_backfill[ind_clay] * s_u_backfill[ind_clay] * circum[ind_clay]
            l_e[ind_clay] = np.exp(
                c0_c    +   c1_c*np.log(t_pipe[ind_clay])       +   c2_c*np.log(sigma_y[ind_clay]) \
                        +   c3_c*np.log(s_u_backfill[ind_clay]) +   c4_c*np.log(alpha_backfill[ind_clay]) \
                        +   c5_c*np.log(pgdef[ind_clay])
            )
        
        # other calcs
        l_to_use = np.minimum(def_length/2, l_e)
        beta_p = t_u/area
        eps_pipe = beta_p*l_to_use/young_mod * (1 + n_param/(1+r_param)*(beta_p*l_to_use/sigma_y)**r_param)
        eps_pipe = np.maximum(np.minimum(eps_pipe*100, 100), 1e-5) # convert to % and apply limit

        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': np.ones(eps_pipe.shape)*0.45,
                'sigma_mu': np.ones(eps_pipe.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['t_u'] = t_u
            output['l_e'] = l_e
            output['l_to_use'] = l_to_use
            output['beta_p'] = beta_p
        
        # return
        return output


# -----------------------------------------------------------
class HutabaratEtal2022_Normal(_PipeStrainBase):
    """
    Model for normal-slip-induced pipe strain using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - normal-slip-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    psi_dip: float, np.ndarray or list
        [deg] pipe-fault dip angle
    h_pipe: float, np.ndarray or list
        [m] burial depth to pipe centerline
    
    Clay:
    s_u_backfill: float, np.ndarray or list
        [kPa] undrained shear strength
    alpha_backfill: float, np.ndarray or list
        adhesion factor
    
    Sand:
    gamma_backfill: float, np.ndarray or list
        [kN/m^3] total unit weight of backfill soil, inferred 
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    soil_density: str, np.ndarray or list
        soil density: medium dense, dense, or very dense for sand
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Hutabarat et al. (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Hutabarat, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation - normal-slip-induced (m)',
                'unit': 'm',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': 'pipe diameter (mm)',
            't_pipe': 'pipe wall thickness (mm)',
            'sigma_y': 'pipe yield stress (kPa)',
            'n_param': 'Ramberg-Osgood parameter',
            'r_param': 'Ramberg-Osgood parameter',
            'l_anchor': 'pipeline anchored length (m)',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'psi_dip': 'pipe-fault dip angle [deg]',
            'h_pipe': 'burial depth to pipe centerline (m)',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': 'soil type (sand/clay) for model',
            'soil_density': 'soil density: medium dense, dense, or very dense for sand',
            'steel_grade': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'clay': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'h_pipe', 'alpha_backfill', 's_u_backfill'],
        },
        'sand': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'h_pipe', 'gamma_backfill', 'phi_backfill', 'delta_backfill'],
        }
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'clay': {
            'level1': ['soil_type', 'soil_density'],
            'level2': ['soil_type', 'soil_density'],
            'level3': ['soil_type', 'soil_density'],
        },
        'sand': {
            'level1': ['soil_type', 'soil_density'],
            'level2': ['soil_type', 'soil_density'],
            'level3': ['soil_type', 'soil_density'],
        }
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """uses soil_type to determine what model parameters to use"""
        soil_type = kwargs.get('soil_type', None)
        if soil_type is None:
            soil_type = ['clay','sand']
        req_rvs_by_level = {}
        req_fixed_by_level = {}
        soils = []
        if len(soil_type) == 0:
            soils = ['clay'] # if soil_type is empty, just use clay as default
        else:
            if 'sand' in soil_type:
                soils.append('sand')
            if 'clay' in soil_type:
                soils.append('clay')
        for i in range(3):
            for each in soils:
                if f'level{i+1}' in req_rvs_by_level:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
                else:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
            req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
            req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))
        return req_rvs_by_level, req_fixed_by_level


    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, sigma_y, n_param, r_param, l_anchor, # infrastructure
        psi_dip, h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        soil_type, soil_density, steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # initialize arrays
        # eps_ult = np.empty_like(pgdef)
        t_u = np.empty_like(pgdef)
        n_q_vd = np.zeros(pgdef.shape)
        n_gamma_d = np.zeros(pgdef.shape)
        q_vd = np.zeros(pgdef.shape)
        
        # for delta_u
        a0_low = np.empty_like(pgdef)
        a1_low = np.empty_like(pgdef)
        a2_low = np.empty_like(pgdef)
        a3_low = np.empty_like(pgdef)
        a4_low = np.empty_like(pgdef)
        a5_low = np.empty_like(pgdef)
        a6_low = np.empty_like(pgdef)
        # for eps_pipe
        b0_low =       np.empty_like(pgdef)
        b2_low =       np.empty_like(pgdef)
        b3_low =       np.empty_like(pgdef)
        b4_low =       np.empty_like(pgdef)
        b5_low =       np.empty_like(pgdef)
        d2_low =       np.empty_like(pgdef)
        d3_low =       np.empty_like(pgdef)
        d41_low =      np.empty_like(pgdef)
        d42_low =      np.empty_like(pgdef)
        d5_low =       np.empty_like(pgdef)
        d6_low =       np.empty_like(pgdef)
        d71_low =      np.empty_like(pgdef)
        d72_low =      np.empty_like(pgdef)
        d8_low =       np.empty_like(pgdef)
        sigma_ln_low = np.empty_like(pgdef)
        # for delta_u
        a0_high = np.empty_like(pgdef)
        a1_high = np.empty_like(pgdef)
        a2_high = np.empty_like(pgdef)
        a3_high = np.empty_like(pgdef)
        a4_high = np.empty_like(pgdef)
        a5_high = np.empty_like(pgdef)
        a6_high = np.empty_like(pgdef)
        # for eps_pipe
        b0_high =       np.empty_like(pgdef)
        b2_high =       np.empty_like(pgdef)
        b3_high =       np.empty_like(pgdef)
        b4_high =       np.empty_like(pgdef)
        b5_high =       np.empty_like(pgdef)
        d2_high =       np.empty_like(pgdef)
        d3_high =       np.empty_like(pgdef)
        d41_high =      np.empty_like(pgdef)
        d42_high =      np.empty_like(pgdef)
        d5_high =       np.empty_like(pgdef)
        d6_high =       np.empty_like(pgdef)
        d71_high =      np.empty_like(pgdef)
        d72_high =      np.empty_like(pgdef)
        d8_high =       np.empty_like(pgdef)
        sigma_ln_high = np.empty_like(pgdef)
        
        # model coefficients
        # determine coefficients based on psi_dip
        #####
        curr_cond = psi_dip <= 30
        if True in curr_cond:
            # for delta_u
            a0_low[curr_cond] = 3.9633
            a1_low[curr_cond] = 0.2937
            a2_low[curr_cond] = 1.2438
            a3_low[curr_cond] = -0.702
            a4_low[curr_cond] = -0.3957
            a5_low[curr_cond] = -0.4051
            a6_low[curr_cond] = 0.0001
            # for eps_pipe
            b0_low[curr_cond] =       0.8018
            b2_low[curr_cond] =       0.0265
            b3_low[curr_cond] =       0
            b4_low[curr_cond] =       0.018
            b5_low[curr_cond] =       0.1417
            d2_low[curr_cond] =       1.1363
            d3_low[curr_cond] =       0.0012
            d41_low[curr_cond] =      0.0038
            d42_low[curr_cond] =      0
            d5_low[curr_cond] =       0.0032
            d6_low[curr_cond] =       2.1297
            d71_low[curr_cond] =      0.001
            d72_low[curr_cond] =      0
            d8_low[curr_cond] =       -0.3867
            sigma_ln_low[curr_cond] = 0.349
            
            # for delta_u
            a0_high[curr_cond] = 3.9633
            a1_high[curr_cond] = 0.2937
            a2_high[curr_cond] = 1.2438
            a3_high[curr_cond] = -0.702
            a4_high[curr_cond] = -0.3957
            a5_high[curr_cond] = -0.4051
            a6_high[curr_cond] = 0.0001
            # for eps_pipe
            b0_high[curr_cond] =       0.8018
            b2_high[curr_cond] =       0.0265
            b3_high[curr_cond] =       0
            b4_high[curr_cond] =       0.018
            b5_high[curr_cond] =       0.1417
            d2_high[curr_cond] =       1.1363
            d3_high[curr_cond] =       0.0012
            d41_high[curr_cond] =      0.0038
            d42_high[curr_cond] =      0
            d5_high[curr_cond] =       0.0032
            d6_high[curr_cond] =       2.1297
            d71_high[curr_cond] =      0.001
            d72_high[curr_cond] =      0
            d8_high[curr_cond] =       -0.3867
            sigma_ln_high[curr_cond] = 0.349
        #####
        psi_low = 30
        psi_high = 45
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        if True in curr_cond:
            # for delta_u
            a0_low[curr_cond] = 3.9633
            a1_low[curr_cond] = 0.2937
            a2_low[curr_cond] = 1.2438
            a3_low[curr_cond] = -0.702
            a4_low[curr_cond] = -0.3957
            a5_low[curr_cond] = -0.4051
            a6_low[curr_cond] = 0.0001
            # for eps_pipe
            b0_low[curr_cond] =       0.8018
            b2_low[curr_cond] =       0.0265
            b3_low[curr_cond] =       0
            b4_low[curr_cond] =       0.018
            b5_low[curr_cond] =       0.1417
            d2_low[curr_cond] =       1.1363
            d3_low[curr_cond] =       0.0012
            d41_low[curr_cond] =      0.0038
            d42_low[curr_cond] =      0
            d5_low[curr_cond] =       0.0032
            d6_low[curr_cond] =       2.1297
            d71_low[curr_cond] =      0.001
            d72_low[curr_cond] =      0
            d8_low[curr_cond] =       -0.3867
            sigma_ln_low[curr_cond] = 0.349
            
            # for delta_u
            a0_high[curr_cond] = 3.7533
            a1_high[curr_cond] = 0.1451
            a2_high[curr_cond] = 1.2497
            a3_high[curr_cond] = -0.461
            a4_high[curr_cond] = 0.3914
            a5_high[curr_cond] = -0.2131
            a6_high[curr_cond] = -0.3414
            # for eps_pipe
            b0_high[curr_cond] =       -1.1082
            b2_high[curr_cond] =       0.1063
            b3_high[curr_cond] =       -0.1439
            b4_high[curr_cond] =       0.2788
            b5_high[curr_cond] =       -0.3103
            d2_high[curr_cond] =       1.2553
            d3_high[curr_cond] =       0.0003
            d41_high[curr_cond] =      0.0052
            d42_high[curr_cond] =      -0.0859
            d5_high[curr_cond] =       0.0006
            d6_high[curr_cond] =       -0.2176
            d71_high[curr_cond] =      -0.0269
            d72_high[curr_cond] =      0.5739
            d8_high[curr_cond] =       0.3446
            sigma_ln_high[curr_cond] = 0.3997
        #####
        psi_low = 45
        psi_high = 60
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        if True in curr_cond:
            # for delta_u
            a0_low[curr_cond] = 3.7533
            a1_low[curr_cond] = 0.1451
            a2_low[curr_cond] = 1.2497
            a3_low[curr_cond] = -0.461
            a4_low[curr_cond] = 0.3914
            a5_low[curr_cond] = -0.2131
            a6_low[curr_cond] = -0.3414
            # for eps_pipe
            b0_low[curr_cond] =       -1.1082
            b2_low[curr_cond] =       0.1063
            b3_low[curr_cond] =       -0.1439
            b4_low[curr_cond] =       0.2788
            b5_low[curr_cond] =       -0.3103
            d2_low[curr_cond] =       1.2553
            d3_low[curr_cond] =       0.0003
            d41_low[curr_cond] =      0.0052
            d42_low[curr_cond] =      -0.0859
            d5_low[curr_cond] =       0.0006
            d6_low[curr_cond] =       -0.2176
            d71_low[curr_cond] =      -0.0269
            d72_low[curr_cond] =      0.5739
            d8_low[curr_cond] =       0.3446
            sigma_ln_low[curr_cond] = 0.3997
            
            # for delta_u
            a0_high[curr_cond] = 4.3183
            a1_high[curr_cond] = -0.0279
            a2_high[curr_cond] = 1.0497
            a3_high[curr_cond] = -0.4691
            a4_high[curr_cond] = 0.2915
            a5_high[curr_cond] = -0.2861
            a6_high[curr_cond] = -0.1348
            # for eps_pipe
            b0_high[curr_cond] =       -2.1277
            b2_high[curr_cond] =       0.1476
            b3_high[curr_cond] =       -0.2183
            b4_high[curr_cond] =       0.4227
            b5_high[curr_cond] =       -0.5372
            d2_high[curr_cond] =       1.252
            d3_high[curr_cond] =       -0.0006
            d41_high[curr_cond] =      0.0053
            d42_high[curr_cond] =      -0.0485
            d5_high[curr_cond] =       0.0013
            d6_high[curr_cond] =       -0.566
            d71_high[curr_cond] =      -0.0321
            d72_high[curr_cond] =      0.8497
            d8_high[curr_cond] =       0.0901
            sigma_ln_high[curr_cond] = 0.5017
        #####
        psi_low = 60
        psi_high = 75
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        if True in curr_cond:
            # for delta_u
            a0_low[curr_cond] = 4.3183
            a1_low[curr_cond] = -0.0279
            a2_low[curr_cond] = 1.0497
            a3_low[curr_cond] = -0.4691
            a4_low[curr_cond] = 0.2915
            a5_low[curr_cond] = -0.2861
            a6_low[curr_cond] = -0.1348
            # for eps_pipe
            b0_low[curr_cond] =       -2.1277
            b2_low[curr_cond] =       0.1476
            b3_low[curr_cond] =       -0.2183
            b4_low[curr_cond] =       0.4227
            b5_low[curr_cond] =       -0.5372
            d2_low[curr_cond] =       1.252
            d3_low[curr_cond] =       -0.0006
            d41_low[curr_cond] =      0.0053
            d42_low[curr_cond] =      -0.0485
            d5_low[curr_cond] =       0.0013
            d6_low[curr_cond] =       -0.566
            d71_low[curr_cond] =      -0.0321
            d72_low[curr_cond] =      0.8497
            d8_low[curr_cond] =       0.0901
            sigma_ln_low[curr_cond] = 0.5017
            
            # for delta_u
            a0_high[curr_cond] = 5.5951
            a1_high[curr_cond] = 0.016
            a2_high[curr_cond] = 1.2641
            a3_high[curr_cond] = -0.5243
            a4_high[curr_cond] = 0.3583
            a5_high[curr_cond] = -0.3592
            a6_high[curr_cond] = -0.2482
            # for eps_pipe
            b0_high[curr_cond] =       -2.345
            b2_high[curr_cond] =       0.1947
            b3_high[curr_cond] =       -0.2044
            b4_high[curr_cond] =       0.4143
            b5_high[curr_cond] =       -0.5571
            d2_high[curr_cond] =       1.0931
            d3_high[curr_cond] =       0.0001
            d41_high[curr_cond] =      0.0035
            d42_high[curr_cond] =      -0.0407
            d5_high[curr_cond] =       0.0016
            d6_high[curr_cond] =       -0.6595
            d71_high[curr_cond] =      -0.0301
            d72_high[curr_cond] =      0.8422
            d8_high[curr_cond] =       0.5068
            sigma_ln_high[curr_cond] = 0.4378
        #####
        psi_low = 75
        psi_high = 90
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        if True in curr_cond:
            # for delta_u
            a0_low[curr_cond] = 5.5951
            a1_low[curr_cond] = 0.016
            a2_low[curr_cond] = 1.2641
            a3_low[curr_cond] = -0.5243
            a4_low[curr_cond] = 0.3583
            a5_low[curr_cond] = -0.3592
            a6_low[curr_cond] = -0.2482
            # for eps_pipe
            b0_low[curr_cond] =       -2.345
            b2_low[curr_cond] =       0.1947
            b3_low[curr_cond] =       -0.2044
            b4_low[curr_cond] =       0.4143
            b5_low[curr_cond] =       -0.5571
            d2_low[curr_cond] =       1.0931
            d3_low[curr_cond] =       0.0001
            d41_low[curr_cond] =      0.0035
            d42_low[curr_cond] =      -0.0407
            d5_low[curr_cond] =       0.0016
            d6_low[curr_cond] =       -0.6595
            d71_low[curr_cond] =      -0.0301
            d72_low[curr_cond] =      0.8422
            d8_low[curr_cond] =       0.5068
            sigma_ln_low[curr_cond] = 0.4378
            
            # for delta_u
            a0_high[curr_cond] = 14.5751
            a1_high[curr_cond] = 0.1356
            a2_high[curr_cond] = 2.999
            a3_high[curr_cond] = -0.9471
            a4_high[curr_cond] = 0.6603
            a5_high[curr_cond] = -1.2489
            a6_high[curr_cond] = -0.4414
            # for eps_pipe
            b0_high[curr_cond] =       5.1354
            b2_high[curr_cond] =       -0.0496
            b3_high[curr_cond] =       0.4459
            b4_high[curr_cond] =       -0.8371
            b5_high[curr_cond] =       0.6309
            d2_high[curr_cond] =       0.9139
            d3_high[curr_cond] =       0.0025
            d41_high[curr_cond] =      0.0016
            d42_high[curr_cond] =      -0.0975
            d5_high[curr_cond] =       0.0012
            d6_high[curr_cond] =       0.4648
            d71_high[curr_cond] =      0.0008
            d72_high[curr_cond] =      0.0679
            d8_high[curr_cond] =       0.5898
            sigma_ln_high[curr_cond] = 0.3475

        # setup
        # --------------------
        young_mod = 2e8 # kpa, hard-coded now
        # --------------------
        phi_rad = np.radians(phi_backfill)
        pi = np.pi
        d_pipe = d_pipe / 1000 # mm to meter
        t_pipe = t_pipe / 1000 # mm to meter
        d_in = d_pipe - 2*t_pipe # m
        circum = pi * d_pipe # m
        area = pi * ((d_pipe)**2 - (d_in)**2) / 4 # m^2
        h_d_ratio = h_pipe/d_pipe
        h_d_ratio = np.minimum(np.maximum(h_d_ratio,1.8),11.5) # limit h_d_ratio to 1.8 and 11.5
        d_t_ratio = d_pipe/t_pipe

        # find where sigma_y, n_param, and r_param are nan
        sigma_y_isnan = np.isnan(sigma_y)
        n_param_isnan = np.isnan(n_param)
        r_param_isnan = np.isnan(r_param)
        # initialize sigma_ult using X-52 properties
        sigma_ult = np.ones(pgdef.shape) * 455*1000 # kPa
        
        # if steel-grade is provided, use properties informed by steel grade
        # for steel_grade = "NA", set to "X-52"
        steel_grade[steel_grade=='NA'] = 'X-52'
        # Grade-B
        grade = 'Grade-B'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 241*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 8
        sigma_ult[cond] = 344*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.099377366 # %
        # Grade X-42
        grade = 'X-42'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 290*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 9
        sigma_ult[cond] = 414*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.642491598 # %
        # Grade X-52
        grade = 'X-52'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 359*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 10
        sigma_ult[cond] = 455*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.904124241 # %
        # Grade X-60
        grade = 'X-60'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 414*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 12
        sigma_ult[cond] = 517*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.431334401 # %
        # Grade X-70
        grade = 'X-70'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 483*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 14
        r_param[np.logical_and(cond,r_param_isnan)] = 15
        sigma_ult[cond] = 565*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.769051799 # %
        # Grade X-80
        grade = 'X-80'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 552*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 15
        r_param[np.logical_and(cond,r_param_isnan)] = 20
        sigma_ult[cond] = 625*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.846493399 # %
        # if any of the params are still missing, use default grade of X-52
        sigma_y[sigma_y_isnan] = 359*1000 # kPa
        n_param[n_param_isnan] = 8
        r_param[r_param_isnan] = 10
        
        # compute eps_ult (%)
        eps_ult = sigma_ult/young_mod * (1 + n_param/(1+r_param)*(sigma_ult/sigma_y)**r_param) * 100
        
        # calculations
        # find indices with sand and clay
        ind_sand = soil_type=='sand'
        ind_clay = soil_type=='clay'
        # for sand
        if True in ind_sand:
            t_u[ind_sand] = gamma_backfill[ind_sand] * h_pipe[ind_sand] \
                            * np.tan(delta_backfill[ind_sand] * phi_rad[ind_sand]) \
                            * circum[ind_sand]
            # depending on soil density
            ind_medium_dense = soil_density == 'medium dense'
            ind_dense = soil_density == 'dense'
            ind_very_dense = soil_density == 'very dense'
            if True in ind_medium_dense:
                curr_ind = np.logical_and(ind_sand,ind_medium_dense)
                n_q_vd[curr_ind] = \
                    13.082 + -1.038*h_d_ratio[curr_ind] + 0.579*(h_d_ratio[curr_ind])**2 \
                    + -0.072*(h_d_ratio[curr_ind])**3 + 0.0027*(h_d_ratio[curr_ind])**4
            if True in ind_dense:
                curr_ind = np.logical_and(ind_sand,ind_dense)
                n_q_vd[curr_ind] = \
                    17.168 + -1.571*h_d_ratio[curr_ind] + 0.585*(h_d_ratio[curr_ind])**2 \
                    + -0.063*(h_d_ratio[curr_ind])**3 + 0.0022*(h_d_ratio[curr_ind])**4
            if True in ind_very_dense:
                curr_ind = np.logical_and(ind_sand,ind_very_dense)
                n_q_vd[curr_ind] = \
                    22.265 + -3.567*h_d_ratio[curr_ind] + 1.083*(h_d_ratio[curr_ind])**2 \
                    + -0.103*(h_d_ratio[curr_ind])**3 + 0.0032*(h_d_ratio[curr_ind])**4
            n_gamma_d[ind_sand] = np.minimum(80, np.exp(0.18*phi_backfill[ind_sand]-2.5))
            q_vd[ind_sand] = \
                n_q_vd[ind_sand] * gamma_backfill[ind_sand] * h_pipe[ind_sand] * d_pipe[ind_sand] \
                + 0.5 * gamma_backfill[ind_sand] * d_pipe[ind_sand]**2 * n_gamma_d[ind_sand]
        # for clay
        if True in ind_clay:
            t_u[ind_clay] = alpha_backfill[ind_clay] * s_u_backfill[ind_clay] * circum[ind_clay]
            q_vd[ind_clay] = 5.14 * s_u_backfill[ind_clay] * d_pipe[ind_clay]

        # intermediate calcs
        ln_l_anchor = np.log(l_anchor)
        ln_d_pipe = np.log(d_pipe)
        ln_d_t_ratio = np.log(d_t_ratio)
        ln_eps_ult = np.log(eps_ult)
        ln_q_vd = np.log(q_vd)
        ln_t_u = np.log(t_u)

        # calculate delta_u
        delta_u_low = np.exp(
            a0_low + \
            a1_low * ln_l_anchor + \
            a2_low * ln_d_pipe + \
            a3_low * ln_d_t_ratio + \
            a4_low * ln_eps_ult + \
            a5_low * ln_q_vd + \
            a6_low * ln_t_u
        )
        delta_u_high = np.exp(
            a0_high + \
            a1_high * ln_l_anchor + \
            a2_high * ln_d_pipe + \
            a3_high * ln_d_t_ratio + \
            a4_high * ln_eps_ult + \
            a5_high * ln_q_vd + \
            a6_high * ln_t_u
        )
        
        # prepare for eps_pipe calculation
        # deformation flag
        f_delta_f_low = np.zeros(pgdef.shape)
        f_delta_f_low[pgdef < delta_u_low] = 1
        f_delta_f_high = np.zeros(pgdef.shape)
        f_delta_f_high[pgdef < delta_u_high] = 1
        # anchorlage length flag
        f_l_anchor = np.zeros(pgdef.shape)
        f_l_anchor[l_anchor < 100] = 1
        # soil type flag
        f_soil_type = np.zeros(pgdef.shape)
        f_soil_type[soil_type == 'sand'] = 1
        # intermediate calculations
        b1_low = (
            f_delta_f_low*(
                d2_low + \
                d3_low * t_u + \
                d41_low * f_l_anchor*(l_anchor-100) + \
                d42_low * (1-f_l_anchor) + \
                d5_low * d_t_ratio
            ) + \
            (1-f_delta_f_low) * (
                d6_low + \
                d71_low * f_l_anchor*(l_anchor-100) + \
                d72_low * (1-f_l_anchor) + \
                d8_low * ln_eps_ult
            )
        )
        b1_high = (
            f_delta_f_high*(
                d2_high + \
                d3_high * t_u + \
                d41_high * f_l_anchor*(l_anchor-100) + \
                d42_high * (1-f_l_anchor) + \
                d5_high * d_t_ratio
            ) + \
            (1-f_delta_f_high) * (
                d6_high + \
                d71_high * f_l_anchor*(l_anchor-100) + \
                d72_high * (1-f_l_anchor) + \
                d8_high * ln_eps_ult
            )
        )
        # other intermediate calcs
        ln_delta_f_delta_u_ratio_low = np.log(pgdef/delta_u_low)
        ln_delta_f_delta_u_ratio_high = np.log(pgdef/delta_u_high)
        
        # pipe strain
        ln_eps_pipe_low = \
            b0_low + \
            b1_low * ln_delta_f_delta_u_ratio_low + \
            b2_low * ln_d_t_ratio + \
            (
                b3_low * f_soil_type * ln_t_u + \
                b4_low * ln_q_vd
            ) + \
            b5_low * ln_d_pipe
        ln_eps_pipe_high = \
            b0_high + \
            b1_high * ln_delta_f_delta_u_ratio_high + \
            b2_high * ln_d_t_ratio + \
            (
                b3_high * f_soil_type * ln_t_u + \
                b4_high * ln_q_vd
            ) + \
            b5_high * ln_d_pipe
        
        # interpolate pipe strain
        interp_factor = np.empty_like(pgdef)
        #####
        curr_cond = psi_dip <= 30
        interp_factor[curr_cond] = 0
        #####
        psi_low = 30
        psi_high = 45
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        interp_factor[curr_cond] = (psi_dip[curr_cond] - psi_low)/(psi_high - psi_low)
        #####
        psi_low = 45
        psi_high = 60
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        interp_factor[curr_cond] = (psi_dip[curr_cond] - psi_low)/(psi_high - psi_low)
        #####
        psi_low = 60
        psi_high = 75
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        interp_factor[curr_cond] = (psi_dip[curr_cond] - psi_low)/(psi_high - psi_low)
        #####
        psi_low = 75
        psi_high = 90
        curr_cond = np.logical_and(psi_dip > psi_low, psi_dip <= psi_high)
        interp_factor[curr_cond] = (psi_dip[curr_cond] - psi_low)/(psi_high - psi_low)

        # interpolate for pipe strain and raise by exp
        eps_pipe = np.exp(ln_eps_pipe_low + interp_factor*(ln_eps_pipe_high - ln_eps_pipe_low))
        eps_pipe = np.maximum(np.minimum(eps_pipe, 100), 1e-5) # apply limits
        
        # sigma
        sigma = sigma_ln_low + interp_factor*(sigma_ln_high - sigma_ln_low)
        
        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': sigma,
                'sigma_mu': np.ones(eps_pipe.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['d_t_ratio'] = d_t_ratio
            output['sigma_ult'] = sigma_ult
            output['eps_ult'] = eps_ult
            output['h_d_ratio'] = h_d_ratio
            output['t_u'] = t_u
            output['n_q_vd'] = n_q_vd
            output['n_gamma_d'] = n_gamma_d
            output['q_vd'] = q_vd
            output['ln_q_vd'] = ln_q_vd
            output['ln_t_u'] = ln_t_u
            output['f_l_anchor'] = f_l_anchor
            output['f_soil_type'] = f_soil_type
        
        # return
        return output
    
    
# -----------------------------------------------------------
class HutabaratEtal2022_Reverse(_PipeStrainBase):
    """
    Model for reverse-slip-induced pipe strain using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - reverse-slip-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    psi_dip: float, np.ndarray or list
        [deg] pipe-fault dip angle
    h_pipe: float, np.ndarray or list
        [m] burial depth to pipe centerline
    
    Clay:
    s_u_backfill: float, np.ndarray or list
        [kPa] undrained shear strength
    alpha_backfill: float, np.ndarray or list
        adhesion factor
    
    Sand:
    gamma_backfill: float, np.ndarray or list
        [kN/m^3] total unit weight of backfill soil, inferred 
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Hutabarat et al. (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Hutabarat, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation - reverse-slip-induced (m)',
                'unit': 'm',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': 'pipe diameter (mm)',
            't_pipe': 'pipe wall thickness (mm)',
            'l_anchor': 'pipeline anchored length (m)',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'psi_dip': 'pipe-fault dip angle [deg]',
            'h_pipe': 'burial depth to pipe centerline (m)',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': 'soil type (sand/clay) for model',
            # 'soil_density': 'soil_density': 'soil density: soft, medium stiff, or stiff for clay; medium dense, dense, or very dense for sand',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'clay': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'h_pipe', 'alpha_backfill', 's_u_backfill'],
        },
        'sand': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'h_pipe', 'gamma_backfill', 'phi_backfill', 'delta_backfill'],
        }
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'clay': {
            'level1': ['soil_type'],
            'level2': ['soil_type'],
            'level3': ['soil_type'],
        },
        'sand': {
            'level1': ['soil_type'],
            'level2': ['soil_type'],
            'level3': ['soil_type'],
        }
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """uses soil_type to determine what model parameters to use"""
        soil_type = kwargs.get('soil_type', None)
        if soil_type is None:
            soil_type = ['clay','sand']
        req_rvs_by_level = {}
        req_fixed_by_level = {}
        soils = []
        if len(soil_type) == 0:
            soils = ['clay'] # if soil_type is empty, just use clay as default
        else:
            if 'sand' in soil_type:
                soils.append('sand')
            if 'clay' in soil_type:
                soils.append('clay')
        for i in range(3):
            for each in soils:
                if f'level{i+1}' in req_rvs_by_level:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
                else:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
            req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
            req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))
        return req_rvs_by_level, req_fixed_by_level


    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, l_anchor, # infrastructure
        psi_dip, h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        # soil_type, soil_density, steel_grade, # fixed/toggles
        soil_type, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # initialize arrays
        t_u = np.empty_like(pgdef)

        # setup
        # --------------------
        young_mod = 210000000 # kpa, hard-coded now
        # --------------------
        phi_rad = np.radians(phi_backfill)
        pi = np.pi
        d_pipe = d_pipe / 1000 # mm to meter
        t_pipe = t_pipe / 1000 # mm to meter
        d_in = d_pipe - 2*t_pipe # m
        circum = pi * d_pipe # m
        area = pi * ((d_pipe)**2 - (d_in)**2) / 4 # m^2
        h_d_ratio = h_pipe/d_pipe
        h_d_ratio = np.minimum(np.maximum(h_d_ratio,1.8),11.5) # limit h_d_ratio to 1.8 and 11.5
        d_t_ratio = d_pipe/t_pipe
        ln_d_t_ratio = np.log(d_t_ratio)
        
        # calculations
        # find indices with sand and clay
        ind_sand = soil_type=='sand'
        ind_clay = soil_type=='clay'
        # for sand
        if True in ind_sand:
            t_u[ind_sand] = gamma_backfill[ind_sand] * h_pipe[ind_sand] \
                            * np.tan(delta_backfill[ind_sand] * phi_rad[ind_sand]) \
                            * circum[ind_sand]
        # for clay
        if True in ind_clay:
            t_u[ind_clay] = alpha_backfill[ind_clay] * s_u_backfill[ind_clay] * circum[ind_clay]

        # factors for eps_pipe
        f_psi_dip = np.zeros(pgdef.shape)
        cond = np.logical_and(psi_dip >= 60, psi_dip < 90)
        if True in cond:
            f_psi_dip[cond] = 60 - psi_dip[cond]
        f_d_pipe = np.zeros(pgdef.shape)
        f_d_pipe[d_pipe < 0.5] = 1
        
        # get coefficients
        b0 = -4.11127 + 0.6064*d_pipe + 0.002805*l_anchor + 0.038944*f_psi_dip
        b1 = 2.29445 + (-0.04675*d_pipe) + (-0.00104*l_anchor) + (-0.09201*f_psi_dip)
        b2 = 0.42882 + 0.09845*d_pipe + 0.0006*l_anchor + 0.01203*f_psi_dip
        b3 = 2.64335 + (-0.36353*d_pipe) + 0.00086*l_anchor + (-0.05422*f_psi_dip)
        b4 = -4.57877 + (-0.04142*(ln_d_t_ratio**2)) + \
            (0.9346*ln_d_t_ratio) + (0.4714*np.log(t_u)) + \
            (0.00007*(180-psi_dip)) + (-5.2467*f_d_pipe*(d_pipe-0.5)) + \
            (-0.28986*(1-f_d_pipe))
        
        # calculate pipe strain
        metric_inside_atanh = (np.log(pgdef) - b0) / b1
        # limit atanh between -5 and 5
        atanh_metric = np.ones(pgdef.shape)*5
        cond = np.logical_and(metric_inside_atanh>-1,metric_inside_atanh<1)
        atanh_metric[cond] = np.arctanh(metric_inside_atanh[cond])
        cond = metric_inside_atanh<=-1
        atanh_metric[cond] = -5
        eps_pipe = np.exp(atanh_metric/b2 - b3 + b4)
        eps_pipe = np.maximum(np.minimum(eps_pipe, 100), 1e-5) # limit to 1e-5 % (avoid 0%) and 100 %
        
                    
        # sigma, from mixture 2 normal model
        mu_1 = -0.283
        mu_2 = -0.625
        sigma_1 = 0.46
        sigma_2 = 0.709
        weight_1 = 0.688
        weight_2 = 0.312
        sigma_mix = np.sqrt(
            weight_1*sigma_1**2 + weight_2*sigma_2**2 + \
            weight_1*weight_2*(mu_1-mu_2)**2
        )
        sigma = np.ones(pgdef.shape)*sigma_mix
        
        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': sigma,
                'sigma_mu': np.ones(eps_pipe.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['d_t_ratio'] = d_t_ratio
            output['t_u'] = t_u
            output['f_psi_dip'] = f_psi_dip
            output['f_d_pipe'] = f_d_pipe
            output['ln_d_t_ratio'] = ln_d_t_ratio
            output['b0'] = b0
            output['b1'] = b1
            output['b2'] = b2
            output['b3'] = b3
            output['b4'] = b4
            output['atanh_metric'] = atanh_metric
            
        # return
        return output
    
    
# -----------------------------------------------------------
class HutabaratEtal2022_SSComp(_PipeStrainBase):
    """
    Model for strike-slip-compression-induced pipe strain using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - strike-slip-compression-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    beta_crossing: float, np.ndarray or list
        [deg] pipe-fault dip angle
        
    Fixed:

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Hutabarat et al. (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Hutabarat, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation - strike-slip-compression-induced (m)',
                'unit': 'm',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': 'pipe diameter (mm)',
            'l_anchor': 'pipeline anchored length (m)',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'beta_crossing': 'pipe-fault dip angle [deg]',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': ['d_pipe'],
        'level3': ['d_pipe'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': [],
        'level2': [],
        'level3': [],
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
    

    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, l_anchor, # infrastructure
        beta_crossing, # geotech - general
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # setup
        d_pipe = d_pipe / 1000 # mm to meter

        # factors
        f_beta_crossing = np.zeros(pgdef.shape)
        cond = np.logical_and(beta_crossing > 90, beta_crossing <= 120)
        if True in cond:
            f_beta_crossing[cond] = beta_crossing[cond] - 120
        
        # get coefficients
        b0 = -6.50785 + 0.98692*d_pipe + 0.01601*l_anchor + (-0.04575*f_beta_crossing)
        b1 = 4.54097 - 0.01093*l_anchor
        b2 = 0.34262 + (-0.10918*d_pipe) + 0.00197*l_anchor+ 0.0027*f_beta_crossing
        
        # calculate pipe strain
        metric_inside_atanh = (np.log(pgdef) - b0) / b1
        # limit atanh between -5 and 5
        atanh_metric = np.ones(pgdef.shape)*5
        cond = np.logical_and(metric_inside_atanh>-1,metric_inside_atanh<1)
        atanh_metric[cond] = np.arctanh(metric_inside_atanh[cond])
        cond = metric_inside_atanh<=-1
        atanh_metric[cond] = -5
        eps_pipe = np.exp(atanh_metric/b2 - 4)
        eps_pipe = np.maximum(np.minimum(eps_pipe, 100), 1e-5) # limit to 1e-5 % (avoid 0%) and 100 %
        
        # sigma
        sigma = np.ones(pgdef.shape)*0.571
        
        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': sigma,
                'sigma_mu': np.ones(eps_pipe.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['f_beta_crossing'] = f_beta_crossing
            output['b0'] = b0
            output['b1'] = b1
            output['b2'] = b2
            output['atanh_metric'] = atanh_metric
            
        # return
        return output
    
    
# -----------------------------------------------------------
class HutabaratEtal2022_SSTens_85to90(_PipeStrainBase):
    """
    Model for strike-slip-tension-induced pipe strain with beta between 85 and 90, using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - strike-slip-tension-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    h_pipe: float, np.ndarray or list
        [m] burial depth to pipe centerline
    
    Clay:
    s_u_backfill: float, np.ndarray or list
        [kPa] undrained shear strength
    alpha_backfill: float, np.ndarray or list
        adhesion factor
    
    Sand:
    gamma_backfill: float, np.ndarray or list
        [kN/m^3] total unit weight of backfill soil, inferred 
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Hutabarat et al. (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Hutabarat, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation - strike-slip-tension-induced (m)',
                'unit': 'm',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
            'd_pipe': 'pipe diameter (mm)',
            't_pipe': 'pipe wall thickness (mm)',
            'sigma_y': 'pipe yield stress (kPa)',
            'n_param': 'Ramberg-Osgood parameter',
            'r_param': 'Ramberg-Osgood parameter',
            'l_anchor': 'pipeline anchored length (m)',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            # 'beta_crossing': 'pipe-fault dip angle [deg]',
            'h_pipe': 'burial depth to pipe centerline (m)',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': 'soil type (sand/clay) for model',
            # 'soil_density': 'soil density: soft, medium stiff, or stiff for clay; medium dense, dense, or very dense for sand',
            'steel_grade': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'clay': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'h_pipe', 'alpha_backfill', 's_u_backfill'],
        },
        'sand': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe'],
            'level3': ['d_pipe', 't_pipe', 'h_pipe', 'gamma_backfill', 'phi_backfill', 'delta_backfill'],
        }
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'clay': {
            'level1': ['soil_type'],
            'level2': ['soil_type'],
            'level3': ['soil_type'],
        },
        'sand': {
            'level1': ['soil_type'],
            'level2': ['soil_type'],
            'level3': ['soil_type'],
        }
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """uses soil_type to determine what model parameters to use"""
        soil_type = kwargs.get('soil_type', None)
        if soil_type is None:
            soil_type = ['clay','sand']
        req_rvs_by_level = {}
        req_fixed_by_level = {}
        soils = []
        if len(soil_type) == 0:
            soils = ['clay'] # if soil_type is empty, just use clay as default
        else:
            if 'sand' in soil_type:
                soils.append('sand')
            if 'clay' in soil_type:
                soils.append('clay')
        for i in range(3):
            for each in soils:
                if f'level{i+1}' in req_rvs_by_level:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
                else:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL[each][f'level{i+1}']
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL[each][f'level{i+1}']
            req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
            req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))
        return req_rvs_by_level, req_fixed_by_level


    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, sigma_y, n_param, r_param, l_anchor, # infrastructure
        h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        # soil_type, soil_density, steel_grade, # fixed/toggles
        soil_type, steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # initialize arrays
        # sigma_ult = np.empty_like(pgdef)
        # eps_ult = np.empty_like(pgdef)
        t_u = np.empty_like(pgdef)
        
        # setup
        # --------------------
        young_mod = 2e8 # kpa, hard-coded now
        # --------------------
        phi_rad = np.radians(phi_backfill)
        pi = np.pi
        d_pipe = d_pipe / 1000 # mm to meter
        t_pipe = t_pipe / 1000 # mm to meter
        d_in = d_pipe - 2*t_pipe # m
        circum = pi * d_pipe # m
        area = pi * ((d_pipe)**2 - (d_in)**2) / 4 # m^2
        h_d_ratio = h_pipe/d_pipe
        h_d_ratio = np.minimum(np.maximum(h_d_ratio,1.8),11.5) # limit h_d_ratio to 1.8 and 11.5
        d_t_ratio = d_pipe/t_pipe
        ln_d_t_ratio = np.log(d_t_ratio)
        
        # find where sigma_y, n_param, and r_param are nan
        sigma_y_isnan = np.isnan(sigma_y)
        n_param_isnan = np.isnan(n_param)
        r_param_isnan = np.isnan(r_param)
        # initialize sigma_ult using X-52 properties
        sigma_ult = np.ones(pgdef.shape) * 455*1000 # kPa
        
        # if steel-grade is provided, use properties informed by steel grade
        # for steel_grade = "NA", set to "X-52"
        steel_grade[steel_grade=='NA'] = 'X-52'
        # Grade-B
        grade = 'Grade-B'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 241*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 8
        sigma_ult[cond] = 344*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.099377366 # %
        # Grade X-42
        grade = 'X-42'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 290*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 9
        sigma_ult[cond] = 414*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.642491598 # %
        # Grade X-52
        grade = 'X-52'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 359*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 10
        sigma_ult[cond] = 455*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.904124241 # %
        # Grade X-60
        grade = 'X-60'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 414*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 12
        sigma_ult[cond] = 517*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.431334401 # %
        # Grade X-70
        grade = 'X-70'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 483*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 14
        r_param[np.logical_and(cond,r_param_isnan)] = 15
        sigma_ult[cond] = 565*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.769051799 # %
        # Grade X-80
        grade = 'X-80'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 552*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 15
        r_param[np.logical_and(cond,r_param_isnan)] = 20
        sigma_ult[cond] = 625*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.846493399 # %
        # if any of the params are still missing, use default grade of X-52
        sigma_y[sigma_y_isnan] = 359*1000 # kPa
        n_param[n_param_isnan] = 8
        r_param[r_param_isnan] = 10
        
        # compute eps_ult (%)
        eps_ult = sigma_ult/young_mod * (1 + n_param/(1+r_param)*(sigma_ult/sigma_y)**r_param) * 100
        
        # calculations
        # find indices with sand and clay
        ind_sand = soil_type=='sand'
        ind_clay = soil_type=='clay'
        # for sand
        if True in ind_sand:
            t_u[ind_sand] = gamma_backfill[ind_sand] * h_pipe[ind_sand] \
                            * np.tan(delta_backfill[ind_sand] * phi_rad[ind_sand]) \
                            * circum[ind_sand]
        # for clay
        if True in ind_clay:
            t_u[ind_clay] = alpha_backfill[ind_clay] * s_u_backfill[ind_clay] * circum[ind_clay]

        # coefficients
        a0 = -0.15507
        a2 = 0.05203
        c3 = 0.00081
        c4 = 0.00314
        b0 = -2.30723
        b2 = 0.58524
        b3 = 0.20132
        e1 = 1.42749
        e2 = -0.00711
        e3 = 0.00514
        e4 = -1.42906
        e5 = 0.04920
        e6 = 0.14514
        e7 = 0.02017
        e8 = -1.03203
    
        # intermediate calcs for delta_u
        a1 = c3*t_u + c4*np.log(eps_ult)
        ln_delta_o = a0 + a1*ln_d_t_ratio + a2*np.log(l_anchor)
        delta_o = np.exp(ln_delta_o)
    
        # calculate delta_u
        ln_delta_u = ln_delta_o + 1
        ln_delta_u[l_anchor<15] = ln_delta_o[l_anchor<15] + 0.75
        ln_delta_u[l_anchor>50] = ln_delta_o[l_anchor>50] + 1.5
        delta_u = np.exp(ln_delta_u)
    
        # factors for eps_pipe
        f_delta_o = np.zeros(pgdef.shape)
        f_delta_o[pgdef < delta_o] = 1
        f_delta_u = np.zeros(pgdef.shape)
        f_delta_u[pgdef > delta_u] = 1
        # soil type flag
        f_soil_type = np.zeros(pgdef.shape)
        f_soil_type[soil_type == 'sand'] = 1
    
        # get coefficients
        b1 = \
            f_delta_o*np.log(pgdef/delta_o)*(
                e1 + e2*t_u + e3*d_t_ratio
            ) + \
            f_delta_u*np.log(pgdef/delta_u)*(
                e4 + e5*t_u + e6*l_anchor + e7*d_t_ratio + e8*np.log(eps_ult)
            )
    
        # calculate pipe strain, and limit to 1e-5% and 100%
        eps_pipe = np.exp(b0 + b1*pgdef + b2*ln_d_t_ratio + f_soil_type*b3*np.log(t_u))
        eps_pipe = np.maximum(np.minimum(eps_pipe, 100), 1e-5)
    
        # sigma
        sigma = np.ones(pgdef.shape)*0.723
        
        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': sigma,
                'sigma_mu': np.ones(eps_pipe.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['d_t_ratio'] = d_t_ratio
            output['sigma_ult'] = sigma_ult
            output['eps_ult'] = eps_ult
            output['t_u'] = t_u
            output['a1'] = a1
            output['ln_delta_o'] = ln_delta_o
            output['ln_delta_u'] = ln_delta_u
            output['f_delta_o'] = f_delta_o
            output['f_delta_u'] = f_delta_u
            output['f_soil_type'] = f_soil_type
            output['b1'] = b1
            
        # return
        return output
    
    
# -----------------------------------------------------------
class HutabaratEtal2022_SSTens_5to85(HutabaratEtal2022_SSTens_85to90):
    """
    Model for strike-slip-tension-induced pipe strain with beta between 5 and 85, using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - strike-slip-tension-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    h_pipe: float, np.ndarray or list
        [m] burial depth to pipe centerline
    
    Clay:
    s_u_backfill: float, np.ndarray or list
        [kPa] undrained shear strength
    alpha_backfill: float, np.ndarray or list
        adhesion factor
    
    Sand:
    gamma_backfill: float, np.ndarray or list
        [kN/m^3] total unit weight of backfill soil, inferred 
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'beta_crossing': 'pipe-fault dip angle [deg]',
            'h_pipe': 'burial depth to pipe centerline (m)',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
        }
    }
    

    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, sigma_y, n_param, r_param, l_anchor, # infrastructure
        beta_crossing, h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        # soil_type, soil_density, steel_grade, # fixed/toggles
        soil_type, steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # initialize arrays
        # sigma_ult = np.empty_like(pgdef)
        # eps_ult = np.empty_like(pgdef)
        t_u = np.empty_like(pgdef)
        
        # setup
        # --------------------
        young_mod = 2e8 # kpa, hard-coded now
        # --------------------
        phi_rad = np.radians(phi_backfill)
        pi = np.pi
        d_pipe = d_pipe / 1000 # mm to meter
        t_pipe = t_pipe / 1000 # mm to meter
        d_in = d_pipe - 2*t_pipe # m
        circum = pi * d_pipe # m
        area = pi * ((d_pipe)**2 - (d_in)**2) / 4 # m^2
        h_d_ratio = h_pipe/d_pipe
        h_d_ratio = np.minimum(np.maximum(h_d_ratio,1.8),11.5) # limit h_d_ratio to 1.8 and 11.5
        d_t_ratio = d_pipe/t_pipe
        ln_d_t_ratio = np.log(d_t_ratio)
        
        # find where sigma_y, n_param, and r_param are nan
        sigma_y_isnan = np.isnan(sigma_y)
        n_param_isnan = np.isnan(n_param)
        r_param_isnan = np.isnan(r_param)
        # initialize sigma_ult using X-52 properties
        sigma_ult = np.ones(pgdef.shape) * 455*1000 # kPa
        
        # if steel-grade is provided, use properties informed by steel grade
        # for steel_grade = "NA", set to "X-52"
        steel_grade[steel_grade=='NA'] = 'X-52'
        # Grade-B
        grade = 'Grade-B'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 241*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 8
        sigma_ult[cond] = 344*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.099377366 # %
        # Grade X-42
        grade = 'X-42'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 290*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 3
        r_param[np.logical_and(cond,r_param_isnan)] = 9
        sigma_ult[cond] = 414*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.642491598 # %
        # Grade X-52
        grade = 'X-52'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 359*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 10
        sigma_ult[cond] = 455*1000 # kPa
        # eps_ult[steel_grade==grade] = 1.904124241 # %
        # Grade X-60
        grade = 'X-60'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 414*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 8
        r_param[np.logical_and(cond,r_param_isnan)] = 12
        sigma_ult[cond] = 517*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.431334401 # %
        # Grade X-70
        grade = 'X-70'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 483*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 14
        r_param[np.logical_and(cond,r_param_isnan)] = 15
        sigma_ult[cond] = 565*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.769051799 # %
        # Grade X-80
        grade = 'X-80'
        cond = steel_grade==grade
        sigma_y[np.logical_and(cond,sigma_y_isnan)] = 552*1000 # kPa
        n_param[np.logical_and(cond,n_param_isnan)] = 15
        r_param[np.logical_and(cond,r_param_isnan)] = 20
        sigma_ult[cond] = 625*1000 # kPa
        # eps_ult[steel_grade==grade] = 2.846493399 # %
        # if any of the params are still missing, use default grade of X-52
        sigma_y[sigma_y_isnan] = 359*1000 # kPa
        n_param[n_param_isnan] = 8
        r_param[r_param_isnan] = 10
        
        # compute eps_ult (%)
        eps_ult = sigma_ult/young_mod * (1 + n_param/(1+r_param)*(sigma_ult/sigma_y)**r_param) * 100
        ln_eps_ult = np.log(eps_ult)
        
        # calculations
        # find indices with sand and clay
        ind_sand = soil_type=='sand'
        ind_clay = soil_type=='clay'
        # for sand
        if True in ind_sand:
            t_u[ind_sand] = gamma_backfill[ind_sand] * h_pipe[ind_sand] \
                            * np.tan(delta_backfill[ind_sand] * phi_rad[ind_sand]) \
                            * circum[ind_sand]
        # for clay
        if True in ind_clay:
            t_u[ind_clay] = alpha_backfill[ind_clay] * s_u_backfill[ind_clay] * circum[ind_clay]
        
        # initialize coeffs
        # coeff set 1
        a0 = np.zeros(beta_crossing.shape)
        a2 = np.zeros(beta_crossing.shape)
        c3 = np.zeros(beta_crossing.shape)
        c41 = np.zeros(beta_crossing.shape)
        c42 = np.zeros(beta_crossing.shape)
        c51 = np.zeros(beta_crossing.shape)
        c52 = np.zeros(beta_crossing.shape)
        # coeff set 2
        b0 = np.zeros(beta_crossing.shape)
        b2 = np.zeros(beta_crossing.shape)
        b3 = np.zeros(beta_crossing.shape)
        d2 = np.zeros(beta_crossing.shape)
        d3 = np.zeros(beta_crossing.shape)
        d41 = np.zeros(beta_crossing.shape)
        d42 = np.zeros(beta_crossing.shape)
        d5 = np.zeros(beta_crossing.shape)
        d6 = np.zeros(beta_crossing.shape)
        d71 = np.zeros(beta_crossing.shape)
        d72 = np.zeros(beta_crossing.shape)
        d8 = np.zeros(beta_crossing.shape)
        sigma = np.zeros(beta_crossing.shape)
        
        # additional conditions
        cond_beta_btw_5_45 = np.logical_and(beta_crossing>=5, beta_crossing<45)
        cond_beta_btw_45_85 = np.logical_and(beta_crossing>=45, beta_crossing<=85)
        
        # get coefficients
        if True in cond_beta_btw_5_45:
            beta_crossing_cond = beta_crossing[cond_beta_btw_5_45]
            #coeff set 1
            a0[cond_beta_btw_5_45] =  -0.05402 * (beta_crossing_cond-45) - 1.82829
            a2[cond_beta_btw_5_45] =   0.01347 * (beta_crossing_cond-45) + 0.37664
            c3[cond_beta_btw_5_45] =  -0.00301 * (beta_crossing_cond-45) - 0.01591
            c41[cond_beta_btw_5_45] =  0.02182 * (beta_crossing_cond-45) + 0.49488
            c42[cond_beta_btw_5_45] =  0.02436 * (beta_crossing_cond-45) + 0.47831
            c51[cond_beta_btw_5_45] = -0.00001 * (beta_crossing_cond-45) - 0.00165
            c52[cond_beta_btw_5_45] =  0.00228 * (beta_crossing_cond-45) + 0.10021
            # coeff set 2
            b0[cond_beta_btw_5_45] =    -0.02174 * (beta_crossing_cond-45) + 0.16235
            b2[cond_beta_btw_5_45] =     0.00203 * (beta_crossing_cond-45) + 0.24407
            b3[cond_beta_btw_5_45] =    -0.02801 * (beta_crossing_cond-45) + 1.64437
            d2[cond_beta_btw_5_45] =    -0.00010 * (beta_crossing_cond-45) - 0.00387
            d3[cond_beta_btw_5_45] =    -0.00114 * (beta_crossing_cond-45) + 0.00514
            d41[cond_beta_btw_5_45] =    0.01436 * (beta_crossing_cond-45) - 0.12124
            d42[cond_beta_btw_5_45] =    0.00002 * (beta_crossing_cond-45) + 0.00092
            d5[cond_beta_btw_5_45] =     0.01326 * (beta_crossing_cond-45) + 0.97745
            d6[cond_beta_btw_5_45] =     0.00061 * (beta_crossing_cond-45) + 0.00602
            d71[cond_beta_btw_5_45] =   -0.00728 * (beta_crossing_cond-45) - 0.06927
            d72[cond_beta_btw_5_45] =    0.01480 * (beta_crossing_cond-45) + 0.46008
            d8[cond_beta_btw_5_45] =     0.00272 * (beta_crossing_cond-45) + 0.11565
            sigma[cond_beta_btw_5_45] =  0.00302 * (beta_crossing_cond-45) + 0.53947
        if True in cond_beta_btw_45_85:
            beta_crossing_cond = beta_crossing[cond_beta_btw_45_85]
            #coeff set 1
            a0[cond_beta_btw_45_85] =   0.00735 * (beta_crossing_cond-75) - 1.60779
            a2[cond_beta_btw_45_85] =   0.00484 * (beta_crossing_cond-75) + 0.52187
            c3[cond_beta_btw_45_85] =  -0.02185 * (beta_crossing_cond-75) - 0.67156
            c41[cond_beta_btw_45_85] =  0.05619 * (beta_crossing_cond-75) + 2.18087
            c42[cond_beta_btw_45_85] =  0.06430 * (beta_crossing_cond-75) + 2.40733
            c51[cond_beta_btw_45_85] =                                    - 0.00153
            c52[cond_beta_btw_45_85] =  0.00144 * (beta_crossing_cond-75) + 0.14358
            # coeff set 2
            b0[cond_beta_btw_45_85] =    -0.02787 * (beta_crossing_cond-75) - 0.67388
            b2[cond_beta_btw_45_85] =     0.00361 * (beta_crossing_cond-75) + 0.35249
            b3[cond_beta_btw_45_85] =     0.00794 * (beta_crossing_cond-75) + 1.88270
            d2[cond_beta_btw_45_85] =    -0.00002 * (beta_crossing_cond-75) - 0.00456
            d3[cond_beta_btw_45_85] =     0.00057 * (beta_crossing_cond-75) + 0.02215
            d41[cond_beta_btw_45_85] =   -0.00844 * (beta_crossing_cond-75) - 0.37439
            d42[cond_beta_btw_45_85] =    0.00002 * (beta_crossing_cond-75) + 0.00156
            d5[cond_beta_btw_45_85] =    -0.00799 * (beta_crossing_cond-75) + 0.73788
            d6[cond_beta_btw_45_85] =    -0.00081 * (beta_crossing_cond-75) - 0.01826
            d71[cond_beta_btw_45_85] =    0.00522 * (beta_crossing_cond-75) + 0.08748
            d72[cond_beta_btw_45_85] =   -0.00924 * (beta_crossing_cond-75) + 0.18293
            d8[cond_beta_btw_45_85] =     0.00122 * (beta_crossing_cond-75) + 0.15234
            sigma[cond_beta_btw_45_85] =  0.00428 * (beta_crossing_cond-75) + 0.66796

        # toggles
        f_t_u = np.zeros(t_u.shape)
        f_t_u[t_u<70] = 1
        f_d_t_ratio = np.zeros(d_t_ratio.shape)
        f_d_t_ratio[d_t_ratio<100] = 1
        
        # calculate delta_u
        ln_l_anchor = np.log(l_anchor)
        a1 = \
            c3*ln_l_anchor + \
            c41*f_t_u + c42*(1-f_t_u) + \
            c51*f_d_t_ratio*(d_t_ratio-100) + c52*(1-f_d_t_ratio)
        ln_delta_u = a0 + a1*ln_eps_ult + a2*ln_l_anchor
        delta_u = np.exp(ln_delta_u)
        
        # more toggles
        f_delta_f = np.zeros(delta_u.shape)
        f_delta_f[pgdef<delta_u] = 1
        f_l_anchor = np.zeros(l_anchor.shape)
        f_l_anchor[l_anchor<50] = 1
        f_soil_type = np.zeros(pgdef.shape)
        f_soil_type[soil_type == 'sand'] = 1
        
        # get coefficient b1
        b1 = \
            f_delta_f*(
                d2 + d3*t_u + d41*f_l_anchor*(l_anchor-50) + d42*(1-f_l_anchor) + d5*d_t_ratio
            ) + \
            (1-f_delta_f)*(
                d6 + d71*f_l_anchor*(l_anchor-50) + d72*(1-f_l_anchor) + d8*ln_eps_ult
            )
    
        # calculate pipe strain and limit to 1e-5% and 100%
        eps_pipe = np.exp(b0 + b1*(np.log(pgdef)-ln_delta_u) + b2*ln_d_t_ratio + f_soil_type*b3*np.log(t_u))
        eps_pipe = np.maximum(np.minimum(eps_pipe, 100), 1e-5)
    
        # sigma
        # sigma = np.ones(pgdef.shape)*0.723
            
        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': sigma,
                'sigma_mu': np.ones(eps_pipe.shape)*0.3,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['d_t_ratio'] = d_t_ratio
            output['sigma_ult'] = sigma_ult
            output['eps_ult'] = eps_ult
            output['t_u'] = t_u
            output['a0'] = a0
            output['a1'] = a1
            output['a2'] = a2
            output['c3'] = c3
            output['c41'] = c41
            output['c42'] = c42
            output['c51'] = c51
            output['c52'] = c52
            output['b0'] = b0
            output['b2'] = b2
            output['b3'] = b3
            output['d2'] = d2
            output['d3'] = d3
            output['d41'] = d41
            output['d42'] = d42
            output['d5'] = d5
            output['d6'] = d6
            output['d71'] = d71
            output['d72'] = d72
            output['d8'] = d8
            output['f_t_u'] = f_t_u
            output['f_d_t_ratio'] = f_d_t_ratio
            output['d8'] = d8
            output['d8'] = d8
            output['ln_delta_u'] = ln_delta_u
            output['f_delta_f'] = f_delta_f
            output['f_l_anchor'] = f_l_anchor
            output['f_soil_type'] = f_soil_type
            output['b1'] = b1
            
        # return
        return output
    
    
# -----------------------------------------------------------
class HutabaratEtal2022_SSTens(HutabaratEtal2022_SSTens_5to85):
    """
    Model for strike-slip-tension-induced pipe strain using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation - strike-slip-tension-induced
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    h_pipe: float, np.ndarray or list
        [m] burial depth to pipe centerline
    
    Clay:
    s_u_backfill: float, np.ndarray or list
        [kPa] undrained shear strength
    alpha_backfill: float, np.ndarray or list
        adhesion factor
    
    Sand:
    gamma_backfill: float, np.ndarray or list
        [kN/m^3] total unit weight of backfill soil, inferred 
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe : float, np.ndarray
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """
    
    
    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, sigma_y, n_param, r_param, l_anchor, # infrastructure
        beta_crossing, h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        soil_type, steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        #####
        # two separate paths:
        # 1) 10 <= beta <= 85 deg
        # 2) 85 < beta <= 90 deg
        #####
        
        # initialize output variables
        eps_pipe = np.ones(pgdef.shape)*1e-5
        sigma_eps_pipe = np.ones(pgdef.shape)*0.01
        sigma_mu_eps_pipe = np.ones(pgdef.shape)*0.3
        
        # initialize intermediate variable for checking cases
        case_to_run = np.empty(pgdef.shape, dtype="<U20")
         
        # 1) check for cases at head scarp (normal)
        cond = np.logical_and(beta_crossing >= 5, beta_crossing <= 85)
        if True in cond:
            case_to_run[cond] = 'SSTens_5to85'
            output = HutabaratEtal2022_SSTens_5to85._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], steel_grade[cond], # fixed/toggles
            )
            eps_pipe[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe[cond] = output['eps_pipe']['sigma_mu']
            
        # 2) check for cases for strike-slip tension
        cond = np.logical_and(beta_crossing > 85, beta_crossing <= 90)
        if True in cond:
            case_to_run[cond] = 'SSTens_85to90'
            output = HutabaratEtal2022_SSTens_85to90._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], steel_grade[cond], # fixed/toggles
            )
            eps_pipe[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe[cond] = output['eps_pipe']['sigma_mu']
        
        # prepare outputs
        output = {
            'eps_pipe': {
                'mean': eps_pipe,
                'sigma': sigma_eps_pipe,
                # 'sigma_mu': np.ones(pgdef.shape)*0.3,
                'sigma_mu': sigma_mu_eps_pipe,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['case_to_run'] = case_to_run
        
        # return
        return output