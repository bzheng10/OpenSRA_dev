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
from numpy import cos, sin, radians, logical_and, logical_or
# from numba import jit
# from numba import njit

# OpenSRA modules and classes
from src.base_class import BaseModel
from src.dm._pipe_strain_base_models import HutabaratEtal2022_Normal, HutabaratEtal2022_Reverse
from src.dm._pipe_strain_base_models import HutabaratEtal2022_SSComp, HutabaratEtal2022_SSTens


# -----------------------------------------------------------
class SurfaceFaultRuptureInducedPipeStrain(BaseModel):
    "Inherited class specfic to surface-fault-rupture-induced pipe strain"

    def __init__(self):
        super().__init__()
    
    
# -----------------------------------------------------------
class HutabaratEtal2022(SurfaceFaultRuptureInducedPipeStrain):
    """
    Model for landslide-induced pipe strain using Hutabarat et al. (2022).
    
    Parameters
    ----------
    From upstream PBEE:
    pgdef: float, np.ndarray or list
        [m] permanent ground deformation
    
    Infrastructure:
    d_pipe: float, np.ndarray or list
        [mm] pipe outside diameter
    t_pipe: float, np.ndarray or list
        [mm] pipe wall thickness
    l_anchor: float, np.ndarray or list
        [m] pipeline anchored length
        
    Geotechnical/geologic:
    General:
    beta_crossing: float, np.ndarray or list
        [deg] pipe-fault crossing angle
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
        [kN/m^3] total unit weight of backfill soil
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    soil_density: str, np.ndarray or list
        soil density: soft, medium stiff, or stiff for clay; medium dense, dense, or very dense for sand
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe_comp : float, np.ndarray
        [%] longitudinal pipe strain in compression
    eps_pipe_tens : float, np.ndarray
        [%] longitudinal pipe strain in tension
    
    References
    ----------
    .. [1] Hutabarat, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Hutabarat et al. (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Authors, Year, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe_comp': {
                'desc': 'longitudinal pipe strain in compression (%)',
                'unit': '%',
            },
            'eps_pipe_tens': {
                'desc': 'longitudinal pipe strain in tension (%)',
                'unit': '%',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
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
            'sigma_y': 'pipe yield stress (kPa) - Bain et al. (2022) only',
            'n_param': 'Ramberg-Osgood parameter - Bain et al. (2022) only',
            'r_param': 'Ramberg-Osgood parameter - Bain et al. (2022) only',
            'l_anchor': 'pipe wall thickness (m) - Hutabarat et al. (2022) only',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'beta_crossing': 'pipe-fault crossing angle [deg]',
            'theta_slip': 'slip direction (azimuth) [deg]',
            'psi_dip': 'pipe-fault dip angle [deg]',
            'h_pipe': 'burial depth to pipe centerline (m)',
            'def_length': 'length of ground deformation zone (m) - Bain et al. (2022) only',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand - Bain et al. (2022) only',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': 'soil type (sand/clay) for model',
            'soil_density': 'soil density: soft, medium stiff, or stiff for clay; medium dense, dense, or very dense for sand',
            'steel_grade': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'clay': ['d_pipe', 't_pipe', 'sigma_y', 'h_pipe', 'alpha_backfill', 's_u_backfill'],
        'sand': ['d_pipe', 't_pipe', 'sigma_y', 'h_pipe', 'gamma_backfill', 'phi_backfill', 'delta_backfill'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'clay': ['soil_type', 'soil_density'],
        'sand': ['soil_type', 'soil_density'],
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
        if 'sand' in soil_type:
            soils.append('sand')
        if 'clay' in soil_type:
            soils.append('clay')
        for i in range(3):
            for each in soils:
                if f'level{i+1}' in req_rvs_by_level:
                    req_rvs_by_level[f'level{i+1}'] += cls._REQ_MODEL_RV_FOR_LEVEL[each]
                    req_fixed_by_level[f'level{i+1}'] += cls._REQ_MODEL_FIXED_FOR_LEVEL[each]
                else:
                    req_rvs_by_level[f'level{i+1}'] = cls._REQ_MODEL_RV_FOR_LEVEL[each]
                    req_fixed_by_level[f'level{i+1}'] = cls._REQ_MODEL_FIXED_FOR_LEVEL[each]
            req_rvs_by_level[f'level{i+1}'] = sorted(list(set(req_rvs_by_level[f'level{i+1}'])))
            req_fixed_by_level[f'level{i+1}'] = sorted(list(set(req_fixed_by_level[f'level{i+1}'])))
        return req_rvs_by_level, req_fixed_by_level


    @staticmethod
    # @njit
    def _model(
        pgdef, # upstream PBEE RV
        d_pipe, t_pipe, sigma_y, n_param, r_param, l_anchor, # infrastructure
        beta_crossing, theta_slip, psi_dip, h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        soil_type, soil_density, steel_grade, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        #####
        # one of the ten options:
        # 
        # beta_crossing <= 90
        # 1)        theta_slip = -180   strike-slip compression
        # 2) -180 < theta_slip < -90    strike-slip compression and normal-slip
        # 3)        theta_slip = -90                                normal-slip
        # 4)  -90 < theta_slip < 0      strike-slip tension     and normal-slip
        # 5)        theta_slip = 0      strike-slip tension
        # 
        # beta_crossing > 90
        # 6)        theta_slip = 0      strike-slip compression
        # 7)    0 < theta_slip < 90     strike-slip compression and reverse-slip
        # 8)        theta_slip = 90                                 reverse-slip
        # 9)   90 < theta_slip < 180    strike-slip tension     and reverse-slip
        # 10)       theta_slip = 180    strike-slip tension
        #####
        
        # initialize output variables
        eps_pipe_comp = np.ones(pgdef.shape)*1e-5
        eps_pipe_tens = np.ones(pgdef.shape)*1e-5
        sigma_eps_pipe_comp = np.ones(pgdef.shape)*0.01
        sigma_eps_pipe_tens = np.ones(pgdef.shape)*0.01
        sigma_mu_eps_pipe_comp = np.ones(pgdef.shape)*0.3
        sigma_mu_eps_pipe_tens = np.ones(pgdef.shape)*0.3
        
        # initialize intermediate variable for checking cases
        case_to_run = np.empty(pgdef.shape, dtype="<U20")
         
        ###################################################
        # for all cases where beta_crossing <= 90
        cond1 = beta_crossing <= 90
        
        # 1)        theta_slip = -180   strike-slip compression
        cond2 = theta_slip == -180
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSComp'
            ##################
            # prep
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSComp
            pgdef_to_use = pgdef_cond
            output = HutabaratEtal2022_SSComp._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                beta_cond, # geotech - general
            )
            ##################
            eps_pipe_comp[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_comp[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_comp[cond] = output['eps_pipe']['sigma_mu']
        
        # 2) -180 < theta_slip < -90    strike-slip compression and normal-slip
        cond2 = logical_and(theta_slip>-180, theta_slip<-90)
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSComp_and_Normal'
            ##################
            # prep
            theta_cond = theta_slip[comb_cond].copy()
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSComp
            pgdef_to_use = pgdef_cond * np.abs(cos(radians(theta_cond)))
            output1 = HutabaratEtal2022_SSComp._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                beta_cond, # geotech - general
            )
            # Normal
            pgdef_to_use = pgdef_cond * np.abs(sin(radians(theta_cond)) / cos(radians(90-beta_cond)))
            output2 = HutabaratEtal2022_Normal._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], soil_density[cond], steel_grade[cond], # fixed/toggles
            )
            # post - pick worst case, likely comp
            eps_pipe_comp_curr = np.ones(pgdef.shape)*1e-5
            eps_pipe_tens_curr = np.ones(pgdef.shape)*1e-5
            sigma_eps_pipe_comp_curr = np.ones(pgdef.shape)*0.01
            sigma_eps_pipe_tens_curr = np.ones(pgdef.shape)*0.01
            sigma_mu_eps_pipe_comp_curr = np.ones(pgdef.shape)*0.3
            sigma_mu_eps_pipe_tens_curr = np.ones(pgdef.shape)*0.3
            # where SSComp > Normal
            ind_SSComp_gt_Normal = output1['eps_pipe']['mean'] > output2['eps_pipe']['mean']
            eps_pipe_comp_curr[ind_SSComp_gt_Normal] = output1['eps_pipe']['mean'][ind_SSComp_gt_Normal]
            sigma_eps_pipe_comp_curr[ind_SSComp_gt_Normal] = output1['eps_pipe']['sigma'][ind_SSComp_gt_Normal]
            sigma_mu_eps_pipe_comp_curr[ind_SSComp_gt_Normal] = output1['eps_pipe']['sigma_mu'][ind_SSComp_gt_Normal]
            # where Normal > SSComp
            ind_Normal_gt_SSComp = output2['eps_pipe']['mean'] > output1['eps_pipe']['mean']
            eps_pipe_tens_curr[ind_Normal_gt_SSComp] = output2['eps_pipe']['mean'][ind_Normal_gt_SSComp]
            sigma_eps_pipe_tens_curr[ind_Normal_gt_SSComp] = output2['eps_pipe']['sigma'][ind_Normal_gt_SSComp]
            sigma_mu_eps_pipe_tens_curr[ind_Normal_gt_SSComp] = output2['eps_pipe']['sigma_mu'][ind_Normal_gt_SSComp]
            ##################
            eps_pipe_comp[cond] = eps_pipe_comp_curr
            sigma_eps_pipe_comp[cond] = sigma_eps_pipe_comp_curr
            sigma_mu_eps_pipe_comp[cond] = sigma_mu_eps_pipe_comp_curr
            eps_pipe_tens[cond] = eps_pipe_tens_curr
            sigma_eps_pipe_tens[cond] = sigma_eps_pipe_tens_curr
            sigma_mu_eps_pipe_tens[cond] = sigma_mu_eps_pipe_tens_curr
        
        # 3)        theta_slip = -90                                normal-slip
        cond2 = theta_slip == -90
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'Normal'
            ##################
            # prep
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSTens
            pgdef_to_use = pgdef_cond / np.abs(cos(radians(90-beta_cond)))
            output = HutabaratEtal2022_SSTens._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], sigma_y[comb_cond], n_param[comb_cond], r_param[comb_cond], l_anchor[comb_cond], # infrastructure
                h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], steel_grade[comb_cond], # fixed/toggles
            )
            ##################
            eps_pipe_tens[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_tens[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_tens[cond] = output['eps_pipe']['sigma_mu']
        
        # 4)  -90 < theta_slip < 0      strike-slip tension     and normal-slip
        cond2 = logical_and(theta_slip>-90, theta_slip<0)
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSTens_and_Normal'
            ##################
            # prep
            theta_cond = theta_slip[comb_cond].copy()
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSTens
            pgdef_to_use = pgdef_cond * np.abs(cos(radians(theta_cond)))
            output1 = HutabaratEtal2022_SSTens._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], sigma_y[comb_cond], n_param[comb_cond], r_param[comb_cond], l_anchor[comb_cond], # infrastructure
                h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], steel_grade[comb_cond], # fixed/toggles
            )
            # Normal
            pgdef_to_use = pgdef_cond * np.abs(sin(radians(theta_cond)) / cos(radians(90-beta_cond)))
            output2 = HutabaratEtal2022_Normal._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], soil_density[cond], steel_grade[cond], # fixed/toggles
            )
            # post - SRSS tension and normal
            eps_pipe_tens_curr = (output1['eps_pipe']['mean']**2 + output2['eps_pipe']['mean']**2)**0.5
            sigma_eps_pipe_tens_curr = ((output1['eps_pipe']['sigma']**2 + output2['eps_pipe']['sigma']**2)/2)**0.5
            sigma_mu_eps_pipe_tens_curr = ((output1['eps_pipe']['sigma_mu']**2 + output2['eps_pipe']['sigma_mu']**2)/2)**0.5
            ##################
            eps_pipe_tens[cond] = eps_pipe_tens_curr
            sigma_eps_pipe_tens[cond] = sigma_eps_pipe_tens_curr
            sigma_mu_eps_pipe_tens[cond] = sigma_mu_eps_pipe_tens_curr

        # 5)        theta_slip = 0      strike-slip tension
        cond2 = theta_slip == -180
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSTens'
            ##################
            # prep
            pgdef_cond = pgdef[comb_cond].copy()
            # SSTens
            pgdef_to_use = pgdef_cond
            output = HutabaratEtal2022_SSTens._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], sigma_y[comb_cond], n_param[comb_cond], r_param[comb_cond], l_anchor[comb_cond], # infrastructure
                h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], steel_grade[comb_cond], # fixed/toggles
            )
            ##################
            eps_pipe_tens[comb_cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_tens[comb_cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_tens[comb_cond] = output['eps_pipe']['sigma_mu']
        
        ###################################################
        # for all cases where beta_crossing > 90
        cond1 = beta_crossing > 90
        
        # 6)        theta_slip = 0      strike-slip compression
        cond2 = theta_slip == 0
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSComp'
            ##################
            # prep
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSComp
            pgdef_to_use = pgdef_cond
            output = HutabaratEtal2022_SSComp._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                beta_cond, # geotech - general
            )
            ##################
            eps_pipe_comp[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_comp[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_comp[cond] = output['eps_pipe']['sigma_mu']
        
        # 7)    0 < theta_slip < 90     strike-slip compression and reverse-slip
        cond2 = logical_and(theta_slip>0, theta_slip<90)
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSComp_and_Reverse'
            ##################
            # prep
            theta_cond = theta_slip[comb_cond].copy()
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSComp
            pgdef_to_use = pgdef_cond * np.abs(cos(radians(theta_cond)))
            output1 = HutabaratEtal2022_SSComp._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                beta_cond, # geotech - general
            )
            # Reverse
            pgdef_to_use = pgdef_cond * np.abs(sin(radians(theta_cond)) / cos(radians(90-beta_cond)))
            output2 = HutabaratEtal2022_Reverse._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                psi_dip[comb_cond], h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], # fixed/toggles
            )
            # post - SRSS comp and reverse
            eps_pipe_comp_curr = (output1['eps_pipe']['mean']**2 + output2['eps_pipe']['mean']**2)**0.5
            sigma_eps_pipe_comp_curr = ((output1['eps_pipe']['sigma']**2 + output2['eps_pipe']['sigma']**2)/2)**0.5
            sigma_mu_eps_pipe_comp_curr = ((output1['eps_pipe']['sigma_mu']**2 + output2['eps_pipe']['sigma_mu']**2)/2)**0.5
            ##################
            eps_pipe_comp[cond] = eps_pipe_comp_curr
            sigma_eps_pipe_comp[cond] = sigma_eps_pipe_comp_curr
            sigma_mu_eps_pipe_comp[cond] = sigma_mu_eps_pipe_comp_curr
        
        # 8)        theta_slip = 90                                 reverse-slip
        cond2 = theta_slip == 90
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'Reverse'
            ##################
            # prep
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # Reverse
            pgdef_to_use = pgdef_cond / np.abs(cos(radians(90-beta_cond)))
            output = HutabaratEtal2022_Reverse._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                psi_dip[comb_cond], h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], # fixed/toggles
            )
            ##################
            eps_pipe_comp[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_comp[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_comp[cond] = output['eps_pipe']['sigma_mu']
        
        # 9)   90 < theta_slip < 180    strike-slip tension     and reverse-slip
        cond2 = logical_and(theta_slip>90, theta_slip<180)
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSTens_and_Reverse'
            ##################
            # prep
            theta_cond = theta_slip[comb_cond].copy()
            beta_cond = beta_crossing[comb_cond].copy()
            pgdef_cond = pgdef[comb_cond].copy()
            # SSTens
            pgdef_to_use = pgdef_cond * np.abs(cos(radians(theta_cond-90)))
            output1 = HutabaratEtal2022_SSTens._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], sigma_y[comb_cond], n_param[comb_cond], r_param[comb_cond], l_anchor[comb_cond], # infrastructure
                h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], steel_grade[comb_cond], # fixed/toggles
            )
            # Reverse
            pgdef_to_use = pgdef_cond * np.abs(sin(radians(theta_cond)) / cos(radians(90-beta_cond)))
            output2 = HutabaratEtal2022_Reverse._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], l_anchor[comb_cond], # infrastructure
                psi_dip[comb_cond], h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], # fixed/toggles
            )
            # post - pick worst case, likely reverse
            eps_pipe_comp_curr = np.ones(pgdef.shape)*1e-5
            eps_pipe_tens_curr = np.ones(pgdef.shape)*1e-5
            sigma_eps_pipe_comp_curr = np.ones(pgdef.shape)*0.01
            sigma_eps_pipe_tens_curr = np.ones(pgdef.shape)*0.01
            sigma_mu_eps_pipe_comp_curr = np.ones(pgdef.shape)*0.3
            sigma_mu_eps_pipe_tens_curr = np.ones(pgdef.shape)*0.3
            # where SSComp > Normal
            ind_SSTens_gt_Reverse = output1['eps_pipe']['mean'] > output2['eps_pipe']['mean']
            eps_pipe_tens_curr[ind_SSTens_gt_Reverse] = output2['eps_pipe']['mean'][ind_SSTens_gt_Reverse]
            sigma_eps_pipe_tens_curr[ind_SSTens_gt_Reverse] = output2['eps_pipe']['sigma'][ind_SSTens_gt_Reverse]
            sigma_mu_eps_pipe_tens_curr[ind_SSTens_gt_Reverse] = output2['eps_pipe']['sigma_mu'][ind_SSTens_gt_Reverse]
            # where Normal > SSComp
            ind_Reverse_gt_SSTens = output2['eps_pipe']['mean'] > output1['eps_pipe']['mean']
            eps_pipe_comp_curr[ind_Reverse_gt_SSTens] = output1['eps_pipe']['mean'][ind_Reverse_gt_SSTens]
            sigma_eps_pipe_comp_curr[ind_Reverse_gt_SSTens] = output1['eps_pipe']['sigma'][ind_Reverse_gt_SSTens]
            sigma_mu_eps_pipe_comp_curr[ind_Reverse_gt_SSTens] = output1['eps_pipe']['sigma_mu'][ind_Reverse_gt_SSTens]
            ##################
            eps_pipe_comp[cond] = eps_pipe_comp_curr
            sigma_eps_pipe_comp[cond] = sigma_eps_pipe_comp_curr
            sigma_mu_eps_pipe_comp[cond] = sigma_mu_eps_pipe_comp_curr
            eps_pipe_tens[cond] = eps_pipe_tens_curr
            sigma_eps_pipe_tens[cond] = sigma_eps_pipe_tens_curr
            sigma_mu_eps_pipe_tens[cond] = sigma_mu_eps_pipe_tens_curr
        
        # 10)       theta_slip = 180    strike-slip tension
        cond2 = theta_slip == 180
        comb_cond = logical_and(cond1,cond2)
        if True in comb_cond:
            case_to_run[comb_cond] = 'SSTens'
            ##################
            # prep
            pgdef_cond = pgdef[comb_cond].copy()
            # SSTens
            pgdef_to_use = pgdef_cond
            output = HutabaratEtal2022_SSTens._model(
                pgdef_to_use, # upstream PBEE RV
                d_pipe[comb_cond], t_pipe[comb_cond], sigma_y[comb_cond], n_param[comb_cond], r_param[comb_cond], l_anchor[comb_cond], # infrastructure
                h_pipe[comb_cond], # geotech - general
                alpha_backfill[comb_cond], s_u_backfill[comb_cond], # clay
                gamma_backfill[comb_cond], phi_backfill[comb_cond], delta_backfill[comb_cond], # sand
                soil_type[comb_cond], steel_grade[comb_cond], # fixed/toggles
            )
            ##################
            eps_pipe_tens[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_tens[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_tens[cond] = output['eps_pipe']['sigma_mu']
        
        # prepare outputs
        output = {
            'eps_pipe_comp': {
                'mean': eps_pipe_comp,
                'sigma': sigma_eps_pipe_comp,
                # 'sigma_mu': np.ones(pgdef.shape)*0.3,
                'sigma_mu': sigma_mu_eps_pipe_comp,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            'eps_pipe_tens': {
                'mean': eps_pipe_tens,
                'sigma': sigma_eps_pipe_tens,
                # 'sigma_mu': np.ones(pgdef.shape)*0.3,
                'sigma_mu': sigma_mu_eps_pipe_tens,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['case_to_run'] = case_to_run
        
        # return
        return output