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
from src.dm._pipe_strain_base import HutabaratEtal2022_Normal, HutabaratEtal2022_Reverse
from src.dm._pipe_strain_base import HutabaratEtal2022_SSComp, HutabaratEtal2022_SSTens


# -----------------------------------------------------------
class LandslideInducedPipeStrain(BaseModel):
    "Inherited class specfic to lateral-spread-induced pipe strain"

    def __init__(self):
        super().__init__()
    
    
# -----------------------------------------------------------
class HutabaratEtal2022(LandslideInducedPipeStrain):
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
        soil density: medium dense, dense, or very dense for sand
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
    .. [1] Bain, C., Bray, J.D., and co., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Bain et al. (2022) and Hutabarat et al. (2022)'       # Name of the model
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
            'sigma_y': 'pipe yield stress (kPa)',
            'n_param': 'Ramberg-Osgood parameter',
            'r_param': 'Ramberg-Osgood parameter',
            'l_anchor': 'pipeline anchored length (m)',
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'beta_crossing': 'pipe-fault crossing angle [deg]',
            'psi_dip': 'pipe-fault dip angle [deg]',
            'h_pipe': 'burial depth to pipe centerline (m)',
            'alpha_backfill': 'adhesion factor for clay - for clay',
            's_u_backfill': 'undrained shear strength (kPa) - for clay',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
            'primary_mech': 'mechanisms for slip (normal, reverse, and/or SS compression, tension)',
            'transition_weight_factor': 'for oblique slips, linear weight from 0 to 1 with beta_crossing',
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
        beta_crossing, psi_dip, h_pipe, # geotech - general
        alpha_backfill, s_u_backfill, # clay
        gamma_backfill, phi_backfill, delta_backfill, # sand
        soil_type, soil_density, steel_grade, # fixed/toggles,
        primary_mech, transition_weight_factor, # additional fixed params determined internally
        return_inter_params=False # to get intermediate params
    ):
        #####
        # one of the 8 options:
        # 1) normal only:
        # 2) reverse only:
        # 3) strike-slip tension only:
        # 4) strike-slip compression only:
        # 5) normal and strike-slip tension:
        # 6) normal and strike-slip compression:
        # 7) reverse and strike-slip tension:
        # 8) reverse and strike-slip compression:
        #####
        
        # for PGD less than 5 cm or 0.05 m, set to 0
        pgdef[pgdef<0.05] = 0
        
        # initialize output variables
        eps_pipe_comp = np.ones(pgdef.shape)*1e-5
        eps_pipe_tens = np.ones(pgdef.shape)*1e-5
        sigma_eps_pipe_comp = np.ones(pgdef.shape)*0.01
        sigma_eps_pipe_tens = np.ones(pgdef.shape)*0.01
        sigma_mu_eps_pipe_comp = np.ones(pgdef.shape)*0.3
        sigma_mu_eps_pipe_tens = np.ones(pgdef.shape)*0.3
        
        # 1) normal only:
        cond = primary_mech == 'Normal'
        if True in cond:
            output = HutabaratEtal2022_Normal._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], soil_density[cond], steel_grade[cond], # fixed/toggles
            )
            eps_pipe_tens[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_tens[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_tens[cond] = output['eps_pipe']['sigma_mu']
            
        # 2) reverse only:
        cond = primary_mech == 'Reverse'
        if True in cond:
            output = HutabaratEtal2022_Reverse._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], # fixed/toggles
            )
            eps_pipe_comp[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_comp[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_comp[cond] = output['eps_pipe']['sigma_mu']
            
        # 3) strike-slip tension only:
        cond = primary_mech == 'SSTens'
        if True in cond:
            output = HutabaratEtal2022_SSTens._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], steel_grade[cond], # fixed/toggles
            )
            eps_pipe_tens[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_tens[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_tens[cond] = output['eps_pipe']['sigma_mu']
            
        # 4) strike-slip compression only:
        cond = primary_mech == 'SSComp'
        if True in cond:
            output = HutabaratEtal2022_SSComp._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], # geotech - general
            )
            eps_pipe_comp[cond] = output['eps_pipe']['mean']
            sigma_eps_pipe_comp[cond] = output['eps_pipe']['sigma']
            sigma_mu_eps_pipe_comp[cond] = output['eps_pipe']['sigma_mu']
        
        # 5) normal and strike-slip tension:
        cond = primary_mech == 'Normal_SSTens'
        if True in cond:
            ##################
            # Normal
            output1 = HutabaratEtal2022_Normal._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], soil_density[cond], steel_grade[cond], # fixed/toggles
            )
            # SSTens
            output2 = HutabaratEtal2022_SSTens._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], steel_grade[cond], # fixed/toggles
            )
            # post - linearly weight normal and ss_tens strains and variance
            weight_cond = transition_weight_factor[cond]
            inv_weight_cond = 1 - weight_cond
            eps_pipe_tens_curr = \
                output1['eps_pipe']['mean']*weight_cond + \
                output2['eps_pipe']['mean']*inv_weight_cond
            sigma_eps_pipe_tens_curr = ((
                output1['eps_pipe']['sigma']**2*weight_cond + \
                output2['eps_pipe']['sigma']**2*inv_weight_cond
            ))**0.5
            sigma_mu_eps_pipe_tens_curr = ((
                output1['eps_pipe']['sigma_mu']**2*weight_cond +\
                output2['eps_pipe']['sigma_mu']**2*inv_weight_cond
            ))**0.5
            ##################
            eps_pipe_tens[cond] = eps_pipe_tens_curr
            sigma_eps_pipe_tens[cond] = sigma_eps_pipe_tens_curr
            sigma_mu_eps_pipe_tens[cond] = sigma_mu_eps_pipe_tens_curr
        
        # 6) normal and strike-slip compression:
        cond = primary_mech == 'Normal_SSComp'
        if True in cond:
            subset_shape = pgdef[cond].shape
            ##################
            # Normal
            output1 = HutabaratEtal2022_Normal._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], soil_density[cond], steel_grade[cond], # fixed/toggles
            )
            # SSComp
            output2 = HutabaratEtal2022_SSComp._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], # geotech - general
            )
            # post - pick worst case, likely comp
            eps_pipe_comp_curr = np.ones(subset_shape)*1e-5
            eps_pipe_tens_curr = np.ones(subset_shape)*1e-5
            sigma_eps_pipe_comp_curr = np.ones(subset_shape)*0.01
            sigma_eps_pipe_tens_curr = np.ones(subset_shape)*0.01
            sigma_mu_eps_pipe_comp_curr = np.ones(subset_shape)*0.3
            sigma_mu_eps_pipe_tens_curr = np.ones(subset_shape)*0.3
            # where Normal > SSComp
            ind_1_gt_2 = output1['eps_pipe']['mean'] >= output2['eps_pipe']['mean']
            eps_pipe_tens_curr[ind_1_gt_2] = output1['eps_pipe']['mean'][ind_1_gt_2]
            sigma_eps_pipe_tens_curr[ind_1_gt_2] = output1['eps_pipe']['sigma'][ind_1_gt_2]
            sigma_mu_eps_pipe_tens_curr[ind_1_gt_2] = output1['eps_pipe']['sigma_mu'][ind_1_gt_2]
            # where SSComp > Normal
            ind_2_gt_1 = output2['eps_pipe']['mean'] > output1['eps_pipe']['mean']
            eps_pipe_comp_curr[ind_2_gt_1] = output2['eps_pipe']['mean'][ind_2_gt_1]
            sigma_eps_pipe_comp_curr[ind_2_gt_1] = output2['eps_pipe']['sigma'][ind_2_gt_1]
            sigma_mu_eps_pipe_comp_curr[ind_2_gt_1] = output2['eps_pipe']['sigma_mu'][ind_2_gt_1]
            ##################
            eps_pipe_comp[cond] = eps_pipe_comp_curr
            sigma_eps_pipe_comp[cond] = sigma_eps_pipe_comp_curr
            sigma_mu_eps_pipe_comp[cond] = sigma_mu_eps_pipe_comp_curr
            eps_pipe_tens[cond] = eps_pipe_tens_curr
            sigma_eps_pipe_tens[cond] = sigma_eps_pipe_tens_curr
            sigma_mu_eps_pipe_tens[cond] = sigma_mu_eps_pipe_tens_curr
        
        # 7) reverse and strike-slip tension:
        cond = primary_mech == 'Reverse_SSTens'
        if True in cond:
            subset_shape = pgdef[cond].shape
            ##################
            # Reverse
            output1 = HutabaratEtal2022_Reverse._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], # fixed/toggles
            )
            # SSTens
            output2 = HutabaratEtal2022_SSTens._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], sigma_y[cond], n_param[cond], r_param[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], steel_grade[cond], # fixed/toggles
            )
            # post - pick worst case, likely reverse
            eps_pipe_comp_curr = np.ones(subset_shape)*1e-5
            eps_pipe_tens_curr = np.ones(subset_shape)*1e-5
            sigma_eps_pipe_comp_curr = np.ones(subset_shape)*0.01
            sigma_eps_pipe_tens_curr = np.ones(subset_shape)*0.01
            sigma_mu_eps_pipe_comp_curr = np.ones(subset_shape)*0.3
            sigma_mu_eps_pipe_tens_curr = np.ones(subset_shape)*0.3
            # where Reverse > SSTens
            ind_1_gt_2 = output1['eps_pipe']['mean'] >= output2['eps_pipe']['mean']
            eps_pipe_comp_curr[ind_1_gt_2] = output1['eps_pipe']['mean'][ind_1_gt_2]
            sigma_eps_pipe_comp_curr[ind_1_gt_2] = output1['eps_pipe']['sigma'][ind_1_gt_2]
            sigma_mu_eps_pipe_comp_curr[ind_1_gt_2] = output1['eps_pipe']['sigma_mu'][ind_1_gt_2]
            # where SSTens > Reverse
            ind_2_gt_1 = output2['eps_pipe']['mean'] > output1['eps_pipe']['mean']
            eps_pipe_tens_curr[ind_2_gt_1] = output2['eps_pipe']['mean'][ind_2_gt_1]
            sigma_eps_pipe_tens_curr[ind_2_gt_1] = output2['eps_pipe']['sigma'][ind_2_gt_1]
            sigma_mu_eps_pipe_tens_curr[ind_2_gt_1] = output2['eps_pipe']['sigma_mu'][ind_2_gt_1]
            ##################
            eps_pipe_comp[cond] = eps_pipe_comp_curr
            sigma_eps_pipe_comp[cond] = sigma_eps_pipe_comp_curr
            sigma_mu_eps_pipe_comp[cond] = sigma_mu_eps_pipe_comp_curr
            eps_pipe_tens[cond] = eps_pipe_tens_curr
            sigma_eps_pipe_tens[cond] = sigma_eps_pipe_tens_curr
            sigma_mu_eps_pipe_tens[cond] = sigma_mu_eps_pipe_tens_curr
        
        # 8) reverse and strike-slip compression:
        cond = primary_mech == 'Reverse_SSComp'
        if True in cond:
            ##################
            # Reverse
            output1 = HutabaratEtal2022_Reverse._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], t_pipe[cond], l_anchor[cond], # infrastructure
                psi_dip[cond], h_pipe[cond], # geotech - general
                alpha_backfill[cond], s_u_backfill[cond], # clay
                gamma_backfill[cond], phi_backfill[cond], delta_backfill[cond], # sand
                soil_type[cond], # fixed/toggles
            )
            # SSComp
            output2 = HutabaratEtal2022_SSComp._model(
                pgdef[cond], # upstream PBEE RV
                d_pipe[cond], l_anchor[cond], # infrastructure
                beta_crossing[cond], # geotech - general
            )
            # post - linearly weight reverse and ss_comp strains and variance
            weight_cond = transition_weight_factor[cond]
            inv_weight_cond = 1 - weight_cond
            eps_pipe_comp_curr = \
                output1['eps_pipe']['mean']*weight_cond + \
                output2['eps_pipe']['mean']*inv_weight_cond
            sigma_eps_pipe_comp_curr = ((
                output1['eps_pipe']['sigma']**2*weight_cond + \
                output2['eps_pipe']['sigma']**2*inv_weight_cond
            ))**0.5
            sigma_mu_eps_pipe_comp_curr = ((
                output1['eps_pipe']['sigma_mu']**2*weight_cond +\
                output2['eps_pipe']['sigma_mu']**2*inv_weight_cond
            ))**0.5
            ##################
            eps_pipe_comp[cond] = eps_pipe_comp_curr
            sigma_eps_pipe_comp[cond] = sigma_eps_pipe_comp_curr
            sigma_mu_eps_pipe_comp[cond] = sigma_mu_eps_pipe_comp_curr
        
        # where_nan = np.isnan(eps_pipe_tens)
        # if True in where_nan:
        #     print(primary_mech[where_nan])
        #     print(transition_weight_factor[where_nan])
        #     print(pgdef[where_nan])
        #     print(d_pipe[where_nan])
        #     print(t_pipe[where_nan])
        #     print(sigma_y[where_nan])
        #     print(n_param[where_nan])
        #     print(r_param[where_nan])
        #     print(l_anchor[where_nan])
        #     print(beta_crossing[where_nan])
        #     print(h_pipe[where_nan])
        #     print(alpha_backfill[where_nan])
        #     print(s_u_backfill[where_nan])
        #     print(gamma_backfill[where_nan])
        #     print(phi_backfill[where_nan])
        #     print(delta_backfill[where_nan])
        #     print(soil_type[where_nan])
        #     print(steel_grade[where_nan])
        
        # prepare outputs
        output = {
            'eps_pipe_comp': {
                'mean': eps_pipe_comp,
                'sigma': sigma_eps_pipe_comp,
                'sigma_mu': sigma_mu_eps_pipe_comp,
                'dist_type': 'lognormal',
                'unit': '%'
            },
            'eps_pipe_tens': {
                'mean': eps_pipe_tens,
                'sigma': sigma_eps_pipe_tens,
                'sigma_mu': sigma_mu_eps_pipe_tens,
                'dist_type': 'lognormal',
                'unit': '%'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output