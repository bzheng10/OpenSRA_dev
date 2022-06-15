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
class PipeStrain(BaseModel):
    "Inherited class specfic to pipe strain"
    
    _RETURN_PBEE_META = {
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        'type': 'pipe strain',       # Type of model (e.g., liquefaction, landslide, pipe strain)
        'variable': [
            'eps_pipe'
        ] # Return variable for PBEE category, e.g., pgdef, eps_pipe
    }

    def __init__(self):
        super().__init__()
    
    
# -----------------------------------------------------------
class BainEtal2022(PipeStrain):
    """
    Model for transient pipe strain using Bain et al. (2022).
    
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
    sigma_y: float, np.ndarray or list
        [kpa] pipe yield stress
    n_param: float, np.ndarray or list
        Ramberg-Osgood parameter
    r_param: float, np.ndarray or list
        Ramberg-Osgood parameter
        
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
        total unit weight of backfill soil [kN/m^3]
    h_cover: float, np.ndarray or list
        [m] burial depth to pipe centerline
    phi_backfill: float, np.ndarray or list
        [deg] friction angle of backfill
    delta_backfill: float, np.ndarray or list
        sand-pipe interface friction angle ratio
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    welding: boolean, np.ndarray or list
        welded or seamless for SMLS
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_pipe : float
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
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'eps_pipe': {
                'desc': 'longitudinal pipe strain (%)',
                'unit': '%',
                'mean': None,
                'aleatory': 0.45,
                'epistemic': {
                    'coeff': 0.4, # base uncertainty, based on coeffcients
                    'input': None, # sigma_mu uncertainty from input parameters
                    'total': None # SRSS of coeff and input sigma_mu uncertainty
                },
                'dist_type': 'lognormal',
            }
        }
    }
    _INPUT_PBEE_META = {
        'category': 'EDP',        # Input category in PBEE framework, e.g., IM, EDP, DM
        'variable': 'pgdef'        # Input variable for PBEE category, e.g., pgdef, eps_pipe
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
                'mean': None,
                'aleatory': None,
                'epistemic': None,
                'dist_type': 'lognormal'
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
            'h_cover': 'soil cover to centerline of pipeline (m) - for sand',
            'gamma_backfill': 'total unit weight of backfill soil (kN/m^3) - for sand',
            'phi_backfill': 'backfill friction angle (deg) - for sand',
            'delta_backfill': 'sand/pipe interface friction angle ratio - for sand',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': 'soil type (sand/clay) for model',
            'welding': 'welded or seamless (for SMLS)',
            'steel_grade': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'clay': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe', 'sigma_y'],
            'level3': ['d_pipe', 't_pipe', 'sigma_y', 'def_length', 'alpha_backfill', 's_u_backfill'],
        },
        'sand': {
            'level1': [],
            'level2': ['d_pipe', 't_pipe', 'sigma_y'],
            'level3': ['d_pipe', 't_pipe', 'sigma_y', 'def_length', 'h_cover', 'gamma_backfill', 'phi_backfill', 'delta_backfill'],
        }
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'clay': {
            'level1': ['soil_type'],
            'level2': ['soil_type', 'weld_flag', 'steel_grade'],
            'level3': ['soil_type', 'weld_flag', 'steel_grade'],
        },
        'sand': {
            'level1': ['soil_type'],
            'level2': ['soil_type', 'weld_flag', 'steel_grade'],
            'level3': ['soil_type', 'weld_flag', 'steel_grade'],
        }
    }
    _MODEL_INTERNAL = {
        'n_sample': 1,
        'n_site': 1,
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = True
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    # _SUB_CLASS = [
    #     '_BainEtAl2022_Clay', '_BainEtAl2022_Sand'
    # ]


    def __init__(self):
        """create instance"""
        super().__init__()
    
    
    @classmethod
    def get_req_rv_and_fix_params(cls, kwargs):
        """uses soil_type to determine what model parameters to use"""
        soil_type = kwargs.get('soil_type')
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
        alpha_backfill, s_u_backfill, def_length, # clay
        h_cover, gamma_backfill, phi_backfill, delta_backfill, # sand
        soil_type, welding=None, steel_grade=None, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # model coefficients
        # sand
        c0_s =  0.188     # constant
        c1_s =  0.853     # ln(t_pipe)
        c2_s =  0.018     # ln(d_pipe)
        c3_s =  0.751     # ln(sigma_y)
        c4_s = -0.862     # ln(h_cover)
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
        young_mod = 200000000 # kpa, hard-coded now
        # --------------------
        t_u = np.empty_like(pgdef)
        l_e = np.empty_like(pgdef)
        phi_rad = np.radians(phi_backfill)
        pi = np.pi
        d_in = d_pipe - 2*t_pipe # mm
        circum = pi * d_pipe/1000 # m
        area = pi * ((d_pipe/1000)**2 - (d_in/1000)**2) / 4 # m^2

        # calculations
        # find indices with sand and clay
        ind_sand = np.where(soil_type=='sand')[0]
        ind_clay = np.where(soil_type=='clay')[0]
        # for sand
        if len(ind_sand) > 0:
            t_u[ind_sand] = gamma_backfill[ind_sand] * h_cover[ind_sand] \
                            * np.tan(phi_rad[ind_sand]*delta_backfill[ind_sand]) \
                            * circum[ind_sand]
            l_e[ind_sand] = np.exp(
                c0_s    +   c1_s*np.log(t_pipe[ind_sand])           +   c2_s*np.log(d_pipe[ind_sand])
                        +   c3_s*np.log(sigma_y[ind_sand])          +   c4_s*np.log(h_cover[ind_sand])
                        +   c5_s*np.log(gamma_backfill[ind_sand])   +   c6_s*np.log(phi_backfill[ind_sand])
                        +   c7_s*np.log(delta_backfill[ind_sand])   +   c8_s*np.log(pgdef[ind_sand])
            )
        # for clay
        if len(ind_clay) > 0:
            t_u[ind_clay] = alpha_backfill[ind_clay] * s_u_backfill[ind_clay] * circum[ind_clay]
            l_e[ind_clay] = np.exp(
                c0_c    +   c1_c*np.log(t_pipe[ind_clay])       +   c2_c*np.log(sigma_y[ind_clay]) \
                        +   c3_c*np.log(s_u_backfill[ind_clay]) +   c4_c*np.log(alpha_backfill[ind_clay]) \
                        +   c5_c*np.log(pgdef[ind_clay])
            )
        # other calcs
        l_to_use = np.minimum(def_length/2, l_e)
        beta_p = t_u/area
        eps_pipe = beta_p*l_to_use/young_mod * (1 + n_param/(1+r_param)*(beta_p*l_to_use/sigma_y)**r_param) * 100 # %

        # prepare outputs
        output = {'eps_pipe': eps_pipe}
        # get intermediate values if requested
        if return_inter_params:
            output['t_u'] = t_u
            output['l_e'] = l_e
            output['l_to_use'] = l_to_use
            output['beta_p'] = beta_p
        
        # return
        return output


# -----------------------------------------------------------
def ShinozukaKoike1979(eps_g, D, t, E, G, lam, **kwargs):
    """
    Computes the transient pipe strain given transient ground strain using the Shinozuka & Koike (1979) approach, as described in O'Rourke and Liu (2004).
    
    Parameters
    ----------
    eps_g : float
        [%] transient ground strain
    D : float
        [m] outer diameter of pipe
    t : float
        [m] wall thickness of pipe
    E : float
        [kPa] modulus of elasticity of pipe
    G : float
        [kPa] shear modulus
    lam : float
        [m] wavelength
    gamma_cr : float, optional
        [%] critical shear strain before slippage occurs; default = 0.1%
    q : float, optional
        degree of slippage at the pipe-soil interface, ranges from 1 to :math:`\pi`/2; default = :math:`\pi`/2 (slippage over the whole pipe length)
    
    Returns
    -------
    eps_pipe : float
        [%] transient pipe (structural) strain
    
    References
    ----------
    .. [1] Shinozuka, M., and Koike, T., 1979, Estimation of Structural Strain in Underground Lifeline Pipes, Lifeline Earthquake Engineering-Buried Pipelines, Seismic Risk and instrumentation, ASME, New York, pp. 31-48.
    .. [2] O’Rourke, M.J., and Liu, J., 2004, The Seismic Design of Buried and Offshore Pipelines, MCEER Monograph No. 4, 2012, MCEER, University at Buffalo, Buffalo, NY.
    
    """
    
    pass

    # # ground to pipe conversion factor
    # A = np.pi*(D**2 - (D-2*t)**2) # cross-sectional area of pipe
    # Kg = 2*np.pi*G # equivalent spring constant to reflect soil-structural interaction
    # beta_0 = 1/(1 + (2*np.pi/lam)**2 * A*E/Kg) # eq. 10.5
    
    # # shear strain at soil-pipe interface
    # gamma_0 = 2*np.pi/lam*E*t/G*eps_g*beta_0 # eq. 10.6
        
    # # critical shear strain, default = 0.1%
    # # if gamma_0 <= gamma_cr, no slippage
    # # if gamma_0 > gamma_cr, slips
    # gamma_cr = kwargs.get('gamma_cr',0.1)
    
    # # ground to pipe conversion factor, for large ground movement, i.e., gamma_0 > gamma_cr
    # q = kwargs.get('q',np.pi/2)
    # beta_c = gamma_cr/gamma_0*q*beta_0 # eq. 10.8
    
    # # pipe axial strain
    # eps_pipe = beta_c*eps_g # eq. 10.9
        
    # #
    # return eps_pipe
    
    
# -----------------------------------------------------------
def ORourkeElhmadi1988(**kwargs):
    """
    Computes the transient pipe strain given transient ground strain using the O'Rourke & El Hmadi (1988) approach, as described in O'Rourke and Liu (2004).
    
    Parameters
    ----------
    eps_g : float
        [%] transient ground strain
    D : float
        [m] outer diameter of pipe
    t : float
        [m] wall thickness of pipe
    E : float
        [kPa] modulus of elasticity of pipe
    tu : float
        [kPa] maximum frictional resistance at the shear interface
    lam : float
        [m] wavelength
    
    Returns
    -------
    eps_pipe : float
        [%] transient pipe (structural) strain
    
    References
    ----------
    .. [1] O’Rourke, M.J., and El Hmadi, K.E., 1988, Analysis of Continuous Buried Pipelines for Seismic Wave Effects, Earthquake Engineering and Structural Dynamics, v. 16, pp. 917-929.
    .. [2] O’Rourke, M.J., and Liu, J., 2004, The Seismic Design of Buried and Offshore Pipelines, MCEER Monograph No. 4, 2012, MCEER, University at Buffalo, Buffalo, NY.
    
    """
    
    pass

    # # cross-sectional area of pipe
    # A = np.pi*(D**2 - (D-2*t)**2)
    
    # # strain due to friction forces acting over 1/4 of wavelength
    # # controls when ground strain becomes large
    # eps_f = tu*(lam/4)/(A*E) # eq. 10.15
    
    # # pipe axial strain
    # eps_pipe = min(eps_g, eps_f) # eq. 10.16
        
    # #
    # return eps_pipe