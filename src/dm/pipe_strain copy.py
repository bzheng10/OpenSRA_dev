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
import logging

# data manipulation modules
import numpy as np
# from numpy import tan, radians, where
# from numba import jit
from numba import njit

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class PipeStrain(BaseModel):
    "Inherited class specfic to pipe strain"
    
    _RETURN_PBEE_META = {
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        'type': 'pipe strain',       # Type of model (e.g., liquefaction, landslide, pipe strain)
        'variable': 'eps_p'        # Return variable for PBEE category, e.g., pgdef, eps_p
    }

    def __init__(self):
        super().__init__()
    
    
# -----------------------------------------------------------
class BainEtAl2022(PipeStrain):
    """
    Base model for transient pipe strain using Bain et al. (2022).
    
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
    n: float, np.ndarray or list
        Ramberg-Osgood parameter
    r: float, np.ndarray or list
        Ramberg-Osgood parameter
        
    Geotechnical:
    def_length: float, np.ndarray or list
        [m] length of ground deformation zone
    s_u: float, np.ndarray or list
        [kPa] (FOR CLAY MODEL) undrained shear strength
    adhesion: float, np.ndarray or list
        adhesion factor
        
    Fixed:
    soil_type: str, np.ndarray or list
        soil type (sand/clay) for model
    weld_flag: boolean, np.ndarray or list
        welded (True/False)
    steel_grade: str, np.ndarray or list
        steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80

    Returns
    -------
    eps_p : float
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Bain, C., and Bray, J.D., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
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
            'eps_p': {
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
        'variable': 'pgdef'        # Input variable for PBEE category, e.g., pgdef, eps_p
    }
    _INPUT_PBEE_RV = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
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
            'd_pipe': {
                'desc': 'pipe diameter (mm)',
                'unit': 'mm',
                'mean': {'level1': 610, 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 25, 'level2': 1, 'level3': 1},
                'low': {'level1': 102, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'high': {'level1': 1067, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'dist_type': 'normal'
            },
            't_pipe': {
                'desc': 'pipe wall thickness (mm)',
                'unit': 'mm',
                'mean': {'level1': 10.2, 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 40, 'level2': 5, 'level3': 5},
                'low': {'level1': 2.5, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'high': {'level1': 20.3, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'dist_type': 'normal'
            },
            'sigma_y': {
                'desc': 'pipe yield stress (kPa)',
                'unit': 'kPa',
                'mean': {'level1': 359000, 'level2': 'user provided', 'level3': 'user provided'},
                'cov': {'level1': 15, 'level2': 7.5, 'level3': 7.5},
                'low': {'level1': 240000, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'high': {'level1': 600000, 'level2': 'depends on mean', 'level3': 'depends on mean'},
                'dist_type': 'normal'
            },
            'n_param': {
                'desc': 'Ramberg-Osgood parameter',
                'unit': 'unitless',
                'mean': 14,
                'sigma': 3,
                'low': 14-2*3,
                'high': 14+2*3,
                'dist_type': 'normal'
            },
            'r_param': {
                'desc': 'Ramberg-Osgood parameter',
                'unit': 'unitless',
                'mean': 8.5,
                'sigma': 1.5,
                'low': 14-2*3,
                'high': 14+2*3,
                'dist_type': 'normal'
            },
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'soil_type': {
                'desc': 'soil type (sand/clay) for model',
                'unit': 'unitless',
            },
            'weld_flag': {
                'desc': 'welded (True/False)',
                'unit': 'unitless',
            },
            'steel_grade': {
                'desc': 'steel grade: Grade-B, X-42, X-52, X-60, X-70, X-80',
                'unit': 'unitless',
            }
        }
    }
    _MODEL_INTERNAL = {
        'n_sample': 1,
        'n_site': 1,
    }
    _SUB_CLASS = [
        '_BainEtAl2022_Clay', '_BainEtAl2022_Sand'
    ]


    def __init__(self):
        """create instance"""
        super().__init__()
    

    def _perform_calc(self):
        """Performs calculations"""
        # pull inputs locally
        n_sample = self._inputs['n_sample']
        n_site = self._inputs['n_site']

        # pgd = self._convert_to_ndarray(self._inputs['pgd'], n_sample)
        # d_pipe = self._convert_to_ndarray(self._inputs['d_pipe'], n_sample)
        # def_length = self._convert_to_ndarray(self._inputs['def_length'], n_sample)
        # t_pipe = self._convert_to_ndarray(self._inputs['t_pipe'], n_sample)
        # sigma_y = self._convert_to_ndarray(self._inputs['sigma_y'], n_sample)
        # soil_type = self._convert_to_ndarray(self._inputs['soil_type'], n_sample)
        # gamma_t = self._convert_to_ndarray(self._inputs['gamma_t'], n_sample)
        # h_cover = self._convert_to_ndarray(self._inputs['h_cover'], n_sample)
        # phi = self._convert_to_ndarray(self._inputs['phi'], n_sample)
        # delta = self._convert_to_ndarray(self._inputs['delta'], n_sample)
        # k0 = self._convert_to_ndarray(self._inputs['k0'], n_sample)
        # adhesion = self._convert_to_ndarray(self._inputs['adhesion'], n_sample)
        # s_u = self._convert_to_ndarray(self._inputs['s_u'], n_sample)
        # n = self._convert_to_ndarray(self._inputs['n'], n_sample)
        # r = self._convert_to_ndarray(self._inputs['r'], n_sample)


        # pgd = self._inputs['pgd']
        # d_pipe = self._inputs['d_pipe']
        # l = self._inputs['l']
        # t_pipe = self._inputs['t_pipe']
        # sigma_y = self._inputs['sigma_y']
        # soil_type = self._inputs['soil_type']
        # gamma_t = self._inputs['gamma_t']
        # d_p = self._inputs['d_p']
        # phi = self._inputs['phi']
        # delta = self._inputs['delta']
        # k0 = self._inputs['k0']
        # adhesion = self._inputs['adhesion']
        # s_u = self._inputs['s_u']
        # n = self._inputs['n']
        # r = self._inputs['r']

        # calculations
        # eps_p = np.asarray([
        #     self._model(pgd[i], l[i], d_pipe[i], t_pipe[i], sigma_y[i], 
        #                 soil_type[i], gamma_t[i], d_p[i], phi[i], delta[i], 
        #                 k0[i], adhesion[i], s_u[i], n[i], r[i]
        #     ) for i in range(n_sample)])

        
        # loop through number of periods
        if njit_on:
            eps_p = self._model(**inputs)
        else:
            eps_p = self._model.py_func(**inputs)            
        
        # store intermediate params
        self._outputs.update({
            'eps_p': eps_p
        })


    @staticmethod
    @njit
    # @jit(nopython=True)
    def _model(
        pgd, d_pipe, t_pipe, sigma_y, n_param, r_param,
        adhesion, s_u, def_length,
        return_inter_param=False
    ):
        """Model"""
        # model coefficients
        # sand
        c0_s =  0.613     # constant
        c1_s =  0.853     # ln(t_pipe)
        c2_s = -0.084     # ln(d_pipe)
        c3_s =  0.751     # ln(sigma_y)
        c4_s = -0.735     # ln(h_cover)
        c5_s = -0.863     # ln(gamma_t)
        c6_s = -1.005     # ln(phi)
        c7_s = -1.000     # ln(delta)
        c8_s =  0.136     # ln(pgd)
        # clay
        c0_c = -4.019     # constant
        c1_c =  0.876     # ln(t_pipe)
        c2_c =  0.787     # ln(sigma_y)
        c3_c = -0.886     # ln(s_u)
        c4_c = -0.889     # ln(adhesion)
        c5_c =  0.114     # ln(pgd)

        # setup
        # --------------------
        young_mod = 200000000 # kpa, hard-coded now
        # --------------------
        t_u = np.empty_like(pgd)
        l_e = np.empty_like(pgd)
        phi_rad = np.radians(phi)
        d_in = d_pipe - 2*t_pipe # mm
        circum = np.pi * d_pipe/1000 # m
        area = np.pi * ((d_pipe/1000)**2 - (d_in/1000)**2) / 4 # m^2

        # calculations
        # find indices with 
        # ind_sand = np.where(soil_type=='sand')[0]
        # ind_clay = np.where(soil_type=='clay')[0]
        # for sand
        t_u[ind_sand] = gamma_t[ind_sand] * (h_cover[ind_sand]+d_pipe[ind_sand]/1000/2) \
                        * (1+k0[ind_sand])/2*np.tan(phi_rad[ind_sand]*delta[ind_sand]) \
                        * circum[ind_sand]
        l_e[ind_sand] = np.exp(
            c0_s    +   c1_s*np.log(t_pipe[ind_sand])        +   c2_s*np.log(d_pipe[ind_sand])
                    +   c3_s*np.log(sigma_y[ind_sand])  +   c4_s*np.log(h_cover[ind_sand])
                    +   c5_s*np.log(gamma_t[ind_sand])  +   c6_s*np.log(phi[ind_sand])
                    +   c7_s*np.log(delta[ind_sand])    +   c8_s*np.log(pgd[ind_sand])
        )
        # for clay
        t_u[ind_clay] = adhesion[ind_clay] * s_u[ind_clay] * circum[ind_clay]
        l_e[ind_clay] = np.exp(
            c0_c    +   c1_c*np.log(t_pipe[ind_clay])        +   c2_c*np.log(sigma_y[ind_clay]) \
                    +   c3_c*np.log(s_u[ind_clay])      +   c4_c*np.log(adhesion[ind_clay]) \
                    +   c5_c*np.log(pgd[ind_clay])
        )
        # other calcs
        l_to_use = np.minimum(def_length/2, l_e)
        beta_p = t_u/area
        eps_p = beta_p*l_to_use/young_mod * (1 + n/(1+r)*(beta_p*l_to_use/sigma_y)**r) * 100 # %

        # single evaluation
        # phi_rad = np.radians(phi)
        # circum = np.pi * d_pipe
        # area = np.pi * d_pipe**2/4 * (1 - (1-2*t_pipe)**2)

        # # calculations
        # # sand vs clay
        # if 'sand' in soil_type.lower():
        #     t_u = gamma_t * (d_p+d_pipe/1000/2) \
        #                     * (1+k0)/2*np.tan(phi_rad*delta) \
        #                     * circum
        # elif 'clay' in soil_type.lower():
        #     t_u = adhesion * s_u * circum
        # # other calcs
        # beta_p = t_u/area
        # l_e = np.exp(c0 + c1*np.log(t_pipe) + c2*np.log(sigma_y) + c3*np.log(s_u) + c4*np.log(adhesion) + 0.114*np.log(pgd))
        # l_to_use = min(l, l_e/2)
        # eps_p = beta_p*l_to_use/2/young_mod * (1 + n/(1+r)*((beta_p*l_to_use/2/sigma_y)**r))

        # return
        return eps_p
    
# -----------------------------------------------------------
class _BainEtAl2022_Clay(BainEtAl2022):
    """
    Model for transient pipe strain in clay using Bain et al. (2022).
    
    Parameters
    ----------
    See BainEtAl2022_Base for base list of parameters
    
    Geotechnical
    def_length: float, np.ndarray or list
        [m] length of ground deformation zone
    s_u: float, np.ndarray or list
        [kPa] (FOR CLAY MODEL) undrained shear strength
    adhesion: float, np.ndarray or list
        adhesion factor

    Returns
    -------
    eps_p : float
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Bain, C., and Bray, J.D., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Bain et al. (2022) - Clay'       # Name of the model
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical random variables:',
        'params': {
            'def_length': {
                'desc': 'length of ground deformation zone (m)',
                'unit': 'm',
                'mean': {'level1': 100, 'level2': 100, 'level3': 'user provided'},
                'cov': {'level1': 90, 'level2': 70, 'level3': 50},
                'low': {'level1': 10, 'level2': 10, 'level3': 10},
                'high': {'level1': 400, 'level2': 400, 'level3': 400},
                'dist_type': 'lognormal'
            },
            'adhesion': {
                'desc': 'adhesion factor for clay',
                'unit': 'unitless',
                'mean': {'level1': 0.75, 'level2': 0.75, 'level3': 'user provided'},
                'sigma': {'level1': 0.14, 'level2': 0.12, 'level3': 0.12},
                'low': {'level1': 0.5, 'level2': 0.5, 'level3': 0.5},
                'high': {'level1': 1, 'level2': 1, 'level3': 1},
                'dist_type': 'normal'
            },
            's_u': {
                'desc': 'undrained shear strength (kPa)',
                'unit': 'kPa',
                'mean': {'level1': 40, 'level2': 35, 'level3': 'user provided'},
                'cov': {'level1': 45, 'level2': 20, 'level3': 5},
                'low': {'level1': 20, 'level2': 20, 'level3': 20},
                'high': {'level1': 120, 'level2': 100, 'level3': 100},
                'dist_type': 'lognormal'
            },
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': ['d_pipe', 't_pipe', 'sigma_y'],
        'level3': ['d_pipe', 't_pipe', 'sigma_y', 'adhesion', 's_u', 'def_length'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': ['soil_type'],
        'level2': ['soil_type', 'weld_flag', 'steel_grade'],
        'level3': ['soil_type', 'weld_flag', 'steel_grade'],
    }

    def __init__(self):
        """create instance"""
        super().__init__()


# -----------------------------------------------------------
class _BainEtAl2022_Sand(BainEtAl2022):
    """
    Model for transient pipe strain in sand using Bain et al. (2022).
    
    Parameters
    ----------
    See BainEtAl2022_Base for base list of parameters
    
    Geotechnical
    def_length: float, np.ndarray or list
        [m] length of ground deformation zon
    gamma_t: float, np.ndarray or list
        total unit weight of backfill soil [kN/m^3]
    h_cover: float, np.ndarray or list
        [m] burial depth to pipe centerline
    phi: float, np.ndarray or list
        [deg] friction angle of backfill
    delta: float, np.ndarray or list
        sand-pipe interface friction angle ratio
    
    """
    
    _NAME = 'Bain et al. (2022) - Sand'       # Name of the model
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical random variables:',
        'params': {
            'def_length': {
                'desc': 'length of ground deformation zone (m)',
                'unit': 'm',
                'mean': {'level1': 100, 'level2': 100, 'level3': 'user provided'},
                'cov': {'level1': 90, 'level2': 70, 'level3': 50},
                'low': {'level1': 10, 'level2': 10, 'level3': 10},
                'high': {'level1': 400, 'level2': 400, 'level3': 400},
                'dist_type': 'lognormal'
            },
            'gamma_t': {
                'desc': 'total unit weight of backfill soil (kN/m^3)',
                'unit': 'kN/m^3',
                'mean': {'level1': 19, 'level2': 19, 'level3': 'user provided'},
                'cov': {'level1': 9, 'level2': 7, 'level3': 5},
                'low': {'level1': 16, 'level2': 16, 'level3': 16},
                'high': {'level1': 21.5, 'level2': 21.5, 'level3': 21.5},
                'dist_type': 'lognormal'
            },
            'h_cover': {
                'desc': 'soil cover to centerline of pipeline (m)',
                'unit': 'm',
                'mean': {'level1': 1.2, 'level2': 1.2, 'level3': 'user provided'},
                'cov': {'level1': 15, 'level2': 15, 'level3': 10},
                'low': {'level1': 0.6, 'level2': 0.6, 'level3': 0.6},
                'high': {'level1': 6, 'level2': 6, 'level3': 6},
                'dist_type': 'lognormal'
            },
            'phi': {
                'desc': 'backfill friction angle (deg)',
                'unit': 'deg',
                'mean': {'level1': 38, 'level2': 38, 'level3': 'user provided'},
                'cov': {'level1': 15, 'level2': 12, 'level3': 9},
                'low': {'level1': 30, 'level2': 30, 'level3': 30},
                'high': {'level1': 45, 'level2': 45, 'level3': 45},
                'dist_type': 'lognormal'
            },
            'delta': {
                'desc': 'sand/pipe interface friction angle ratio',
                'unit': 'unitless',
                'mean': {'level1': 0.75, 'level2': 0.75, 'level3': 'user provided'},
                'sigma': {'level1': 0.14, 'level2': 0.12, 'level3': 0.1},
                'low': {'level1': 0.5, 'level2': 0.5, 'level3': 0.5},
                'high': {'level1': 1, 'level2': 1, 'level3': 1},
                'dist_type': 'normal'
            },
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': [],
        'level2': ['d_pipe', 't_pipe', 'sigma_y'],
        'level3': ['d_pipe', 't_pipe', 'sigma_y', 'gamma_t', 'h_cover', 'def_length', 'phi', 'delta'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': ['soil_type'],
        'level2': ['soil_type', 'weld_flag', 'steel_grade'],
        'level3': ['soil_type', 'weld_flag', 'steel_grade'],
    }
    
    

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
    eps_p : float
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
    # eps_p = beta_c*eps_g # eq. 10.9
        
    # #
    # return eps_p
    
    
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
    eps_p : float
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
    # eps_p = min(eps_g, eps_f) # eq. 10.16
        
    # #
    # return eps_p