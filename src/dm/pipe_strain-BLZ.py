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
from numba import njit

# OpenSRA modules and classes
from src.baseclass import BaseClass


# -----------------------------------------------------------
class PipeStrain(BaseClass):
    "Inherited class specfic to transient pipe strain"

    TYPE = 'DM'

    def __init__(self):
        super().__init__()
        

# -----------------------------------------------------------
class BainEtAl2022(PipeStrain):
    """
    Compute transient pipe strain using Bain et al. (2022).
    
    Parameters
    ----------
    pgd: float, np.ndarray or list
        [cm] permanent ground deformation
    d: float, np.ndarray or list
        [mm] pipe outside diameter
    t: float, np.ndarray or list
        [mm] pipe wall thickness
    sigma_y: float, np.ndarray or list
        [kpa] pipe yield stress
    soil_type, str, np.ndarray or list
        soil type: **sand** or **clay**
    l_def: float, np.ndarray or list
        [m] length of ground deformation zone

    Parameters for **clay model**
    s_u: float, np.ndarray or list
        [kPa] (FOR CLAY MODEL) undrained shear strength
    alpha: float, np.ndarray or list
        adhesion factor
    
    Parameters for **sand model**
    gamma_t: float, np.ndarray or list
        total unit weight of backfill soil [kN/m3]
    h_cover: float, np.ndarray or list
        [m] burial depth to pipe centerline
    phi: float, np.ndarray or list
        [deg] friction angle of backfill
    delta: float, np.ndarray or list
        sand-pipe interface friction angle ratio
    k0: float, np.ndarray or list, optional
        at-rest earth pressure coefficient; default = **1** (common for pipelines)
    
    Additional fitting parameters for **both models**
    n: float, np.ndarray or list, optional
        Ramberg-Osgood parameter; default = **14**
    r: float, np.ndarray or list, optional
        Ramberg-Osgood parameter; default = **8.5**

    Returns
    -------
    eps_p : float
        [%] longitudinal pipe strain
    
    References
    ----------
    .. [1] Bain, C., and Bray, J.D., 2022, Title, Publication, vol. xx, no. yy, pp. zz-zz.
    
    """

    _NAME = 'Bain et al. (2022)'       # Name of the model
    _ABBREV = 'BB22'                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Bain, C., Bray, J.D., and co., 2022, ',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _MODEL_DIST = {                            # Distribution information
        'type': 'lognormal',
        'mean': None,
        'aleatory': 0.45,
        'epistemic': {
            'coeff': 0.4, # base uncertainty, based on coeffcients
            'input': None, # epistemic uncertainty from input parameters
            'total': None # SRSS of coeff and input epistemic uncertainty
        }
    }
    _REQ_PBEE_CAT = 'EDP'     # Upstream PBEE variable required by model, e.g, IM, EDP, DM
    _REQ_PBEE_RV = 'pgdef'     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
    _MODEL_INPUT_INFRA = {                           # Model inputs (required and optional)
        'required': [                   # List of required inputs
            'pgdef',
            'l_def',
            'd',
            't',
            'sigma_y',
            'soil_type'
        ],
        'OPTIONAL': {                   # Optional inputs with default values
            'n_sample': 1,
            'gamma_t': None,
            'h_cover': None,
            'phi': None,
            'delta': 0.8,
            'k0': 1,
            'alpha': 0.7,
            's_u': None,
            'n': 14,
            'r': 8.5
        }
    }
    OUTPUT = [                          # List of available outputs
        'eps_p'
    ]


    # instantiation
    def __init__(self):
        super().__init__()


    # update calculation method
    def _perform_calc(self):
        """Performs calculations"""
        # pull inputs locally
        n_sample = self._inputs['n_sample']
        n_site = self._inputs['n_site']

        pgd = self._convert_to_ndarray(self._inputs['pgd'], n_sample)
        d = self._convert_to_ndarray(self._inputs['d'], n_sample)
        l_def = self._convert_to_ndarray(self._inputs['l_def'], n_sample)
        t = self._convert_to_ndarray(self._inputs['t'], n_sample)
        sigma_y = self._convert_to_ndarray(self._inputs['sigma_y'], n_sample)
        soil_type = self._convert_to_ndarray(self._inputs['soil_type'], n_sample)
        gamma_t = self._convert_to_ndarray(self._inputs['gamma_t'], n_sample)
        h_cover = self._convert_to_ndarray(self._inputs['h_cover'], n_sample)
        phi = self._convert_to_ndarray(self._inputs['phi'], n_sample)
        delta = self._convert_to_ndarray(self._inputs['delta'], n_sample)
        k0 = self._convert_to_ndarray(self._inputs['k0'], n_sample)
        alpha = self._convert_to_ndarray(self._inputs['alpha'], n_sample)
        s_u = self._convert_to_ndarray(self._inputs['s_u'], n_sample)
        n = self._convert_to_ndarray(self._inputs['n'], n_sample)
        r = self._convert_to_ndarray(self._inputs['r'], n_sample)


        # pgd = self._inputs['pgd']
        # d = self._inputs['d']
        # l = self._inputs['l']
        # t = self._inputs['t']
        # sigma_y = self._inputs['sigma_y']
        # soil_type = self._inputs['soil_type']
        # gamma_t = self._inputs['gamma_t']
        # d_p = self._inputs['d_p']
        # phi = self._inputs['phi']
        # delta = self._inputs['delta']
        # k0 = self._inputs['k0']
        # alpha = self._inputs['alpha']
        # s_u = self._inputs['s_u']
        # n = self._inputs['n']
        # r = self._inputs['r']

        # calculations
        # eps_p = np.asarray([
        #     self._model(pgd[i], l[i], d[i], t[i], sigma_y[i], 
        #                 soil_type[i], gamma_t[i], d_p[i], phi[i], delta[i], 
        #                 k0[i], alpha[i], s_u[i], n[i], r[i]
        #     ) for i in range(n_sample)])

        
        eps_p = self._model(
            pgd, l_def, d, t, sigma_y, 
            soil_type, gamma_t, h_cover, phi, delta, 
            k0, alpha, s_u, n, r
        )
        
        # store intermediate params
        self._inters.update({
            'eps_p': eps_p
        })


    @staticmethod
    @njit
    # @jit(nopython=True)
    def _model(pgd, l_def, d, t, sigma_y, soil_type, gamma_t, h_cover, phi, delta, k0, alpha, s_u, n, r):
        """Model"""
        # model coefficients
        # sand
        c0_s =  0.613     # constant
        c1_s =  0.853     # ln(t)
        c2_s = -0.084     # ln(d)
        c3_s =  0.751     # ln(sigma_y)
        c4_s = -0.735     # ln(h_cover)
        c5_s = -0.863     # ln(gamma_t)
        c6_s = -1.005     # ln(phi)
        c7_s = -1.000     # ln(delta)
        c8_s =  0.136     # ln(pgd)
        # clay
        c0_c = -4.019     # constant
        c1_c =  0.876     # ln(t)
        c2_c =  0.787     # ln(sigma_y)
        c3_c = -0.886     # ln(s_u)
        c4_c = -0.889     # ln(alpha)
        c5_c =  0.114     # ln(pgd)

        # setup
        # --------------------
        young_mod = 200000000 # kpa, hard-coded now
        # --------------------
        t_u = np.empty_like(pgd)
        l_e = np.empty_like(pgd)
        phi_rad = np.radians(phi)
        d_in = d - 2*t # mm
        circum = np.pi * d/1000 # m
        area = np.pi * ((d/1000)**2 - (d_in/1000)**2) / 4 # m^2

        # calculations
        # find indices with 
        ind_sand = np.where(soil_type=='sand')[0]
        ind_clay = np.where(soil_type=='clay')[0]
        # for sand
        t_u[ind_sand] = gamma_t[ind_sand] * (h_cover[ind_sand]+d[ind_sand]/1000/2) \
                        * (1+k0[ind_sand])/2*np.tan(phi_rad[ind_sand]*delta[ind_sand]) \
                        * circum[ind_sand]
        l_e[ind_sand] = np.exp(
            c0_s    +   c1_s*np.log(t[ind_sand])        +   c2_s*np.log(d[ind_sand])
                    +   c3_s*np.log(sigma_y[ind_sand])  +   c4_s*np.log(h_cover[ind_sand])
                    +   c5_s*np.log(gamma_t[ind_sand])  +   c6_s*np.log(phi[ind_sand])
                    +   c7_s*np.log(delta[ind_sand])    +   c8_s*np.log(pgd[ind_sand])
        )
        # for clay
        t_u[ind_clay] = alpha[ind_clay] * s_u[ind_clay] * circum[ind_clay]
        l_e[ind_clay] = np.exp(
            c0_c    +   c1_c*np.log(t[ind_clay])        +   c2_c*np.log(sigma_y[ind_clay]) \
                    +   c3_c*np.log(s_u[ind_clay])      +   c4_c*np.log(alpha[ind_clay]) \
                    +   c5_c*np.log(pgd[ind_clay])
        )
        # other calcs
        l_to_use = np.minimum(l_def/2, l_e)
        beta_p = t_u/area
        eps_p = beta_p*l_to_use/young_mod * (1 + n/(1+r)*(beta_p*l_to_use/sigma_y)**r) * 100 # %

        # single evaluation
        # phi_rad = np.radians(phi)
        # circum = np.pi * d
        # area = np.pi * d**2/4 * (1 - (1-2*t)**2)

        # # calculations
        # # sand vs clay
        # if 'sand' in soil_type.lower():
        #     t_u = gamma_t * (d_p+d/1000/2) \
        #                     * (1+k0)/2*np.tan(phi_rad*delta) \
        #                     * circum
        # elif 'clay' in soil_type.lower():
        #     t_u = alpha * s_u * circum
        # # other calcs
        # beta_p = t_u/area
        # l_e = np.exp(c0 + c1*np.log(t) + c2*np.log(sigma_y) + c3*np.log(s_u) + c4*np.log(alpha) + 0.114*np.log(pgd))
        # l_to_use = min(l, l_e/2)
        # eps_p = beta_p*l_to_use/2/young_mod * (1 + n/(1+r)*((beta_p*l_to_use/2/sigma_y)**r))

        # return
        return eps_p


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