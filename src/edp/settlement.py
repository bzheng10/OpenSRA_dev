# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for ground settlement
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import numpy as np
from scipy import sparse

# OpenSRA modules and functions
# import src.EDP.LateralSpread

# OpenSRA modules and classes
from src.base_class import BaseModel

 
# -----------------------------------------------------------
class GroundSettlement(BaseModel):
    "Inherited class specfic to liquefaction-induced ground settlement"
    
    # _RETURN_PBEE_META = {
    #     'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
    #     'type': 'ground settlement',       # Type of model (e.g., liquefaction, landslide, pipe strain)
    #     'variable': [
    #         'pgdef',
    #     ] # Return variable for PBEE category, e.g., pgdef, eps_pipe
    # }

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PGE2022(GroundSettlement):
    """
    Compute volumetric settlement at a given location with the methodology from the PG&E (2022) report.
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    pge_b: float, np.ndarray or list
        [g] coefficent b from pge rasters
    pge_c: float, np.ndarray or list
        [dimensionless] coefficent c from pge rasters
    msf_max: float, np.ndarray or list
        [dimensionless] maximum magnitude scale factor
    ds: float, np.ndarray or list
        [m] settlement coeffient from subsurface conditions
    sigs: float, np.ndarray or list
        [dimensionless] log-normal standard deviation for settlement 

    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
    sigma_pgdef : float, np.ndarray
        aleatory variability for ln(pgdef)
    
    References
    ----------
    .. [1] Pacific Gas & Electric (PG&E), 202, San Francisco Bay Area Liquefaction and Lateral Spread Study, Pacific Gas & Electric (PG&E) Revised Report.

    
    """

    _NAME = 'PG&E (2022)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Pacific Gas & Electric (PG&E), 2023, San Francisco Bay Area Liquefaction and Lateral Spread Study, Pacific Gas & Electric (PG&E) Revised Report.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
                # 'dist_type': 'lognormal',
            },
            # 'sigma_pgdef': {
            #     'desc': 'aleatory variability for ln(pgdef)',
            #     'unit': '',
            #     'mean': None,
            # },
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': [
    #     ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Return category in PBEE framework, e.g., IM, EDP, DM
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
    _INPUT_DIST_VARY_WITH_LEVEL = False
    # _INPUT_DIST_VARY_WITH_LEVEL = True
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
        'pge_b': 'coefficent b from pge rasters (g)',
            'pge_c': 'coefficent c from pge rasters',
            'msf_max': 'maximum magnitude scale factor',
            'ds': 'settlement coeffient from subsurface conditions (m)',
            'sigs': 'log-normal standard deviation for settlement',
            'dl': 'lateral spreading coeffient from subsurface conditions (m)',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        # 'prob_liq'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'pge_b', 'pge_c', 'msf_max', 'ds', 'sigs', 'dl'
        # 'liq_susc'
    }
    # _MODEL_INTERNAL = {
    #     'n_sample': 1,
    #     'n_site': 1,
    # }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        pga, mag, # upstream PBEE RV
        pge_b, pge_c, msf_max, ds, sigs, dl, # geotechnical/geologic
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # initialize arrays
        msf = np.empty(pga.shape)
        med_ln_dl = np.empty(pga.shape)
        
        # magnitude scaling factor value
        msf = 1 + (msf_max - 1)*(8.64 * np.exp(-1 * mag / 4) - 1.325)
        
        # probability of liquefaction
        ds[ds==0] = min(np.min(ds[ds>0]),1e-10) # avoid ds == 0 for log
        dl[dl==0] = min(np.min(dl[dl>0]),1e-10) # avoid dl == 0 for log
        med_ln_ds = np.log(ds / (1 + ((pga / msf) / pge_b) ** pge_c))
        med_ln_dl = np.log(dl / (1 + ((pga / msf) / pge_b) ** pge_c))
        pgdef = np.exp(med_ln_ds) + (0.5*np.exp(med_ln_dl))
        pgdef = np.maximum(pgdef,1e-5)
        
        # catch some isnan sigs
        sigs[np.isnan(sigs)] = 1e-3
        
        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': sigs,
                'sigma_mu': np.ones(med_ln_ds.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'm'
            },
            # 'pgdef': pgdef,
            # 'sigma_pgdef': sigma_pgdef,
            # 'sigma_mu_pgdef': sigma_mu_pgdef,
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output


# -----------------------------------------------------------
class CetinEtal2009(GroundSettlement):
    """
    Compute volumetric strain following the Cetin et al. (2009) probabilistic method.

    Parameters
    ----------
    From upstream PBEE:
    pga : float
        [g] peak ground acceleration
    
    Geotechnical/geologic:
    n1_60_cs : float, np.ndarray or list
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
    sv0_tot : float, np.ndarray or list
        [kPa] initial vertical total stress
    sv0_eff : float, np.ndarray or list
        [kPa] initial vertical effective stress
    r_d : float, np.ndarray or list
        stress reduction factor with depth
    d_r : float, np.ndarray or list
        [%] relative density
    d_h : float, np.ndarray or list
        [m] layer thickness
    prob_liq : float, np.ndarray or list
        [%] probability of liquefaction
    
    Fixed:
    mag : float, np.ndarray or list
        Moment magnitude
    # vs12 : float, optional
    #     [m/s] time-averaged shear wave velocity in the upper 12 meters, default=150 m/s
    # flag_df : boolean, optional
    #     flag to calculate depth-weight factor, to increase settlement contribution from depths less than **z_cr** and reduce contribution from depths greater than **z_cr** (cetin et al., 2009), default: True
    # z_crit : float, optional
    #     [m] critical depth for where depth-weighted factor is equal to 1 (cetin et al., 2009), default: 18 m

    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
    sigma_pgdef : float, np.ndarray
        aleatory variability for ln(pgdef)

    References
    ----------
    .. [1] Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 3, pp. 387-398.

    """
    
    _NAME = 'Cetin et al. (2009)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, ',
        'Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, ',
        'Journal of Geotechnical and Geoenvironmental Engineering, ',
        'vol. 135, no. 3, pp. 387-398.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
                # 'mean': None,
                # 'aleatory': None,
                # 'epistemic': 0.25, # base model uncertainty, does not include input uncertainty,
                # 'dist_type': 'lognormal',
            },
            # 'sigma_pgdef': {
            #     'desc': 'aleatory variability for ln(pgdef)',
            #     'unit': '',
            #     # 'mean': None,
            # },
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': [
    #         'pga'
    #     ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Return category in PBEE framework, e.g., IM, EDP, DM
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
    _INPUT_DIST_VARY_WITH_LEVEL = True
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        "params": {
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {
            'n1_60_cs': 'corrected blow count',
            'sv0_tot': 'initial vertical total stress (kPa)',
            'sv0_eff': 'initial vertical effective stress (kPa)',
            'r_d': 'stress reduction factor with depth',
            'd_r': 'relative density (%)',
            'd_h': 'layer_thickness',
            'prob_liq': 'probability of liquefaction (%)',
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            # 'mag': 'moment magnitude',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': ['n1_60_cs', 'sv0_tot', 'sv0_eff', 'r_d', 'd_r', 'd_h', 'prob_liq'],
        'level2': ['n1_60_cs', 'sv0_tot', 'sv0_eff', 'r_d', 'd_r', 'd_h', 'prob_liq'],
        'level3': ['n1_60_cs', 'sv0_tot', 'sv0_eff', 'r_d', 'd_r', 'd_h', 'prob_liq'],
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': [],
        'level2': [],
        'level3': [],
    }
    # _MODEL_INTERNAL = {
    #     'n_sample': 1,
    #     'n_site': 1,
    # }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    @staticmethod
    # @njit
    def _model(
        pga, mag, # upstream PBEE RV
        n1_60_cs, sv0_tot, sv0_eff, r_d, d_r, d_h, prob_liq, # geotechnical/geologic
        vs12=None, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""

        # Multidirectional shaking effects
        k_md = 0.361*np.log(d_r) - 0.579 # eq. 3

        # Magnitude scaling factor
        k_mag = 87.1/mag**2.217 # eq. 4

        # Overburden correction factor
        f = 1 - 0.005*d_r
        k_sigma = (sv0_eff/101.3)**(f-1) # eq. 5
        
        # calculate depth to bottom and middle of each layer using layer thickness
        z_bot = np.cumsum(d_h)
        z_top = z_bot - d_h
        z_mid = (z_top + z_bot)/2
        h_total = z_bot[-1]
        
        # calculate r_d
        if vs12 is None:
            vs12 = 150 # m/s, 150 to 200, use lower estimate
        # for depth < 20 meters, eq. 8 in Cetin et al. (2004)
        numer = -23.013 - 2.949*pga + 0.999*mag + 0.0525*vs12
        denom_top = 16.258 + 0.201*np.exp(0.341*(-z_mid + 0.0785*vs12 + 7.586))
        denom_bot = 16.258 + 0.201*np.exp(0.341*(0.0785*vs12 + 7.586))
        r_d = (1 + numer/denom_top) / (1 + numer/denom_bot)
        # for depth >= 20 meters
        # ind_z_ge_20 = np.where(z_mid>=20)[0]
        # ind_z_ge_20 = z_mid>=20
        denom_top = 16.258 + 0.201*np.exp(0.341*(-20 + 0.0785*vs12 + 7.586))
        r_d[z_mid>=20] = (1 + numer[z_mid>=20]/denom_top[z_mid>=20]) / \
                         (1 + numer[z_mid>=20]/denom_bot[z_mid>=20])
        
        # cyclic stress ratio (demand)
        csr_field = 0.65 * pga * sv0_tot/sv0_eff * r_d # Seed and Idriss (1971)
        csr = csr_field/k_md/k_mag/k_sigma # eq. 2, CSR corrected for unidirectionality in lab, magnitude, and overburden
        # limit csr to 0.05 to 0.60
        csr = np.minimum(np.maximum(n1_60_cs,0.05),0.60)

        # limit n1_60_cs to 5 and 40
        n1_60_cs = np.minimum(np.maximum(n1_60_cs,5),40)

        # get volumetric strain
        ln_eps_vol = np.log(
            1.879*np.log(
                (780.416*np.log(csr) - n1_60_cs + 2442.465) / \
                (636.613*n1_60_cs + 306.732)
            ) + 5.583
        )
        eps_vol = np.exp(ln_eps_vol) * 100 # %

        # maximum volumetric strain, after Huang (2008)
        eps_vol_max = 9.765 - 2.427*np.log(n1_60_cs) # %
        eps_vol = min(eps_v, eps_v_max) # %
        
        # integrate strain over profile to get settlement
        df = 1-z_mid/1.8 # calculate depth-weighted factor
        eps_vol_eq = sum(eps_vol*d_h*df) / sum(d_h*df) # %
        pgdef = eps_vol_eq * h_total * 100 # m to cm
        
        # correction factor suggested in Peterson (2016)
        pgdef *= pgdef * 1.15
        
        # condition on prob_liq
        pgdef *= prob_liq/100
        
        # convert from cm to m
        pgdef = pgdef/100
        
        # sigma
        # sigma_pgdef = np.ones(pgdef.shape)*0.689
        
        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': np.ones(pgdef.shape)*0.689,
                'sigma_mu': np.ones(pgdef.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'm'
            },
            # 'pgdef': pgdef,
            # 'sigma_pgdef': sigma_pgdef,
        }
        # get intermediate values if requested
        if return_inter_params:
            output['r_d'] = r_d
            output['csr'] = csr
            output['k_md'] = k_md
            output['k_mag'] = k_mag
            output['k_sigma'] = k_sigma
            output['eps_vol'] = eps_vol
            output['eps_vol_max'] = eps_vol_max
            output['df'] = df
        
        # return
        return output


# -----------------------------------------------------------
class Hazus2020(GroundSettlement):
    """
    Compute volumetric settlement at a given location using a simplified deterministic approach (after Tokimatsu and Seed, 1987).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Geotechnical/geologic:
    prob_liq: float, np.ndarray or list
        probability of liquefaction
    
    Fixed:
    liq_susc: str, np.ndarray or list
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
    sigma_pgdef : float, np.ndarray
        aleatory variability for ln(pgdef)
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

    
    """

    _NAME = 'Hazus (FEMA, 2014)'       # Name of the model
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
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
                # 'dist_type': 'lognormal',
            },
            # 'sigma_pgdef': {
            #     'desc': 'aleatory variability for ln(pgdef)',
            #     'unit': '',
            #     'mean': None,
            # },
        }
    }
    # _INPUT_PBEE_META = {
    #     'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
    #     'variable': [
    #     ] # Input variable for PBEE category, e.g., pgdef, eps_pipe
    # }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
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
            'prob_liq': 'probability of liquefaction'
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'liq_susc': 'liquefaction susceptibility category',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
        'level1': ['prob_liq'],
        'level2': ['prob_liq'],
        'level3': ['prob_liq'],
        # 'prob_liq'
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
        'level1': ['liq_susc'],
        'level2': ['liq_susc'],
        'level3': ['liq_susc'],
        # 'liq_susc'
    }
    # _MODEL_INTERNAL = {
    #     'n_sample': 1,
    #     'n_site': 1,
    # }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        prob_liq, # geotechnical/geologic
        liq_susc, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # initialize arrays
        
        # get threshold pga against liquefaction, in cm
        pgdef = np.ones(liq_susc.shape)*np.nan
        pgdef[liq_susc=='very high'] = 30
        pgdef[liq_susc=='high'] = 15
        pgdef[liq_susc=='moderate'] = 5
        pgdef[liq_susc=='low'] = 2.5
        pgdef[liq_susc=='very low'] = 1
        pgdef[liq_susc=='none'] = 1e-3
        
        # condition with prob_liq
        pgdef = pgdef * prob_liq
        
        # convert from cm to m
        pgdef = pgdef/100
        
        # limit deformations to 1e-5
        pgdef = np.maximum(pgdef,1e-5)
        
        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': np.ones(pgdef.shape)*0.8,
                'sigma_mu': np.ones(pgdef.shape)*0.25,
                'dist_type': 'lognormal',
                'unit': 'm'
            },
            # 'pgdef': pgdef,
            # 'sigma_pgdef': sigma_pgdef,
            # 'sigma_mu_pgdef': sigma_mu_pgdef,
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output


# -----------------------------------------------------------
def IdrissBoulanger2008(FS_liq, N1_60_cs=None, qc_1N_cs=None):
    """
    Compute volumetric strain following Idriss & Boulanger (2008); must specify either **N1_60_cs** or **qc_1N_cs**.

    Parameters
    ----------
    FS_liq : float
        factor of safety against liquefaction triggering
    N1_60_cs : float, optional
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
    qc_1N_cs : float, optional
        CPT tip resistance corrected for overburden and fines content

    Returns
    -------
    e_vol : float
        [%] volumetric strain

    References
    ----------
    .. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.

    """
    
    # Use ls.idriss_boulanger_2008_ls to compute shear strain and relative density
    gamma_max, Dr = ls.idriss_boulanger_2008_ls(FS_liq, N1_60_cs, qc_1N_cs)
    
    # Estimate maximum volumetric strain from maximum shear strain
    # eps_v = 1.5*np.exp(-0.369 * N1_60_cs**(0.5)) * min(0.08,gamma_max) * 100 # %, using N1_60_cs
    eps_v = 1.5*np.exp(-2.5 * Dr/100) * min(0.08,gamma_max/100) * 100 # %, using Dr, eq. 95 in Idriss & Boulanger (2008)

    # Correction suggested by Cetin et al. (2009)
    # eps_v = eps_v * 0.9

    #
    return eps_v
        

# -----------------------------------------------------------
def ZhangEtal2002(FS_liq, qc_1N_cs):
    """
    Compute volumetric strain from CPT following the Zhang et al. (2002) method.

    Parameters
    ----------
    FS_liq : float, array
        factor of safety against liquefaction triggering
    qc_1N_cs : float, array
        CPT tip resistance corrected for overburden and fines content

    Returns
    -------
    eps_v : float
        [%] volumetric strain

    References
    ----------
    .. [1] Zhang, G., Robertson, P.K., and Brachman, R.W.I., 2002, Estimating liquefaction-induced ground settlements from CPT for level ground, Canadian Geotechnical Journal, vol. 39, no. 5, pp. 1168-1180.

    """

    #
    # if qc_1N_cs is None or FS_liq is None:
        #
        # logging.info(f"Must specify qc_1N_cs and FS_liq to proceed with method")
        
        #
        # return None
        
    # else:
    # Check if params are arrays and convert if not
    if not isinstance(FS_liq,np.ndarray):
        FS_liq = np.asarray([FS_liq])
        
    if not isinstance(qc_1N_cs,np.ndarray):
        qc_1N_cs = np.asarray([qc_1N_cs])
        
    # Since relationships in Zhang et al. (2002) for discrete Dr, first compute strain at bounds, then interpolate
    # 10 discrete FS values where relationships are given, 11 bins total (e.g., bin 1 = FS <= 0.5, bin 11 = FS >= 2):
    FS_bin_range = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2])
    
    # Bound applicable range for FS_liq and qc_1N_cs
    FS_liq = np.maximum(np.minimum(FS_liq,2),0.5)
    qc_1N_cs = np.maximum(np.minimum(qc_1N_cs,200),33)
    
    # Determine FS_liq upper and lower bin values
    bin_arr = np.digitize(FS_liq,FS_bin_range)
    
    bin_arr_high = bin_arr
    bin_arr_high[bin_arr_high>len(FS_bin_range)-1] = len(FS_bin_range)-1 # upper limit = number of bin range
    bin_arr_low = bin_arr-1
    bin_arr_low[bin_arr_low<0] = 0 # lower limit = 0
    
    FS_liq_high = FS_bin_range[bin_arr_high]
    FS_liq_low = FS_bin_range[bin_arr_low]
    
    # Appendix A in Zhang et al. (2002) provides relationships for volumetric strain in the form of:
    # --- eps_v = c1 * (qc_1N_cs) ^ c2, where c1 and c2 are dependent on FS_liq and qc_1N_cs
    # Determine coeffcients c1 and c2 at the FS_liq upper and lower bounds
    # Start with FS >= 2, no volumetric strain
    c1_high = np.zeros(FS_liq.shape)
    c2_high = np.zeros(FS_liq.shape)
    c1_low = np.zeros(FS_liq.shape)
    c2_low = np.zeros(FS_liq.shape)
    
    # FS_liq >= 1.3 and < 2
    c1_low[FS_liq < 2] = 7.6
    c2_low[FS_liq < 2] = -0.71
    
    # FS_liq >= 1.2 and < 1.3
    c1_high[FS_liq < 1.3] = 7.6
    c2_high[FS_liq < 1.3] = -0.71
    c1_low[FS_liq < 1.3] = 9.7
    c2_low[FS_liq < 1.3] = -0.69
    
    # FS_liq >= 1.1 and < 1.2
    c1_high[FS_liq < 1.2] = 9.7
    c2_high[FS_liq < 1.2] = -0.69
    c1_low[FS_liq < 1.2] = 11
    c2_low[FS_liq < 1.2] = -0.65
    
    # FS_liq >= 1.0 and < 1.1
    c1_high[FS_liq < 1.1] = 11
    c2_high[FS_liq < 1.1] = -0.65
    c1_low[FS_liq < 1.1] = 64
    c2_low[FS_liq < 1.1] = -0.93
    
    # FS_liq >= 0.9 and < 1.0
    c1_high[FS_liq < 1.0] = 64
    c2_high[FS_liq < 1.0] = -0.93
    c1_low[FS_liq < 1.0] = 1430
    c2_low[FS_liq < 1.0] = -1.48
    
    # FS_liq >= 0.8 and < 0.9
    c1_high[FS_liq < 0.9] = 1430
    c2_high[FS_liq < 0.9] = -1.48
    c1_low[FS_liq < 0.9] = 1690
    c2_low[FS_liq < 0.9] = -1.46
    
    # FS_liq >= 0.7 and < 0.8
    c1_high[FS_liq < 0.8] = 1690
    c2_high[FS_liq < 0.8] = -1.46
    c1_low[FS_liq < 0.8] = 1701
    c2_low[FS_liq < 0.8] = -1.42
    
    # FS_liq >= 0.6 and < 0.7
    c1_high[FS_liq < 0.7] = 1701
    c2_high[FS_liq < 0.7] = -1.42
    c1_low[FS_liq < 0.7] = 2411
    c2_low[FS_liq < 0.7] = -1.45
    
    # FS_liq > 0.5 and < 0.6
    c1_high[FS_liq < 0.6] = 2411
    c2_high[FS_liq < 0.6] = -1.45
    c1_low[FS_liq < 0.6] = 102
    c2_low[FS_liq < 0.6] = -0.82
    
    # FS_liq <= 0.5
    c1_high[FS_liq <= 0.5] = 102
    c2_high[FS_liq <= 0.5] = -0.82
    
    # Compute volumetric strain at bounds
    eps_v_high = c1_high*qc_1N_cs**c2_high
    eps_v_low = c1_low*qc_1N_cs**c2_low
    
    # Apply limiting strain
    eps_v_high = np.minimum(eps_v_high,102*qc_1N_cs**(-0.82))
    eps_v_low = np.minimum(eps_v_low,102*qc_1N_cs,33**(-0.82))
    
    # Interpolate for volumetric strain at FS_liq
    eps_v = eps_v_low + (eps_v_high - eps_v_low)*(FS_liq - FS_liq_low)/(FS_liq_high - FS_liq_low)
    
    #
    return eps_v
        

# -----------------------------------------------------------
def CetinEtal2009_superseded(**kwargs):
    """
    Compute volumetric strain following the Cetin et al. (2009) probabilistic method.

    Parameters
    ----------
    N1_60_cs : float
        [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [kPa] initial vertical total stress
    sv0_eff : float
        [kPa] initial vertical effective stress
    rd : float
        stress reduction factor with depth
    Dr : float
        [%] relative density
    M : float
        Moment magnitude
    patm : float, optional
        [kPa] atmospheric pressure; **default = 101.3 kPa**

    Returns
    -------
    eps_v : float
        [%] volumetric strain

    References
    ----------
    .. [1] Cetin, K.O., Bilge, H.T., Wu, J., Kammerer, A.M., and Seed, R.B., 2009, Probabilistic Model for the Assessment of Cyclically Induced Reconsolidation (Volumetric) Settlements, Journal of Geotechnical and Geoenvironmental Engineering, vol. 135, no. 3, pp. 387-398.

    """

    ############ Inputs ############
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    Dr = kwargs.get('Dr',None) # %, relative density
    patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    M = kwargs.get('M',None) # moment magnitude

    # Multidirectional shaking effects
    if Dr is None:
        K_md = 1.0
    else:
        K_md = 0.361*np.log(Dr) - 0.579 # eq. 3

    # Magnitude scaling factor
    if M is None:
        K_M = 1.0
    else:
        K_M = 87.1/M**2.217 # eq. 4

    # Overburden correction factor
    if Dr is None:
        K_sigma = 1.0 # eq. 31
    else:
        f = 1 - 0.005*Dr # factor for K_sigma, eq. 5
        K_sigma = (sv0_eff/patm)**(f-1) # eq. 5

    if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
        csr = None
        print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
    else:
        # Cyclic stress ratio (demand)
        csr_field = 0.65 * amax * sv0_tot/sv0_eff * rd # Seed and Idriss (1971)
        csr = csr_field/K_md/K_M/K_sigma # eq. 2, CSR corrected for unidirectionality in lab, magnitude, and overburden

    #
    if N1_60_cs < 5 or N1_60_cs > 40:
        print('N1_60_cs is outside the range of 5 to 40 given by Cetin et al. (2009)')
    if csr < 5 or csr > 40:
        print('CSR_SS_20_1D_1_atm is outside the range of 0.05 to 0.60 given by Cetin et al. (2009)')

    #
    ln_eps_v = np.log(1.879*np.log((780.416*np.log(csr) - N1_60_cs + 2442.465)/(636.613*N1_60_cs + 306.732)) + 5.583)
    eps_v = np.exp(ln_eps_v) * 100 # %
    sigma = 0.689

    # maximum volumetric strain, after Huang (2008)
    eps_v_max = 9.765 - 2.427*np.log(N1_60_cs) # %

    # volumetric strain as the minimum of the correlated and maximum values
    eps_v = min(eps_v, eps_v_max) # %

    #
    eps_v = eps_v * 1.15 # correction factor suggested in Peterson (2016)

    #
    return eps_v


# -----------------------------------------------------------
def Hazus2014_superseded(**kwargs):
    """
    Compute volumetric settlement at a given location using a simplified deterministic approach (after Tokimatsu and Seed, 1987).

    Parameters
    ----------
    liq_susc : str
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    pgd_gs : float
        [cm] permanent ground deformation (settlement)

    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Buildin Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
    .. [2] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

    """

    # get inputs
    liq_susc = kwargs.get('liq_susc',None) # liquefaction susceptibility category
    # n_site = kwargs.get('n_site',len(liq_susc)) # number of sites
    n_event = kwargs.get('n_event',1) # number of events
    
    # get threshold pga against liquefaction
    pgd_gs = np.zeros(liq_susc.shape)
    pgd_gs[liq_susc=='very high'] = 12 # inches
    pgd_gs[liq_susc=='high'] = 6 # inches
    pgd_gs[liq_susc=='moderate'] = 2 # inches
    pgd_gs[liq_susc=='low'] = 1 # inches
    pgd_gs[liq_susc=='very low'] = 0 # inches
    pgd_gs[liq_susc=='none'] = 0 # inches

    # convert to cm and repeat for number for events
    pgd_gs = np.tile(pgd_gs*2.54,[n_event,1])

    # convert to coo_matrix
    pgd_gs = sparse.coo_matrix(pgd_gs)
    
    # probability distribution
    prob_dist_type = 'Uniform'
    factor_aleatory = 3
    factor_epistemic = 4

    # store outputs
    output = {}
    output.update({'pgd_gs': pgd_gs})
    output.update({'prob_dist': {'type': prob_dist_type,
                                'factor_aleatory': factor_aleatory,
                                'factor_epistemic': factor_epistemic}})

    #
    return output


# # -----------------------------------------------------------
# def IshiharaYoshimine1992(FS_liq, N1_60_cs=None, qc_1N_cs=None):
#     """
#     Compute volumetric strain following Ishihara and Yoshimine (1992); must specify either **N1_60_cs** or **qc_1N_cs**.

#     Parameters
#     ----------
#     FS_liq : float
#         factor of safety against liquefaction triggering
#     N1_60_cs : float, optional
#         [blows/ft] SPT blow count corrected for energy and equipment, overburden, and fines content; used to estimate :math:`Dr = \sqrt{(N_{1-60})_{cs}/46}`
#     qc_1N_cs : float, optional
#         CPT tip resistance corrected for overburden and fines content; used to estimate :math:`Dr = \sqrt{(N_{1-60})_{cs}/46}`

#     Returns
#     -------
#     eps_v : float
#         [%] volumetric strain

#     References
#     ----------
#     .. [1] Ishihara, K., and Yoshimine, M., 1992, Evaluation of Settlements in Sand Deposits Following Liquefaction During Earthquakes, Soils and Foundations, vol. 32, no. 1, pp. 173-188.
#     .. [2] Franke, K.W., Ekstrom, L.T., Ulmer, K.J., Astorga, L., and Error, B., 2016, Simplified Standard Penetration Test Performance-Based Assessment of Liquefaction and Effects, Brigham Young University, Report No. UT-16.16, Provo, UT, USA.
#     .. [3] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.

#     """

#     #
#     # eps_v = idriss_boulanger_2008_gs(FS_liq, N1_60_cs, qc_1N_cs)
    
#     #
#     # return eps_v


# # -----------------------------------------------------------
# def TokimatsuSeed1987(**kwargs):
#     """
#     Compute volumetric strain following the Tokimatsu and Seed (1987) deterministic method.

#     Parameters
#     ----------
#     TBD : float
#         TBD

#     Returns
#     -------
#     eps_v : float
#         [%] volumetric strain

#     References
#     ----------
#     .. [1] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

#     """

#     print('Placeholder - under development')

#     return None