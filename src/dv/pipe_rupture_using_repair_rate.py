# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for buckling
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import numpy as np
from scipy.stats import norm, lognorm

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class PipeRuptureUsingRepairRate(BaseModel):
    "Inherited class specfic to pipe rupture"

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class Hazus2020(PipeRuptureUsingRepairRate):
    """
    Compute probability of pipe rupture using Hazus (FEMA, 2020).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Infrastructure:
    
    Fixed:
        
    Returns
    -------
    repair_rate_crit : np.ndarray or list
        [repairs/km] mean critical repair rate
        
    References
    ----------
    .. [1] Hazus (FEMA, 2020)
    
    """

    # class definitions
    NAME = 'Hazus (FEMA, 2020)'   # Name of the model
    ABBREV = None                 # Abbreviated name of the model
    REF = "".join([                 # Reference for the model
        'TBD',
        'Title, ',
        'Publication, ',
        'vol. xx, no. yy, pp. zz-zz.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'repair_rate_crit': {
                'desc': 'mean critical repair rate',
                'unit': 'repairs/km',
            },
        }
    }
    _INPUT_PBEE_META = {
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'DM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {
        }
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}


    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()


    @staticmethod
    # @njit
    def _model(
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # model coefficients

        # calculations
        repair_rate_crit = 0.1 # repairs per km
        
        # prepare outputs
        output = {
            'repair_rate_crit': {
                'mean': repair_rate_crit,
                'sigma': np.sqrt(0.85**2-0.25**2),
                'sigma_mu': 0.25,
                'dist_type': 'lognormal',
                'unit': ''
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output


# -----------------------------------------------------------
def _ALA2001(**kwargs):
    """
    Compute repair rates for buried pipelines induced by both transient and permanent ground deformations.
    - for pipe diameters, small < 12 inches
    
    Parameters
    ----------
    From upstream PBEE:
    pgv: float, np.ndarray or list
        [cm/sec] peak ground velocity
    
    Infrastructure:
        
    Geotechnical/geologic:
    
    Fixed:
    l_segment : float, np.ndarray or list
        [m] pipe segment length
    mat_type: str, np.ndarray or list
        material type: cast-iron, concrete, steel, pre-tcc
    install_year: str, ndarray or list
        year of installation, used for steel pipes only
        
    Returns
    -------
    rr_ave : np.ndarray or list
        average annual number of repairs
    sigma_rr_ave : float, np.ndarray
        aleatory variability
    sigma_mu_rr_ave : float, np.ndarray
        epistemic uncertainty
        
    References
    ----------
    .. [1] ALA (2001)
    
    """

    # class definitions
    NAME = 'ALA (2001)'   # Name of the model
    ABBREV = None                 # Abbreviated name of the model
    REF = "".join([                 # Reference for the model
        'TBD',
        'TBD',
        'TBD',
        'TBD'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'DV',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'rr_ave': {
                'desc': 'average annual number of repairs',
                'unit': '',
            },
        }
    }
    _INPUT_PBEE_META = {
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'pgv': {
                'desc': 'peak ground velocity (cm/s)',
                'unit': 'cm/sec',
            }
        }
    }
    _INPUT_DIST_VARY_WITH_LEVEL = False
    _N_LEVEL = 3
    _MODEL_INPUT_INFRA = {
        "desc": 'Infrastructure random variables:',
        'params': {}
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            'l_segment': 'pipe segment length (m)',
            'mat_type': 'cast-iron, concrete, steel, pre-tcc, default to steel',
            'install_year': 'installation year, used for steel pipes only',
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {}
    _REQ_MODEL_FIXED_FOR_LEVEL = {}
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    

    # instantiation
    def __init__(self):
        """create instance"""
        super().__init__()


    @staticmethod
    # @njit
    def _model(
        pgv, # upstream
        l_segment, mat_type, install_year, joint_type, # fixed/toggles,
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # first get k1 correction for pipe properties
        k1 = np.ones(pgv.shape) # default to 1
        
        # cast-iron pipes
        k1[mat_type=='cast-iron'] = 0.7
        k1[mat_type=='concrete'] = 1.0
        k1[mat_type=='steel'] = 0.3 # for install year between 1950 and 1970
        k1[np.logical_and(mat_type=='steel',install_year<1950)] = 0.6 # if install year < 1950
        k1[np.logical_and(mat_type=='steel',install_year>1970)] = 0.6 # if install year > 1970
        k1[mat_type=='pre-tcc'] = 1.0
        
        # calculate mean annual repair rate
        annual_repair_rate = 0.00
        ann_repair
        
        pass