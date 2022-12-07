# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for surface fault rupture displacements
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import numpy as np
from scipy import sparse

# OpenSRA modules and classes
from src.base_class import BaseModel


# -----------------------------------------------------------
class SurfaceFaultRupture(BaseModel):
    "Inherited class specfic to surface fault rupture"

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class PetersenEtal2011(SurfaceFaultRupture):
    """
    Compute surface fault rupture displacement using Petersen et al. (2021).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Geotechnical/geologic:
    
    Fixed:
    
    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
        
    References
    ----------
    .. [1] Petersen et al. (2011) - To be completed.
    
    """

    _NAME = 'Petersen et al. (2011)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Petersen, et al., 2011, ',
        'XX, ',
        'YY, ',
        'ZZ.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
            }
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
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
        "params": {
            'norm_dist': {
                'desc': 'normalized distance of crossing from fault end (max=0.5)',
                'unit': '',
            },
            'f_r': {
                'desc': 'faulting frequency (nonzero for secondary hazard), recommended by LCI',
                'unit': '',
            },
            'f_ds': {
                'desc': 'displacement scale factor (nonzero for secondary hazard), recommended by LCI',
                'unit': '',
            },
        }
    }
    _MODEL_INPUT_GEO = {
        "desc": 'Geotechnical/geologic random variables:',
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {}
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        mag, norm_dist, f_r, f_ds, # upstream PBEE RV
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # disp magnitude by Petersen et al. (2011)
        x_star = (1 - ((norm_dist-0.5)/0.5)**2)**0.5
        # calculate deformation
        ln_pgdef = 1.7927*mag + 3.3041*x_star -11.2192 # ln(cm)
        
        # calculate probability of surface rupture
        term = np.exp(-12.51+2.053*mag)
        prob_surf_rup = term / (1+term)
        
        # apply prob and scale factors as recommended by LCI
        # - prob of surface rupture
        # - faulting frequency prob
        # - displacement scale factor
        ln_pgdef = ln_pgdef * prob_surf_rup * f_r * f_ds # applied to ln(d)
        
        # convert to meters and limit to 1e-5 m to avoid ln(0)
        pgdef = np.maximum(np.exp(ln_pgdef)/100, 1e-5) # m

        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': np.ones(pgdef.shape)*0.498,
                'sigma_mu': np.ones(pgdef.shape)*1.0197,
                'dist_type': 'lognormal',
                'unit': 'm'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['x_star'] = x_star
            output['prob_surf_rup'] = prob_surf_rup
        
        # return
        return output


# -----------------------------------------------------------
class WellsCoppersmith1994(SurfaceFaultRupture):
    """
    Compute surface fault rupture displacement using Wells & Coppersmith (1994).
    
    Parameters
    ----------
    From upstream PBEE:
    mag: float, np.ndarray or list
        moment magnitude
    rake : float, np.ndarray or list
        [deg] rake angle, used to infer fault mechanism: strike-slip, reverse, normal; if not given, default to all
    
    Geotechnical/geologic:
    
    Fixed:
    flag_crossing : boolean, np.adarray or list
        fault crossing with component: True/False
    disp_model : str
        displacement model: "maximum" or "average" displacements; default="maximum"
    
    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
        
    References
    ----------
    .. [1] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
    
    """

    _NAME = 'Wells and Coppersmith (1994)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Wells, D.L., and Coppersmith, K.J., 1994, ',
        'New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, ',
        'Bulletin of the Seismological Society of America, ',
        'vol. 84, no. 4, pp. 974-1002.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
            'mag': {
                'desc': 'moment magnitude',
                'unit': '',
            },
            'rake': {
                'desc': 'rake angle (deg)',
                'unit': 'deg',
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
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {
            # 'flag_crossing'
        }
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        mag, rake, # upstream PBEE RV
        disp_model='maximum', # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # determine indices for strike-slip, reverse, normal, or "all"
        ind_rake_nm = np.where(np.logical_and(rake>=-150,rake<=-30))[0] # normal
        ind_rake_rv = np.where(np.logical_and(rake>=30,rake<=150))[0] # reverse
        ind_rake_ss = np.where(
            np.logical_or(
                np.logical_or(
                    np.logical_and(rake>=-180,rake<-150),
                    np.logical_and(rake>150,rake<=180)
                ),
                np.logical_and(rake>-30,rake<30)
            )
        )[0] # strike_slip
        set_all = set(np.arange(len(rake)))
        set_nm_rv_ss = set(ind_rake_nm).union(set(ind_rake_rv),set(ind_rake_ss))
        ind_rake_all = list(set_all.difference(set_nm_rv_ss))
        
        # initialize array
        a = np.empty(rake.shape)
        b = np.empty(rake.shape)
        sigma = np.empty(rake.shape)
        
        # Wells & Coppersmith (1994) coefficients
        if disp_model.lower() == 'maximum':
            # normal
            a[ind_rake_nm] = -5.90
            b[ind_rake_nm] = 0.89
            sigma[ind_rake_nm] = 0.38
            # reverse
            a[ind_rake_rv] = -1.84
            b[ind_rake_rv] = 0.29
            sigma[ind_rake_rv] = 0.42
            # strike-slip
            a[ind_rake_ss] = -7.03
            b[ind_rake_ss] = 1.03
            sigma[ind_rake_ss] = 0.34
            # all
            a[ind_rake_all] = -5.46
            b[ind_rake_all] = 0.82
            sigma[ind_rake_all] = 0.42
        elif disp_model.lower() == 'average':
            # normal
            a[ind_rake_nm] = -4.45
            b[ind_rake_nm] = 0.63
            sigma[ind_rake_nm] = 0.33
            # reverse
            a[ind_rake_rv] = -0.74
            b[ind_rake_rv] = 0.08
            sigma[ind_rake_rv] = 0.38
            # strike-slip
            a[ind_rake_ss] = -6.32
            b[ind_rake_ss] = 0.90
            sigma[ind_rake_ss] = 0.28
            # all
            a[ind_rake_all] = -4.80
            b[ind_rake_all] = 0.69
            sigma[ind_rake_all] = 0.36
        # convert to natural log base
        a = a*np.log(10)
        b = b*np.log(10)
        sigma = sigma*np.log(10)
        
        # calculate deformation
        pgdef = np.exp(a + b*mag) # m

        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': sigma, 
                'sigma_mu': np.ones(pgdef.shape)*0.800, # using same values as Thompson 2021
                'dist_type': 'lognormal',
                'unit': 'm'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['a'] = a
            output['b'] = b
        
        # return
        return output


# -----------------------------------------------------------
class Hazus2020(SurfaceFaultRupture):
    """
    Compute surface fault rupture displacement using Hazus (FEMA, 2020), modified after Wells and Coppersmith (1994).
    
    Parameters
    ----------
    From upstream PBEE:
    mag: float, np.ndarray or list
        moment magnitude
    
    Geotechnical/geologic:
    
    Fixed:
    
    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
        
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
    
    """

    _NAME = 'Wells and Coppersmith (1994)'       # Name of the model
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
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
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
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {}
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        mag, # upstream PBEE RV
        return_inter_params=False # to get intermediate params
    ):
        """Model"""

        # initialize arrays
        
        # coefficient suggested by Hazus (FEMA 2020) for median maximum displacement
        a = -5.26*np.log(10) # for ln(D)
        b = 0.79*np.log(10) # for log10(D)
            
        # calculate deformation
        pgdef = np.exp(a + b*mag) # m

        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': np.ones(pgdef.shape)*0.498, # using same values as Thompson 2021
                'sigma_mu': np.ones(pgdef.shape)*0.800, # using same values as Thompson 2021
                'dist_type': 'lognormal',
                'unit': 'm'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['a'] = a
            output['b'] = b
        
        # return
        return output


# -----------------------------------------------------------
class Thompson2021(SurfaceFaultRupture):
    """
    Compute surface fault rupture displacement using Thompson (2021), average of Wells and Coppersmith (1994); Hecker et al. (2013); Wells and Youngs (2015).
    
    Parameters
    ----------
    From upstream PBEE:
    mag: float, np.ndarray or list
        moment magnitude
    
    Geotechnical/geologic:
    
    Fixed:
    
    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
        
    References
    ----------
    .. [1] Thompson, S, 2021, Fault Displacement Hazard Characterization for OpenSRA, California Energy Commission, Publication Number: CEC-500-202X-XXX.
    .. [2] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
    
    """

    _NAME = 'Thompson (2020)'       # Name of the model
    _ABBREV = None                     # Abbreviated name of the model
    _REF = "".join([                     # Reference for the model
        'Thompson, S, 2021, ',
        'Fault Displacement Hazard Characterization for OpenSRA, ',
        'California Energy Commission, ',
        'Publication Number: CEC-500-202X-XXX.'
    ])
    _RETURN_PBEE_DIST = {                            # Distribution information
        'category': 'EDP',        # Return category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'returned PBEE upstream random variables:',
        'params': {
            'pgdef': {
                'desc': 'permanent ground deformation (m)',
                'unit': 'm',
            },
        }
    }
    _INPUT_PBEE_DIST = {     # Randdom variable from upstream PBEE category required by model, e.g, pga, pgdef, pipe_strain
        'category': 'IM',        # Input category in PBEE framework, e.g., IM, EDP, DM
        "desc": 'PBEE upstream random variables:',
        'params': {
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
        'params': {}
    }
    _MODEL_INPUT_FIXED = {
        'desc': 'Fixed input variables:',
        'params': {}
    }
    _REQ_MODEL_RV_FOR_LEVEL = {
    }
    _REQ_MODEL_FIXED_FOR_LEVEL = {
    }
    _REQ_PARAMS_VARY_WITH_CONDITIONS = False
    _MODEL_FORM_DETAIL = {}
    _MODEL_INPUT_RV = {}
    
    
    @staticmethod
    # @njit
    def _model(
        mag, # upstream PBEE RV
        return_inter_params=False # to get intermediate params
    ):
        """Model"""

        # initialize arrays
        
        # coefficient suggested by median displacement
        a = -10.181 # for ln(D)
        b = 1.464 # for ln(D)
            
        # calculate deformation
        pgdef = np.exp(a + b*mag) # m

        # prepare outputs
        output = {
            'pgdef': {
                'mean': pgdef,
                'sigma': np.ones(pgdef.shape)*0.498,
                'sigma_mu': np.ones(pgdef.shape)*0.800,
                'dist_type': 'lognormal',
                'unit': 'm'
            },
        }
        # get intermediate values if requested
        if return_inter_params:
            output['a'] = a
            output['b'] = b
        
        # return
        return output


# -----------------------------------------------------------
def h14(**kwargs):
    """
    Compute surface fault rupture displacement using Wells & Coppersmith (1994) with modified parameters
    
    Parameters
    ----------
    M : float
        moment magnitude
    
    Returns
    -------
    d : float
        [cm] surface fault rupture displacement
        
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
    .. [2] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
    
    """
    
    # return 10**(-5.26 + 0.79*M) * 100
    kwargs['flag_hazus'] = True
    output = wc94(**kwargs)
    #
    return output
    
    
# -----------------------------------------------------------
def wc94(**kwargs):
    """
    Compute surface fault rupture displacement using Wells & Coppersmith (1994).
    
    Parameters
    ----------
    M : float
        moment magnitude
    mat_seg2calc : matrix
        matrix (n_event by n_site) with sites to calculate flagged as 1, else 0
    d_type : str
        displacement type: either **maximum** or **average**
    fault_type : str
        type of fault, enter bolded keywords: **ss** (strike-slip), **r** (reverse), **n** (normal), or **all**
    
    Returns
    -------
    d : float
        [cm] surface fault rupture displacement
        
    References
    ----------
    .. [1] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
    
    """
    
    # get inputs
    M = kwargs.get('M',None) # moment magnitude
    n_site = kwargs.get('n_site',None) # number of sites
    n_event = kwargs.get('n_event',len(M)) # number of ruptures
    # mat_seg2calc = kwargs.get('mat_seg2calc',None) # matrix (n_event by n_site) with sites to calculate flagged as 1, else 0
    d_type = kwargs.get('d_type','max') # displacement type
    fault_type = kwargs.get('fault_type','all') # fault type
    return_param = kwargs.get('return_param',None) #
    # nsamp_pgd = kwargs.get('nsamp_pgd',3) # number of samples for pgd
    # eps_aleatory = kwargs.get('eps_aleatory',[1.65,0,-1.65])
    # eps_epistemic = kwargs.get('eps_epistemic',0)
    n_sample = kwargs.get('n_sample',1) # number of samples, default to 1
    
    # default return_param
    if return_param is None:
        # return_param = ['pgd_surf', 'p_surf']
        return_param = ['pgd_surf']
    
    # fault_crossings
    fault_crossings = kwargs.get('fault_crossings',None)
    if fault_crossings is None:
        seg_with_crossing = sparse.coo_matrix([n_event,n_site])
    else:
        row = []
        col = []
        for i in range(len(fault_crossings)):
            crossings_i = fault_crossings['ListOfSegmentIDsWithCrossings'][i]
            if len(crossings_i) > 0:
                for j in crossings_i:
                    row.append(i)
                    col.append(j)
        seg_with_crossing = sparse.coo_matrix((
            [1]*len(row), (row, col)), shape=(n_event, n_site))
    
    # preset list
    # if mat_seg2calc is None:
        # mat_seg2calc = sparse.coo_matrix([n_event,n_site])
    
    M_sparse = M[seg_with_crossing.row]
    # M_sparse = sparse.coo_matrix((M[mat_seg2calc.row],(mat_seg2calc.row,mat_seg2calc.col)),shape=mat_seg2calc.shape)
    
    # Wells & Coppersmith (1994) coefficients
    if 'max' in d_type.lower():
        if fault_type.lower() == 'ss':
            a = -7.03
            b = 1.03
        elif fault_type.lower() == 'r':
            a = -1.84
            b = 0.29
        elif fault_type.lower() == 'n':
            a = -5.90
            b = 0.89
        elif fault_type.lower() == 'all':
            a = -5.46
            b = 0.82
        else: # default to 'all'
            a = -5.46
            b = 0.82
    elif 'ave' in d_type.lower():
        if fault_type.lower() == 'ss':
            a = -6.32
            b = 0.90
        elif fault_type.lower() == 'r':
            a = -0.74
            b = 0.08
        elif fault_type.lower() == 'n':
            a = -4.45
            b = 0.63
        elif fault_type.lower() == 'all':
            a = -4.80
            b = 0.69
        else: # default to 'all'
            a = -4.80
            b = 0.69
    # convert to exp base
    a = a*np.log(10)
    b = b*np.log(10)
                
    # See if Thompson's coefficients are requested
    flag_thompson = kwargs.get('flag_thompson',False)
    # coefficient suggested by Steve Thompson for median AVERAGE displacement
    if flag_thompson: # update to 
        a = -10.1813 # for ln(D)
        b = 1.4644 # for ln(D)
         
    # See if Hazus coefficients are requested
    flag_hazus = kwargs.get('flag_hazus',False)
    # coefficient suggested by Steve Thompson for median AVERAGE displacement
    if flag_hazus: # update to 
        a = -5.26*np.log(10) # for ln(D)
        b = 0.79*np.log(10) # for ln(D)
    
    # probability distribution
    prob_dist_type = 'Lognormal'
    sigma_total = 0.942 # for ln(D)
    sigma_aleatory = 0.498 # for ln(D)
    sigma_epistemic = 0.800 # for ln(D)
    
    # pgd_surf = {}
    
    # if eps_epistemic == 999:
        # sigma = sigma_total
        # eps_epistemic = 0
    # else:
        # sigma = sigma_aleatory
        
    # for i in range(len(eps_aleatory)):
        # Displacement for surface fault rupture (cm), log(D) = a + b*M, D in meter -> *100 to convert to cm
        # pgd_surf_i = np.exp(a + b*M_sparse + eps_aleatory[i]*sigma + eps_epistemic*sigma_epistemic) * 100
        # pgd_surf.update({i:sparse.coo_matrix((pgd_surf_i[mat_seg2calc.row],(mat_seg2calc.row,mat_seg2calc.col)),shape=mat_seg2calc.shape)})
    
    # calculate deformation and multiply by 100 to get to cm
    pgd_surf = np.exp(a + b*M_sparse) * 100
    # return pgd_surf
    
    # convert to sparse matrix
    pgd_surf = sparse.coo_matrix(
        (pgd_surf,(seg_with_crossing.row,seg_with_crossing.col)),
        shape=(n_event, n_site)
    )
    # return pgd_surf
    
    # probability
    # for k in range(n_sample):
        # p_surf[k] = sparse.coo_matrix(
            # ([1]*len(pgd_surf),(seg_with_crossing.row,seg_with_crossing.col)),
            # shape=seg_with_crossing.shape
        # )
    # p_surf = sparse.coo_matrix(
        # (np.ones(len(M_sparse)),(seg_with_crossing.row,seg_with_crossing.col)),
        # shape=(n_event, n_site)
    # )
    
    # store outputs
    output = {}
    #
    if 'pgd_surf' in return_param:
        output.update({'pgd_surf': pgd_surf})
    # if 'p_surf' in return_param:
        # output.update({'p_surf': p_surf})
    # store probability distribution parameters
    output.update({'prob_dist': {
        'type': prob_dist_type,
        'sigma_total': sigma_total,
        'sigma_aleatory': sigma_aleatory,
        'sigma_epistemic': sigma_epistemic}})
    #
    return output
    
# -----------------------------------------------------------
def t20(**kwargs):
    """
    Compute surface fault rupture displacement using Wells & Coppersmith (1994).
    
    Parameters
    ----------
    M : float
        moment magnitude
    mat_seg2calc : matrix
        matrix (n_event by n_site) with sites to calculate flagged as 1, else 0
    d_type : str
        displacement type: either **maximum** or **average**
    fault_type : str
        type of fault, enter bolded keywords: **ss** (strike-slip), **r** (reverse), **n** (normal), or **all**
    
    Returns
    -------
    d : float
        [cm] surface fault rupture displacement
        
    References
    ----------
    .. [1] Wells, D.L., and Coppersmith, K.J., 1994, New Empirical Relationships Among Magnitude, Rupture Length, Rupture width, Rupture Area, and Surface Displacement, Bulletin of the Seismological Society of America, vol. 84, no. 4, pp. 974-1002.
    
    """
    
    #
    kwargs['flag_thompson'] = True
    output = wc94(**kwargs)
    #
    return output