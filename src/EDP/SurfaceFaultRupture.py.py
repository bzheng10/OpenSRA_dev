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


# -----------------------------------------------------------
def Hazus2014(M):
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
    
    return 10**(-5.26 + 0.79*M) * 100
    
    
# -----------------------------------------------------------
def WellsCoppersmith1994(**kwargs):
    """
    Compute surface fault rupture displacement using Wells & Coppersmith (1994).
    
    Parameters
    ----------
    M : float
        moment magnitude
    mat_seg2calc : matrix
        matrix (n_rup by n_site) with sites to calculate flagged as 1, else 0
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
    n_rup = kwargs.get('n_rup',None) # number of ruptures
    mat_seg2calc = kwargs.get('mat_seg2calc',None) # matrix (n_rup by n_site) with sites to calculate flagged as 1, else 0
    d_type = kwargs.get('d_type','max') # displacement type
    fault_type = kwargs.get('fault_type','all') # fault type
    return_param = kwargs.get('return_param',None) #
    # nsamp_pgd = kwargs.get('nsamp_pgd',3) # number of samples for pgd
    # eps_aleatory = kwargs.get('eps_aleatory',[1.65,0,-1.65])
    # eps_epistemic = kwargs.get('eps_epistemic',0)
    
    # See if Thompson's coefficients are requested
    flag_thompson = kwargs.get('flag_thompson',False)
    
    # default return_param
    if return_param is None:
        return_param = ['pgd_surf']
        
    # preset list
    if mat_seg2calc is None:
        mat_seg2calc = sparse.coo_matrix([n_rup,n_site])
    
    M_sparse = M[mat_seg2calc.row]
    # M_sparse = sparse.coo_matrix((M[mat_seg2calc.row],(mat_seg2calc.row,mat_seg2calc.col)),shape=mat_seg2calc.shape)
    
    # coefficients
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
            
    # coefficient suggested by Steve Thompson for median AVERAGE displacement
    if flag_thompson: # update to 
        a = -10.1813 # for ln(D)
        b = 1.4644 # for ln(D)
    
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
        
    pgd_surf = np.exp(a + b*M_sparse) * 100
    pgd_surf = sparse.coo_matrix((pgd_surf[mat_seg2calc.row],(mat_seg2calc.row,mat_seg2calc.col)),shape=mat_seg2calc.shape)
    
    # store outputs
    output = {}
    output.update({'pgd_surf': pgd_surf})
    # store probability distribution parameters
    output.update({'prob_dist': {'type': prob_dist_type,
                                'sigma_total': sigma_total,
                                'sigma_aleatory': sigma_aleatory,
                                'sigma_epistemic': sigma_epistemic}})
    
    #
    return output
    
# -----------------------------------------------------------
def Thompson2020(**kwargs):
    """
    Compute surface fault rupture displacement using Wells & Coppersmith (1994).
    
    Parameters
    ----------
    M : float
        moment magnitude
    mat_seg2calc : matrix
        matrix (n_rup by n_site) with sites to calculate flagged as 1, else 0
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
    
    kwargs['flag_thompson'] = True
    output = WellsCoppersmith1994(**kwargs)
    
    return output