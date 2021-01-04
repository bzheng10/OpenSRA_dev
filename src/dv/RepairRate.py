# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for repair rates
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
def ALA2001(**kwargs):
    """
    Compute repair rates for buried pipelines induced by both transient and permanent ground deformations.
    
    Parameters
    ----------
    By transient demand
    pgv : float
        [cm/s] peak ground velocity
    
    By ground deformation
    pgd : float
        [cm] permanent ground deformation (e.g., lateral spreading, ground settlement, landslide)
    pgd_label : str
        string that describes the permanent ground deformation (e.g., 'ls', 'gs', 'land', etc.)
    pgd_cutoff : float, optional
        [cm] cutoff **PGD** below which the repair rate is negligible; default = 4 inch (2 to 4 in suggested in O'Rourke Memo to OpenSRA)
    k2 : float, optional
        coefficient for pipe type; default = 1
        
    Other parameters
    return_param : str
        choose variables to return; specify any number of the return variables; default = [**'rr_pgv'**, **'rr_pgd'**]
    
    Returns
    -------
    rr_pgv : float
        [repairs/km] repair rate due to transient ground demand (**PGV**)
    rr_pgd : float
        [repairs/km] repair rate due to permanent ground failure (**PGD**)
    rr_leak : float
        [repairs/km] number of leaks per km
    rr_break : float
        [repairs/km] number of breaks per km
            
    References
    ----------
    .. [1] American Lifeline Alliance (ALA), 2001, Seismic Fragility Formulations for Water Systems, Parts 1 and 2 – Guideline and Appendices prepared for ASCE and FEMA, April, https://www.americanlifelinesalliance.com/pdf/Part_1_Guideline.pdf.
    .. [2] O'Rourke, T.D., 2020, Memo to OpenSRA.
    
    """
    
    # get inputs
    pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
    pgd_label = kwargs.get('pgd_label',None) # label for type of PGD
    pgd = kwargs.get(pgd_label,None) # cm, permanent ground deformation
    return_param = kwargs.get('return_param',['rr_pgv', 'rr_pgd']) # return param
    pgd_cutoff = kwargs.get('pgd_cutoff',4*2.54) # cm (4 in)
    k1 = kwargs.get('k1',1.0)
    k2 = kwargs.get('k2',1.0)
    flag_orourke_pgv = kwargs.get('flag_orourke_pgv',False) # True to use O'Rourke's form, False to stay with ALA
    n_samp_im = kwargs.get('n_samp_im',1) # number of IM samples, default to 1
    # n_samp_edp = kwargs.get('n_samp_edp',1) # number of EDP samples, default to 1
    n_site = kwargs.get('n_site') # number of sites
    n_rup = kwargs.get('n_rup') # number of sites
    l_seg = kwargs.get('l_seg',np.ones(n_site)) # km, segment length
    
    prob = None
    for key in kwargs.keys():
        if 'p_' in key:
            prob = kwargs.get(key,None) # %, probability of triggering
            # logging.debug(f"\t\tPulled probability from kwargs.")
            
    # preset dict
    if 'rr_pgv' in return_param:
        rr_pgv = {}
    if 'rr_pgd' in return_param:
        rr_pgd = {}
    if 'rr_leak' in return_param:
        rr_leak = {}
    if 'rr_break' in return_param:
        rr_break = {}
    
    #
    if pgd_label is None or not 'surf' in pgd_label:
        # loop through all realizations
        for k in range(n_samp_im):
        
            # Repair rates by PGV
            if pgv is None:
                rr_pgv_k = sparse.coo_matrix(np.zeros([n_rup,n_site]))
                # logging.debug(f"\t\tPGV is None.")
            else:
                if flag_orourke_pgv:
                    rr_pgv_k = pgv[k].power(2.50)*np.exp(-11) # PGV in cm/s, repair rate in repairs/km
                    # logging.debug(f"\t\tCalculated rr_pgv with O'Rourke.")
                else:
                    rr_pgv_k = pgv[k]/2.54*0.00187*k1 # convert PGV from cm/sec to in/sec, repair rate in repairs/1000 ft
                    rr_pgv_k = rr_pgv_k/0.3048 # correct from rr/1000 ft to rr/km
                    # logging.debug(f"\t\tCalculated rr_pgv with 'ALA'.")
                
            # Repair rates by PGD
            if pgd is None:
                rr_pgd_k = sparse.coo_matrix(np.zeros([n_rup,n_site]))
                # logging.debug(f"\t\tPGD is None.")
            else:
                # see if pgd contains multiple samples (dictionary of matrices versus matrix)
                if type(pgd) is dict:
                    pgd_k = pgd[k]
                    # logging.debug(f"\t\tFor {pgd_label}: pgd is dictionary.")
                else:
                    pgd_k = pgd
                    # logging.debug(f"\t\tFor {pgd_label}: pgd is matrix.")
                #
                shape = pgd_k.shape
                rows = pgd_k.row
                cols = pgd_k.col
                # logging.debug(f"\t\tMinimum(pgd.data) = {min(pgd_k.data)}.")
                # pgd_k.data[pgd_k.data<=pgd_cutoff] = 0 # apply cutoff correction, only for O'Rourke
                pgd_k.data = np.maximum(pgd_k.data-pgd_cutoff,0) # shift by pgd_cutoff, only for O'Rourke
                # logging.debug(f"\t\tMinimum(pgd.data) = {min(pgd_k.data)}.")
                pgd_k = pgd_k/2.54 # convert to inches
                # pgd_k = sparse.coo_matrix((np.maximum(pgd[k].data-pgd_cutoff,0)/2.54,(rows,cols)),shape=shape) # apply cutoff correction, limit minimum value to 0, and convert to inches
                rr_pgd_k = pgd_k.power(0.319)*k2*1.06 # pgd in inches, repair rate in repairs/1000 ft
                rr_pgd_k = rr_pgd_k/0.3048 # correct from rr/1000 ft to rr/km
                # logging.debug(f"\t\tCalculated rr_pgd.")
                rr_pgd_k = rr_pgd_k.multiply(prob[k]/100) # multiply by probability of triggering
                # logging.debug(f"\t\tFor {pgd_label}: multiply by probability.")
            
            # store rr
            if 'rr_pgv' in return_param:
                rr_pgv.update({k:rr_pgv_k})
            if 'rr_pgd' in return_param:
                rr_pgd.update({k:rr_pgd_k})
            
            # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
            if 'rr_leak' in return_param: 
                rr_leak.update({k:rr_pgv_k*0.8 + rr_pgd_k*0.2})
            if 'rr_break' in return_param:
                rr_break.update({k:rr_pgv_k*0.2 + rr_pgd_k*0.8})
            
    else:
        # Repair rates by PGV
        rr_pgv_k = sparse.coo_matrix(np.zeros([n_rup,n_site]))
            
        # Repair rates by PGD for surface fault rupture
        shape = pgd.shape
        rows = pgd.row
        cols = pgd.col
        pgd_k = pgd
        
        # simplified fault crossing damage
        p_no_fail = np.ones(pgd_k.data.shape)
        p_no_fail[np.logical_and(pgd_k.data>0,pgd_k.data<=12)] = 0.50
        p_no_fail[np.logical_and(pgd_k.data>12,pgd_k.data<=24)] = 0.25
        p_no_fail[pgd_k.data>24] = 0.05
        
        # 
        rr_pgd_k = -np.log(p_no_fail)/l_seg[cols] # number of breaks per km, with l_seg in km
        
        #
        ind_red = np.where(rr_pgd_k>0)[0]
        rows_red = rows[ind_red]
        cols_red = cols[ind_red]
        rr_pgd_k = sparse.coo_matrix((rr_pgd_k[ind_red],(rows_red,cols_red)),shape=(n_rup,n_site))
        
        # store rr
        if 'rr_pgv' in return_param:
            rr_pgv.update({'all':rr_pgv_k})
        if 'rr_pgd' in return_param:
            rr_pgd.update({'all':rr_pgd_k})
        
        # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
        if 'rr_leak' in return_param: 
            rr_leak.update({'all':rr_pgv_k*0.8 + rr_pgd_k*0.2})
        if 'rr_break' in return_param:
            rr_break.update({'all':rr_pgv_k*0.2 + rr_pgd_k*0.8})
    
    # store outputs
    output = {}
    #
    if 'rr_pgv' in return_param:
        output.update({'rr_pgv': rr_pgv.copy()})
    if 'rr_pgd' in return_param:
        output.update({'rr_pgd': rr_pgd.copy()})
    if 'rr_leak' in return_param:
        output.update({'rr_leak': rr_leak.copy()})
    if 'rr_break' in return_param:
        output.update({'rr_break': rr_break.copy()})
    
    #
    return output
    
    
# -----------------------------------------------------------
def ORourke2020(**kwargs):
    """
    Compute repair rates for buried pipelines induced by both transient and permanent ground deformations.
    
    Parameters
    ----------
    By transient demand
    pgv : float
        [cm/s] peak ground velocity
    
    By ground deformation
    pgd : float
        [cm] permanent ground deformation (e.g., lateral spreading, ground settlement, landslide)
    pgd_label : str
        string that describes the permanent ground deformation (e.g., 'ls', 'gs', 'land', etc.)
    pgd_cutoff : float, optional
        [cm] cutoff **PGD** below which the repair rate is negligible; default = 5 cm (2 to 4 in suggested in O'Rourke Memo to OpenSRA)
    k2 : float, optional
        coefficient for pipe type; default = 1
        
    Other parameters
    return_param : str
        choose variables to return; specify any number of the return variables; default = [**'rr_pgv'**, **'rr_pgd'**]
    
    Returns
    -------
    rr_pgv : float
        [repairs/km] repair rate due to transient ground demand (**PGV**)
    rr_pgd : float
        [repairs/km] repair rate due to permanent ground failure (**PGD**)
    rr_leak : float
        [repairs/km] number of leaks per km
    rr_break : float
        [repairs/km] number of breaks per km
            
    References
    ----------
    .. [1] American Lifeline Alliance (ALA), 2001, Seismic Fragility Formulations for Water Systems, Parts 1 and 2 – Guideline and Appendices prepared for ASCE and FEMA, April, https://www.americanlifelinesalliance.com/pdf/Part_1_Guideline.pdf.
    .. [2] O'Rourke, T.D., 2020, Memo to OpenSRA.
    
    """ 
    
    output = ALA2001(**kwargs)
    
    return output


# -----------------------------------------------------------
def Hazus2014(**kwargs):
    """
    Compute repair rates for buried pipelines induced by both transient and permanent ground deformations.
    
    Parameters
    ----------
    By transient demand
    pgv : float
        [cm/s] peak ground velocity
    
    By ground deformation
    prob : float
        [%] probability of hazard to trigger demand (e.g., liquefaction, landslide)
    pgd : float
        [cm] permanent ground deformation (e.g., lateral spreading, ground settlement, landslide)
    pgd_label : str
        string that describes the permanent ground deformation (e.g., 'ls', 'gs', 'land', etc.)
        
    Other parameters
    return_param : str
        choose variables to return; specify any number of the return variables; default = [**'rr_pgv'**, **'rr_pgd'**]
    pipe_stiff : str
        either **brittle** or **ductile** for pipe segment; default = **brittle**
    
    Returns
    -------
    rr_pgv : float
        [repairs/km] repair rate due to transient ground demand (**PGV**)
    rr_pgd : float
        [repairs/km] repair rate due to permanent ground failure (**PGD**)
    rr_leak : float
        [repairs/km] number of leaks per km
    rr_break : float
        [repairs/km] number of breaks per km
            
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
    .. [2] O'Rourke, M., and Ayala, G., 1993, Pipeline Damage Due to Wave Propagation, Journal of Geotechnical Engineering, vol. 119, no. 9, pp. 1490-1498.
    .. [3] Honegger D.G., and Eguchi R.T., 1992, Determination of Relative Vulnerabilities to Seismic Damage for San Diego County Water Authority (SDCWA) Water Transmission Pipelines.
    
    """
    
    # get inputs
    pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity
    pgd_label = kwargs.get('pgd_label',None) # label for type of PGD
    pgd = kwargs.get(pgd_label,None) # cm, permanent ground deformation
    pipe_stiff = kwargs.get('pipe_stiff','brittle') # brittle or ductile for pipe; default = brittle
    return_param = kwargs.get('return_param',['rr_pgv', 'rr_pgd']) # return param
    n_samp_im = kwargs.get('n_samp_im',1) # number of samples, default to 1
    n_site = kwargs.get('n_site') # number of sites
    n_rup = kwargs.get('n_rup') # number of sites
    
    for key in kwargs.keys():
        if 'p_' in key:
            prob = kwargs.get(key,None) # %, probability of triggering
            # logging.debug(f"\t\tPulled probability from kwargs.")

    # correction factor for ductility of pipes
    if type(pipe_stiff) is str:
        rr_multi = np.asarray([0.3 if pipe_stiff == 'ductile' else 1])
    else:
        rr_multi = np.ones(n_site)
        rr_multi[pipe_stiff=='ductile'] = 0.3
        rr_multi = np.transpose(np.repeat(rr_multi,n_rup))
    
    # preset dict
    if 'rr_pgv' in return_param:
        rr_pgv = {}
    if 'rr_pgd' in return_param:
        rr_pgd = {}
    if 'rr_leak' in return_param:
        rr_leak = {}
    if 'rr_break' in return_param:
        rr_break = {}

    #
    if pgd_label is None or not 'surf' in pgd_label:
        # loop through all realizations
        for k in range(n_samp_im):
        
            # Repair rates by PGV
            if pgv is None:
                rr_pgv_k = sparse.coo_matrix(np.zeros([n_rup,n_site]))
                # logging.debug(f"\t\tPGV is None.")
            else:
                rr_pgv_k = pgv[k].power(2.25)*0.0001 # PGV in cm/sec, repair rate in repairs/km
                # logging.debug(f"\t\tCalculated rr_pgv.")
                
            # Repair rates by PGD
            if pgd is None:
                rr_pgd_k = sparse.coo_matrix(np.zeros([n_rup,n_site]))
            else:
                if type(pgd) is dict:
                    rr_pgd_k = (pgd[k]/2.54).power(0.56) # convert PGD to inches, repair rate in repairs/km
                    # logging.debug(f"\t\tCalculated rr_pgd with pgd[k].")
                else:
                    rr_pgd_k = (pgd/2.54).power(0.56) # convert PGD to inches, repair rate in repairs/km
                    # logging.debug(f"\t\tCalculated rr_pgd with pgd.")
                rr_pgd_k = rr_pgd_k.multiply(prob[k]/100) # multiply by probability of triggering
                # logging.debug(f"\t\tMultiplied by probability.")
                
            # correct for pipe_stiff
            if 'rr_pgv' in return_param:
                rr_pgv.update({k:rr_pgv_k.multiply(rr_multi)})
            else:
                rr_pgv_k = rr_pgv_k.multiply(rr_multi)
                
            if 'rr_pgd' in return_param:
                rr_pgd.update({k:rr_pgd_k.multiply(rr_multi)})
            else:
                rr_pgd_k = rr_pgd_k.multiply(rr_multi)
            
            # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
            if 'rr_leak' in return_param: 
                rr_leak.update({k:rr_pgv*0.8 + rr_pgd*0.2})
            if 'rr_break' in return_param:
                rr_break.update({k:rr_pgv*0.2 + rr_pgd*0.8})
                
    else:
        # Repair rates by PGV
        rr_pgv_k = sparse.coo_matrix(np.zeros([n_rup,n_site]))
            
        # Repair rates by PGD
        rr_pgd_k = (pgd/2.54).power(0.56) # convert PGD to inches, repair rate in repairs/km
            
        # correct for pipe_stiff
        if 'rr_pgv' in return_param:
            rr_pgv.update({'all':rr_pgv_k.multiply(rr_multi)})
        else:
            rr_pgv_k = rr_pgv_k.multiply(rr_multi)
            
        if 'rr_pgd' in return_param:
            rr_pgd.update({'all':rr_pgd_k.multiply(rr_multi)})
        else:
            rr_pgd_k = rr_pgd_k.multiply(rr_multi)
        
        # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
        if 'rr_leak' in return_param: 
            rr_leak.update({'all':rr_pgv*0.8 + rr_pgd*0.2})
        if 'rr_break' in return_param:
            rr_break.update({'all':rr_pgv*0.2 + rr_pgd*0.8})
        
    # store outputs
    output = {}
    #
    if 'rr_pgv' in return_param:
        output.update({'rr_pgv': rr_pgv.copy()})
    if 'rr_pgd' in return_param:
        output.update({'rr_pgd': rr_pgd.copy()})
    if 'rr_leak' in return_param:
        output.update({'rr_leak': rr_leak.copy()})
    if 'rr_break' in return_param:
        output.update({'rr_break': rr_break.copy()})
    
    #
    return output