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
    n_sample = kwargs.get('n_sample',1) # number of IM samples, default to 1
    # n_samp_edp = kwargs.get('n_samp_edp',1) # number of EDP samples, default to 1
    n_site = kwargs.get('n_site') # number of sites
    n_event = kwargs.get('n_event',1) # number of events
    
    # O'Rourke (2020) corrections
    flag_orourke_pgv = kwargs.get('flag_orourke_pgv',False) # True to use O'Rourke's form, False to stay with ALA
    pgd_cutoff = kwargs.get('PGDCutoff',4*2.54) # cm (4 in)
    flag_use_gmpgv = kwargs.get('ConvertToGMPGV',False)
    if flag_orourke_pgv is False:
        pgd_cutoff = 0
        flag_use_gmpgv = False
    
    # component properties
    length = kwargs.get('ComponentLength',np.ones(n_site)) # km, component length
    if isinstance(length,str):
        length = np.ones(n_site)
    material = kwargs.get('ComponentMaterial',None) # component material
    joint = kwargs.get('ComponentJointType',None) # component joint
    diameter = kwargs.get('ComponentDiameter',None) # in, component diameter
    
    # see if material is provided, if not, see if k1 and k2 are provided explicitly
    if material is None:
        k1 = kwargs.get('k1',np.ones(n_site)) # factor for rr_pgv
        k2 = kwargs.get('k2',np.ones(n_site)) # factor for rr_pgd
    else:
        # check for material data availability
        if isinstance(material,str):
            material = ['castiron']*n_site
        else:
            # convert material property strings to lower cases
            material = [item.lower() for item in material]
        # check for joint data availability
        if joint is None or isinstance(joint,str):
            joint = ['']*n_site
        else:
            # convert joint property strings to lower cases
            joint = [item.lower() for item in joint]
        # check for diameter data availability
        if diameter is None or isinstance(diameter,str):
            diameter = np.ones(n_site)
        
        # get correction factors for material, joint type, and diameter
        #   to add soil corrosivity later
        k1 = np.ones(n_site) # default value = 1
        k2 = np.ones(n_site) # default value = 1
        
        if flag_orourke_pgv: # O'Rourke (2020)
            # check cases
            bool_castIron = ['cast' in item for item in material] # cast iron
            bool_ductileIron = ['ductile' in item for item in material] # ductile iron
            bool_ductileIron_pushOn = np.logical_and(bool_ductileIron,['push' in item for item in joint]) # ductile iron with push-on joints
            bool_ductileIron_fieldLok = np.logical_and(bool_ductileIron,['field' in item for item in material]) # ductile iron with field lok joints
            bool_asbestosCement = ['asbestos' in item for item in material] # asbestos cement
            bool_steel = ['steel' in item for item in material] # welded steel
            bool_steel_water = np.logical_and(bool_steel,['water' in item for item in material]) # water welded steel
            bool_steel_rubberGasket = np.logical_and(bool_steel,['gasket' in item for item in material]) # steel with rubber gasket joints
            bool_steel_SEAW = np.logical_and(bool_steel,[(' shield' in item) | ('seaw' in item) for item in material]) # steel with SEAW welds
            bool_steel_UEAW = np.logical_and(bool_steel,[('unshield' in item) | ('ueaw' in item) for item in material]) # steel with UEAW welds
            bool_steel_oxy = np.logical_and(bool_steel,['oxy' in item for item in material]) # steel with oxyacetylene welds
            bool_poly = [('hdpe' in item) | ('mdpe' in item) | ('polyethylene' in item) for item in material] # polyethylene (high or medium density)

            # set values
            k1[bool_castIron] = 1.0
            k1[bool_asbestosCement] = 1.5
            k1[bool_ductileIron_pushOn] = 1.0
            k1[bool_ductileIron_fieldLok] = 0.5
            k1[bool_steel_water] = 0.1
            k1[bool_steel_rubberGasket] = 1.0
            k1[bool_steel_SEAW] = 0.05
            k1[bool_steel_UEAW] = 0.2
            k1[bool_steel_oxy] = 1.0
            k1[bool_poly] = 0.05
            
            # k2 = k1 for O'Rourke (2020)
            k2 = k1
            
        else: # ALA (2001)
            # check cases
            bool_castIron = ['cast' in item for item in material] # cast iron
            bool_castIron_cement = np.logical_and(bool_castIron,['cement joint' in item for item in joint]) # cast iron with cement joint
            bool_castIron_rubberGasket = np.logical_and(bool_castIron,['gasket' in item for item in material]) # cast iron with rubber gasket joint
            bool_castIron_mechRestrain = np.logical_and(bool_castIron,['mech' in item for item in material]) # cast iron with mechanical restrained joint
            bool_steel = ['steel' in item for item in material] # welded steel
            bool_steel_arcWeld = np.logical_and(bool_steel,[('arc' in item) | ('eaw' in item) for item in joint]) # steel with arc-welded joints
            bool_steel_arcWeld_smallDiam = np.logical_and(bool_steel_arcWeld,diameter<16) # steel with arc-welded joints with small diameter
            bool_steel_arcWeld_largeDiam = np.logical_and(bool_steel_arcWeld,diameter>=16) # steel with arc-welded joints with large diameter
            bool_steel_rubberGasket = np.logical_and(bool_steel,['gasket' in item for item in joint]) # steel with rubber gasket joint
            bool_steel_screwed = np.logical_and(bool_steel,['screw' in item for item in joint]) # steel with screwed joint
            bool_steel_riveted = np.logical_and(bool_steel,['rivet' in item for item in joint]) # steel with riveted joint
            bool_asbestosCement = ['asbestos' in item for item in material] # asbestos cement
            bool_asbestosCement_rubberGasket = np.logical_and(bool_asbestosCement,['gasket' in item for item in joint]) # asbestos cement with rubber gasket joint
            bool_asbestosCement_cement = np.logical_and(bool_asbestosCement,['cement joint' in item for item in joint]) # asbestos cement with cement joint
            bool_concrete = ['concrete' in item for item in material] # concrete
            bool_concrete_arcWeld = np.logical_and(bool_concrete,[('arc' in item) | ('eaw' in item) for item in joint]) # asbestos cement with arc-welded joint
            bool_concrete_cement = np.logical_and(bool_concrete,['cement joint' in item for item in joint]) # asbestos cement with cement joint
            bool_concrete_rubberGasket = np.logical_and(bool_concrete,['gasket' in item for item in joint]) # asbestos cement with rubber gasket joint
            bool_pvc = ['pvc' in item for item in material] # pvc
            bool_ductileIron = ['ductile' in item for item in material] # ductile iron

            # set values
            k1[bool_castIron_cement] = 1.0
            k1[bool_castIron_rubberGasket] = 0.8
            k1[bool_steel_arcWeld_smallDiam] = 0.6
            k1[bool_steel_arcWeld_largeDiam] = 0.15
            k1[bool_steel_rubberGasket] = 0.7
            k1[bool_steel_screwed] = 1.3
            k1[bool_steel_riveted] = 1.3
            k1[bool_asbestosCement_rubberGasket] = 0.5
            k1[bool_asbestosCement_cement] = 1.0
            k1[bool_concrete_arcWeld] = 0.7
            k1[bool_concrete_cement] = 1.0
            k1[bool_concrete_rubberGasket] = 0.8
            k1[bool_pvc] = 0.5
            k1[bool_ductileIron] = 0.5

            # set values
            k2[bool_castIron_cement] = 1.0
            k2[bool_castIron_rubberGasket] = 0.8
            k2[bool_castIron_mechRestrain] = 0.7
            k2[bool_steel_arcWeld_largeDiam] = 0.15
            k2[bool_steel_rubberGasket] = 0.7
            k2[bool_asbestosCement_rubberGasket] = 0.8
            k2[bool_asbestosCement_cement] = 1.0
            k2[bool_concrete_arcWeld] = 0.6
            k2[bool_concrete_cement] = 1.0
            k2[bool_concrete_rubberGasket] = 0.7
            k2[bool_pvc] = 0.8
            k2[bool_ductileIron] = 0.5
            
    # repeat k1 and k2 over all scenarios ((n_site,) to (n_event,n_site))
    k1 = np.repeat(np.expand_dims(k1,axis=0), n_event, axis=0)
    k2 = np.repeat(np.expand_dims(k2,axis=0), n_event, axis=0)
            
    # get probability
    prob = None
    for key in kwargs.keys():
        if 'p_' in key:
            prob = kwargs.get(key,None) # %, probability of hazard (e.g., liquefaction)
            # logging.debug(f"\t\tPulled probability from kwargs.")
            
    # preset dict
    if 'rr_pgv' in return_param:
        rr_pgv = {}
    if 'rr_pgd' in return_param:
        rr_pgd = {}
    # if 'rr_leak' in return_param:
        # rr_leak = {}
    # if 'rr_break' in return_param:
        # rr_break = {}
    
    # demands with pre-computed probability (e.g., LateralSpread, GroundSettlement, Landslide)
    if pgd_label is None or not 'surf' in pgd_label:
        # loop through all realizations
        for k in range(n_sample):
        
            # Repair rates by PGV
            if pgv is None:
                rr_pgv_k = sparse.coo_matrix(np.zeros([n_event,n_site]))
                # logging.debug(f"\t\tPGV is None.")
            else:
                # get pgv for current sample
                pgv_k = pgv[k]
                if flag_orourke_pgv:
                    if flag_use_gmpgv:
                        pgv_k = pgv_k/1.21 # convert from PGV to GMPGV, PGV = 1.21G*MPGV
                    rr_pgv_k = pgv_k.power(2.50)*np.exp(-11) # PGV in cm/s, repair rate in repairs/km
                    rr_pgv_k = rr_pgv_k.multiply(k1) # correction for material
                    # logging.debug(f"\t\tCalculated rr_pgv with O'Rourke.")
                else:
                    rr_pgv_k = pgv_k/2.54 # convert PGV from cm/sec to in/sec, 
                    rr_pgv_k = rr_pgv_k.multiply(k1)*0.00187 # repair rate in repairs/1000 ft, with correction for material, joint, diameter
                    rr_pgv_k = rr_pgv_k/0.3048 # correct from rr/1000 ft to rr/km
                    # logging.debug(f"\t\tCalculated rr_pgv with 'ALA'.")
                
            # Repair rates by PGD
            if pgd is None:
                rr_pgd_k = sparse.coo_matrix(np.zeros([n_event,n_site]))
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
                pgd_k = pgd_k/2.54 # convert from cm to inches
                # pgd_k = sparse.coo_matrix((np.maximum(pgd[k].data-pgd_cutoff,0)/2.54,(rows,cols)),shape=shape) # apply cutoff correction, limit minimum value to 0, and convert to inches
                rr_pgd_k = pgd_k.power(0.319)*1.06 # pgd in inches, repair rate in repairs/1000 ft
                rr_pgd_k = rr_pgd_k.multiply(k2) # correction for material, joint, diameter
                rr_pgd_k = rr_pgd_k/0.3048 # correct from rr/1000 ft to rr/km
                # logging.debug(f"\t\tCalculated rr_pgd.")
                rr_pgd_k = rr_pgd_k.multiply(prob[k]/100) # multiply by probability of hazard (e.g., liquefaction)
                # logging.debug(f"\t\tFor {pgd_label}: multiply by probability.")
            
            # print(f"rr_pgv_k = {rr_pgv_k.toarray()[0]}")
            # print(f"rr_pgd_k = {rr_pgd_k.toarray()[0]}")
            # sys.exit()
            
            # store rr
            if 'rr_pgv' in return_param:
                rr_pgv.update({k:rr_pgv_k})
            if 'rr_pgd' in return_param:
                rr_pgd.update({k:rr_pgd_k})
            
            # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
            # if 'rr_leak' in return_param: 
                # rr_leak.update({k:rr_pgv_k*0.8 + rr_pgd_k*0.2})
            # if 'rr_break' in return_param:
                # rr_break.update({k:rr_pgv_k*0.2 + rr_pgd_k*0.8})
           
    # demands that have prob = 1 (PGV) or with no pre-computed probability (SurfaceFaultRupture)
    else:
        # loop through all realizations
        for k in range(n_sample):
            # Repair rates by PGV
            rr_pgv_k = sparse.coo_matrix(np.zeros([n_event,n_site]))
            
            # see if pgd contains multiple samples (dictionary of matrices versus matrix)
            if type(pgd) is dict:
                pgd_k = pgd[k]
                # logging.debug(f"\t\tFor {pgd_label}: pgd is dictionary.")
            else:
                pgd_k = pgd
                # logging.debug(f"\t\tFor {pgd_label}: pgd is matrix.")
            
            # Repair rates by PGD for surface fault rupture
            shape = pgd_k.shape
            rows = pgd_k.row
            cols = pgd_k.col
            # pgd_k = pgd
            
            # simplified fault crossing damage
            p_no_fail = np.ones(pgd_k.data.shape)
            p_no_fail[np.logical_and(pgd_k.data>0,pgd_k.data<=12)] = 0.50
            p_no_fail[np.logical_and(pgd_k.data>12,pgd_k.data<=24)] = 0.25
            p_no_fail[pgd_k.data>24] = 0.05
            
            # 
            rr_pgd_k = -np.log(p_no_fail)/length[cols] # number of breaks per km, with length in km
            
            #
            ind_red = np.where(rr_pgd_k>0)[0]
            rows_red = rows[ind_red]
            cols_red = cols[ind_red]
            rr_pgd_k = sparse.coo_matrix((rr_pgd_k[ind_red],(rows_red,cols_red)),shape=(n_event,n_site))
            
            # store rr
            if 'rr_pgv' in return_param:
                # rr_pgv.update({'all':rr_pgv_k})
                rr_pgv.update({k:rr_pgv_k})
            if 'rr_pgd' in return_param:
                # rr_pgd.update({'all':rr_pgd_k})
                rr_pgd.update({k:rr_pgd_k})
            
            # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
            # if 'rr_leak' in return_param: 
                # rr_leak.update({'all':rr_pgv_k*0.8 + rr_pgd_k*0.2})
                # rr_leak.update({k:rr_pgv_k*0.8 + rr_pgd_k*0.2})
            # if 'rr_break' in return_param:
                # rr_break.update({'all':rr_pgv_k*0.2 + rr_pgd_k*0.8})
                # rr_break.update({k:rr_pgv_k*0.2 + rr_pgd_k*0.8})
    
    # store outputs
    output = {}
    #
    if 'rr_pgv' in return_param:
        output.update({'rr_pgv': rr_pgv.copy()})
    if 'rr_pgd' in return_param:
        output.update({'rr_pgd': rr_pgd.copy()})
    # if 'rr_leak' in return_param:
        # output.update({'rr_leak': rr_leak.copy()})
    # if 'rr_break' in return_param:
        # output.update({'rr_break': rr_break.copy()})
    
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
    
    kwargs['flag_orourke_pgv'] = True
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
    n_sample = kwargs.get('n_sample',1) # number of samples, default to 1
    n_site = kwargs.get('n_site') # number of sites
    n_event = kwargs.get('n_event',1) # number of events
    
    # get probability
    for key in kwargs.keys():
        if 'p_' in key:
            prob = kwargs.get(key,None) # %, probability of hazard (e.g., liquefaction)
            # logging.debug(f"\t\tPulled probability from kwargs.")

    # get correction factors for material, joint type, and diameter
        #   to add soil corrosivity later

    # component properties
    material = kwargs.get('ComponentMaterial',None) # component material
    joint = kwargs.get('ComponentJointType',None) # component joint
    # diameter = kwargs.get('ComponentDiameter',None) # in, component diameter
    year_of_install = kwargs.get('YearOfInstallation',None) # year of installation
    # see if material info is provided, if not, use check for stiffness description
    if material is None:
        # correction factor for ductility of pipes
        if type(pipe_stiff) is str:
            multiplier = np.asarray([0.3 if pipe_stiff == 'ductile' else 1])
        else:
            multiplier = np.ones(n_site) # default value = 1
            multiplier[pipe_stiff=='ductile'] = 0.3
            multiplier = np.transpose(np.repeat(multiplier,n_event))
    else:
        # check for material data availability
        if isinstance(material,str):
            material = ['castiron']*n_site
        else:
            # convert material property strings to lower cases
            material = [item.lower() for item in material]
        # check for joint data availability
        if joint is None or isinstance(joint,str):
            joint = ['']*n_site
        else:
            # convert joint property strings to lower cases
            joint = [item.lower() for item in joint]
        #   to add soil corrosivity later
        multiplier = np.ones(n_site) # default value = 1
        # check cases
        bool_brittle = [('cast' in item) | ('asbestos' in item) | ('concrete' in item) for item in material] # brittle pipe
        bool_ductile = [('ductile' in item) | ('steel' in item) | ('pvc' in item) for item in material] # ductile pipe
        # set values
        multiplier[bool_brittle] = 1.0
        multiplier[bool_ductile] = 0.3
        # for steel pipes specifically, check for year of installation data availability
        if year_of_install is None or isinstance(year_of_install,str):
            year_of_install = np.ones(n_site)*1930
        bool_steel_pre1935 = np.logical_and(['steel' in item for item in material],year_of_install<1935) # steel pipes pre-1935 classified as brittle
        multiplier[bool_steel_pre1935] = 1.0
    
    # repeat k1 and k2 over all scenarios ((n_site,) to (n_event,n_site))
    multiplier = np.repeat(np.expand_dims(multiplier,axis=0), n_event, axis=0)
    
    # preset dict
    if 'rr_pgv' in return_param:
        rr_pgv = {}
    if 'rr_pgd' in return_param:
        rr_pgd = {}
    # if 'rr_leak' in return_param:
        # rr_leak = {}
    # if 'rr_break' in return_param:
        # rr_break = {}

    #
    # if pgd_label is None or not 'surf' in pgd_label:
    # if pgd_label is None:
    # loop through all realizations
    for k in range(n_sample):
    
        # Repair rates by PGV
        if pgv is None:
            rr_pgv_k = sparse.coo_matrix(np.zeros([n_event,n_site]))
            # logging.debug(f"\t\tPGV is None.")
        else:
            rr_pgv_k = pgv[k].power(2.25)*0.0001 # PGV in cm/sec, repair rate in repairs/km
            # logging.debug(f"\t\tCalculated rr_pgv.")
            
        # Repair rates by PGD
        if pgd is None:
            rr_pgd_k = sparse.coo_matrix(np.zeros([n_event,n_site]))
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
            rr_pgv.update({k:rr_pgv_k.multiply(multiplier)})
        else:
            rr_pgv_k = rr_pgv_k.multiply(multiplier)
            
        if 'rr_pgd' in return_param:
            rr_pgd.update({k:rr_pgd_k.multiply(multiplier)})
        else:
            rr_pgd_k = rr_pgd_k.multiply(multiplier)
        
        # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
        # if 'rr_leak' in return_param: 
            # rr_leak.update({k:rr_pgv*0.8 + rr_pgd*0.2})
        # if 'rr_break' in return_param:
            # rr_break.update({k:rr_pgv*0.2 + rr_pgd*0.8})
                
    #else:
    #    # Repair rates by PGV
    #    rr_pgv_k = sparse.coo_matrix(np.zeros([n_event,n_site]))
    #        
    #    # Repair rates by PGD
    #    rr_pgd_k = (pgd/2.54).power(0.56) # convert PGD to inches, repair rate in repairs/km
    #        
    #    # correct for pipe_stiff
    #    if 'rr_pgv' in return_param:
    #        rr_pgv.update({'all':rr_pgv_k.multiply(multiplier)})
    #    else:
    #        rr_pgv_k = rr_pgv_k.multiply(multiplier)
    #        
    #    if 'rr_pgd' in return_param:
    #        rr_pgd.update({'all':rr_pgd_k.multiply(multiplier)})
    #    else:
    #        rr_pgd_k = rr_pgd_k.multiply(multiplier)
    #    
    #    # break rr_pgv and rr_pgd into rr_leak and rr_break and combine
    #    if 'rr_leak' in return_param: 
    #        rr_leak.update({'all':rr_pgv*0.8 + rr_pgd*0.2})
    #    if 'rr_break' in return_param:
    #        rr_break.update({'all':rr_pgv*0.2 + rr_pgd*0.8})
        
    # store outputs
    output = {}
    #
    if 'rr_pgv' in return_param:
        output.update({'rr_pgv': rr_pgv.copy()})
    if 'rr_pgd' in return_param:
        output.update({'rr_pgd': rr_pgd.copy()})
    # if 'rr_leak' in return_param:
        # output.update({'rr_leak': rr_leak.copy()})
    # if 'rr_break' in return_param:
        # output.update({'rr_break': rr_break.copy()})
    
    #
    return output