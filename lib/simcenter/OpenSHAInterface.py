# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Python interface to OpenSHA
#
# Created: December 10, 2020
# @author: Wael Elhaddad (SimCenter)
#          Kuanshi Zhong (Stanford University)
#          Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import os
import logging
import json
import sys
import numpy as np
import jpype
from jpype import imports
from jpype.types import *
from scipy import sparse

# OpenSRA modules
from src.Fcn_Common import check_common_member, get_closest_pt, get_haversine_dist

# Using JPype to load OpenSHA in JVM
jpype.addClassPath('./opensha/OpenSHA-1.5.2.jar')
if not jpype.isJVMStarted():
    jpype.startJVM("-Xmx8G", convertStrings=False)

from java.io import *
from java.lang import *
from java.lang.reflect import *
from java.util import ArrayList

from org.opensha.commons.data import *
from org.opensha.commons.data.siteData import *
from org.opensha.commons.data.function import *
from org.opensha.commons.exceptions import ParameterException
from org.opensha.commons.geo import *
from org.opensha.commons.param import *
from org.opensha.commons.param.event import *
from org.opensha.commons.param.constraint import *
from org.opensha.commons.util import ServerPrefUtils

from org.opensha.sha.earthquake import *
from org.opensha.sha.earthquake.param import *
from org.opensha.sha.earthquake.rupForecastImpl.Frankel02 import Frankel02_AdjustableEqkRupForecast
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF1 import WGCEP_UCERF1_EqkRupForecast
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final import UCERF2
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final.MeanUCERF2 import MeanUCERF2
from org.opensha.sha.faultSurface import *
from org.opensha.sha.imr import *
from org.opensha.sha.imr.attenRelImpl import *
from org.opensha.sha.imr.attenRelImpl.ngaw2 import *
from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *
from org.opensha.sha.imr.param.IntensityMeasureParams import *
from org.opensha.sha.imr.param.OtherParams import *
from org.opensha.sha.imr.param.SiteParams import Vs30_Param
from org.opensha.sha.calc import *
from org.opensha.sha.util import *
from scratch.UCERF3.erf.mean import MeanUCERF3

from org.opensha.sha.gcim.imr.attenRelImpl import *
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *
from org.opensha.sha.gcim.calc import *


# -----------------------------------------------------------
def setup_opensha(setup_config, other_config_param, site_data):
    """
    Sets up interface to OpenSHA to retrieve GM predictions
    
    """
    
    # Get Locations
    locs = site_data[['Latitude','Longitude']].values
    vs30 = site_data['VS30 (m/s)'].values
    
    # Initialize ERF
    logging.info(f"\n\n     *****Runtime messages from OpenSHA*****\n")
    erf_name = setup_config['IntensityMeasure']['SourceParameters']['SeismicSourceModel']
    erf = getERF(erf_name, update_flag=True)
    logging.info(f"\n\n     *****Runtime messages from OpenSHA*****\n")
    logging.info(f'\t... initialized ERF "{erf_name}"')
    
    # Initialize IMR
    gmpe_name = setup_config['IntensityMeasure']['SourceParameters']['GroundMotionModel']
    try:
        imr = CreateIMRInstance(gmpe_name)
        logging.info(f'\t... created instance of IMR "{gmpe_name}"')
    except:
        print('Please check GMPE name.')
        return 1, station_info
    
    # Set user-defined Vs30
    siteSpec = [{
        'Location': {
            'Latitude': locs[i,0],
            'Longitude': locs[i,1]
        },
        'Vs30': vs30[i]
    } for i in range(len(locs))]
    sites = get_site_prop(imr, siteSpec)
    logging.info(f"\t... set Vs30 to sites")
    
    # Set max distance
    if setup_config['IntensityMeasure']['SourceParameters']['Filter']['Distance']['ToInclude']:
        r_max = setup_config['IntensityMeasure']['SourceParameters']['Filter']['Distance']['Maximum']
        imr.setUserMaxDistance(r_max)
        logging.info(f"\t... set max distance to {r_max} km")
    
    #
    return erf, imr, sites


# -----------------------------------------------------------
def filter_ruptures(erf, locs, filter_criteria, rupture_list, save_name,
                    rup_seg_file, pt_src_file):
    """
    Filter rupture scenarios; locs should be in [lon,lat] order
    
    """
    
    # convert to np array
    rupture_list = np.asarray(rupture_list)

    # filter by rate_cutoff (or return period)
    if 'ReturnPeriod' in filter_criteria:
        rate_cutoff = 1/filter_criteria['ReturnPeriod']['Maximum']
        # rupture_list = np.asarray([row for row in rupture_list if row[3] >= rate_cutoff])
        rupture_list = rupture_list[rupture_list[:,3]>=rate_cutoff]
        logging.info(f"\t\t... filtered scenarios with mean annual rates less than {rate_cutoff}")
        logging.info(f"\t\t\t  length after filter by rate = {len(rupture_list)}")

    # filter by magnitude range
    if 'Magnitude' in filter_criteria:
        mag_min = filter_criteria['Magnitude']['Minimum']
        mag_max = filter_criteria['Magnitude']['Maximum']
        # rupture_list = np.asarray([row for row in rupture_list if row[2] >= mag_min and row[2] <= mag_max])
        rupture_list = rupture_list[np.where(logical_and(rupture_list[:,2]>=mag_min,rupture_list[:,2]<=mag_max))]
        logging.info(f"\t\t... filtered scenarios with moment magnitudes outside of {mag_min} and {mag_max}")
        logging.info(f"\t\t\t  length after filter by mag = {len(rupture_list)}")

    # filter by point source
    list_pt_src = np.loadtxt(pt_src_file) # read list of point sources in source model
    if not 'PointSource' in filter_criteria:
        rupture_list = rupture_list[np.where(np.in1d(rupture_list[:,0], list_pt_src[:,0],invert=True))[0]]
        logging.info(f"\t\t... point sources removed from list of scenarios")
        logging.info(f"\t\t\t  length after filter by pt src = {len(rupture_list)}")

    # read list of rupture segments in source model
    if 'Distance' in filter_criteria:
        #
        r_max = filter_criteria['Distance']['Maximum']
        with open(rup_seg_file, 'r') as read_file:
            list_rup_seg = json.load(read_file)
        read_file.close()
        # check site locations against all rupture segments and see if shortest distance is within r_max
        seg_pass_r_max = []
        for seg in list_rup_seg:
            flag_r_max = check_r_max(locs,list_rup_seg[seg]['trace'],'seg',r_max)
            if flag_r_max == True:
                seg_pass_r_max.append(int(seg))
        logging.info(f"\t\t... obtained list of {len(seg_pass_r_max)} segments that are within {r_max} km of site locations")
    
        # check if point sources are needed
        pt_src_pass_r_max = []
        if 'PointSource' in filter_criteria:
            # check site locations against all point sources and see if shortest distance is within r_max
            for pt_src in list_pt_src:
                flag_r_max = check_r_max(locs,pt_src[1:3],'pt',r_max)
                if flag_r_max == True:
                    pt_src_pass_r_max.append(int(pt_src[0]))
            logging.info(f"\t\t... obtained list of {len(pt_src_pass_r_max)} point sources that are within {r_max} km of site locations")

        # get rupture section class from OpenSHA through EQHazard
        _, rupSourceSections = get_rupture_set(erf)
        
        # compare list of rupture segments that are wihtin r_max and with the rupture segments in each source
        rupture_list_temp = []
        for i in range(len(rupture_list)):
            try:
                # get list of rupture segments for current source index
                listSeg = list(rupSourceSections[int(rupture_list[i,0])].toArray())
                # check for common members of rupture segments
                if check_common_member(listSeg,seg_pass_r_max):
                    rupture_list_temp.append(rupture_list[i])
            except:
                # check source index for point source with those that pass r_max
                if rupture_list[i,0] in pt_src_pass_r_max:
                    rupture_list_temp.append(rupture_list[i])
        rupture_list = np.asarray(rupture_list_temp)
        logging.info(f"\t\t... filtered scenarios that are more than {r_max} km from sites")
        logging.info(f"\t\t\t  length after filter by rmax = {len(rupture_list)}")
    
    # store filtered list of rupture metainfo
    if '.txt' in save_name:
        if len(rupture_list) > 0:
            np.savetxt(save_name, rupture_list, fmt='%i %i %6.3f %6.3e')
        else:
            np.savetxt(save_name, rupture_list)
    else:
        logging.info('\t\tlimited to ".txt" output file type only')

    logging.info(f"\t... filtered rupture scenarios by rmax and exported to:")
    logging.info(f"\t\t{save_name,}")
    
    # create return dictionary
    output = {
        'src':rupture_list[:,0].astype(np.int32),
        'rup':rupture_list[:,1].astype(np.int32),
        'mag':rupture_list[:,2],
        'rate':rupture_list[:,3]}
        
    #
    return output
    

# -----------------------------------------------------------
def check_r_max(locs, seg_trace, seg_type, r_max):
    """
    check for list of rupture segments that are within r_max; locs in (lon, lat)
    
    """

    #
    if seg_type == 'seg': # finite source
        for i in range(len(seg_trace)-1):
            seg_sub = [seg_trace[i][:2],seg_trace[i+1][:2]]
            for site_j in locs:
                _,dist = get_closest_pt(site_j,seg_sub)
                if dist <= r_max:
                    return True
        return False

    elif seg_type == 'pt': # point source
        for site_j in locs:
            dist = get_haversine_dist(site_j[0],site_j[1],seg_trace[0],seg_trace[1])
            if dist <= r_max:
                return True
        return False
    

# -----------------------------------------------------------
def get_rupture_set(erf):
    #Read the fault system solution
    sol = erf.getSolution()

    #Read the rupture set
    rupSet = sol.getRupSet()

    #Get the number of section
    numSections = rupSet.getNumSections()

    #Map rupture sources to sections
    rupSourceSections = rupSet.getSectionIndicesForAllRups()
    
    return rupSet, rupSourceSections


# -----------------------------------------------------------
def get_src_rup_M_rate(erf=None, rupture_list_file=None, ind_range=['all'], rate_cutoff=None,
                        rup_group_file=None, rup_per_group=None, file_type='txt'):
    """
    Gets sources, ruptures, rates, and mags from OpenSHA and metadata exported. If metadata file already exists, use existing.
    
    Parameters
    ----------
    rupture_list_file : str
        file name of the file containing rupture metadata
    proc : object
        OpenSHA processor object; used if **rupture_list_file** does not exist to obtain source parameters
    ind_range : str/list, optional
        define indices to extract source parameters for: options are **['all']**, **[index]** or **[low, high]** (brackets are required; replace index, low, and high with integers)
    
    Returns
    -------
    gm_source_info : dictionary
        contains list of source indices, rupture indices, moment magnitudes, and mean annual rates
    
    """
    
    # see rupture_list_file is given
    if rupture_list_file is None:
        # get and store rates and Mw
        nSources = erf.getNumSources()
        if rate_cutoff is None: # no cutoff on rate
            src_rup = [[i, j]+get_mag_rate(erf,i,j)
                        for i in range(nSources) for j in range(erf.getNumRuptures(i))]
        else: # apply rate_cutoff
            src_rup = [[i, j]+get_mag_rate(erf,i,j)
                        for i in range(nSources) for j in range(erf.getNumRuptures(i))
                        if get_mag_rate(erf, i,j)[1] > rate_cutoff]
        #
        return src_rup
    else:
        # see if extension is provided, if not, add it
        if not '.'+file_type in rupture_list_file:
            rupture_list_file = rupture_list_file+'.'+file_type

        # initialize dictionaries
        src_rup = {'src':None,'rup':None,'mag':None,'rate':None}
        # load rupture_list_file
        # txt file format
        if 'txt' in file_type:
            f = np.loadtxt(rupture_list_file,unpack=True)
            if len(ind_range) == 1:
                if ind_range[0] == 'all':
                    src_rup['src'] = f[0].astype(np.int32)
                    src_rup['rup'] = f[1].astype(np.int32)
                    src_rup['mag'] = f[2]
                    src_rup['rate'] = f[3]
                else:
                    src_rup['src'] = f[0,ind_range[0]].astype(np.int32)
                    src_rup['rup'] = f[1,ind_range[0]].astype(np.int32)
                    src_rup['mag'] = f[2,ind_range[0]]
                    src_rup['rate'] = f[3,ind_range[0]]
            elif len(ind_range) == 2:
                src_rup['src'] = f[0,ind_range[0]:ind_range[1]].astype(np.int32)
                src_rup['rup'] = f[1,ind_range[0]:ind_range[1]].astype(np.int32)
                src_rup['mag'] = f[2,ind_range[0]:ind_range[1]]
                src_rup['rate'] = f[3,ind_range[0]:ind_range[1]]
        #
        return src_rup
 
    
# -----------------------------------------------------------
def get_mag_rate(erf, src, rup):
    """
    This extracts the mean annual rate and moment magnitude for target scenario (source + rupture index)
    
    Parameters
    ----------
    erf : object
        OpenSHA ERF object
    src : int
        source index used by OpenSHA
    rup : int
        rupture index used by OpenSHA
        
    Returns
    -------
    mag : float
        moment magnitude of scenario
    rate : float
        mean annual rate for scenario
    
    """
    # Set current scenario
    rup = erf.getSource(src).getRupture(rup)
    # Return mean annual rate and moment magnitude
    return [rup.getMag(), rup.getMeanAnnualRate(1)]


# -----------------------------------------------------------
def get_IM(erf, imr, sites, src_list, rup_list, list_im=['PGA','PGV'],
    saveDir=None, store_file_type='txt', r_max=10000):
    """
    Get IM from OpenSHA; developed by Kuanshi Zhong (SimCenter), modified by Barry Zheng
    
    """
    
    # IM param map to save name
    list_param = {
        'Mean': 'mean',
        'InterEvStdDev': 'stdev_inter',
        'IntraEvStdDev': 'stdev_intra',
        'TotalStdDev': 'stdev_total'}
    # Get available stdDev options
    try:
        stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
        hasIEStats = stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTER) and \
            stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA)
    except:
        stdDevParam = None
        hasIEStats = False
    # Set up output
    shape = (len(src_list),len(sites))
    output = {}
    for im in list_im:
        output[im] = {'Mean': np.zeros(shape)}
        output[im].update({'TotalStdDev': np.zeros(shape)})
        if hasIEStats:
            output[im].update({'InterEvStdDev': np.zeros(shape)})
            output[im].update({'IntraEvStdDev': np.zeros(shape)})
    # Loop through scenarios
    for i in range(len(src_list)):
        # Setting up imr
        currentRupture = erf.getSource(src_list[i]).getRupture(rup_list[i])
        imr.setEqkRupture(currentRupture)
        # Looping over sites
        for j in range(len(sites)):
            # Set up the site in the imr
            imr.setSite(sites[j])
            # Get distance with rupture
            DistanceRup = imr.getStdDevIndependentParams().getValue('DistanceRup')
            DistanceJB = imr.getStdDevIndependentParams().getValue('DistanceJB')
            # loop through IM
            for im in list_im:
                # Check if distances are within r_max, and get value only if true
                if DistanceRup <= r_max and DistanceJB <= r_max:                    
                    imr.setIntensityMeasure(im)
                    output[im]['Mean'][i,j] = float(np.exp(imr.getMean()))
                    output[im]['TotalStdDev'][i,j] = float(imr.getStdDev())
                    if hasIEStats:
                        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
                        output[im]['InterEvStdDev'][i,j] = float(imr.getStdDev())
                        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
                        output[im]['IntraEvStdDev'][i,j] = float(imr.getStdDev())
    # Convert to COO matrix and export if saveDir is given
    for im_i in list_im:
        for param_j in output[im_i].keys():
            output[im_i][param_j] = sparse.coo_matrix(output[im_i][param_j])
            if saveDir is not None:
                saveName = os.path.join(saveDir,im_i,list_param[param_j]+'.'+store_file_type)
                np.savetxt(saveName,output[im_i][param_j].toarray(),fmt='%5.3f')
    #
    return output
    
    
# -----------------------------------------------------------
def get_site_prop(imr, siteSpec):
    """
    Get and update site properties; developed by Kuanshi Zhong (SimCenter), modified by Barry Zheng
    
    """

    # GMPE
    # try:
        # imr = CreateIMRInstance(gmpe_name)
    # except:
        # print('Please check GMPE name.')
        # return 1
    # Site data
    sites = ArrayList()
    for cur_site in siteSpec:
        cur_loc = Location(cur_site['Location']['Latitude'], cur_site['Location']['Longitude'])
        sites.add(Site(cur_loc))
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()
    try:
        availableSiteData = siteDataProviders.getAllAvailableData(sites)
    except:
        print('Error in getAllAvailableData')
        return 1
    siteTrans = SiteTranslator()
    # Looping over all sites
    # site_prop = []
    for i in range(len(siteSpec)):
        site_tmp = dict()
        # Current site
        site = sites.get(i)
        # Location
        cur_site = siteSpec[i]
        locResults = {'Latitude': cur_site['Location']['Latitude'],
                      'Longitude': cur_site['Location']['Longitude']}
        cur_loc = Location(cur_site['Location']['Latitude'], cur_site['Location']['Longitude'])
        siteDataValues = ArrayList()
        for j in range(len(availableSiteData)):
            siteDataValues.add(availableSiteData.get(j).getValue(i))
        imrSiteParams = imr.getSiteParams()
        siteDataResults = []
        # Setting site parameters
        for j in range(imrSiteParams.size()):
            siteParam = imrSiteParams.getByIndex(j)
            newParam = Parameter.clone(siteParam)
            siteDataFound = siteTrans.setParameterValue(newParam, siteDataValues)
            if (str(newParam.getName())=='Vs30' and bool(cur_site.get('Vs30', None))):
                newParam.setValue(Double(cur_site['Vs30']))
                siteDataResults.append({'Type': 'Vs30',
                                        'Value': float(newParam.getValue()),
                                        'Source': 'User Defined'})
            elif (str(newParam.getName())=='Vs30 Type' and bool(cur_site.get('Vs30', None))):
                newParam.setValue("Measured")
                siteDataResults.append({'Type': 'Vs30 Type',
                                        'Value': 'Measured',
                                        'Source': 'User Defined'})
            elif siteDataFound:
                provider = "Unknown"
                provider = get_DataSource(newParam.getName(), siteDataValues)
                if 'String' in str(type(newParam.getValue())):
                    tmp_value = str(newParam.getValue())
                elif 'Double' in str(type(newParam.getValue())):
                    tmp_value = float(newParam.getValue())
                    if str(newParam.getName())=='Vs30':
                            cur_site.update({'Vs30': tmp_value})
                else:
                    tmp_value = str(newParam.getValue())
                siteDataResults.append({'Type': str(newParam.getName()),
                                        'Value': tmp_value,
                                        'Source': str(provider)})
            else:
                newParam.setValue(siteParam.getDefaultValue())
                siteDataResults.append({'Type': str(siteParam.getName()),
                                        'Value': float(siteParam.getDefaultValue()),
                                        'Source': 'Default'})
            site.addParameter(newParam)
            # End for j
        # Updating site specifications
        # siteSpec[i] = cur_site
        site_tmp.update({'Location': locResults,
                         'SiteData': siteDataResults})
        # site_prop.append(site_tmp)

    # Return
    # return siteSpec, sites, site_prop
    return sites


# -----------------------------------------------------------
def get_DataSource(paramName, siteData):
    """
    Fetch data source; developed by Kuanshi Zhong (SimCenter)
    
    """
    
    typeMap = SiteTranslator.DATA_TYPE_PARAM_NAME_MAP
    for dataType in typeMap.getTypesForParameterName(paramName):
        if dataType == SiteData.TYPE_VS30:
            for dataValue in siteData:
                if dataValue.getDataType() != dataType:
                    continue
                vs30 = Double(dataValue.getValue())
                if (not vs30.isNaN()) and (vs30 > 0.0):
                    return dataValue.getSourceName()
        elif (dataType == SiteData.TYPE_DEPTH_TO_1_0) or (dataType == SiteData.TYPE_DEPTH_TO_2_5):
             for dataValue in siteData:
                if dataValue.getDataType() != dataType:
                    continue
                depth = Double(dataValue.getValue())
                if (not depth.isNaN()) and (depth > 0.0):
                    return dataValue.getSourceName()
    return 1


# -----------------------------------------------------------
def getERF(erf_name, update_flag):
    """
    get ERF; developed by Kuanshi Zhong (SimCenter); modified by Barry Zheng
    
    """
    
    # Available SSC and names to use in setup configuration:
    #     WGCEP (2007) UCERF2 - Single Branch
    #     USGS/CGS 2002 Adj. Cal. ERF
    #     WGCEP UCERF 1.0 (2005)
    #     Mean UCERF3
    #     Mean UCERF3 FM3.1
    #     Mean UCERF3 FM3.2
    #     WGCEP Eqk Rate Model 2 ERF
    
    # Initialization
    erf = None
    # ERF model options
    if erf_name == 'WGCEP (2007) UCERF2 - Single Branch':
        erf = MeanUCERF2()
    elif erf_name == 'USGS/CGS 2002 Adj. Cal. ERF':
        erf = Frankel02_AdjustableEqkRupForecast()
    elif erf_name == 'WGCEP UCERF 1.0 (2005)':
        erf = WGCEP_UCERF1_EqkRupForecast()
    elif erf_name == 'Mean UCERF3':
        tmp = MeanUCERF3()
        tmp.setPreset(MeanUCERF3.Presets.BOTH_FM_BRANCH_AVG)
        erf = tmp
        del tmp
    elif erf_name == 'Mean UCERF3 FM3.1':
        tmp = MeanUCERF3()
        tmp.setPreset(MeanUCERF3.Presets.FM3_1_BRANCH_AVG)
        erf = tmp
        del tmp
    elif erf_name == 'Mean UCERF3 FM3.2':
        tmp = MeanUCERF3()
        tmp.setPreset(MeanUCERF3.Presets.FM3_2_BRANCH_AVG)
        erf = tmp
        del tmp
    elif erf_name == 'WGCEP Eqk Rate Model 2 ERF':
        erf = UCERF2()
    else:
        print('Please check the ERF model name.')

    if erf_name and update_flag:
        erf.updateForecast()
    # return
    return erf
    

# -----------------------------------------------------------
def CreateIMRInstance(gmpe_name):
    """
    create IMR instance; developed by Kuanshi Zhong (SimCenter); modified by Barry Zheng
    
    """
    
    # Available GMMs and names to use in setup configuration:
    #     Abrahamson, Silva & Kamai (2014)
    #     Boore, Stewart, Seyhan & Atkinson (2014)
    #     Campbell & Bozorgnia (2014)
    #     Chiou & Youngs (2014)
    #     Bommer et al. (2009)
    #     Afshari & Stewart (2016)
    #     NGAWest2 2014 Averaged Attenuation Relationship
    #     NGAWest2 2014 Averaged No Idriss
    
    # GMPE name map
    gmpe_map = {str(ASK_2014.NAME): ASK_2014_Wrapper.class_.getName(),
                str(BSSA_2014.NAME): BSSA_2014_Wrapper.class_.getName(),
                str(CB_2014.NAME): CB_2014_Wrapper.class_.getName(),
                str(CY_2014.NAME): CY_2014_Wrapper.class_.getName(),
                str(KS_2006_AttenRel.NAME): KS_2006_AttenRel.class_.getName(),
                str(BommerEtAl_2009_AttenRel.NAME): BommerEtAl_2009_AttenRel.class_.getName(),
                str(AfshariStewart_2016_AttenRel.NAME): AfshariStewart_2016_AttenRel.class_.getName(),
                str(NGAWest_2014_Averaged_AttenRel.NAME): NGAWest_2014_Averaged_AttenRel.class_.getName(),
                str(NGAWest_2014_Averaged_AttenRel.NGAWest_2014_Averaged_AttenRel_NoIdriss.NAME): NGAWest_2014_Averaged_AttenRel.NGAWest_2014_Averaged_AttenRel_NoIdriss.class_.getName()}
    # Check if NGAWest2 average relationship is required
    if 'NGAWest2 2014 Averaged' in gmpe_name:
        # Different initialization for NGAWest2 Average IMR
        # Second arg in class call = boolean for Idriss
        if 'No Idriss' in gmpe_name:
            imr = NGAWest_2014_Averaged_AttenRel(None, False)
        else:
            imr = NGAWest_2014_Averaged_AttenRel(None, True)
    else:
        # Mapping GMPE name
        imrClassName = gmpe_map.get(gmpe_name, None)
        if imrClassName is None:
            return imrClassName
        # Getting the java class
        imrClass = Class.forName(imrClassName)
        ctor = imrClass.getConstructor()
        imr = ctor.newInstance()
    # Setting default parameters
    imr.setParamDefaults()
    # return
    return imr
