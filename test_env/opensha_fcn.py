# import os, logging, json, sys, importlib, time, pathlib, queue
import os
import numpy as np
import pandas as pd
# from scipy import sparse
# import pyproj
# from shapely.geometry import LineString, Point, MultiPoint, MultiLineString, Polygon, MultiPolygon, box
# from shapely.ops import unary_union, linemerge, transform, polygonize, cascaded_union
# from shapely.strtree import STRtree
# from rtree import index
# from shapely.prepared import prep
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# import itertools
import jpype
from jpype import imports
from jpype.types import *
# import numba
# from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


# OpenSRA modules
# from src.util import check_common_member, get_closest_pt, get_haversine_dist
# from src.im import opensha as sha

# Using JPype to load OpenSHA in JVM
opensha_dir = os.path.join(os.path.dirname(os.getcwd()),'OpenSRA','lib','OpenSHA')
jpype.addClassPath(os.path.join(opensha_dir,'OpenSHA-1.5.2.jar'))
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
from org.opensha.sha.faultSurface.cache import SurfaceCachingPolicy
from org.opensha.sha.imr import *
from org.opensha.sha.imr.attenRelImpl import *
from org.opensha.sha.imr.attenRelImpl.ngaw2 import *
from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *
from org.opensha.sha.imr.param.IntensityMeasureParams import *
from org.opensha.sha.imr.param.OtherParams import *
from org.opensha.sha.imr.param.SiteParams import *
from org.opensha.sha.calc import *
from org.opensha.sha.calc.params import *
from org.opensha.sha.util import *
from scratch.UCERF3.erf.mean import MeanUCERF3

from org.opensha.sha.gcim.imr.attenRelImpl import *
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *
from org.opensha.sha.gcim.calc import *



def get_erf(erf_name):

    """
    get ERF; developed by Kuanshi Zhong (SimCenter); modified by Barry Zheng
    
    """
    
    # from org.opensha.sha.earthquake.rupForecastImpl.Frankel02 import Frankel02_AdjustableEqkRupForecast
    # from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF1 import WGCEP_UCERF1_EqkRupForecast
    # from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final import UCERF2
    # from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final.MeanUCERF2 import MeanUCERF2
    
    
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
    elif erf_name.startswith('Mean UCERF3'):
        erf = MeanUCERF3()
        if erf_name.endswith('FM3.1'): # Branch 3.1
            erf.setPreset(MeanUCERF3.Presets.FM3_1_BRANCH_AVG)
        elif erf_name.endswith('FM3.2'): # Branch 3.2
            erf.setPreset(MeanUCERF3.Presets.FM3_2_BRANCH_AVG)
        else: # Unspecified (both branches)
            erf.setPreset(MeanUCERF3.Presets.BOTH_FM_BRANCH_AVG)
    elif erf_name == 'WGCEP Eqk Rate Model 2 ERF':
        erf = UCERF2()
    else:
        raise ValueError(f'The ERF model "{erf_name}" is not supported.')
    # update forecast
    erf.updateForecast()
    # return
    return erf


def get_imr(imr_name):
    """
    create IMR instance; developed by Kuanshi Zhong (SimCenter); modified by Barry Zheng
    
    """
    # from org.opensha.sha.imr.attenRelImpl import *
    # from org.opensha.sha.imr.attenRelImpl.ngaw2 import *
    # from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *
    
    # Available GMMs and names to use in setup configuration:
    #     Abrahamson, Silva & Kamai (2014)
    #     Boore, Stewart, Seyhan & Atkinson (2014)
    #     Campbell & Bozorgnia (2014)
    #     Chiou & Youngs (2014)
    #     Idriss (2014)
    #     Bommer et al. (2009)
    #     Afshari & Stewart (2016)
    #     NGAWest2 2014 Averaged Attenuation Relationship
    #     NGAWest2 2014 Averaged No Idriss
    
    # GMPE name map
    imr_map = {
        str(ASK_2014.NAME): ASK_2014_Wrapper.class_.getName(),
        str(BSSA_2014.NAME): BSSA_2014_Wrapper.class_.getName(),
        str(CB_2014.NAME): CB_2014_Wrapper.class_.getName(),
        str(CY_2014.NAME): CY_2014_Wrapper.class_.getName(),
        str(Idriss_2014.NAME): Idriss_2014_Wrapper.class_.getName(),
        str(KS_2006_AttenRel.NAME): KS_2006_AttenRel.class_.getName(),
        str(BommerEtAl_2009_AttenRel.NAME): BommerEtAl_2009_AttenRel.class_.getName(),
        str(AfshariStewart_2016_AttenRel.NAME): AfshariStewart_2016_AttenRel.class_.getName(),
        str(NGAWest_2014_Averaged_AttenRel.NAME): NGAWest_2014_Averaged_AttenRel.class_.getName(),
        str(NGAWest_2014_Averaged_AttenRel.NGAWest_2014_Averaged_AttenRel_NoIdriss.NAME): NGAWest_2014_Averaged_AttenRel.NGAWest_2014_Averaged_AttenRel_NoIdriss.class_.getName()
    }
    # Check if NGAWest2 average relationship is requested
    if 'NGAWest2 2014 Averaged' in imr_name:
        # Different initialization for NGAWest2 Average IMR
        # Second arg in class call = boolean for Idriss
        if 'No Idriss' in imr_name:
            imr = NGAWest_2014_Averaged_AttenRel(None, False)
        else:
            imr = NGAWest_2014_Averaged_AttenRel(None, True)
    else:
        # Mapping GMPE name
        imrClassName = imr_map.get(imr_name, None)
        if imrClassName is None:
            raise ValueError(f'The GM model "{imr_name}" is not supported.')
            # return imrClassName
        # Getting the java class
        imrClass = Class.forName(imrClassName)
        ctor = imrClass.getConstructor()
        imr = ctor.newInstance()
    # Setting default parameters
    imr.setParamDefaults()
    # return
    return imr
    

def get_im(erf, imr, sites, src_list, rup_list, site_range_to_run=None, dist_max=200, id=1):
    """
    """
    if site_range_to_run is None:
        site_range_to_run = [0, len(sites)]
        
    n_sites = site_range_to_run[1] - site_range_to_run[0]
    n_sources = len(src_list)
    print(id, site_range_to_run, n_sites, imr.getName())
    
    im_list = ['PGA','PGV']
    output = {}
    for im in im_list:
        output[im] = {
            'Mean': np.ones((n_sources,n_sites))*-10,
            'TotalStdDev': np.ones((n_sources,n_sites))*0,
            'InterEvStdDev': np.ones((n_sources,n_sites))*0,
            'IntraEvStdDev': np.ones((n_sources,n_sites))*0
        }
    
    # see if inter and intra sigmas are available
    stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
    if stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTER) and \
        stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA):
        hasIEStats = True
    else:
        hasIEStats = False
        
    for j in range(n_sites):
        site_ind = site_range_to_run[0]+j
        
        # Set up the site in the imr
        imr.setSite(sites[site_ind])
        
        # Loop through scenarios
        for i in range(len(src_list)):
            # Setting up imr
            currentRupture = erf.getSource(src_list[i]).getRupture(rup_list[i])
            imr.setEqkRupture(currentRupture)
            
            # Get distance with rupture
            try:
                DistanceRup = imr.getStdDevIndependentParams().getValue('DistanceRup')
            except jpype.JException as exception:
                DistanceRup = np.nan
            try:
                DistanceJB = imr.getStdDevIndependentParams().getValue('DistanceJB')
            except jpype.JException as exception:
                DistanceJB = np.nan
                
            # Get max distance from rupture and JB distance
            if DistanceRup > dist_max or DistanceJB > dist_max or (DistanceRup is None and DistanceJB is None):
                # for im in im_list:
                #     output[im]['Mean'][i,j] = -10
                #     output[im]['TotalStdDev'][i,j] = 0
                #     output[im]['InterEvStdDev'][i,j] = 0
                #     output[im]['IntraEvStdDev'][i,j] = 0
                pass
            else:
                # loop through IM
                for im in im_list:
                    imr.setIntensityMeasure(im)
                    output[im]['Mean'][i,j] = float(imr.getMean())
                    output[im]['TotalStdDev'][i,j] = float(imr.getStdDev())
                    if hasIEStats:
                        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
                        output[im]['InterEvStdDev'][i,j] = float(imr.getStdDev())
                        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
                        output[im]['IntraEvStdDev'][i,j] = float(imr.getStdDev())
                        
    return output