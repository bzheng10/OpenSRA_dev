# Python
import os
import numpy as np
import pandas as pd

# Java
import jpype
from jpype import imports
from jpype.types import *

# OpenSRA modules
from src.im import opensha as sha
from src import util as src_util

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
from org.opensha.sha.calc.hazardMap import HazardCurveSetCalculator
from org.opensha.sha.util import *
from scratch.UCERF3.erf.mean import MeanUCERF3

from org.opensha.sha.gcim.imr.attenRelImpl import *
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *
from org.opensha.sha.gcim.calc import *


# setup ERF
# erf_name = "Mean UCERF3"
erf_name = "Mean UCERF3 FM3.1"
# erf_name = "Mean UCERF3 FM3.2"
erf_base_dir = os.path.join(opensha_dir,"ERF")
erf_dir = os.path.join(erf_base_dir,erf_name)


# start OpenSHA instance
erf = sha.getERF(erf_name)

# time span
t = 1
ts = TimeSpan(TimeSpan.NONE, TimeSpan.YEARS)
ts.setDuration(t)
erf.setTimeSpan(ts)
erf.updateForecast()

# background
# erf.setParameter(IncludeBackgroundParam.NAME, IncludeBackgroundOption.INCLUDE)
erf.setParameter(IncludeBackgroundParam.NAME, IncludeBackgroundOption.EXCLUDE)

# get ruptures
n_sources = int(erf.getNumSources())
src_list = erf.getSourceList()

# rupture data
collect_info = [
    'getMag',
    'getMeanAnnualRate',
    'getAveRake',
    'getSectionsForRupture'
]
mag_deci = 3
angle_deci = 1
ts = 1 # year
info = {}
for each in collect_info:
    info[each] = []
for i in range(n_sources):
    meta = erf.getNthRupture(i)
    for each in collect_info:
        if each == 'getSectionsForRupture':
            info[each].append(
                np.asarray(list(erf.getSolution().getRupSet().getSectionsIndicesForRup(i)))
            )
        elif each == 'getMag':
            info[each].append(
                np.round(getattr(meta,each)(), decimals=mag_deci)
            )
        elif each == 'getAveRake':
            info[each].append(
                np.round(getattr(meta,each)(), decimals=angle_deci)
            )
        else:
            info[each].append(
                float(getattr(meta,each,)(ts))
            )

# convert to DataFrame
df_rupture_info = pd.DataFrame(list(range(n_sources)),columns=["SourceId"])
for each in info:
    header = each.replace('get','').replace('Ave','')
    df_rupture_info[header] = info[each]
df_rupture_info.MeanAnnualRate = df_rupture_info.MeanAnnualRate.apply(lambda x: '%.3e' % x)
df_rupture_info = df_rupture_info.astype({"MeanAnnualRate": float})
df_rupture_info[:10]

# export information
export_path = os.path.join(erf_dir, 'Ruptures.h5')
df_rupture_info.to_hdf(
    export_path, key='rupture', mode='w', complevel=9
)



# get sections
n_sections = erf.getSolution().getRupSet().getNumSections()

# get attributes from OpenSHA
collect_info = [
    'getSectionId',
    'getName',
    # 'getOrigAveSlipRate',
    # 'getOrigSlipRateStdDev',
    'getAveDip',
    # 'getAveRake',
    'getOrigAveUpperDepth',
    'getAveLowerDepth',
    # 'getOrigDownDipWidth',
    # 'getAseismicSlipFactor',
    # 'getCouplingCoeff',
    'getDipDirection',
    'getFaultTrace',
    'getRupturesForSection'
]
length_deci = 4
angle_deci = 1
info = {}
for each in collect_info:
    info[each] = []
for i in range(n_sections):
    meta = erf.getSolution().getRupSet().getFaultSectionData(i)
    for each in collect_info:
        if each == 'getFaultTrace':
            trace_list = []
            for loc in meta.getFaultTrace():
                trace_list.append(
                    np.round([
                        loc.getLongitude(),
                        loc.getLatitude(),
                        loc.getDepth()]
                    ,decimals=length_deci)
                )
            info[each].append(trace_list)
        elif each == 'getRupturesForSection':
            info[each].append(
                np.asarray(list(erf.getSolution().getRupSet().getRupturesForSection(i)))
            )
        elif each == 'getOrigAveUpperDepth' or each == 'getAveLowerDepth' or each == 'getOrigDownDipWidth':
            info[each].append(
                np.round(getattr(meta,each)(), decimals=length_deci)
            )
        elif each == 'getAveDip' or each == 'getDipDirection' or each == 'getAveRake':
            info[each].append(
                np.round(getattr(meta,each)(), decimals=angle_deci)
            )
        elif each == 'getName':
            info[each].append(
                str(getattr(meta,each)())
            )
        else:
            info[each].append(
                getattr(meta,each)()
            )
            
# convert to DataFrame
df_section_info = pd.DataFrame(None)
for each in info:
    header = each.replace('get','').replace('Ave','').replace('Orig','')
    df_section_info[header] = info[each]
df_section_info[:10]

# export information
export_path = os.path.join(erf_dir, 'Sections.h5')
df_section_info.to_hdf(
    export_path, key='section', mode='w', complevel=9
)