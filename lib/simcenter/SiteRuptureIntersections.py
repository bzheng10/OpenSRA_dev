#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### Get fault crossings
#####
##### Created: August 5, 2020
##### @author: Wael Elhaddad (formerly SimCenter)
##### modified: Barry Zheng
#####################################################################################################################


#####################################################################################################################
##### Python modules
import jpype, h5py, os, logging
import jpype.imports
from jpype.types import *
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import numpy as np

##### Using JPype to load EQHazard in JVM
jpype.addClassPath('EQHazard_ngawest2_noIdriss.jar')
jpype.startJVM("-Xmx8G", convertStrings=False)

##### Importing needed EQHazard and OpenSHA classes from Java 
from org.opensha.commons.geo import Location
from org.opensha.sha.earthquake import ProbEqkRupture
from org.opensha.sha.faultSurface import RuptureSurface
from org.designsafe.ci.simcenter import RegionalProcessor
from org.designsafe.ci.simcenter import EQHazardCalc


#####################################################################################################################
def load_src_rup_M_rate(file):
    """
    Load sources, ruptures, rates, and mags from pre-run. If **file** doesn't exist, compute and export
    
    Parameters
    ----------
    file : str
    file name of the hdf5 file containing the information
    
    Returns
    -------
    src : int/list
    source index used by OpenSHA
    rup : int/list
    rupture index used by OpenSHA
    M : float/list
    moment magnitude of scenario
    rate : float/list
    mean annual rate for scenario
    
    """
    
    ## see if extension is provided, if not, add it
    if not '.hdf5' in file:
        file = file+'.hdf5'
    
    ## see if file already exists
    if os.path.exists(file):
        ## load file
        with h5py.File(file, 'r') as f:
            ## pull information
            src = f.get('src')[:]
            rup = f.get('rup')[:]
            M = f.get('M')[:]
            rate = f.get('rate')[:]
        f.close()
    
    return src, rup, M, rate
    
    
#####################################################################################################################
def get_trace(rupSet, rupSourceSections, src, saveDir):
    """
    """
    
    ## define save file
    saveFile = saveDir + '/src_'+str(src)+'.txt'
    
    try:
        ## get list of segments
        listSeg = np.asarray(rupSourceSections[src].toArray())

        ## get list of nodes for all segments in current source
        nodes = []
        for j in range(len(listSeg)):
            section = rupSet.getFaultSectionData(listSeg[j])
            trace = section.getFaultTrace()
            for point in trace:
                nodes.append([point.getLongitude(),point.getLatitude(),point.getDepth()])
        nodes = np.asarray(nodes)
        
    except:
        ## point sources
        processor.setCurrentRupture(src,0)
        rupture = processor.getRupture()
        surface = rupture.getRuptureSurface()
        nodes = np.asarray([[surface.getLocation().getLongitude(), 
                            surface.getLocation().getLatitude(), 
                            surface.getLocation().getDepth()]])
    
    ## pull lon lat
    coords = np.transpose([nodes[:,0],nodes[:,1]])
    
    ## save trace into file
    np.savetxt(saveFile,nodes,fmt='%10.8f')
    
    ##
    return coords
    
    
##################################################################################################################### 
def get_intersect(src, coords, lines, saveDir):
    """
    """
    
    ## define save file
    saveFile = saveDir + '/src_'+str(src)+'.txt'

    if len(coords) == 1:
        ## point source, no intersection
        intersect = np.array([])

    else:
        ## create linestring shape using coordinates of segments
        rup_shape = LineString(coords)
        
        ## 
        intersect = [j for j, line in enumerate(lines) if line.intersects(rup_shape)]

    ## save intersections into file
    np.savetxt(saveFile,intersect,fmt='%i')

    ##
#     return intersect