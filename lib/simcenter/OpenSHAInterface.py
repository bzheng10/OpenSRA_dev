#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
#####
##### Copyright(c) 2020-2022 The Regents of the University of California and
##### Slate Geotechnical Consultants. All Rights Reserved.
#####
##### Python wrapper for RegionalProcessor
#####
##### Created: August 5, 2020
##### @author: Wael Elhaddad (formerly SimCenter)
##### modified: Barry Zheng
#####################################################################################################################


#####################################################################################################################
##### Python modules
import jpype, os, logging, json, sys
import jpype.imports
from jpype.types import *
import numpy as np
from scipy import sparse
from shapely.geometry import LineString
# import h5py

##### OpenSRA modules
from src.fcn_gen import common_member, get_closest_pt, get_haversine_dist

##### Using JPype to load EQHazard in JVM
jpype.addClassPath('EQHazard_ngawest2_noIdriss.jar')
if not jpype.isJVMStarted():
    jpype.startJVM("-Xmx8G", convertStrings=False)

##### Importing needed EQHazard and OpenSHA classes from Java
from org.opensha.commons.geo import Location
from org.opensha.sha.earthquake import ProbEqkRupture
from org.designsafe.ci.simcenter import RegionalProcessor
from org.opensha.sha.faultSurface import RuptureSurface
from org.designsafe.ci.simcenter import EQHazardCalc



#####################################################################################################################
def filter_src_by_rmax(site_loc, rmax, rup_meta_file_tr, rup_meta_file_tr_rmax,
                        rup_seg_file, pt_src_file, rup_group_file, rup_per_group,
                        flag_include_point_source=True, file_type='txt'):

    ## read list of rupture segments in source model
    with open(rup_seg_file, 'r') as read_file:
        list_rup_seg = json.load(read_file)
    read_file.close()
    
    ## read list of point sources in source model
    list_pt_src = np.loadtxt(pt_src_file)
    
    ## check site locations against all rupture segments and see if shortest distance is within rmax
    seg_pass_rmax = []
    for seg in list_rup_seg:
        flag_rmax = check_rmax(site_loc,list_rup_seg[seg]['trace'],'seg',rmax)
        if flag_rmax == True:
            seg_pass_rmax.append(int(seg))
    logging.info(f"\tObtained list of {len(seg_pass_rmax)} segments that are within {rmax} km of site locations")
    
    ## check if point sources are needed
    if flag_include_point_source:
        ## check site locations against all point sources and see if shortest distance is within rmax
        pt_src_pass_rmax = []
        for pt_src in list_pt_src:
            flag_rmax = check_rmax(site_loc,pt_src[1:3],'pt',rmax)
            if flag_rmax == True:
                pt_src_pass_rmax.append(int(pt_src[0]))
        logging.info(f"\tObtained list of {len(pt_src_pass_rmax)} point sources that are within {rmax} km of site locations")
    else:
        logging.info(f"\tPoint sources not included")
    
    ## import list of source indices filtered by return period only
    gm_source_info = load_src_rup_M_rate(rup_meta_file_tr, ['all'], None)
    src_list = gm_source_info['src']
    rup_list = gm_source_info['rup']
    M_list = gm_source_info['M']
    rate_list = gm_source_info['rate']
    
    ## get rupture section class from OpenSHA through EQHazard
    _, rupSourceSections = set_up_get_rup()
    
    ## compare list of rupture segments that are wihtin rmax and with the rupture segments in each source
    rup_meta_tr_rmax = []
    for i in range(len(src_list)):
        try:
            ## get list of rupture segments for current source index
            listSeg = list(rupSourceSections[src_list[i]].toArray())

            ## check for common members of rupture segments
            if common_member(listSeg,seg_pass_rmax):
                rup_meta_tr_rmax.append([src_list[i],rup_list[i],M_list[i],rate_list[i]])
        except:
            ## check if point sources are needed
            if flag_include_point_source:
                ## check source index for point source with those that pass rmax
                if src_list[i] in pt_src_pass_rmax:
                    rup_meta_tr_rmax.append([src_list[i],rup_list[i],M_list[i],rate_list[i]])
            else:
                pass
    logging.info(f"\tObtained list of {len(rup_meta_tr_rmax)} sources indices that are within {rmax} km of site locations")

    ## store filtered list of rupture metainfo
    ## to txt format
    if '.'+file_type in rup_meta_file_tr_rmax:
        np.savetxt(rup_meta_file_tr_rmax, rup_meta_tr_rmax, fmt='%i %i %6.4f %6.4e')

    ## to hdf5 format
    elif '.'+file_type in rup_meta_file_tr_rmax:
        # rup_meta_tr_rmax = np.transpose(np.asarray(rup_meta_tr_rmax))
        rup_meta_tr_rmax = np.transpose(rup_meta_tr_rmax)
        ##
        with h5py.File(rup_meta_file_tr_rmax, 'w') as f:
            dset1 = f.create_dataset('src', data=rup_meta_tr_rmax[0].astype(np.int32))
            dset2 = f.create_dataset('rup', data=rup_meta_tr_rmax[1].astype(np.int32))
            dset3 = f.create_dataset('M', data=rup_meta_tr_rmax[2])
            dset4 = f.create_dataset('rate', data=rup_meta_tr_rmax[3])
        f.close()
    
    ## generate and store list of rupture groups
    n_rup_groups = int(np.ceil(len(rup_meta_tr_rmax)/rup_per_group))
    print(len(rup_meta_tr_rmax),rup_per_group)
    print(n_rup_groups)
    list_rup_group = [str(rup_per_group*i)+'_'+str(rup_per_group*(i+1)-1) for i in range(n_rup_groups)]
    np.savetxt(rup_group_file,list_rup_group,fmt='%s')
    logging.info(f"\tNumber of rupture groups = {n_rup_groups}")


#####################################################################################################################
##### check for list of rupture segments that are within rmax
#####################################################################################################################
def check_rmax(site_loc,seg_trace,seg_type,rmax):
    """
    check for list of rupture segments that are within rmax
    
    """
    ##
    if seg_type == 'seg':
        ##
        for i in range(len(seg_trace)-1):
            seg_sub = [seg_trace[i][:2],seg_trace[i+1][:2]]
            ##
            for site_j in site_loc:
                ##
                _,dist = get_closest_pt(site_j,seg_sub)
                ##
                if dist <= rmax:
                    ##
                    return True
        ##
        return False

    elif seg_type == 'pt':
        ##
        for site_j in site_loc:
            ##
            dist = get_haversine_dist(site_j[0],site_j[1],seg_trace[0],seg_trace[1])
            ##
            if dist <= rmax:
                ##
                return True
        ##
        return False
    

#####################################################################################################################
def set_up_get_rup():

    #Initializing the interface to EQHazardCalc
    calc = EQHazardCalc()

    #Create UCERF3 instance
    ucerf3 = calc.getERF("Mean UCERF3 FM3.1")

    #Read the fault system solution
    sol = ucerf3.getSolution()

    #Read the rupture set
    rupSet = sol.getRupSet()

    #Get the number of section
    numSections = rupSet.getNumSections()

    #Map rupture sources to sections
    rupSourceSections = rupSet.getSectionIndicesForAllRups()
    
    return rupSet, rupSourceSections


#####################################################################################################################
def get_fault_xing(processor, start_loc, end_loc, trace_dir, intersect_dir, rup_meta_file, ind_range=['all']):

    ## create line shapes with start-end locations
    lines = [LineString([start_loc[i],end_loc[i]]) for i in range(len(start_loc))]

    #Initializing the interface to EQHazardCalc
    calc = EQHazardCalc()

    #Create UCERF3 instance
    ucerf3 = calc.getERF("Mean UCERF3 FM3.1")

    #Read the fault system solution
    sol = ucerf3.getSolution()

    #Read the rupture set
    rupSet = sol.getRupSet()

    #Get the number of section
    numSections = rupSet.getNumSections()

    #Map rupture sources to sections
    rupSourceSections = rupSet.getSectionIndicesForAllRups()

    ##
    gm_source_info = load_src_rup_M_rate(rup_meta_file, ind_range)
    
    ##
    for src_unique_i in np.unique(gm_source_info['src']):
    
        ## get traces
        coords = get_trace(processor, rupSet, rupSourceSections, src_unique_i, trace_dir)
    
        ## get intersections
        get_intersect(src_unique_i, coords, lines, intersect_dir)


#####################################################################################################################
def get_trace(processor, rupSet, rupSourceSections, src_unique_i, saveDir):
    """
    """
    
    ## define save file
    saveFile = os.path.join(saveDir,'src_'+str(src_unique_i)+'.txt')
    
    try:
        ## get list of segments
        listSeg = np.asarray(rupSourceSections[src_unique_i].toArray())

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
        processor.setCurrentRupture(src_unique_i,0)
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
def get_intersect(src_unique_i, coords, lines, saveDir):
    """
    """
    
    ## define save file
    saveFile = os.path.join(saveDir,'src_'+str(src_unique_i)+'.txt')

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


#####################################################################################################################
##
def init_processor(case_to_run, path_siteloc, path_vs30=None, numThreads=1, rmax_cutoff=100):
    """
    Setup processor to save run time
    
    multiple cases to run:
    1 = get vs30 from OpenSHA
    2 = get list of rupture scenarios passing criteria
    3 = get GM predictions from OpenSHA
    
    """
    
    #locations = ReadLocations()#Read a specific number of locations
    logging.debug(f"read locations")
    locations = ReadLocations(path_siteloc)
    
    #Initializing the interface to EQHazard
    logging.debug(f"initialize processor")
    processor = initEQHazardInterface(numThreads) #Using 4 threads
    
    #Setting locations
    logging.debug(f"set locations")
    processor.setLocations(locations)

    ##
    if case_to_run == 1:
        #Calculate Vs30 and save it, if needed (e.g. new set of nodes)
        processor.obtainVs30()
        vs30 = processor.getVs30()
        saveJArray(vs30, path_vs30)
        
    ##
    else:
        #Load Vs30 from a file if we had it saved to save time
        logging.debug(f"load Vs30 into OpenSHA")
        vs30 = loadJArray(path_vs30)
        processor.setVs30(vs30)
        logging.debug("Vs30: {}....".format(vs30[0:5]))
        
        #Setting max distance (cut-off)
        logging.debug(f"set rmax")
        processor.setMaxDistance(rmax_cutoff)
            
        ##
        return processor
    
    
#####################################################################################################################
def runHazardAnalysis(processor, rup_meta_file, ind_range, saveDir, store_file_type='npz', list_out=None,
                        list_im=['pga','pgv'], list_param=['mean','inter','intra']):
    """
    Main entry method to run the hazard analysis
    """
    #Setting logging level (e.g. DEBUG or INFO)
    # setLogging(logging.DEBUG)
    
    ## get list of outputs
    if list_out is None:
        list_out = []
        for im_i in list_im:
            for param_j in list_param:
                list_out.append(im_i+'_'+param_j)
    
    ## Load sources, ruptures, rates and mags from pre-run
    logging.debug(f"get src and rup indices")
    gm_source_meta = load_src_rup_M_rate(rup_meta_file, processor, ind_range)
    # np.savetxt(r'C:\Users\barry\Desktop\opensra_results_for_chris\src.txt',gm_source_meta['src'])
    # np.savetxt(r'C:\Users\barry\Desktop\opensra_results_for_chris\rup.txt',gm_source_meta['rup'])
    # np.savetxt(r'C:\Users\barry\Desktop\opensra_results_for_chris\M.txt',gm_source_meta['M'])
    # np.savetxt(r'C:\Users\barry\Desktop\opensra_results_for_chris\rate.txt',gm_source_meta['rate'])
    # sys.exit()
    
    ## get IM predictions
    out = np.asarray([get_IM_from_opensha(processor, gm_source_meta['src'][i],
                                        gm_source_meta['rup'][i]) for i in range(len(gm_source_meta['src']))])

    ## get shape of output file
    shape = out.shape
    
    ## convert to sparse matrix and export
    for i in range(len(list_out)):
        saveName = os.path.join(saveDir,list_out[i]+'.'+store_file_type)
        if 'mean' in list_out[i]:
            out_i = np.reshape(out[:,[i][:]],[shape[0],shape[2]])
            ## add 10 to all values to get invalid values (-10) to 0 for conversion to sparse matrix
            out_i = out_i+10
            # zero_loc = out_i<-9.9 # find where values = -10 (cases that exceed rmax)
            # out_i = np.round(np.exp(out_i),decimals=4)
            # out_i[zero_loc] = 0
            if store_file_type == 'npz':
                sparse.save_npz(saveName, sparse.csc_matrix(out_i)) # csc matrix is more efficient for large number of sites, where columns = sites
                # sparse.save_npz(saveName, sparse.csc_matrix(np.reshape(out[:,[i][:]],[shape[0],shape[2]])+10))
            elif store_file_type == 'txt':
                coo_mat = sparse.coo_matrix(out_i) # coo matrix is easier to understand and reconstruct
                np.savetxt(saveName,np.transpose([coo_mat.row,coo_mat.col,coo_mat.data-10]),fmt='%i %i %5.3f')
        else:
            if store_file_type == 'npz':
                sparse.save_npz(saveName, sparse.csc_matrix(np.reshape(out[:,[i][:]],[shape[0],shape[2]]))) # csc matrix is more efficient for large number of sites, where columns = sites
            elif store_file_type == 'txt':
                coo_mat = sparse.coo_matrix(np.reshape(out[:,[i][:]],[shape[0],shape[2]]))  # coo matrix is easier to understand and reconstruct
                np.savetxt(saveName,np.transpose([coo_mat.row,coo_mat.col,coo_mat.data]),fmt='%i %i %5.3f')
    
    out = None
    ##
    # return out
    
    
#####################################################################################################################
def get_IM_from_opensha(proc, src_ind, rup_ind):
    """
    get PGA and PGV means and stdevs from OpenSHA
    """
    logging.debug(f"==============src = {src_ind}, rup = {rup_ind}=========================")
    
    #Setting the current rupture
    logging.debug(f"set current scenario")
    proc.setCurrentRupture(src_ind, rup_ind)
    
    #Calculating distances for all sites from the current rupture surface
    logging.debug(f"calculate distances")
    proc.calculateDistances()

    #Setting the Intensity measured needed
    logging.debug(f"calculate PGA")
    proc.setIM('PGA')
    
    #Calculating intensity measures using GMM
    proc.calculateIMs()
    
    #Reading the calculated means
    # pgaMean = np.round(proc.getMeans(),decimals=3)
    pgaMean = np.round(proc.getMeans(),decimals=6)
    #Reading the calculated Std. Devs.
    pgaInter = np.round(proc.getInterEvStdDevs(),decimals=3)
    pgaIntra = np.round(proc.getIntraEvStdDevs(),decimals=3)
    
    #Setting the Intensity measured needed
    logging.debug(f"calculate PGV")
    proc.setIM('PGV')
    
    #Calculating intensity measures using GMM
    proc.calculateIMs()
    
    #Reading the calculated means
    # pgvMean = np.round(proc.getMeans(),decimals=3)
    pgvMean = np.round(proc.getMeans(),decimals=6)
    #Reading the calculated Std. Devs.
    pgvInter = np.round(proc.getInterEvStdDevs(),decimals=3)
    pgvIntra = np.round(proc.getIntraEvStdDevs(),decimals=3)
    
    ##
    return pgaMean, pgaInter, pgaIntra, pgvMean, pgvInter, pgvIntra
    
    
#####################################################################################################################
def load_src_rup_M_rate(rup_meta_file, proc=None, ind_range=['all'], rate_cutoff=1/10000,
                        rup_group_file=None, rup_per_group=None, file_type='txt'):
    """
    Load sources, ruptures, rates, and mags from pre-run. If **rup_meta_file** doesn't exist, compute and export
    
    Parameters
    ----------
    rup_meta_file : str
        file name of the hdf5 file containing the information
    proc : object
        OpenSHA processor object; used if **rup_meta_file** does not exist to obtain source parameters
    ind_range : str/list, optional
        define indices to extract source parameters for: options are **['all']**, **[index]** or **[low, high]** (brackets are required; replace index, low, and high with integers)
    
    Returns
    -------
    gm_source_info : dictionary
        contains list of source indices, rupture indices, moment magnitudes, and mean annual rates
    
    """
    
    ## see if extension is provided, if not, add it
    if not '.'+file_type in rup_meta_file:
        rup_meta_file = rup_meta_file+'.'+file_type
    # if not '.hdf5' in rup_meta_file:
        # rup_meta_file = rup_meta_file+'.hdf5'
    # if not '.txt' in rup_meta_file:
    #     rup_meta_file = rup_meta_file+'.txt'
    
    ## initialize dictionaries
    gm_source_info = {'src':None,'rup':None,'M':None,'rate':None}
    keys = list(gm_source_info)
    
    ## see if rup_meta_file already exists
    if os.path.exists(rup_meta_file):

        ## load rup_meta_file

        ## txt file format
        if 'txt' in file_type:
            f = np.loadtxt(rup_meta_file,unpack=True)
            if len(ind_range) == 1:
                if ind_range[0] == 'all':
                    gm_source_info['src'] = f[0].astype(np.int32)
                    gm_source_info['rup'] = f[1].astype(np.int32)
                    gm_source_info['M'] = f[2]
                    gm_source_info['rate'] = f[3]
                else:
                    gm_source_info['src'] = f[0,ind_range[0]].astype(np.int32)
                    gm_source_info['rup'] = f[1,ind_range[0]].astype(np.int32)
                    gm_source_info['M'] = f[2,ind_range[0]]
                    gm_source_info['rate'] = f[3,ind_range[0]]
            elif len(ind_range) == 2:
                gm_source_info['src'] = f[0,ind_range[0]:ind_range[1]].astype(np.int32)
                gm_source_info['rup'] = f[1,ind_range[0]:ind_range[1]].astype(np.int32)
                gm_source_info['M'] = f[2,ind_range[0]:ind_range[1]]
                gm_source_info['rate'] = f[3,ind_range[0]:ind_range[1]]

        ## hdf5 file format
        elif 'hdf5' in file_type:
            with h5py.File(rup_meta_file, 'r') as f:
                if len(ind_range) == 1:
                    if ind_range[0] == 'all':
                        for key in keys:
                            gm_source_info[key] = f.get(key)[:]
                    else:
                        for key in keys:
                            gm_source_info[key] = f.get(key)[ind_range[0]]
                elif len(ind_range) == 2:
                    for key in keys:
                        gm_source_info[key] = f.get(key)[ind_range[0]:ind_range[1]]
            f.close()

        ##
        return gm_source_info
        # return src, rup, M, rate
    
    else:
        ## get and store rates and Mw
        nSources = proc.getNumSources()
        src_rup = [[i, j, get_M_rate(proc,i,j)[1], get_M_rate(proc,i,j)[0]]
                    for i in range(nSources) for j in range(proc.getNumRuptures(i))
                    if get_M_rate(proc, i,j)[0] > rate_cutoff]

        ## write to txt format
        if 'txt' in file_type:
            np.savetxt(rup_meta_file, src_rup, fmt='%i %i %6.4f %6.4e')

        ## write to hdf5 format
        elif 'hdf5' in file_type:
            ## restructure datafile
            src_rup = np.transpose(src_rup)
            ##
            with h5py.File(rup_meta_file, 'w') as f:
                dset1 = f.create_dataset('src', data=src_rup[0].astype(np.int32))
                dset2 = f.create_dataset('rup', data=src_rup[1].astype(np.int32))
                dset3 = f.create_dataset('M', data=src_rup[2])
                dset4 = f.create_dataset('rate', data=src_rup[3])
            f.close()

        ## generate rupture group file
        if rup_group_file is not None:
            ## generate and store list of rupture groups
            n_rup_groups = int(np.ceil(len(src_rup[0])/rup_per_group))
            list_rup_group = [str(rup_per_group*i)+'_'+str(rup_per_group*(i+1)-1) for i in range(n_rup_groups)]
            np.savetxt(rup_group_file,list_rup_group,fmt='%s')
            logging.info(f"\tNumber of rupture groups = {n_rup_groups} (each group contains {rup_per_group} ruptures)")
    
    
#####################################################################################################################
def get_M_rate(proc, src, rup):
    """
    This extracts the mean annual rate and moment magnitude for a given scenario (source + rupture index)
    
    Parameters
    ----------
    proc : object
        OpenSHA processor object
    src : int
        source index used by OpenSHA
    rup : int
        rupture index used by OpenSHA
        
    Returns
    -------
    M : float
        moment magnitude of scenario
    rate : float
        mean annual rate for scenario
    
    """
    #Setting the current rupture
    proc.setCurrentRupture(src, rup)
    #Get the current rupture and reading magnitude and rate
    rup = proc.getRupture()
    return [rup.getMeanAnnualRate(1), rup.getMag()]
    
    
#####################################################################################################################
def saveJArray(array, filename):
    """
    Save Java array to a text file one value per line
    """
    values = []
    for value in array:
        values.append(str(value) + '\n')
    
    with open(filename, 'w+') as arrayFile:
        arrayFile.writelines(values)
    
    
#####################################################################################################################
def loadJArray(filename):
    """
    Load Java array from a text file
    """
    with open(filename, 'r') as arrayFile:
        lines = arrayFile.readlines()
        array = JArray(JDouble)(len(lines))
        
        for i in range(len(lines)):
            array[i] = float(lines[i])
        
        return array
    
    
#####################################################################################################################
def ReadLocations(path_siteloc,count=None):
    """
    This method reads the locations from the csv file
    """
    lines = []
    
    # Reading all lines in the file
    with open(path_siteloc, 'r') as locationsFile:
    # with open('seg_node_full_1k.txt', 'r') as locationsFile:
        lines = locationsFile.readlines()
    if count:
        locations = JArray(Location)(count)
    else:
        locations = JArray(Location)(len(lines))
    
    i = 0
    for line in lines:
        if count and i >= count:
            break
        tokens = line.split(',')
        latitude = float(tokens[1])
        longitude = float(tokens[0])
        locations[i] = Location(latitude, longitude)
        i = i + 1
    
    return locations
    
    
#####################################################################################################################
def initEQHazardInterface(numThreads):
    """
    This method creates and returns an instance of the RegionalProcessor class
    """
    processor = RegionalProcessor(numThreads)
    return processor