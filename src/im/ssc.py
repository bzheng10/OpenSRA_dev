# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Base classes used in OpenSRA
#
# Created: April 1, 2022
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python base modules
import os
import logging
import json
import sys
import xml.etree.ElementTree as ET
# import inspect

# data manipulation modules
import numpy as np
# from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas import DataFrame, read_hdf
from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from shapely.geometry import Polygon
# from scipy.interpolate import interp1d


# geospatial processing modules


# OpenQuake for distance calculations
from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.mesh import Mesh

# efficient processing modules
# import numba as nb
from numba import njit, jit

# OpenSRA modules
from src.util import from_dict_to_array, wgs84_to_utm, utm_to_wgs84
# from src.im import gmc


# -----------------------------------------------------------
class SeismicSource(object):
    """
    Class for running seismic hazard
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    
    # class definitions
    SUPPORTE_SOURCE_TYPE = ['UCERF3', 'ShakeMap','UserDefinedRupture', 'UserDefinedGM'] # supported sourcess
    
    
    # instantiation
    def __init__(self):
        """Create an instance of the class"""

        # get inputs
        # self.source_type = source_type
        # check source type against supported source types
        # self._check_source_support()
        
        # additional setup
        # OpenQuake surfaces
        # self.oq_objects = {
        #     'points': None,
        #     'mesh': None,
        #     'surfaces': None
        # }
        
        # initialize empty params
        self._n_event = 0
    
    
    # @property
    # def supported_source_types(self):
    #     """supported sources"""
    #     return self.SUPPORTE_SOURCE_TYPE
    
    
    # def _check_source_support(self):
    #     """check for support"""
    #     self._source_type_support = False
    #     if self.source_type.lower() in self.supported_source_types:
    #         self._source_type_support = True
    #     else:
    #         raise NotImplementedError(
    #             f'"{self.source_type}" is not a supported type for seismic source characterization;' + 
    #             f'supported source types include: {*self.supported_source_types,}'
    #         )
            
    
# -----------------------------------------------------------
class UCERF_CompleteInventory(SeismicSource):
    """
    UCERF class
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    
    # class definitions
    # SUPPORTE_UCERF_MODEL = ['Mean UCERF3', 'Mean UCERF3 FM3.1','Mean UCERF3 FM3.2'] # supported sourcess
    SUPPORTE_UCERF_MODEL = ['Mean UCERF3 FM3.1'] # supported sourcess

    
    # instantiation
    def __init__(self, ucerf_model='Mean UCERF3 FM3.1'):
        """Create an instance of the class"""
        
        # invoke parent function
        super().__init__()
        
        # get inputs
        self.ucerf_model = ucerf_model
        # check against supported ucerf models
        self._check_ucerf_support()
        
        # initialize empty params
        self.oq_surfaces = None # OpenQuake surfaces
        self._max_dist = None
        self._mag_min = None
        self._mag_max = None
        self._rate_min = None
        self._rate_max = None
        self._mesh_spacing = None
        
        # preprocess
        self._load_rupture_and_section() # loads ruptures and sections for UCERF models
        
        
    @property
    def supported_ucerf_model(self):
        """supported models"""
        return self.SUPPORTE_UCERF_MODEL
    
    
    def _check_ucerf_support(self):
        """check for support"""
        self._ucerf_model_support = False
        if self.ucerf_model in self.supported_ucerf_model:
            self._ucerf_model_support = True
        else:
            raise NotImplementedError(
                f'"{self.ucerf_model}" is not a supported type for seismic source characterization;' + 
                f'supported source types include: {*self.supported_ucerf_model,}'
            )
    
    
    def _load_rupture_and_section(self):
        """load ruptures and sections for ucerf model from provided library"""
        # make file paths
        ###############################################
        curr_path = os.path.realpath(__file__)
        self._ucerf_base_dir = os.path.join("..","..","OpenSRA","lib","OpenSHA","ERF")
        ###############################################
        self.ucerf_model_dir = os.path.join(self._ucerf_base_dir,self.ucerf_model)
        # load ucerf files
        self.df_rupture = read_hdf(os.path.join(self.ucerf_model_dir,'Ruptures.h5'))
        self.df_section = read_hdf(os.path.join(self.ucerf_model_dir,"Sections.h5"))
        self._n_event = self.df_rupture.shape[0]
        logging.info(f"\t- Loaded rupture and section information")
    
    
    def process_rupture(
        self,
        lon,
        lat,
        max_dist=200, # km,
        mag_min=None,
        mag_max=None,
        rate_min=None,
        rate_max=None,
        mesh_spacing=1, # km
    ):
        """a series of actions to process and filter rupture scenarios"""
        # get inputs)
        self._max_dist = max_dist
        self._mag_min = mag_min
        self._mag_max = mag_max
        self._rate_min = rate_min
        self._rate_max = rate_max
        self._mesh_spacing = mesh_spacing
        self._oq_obj = {
            'points': None,
            'mesh': None,
            'surfaces': None,
        }
        # some preprocessing
        logging.info(f"Some preprocessing for fault ruptures...")
        self._n_site = len(lon)
        self._make_oq_loc_obj(lon, lat)
        self._generate_fault_surfaces() # make fault surfaces for sections
        # run actions
        logging.info(f"Screening ruptures...")
        self._find_sections_within_max_dist() # get sections that are within max distance
        self._get_distances_to_sections() # calculate distance metrics from sites to sections within max distance
        self._filter_ucerf_scenarios() # filter by sections within max dist, then ranges of magnitude and annual rates
    
    
    def _make_oq_loc_obj(self, lon, lat):
        """make OpenQuake points and mesh from lat lon"""
        if self._oq_obj['points'] is not None:
            self._oq_obj['points'] = None
            self._oq_obj['mesh'] = None
        self._oq_obj['points'] = [Point(lon[i],lat[i]) for i in range(len(lon))]
        self._oq_obj['mesh'] = Mesh.from_points_list(self._oq_obj['points'])
    
    
    def _generate_fault_surfaces(self):
        """generate OpenQuake simple fault surfaces"""
        # create surfaces for all sections
        if self._oq_obj['surfaces'] is not None:
            self._oq_obj['surfaces']  = None
        self._oq_obj['surfaces'] = [
            SimpleFaultSurface.from_fault_data(
                fault_trace=Line(
                    [Point(
                        self.df_section.FaultTrace[ind][i][0],
                        self.df_section.FaultTrace[ind][i][1],
                        self.df_section.FaultTrace[ind][i][2]
                    ) for i in range(len(self.df_section.FaultTrace[ind]))]),
                upper_seismogenic_depth=self.df_section.UpperDepth[ind],
                lower_seismogenic_depth=self.df_section.LowerDepth[ind],
                dip=self.df_section.Dip[ind],
                mesh_spacing=self._mesh_spacing # km
            ) for ind in range(self.df_section.shape[0])
        ]
        logging.info(f"\t- Generated simple fault surfaces for sections")
        
        
    def _find_sections_within_max_dist(self):
        """find list of sections within max distance and sites within max distance of said section"""
        self.sites_for_section_in_maxdist = {
            i: np.where(self._oq_obj['surfaces'][i].get_min_distance(self._oq_obj['mesh']) <= self._max_dist)[0] 
            for i in range(self.df_section.shape[0])
            if min(self._oq_obj['surfaces'][i].get_min_distance(self._oq_obj['mesh'])) <= self._max_dist
        }
        # get dictionary keys
        self.section_in_maxdist = list(self.sites_for_section_in_maxdist.keys())
        logging.info(f"\t- Retrieved sections within max distance of {self._max_dist} km from sites")
    
    
    @staticmethod
    def _get_dist_from_sites_to_section(section_id, list_of_sites, oq_points, oq_surfaces, decimals=1):
        """calculates distance metrics using OpenQuake"""
        # create reduced mesh with current set of sites
        loc_mesh = Mesh.from_points_list([oq_points[site] for site in list_of_sites])
        # calculate distances
        r_rup = np.round(oq_surfaces[section_id].get_min_distance(loc_mesh),decimals=decimals)
        r_jb = np.round(oq_surfaces[section_id].get_joyner_boore_distance(loc_mesh),decimals=decimals)
        r_x = np.round(oq_surfaces[section_id].get_rx_distance(loc_mesh),decimals=decimals)
        r_y0 = np.round(oq_surfaces[section_id].get_ry0_distance(loc_mesh),decimals=decimals)
        # return
        return r_rup, r_jb, r_x, r_y0
    
    
    def _get_distances_to_sections(self):
        """calculates distances from sites to sections, to be used for determining controlling fault parameters for GMPE"""
        # initialize table to store distances for sites within dist_max
        r_rup_for_section = {}
        r_jb_for_section = {}
        r_x_for_section = {}
        r_y0_for_section = {}
        # loop through each section within max distance and get all distance metrics for NGAW2 GMPEs
        for section in self.section_in_maxdist:
            # # calculate distances
            r_rup, r_jb, r_x, r_y0 = self._get_dist_from_sites_to_section(
                section,
                self.sites_for_section_in_maxdist[section],
                self._oq_obj['points'],
                self._oq_obj['surfaces']
            )
            r_rup_for_section[section] = r_rup
            r_jb_for_section[section] = r_jb
            r_x_for_section[section] = r_x
            r_y0_for_section[section]  = r_y0
        # convert from dictionary to array and store
        shape = (self.df_section.shape[0],self._n_site)
        self.r_rup_table = from_dict_to_array(r_rup_for_section,self.sites_for_section_in_maxdist,shape,fill_zeros=999)
        self.r_jb_table = from_dict_to_array(r_jb_for_section,self.sites_for_section_in_maxdist,shape,fill_zeros=999)
        self.r_x_table = from_dict_to_array(r_x_for_section,self.sites_for_section_in_maxdist,shape,fill_zeros=999)
        self.r_y0_table = from_dict_to_array(r_y0_for_section,self.sites_for_section_in_maxdist,shape,fill_zeros=999)
        logging.info(f"\t- Calculated r_rup, r_jb, r_x, and r_y0 between sites and sections")
        
        
    def _filter_ucerf_scenarios(self):
        """list of scenarios in max distance, magnitude, and mean annual rate"""
        logging.info(f"\t- Filtered rupture scenarios by:")
        # filter by distance
        scenario_in_maxdist = np.unique(np.hstack(self.df_section.RupturesForSection[self.section_in_maxdist]))
        self.df_rupture = self.df_rupture.iloc[scenario_in_maxdist,:].copy().reset_index(drop=True)
        logging.info(f"\t\t- max distance: {self._max_dist} km")
        # filter by magnitudes
        if self._mag_min is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.Magnitude>=self._mag_min].reset_index(drop=True)
            logging.info(f"\t\t- min magnitude: {self._mag_min}")
        if self._mag_max is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.Magnitude<=self._mag_max].reset_index(drop=True)
            logging.info(f"\t\t- min magnitude: {self._mag_max}")
        # filter for mean annual rate
        if self._rate_min is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.AnnualRate>=self._rate_min].reset_index(drop=True)
            logging.info(f"\t\t- min mean annual rate: {self._rate_min}")
        if self._rate_max is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.AnnualRate<=self._rate_max].reset_index(drop=True)  
            logging.info(f"\t\t- min mean annual rate: {self._rate_min}")
        self._n_event = self.df_rupture.shape[0]
        logging.info(f"\t- Number of events remaining after filter: {self._n_event}")


    @staticmethod
    @njit
    def get_controlling_section_for_site(
        sections_for_scenario,
        r_rup_table,
        sites_for_rupture_within_maxdist
    ):
        """return section in rupture that is closest to sites within max distance of rupture"""
        return [
            sections_for_scenario[
                r_rup_table[sections_for_scenario,site]==min(r_rup_table[sections_for_scenario,site])
            ][0] for site in sites_for_rupture_within_maxdist
        ]
        

    def _get_gmpe_input_for_event_i(self, i, im_list, site_data):
        """compile inputs for event i for GMPE calculations"""
        # sections for current scenario
        sections_for_scenario = self.df_rupture.SectionsForRupture.iloc[i]
        # find list of sites within max distance
        sites_for_rupture_within_maxdist = np.unique(np.hstack([
            self.sites_for_section_in_maxdist[section] for section in self.sites_for_section_in_maxdist.keys()
            if section in sections_for_scenario]))
        # set up table of inputs
        inputs_for_gmpe = DataFrame(
            None,
            index=list(range(len(sites_for_rupture_within_maxdist))),
            columns=[
                'site_id',
                'source_id',
                'mag',
                'rate',
                'section_id_control',
                'dip',
                'rake',
                'z_tor',
                'z_bor',
                'r_rup',
                'r_jb',
                'r_x',
                'r_y0'
            ]
        )
        # get rupture characteristics
        inputs_for_gmpe.site_id = site_data['site_id'][sites_for_rupture_within_maxdist]
        inputs_for_gmpe.source_id = self.df_rupture.SourceId[i]
        inputs_for_gmpe.mag = self.df_rupture.Magnitude[i]
        inputs_for_gmpe.rate = self.df_rupture.AnnualRate[i]
        inputs_for_gmpe.rake = self.df_rupture.Rake[i]
        # get controlling section id
        section_control = self.get_controlling_section_for_site(
            sections_for_scenario,
            self.r_rup_table,
            sites_for_rupture_within_maxdist
        )
        # get rest of gmpe inputs based on controlling section id    
        inputs_for_gmpe.section_id_control = section_control
        inputs_for_gmpe.r_rup = self.r_rup_table[section_control,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.r_jb = self.r_jb_table[section_control,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.r_x = self.r_x_table[section_control,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.r_y0 = self.r_y0_table[section_control,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.dip = self.df_section.Dip[section_control].values
        inputs_for_gmpe.z_tor = self.df_section.UpperDepth[section_control].values
        inputs_for_gmpe.z_bor = self.df_section.LowerDepth[section_control].values
        # add in other site params if available
        for site_param in ['vs30','z1p0','z2p5','vs30_source']:
            if site_data[site_param] is not None:
                inputs_for_gmpe[site_param] = site_data[site_param][sites_for_rupture_within_maxdist]
        # convert vs30_source to numerics: 0=inferred/estimated, 1=measured
        vs30_source_num = np.ones(inputs_for_gmpe.vs30_source.shape)
        vs30_source_num[inputs_for_gmpe.vs30_source=='Inferred'] == 0
        inputs_for_gmpe['vs30_source'] = vs30_source_num
        # convert inputs into dictionary and return
        kwargs = {}
        for col in inputs_for_gmpe.columns:
            kwargs[col] = inputs_for_gmpe[col].values
        kwargs['period_out'] = im_list
        return kwargs
    

# -----------------------------------------------------------
class ShakeMap(SeismicSource):
    """
    ShakeMap class
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    
    # instantiation
    def __init__(self, sm_dir, event_names=None, im_list=['pga','pgv']):
        """Create an instance of the class"""
        
        # invoke parent function
        super().__init__()
        
        # get inputs
        self._sm_dir = sm_dir
        if event_names is None:
            self.event_names = [
                case for case in os.listdir(sm_dir) 
                if os.path.isdir(os.path.join(sm_dir,case)) and 'rupture.json' in os.listdir(os.path.join(sm_dir,case))
            ]
        else:
            self.event_names = event_names
        self.im_list = im_list
        self._n_event = len(self.event_names)
        # self.event_id = np.arange(self._n_event)
        # self.rate = np.ones(self._n_event) # set annual rates to 1 for ShakeMap events
        # self.event_metadata = np.zeros((self._n_event ,3)) # [event_id, magnitude, annual_rate]
        
        
        # initialize empty params
        # self.mag = None
        self.sm_grid_data = {}
        self.sm_summary = {}
        self.oq_surfaces = None # OpenQuake surfaces
        self._max_dist = None
        self._mag_min = None
        self._mag_max = None
        self._rate_min = None
        self._rate_max = None
        self._mesh_spacing = None
        self._gm_meta = {}
        self._sigma_meta = {}
        self._index_for_ims = {}
        
        # preprocess
        self._check_content_in_sm_dir()
        self._parse_sm_files() # loads ruptures and sections for UCERF models
    
    
    def _check_content_in_sm_dir(self):
        """check to make sure each ShakeMap folder contains the required files"""
        for event_i in self.event_names:
            # see what files are given
            files = os.listdir(os.path.join(self._sm_dir,event_i))
            if 'grid.xml' in files and 'grid.xml' in files and 'grid.xml' in files:
                pass
            else:
                raise ValueError(f'Requires "grid.xml", "uncertainty.xml", and "rupture.json"; one of them is missing')
    
    
    def _parse_sm_files(self):
        """load data from ShakeMap folders"""
        # lists for tracking rupture information
        dip = []
        rake = []
        mag = []
        dzs = []
        planes_for_events = []
        traces = []
        ref = []
        z_tor = []
        z_bor = []
        dip_dir = []
        # loop through events
        for i, event_i in enumerate(self.event_names):
            curr_sm_dir = os.path.join(self._sm_dir,event_i)
            self.sm_summary['Event_'+str(i)] = {
                'Label': event_i,
                'Metadata': {},
                'Units': {}
            }
            
            # empty dictionary
            gm_dict = None
            sigma_dict = None
            
            # -----------------------------------------------------------
            # first parse "grid.xml"
            root = ET.parse(os.path.join(curr_sm_dir,'grid.xml')).getroot()
            # list of unique tags in ShakeMap "grid.xml"
            grid_xml_unique_tag = ['root','event','grid_specification','event_specific_uncertainty','grid_field','grid_data']

            # create dictionary with ShakeMap tags
            gm_dict = dict.fromkeys(grid_xml_unique_tag)
            # update dictionary with root attributes
            gm_dict['root'] = root.attrib
            for key in list(gm_dict['root']):
                if '}' in key:
                    new_key = key[key.find('}')+1:]
                    gm_dict['root'][new_key] = gm_dict['root'][key]
                    gm_dict['root'].pop(key)

            # loop through all tags in the xml file
            for child in root:
                current_tag = child.tag[child.tag.find('}')+1:] # get current tag
                # if tag does not exist as a key, initialize it as a dictionary
                if not isinstance(gm_dict[current_tag],dict):
                    gm_dict[current_tag] = {}
                current_attribute = child.attrib # get list of attributes for current child

                # skip 'event_specific_uncertainty'; uncertainty information will be imported from "uncertainty.xml"
                if current_tag == 'event_specific_uncertainty':
                    pass
                # read 'grid_field' information
                elif current_tag == 'grid_field':
                    if not child.get('index') in gm_dict[current_tag].keys():
                        gm_dict[current_tag].update({child.get('index'):{}})
                        for sub_attrib in current_attribute:
                            if not sub_attrib == 'index':
                                try:
                                    gm_dict[current_tag][child.get('index')].update({sub_attrib:float(child.get(sub_attrib))})
                                except:
                                    gm_dict[current_tag][child.get('index')].update({sub_attrib:child.get(sub_attrib)})
                # parse 'grid_data' into matrix
                elif current_tag == 'grid_data':
                    gm_dict[current_tag] = np.asarray([
                        list(map(float, line.split())) 
                        for line in child.text.splitlines() if len(line) > 0
                    ])
                # read other tags
                else:
                    for sub_attrib in current_attribute:
                        try:
                            gm_dict[current_tag].update({sub_attrib:float(child.get(sub_attrib))})
                        except:
                            gm_dict[current_tag].update({sub_attrib:child.get(sub_attrib)})

            # from 'grid_field' get positions for coordinates and target IMs
            index_for_coords = [0]*2 # order = [lon, lat]
            index_for_ims = [0]*len(self.im_list) # order = [im_list1, im_list2, ...]
            count_im_list = 0
            for field in gm_dict['grid_field']:
                # check for 'lon'
                if gm_dict['grid_field'][field]['name'].lower() in 'longitude':
                    index_for_coords[0] = int(field)-1
                # check for 'lat'
                if gm_dict['grid_field'][field]['name'].lower() in 'latitude':
                    index_for_coords[1] = int(field)-1
                # check for IMs
                for im in self.im_list:
                    if im.lower() == gm_dict['grid_field'][field]['name'].lower():
                        index_for_ims[count_im_list] = int(field)-1
                        count_im_list += 1
                        break
            index_for_ims_in_mean_file = index_for_ims
            self._index_for_ims[event_i] = {'mean': index_for_ims_in_mean_file}
            # extract coordinates and target IMs from grid data and convert into DataFrame
            coord_df = DataFrame(gm_dict['grid_data'][:,index_for_coords], columns=['lon','lat'])
            mean_df = DataFrame(gm_dict['grid_data'][:,index_for_ims_in_mean_file], columns=[im+'_mean' for im in self.im_list])
            
            # drop gm_dict['grid_data'] from dictionary for memory
            gm_dict.pop('grid_data',None)
            # store gm_dict into instance
            self._gm_meta[event_i] = gm_dict.copy()
            
            # -----------------------------------------------------------
            # parse "uncertainty.xml"
            root = ET.parse(os.path.join(curr_sm_dir,'uncertainty.xml')).getroot()
            # list of unique tags in ShakeMap "grid.xml"
            grid_xml_unique_tag = ['root','event','grid_specification','event_specific_uncertainty','grid_field','grid_data']

            # create dictionary with ShakeMap tags
            sigma_dict = dict.fromkeys(grid_xml_unique_tag)

            # loop through all tags in the xml file
            for child in root:
                current_tag = child.tag[child.tag.find('}')+1:] # get current tag
                # if tag does not exist as a key, initialize it as a dictionary
                if not isinstance(sigma_dict[current_tag],dict):
                    sigma_dict[current_tag] = {}
                current_attribute = child.attrib # get list of attributes for current child

                # read 'grid_field' information
                if current_tag == 'grid_field':
                    if not child.get('index') in sigma_dict[current_tag].keys():
                        sigma_dict[current_tag].update({child.get('index'):{}})
                        for sub_attrib in current_attribute:
                            if not sub_attrib == 'index':
                                try:
                                    sigma_dict[current_tag][child.get('index')].update({sub_attrib:float(child.get(sub_attrib))})
                                except:
                                    sigma_dict[current_tag][child.get('index')].update({sub_attrib:child.get(sub_attrib)})
                # parse 'grid_data' into matrix
                elif current_tag == 'grid_data':
                    sigma_dict[current_tag] = np.asarray([
                        list(map(float, line.split())) 
                        for line in child.text.splitlines() if len(line) > 0
                    ])
                # skip tags except 'grid_field' and 'grid_data'
                else:
                    pass

            # from 'grid_field' get positions for coordinates and target IMs
            index_for_coords = [0]*2 # order = [lon, lat]
            index_for_ims = [0]*len(self.im_list) # order = [im_list1, im_list2, ...]
            count_im_list = 0
            for field in sigma_dict['grid_field']:
                # check for 'lon'
                if sigma_dict['grid_field'][field]['name'].lower() in 'longitude':
                    index_for_coords[0] = int(field)-1
                # check for 'lat'
                if sigma_dict['grid_field'][field]['name'].lower() in 'latitude':
                    index_for_coords[1] = int(field)-1
                # check for IMs
                for im_i in self.im_list:
                    if f'std{im_i.lower()}' == sigma_dict['grid_field'][field]['name'].lower():
                        index_for_ims[count_im_list] = int(field)-1
                        count_im_list += 1
                        break
            index_for_ims_in_sigma_file = index_for_ims
            self._index_for_ims[event_i]['sigma'] = index_for_ims_in_sigma_file
            # extract coordinates and target IMs from grid data
            sigma_df = DataFrame(sigma_dict['grid_data'][:,index_for_ims_in_sigma_file], columns=[im+'_sigma' for im in self.im_list])

            # drop gm_dict['grid_data'] from dictionary for memory
            sigma_dict.pop('grid_data',None)
            # store gm_dict into instance
            self._sigma_meta[event_i] = sigma_dict.copy()
            
            # combine dfs into one
            self.sm_grid_data[event_i] = pd.concat([coord_df, mean_df, sigma_df], axis=1)

            # -----------------------------------------------------------
            # parse rupture file
            with open(os.path.join(curr_sm_dir,'rupture.json'),'r') as f:
                sm_rupture = json.load(f)
            mag.append(sm_rupture['metadata']['mag'])
            rake.append(sm_rupture['metadata']['rake'])
            ref.append(event_i)
            planes_for_events.append(np.asarray(sm_rupture['features'][0]['geometry']['coordinates'][0]))
            for plane in planes_for_events[-1]:
                plane_wgs84 = np.zeros(plane.shape)
                plane_wgs84[:,2] = plane[:,2]
                plane_wgs84[:,0],plane_wgs84[:,1],_,_ = wgs84_to_utm(lon=plane[:,0],lat=plane[:,1],force_zone_num=10)
                plane_wgs84[:,0] = plane_wgs84[:,0]/1000
                plane_wgs84[:,1] = plane_wgs84[:,1]/1000
                top_edge_i = []
                bot_edge_i = []
                trace_i = []
                for j in range(len(plane_wgs84)-1):
                    if plane_wgs84[j,2] != plane_wgs84[j+1,2]:
                        dhoriz = ((plane_wgs84[j,0]-plane_wgs84[j+1,0])**2 + (plane_wgs84[j,1]-plane_wgs84[j+1,1])**2) ** 0.5
                        dy = plane_wgs84[j+1,1]-plane_wgs84[j,1]
                        dx = plane_wgs84[j+1,0]-plane_wgs84[j,0]
                        dip_dir.append(np.round(np.arctan(dy/dx)*180/np.pi,1))
                        if dx < 0:
                            dip_dir[-1] = -dip_dir[-1]
                        dz = abs(plane_wgs84[j,2]-plane_wgs84[j+1,2])
                        dip.append(np.round(np.arctan(dz/dhoriz)*180/np.pi,1))
                        dzs.append(dz)
                        z_tor.append(min(plane_wgs84[j,2],plane_wgs84[j+1,2]))
                        z_bor.append(max(plane_wgs84[j,2],plane_wgs84[j+1,2]))
                        n_top_edge = j+1
                        break
                if z_tor[-1] > 0:
                    top_edge_j = plane_wgs84[:n_top_edge]
                    bot_edge_j = plane_wgs84[len(plane_wgs84)-2:n_top_edge-1:-1]
                    trace_j = []
                    surf_diag_to_trace = z_tor[-1]/np.tan(dip[-1]*np.pi/180)
                    if dip_dir[-1] < 0:
                        x_trace = top_edge_j[:,0] + np.cos(dip_dir[-1]*np.pi/180)*surf_diag_to_trace
                    else:
                        x_trace = top_edge_j[:,0] - np.cos(dip_dir[-1]*np.pi/180)*surf_diag_to_trace
                    y_trace = top_edge_j[:,1] - np.sin(dip_dir[-1]*np.pi/180)*surf_diag_to_trace
                    x_trace_wgs, y_trace_wgs = utm_to_wgs84(x_trace*1000,y_trace*1000,zone=10)
                    trace_j = np.asarray(np.transpose([
                        np.round(x_trace_wgs,6),
                        np.round(y_trace_wgs,6),
                        np.zeros(len(x_trace))
                    ]))
                else:
                    trace_j = plane[:n_top_edge,:]
                traces.append(np.flipud(trace_j).tolist())

            # store ShakeMap file information
            for j, im in enumerate(self.im_list):
                self.sm_summary['Event_'+str(i)]['Units'][im] = gm_dict['grid_field'][str(index_for_ims_in_mean_file[j]+1)]['units']
            self.sm_summary['Event_'+str(i)]['Metadata'] = gm_dict['event']            
            
        # storing event rupture details
        self.df_rupture = pd.DataFrame.from_dict({
            'EventID': list(range(len(traces))),
            'EventName': ref,
            'Magnitude': mag,
            'AnnualRate': np.ones(len(traces)),
            'Dip': dip,
            'Rake': rake,
            'DipDir': dip_dir,
            'UpperDepth': z_tor,
            'LowerDepth': z_bor,
            'FaultTrace': traces,
        })
    
    
    def process_rupture(
        self,
        mag_min=None,
        mag_max=None,
    ):
        """a series of actions to process and filter rupture scenarios"""
        # get inputs
        self._mag_min = mag_min
        self._mag_max = mag_max
        # some preprocessing
        # run actions
        logging.info(f"Screening ruptures...")
        logging.info(f"\t- Filtered rupture scenarios by:")
        # filter by magnitudes
        if self._mag_min is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.Magnitude>=self._mag_min].reset_index(drop=True)
            logging.info(f"\t\t- min magnitude: {self._mag_min}")
        if self._mag_max is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.Magnitude<=self._mag_max].reset_index(drop=True)
            logging.info(f"\t\t- min magnitude: {self._mag_max}")
        self._n_event = self.df_rupture.shape[0]
        logging.info(f"\t- Number of events remaining after filter: {self._n_event}")
    
    
    def _sample_gm_from_map_i(self, i, lon, lat, out_of_bound_value=-10, aleatory_default=0.01):
        """sample ground motions from maps for event i using nearest neighbor"""
        # num sites
        n_site = len(lon)
        # other items
        im_list = self.im_list
        # current event name
        event_i = self.event_names[i]
        gm_meta_i = self._gm_meta[event_i].copy()
        grid_data_i = self.sm_grid_data[event_i].copy()
        index_for_ims_mean = self._index_for_ims[event_i]['mean']
        index_for_ims_sigma = self._index_for_ims[event_i]['sigma']
        # use sindex in geopandas for nearest neighbor
        gdf_gm = GeoDataFrame(geometry=points_from_xy(
            grid_data_i.lon.values,grid_data_i.lat.values))
        # make boundary to first filter points within ShakeMap extent
        bound = GeoDataFrame(geometry=[Polygon([
            [gm_meta_i['grid_specification']['lon_min'], gm_meta_i['grid_specification']['lat_min']],
            [gm_meta_i['grid_specification']['lon_min'], gm_meta_i['grid_specification']['lat_max']],
            [gm_meta_i['grid_specification']['lon_max'], gm_meta_i['grid_specification']['lat_max']],
            [gm_meta_i['grid_specification']['lon_max'], gm_meta_i['grid_specification']['lat_min']],
        ])])
        # turn sites into geoseries
        gs_sites = GeoSeries(points_from_xy(lon, lat))
        # get points within boundary
        sites_in_bound = gs_sites.sindex.query_bulk(bound.geometry, predicate='intersects')[1]
        sites_not_in_bound = list(set(list(range(n_site))).difference(set(sites_in_bound)))
        # for sites in bound, get nearest neighbor
        # first make new geoseries
        gs_sites_in_bound = GeoSeries(points_from_xy(lon[sites_in_bound],lat[sites_in_bound]))
        # get nearest neighbor for sites
        nearest_sm_node = gdf_gm.sindex.nearest(gs_sites_in_bound)[1]
        # initialize array for interpolated means and uncertainty
        site_gm = np.zeros((n_site,len(im_list)))
        site_sigma = np.zeros((n_site,len(im_list)))
        site_aleatory = np.zeros((n_site,len(im_list)))
        site_epistemic = np.zeros((n_site,len(im_list)))
        
        # interpolate for median IMs
        for i in range(len(im_list)):
            # get im mean values
            site_gm[sites_in_bound,i] = grid_data_i[f'{im_list[i]}_mean'].values[nearest_sm_node]
            
            # multiply by 100 if IM values are in %
            for pct_str in ['%','pct']:
                if pct_str in gm_meta_i['grid_field'][str(index_for_ims_mean[i]+1)]['units']:
                    site_gm[sites_in_bound,i] = site_gm[sites_in_bound,i]/100
                    gm_meta_i['grid_field'][str(index_for_ims_mean[i]+1)]['units'] = \
                        gm_meta_i['grid_field'][str(index_for_ims_mean[i]+1)]['units'].replace(pct_str,'')
                    break

            # get im sigma values
            site_sigma[sites_in_bound,i] = grid_data_i[f'{im_list[i]}_sigma'].values[nearest_sm_node]
            # site_sigma[sites_not_in_bound,i] = out_of_bound_value
            site_aleatory[sites_in_bound,i] = aleatory_default
            site_epistemic[sites_in_bound,i] = np.sqrt(
                site_sigma[sites_in_bound,i]**2 - \
                site_aleatory[sites_in_bound,i]**2
            )
            
        # update metadata
        self._gm_meta[event_i] = gm_meta_i.copy()
        # convert to log for mean
        site_gm[sites_in_bound,:] = np.log(site_gm[sites_in_bound,:])
        # set out_of_bound values
        site_gm[sites_not_in_bound,:] = out_of_bound_value  
        # site_aleatory[sites_not_in_bound,:] = out_of_bound_value
        # site_epistemic[sites_not_in_bound,:] = out_of_bound_value
        # return
        return site_gm, site_aleatory, site_epistemic


# -----------------------------------------------------------
class UserDefinedRupture(SeismicSource):
    """
    User defined class
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    
    # class definitions
    
    
    # instantiation
    def __init__(self, fpath=None):
        """Create an instance of the class"""
        
        # invoke parent function
        super().__init__()
        
        # get inputs
        self.fpath = fpath
        
        # load file
        self._load_rupture_file() # loads ruptures from user specified file
        self._check_consistency() # check table header to make sure it contains the necessary columns
        
        # initialize empty params
        self.oq_surfaces = None # OpenQuake surfaces
        self._max_dist = None
        self._mag_min = None
        self._mag_max = None
        self._rate_min = None
        self._rate_max = None
        self._mesh_spacing = None
    
    
    def _load_rupture_file(self):
        """load rupture file provided by the user. Only allowed CSV format at this stage"""
        if self.fpath is None:
            return ValueError("Must provided fpath (path to rupture CSV file)")
        else:
            # read
            self.df_rupture = pd.read_csv(self.fpath)
            # convert fault trace from string to list of coordinates
            self.df_rupture.FaultTrace = self.df_rupture.FaultTrace.apply(lambda x: json.loads(x))
            # get number of events
            self._n_event = self.df_rupture.shape[0]
            logging.info("Loaded user-defined rupture file")
            
    
    def _check_consistency(self):
        """check table column headers for consistency"""
        # check for following column names
        req_cols = [
            'EventID',
            'EventName',
            'Magnitude',
            'AnnualRate',
            'Dip',
            'Rake',
            'DipDir',
            'UpperDepth',
            'LowerDepth',
            'FaultTrace'
        ]
        missing_cols = []
        for col in req_cols:
            if not col in self.df_rupture:
                missing_cols.append(col)
        if len(missing_cols)>0:
            logging.info(f'The table is missing the following columns/headers:')
            logging.info(f'\t {", ".join(missing_cols)}')
            
    
    def process_rupture(
        self,
        lon,
        lat,
        max_dist=200, # km,
        mag_min=None,
        mag_max=None,
        rate_min=None,
        rate_max=None,
        mesh_spacing=1, # km
    ):
        """a series of actions to process and filter rupture scenarios"""
        # get inputs)
        self._max_dist = max_dist
        self._mag_min = mag_min
        self._mag_max = mag_max
        self._rate_min = rate_min
        self._rate_max = rate_max
        self._mesh_spacing = mesh_spacing
        self._oq_obj = {
            'points': None,
            'mesh': None,
            'surfaces': None,
        }
        # some preprocessing
        logging.info(f"Some preprocessing for fault ruptures...")
        self._n_site = len(lon)
        self._make_oq_loc_obj(lon, lat)
        self._generate_fault_surfaces() # make fault surfaces for ruptures
        # run actions
        logging.info(f"Screening ruptures...")
        self._find_ruptures_within_max_dist() # get ruptures that are within max distance
        self._get_distances_to_ruptures() # calculate distance metrics from sites to ruptures within max distance
        self._filter_rupture() # filter ruptures within max dist, then ranges of magnitude and annual rates
    
    
    def _make_oq_loc_obj(self, lon, lat):
        """make OpenQuake points and mesh from lat lon"""
        if self._oq_obj['points'] is not None:
            self._oq_obj['points'] = None
            self._oq_obj['mesh'] = None
        self._oq_obj['points'] = [Point(lon[i],lat[i]) for i in range(len(lon))]
        self._oq_obj['mesh'] = Mesh.from_points_list(self._oq_obj['points'])
    
    
    def _generate_fault_surfaces(self):
        """generate OpenQuake simple fault surfaces"""
        # create surfaces for all ruptures
        if self._oq_obj['surfaces'] is not None:
            self._oq_obj['surfaces']  = None
        
        # count = 0
        # for ind in range(self.df_rupture.shape[0]):
        #     try:
        #         SimpleFaultSurface.from_fault_data(
        #             fault_trace=Line(
        #                 [Point(
        #                     self.df_rupture.FaultTrace[ind][i][0],
        #                     self.df_rupture.FaultTrace[ind][i][1],
        #                     # self.df_rupture.FaultTrace[ind][i][2]
        #                     self.df_rupture.UpperDepth[ind]
        #                 ) for i in range(len(self.df_rupture.FaultTrace[ind]))]),
        #             upper_seismogenic_depth=self.df_rupture.UpperDepth[ind],
        #             lower_seismogenic_depth=self.df_rupture.LowerDepth[ind],
        #             dip=self.df_rupture.Dip[ind],
        #             mesh_spacing=self._mesh_spacing # km
        #         )
        #     except ValueError:
        #         print(ind+1)
        #         count += 1
            
        #     if ind == 0:
        #         print(SimpleFaultSurface.from_fault_data(
        #             fault_trace=Line(
        #                 [Point(
        #                     self.df_rupture.FaultTrace[ind][i][0],
        #                     self.df_rupture.FaultTrace[ind][i][1],
        #                     # self.df_rupture.FaultTrace[ind][i][2]
        #                     self.df_rupture.UpperDepth[ind]
        #                 ) for i in range(len(self.df_rupture.FaultTrace[ind]))]),
        #             upper_seismogenic_depth=self.df_rupture.UpperDepth[ind],
        #             lower_seismogenic_depth=self.df_rupture.LowerDepth[ind],
        #             dip=self.df_rupture.Dip[ind],
        #             mesh_spacing=self._mesh_spacing # km
        #         ))
        # print(count)
                # print(self.df_rupture.loc[ind])
                # sys.exit()
        self._oq_obj['surfaces'] = [
            SimpleFaultSurface.from_fault_data(
                fault_trace=Line(
                    [Point(
                        self.df_rupture.FaultTrace[ind][i][0],
                        self.df_rupture.FaultTrace[ind][i][1],
                        # self.df_rupture.FaultTrace[ind][i][2]
                        self.df_rupture.UpperDepth[ind]
                    ) for i in range(len(self.df_rupture.FaultTrace[ind]))]),
                upper_seismogenic_depth=self.df_rupture.UpperDepth[ind],
                lower_seismogenic_depth=self.df_rupture.LowerDepth[ind],
                dip=self.df_rupture.Dip[ind],
                mesh_spacing=self._mesh_spacing # km
            ) for ind in range(self.df_rupture.shape[0])
        ]
        logging.info(f"\t- Generated simple fault surfaces for ruptures")
        
        
    def _find_ruptures_within_max_dist(self):
        """find list of ruptures within max distance and sites within max distance of said rupture"""
        self.sites_for_rupture_in_maxdist = {
            i: np.where(self._oq_obj['surfaces'][i].get_min_distance(self._oq_obj['mesh']) <= self._max_dist)[0] 
            for i in range(self.df_rupture.shape[0])
            if min(self._oq_obj['surfaces'][i].get_min_distance(self._oq_obj['mesh'])) <= self._max_dist
        }
        # get dictionary keys
        self.rupture_in_maxdist = list(self.sites_for_rupture_in_maxdist.keys())
        logging.info(f"\t- Retrieved ruptures within max distance of {self._max_dist} km from sites")
    
    
    @staticmethod
    def _get_dist_from_sites_to_rupture(rupture_id, list_of_sites, oq_points, oq_surfaces, decimals=1):
        """calculates distance metrics using OpenQuake"""
        # create reduced mesh with current set of sites
        loc_mesh = Mesh.from_points_list([oq_points[site] for site in list_of_sites])
        # calculate distances
        r_rup = np.round(oq_surfaces[rupture_id].get_min_distance(loc_mesh),decimals=decimals)
        r_jb = np.round(oq_surfaces[rupture_id].get_joyner_boore_distance(loc_mesh),decimals=decimals)
        r_x = np.round(oq_surfaces[rupture_id].get_rx_distance(loc_mesh),decimals=decimals)
        r_y0 = np.round(oq_surfaces[rupture_id].get_ry0_distance(loc_mesh),decimals=decimals)
        # return
        return r_rup, r_jb, r_x, r_y0
    
    
    def _get_distances_to_ruptures(self):
        """calculates distances from sites to ruptures, to be used for determining controlling fault parameters for GMPE"""
        # initialize table to store distances for sites within dist_max
        r_rup_for_rupture = {}
        r_jb_for_rupture = {}
        r_x_for_rupture = {}
        r_y0_for_rupture = {}
        # loop through each rupture within max distance and get all distance metrics for NGAW2 GMPEs
        for rupture in self.rupture_in_maxdist:
            # # calculate distances
            r_rup, r_jb, r_x, r_y0 = self._get_dist_from_sites_to_rupture(
                rupture,
                self.sites_for_rupture_in_maxdist[rupture],
                self._oq_obj['points'],
                self._oq_obj['surfaces']
            )
            r_rup_for_rupture[rupture] = r_rup
            r_jb_for_rupture[rupture] = r_jb
            r_x_for_rupture[rupture] = r_x
            r_y0_for_rupture[rupture]  = r_y0
        # convert from dictionary to array and store
        shape = (self.df_rupture.shape[0],self._n_site)
        self.r_rup_table = from_dict_to_array(r_rup_for_rupture,self.sites_for_rupture_in_maxdist,shape,fill_zeros=999)
        self.r_jb_table = from_dict_to_array(r_jb_for_rupture,self.sites_for_rupture_in_maxdist,shape,fill_zeros=999)
        self.r_x_table = from_dict_to_array(r_x_for_rupture,self.sites_for_rupture_in_maxdist,shape,fill_zeros=999)
        self.r_y0_table = from_dict_to_array(r_y0_for_rupture,self.sites_for_rupture_in_maxdist,shape,fill_zeros=999)
        logging.info(f"\t- Calculated r_rup, r_jb, r_x, and r_y0 between sites and ruptures")
        
        
    def _filter_rupture(self):
        """list of scenarios in max distance, magnitude, and mean annual rate"""
        logging.info(f"\t- Filtered rupture scenarios by:")
        # filter by distance
        # rupture_in_maxdist = np.unique(np.hstack(self.df_rupture.RupturesForSection[self.section_in_maxdist]))
        self.df_rupture = self.df_rupture.iloc[self.rupture_in_maxdist,:].copy().reset_index(drop=True)
        logging.info(f"\t\t- max distance: {self._max_dist} km")
        # filter by magnitudes
        if self._mag_min is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.Magnitude>=self._mag_min].reset_index(drop=True)
            logging.info(f"\t\t- min magnitude: {self._mag_min}")
        if self._mag_max is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.Magnitude<=self._mag_max].reset_index(drop=True)
            logging.info(f"\t\t- min magnitude: {self._mag_max}")
        # filter for mean annual rate
        if self._rate_min is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.AnnualRate>=self._rate_min].reset_index(drop=True)
            logging.info(f"\t\t- min mean annual rate: {self._rate_min}")
        if self._rate_max is not None:
            self.df_rupture = self.df_rupture.loc[self.df_rupture.AnnualRate<=self._rate_max].reset_index(drop=True)  
            logging.info(f"\t\t- min mean annual rate: {self._rate_min}")
        self._n_event = self.df_rupture.shape[0]
        logging.info(f"\t- Number of events remaining after filter: {self._n_event}")
        

    def _get_gmpe_input_for_event_i(self, i, im_list, site_data):
        """compile inputs for event i for GMPE calculations"""
        # sections for current scenario
        # sections_for_scenario = self.df_rupture.SectionsForRupture.iloc[i]
        # find list of sites within max distance
        # sites_for_rupture_within_maxdist = np.unique(np.hstack([
        #     self.sites_for_section_in_maxdist[section] for section in self.sites_for_section_in_maxdist.keys()
        #     if section in sections_for_scenario]))
        rup_ind = self.rupture_in_maxdist[i]
        sites_for_rupture_within_maxdist = self.sites_for_rupture_in_maxdist[rup_ind]
        # set up table of inputs
        inputs_for_gmpe = DataFrame(
            None,
            index=list(range(len(sites_for_rupture_within_maxdist))),
            columns=[
                'site_id',
                # 'source_id',
                'event_id',
                'mag',
                'rate',
                # 'section_id_control',
                'dip',
                'rake',
                'z_tor',
                'z_bor',
                'r_rup',
                'r_jb',
                'r_x',
                'r_y0'
            ]
        )
        # get rupture characteristics
        inputs_for_gmpe.site_id = site_data['site_id'][sites_for_rupture_within_maxdist]
        inputs_for_gmpe.event_id = self.df_rupture.EventID[i]
        inputs_for_gmpe.mag = self.df_rupture.Magnitude[i]
        inputs_for_gmpe.rate = self.df_rupture.AnnualRate[i]
        inputs_for_gmpe.rake = self.df_rupture.Rake[i]
        # get controlling section id
        # section_control = self.get_controlling_section_for_site(
        #     sections_for_scenario,
        #     self.r_rup_table,
        #     sites_for_rupture_within_maxdist
        # )
        section_control = None
        # get rest of gmpe inputs based on controlling section id    
        # inputs_for_gmpe.section_id_control = section_control
        inputs_for_gmpe.r_rup = self.r_rup_table[rup_ind,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.r_jb = self.r_jb_table[rup_ind,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.r_x = self.r_x_table[rup_ind,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.r_y0 = self.r_y0_table[rup_ind,sites_for_rupture_within_maxdist]
        inputs_for_gmpe.dip = self.df_rupture.Dip[i]
        inputs_for_gmpe.z_tor = self.df_rupture.UpperDepth[i]
        inputs_for_gmpe.z_bor = self.df_rupture.LowerDepth[i]
        # add in other site params if available
        for site_param in ['vs30','z1p0','z2p5','vs30_source']:
            if site_data[site_param] is not None:
                inputs_for_gmpe[site_param] = site_data[site_param][sites_for_rupture_within_maxdist]
        # convert vs30_source to numerics: 0=inferred/estimated, 1=measured
        vs30_source_num = np.ones(inputs_for_gmpe.vs30_source.shape)
        vs30_source_num[inputs_for_gmpe.vs30_source=='Inferred'] == 0
        inputs_for_gmpe['vs30_source'] = vs30_source_num
        # convert inputs into dictionary and return
        kwargs = {}
        for col in inputs_for_gmpe.columns:
            kwargs[col] = inputs_for_gmpe[col].values
        kwargs['period_out'] = im_list
        return kwargs
    
    

# -----------------------------------------------------------
class UCERF(UserDefinedRupture):
    """
    UCERF class using reduced list of scenarios by Norm Abrahamson (2022)
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    
    # class definitions
    SUPPORTE_UCERF_MODEL = ['Mean UCERF3 FM3.1'] # supported sourcess
    
    
    # instantiation
    def __init__(self, ucerf_model='Mean UCERF3 FM3.1'):
        """Create an instance of the class"""
        
        # invoke parent function
        # super().__init__()
        
        # get inputs
        self.ucerf_model = ucerf_model
        # check against supported ucerf models
        self._check_ucerf_support()
        
        # load file
        self._load_rupture_file() # loads ruptures from user specified file
        
        # initialize empty params
        self.oq_surfaces = None # OpenQuake surfaces
        self._max_dist = None
        self._mag_min = None
        self._mag_max = None
        self._rate_min = None
        self._rate_max = None
        self._mesh_spacing = None
        
        
    @property
    def supported_ucerf_model(self):
        """supported models"""
        return self.SUPPORTE_UCERF_MODEL
    
    
    def _check_ucerf_support(self):
        """check for support"""
        self._ucerf_model_support = False
        if self.ucerf_model in self.supported_ucerf_model:
            self._ucerf_model_support = True
        else:
            raise NotImplementedError(
                f'"{self.ucerf_model}" is not a supported type for seismic source characterization;' + 
                f'supported source types include: {*self.supported_ucerf_model,}'
            )
    
    
    def _load_rupture_file(self):
        """load rupture file provided by the user. Only allowed CSV format at this stage"""
        # search and get for path to OpenSRA directory
        _opensra_dir = os.path.realpath(__file__)
        count = 0
        while not _opensra_dir.endswith('OpenSRA'):
            _opensra_dir = os.path.abspath(os.path.dirname(_opensra_dir))
            # in case can't locate OpenSRA dir and goes into infinite loop
            if count>5:
                raise FileNotFoundError(
                    'URGENT: Cannot locate OpenSRA directory for sourcing UCERF scenarios - contact dev.'
                )
                break
        # get file path to reduced list of ucerf scenarios
        self._ucerf_base_dir = os.path.join(_opensra_dir,"lib","UCERF3","ReducedEvents_Abrahamson2022")
        self.ucerf_model_dir = os.path.join(self._ucerf_base_dir,self.ucerf_model)
        self.ucerf_model_fpath = os.path.join(self.ucerf_model_dir,"UCERF3_reduced_senario_dM0.5.csv")
        # load ucerf rupture file
        self.df_rupture = pd.read_csv(self.ucerf_model_fpath)
        # convert fault trace from string to list of coordinates
        self.df_rupture.FaultTrace = self.df_rupture.FaultTrace.apply(lambda x: json.loads(x))
        # get number of events
        self._n_event = self.df_rupture.shape[0]
        logging.info(f"Loaded {self.ucerf_model} ruptures")