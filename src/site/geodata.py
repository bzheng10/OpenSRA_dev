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


# Python base modules
import os
import logging
import json
import sys
import importlib
import time
import pathlib
import warnings
import copy
import shutil
from itertools import chain
# import itertools

# data manipulation modules
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import interp2d, griddata
from scipy.spatial.distance import cdist
# from scipy import sparse

# geospatial processing modules
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon
from shapely.ops import linemerge, polygonize
from shapely.prepared import prep
import rasterio as rio
import rasterio.features
import rasterio.warp
from rasterio.plot import show, adjust_band
# from shapely.strtree import STRtree
# from rtree import index

# efficient processing modules
# from numba import jit, njit

# plotting modules
if importlib.util.find_spec('matplotlib') is not None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
if importlib.util.find_spec('contextily') is not None:
    import contextily as ctx

# OpenSRA modules
# from src.im import opensha as sha
from src.util import *
from src.site.site_util import polygonize_cells, split_line_by_max_length, get_segment_within_bound


# -----------------------------------------------------------
class GeoData(object):
    """
    Base class for data files
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    
    # class definitions
    SUPPORTED_FTYPE = ['.shp', '.gdb','.tif','.xml', '.csv', '.from_args'] # supported file types
    
    
    # instantiation
    def __init__(self, fpath, crs='EPSG:4326'):
        """Create an instance of the class"""
        
        # get inputs
        self.fpath = fpath
        self.crs = crs
        # check file type against supported file types
        self.ftype = pathlib.Path(self.fpath).suffix
        self._check_ftype_support()
        
        # additional setup
        # pathing
        self._fname = os.path.basename(self.fpath)
        self._fdir = os.path.dirname(self.fpath)
        # initialize instance variables
        self.data = None
        # for plotting
        self._fig = None
        self._ax = None
        self._fignum = None
        # for exporting
        self._fig_spath = None
    
    
    @property
    def supported_ftype(self):
        """supported file types"""
        return self.SUPPORTED_FTYPE
    
    
    def _check_ftype_support(self):
        self._ftype_support = False
        if self.ftype in self.supported_ftype:
            self._ftype_support = True
        else:
            raise NotImplementedError(
                f'"{self.ftype}" is not a supported geodatafile type; supported file types include: {*self.supported_ftype,}'
            )
    
    
    @classmethod
    def sample_csv(cls, table, fpath, cols_to_get='all', 
                   append_header_str='', interp_scheme='nearest', out_of_bound_value=np.nan,
                   use_hull=True
    ):
        """sample values from CSV files for columns to get and update GeoDataFrame table, must be GeoDataFrame"""        
        # import sample set
        sample_set = LocationData(fpath=fpath).data
        table = cls.sample_gdf(
            table=table.copy(), sample_set=sample_set, cols_to_get=cols_to_get, 
            append_header_str=append_header_str, interp_scheme=interp_scheme,
            out_of_bound_value=out_of_bound_value, use_hull=use_hull
        )
        return table
    
    
    @staticmethod
    def sample_gdf(table, sample_set, cols_to_get='all', 
                   append_header_str='', interp_scheme='nearest',
                   out_of_bound_value=np.nan, use_hull=False
    ):
        """sample values from GeoDataFrame table (sample_set) and add to table; table must also be a GeoDataFrame"""
        # check columns to get
        if cols_to_get == 'all':
            cols_to_get = sample_set.columns.drop('geometry')
        if interp_scheme == 'nearest':
            # first determine which locations are within and outside the convex hull of the sample set
            sindex = table.sindex # spatial index for location data
            if use_hull:
                hull = sample_set.geometry.unary_union.convex_hull
                loc_ind_in_hull = sindex.query(hull,predicate='intersects')
                loc_ind_outside_hull = list(set(list(table.index)).difference(set(loc_ind_in_hull)))
                # perform sampling for locations within convex hull
                locs_in_hull = table.loc[loc_ind_in_hull].copy().reset_index(drop=True) # get subset of table in hull
                nearest_pt_in_sample_set = sample_set.sindex.nearest(locs_in_hull.geometry,return_all=False) # sampling
            else:
                nearest_pt_in_sample_set = sample_set.sindex.nearest(table.geometry,return_all=False) # sampling
            # update data table with samples
            for col in cols_to_get:
                # initialize
                if col == 'vs30source':
                    sampled_vals = np.empty(table.shape[0],dtype='<U10')
                    if use_hull:
                        sampled_vals[loc_ind_outside_hull] = 'Inferred'
                else:
                    sampled_vals = np.empty(table.shape[0])*out_of_bound_value
                # get samples
                if use_hull:
                    sampled_vals[loc_ind_in_hull] = sample_set[col].loc[nearest_pt_in_sample_set[1]].values
                else:
                    sampled_vals = sample_set[col].loc[nearest_pt_in_sample_set[1]].values
                # store samples
                table[col+append_header_str] = sampled_vals
            return table
        else:
            raise NotImplementedError('Only "nearest" is available as interpolation scheme')
    
    
    @staticmethod
    def sample_raster(table, fpath,
                      band=1, store_name=None, interp_scheme='nearest',
                      out_of_bound_value=np.nan, invalid_value=np.nan
    ):
        """sample values from raster file and update (Geo)DataFrame datatable"""
        # create raster object
        raster = RasterData(fpath)
        x = table.geometry.x.values
        y = table.geometry.y.values
        # perform sampling
        raster.get_sample(
            x=x, y=y, band=band,
            interp_scheme=interp_scheme, out_of_bound_value=out_of_bound_value, invalid_value=invalid_value
        )
        # update data table with samples
        if store_name is None:
            store_name = get_basename_without_extension(fpath) # use raster file name, without extension
        table[store_name] = raster.sample
        return table
    
    
    @staticmethod
    def sample_shapefile(table, fpath, attr, store_name=None):
        """sample values from shapefile and update GeoDataFrame table, must be GeoDataFrame"""
        # create shapefile object
        shapefile = ShapefileData(fpath)
        # perform sampling
        shapefile.get_sample(attr=attr, site_geometry=table.geometry)
        index = shapefile.site_index_with_sample
        sample = shapefile.sample
        # update data table with samples
        if store_name is None:
            store_name = get_basename_without_extension(fpath) # use raster file name, without extension
        table[store_name] = np.nan # create new column and initialize all values as nan
        table.loc[index, store_name] = sample
        return table
    
    
    @staticmethod
    def sample_xml():
        """sample values from XML file"""
        raise NotImplementedError("Method to be implemented")
            

    def plot(self, show=True, figsize=[16,8], fignum=None):
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
            """plots geodata"""
            # create new or use existing figure by number
            if fignum is None:
                # initialize figure and axes
                if self._fignum is None:
                    # initialize plot
                    self._fig, self._ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, squeeze=False, sharex=True, sharey=True)
                    # track figure number
                    self._fignum = plt.gcf().number
                else:
                    # close all
                    plt.close(self._fignum)
                    # initialize plot
                    self._fig, self._ax = plt.subplots(num=self._fignum, nrows=1, ncols=1, figsize=figsize, squeeze=False, sharex=True, sharey=True)
            else:
                # close all
                plt.close(fignum)
                # initialize plot
                self._fig, self._ax = plt.subplots(num=fignum, nrows=1, ncols=1, figsize=figsize, squeeze=False, sharex=True, sharey=True)
                # track figure number
                self._fignum = fignum

            # other formatting operations
            self._ax[0,0].set_aspect('equal')
            self._ax[0,0].autoscale()
            self._fig.tight_layout()
            
            # show or not
            if show is True:
                plt.show()
    
    
    @staticmethod
    def clip_data_with_polygon(data, polygon, buffer=0):
        """
        Return data intersected by boundary polygon, for data with many entries
        1. data should be a pd.DataFrame object
        2. poly should be a shapely Polygon/MultiPolygin object
        """
        sindex = data.sindex
        query = sindex.query_bulk(polygon, predicate='intersects')
        return np.sort(query[1])
    
    
    def add_basemap(self, source=None, zoom='auto'):
    # def add_basemap(self, source=None, zoom=11):
        """adds basemap to figure"""
        # if contextily is loaded, plot basemap
        if not 'contextily' in sys.modules:
            logging.info('The "contextily" module is not installed; cannot generate basemap')
        else:
            # pass
            # ctx.add_basemap(ax=self._ax[0,0], source=ctx.providers.Stamen.TonerLite, crs=self.crs)
            # ctx.add_basemap(ax=self._ax[0,0], source=ctx.providers.USGS.USTopo, crs=self.crs)
            if source is None:
                # source = ctx.providers.Stamen.TonerLite
                source = ctx.providers.Stamen.Terrain
            ctx.add_basemap(ax=self._ax[0,0], source=source, zoom=zoom, crs=self.crs)
        
    
    def export_plot(self, spath=None, fmt='pdf', dpi='figure', orientation='landscape'):
        """export figure"""
        # only works if matplotlib is loaded
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot export figure')
        else:
            if spath is None:
                # use directory where shapefile is located
                self._fig_spath = os.path.join(os.path.dirname(self.fpath),f'figure.{fmt}')
            else:
                self._fig_spath = spath
            self._fig.savefig(self._fig_spath, transparent=True, orientation=orientation, bbox_inches='tight')


# -----------------------------------------------------------
class CSVData(GeoData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    

    # instantiation
    def __init__(self, fpath, lon_header, lat_header,
                 shape_type='point', crs="EPSG:4326",
                 lon_begin_header=None, lat_begin_header=None,
                 lon_end_header=None, lat_end_header=None,
        ):
        # invoke parent function
        super().__init__(fpath, crs=crs)
        
        # other inputs
        self._shape_type = shape_type
        self._lon_header = lon_header
        self._lat_header = lat_header
        self._lon_begin_header = lon_begin_header
        self._lat_begin_header = lat_begin_header
        self._lon_end_header = lon_end_header
        self._lat_end_header = lat_end_header
        
        # read data
        self._read_data()
        # convert data to GeoDataFrame
        self._convert_to_gdf()
        
        # initialize other attributes
        # geometry coordinates
        self.bound = None
        self._bound_coord = None
        # for exporting
        self._spath = None

    
    def _read_data(self):
        """reads data"""
        if os.path.exists(self.fpath):
            self.data = pd.read_csv(self.fpath)
            
            
            
        else:
            raise ValueError(f'CSV file "{os.path.basename(self.fpath)}" does not exist')

    
    def _convert_to_gdf(self):
        """converts to geopandas.GeoDataFrame"""
        # see if column ID for lon and lat are given
        # if self._lon_header is None or self._lat_header is None:
        #     if self._lon_header is None:
        #         try:
        #             self._lon_header = next(col for col in self.data.columns if 'lon' in col.lower())
        #         except StopIteration:
        #             return ValueError('Program cannot automatically locate column with "Longitude" data; please specific "lon_header"')
        #     if self._lat_header is None:
        #         try:
        #             self._lat_header = next(col for col in self.data.columns if 'lat' in col.lower())
        #         except StopIteration:
        #             return ValueError('Program cannot automatically locate column with "Latitude" data; please specific "lat_header"')
        # convert to GeoDataFrame
        if self._shape_type == 'point':
            # make columns of lat lon using default header names
            self.data['LON'] = self.data[self._lon_header].values
            self.data['LAT'] = self.data[self._lat_header].values
            # make GeoDataFrame
            self.data = GeoDataFrame(
                self.data,
                crs=self.crs,
                geometry=[Point(xy) for xy in zip(self.data.LON, self.data.LAT)]
            )
        elif self._shape_type == 'line':
            # make columns of lat lon using default header names
            self.data['LON_MID'] = self.data[self._lon_header].values
            self.data['LAT_MID'] = self.data[self._lat_header].values
            self.data['LON_BEGIN'] = self.data[self._lon_begin_header].values
            self.data['LAT_BEGIN'] = self.data[self._lat_begin_header].values
            self.data['LON_END'] = self.data[self._lon_end_header].values
            self.data['LAT_END'] = self.data[self._lat_end_header].values
            # make GeoDataFrame
            begin_pair = list(zip(self.data.LON_BEGIN, self.data.LAT_BEGIN))
            end_pair = list(zip(self.data.LON_END, self.data.LAT_END))
            n_pair = len(begin_pair)
            self.data = GeoDataFrame(
                self.data,
                crs=self.crs,
                geometry=[LineString([begin_pair[i],end_pair[i]]) for i in range(n_pair)]
            )
       
        
    def export_data_to_shp(self, spath=None):
        """exports geopandas.GeoDataFrame to shapefile"""
        if spath is None:
            self._spath = self.fpath.replace('.csv','.shp') # file path
        else:
            self._spath = spath
        self.data.to_file(self._spath) # export operation
        logging.info(f"Exported GeoDataFrame to:")
        logging.info(f"\t{self._spath}")
    
    
    def get_bound(self):
        """gets approximate boundary of grid obtained via convex hull"""
        self.bound = self.data.geometry.unary_union.convex_hull
        self._bound_coord = np.asarray(self.bound.boundary.coords)
        
        
    def export_bound(self, spath=None):
        """exports approximate boundary"""
        if spath is None:
            self._bound_spath = self.fpath.replace('.csv','_BOUND.shp') # file path
        else:
            self._bound_spath = spath
        gs = GeoSeries([self.bound], crs=self.crs)
        gs.to_file(self._bound_spath) # export operation
        logging.info(f"Exported boundary to:")
        logging.info(f"\t{self._bound_spath}")
    
    
    def plot(self, show=True, figsize=[16,8], plot_bound=False, facecolor='none', edgecolor='g', marker='.', markersize=6, add_basemap=False, zoom='auto'):
        """plots geodata"""
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
            # invoke parent function
            super().plot(show=False)
            # plot shapedata
            self.data.geometry.plot(ax=self._ax[0,0], facecolor=facecolor, edgecolor=edgecolor, marker=marker, markersize=markersize)
            # plot approximate boundary of grid obtained via convex hull
            if plot_bound:
                if self._bound_coord is None:
                    self.get_bound()
                self._ax[0,0].plot(self._bound_coord[:,0], self._bound_coord[:,1], 'k')
            # add basemap
            if add_basemap:
                self.add_basemap(zoom=zoom)
            # show or not
            if show is True:
                plt.show()


# -----------------------------------------------------------
class DataFromArgs(CSVData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    
    # instantiation
    def __init__(self,
                 lon, lat, lon_header="LON", lat_header="LAT",
                 shape_type='point', crs='EPSG:4326',
                 lon_begin=None, lat_begin=None, lon_begin_header="LON_BEGIN", lat_begin_header="LAT_BEGIN",
                 lon_end=None, lat_end=None, lon_end_header="LON_END", lat_endheader="LAT_END"):
        # invoke parent function
        # GeoData().__init__(fpath)
        # GeoData(fpath)
        
        # check shape type
        if shape_type != 'point' and shape_type != 'line':
            raise ValueError('shape_type must be either "point" or "line"')
        
        # get inputs
        self._shape_type = shape_type
        self.crs = crs
        
        # other inputs
        # append dimensions if dim = 0
        self._lon = self._check_and_expand_dim(lon)
        self._lat = self._check_and_expand_dim(lat)
        self._lon_header = lon_header
        self._lat_header = lat_header
        # if line, also get begin and end points
        if self._shape_type == 'line':
            # begin point
            if lon_begin is None:
                self._lon_begin = self._lon
            else:
                self._lon_begin = self._check_and_expand_dim(lon_begin)
            if lat_begin is None:
                self._lat_begin = self._lat
            else:
                self._lat_begin = self._check_and_expand_dim(lat_begin)
            self._lon_begin_header = lon_begin_header
            self._lat_begin_header = lat_begin_header
            # end point
            if lon_end is None:
                self._lon_end = self._lon
            else:
                self._lon_end = self._check_and_expand_dim(lon_end)
            if lat_end is None:
                self._lat_end = self._lat
            else:
                self._lat_end = self._check_and_expand_dim(lat_end)
            self._lon_end_header = lon_end_header
            self._lat_end_header = lat_end_header
        
        # create data table
        self._create_data_table()
        
        # convert data to GeoDataFrame
        self._convert_to_gdf()
        
        # initialize other attributes
        # geometry coordinates
        # for exporting

    @staticmethod
    def _check_and_expand_dim(x):
        if np.ndim(x) == 0:
            return np.expand_dims(x)
        else:
            return x

    def _create_data_table(self):
        """creates DataFrame for input data"""
        if self._shape_type == 'point':
            self.data = pd.DataFrame(
                zip(self._lon,self._lat),
                columns=[self._lon_header,self._lat_header]
            )
        elif self._shape_type == 'line':
            # calculate length
            dist = get_haversine_dist(
                self._lon_begin, self._lat_begin,
                self._lon_end, self._lat_end,
                unit='km'
            )
            self.data = pd.DataFrame(
                zip(self._lon,self._lat,
                    self._lon_begin,self._lat_begin,
                    self._lon_end,self._lat_end,
                    dist),
                columns=[
                    self._lon_header,self._lat_header,
                    self._lon_begin_header,self._lat_begin_header,
                    self._lon_end_header,self._lat_end_header,
                    'LENGTH_KM'
                    ]
            )
    

    def _convert_to_gdf(self):
        """converts to geopandas.GeoDataFrame"""
        # convert to GeoDataFrame
        if self._shape_type == 'point':
            self.data = GeoDataFrame(
                self.data,
                crs=self.crs,
                geometry=[Point(xy) for xy in zip(self.data[self._lon_header], self.data[self._lat_header])]
            )
        elif self._shape_type == 'line':
            begin_pair = list(zip(self.data[self._lon_begin_header], self.data[self._lat_begin_header]))
            end_pair = list(zip(self.data[self._lon_end_header], self.data[self._lat_end_header]))
            n_pair = len(begin_pair)
            self.data = GeoDataFrame(
                self.data,
                crs=self.crs,
                geometry=[LineString([begin_pair[i],end_pair[i]]) for i in range(n_pair)]
            )
    
    
    def _read_data(self):
        """unused"""
        return "Not used by this class"


# -----------------------------------------------------------
class ShapefileData(GeoData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    

    # instantiation
    def __init__(self, fpath, crs='EPSG:4326', rows_to_keep=None, minimal_init=False, to_cleanup=True):
        # invoke parent function
        super().__init__(fpath, crs)
        
        if minimal_init == False:
            # read data
            self._read_data()
            if to_cleanup:
                # clean up data: remove rows without geometry, expand multi-objects to individual (e.g., MultiLineString->LineString)
                self._cleanup_data(rows_to_keep)
            # other initial processing
            self._geom_type = self.get_geometry_type(self.data)
        
        # initialize other attributes
        # geometry coordinates
        self.data_coord = None
        self._prepped_geom = None
        # for grid around network
        self.data_extent = None
        self._grid_spacing = None
        self.grid = None
        self.grid_node = None
        self.grid_node_table = None
        # for exporting
        self._spath = None
        self._fig_spath = None
        self._grid_spath = None
        self._grid_node_spath = None
        # for sampling
        self.site_geometry = None
        self.site_index_with_sample = None
        self.sample = None
    
    
    def _read_data(self):
        """reads data"""
        if os.path.exists(self.fpath):
            self.data = gpd.read_file(self.fpath)
        else:
            raise ValueError(f'Shapefile "{os.path.basename(fpath)}" does not exist')
        self.change_crs() # convert to default CRS epsg:4326
        self.data[self.data.columns] = self.data[self.data.columns].apply(pd.to_numeric, errors='ignore') # convert to numerics if possible

    
    def change_crs(self):
        """updates crs of locations"""
        if self.ftype == '.shp' or self.ftype == '.gdb':
            try:
                self.data.to_crs(self.crs, inplace=True)
            except ValueError:
                self.data.set_crs(self.crs, inplace=True)
    
    
    def _cleanup_data(self, rows_to_keep=None):
        """
        clean up shapefile data:
        1. keep rows specified by user (default: keep all rows)
        2. remove rows without geometry
        3. expand multi-objects to individual (e.g., MultiLineString->LineString)
        """
        if rows_to_keep is not None:
            self.data = self.data.iloc[rows_to_keep].reset_index(drop=True)
        self.data.drop(np.where(self.data.geometry.isna())[0], inplace=True) # drop rows with no geometry
        # self.data = self.data.explode(ignore_index=True) # expand multi-objects
        self.data = self.data.explode(ignore_index=False, index_parts=True) # expand multi-objects
        if not 'obj_id' in self.data:
            self.data['obj_id'] = [pair[0]+1 for pair in self.data.index]
        # self.data['pipe_sub_id'] = [pair[1] for pair in self.data.index]
        self.data.reset_index(drop=True,inplace=True)
        # go through geometry, if it contains more than two points, split into additional segments
        for i in range(self.data.shape[0]):
            geom = self.data.geometry[i]
            if isinstance(geom,LineString):
                if len(geom.xy[0]) > 2:
                    x,y = geom.xy
                    geom_new = MultiLineString(
                        [LineString([[x[j],y[j]],[x[j+1],y[j+1]]])
                        for j in range(len(x)-1)]
                    )
                    self.data.geometry[i] = geom_new
        # self.data = self.data.explode(ignore_index=True) # expand again
        self.data = self.data.explode(ignore_index=False, index_parts=True) # expand again
        if not 'obj_id' in self.data:
            self.data['obj_id'] = [pair[0]+1 for pair in self.data.index]
        # self.data['pipe_sub_id'] = [pair[1] for pair in self.data.index]
        self.data.reset_index(drop=True,inplace=True)
        # additional operations just for prepackaged CA state boundary shapefile
        # if 'California_State_Boundary' in os.path.basename(self.fpath):
        #     pass            
            # self._ops_for_state_boundary()
    
    
    def _ops_for_state_boundary(self):
        """additional operations just for prepackaged CA state boundary shapefile"""
        self.data = self.data.iloc[[-1]].reset_index(drop=True)
    
    
    @staticmethod
    def get_coord(gdf, geom_type):
        """extract coordinates from shapes"""
        # for Polygon
        # if isinstance(self._geom_type,Polygon):
        if geom_type == 'Polygon':
           return [list(geom.boundary.coords) for geom in gdf.geometry]
        # for LineString
        # elif isinstance(self._geom_type,LineString):
        elif geom_type == 'LineString':
            return [list(geom.coords) for geom in gdf.geometry]
        else:
            return NotImplementedError(f'"{geom_type}" is not supported; valid geometry types are: ("Polygon" and "LineString")')
    
    
    @staticmethod
    def get_extent(gdf):
        """get total extent of shapefile"""
        return gdf.geometry.total_bounds
    
    
    def reload_data(self, fpath=None):
        """Reloads data by reinstantiating class"""
        if fpath is not None:
            self.fpath = fpath
        self.__init__(self.fpath, self.crs)
    

    @staticmethod
    def get_geometry_type(gdf):
        """gets shape type for geometries in a GeoDataFrame based on first geometry in the list"""
        return gdf.geometry[0].geom_type


    @staticmethod
    def prepare_geometry(gdf, buffer=0):
        """apply Shapely's "prep" function on a GeoDataFrame of Polygons for faster boolean operation"""
        return prep(MultiPolygon(list(gdf.geometry)).buffer(buffer))
    

    def make_grid_over_extent(self, grid_extent=None, grid=None, clip=False, spacing=0.1, buffer=0):
        """generate a grid at user-defined spacing that cover the extents of the network"""
        # if grid is not given, generate it
        if grid is None:
            # read inputs
            self._grid_spacing = spacing
            # calculate decimals, for rounding
            spacing_decimal = decimal_count(self._grid_spacing)
            # see if extent is given
            if grid_extent is None:
                # get extent
                if self.data_extent is None:
                    self.data_extent = self.get_extent(self.data)
                grid_extent = self.data_extent
            # ceil/floor to nearest multiple of dlon/dlat
            grid_extent[0] = smart_round(np.floor(grid_extent[0]/self._grid_spacing)*self._grid_spacing,spacing_decimal)
            grid_extent[1] = smart_round(np.floor(grid_extent[1]/self._grid_spacing)*self._grid_spacing,spacing_decimal)
            grid_extent[2] = smart_round(np.ceil(grid_extent[2]/self._grid_spacing)*self._grid_spacing,spacing_decimal)
            grid_extent[3] = smart_round(np.ceil(grid_extent[3]/self._grid_spacing)*self._grid_spacing,spacing_decimal)
            # create grid
            grid = make_grid(
                grid_extent[0], grid_extent[1], # x and y min
                grid_extent[2], grid_extent[3], # x and y max
                self._grid_spacing, self._grid_spacing # dx and dy
            )
            # polygonize cells in grid and convert to GeoSeries
            self.grid = GeoSeries(polygonize_cells(grid), crs=self.crs) # convert to pandas.GeoSeries for geoprocessing
        else:
            self.grid = grid.to_crs(self.crs) # use input grid
            grid_extent = self.grid.geometry[0].geoms[0].exterior.bounds
            self._grid_spacing = smart_round(grid_extent[2]-grid_extent[0])
        
        # clip grid over data
        if clip:
            _query = self._clip_grid_over_data(buffer)
        # get all nodes in grid
        self.get_grid_nodes()
        # get union of all grid cells
        self.grid = GeoSeries(MultiPolygon(list(self.grid)), crs=self.crs)

    
    def _clip_grid_over_data(self, buffer):
        """
        clips grid over data
        1. if the data is a set of Polygons, then keep grid cells that are contained or intersected by the data boundary
        2. if the data is a set of LineStrings, then keep grid cells that intersects at least one LineString (using rtree)
        """
        # for Polygon
        if self._geom_type == 'Polygon':
            # get prepped Polygons
            self._prepped_geom = self.prepare_geometry(self.data, buffer)
            # clip grid with polygon
            self.grid = GeoSeries(
                [cell for cell in self.grid if self._prepped_geom.contains(cell) or self._prepped_geom.intersects(cell)],
                crs=self.crs
            )
        # for LineString
        elif self._geom_type == 'LineString':
            # map grids to rtree and query for intersections
            sindex = self.data.sindex
            query = sindex.query_bulk(self.grid, predicate='intersects')
            self.grid = self.grid[np.unique(query[0])]
        else:
            return NotImplementedError(f'"{self._geom_type}" is not supported; valid geometry types are: ("Polygon" and "LineString")')        

    
    def get_grid_nodes(self, grid_node_table=None, lon_header='LON', lat_header='LAT'):
        """gets all unique nodes in the grid"""
        if grid_node_table is None:
            # compile coodrinates and place into a DataFrame table
            all_node = np.vstack([list(cell.boundary.coords) for cell in self.grid])
            self.grid_node_table = pd.DataFrame(
                all_node,
                columns=['LON','LAT']
            )
        else:
            # self.grid_node_table = pd.DataFrame(
            #     grid_node_table[[lon_header,lat_header]].values,
            #     columns=['LON','LAT']
            # )
            self.grid_node_table = grid_node_table.copy()
        # round to decimals of spacing, remove trailing error
        self.grid_node_table.LON = self.grid_node_table.LON.round(decimal_count(self._grid_spacing))
        self.grid_node_table.LAT = self.grid_node_table.LAT.round(decimal_count(self._grid_spacing))
        self.grid_node_table.drop_duplicates(ignore_index=True,inplace=True) # drops duplicate nodes
        # self.grid_node_table.reset_index(drop=True,inplace=True)
        # make GeoSeries for visualization
        self.grid_node = GeoSeries(
            Point(self.grid_node_table['LON'][i],self.grid_node_table['LAT'][i]) 
            for i in range(self.grid_node_table.shape[0])
        )
        
    
    def _clip_grid_node_with_self_bound(self, buffer=0):
        """remove grid nodes outside the data (self) boundary (shapely Polygon)"""
        bound_polygon = MultiPolygon(list(self.data.geometry))
        node_in_bound = self.clip_data_with_polygon(self.grid_node.copy(),bound_polygon,buffer=buffer)
        self.grid_node_clipped = self.grid_node.loc[node_in_bound].reset_index(drop=True)
        self.grid_node_table_clipped = self.grid_node_table.loc[node_in_bound].reset_index(drop=True)
    
    
    def get_sample(self, attr, site_geometry=None, x=None, y=None):
        """
        get attribute values at intersection between shapefile and list of locations
        users can provide either:
        1. site_geometry: a GeoSeries or GeoDataFrame with a list of Points (sites) as geometry
        2. x (lon) and y (lat) in arrays
        """
        sindex = self.data.sindex
        # only process x and y if site_geometry is not given:
        if site_geometry is None:
            site_geometry = GeoSeries([Point(x[i],y[i]) for i in range(len(x))], crs=self.crs)
        self.site_geometry = site_geometry
        query = sindex.query_bulk(site_geometry, predicate='intersects')
        self.site_index_with_sample = query[0]
        # if attr not specified:
        if attr is None:
            # set the first attribute that does is not "geometry" as attr
            attr = list(self.data.columns.drop('geometry'))[0]
            print(f'Since "attr" to sample from Shapefile is not given, will sample the first attribute: "{attr}"')
        # get value
        if attr in self.data:
            self.sample = self.data.loc[query[1], attr].values
        else:
            if attr.lower() in self.data:
                self.sample = self.data.loc[query[1], attr.lower()].values
            elif attr.upper() in self.data:
                self.sample = self.data.loc[query[1], attr.lower()].values
    
    
    def export_grid(self, export_grid=True, export_grid_node=True, export_fdir=None, grid_spath=None, grid_node_spath=None):
        """exports grid and nodes to shapefile and CSV files"""
        spacing_string = str(self._grid_spacing).replace('.','p')
        # for grid
        if export_grid:
            if grid_spath is None:
                if export_fdir is None:
                    self._grid_spath = self.fpath.replace('.shp',f"_GRID_{spacing_string}.shp") # file path
                else:
                    self._grid_spath = os.path.join(
                        export_fdir,
                        os.path.basename(self.fpath).replace('.shp',f"_GRID_{spacing_string}.shp")
                    ) # file path
            else:
                self._grid_spath = grid_spath
            self.grid.to_file(self._grid_spath) # export operation
            logging.info(f"Exported grid to:")
            logging.info(f"\t{self._grid_spath}")
        # for grid nodes
        if export_grid_node:
            if grid_node_spath is None:
                    if export_fdir is None:
                        self._grid_node_spath = self.fpath.replace('.shp',f"_GRID_NODE_{spacing_string}.csv") # file path
                        # self._grid_node_spath = os.path.join(os.path.dirname(self.fpath),'grid_nodes.csv')
                    else:
                        self._grid_node_spath = os.path.join(
                            export_fdir,
                            os.path.basename(self.fpath).replace('.shp',f"_GRID_NODE_{spacing_string}.csv")
                        ) # file path
            else:
                self._grid_node_spath = grid_node_spath
            self.grid_node_table.to_csv(self._grid_node_spath, index=False)
            logging.info(f"Exported grid nodes to:")
            logging.info(f"\t{self._grid_node_spath}")
    
    
    def _export_grid_node_clipped_with_self_bound(self, grid_node_spath=None):
        """exports grid nodes clipped with self boundary to CSV file"""
        spacing_string = str(self._grid_spacing).replace('.','p')
        # for grid nodes
        if grid_node_spath is None:
            self._grid_node_clipped_spath = self.fpath.replace('.shp',f"_GRID_NODE_IN_BOUND_{spacing_string}.csv") # file path
            # self._grid_node_spath = os.path.join(os.path.dirname(self.fpath),'grid_nodes.csv')
        else:
            self._grid_node_clipped_spath = grid_node_spath
        self.grid_node_table_clipped.to_csv(self._grid_node_clipped_spath, index=False)
        logging.info(f"Exported grid nodes clipped by self boundary to:")
        logging.info(f"\t{self._grid_node_clipped_spath}")
    
    
    def plot(self, show=True, figsize=[16,8], plot_base=True, facecolor='none', edgecolor='k', linewidth=2,
             attr_to_plot=None, vmin=None, vmax=None, cmap='Pastel2', add_basemap=False, plot_grid=True,
             plot_grid_nodes=False, zoom='auto'):
        """plots geodata"""
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
            # invoke parent function
            super().plot(show=False, figsize=figsize)
            # plot shapedata boundary
            if plot_base:
                # self.data.boundary.plot(ax=self._ax[0,0], edgecolor=edgecolor, linewidth=linewidth)
                self.data.geometry.plot(ax=self._ax[0,0], facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
            if attr_to_plot is not None:
                # if vmin is None:
                #     vmin = np.amin(self.data[attr_to_plot])
                # if vmax is None:
                #     vmax = np.amax(self.data[attr_to_plot])
                # plot attributes
                attr_dtype_is_numeric = is_numeric_dtype(self.data[attr_to_plot])
                attr_handle = self.data.plot(
                    column=attr_to_plot, ax=self._ax[0,0],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    missing_kwds={"color": "grey", "edgecolor": "none", "hatch": "///", "label": "Missing values",}
                )
            # plot grid, if generated
            if plot_grid:
                if self.grid is None:
                    logging.info(f'grid has not been generated; first run "make_grid_over_extent"')
                else:
                    self.grid.plot(ax=self._ax[0,0], facecolor='none', edgecolor='g', linewidth=1)
            # plot nearest points
            if plot_grid_nodes:
                if self.grid_node_table is not None:
                    self._ax[0,0].scatter(self.grid_node_table.LON,self.grid_node_table.LAT,c='blue',s=6)
                else:
                    logging.info(f'Nearest grid nodes have not been obtained; first run "get_nearest_grid_nodes"')
            # add basemap
            if add_basemap:
                self.add_basemap(zoom=zoom)
            # show or not
            if show is True:
                plt.show()


# -----------------------------------------------------------
# class LocationData(GeoData):
class LocationData(ShapefileData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    
    def __init__(self, fpath=None, lon=None, lat=None, lon_header='LON', lat_header='LAT',
                 crs='EPSG:4326', rows_to_keep=None):
        """initialize"""
        if fpath is None:
            if lon is None or lat is None:
                raise ValueError('Please specify either "fpath" or both "lon" and "lat"')
            else:
                fpath = "NONE.from_args"
        # invoke parent function
        # super().__init__(fpath, crs=crs)
        super().__init__(fpath=fpath, crs=crs, minimal_init=True)
        
        # compile data into GeoDataFrame
        if self.ftype == '.shp' or self.ftype == '.gdb':
            self.data = ShapefileData(
                fpath=fpath, rows_to_keep=rows_to_keep).data
        elif self.ftype == '.tif':
            self.data = RasterData(fpath=fpath).data
        elif self.ftype == '.xml':
            # self.data = XMLData(fpath).data
            raise NotImplementedError("Not implemented")
        elif self.ftype == '.csv':
            self.data = CSVData(
                fpath=fpath,lon_header=lon_header,lat_header=lat_header).data
        elif self.ftype == '.from_args':
            self.data = DataFromArgs(lon=lon,lat=lat).data
        
        # store other inputs
        self._lon_header = lon_header
        self._lat_header = lat_header
        
        # get x and y values for sampling
        self._get_xy()
        
        # initialize other attributes
        # geometry coordinates
        self.bound = None
        self._bound_coord = None
        # for exporting
        self._spath = None
    
    
    def _get_xy(self):
        """store values to x and y"""
        # see if column ID for lon and lat are given
        if self._lon_header is None or self._lat_header is None:
            try:
                self._x = self.data.LON_MID.to_numpy()
                self._y = self.data.LAT_MID.to_numpy()
            except AttributeError:
                try:
                    self._x = self.data.LON.to_numpy()
                    self._y = self.data.LAT.to_numpy()
                except AttributeError:
                    logging.info(f'Program cannot locate the columns with "Longitude"/"Latitude" data; please reload {self.__class__.__name__} and specify column headers')
        else:
            self._x = self.data[self._lon_header].to_numpy()
            self._y = self.data[self._lat_header].to_numpy()
    
    
    def clip_loc_with_bound(self, bound=None, bound_fpath=None, buffer=0):
        """remove points outside of boundary (shapely Polygon)"""
        if bound is not None:
            self.bound = bound # update self._bound 
        else:
            if bound_fpath is None:
                return ValueError('Must provide either "bound" (GeoDataFrame) or "bound_fpath"')
            else:
                self.bound = ShapefileData(bound_fpath) # read using ShapefileData class
        # get values
        self._bound_geom_type = self.bound._geom_type
        self.bound = self.bound.data
        # combine boundaries into multipolygons
        if self._bound_geom_type == 'Polygon':
            # convert polygons into MultiPolygons
            bound_polygon = MultiPolygon(self.bound.geometry.to_list())
        elif self._bound_geom_type == 'LineString':
            # polygonize LineStrings
            bound_polygon = Polygon(self.bound.geometry.to_list())
        # get points in boundary
        loc_in_bound = self.clip_data_with_polygon(self.data.copy(),bound_polygon,buffer=buffer)
        # map points to rtree and query
        # sindex = self.data.sindex
        # query = sindex.query_bulk(bound_polygon, predicate='intersects')
        # update data with query results and reset index
        self.data = self.data.loc[loc_in_bound].reset_index(drop=True)
    

    def export_data_to_shp(self, spath=None):
        """exports geopandas.GeoDataFrame to shapefile"""
        if spath is None:
            if self.fpath is None:
                logging.info('Must provide "spath" to export file')
            else:
                self._spath = self.fpath.replace(self.ftype,'.shp') # file path
        else:
            self._spath = spath
        self.data.to_file(self._spath) # export operation
        logging.info(f"Exported GeoDataFrame to:")
        logging.info(f"\t{self._spath}")
        
        
    def export_data_to_csv(self, spath=None):
        """exports geopandas.GeoDataFrame to CSV file"""
        if spath is None:
            if self.fpath is None:
                logging.info('Must provide "spath" to export file')
            else:
                self._spath = self.fpath.replace(self.ftype,'.csv') # file path
        else:
            self._spath = spath
        export_data = self.data.copy()
        export_data.drop(['geometry'], axis=1, inplace=True)
        export_data.to_csv(self._spath, index=False) # export operation
        logging.info(f"Exported GeoDataFrame to:")
        logging.info(f"\t{self._spath}")
    

    def make_component_table(self, attr_to_keep='all'):
        """converts list of geometries to table of individual components (similar to make segments for networks)"""
        # put information into lists
        dict_loc = {
            "LON": [],   "LAT": [],
        }
        dict_meta = {}
        if attr_to_keep == 'all':
            attr_to_keep = self.data.columns.drop('geometry')
        for col in self.data.columns:
            for attr in attr_to_keep:
                if attr.lower() in col.lower():
                    dict_meta[col.upper()] = {'attr_id': col, 'val': []}
        # loop through all objects
        for i in range(self.data.shape[0]):
            # get geometry
            geom = self.data.geometry[i]
            # get segment coordinates
            geom_coord = np.round(geom.coords,7)
            dict_loc['LON'] += geom_coord[:,0].tolist()
            dict_loc['LAT'] += geom_coord[:,1].tolist()
            # get meta data from shapefile
            for key in dict_meta:
                dict_meta[key]['val'] += [self.data[dict_meta[key]['attr_id']][i]]
        # form DataFrame
        df_temp = pd.DataFrame.from_dict(dict_loc)
        self.component_table = gpd.GeoDataFrame.from_dict(
            df_temp,
            geometry=gpd.points_from_xy(df_temp.LON, df_temp.LAT)
        )
        for key in dict_meta:
            self.component_table[key] = dict_meta[key]['val']
        self.component_table['ID'] = np.arange(self.component_table.shape[0])+1
        # reorganize and put geometry last
        self.component_table = self.component_table[['ID']+list(self.component_table.columns.drop('ID').drop('geometry'))+['geometry']]
    
    
    def export_component_table(self, sdir=None, sname=None, to_replace=False):
        """if requested, export DataFrame to CSV file"""
        # if export is True:
        if sdir is None:
            self._component_table_sdir = os.path.join(os.path.dirname(self.fpath)) # directory
        else:
            self._component_table_sdir = sdir
        if sname is None:
            self._component_table_spath_csv = os.path.join(self._component_table_sdir,'components.csv')
            self._component_table_spath_shp = os.path.join(self._component_table_sdir,'components.shp')
        else:
            self._component_table_spath_csv = os.path.join(self._component_table_sdir,sname+'.csv')
            self._component_table_spath_shp = os.path.join(self._component_table_sdir,sname+'.shp')
        if os.path.exists(self._component_table_spath_csv):
            if to_replace:
                # shutil.copyfile(self._component_table_spath_csv,self._component_table_spath_csv.replace('.csv','_OLD.csv'))
                self.component_table.to_csv(self._component_table_spath_csv, index=False) # export operation
        else:
                self.component_table.to_csv(self._component_table_spath_csv, index=False) # export operation
        self.component_table.to_file(self._component_table_spath_shp) # export operation
        logging.info(f"Exported table with component data to:")
        logging.info(f"\t{self._component_table_sdir}")
        logging.info(f"\t\t- {os.path.basename(self._component_table_spath_csv)}")
        logging.info(f"\t\t- {os.path.basename(self._component_table_spath_shp)}")
    
    
    def get_nearest_grid_nodes(self):
        # convert location of components to UTM
        pt_utm_x, pt_utm_y, _, _ = wgs84_to_utm(
            lon=self.component_table.LON.values,
            lat=self.component_table.LAT.values,
            force_zone_num=10
        )
        # convert grid nodes to UTM
        grid_node_utm_x, grid_node_utm_y, _, _ = wgs84_to_utm(
            lon=self.grid_node_table.LON.values,
            lat=self.grid_node_table.LAT.values,
            force_zone_num=10
        )
        # calculate distance between location and every grid point and find minimum distance for each location, keep uniques only
        min_index = np.unique(np.argmin(
            cdist(
                list(zip(pt_utm_x,pt_utm_y)),
                list(zip(grid_node_utm_x,grid_node_utm_y)),
                metric='euclidean'
            ), axis=1))
        # add nearest points to component table
        # self.nearest_node_table = self.grid_node_table[['LON','LAT']].loc[min_index].copy().reset_index(drop=True)
        self.nearest_node_table = self.grid_node_table.loc[min_index].copy().reset_index(drop=True)
        self.nearest_node_table.round(decimal_count(self._grid_spacing))
    
    
    def export_nearest_node_table(self, spath=None):
        """if requested, export DataFrame to CSV file"""
        if spath is None:
            self._nearest_node_table_spath = os.path.join(os.path.dirname(self.fpath),'nearest_grid_nodes.csv') # file path
        else:
            self._nearest_node_table_spath = spath
        self.nearest_node_table.to_csv(self._nearest_node_table_spath, index=False) # export operation
        logging.info(f"Exported table with nearest grid nodes to:")
        logging.info(f"\t{self._nearest_node_table_spath}")
    
    
    def plot(self, show=True, figsize=[16,8], plot_base=True, 
             plot_bound=False, facecolor='none', edgecolor='g', marker='o', markersize=6, 
             attr_to_plot=None, vmin=None, vmax=None, cmap='Pastel2', add_basemap=False, 
             plot_nearest_grid_node=False, plot_grid=True, plot_grid_nodes=False, zoom='auto'):
        """plots geodata"""
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
            # invoke parent function
            # super().plot(show=False)
            super().plot(
                show=False, figsize=figsize, plot_base=plot_base,
                attr_to_plot=attr_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, plot_grid=plot_grid,
                plot_grid_nodes=plot_grid_nodes)
            # plot shapedata
            if plot_base:
                self.data.geometry.plot(
                    ax=self._ax[0,0], facecolor=facecolor, edgecolor=edgecolor, 
                    marker=marker, markersize=markersize
                )
            # plot boundary
            if plot_bound:
                if self.bound is not None:
                    self.bound.geometry.plot(ax=self._ax[0,0], facecolor='none', edgecolor='k', linewidth=3)
            # plot approximate boundary of grid obtained via convex hull
            # if plot_bound:
            #     if self._bound_coord is None:
            #         self.get_bound()
            #     self._ax[0,0].plot(self._bound_coord[:,0], self._bound_coord[:,1], 'k')
            # plot nearest points
            if plot_nearest_grid_node:
                if self.nearest_node_table is not None:
                    self._ax[0,0].scatter(self.nearest_node_table.LON,self.nearest_node_table.LAT,c='orange',s=6)
                else:
                    logging.info(f'Nearest grid nodes have not been obtained; first run "get_nearest_grid_nodes"')
            # add basemap
            if add_basemap:
                self.add_basemap(zoom=zoom)
            # show or not
            if show is True:
                plt.show()


# -----------------------------------------------------------
class NetworkData(ShapefileData):
# class NetworkData(GeoData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    

    # instantiation
    # def __init__(self, fpath, crs='EPSG:4326', rows_to_keep=None):
    #     # invoke parent function
    #     super().__init__(fpath, crs, rows_to_keep)
    
    def __init__(self, fpath=None,
                 shape_type='line', crs="EPSG:4326", rows_to_keep=None,
                 lon=None, lat=None, lon_header='LON', lat_header='LAT',
                 lon_begin=None, lat_begin=None, lon_begin_header='LON_BEGIN', lat_begin_header='LAT_BEGIN',
                 lon_end=None, lat_end=None, lon_end_header='LON_END', lat_end_header='LAT_END'
        ):
        """initialize"""
        if fpath is None:
            if lon is None or lat is None:
                raise ValueError('Please specify either "fpath" or both "lon" and "lat"')
            else:
                fpath = "NONE.from_args"
        # invoke parent function
        super().__init__(fpath=fpath, crs=crs, minimal_init=True)
        
        # compile data into GeoDataFrame
        if self.ftype == '.shp' or self.ftype == '.gdb':
            self.data = ShapefileData(fpath=fpath, crs=crs, rows_to_keep=rows_to_keep).data
        elif self.ftype == '.csv':
            self.data = CSVData(
                fpath=fpath,shape_type=shape_type,
                lon_header=lon_header,lat_header=lat_header,
                lon_begin_header=lon_begin_header,lat_begin_header=lat_begin_header,
                lon_end_header=lon_end_header,lat_end_header=lat_end_header,
            ).data
        elif self.ftype == '.from_args':
            self.data = DataFromArgs(
                shape_type=shape_type,
                lon=lon,lat=lat,
                lon_begin=lon_begin,lat_begin=lat_begin,
                lon_end=lon_end,lat_end=lat_end,
            ).data
        # clean up data: remove rows without geometry, expand multi-objects to individual (e.g., MultiLineString->LineString)
        self._cleanup_data(rows_to_keep)
        # count number of segments in each geometry
        self.data['NUM_SEG'] = [len(geom.coords)-1 for geom in self.data.geometry]
        # get total number of components
        self._num_seg_total = np.sum(self.data.NUM_SEG)
        # other initial processing
        self._geom_type = self.get_geometry_type(self.data)
        
        # initialize other attributes        
        # for individual line segments
        self.segment_table = None
        # self._num_seg_total = None
        # for boundary
        self.bound = None
        self._bound_coord = None
        self._prepped_bound = None
        self.plot_nearest_grid_node = None
        # for exporting
        self._segment_table_spath = None
        self._nearest_node_table_spath = None
        
        
    def split_network_by_max_length(self, l_max=1):
        """split network geometry if exceed max length (in km)"""
        # self._l_max = l_max*1.01 # for roundoff errors
        self._l_max = l_max
        # loop through list of geometries
        for i,geom in enumerate(self.data.geometry):
            # split line into segments given maximum length
            coord_split, flag_performed_split = split_line_by_max_length(np.asarray(geom.coords), self._l_max)
            # if performed, update geometry
            if flag_performed_split:
                self.data.geometry[i] = LineString(coord_split)
        # count number of segments in each geometry
        self.data['NUM_SEG'] = [len(geom.coords)-1 for geom in self.data.geometry]
        # get total number of components
        self._num_seg_total = np.sum(self.data.NUM_SEG)


    def clip_segment_with_bound(self, bound=None, bound_fpath=None, buffer=0):
        """remove segments outside of boundary (shapely Polygon)"""
        if bound is not None:
            self.bound = bound # update self._bound 
        else:
            if bound_fpath is None:
                return ValueError('Must provide either "bound" (GeoDataFrame) or "bound_fpath"')
            else:
                self.bound = ShapefileData(bound_fpath).data # read using ShapefileData class
        # get bound coords and prepare bound
        # self._bound_coord = self.get_coord(self.bound, self.get_geometry_type(self.bound))
        self._prepped_bound = self.prepare_geometry(self.bound, buffer)
        # loop through list of geometries
        for i,geom in enumerate(self.data.geometry):
            flag_remove_segment = False
            # see if bound contains geom
            if self._prepped_bound.contains(geom):
                seg_within_all = geom
            else:
                # if boundary does not contain geom
                flag_remove_segment = True
                seg_within_all = get_segment_within_bound(self._prepped_bound, geom)
            # update if flagged that segments have been removed
            if flag_remove_segment:
                # convert to MultiLineString and update gdf
                if len(seg_within_all) == 0:
                    self.data.geometry[i] = None
                elif len(seg_within_all) == 1:
                    self.data.geometry[i] = seg_within_all[0]
                else:
                    self.data.geometry[i] = MultiLineString(seg_within_all)
        # clean up geometry again after clipping
        self._cleanup_data()
        # count number of segments in each geometry
        self.data['NUM_SEG'] = [len(geom.coords)-1 for geom in self.data.geometry]
        # get total number of components
        self._num_seg_total = int(np.sum(self.data.NUM_SEG))
        if self._num_seg_total == 0:
            raise ValueError("No segments remain after clipping with boundary. Rerun with different boundary.")

    
    def export_processed_data(self, spath=None):
        """exports data to s(ave)path; user-defined or auto-generated"""
        if spath is None:
            self._spath = self.fpath.replace('.shp','_PROCESSED.shp')
        else:
            self._spath = spath
        self.data.to_file(self._spath)
        logging.info(f"Exported preprocessed network shapefile to:")
        logging.info(f"\t{self._spath}")
        
        
    # def make_segment_table(self, attr_to_keep=['diam','owner']):
    def make_segment_table(self, attr_to_keep='all'):
        """converts list of geometries to table of individual segments with midpoints as geometry"""
        # put information into lists
        dict_loc = {
            "LON_BEGIN": [], "LAT_BEGIN": [],
            "LON_END": [],   "LAT_END": [],
            "LON_MID": [],   "LAT_MID": [],
            "LENGTH_KM": []
        }
        dict_meta = {}
        if attr_to_keep == 'all':
            attr_to_keep = list(self.data.columns.drop('geometry'))
            for each in dict_loc:
                if each in attr_to_keep:
                    attr_to_keep.remove(each)
        for col in self.data.columns:
            for attr in attr_to_keep:
                if attr.lower() == col.lower():
                    dict_meta[col.upper()] = {'attr_id': col, 'val': []}
        # initialize list for tracking sub segment ids
        sub_seg_id = []
        # loop through all objects
        for i in range(self.data.shape[0]):
            # get number of segments
            num_seg = self.data.NUM_SEG[i]
            # get pipe id for tracking sub segment number
            obj_id = self.data.obj_id[i]
            # start sub segment id counter
            if i == 0:
                obj_id_prev = obj_id # for tracking previous id
                sub_seg_id.append(np.arange(num_seg)+1)
            else:
                if obj_id_prev == obj_id: # same pipe
                    sub_seg_id.append(np.arange(num_seg)+1 + sub_seg_id[-1][-1])
                else:
                    sub_seg_id.append(np.arange(num_seg)+1)
            # get geometry
            geom = self.data.geometry[i]
            # get segment coordinates
            geom_coord = np.round(geom.coords,10)
            dict_loc['LON_BEGIN'] += geom_coord[:-1,0].tolist()
            dict_loc['LAT_BEGIN'] += geom_coord[:-1,1].tolist()
            dict_loc['LON_END'] += geom_coord[1:,0].tolist()
            dict_loc['LAT_END'] += geom_coord[1:,1].tolist()
            dict_loc['LON_MID'] += np.round((geom_coord[:-1,0] + geom_coord[1:,0])/2,10).tolist()
            dict_loc['LAT_MID'] += np.round((geom_coord[:-1,1] + geom_coord[1:,1])/2,10).tolist()
            dict_loc['LENGTH_KM'] += np.round(get_haversine_dist(
                lon1=geom_coord[:-1,0], lat1=geom_coord[:-1,1],
                lon2=geom_coord[1:,0], lat2=geom_coord[1:,1],
                unit='km'
            ),6).tolist()
            # get meta data from shapefile
            for key in dict_meta:
                dict_meta[key]['val'] += [self.data[dict_meta[key]['attr_id']][i]]*num_seg
            # for tracking previous id
            obj_id_prev = obj_id
        # form DataFrame
        # self.segment_table = pd.DataFrame.from_dict(dict_loc)
        df_temp = pd.DataFrame.from_dict(dict_loc)
        self.segment_table = gpd.GeoDataFrame.from_dict(
            df_temp,
            geometry=gpd.points_from_xy(df_temp.LON_MID, df_temp.LAT_MID)
        )
        for key in dict_meta:
            self.segment_table[key] = dict_meta[key]['val']
        # self.segment_table = pd.concat([
        #     pd.DataFrame(np.arange(self.segment_table.shape[0])+1,columns=['ID']),
        #     self.segment_table
        # ],axis=1)
        self.segment_table['ID'] = np.arange(self.segment_table.shape[0])+1
        self.segment_table = self.segment_table[['ID']+list(self.segment_table.columns.drop('ID').drop('geometry'))+['geometry']]
        # add sub segment ID
        self.segment_table['SUB_SEGMENT_ID'] = np.hstack(sub_seg_id)
        self.segment_table.drop('NUM_SEG',axis=1,inplace=True)
        # reorganize and put geometry last
        self.segment_table = self.segment_table[list(self.segment_table.columns.drop('geometry'))+['geometry']]
        # drop segments with zero length
        self.segment_table = self.segment_table[self.segment_table['LENGTH_KM']>=0].reset_index(drop=True)
        
    
    def get_nearest_grid_nodes(self):
        # convert midpoint of segments to UTM
        mid_pt_utm_x, mid_pt_utm_y, _, _ = wgs84_to_utm(
            lon=self.segment_table.LON_MID.values,
            lat=self.segment_table.LAT_MID.values,
            force_zone_num=10
        )
        # convert grid nodes to UTM
        grid_node_utm_x, grid_node_utm_y, _, _ = wgs84_to_utm(
            lon=self.grid_node_table.LON.values,
            lat=self.grid_node_table.LAT.values,
            force_zone_num=10
        )
        # calculate distance between midpoint and every grid point and find minimum distance for each midpoint, keep uniques only
        min_index = np.unique(np.argmin(
            cdist(
                list(zip(mid_pt_utm_x,mid_pt_utm_y)),
                list(zip(grid_node_utm_x,grid_node_utm_y)),
                metric='euclidean'
            ), axis=1))
        # add nearest points to segment table
        # self.nearest_node_table = self.grid_node_table[['LON','LAT']].loc[min_index].copy().reset_index(drop=True)
        self.nearest_node_table = self.grid_node_table.loc[min_index].copy().reset_index(drop=True)
        self.nearest_node_table.round(decimal_count(self._grid_spacing))
    
    
    def export_segment_table(self, sdir=None, sname=None, to_replace=False):
        """if requested, export DataFrame to CSV file"""
        # if export is True:
        # self._segment_table_spath = os.path.join(os.path.dirname(self.fpath),'site_data.csv') # file path
        if sdir is None:
            self._segment_table_sdir = os.path.join(os.path.dirname(self.fpath)) # directory
        else:
            self._segment_table_sdir = sdir
        if sname is None:
            self._segment_table_spath_csv = os.path.join(self._segment_table_sdir,'segments.csv')
            self._segment_table_spath_shp = os.path.join(self._segment_table_sdir,'segments.shp')
        else:
            self._segment_table_spath_csv = os.path.join(self._segment_table_sdir,sname+'.csv')
            self._segment_table_spath_shp = os.path.join(self._segment_table_sdir,sname+'.shp')
        if os.path.exists(self._segment_table_spath_csv):
            if to_replace:
                # shutil.copyfile(self._segment_table_spath_csv,self._segment_table_spath_csv.replace('.csv','_OLD.csv'))
                self.segment_table.to_csv(self._segment_table_spath_csv, index=False) # export operation
        else:
            self.segment_table.to_csv(self._segment_table_spath_csv, index=False) # export operation
        self.segment_table.to_file(self._segment_table_spath_shp) # export operation
        logging.info(f"Exported table with network segment data to:")
        logging.info(f"\t{self._segment_table_sdir}")
        logging.info(f"\t\t- {os.path.basename(self._segment_table_spath_csv)}")
        logging.info(f"\t\t- {os.path.basename(self._segment_table_spath_shp)}")
    
    
    def export_nearest_node_table(self, spath=None):
        """if requested, export DataFrame to CSV file"""
        if spath is None:
            self._nearest_node_table_spath = os.path.join(os.path.dirname(self.fpath),'nearest_grid_nodes.csv') # file path
        else:
            self._nearest_node_table_spath = spath
        self.nearest_node_table.to_csv(self._nearest_node_table_spath, index=False) # export operation
        logging.info(f"Exported table with nearest grid nodes to:")
        logging.info(f"\t{self._nearest_node_table_spath}")
    
    
    def plot(self, show=True, figsize=[16,8], plot_seg_midpt=False, plot_base=True,
             plot_bound=False, facecolor='none', edgecolor='k', linewidth=2,
             attr_to_plot=None, vmin=None, vmax=None, cmap='Pastel2', add_basemap=False,
             plot_nearest_grid_node=False, plot_grid=True, plot_grid_nodes=False, zoom='auto'):
        """plots geodata"""
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
            # invoke parent function
            super().plot(
                show=False, figsize=figsize, plot_base=plot_base,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                attr_to_plot=attr_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, plot_grid=plot_grid,
                plot_grid_nodes=plot_grid_nodes)
            # plot mid point of segments, if generated
            if plot_seg_midpt:
                if self.segment_table is not None:
                    self._ax[0,0].scatter(self.segment_table.LON_MID,self.segment_table.LAT_MID,c='red',s=6)
            # plot boundary
            if plot_bound:
                if self.bound is not None:
                    self.bound.geometry.plot(ax=self._ax[0,0], facecolor='none', edgecolor='k', linewidth=3)
            # plot nearest points
            if plot_nearest_grid_node:
                if self.nearest_node_table is not None:
                    self._ax[0,0].scatter(self.nearest_node_table.LON,self.nearest_node_table.LAT,c='orange',s=6)
                else:
                    logging.info(f'Nearest grid nodes have not been obtained; first run "get_nearest_grid_nodes"')
            # add basemap
            if add_basemap:
                self.add_basemap(zoom=zoom)
            # show or not
            if show is True:
                plt.show()


# -----------------------------------------------------------
class RasterData(GeoData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    

    # instantiation
    def __init__(self, fpath):
        # invoke parent function
        super().__init__(fpath)
        
        # read data
        self._read_data()
        
        # initialize other attributes
        # for sampling
        self.sample = None
        self.x_sample = None
        self.y_sample = None
        self.n_sample = None


    def _read_data(self):
        """reads data"""
        if os.path.exists(self.fpath):
            self.data = rio.open(self.fpath)
        else:
            raise ValueError(f'Raster file "{os.path.basename(self.fpath)}" does not exist')
    
    
    def close(self):
        """closes file"""
        self.data.close()
        
    
    def get_sample(self, x, y, band=1, interp_scheme='nearest', out_of_bound_value=np.nan, invalid_value=np.nan):
        """performs 2D interpolation at (x,y) pairs. Accepted interp_scheme = 'nearest', 'linear', 'cubic', and 'quintic'"""
        self.x_sample = x
        self.y_sample = y
        n_sample = len(x)
        if interp_scheme == 'nearest':
            self.sample = np.array([val[0] for val in self.data.sample(list(zip(self.x_sample,self.y_sample)))])
        else:
            # create x and y ticks for grid
            x_tick = np.linspace(self.data.bounds.left,   self.data.bounds.right, self.data.width,  endpoint=False)
            y_tick = np.linspace(self.data.bounds.bottom, self.data.bounds.top,   self.data.height, endpoint=False)
            # create interp2d function
            interp_function = interp2d(
                x_tick, y_tick, np.flipud(self.data.read(band)),
                kind=interp_scheme, fill_value=out_of_bound_value)
            # get samples
            self.sample = np.transpose([interp_function(self.x_sample[i],self.y_sample[i]) for i in range(n_sample)])[0]
        # clean up invalid values (returned as 1e38 by NumPy)
        self.sample[abs(self.sample)>1e10] = invalid_value
    
    
    def plot(self, show=True, figsize=[16,8], band=1, plot_base=True, cmap='Greens', show_colorbar=True, plot_sample=False, add_basemap=False, zoom='auto'):
        """plots geodata"""
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
            # invoke parent function
            super().plot(show=False, figsize=figsize)
            # plot raster image
            if plot_base:
                plot_data = self.data.read(band)
                bound = self.data.bounds
                extent = [bound.left, bound.right, bound.bottom, bound.top]
                vmin = np.amin(plot_data)
                vmax = np.amax(plot_data)
                ras_handle = self._ax[0,0].imshow(plot_data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
                if show_colorbar:
                    self._fig.colorbar(ras_handle, ax=self._ax[0,0], location='right', anchor=(0,0), shrink=1)
            # plot samples
            if plot_sample:
                plot_sample_data = self.samples
                self._ax[0,0].scatter(
                    x, y, c=self.sample, s=6,
                    alpha=1, edgecolors='k', cmap=cmap, vmin=vmin, vmax=vmax
                )
            # add basemap
            if add_basemap:
                self.add_basemap(zoom=zoom)
            # show or not
            if show is True:
                plt.show()
            
            
# -----------------------------------------------------------
class JSONData(GeoData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    

    # instantiation
    def __init__(self, fpath, lon_header=None, lat_header=None):
        # invoke parent function
        super().__init__(fpath)


    
    def _read_data(self):
        """reads data"""
        if os.path.exists(self.fpath):
            self.data = rio.open(self.fpath)
        else:
            raise valueError("XML file does not exist")
        self.data = pd.read_csv(self.fpath)
        
        
# -----------------------------------------------------------
class XMLData(GeoData):
    """[summary]

    Args:
        object ([type]): [description]
    """
    
    
    # instantiation
    def __init__(self, fdir):
        # invoke parent function
        super().__init__(fpath)
        
    
    def _read_data(self):
        """reads data"""
        pass

    
    
    
                
                
# -----------------------------------------------------------
class test(GeoData):
    def __init__(self, fpath,lon_header='LON_MID',lat_header='LON_MID'):
        if fpath.endswith('csv'):
            self.__class__ = copy.deepcopy(LocationData(fpath,lon_header=lon_header,lat_header=lat_header).__class__)
            self.__dict__ = copy.deepcopy(LocationData(fpath,lon_header=lon_header,lat_header=lat_header).__dict__)
        elif fpath.endswith('shp'):
            self.__class__ = copy.deepcopy(ShapefileData(fpath).__class__)
            self.__dict__ = copy.deepcopy(ShapefileData(fpath).__dict__)
            
            
            
# -----------------------------------------------------------
