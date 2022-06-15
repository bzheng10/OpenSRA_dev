## -----------------------------------------------------------
## Open-Source Seismic Risk Assessment, OpenSRA(TM)
##
## Copyright(c) 2020-2022 The Regents of the University of California and
## Slate Geotechnical Consultants. All Rights Reserved.
##
## Base classes used in OpenSRA
##
## Created: April 1, 2022
## @author: Barry Zheng (Slate Geotechnical Consultants)
## -----------------------------------------------------------
#
#
## Python base modules
#import os
#import logging
#import json
#import sys
#import importlib
#import time
#import pathlib
#import warnings
## import itertools
#
## data manipulation modules
#import numpy as np
#import pandas as pd
#from pandas import DataFrame
#from pandas.api.types import is_numeric_dtype
#from scipy.interpolate import interp2d, griddata
## from scipy import sparse
#
## geospatial processing modules
#import geopandas as gpd
#from geopandas import GeoDataFrame, GeoSeries
#from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon
#from shapely.ops import linemerge, polygonize
#from shapely.prepared import prep
#import rasterio as rio
#import rasterio.features
#import rasterio.warp
#from rasterio.plot import show, adjust_band
## from shapely.strtree import STRtree
## from rtree import index
#
## efficient processing modules
#from numba import jit, njit
#
## plotting modules
## if importlib.util.find_spec('contextily') is not None:
#    # import contextily as ctx
## if importlib.util.find_spec('matplotlib') is not None:
#    # import matplotlib.pyplot as plt
#    # from matplotlib.collections import LineCollection
#
## java modules
## import jpype
## from jpype import imports
## from jpype.types import *
#
## OpenSRA modules
#from src.util import check_common_member, get_closest_pt, get_haversine_dist
#from src.im import opensha as sha