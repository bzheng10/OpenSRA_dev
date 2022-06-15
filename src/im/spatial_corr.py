# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for CPT penetration correction
#
# Created: April 13, 2020
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python Modules
import numpy as np
from numba import jit

# OpenSRA modules and classes
from src.base_class import BaseClass


# -----------------------------------------------------------
class SpatialCorrelation(BaseClass):
    "Inherited class specfic to spatial correlations"

    TYPE = 'Spatial'

    def __init__(self):
        super().__init__()


# -----------------------------------------------------------
class JayaramBaker2009(SpatialCorrelation):
    """
    Compute correlations between spatial ordinates using Jayaram & Baker (2009).
    
    Parameters
    ----------
    d : float, array
        [km] distance
    T : float
        [sec] period
    geo_cond : int, optional
        geologic condition: **1** for variability within soil, **2** for homogenous conditions; default = 2
        
    Returns
    -------
    corr : float, array
        correlation for two sites at a distance of **d**
        
    References
    ----------
    .. [1] Jayaram, N., and Baker, J.W., 2009, Correlation Model for Spatially Distributed Ground‐Motion Intensities, Earthquake Engineering and Structural Dynamics, vol. 38, no. 15, pp. 1687-1708.
    
    """

    NAME = 'Jayaram and Baker (2009)'
    ABBREV = 'JB09'
    REF = "".join([
        'Jayaram, N., and Baker, J.W., 2009, ',
        'Correlation Model for Spatially Distributed Ground‐Motion Intensities, ',
        'Earthquake Engineering and Structural Dynamics, ',
        'vol. 38, no. 15, pp. 1687-1708.'
    ])
    LEVEL = 1
    INPUT = {
        'REQUIRED': {
            'd': {
                'DESC': 'distance [km]'
            },
            'T': {
                'DESC': 'period [sec]'
            }
        },
        'OPTIONAL': {
            'geo_cond': {
                'DESC': 'geologic condition: 1 for variability within soil; 2 for homogenous conditions',
                'DEFAULT': 2
            }
        }
    }
    OUTPUT = {
        'corr': {
            'DESC': 'Correlation values'
        }
    }


    # instantiation
    def __init__(self):
        super().__init__()


    # update calculation method
    def _perform_calc(self):
        """Performs calculations"""
        # pull inputs locally
        d = self._inputs['d']
        T = self._inputs['T']
        geo_cond = self._inputs['geo_cond']

        # calculations
        corr = self._model(d, T, geo_cond)
        
        # store intermediate params
        self._inters.update({
            'corr': corr
        })


    @staticmethod
    @jit(nopython=True)
    def _model(d, T, geo_cond):
        """Model"""
        # additional model parameters
        
        # setup

        # calculations
        if T < 1:
            if geo_cond == 1:
                b = 8.5 + 17.2*T
            elif geo_cond == 2:
                b = 40.7 - 15.0*T
            else:
                raise ValueError(f"geo_cond must be equal 1 or 2 (see documentation).")
        elif T >= 1:
            b = 22.0 + 3.7*T

        # return
        return np.exp(-3*d/b)


# -----------------------------------------------------------
@jit(nopython=True)
def jb09(d, T, geo_cond=2):
    """
    Compute correlations between spatial ordinates using Jayaram & Baker (2009).
    
    Parameters
    ----------
    d : float, array
        [km] distance
    T : float
        [sec] period
    geo_cond : int, optional
        geologic condition: **1** for variability within soil, **2** for homogenous conditions; default = 2
        
    Returns
    -------
    corr : float, array
        correlation for two sites at a distance of **d**
        
    References
    ----------
    .. [1] Jayaram, N., and Baker, J.W., 2009, Correlation Model for Spatially Distributed Ground‐Motion Intensities, Earthquake Engineering and Structural Dynamics, vol. 38, no. 15, pp. 1687-1708.
    
    """
    
    # calculations
    if T < 1:
        if geo_cond == 1:
            b = 8.5 + 17.2*T
        if geo_cond == 2:
            b = 40.7 - 15.0*T
    elif T >= 1:
        b = 22.0 + 3.7*T
    
    #
    return np.exp(-3*d/b)