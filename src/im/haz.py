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
import sys
import importlib
# import inspect

# data manipulation modules
import numpy as np
# from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas import DataFrame, read_csv
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy import sparse

# geospatial processing modules
from geopandas import GeoDataFrame
from shapely.geometry import LineString

# efficient processing modules
# import numba as nb
# from numba import njit

# plotting modules
# if importlib.util.find_spec('matplotlib') is not None:
#     import matplotlib.pyplot as plt
#     from matplotlib.collections import LineCollection
# if importlib.util.find_spec('contextily') is not None:
#     import contextily as ctx

# OpenSRA modules
from src.util import from_dict_to_array
from src.im import gmc, ssc


# -----------------------------------------------------------
class SeismicHazard(object):
    """
    Class for running seismic hazard
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    
    # instantiation
    def __init__(self):
        """Create an instance of the class"""

        # get inputs
        
        # additional setup
        
        # initialize empty params
        self.ssc_name = None
        self.site_data = None
        self.source = None
        self.gmpe = None
        self.gm_pred = None
        self._n_event = 0
        self._n_site = 0
        self.event_id = None
        self.rate = None
        self.mag = None
        # for plotting
        self._fig = None
        self._ax = None
        self._fignum = None
        # for exporting
        self._fig_spath = None


    def set_site_data(self,
        lon,
        lat,
        vs30,
        z1p0=None,
        z2p5=None,
        vs30_source=None,
        site_id=None
    ):
        """set lat, lon, vs30, vs30_source, z1p0, z2p5"""
        # store into back-end
        self._n_site = len(lon)
        self.site_data = {
            'lon': lon,
            'lat': lat,
            'vs30': vs30,
        }
        if z1p0 is not None:
            self.site_data['z1p0'] = z1p0
        if z2p5 is not None:
            self.site_data['z2p5'] = z2p5
        if vs30_source is not None:
            self.site_data['vs30_source'] = vs30_source
        # create site_id if None
        if site_id is None:
            self.site_data['site_id'] = np.arange(self._n_site)
        else:
            self.site_data['site_id'] = site_id
        # convert vs30 to numerics
        if 'vs30_source' in self.site_data:
            if "Inferred" in self.site_data['vs30_source'] or "Measured" in self.site_data['vs30_source']:
                _vs30_source = np.ones(self._n_site)
                _vs30_source[self.site_data['vs30_source']=='Inferred'] = 0
                self.site_data['vs30_source'] = _vs30_source
        else:
            self.site_data['vs30_source'] = np.zeros(self._n_site)
        logging.info(f"Finished setting sites for seismic hazard")
    
    
    def init_ssc(self, ssc_name=None, ucerf_model_name=None, 
                 sm_dir=None, opensra_dir=None, event_names=None, 
                 im_list=['pga','pgv'], user_def_rup_fpath=None):
        """
        Initializes seismic source
        If ssc_name is not specified, default to 'Mean UCERF3 FM3.1'
        For full list of GMPEs, see src.im.gmc
        """
        self.ssc_name = ssc_name
        if ssc_name is None:
            logging.info(f'Initialized "UCERF3": "Mean UCERF3 FM3.1"')
            self.source = getattr(ssc, "UCERF")('Mean UCERF3 FM3.1')
            # self.event_id = self.source.df_rupture.SourceId.values
        else:
            ##########################################
            # for full inventory
            if ssc_name == 'UCERF':
                if ucerf_model_name is None:
                    ucerf_model_name = 'Mean UCERF3 FM3.1'
                logging.info(f'Initialized "UCERF3": "{ucerf_model_name}"')
                self.source = getattr(ssc, "UCERF")(opensra_dir, ucerf_model_name)
                # self.event_id = self.source.df_rupture.SourceId.values # for full inventory
            ##########################################
            elif ssc_name == 'ShakeMap' or 'UserDefined' in ssc_name:
                logging.info(f'Initialized "{ssc_name}" for SSC')
                if ssc_name == 'ShakeMap':
                    # self.source = getattr(ssc, ssc_name)(sm_dir=sm_dir, event_names=event_names, im_list=im_list)
                    self.source = getattr(ssc, ssc_name)(sm_dir=sm_dir, event_names=event_names, im_list=im_list)
                elif ssc_name == 'UserDefinedRupture':
                    self.source = getattr(ssc, ssc_name)(fpath=user_def_rup_fpath)
                elif ssc_name == 'UserDefinedGM':
                    raise NotImplementedError("UserDefinedGM to be implemented")
                else:
                    raise NotImplementedError("Available options for SSC are UCERF3, ShakeMap, UserDefinedRupture, and UserDefinedGM")
            else:
                raise NotImplementedError("Available options for SSC are UCERF3, ShakeMap, UserDefinedRupture, and UserDefinedGM")
        # update params
        self._n_event = self.source._n_event
        self.rate = self.source.df_rupture.AnnualRate.values
        self.mag = self.source.df_rupture.Magnitude.values
    
    
    def process_rupture(self, 
        max_dist=200, # km, for UCERF and user-defined faults, not ShakeMap
        mag_min=None, # all sources
        mag_max=None, # all sources
        rate_min=None, # all sources
        rate_max=None, # all sources
        event_ids_to_keep=None, # all sources
        mesh_spacing=1, # km
    ):
        """performs filter on rupture scenarios"""
        if self.ssc_name == 'UCERF' or self.ssc_name == 'UserDefinedRupture':
            self.source.process_rupture(
                lon=self.site_data['lon'],
                lat=self.site_data['lat'],
                max_dist=max_dist, # km,
                mag_min=mag_min,
                mag_max=mag_max,
                rate_min=rate_min,
                rate_max=rate_max,
                event_ids_to_keep=event_ids_to_keep,
                mesh_spacing=mesh_spacing, # km
            )
            # update params
            self._n_event = self.source._n_event
            ###############################
            # for full inventory
            # if self.ssc_name == 'UCERF':
                # self.event_id = self.source.df_rupture.SourceId.values
            ###############################
            # elif self.ssc_name == 'UserDefinedRupture':
            #     self.event_id = self.source.df_rupture.EventID.values
            self.event_id = self.source.df_rupture.EventID.values
            self.rate = self.source.df_rupture.AnnualRate.values
            self.mag = self.source.df_rupture.Magnitude.values
        elif self.ssc_name == 'ShakeMap':
            self.source.process_rupture(
                mag_min=mag_min,
                mag_max=mag_max,
                event_ids_to_keep=event_ids_to_keep,
            )
            # update params
            self._n_event = self.source._n_event
        else:
            raise NotImplementedError("Under development.")
    
    
    def init_gmpe(self, gmpe_names=None, weights=None):
        """
        Initializes GMPEs
        If inputs are not specified, defaults to ASK14, BSSA14, CB14, CY14, equally weighted
        For full list of GMPEs, see src.im.gmc
        """
        if self.ssc_name == 'ShakeMap':
            # do not initiate if using ShakeMaps
            logging.info("Using IMs from ShakeMaps, no need to initialize GMPEs")
        else:
            # pull generic class
            self.gmpe = getattr(gmc, "WeightedGMPE")()
            # first initialize GMPEs
            if gmpe_names is None:
                self.gmpe.set_gmpes_and_weights()
            else:
                if weights is None:
                    self.gmpe.set_gmpes_and_weights(gmpe_names)
                else:
                    self.gmpe.set_gmpes_and_weights(gmpe_names, weights)
            logging.info(f"Initialized GMPEs:")
            for each in self.gmpe.instance:
                logging.info(f"\t- {each}: weight={self.gmpe.instance[each]['weight']}")
            
    
    def load_gm_pred_from_csv(self, fpath):
        """loads ground motion predictions from CSV files"""
        pass
        
        
    def load_gm_pred_from_sparse(self, fdir):
        """loads ground motion predictions from sparse matrix files exported from OpenSRA"""
        # get files in fdir
        files = os.listdir(fdir)
        # determine IMs available in folder
        im_list = []
        for f in files:
            if f.endswith('.npz'):
                im = f[:f.find('_')]
                if not im in im_list:
                    im_list.append(im)
        # initialize dictionary
        self.gm_pred = {}
        for im in im_list:
            self.gm_pred[im] = {}
            # read parameters: mean, sigma, sigma_mu
            if im+'_mean.npz' in files:
                data = sparse.load_npz(os.path.join(fdir,im+'_mean.npz'))
                self.gm_pred[im]['mean'] = data.toarray()-10
            if im+'_sigma.npz' in files:
                self.gm_pred[im]['sigma'] = sparse.load_npz(os.path.join(fdir,im+'_sigma.npz')).toarray()
            if im+'_sigma_mu.npz' in files:
                self.gm_pred[im]['sigma_mu'] = sparse.load_npz(os.path.join(fdir,im+'_sigma_mu.npz')).toarray()
        # determine and store other metrics
        self.im_list = im_list
        self._n_event = self.gm_pred[im_list[0]]['mean'].shape[0]
        self._n_site = self.gm_pred[im_list[0]]['mean'].shape[1]
        # import other data
        self.site_data = read_csv(os.path.join(fdir,'site_data.csv')).to_dict()
        rup_meta = read_csv(os.path.join(fdir,'rupture_metadata.csv'))
        self.rate = rup_meta['annual_rate'].values
        self.mag = rup_meta['magnitude'].values
        self.event_id = rup_meta['event_id'].values.astype(int)
        

    def get_gm_pred_from_gmc(self, im_list=['pga','pgv'], njit_on=False, n_events_print=1e9):
        """calculates/get ground motion predictions"""
        if self._n_site == 0:
            return f"No sites specified - run self.set_site_data(args)"
        elif self._n_event == 0:
            return f"No scenarios to run - pick from {self.supported_source_types}"
        else:
            if self.ssc_name == "ShakeMap":
                # calculate IMs using GMC
                logging.info(f"Getting ground motions from ShakeMap grids:")
                logging.info(f"\t- Number of events: {self._n_event}")
                logging.info(f"\t- Number of sites: {self._n_site}")
                # setup
                
                shape = (self._n_event, self._n_site)
                self.gm_pred = {}
                im_list = [im.lower() for im in im_list] # convert to lower case
                self.im_list = im_list
                for im in im_list:
                    self.gm_pred[im] = {
                            'mean': np.ones(shape) * -10, # default gm value outside of max dist
                            'sigma': np.zeros(shape),
                            'sigma_mu': np.zeros(shape)
                        }
                logging.info(f"\t- Periods to get: {', '.join(self.im_list)}")
                # loop through number of events
                for i in range(self._n_event):
                # for i, rup_ind in self.source.rupture_in_maxdist:
                    # perform sampling of IMs from ShakeMap
                    site_gm, site_aleatory, site_epistemic = \
                        self.source._sample_gm_from_map_i(i, self.site_data['lon'], self.site_data['lat'])
                    # store outputs
                    for j, im in enumerate(im_list):
                        self.gm_pred[im]['mean'][i] = site_gm[:,j]
                        self.gm_pred[im]['sigma'][i] = site_aleatory[:,j]
                        self.gm_pred[im]['sigma_mu'][i] = site_epistemic[:,j]
                    # print message to track number of events already ran
                    if (i+1) % n_events_print == 0:
                        logging.info(f"\t\t- finished sampling from {self.source.event_names[i]}...")
                logging.info(f">>>>>>>>>>> Finished sampling ground motions from ShakeMaps")
            else:
                # calculate IMs using GMC
                logging.info(f"Calculating ground motion predictions:")
                logging.info(f"\t- Number of events: {self._n_event}")
                logging.info(f"\t- Number of sites: {self._n_site}")
                # check if GMPEs are initialized
                if self.gmpe is None:
                    self.init_gmpe() # use default, which is NGAWest2
                # setup
                # shape = (self._n_event, self._n_site)
                shape = (self._n_event, self._n_site)
                # print(self._n_event, shape)
                self.gm_pred = {}
                im_list = [im.lower() for im in im_list] # convert to lower case
                self.im_list = im_list
                for im in im_list:
                    self.gm_pred[im] = {
                            'mean': np.ones(shape) * -10, # default gm value outside of max dist
                            'sigma': np.zeros(shape),
                            'sigma_mu': np.zeros(shape)
                        }
                logging.info(f"\t- Periods to calculate: {', '.join(self.im_list)}")
                # loop through number of events
                for i in range(self._n_event):
                # for i, rup_ind in enumerate(self.source.rupture_in_maxdist):
                    # each source type has its own version of generating inputs for running GMPE
                    # see if current event contains sites within r_max 
                    # if i in self.source.sites_for_rupture_in_maxdist:
                    # get inputs
                    kwargs = self.source._get_gmpe_input_for_event_i(i, im_list, self.site_data)
                    # run mean model
                    self.gmpe.run_model(kwargs, njit_on=njit_on)
                    # store outputs
                    # try:
                    for j, im in enumerate(im_list):
                        self.gm_pred[im]['mean'][i,kwargs['site_id']] = self.gmpe.model_dist['mean'][j,:]
                        self.gm_pred[im]['sigma'][i,kwargs['site_id']] = self.gmpe.model_dist['aleatory']['sigma'][j,:]
                        self.gm_pred[im]['sigma_mu'][i,kwargs['site_id']] = self.gmpe.model_dist['epistemic'][j,:]
                    # except:
                    #     print(i)
                    # else:
                    #     # store outputs
                    #     for j, im in enumerate(im_list):
                    #         self.gm_pred[im]['mean'][i,:] = -10
                    #         self.gm_pred[im]['sigma'][i,:] = 0.001
                    #         self.gm_pred[im]['sigma_mu'][i,:] = 0.0
                        
                    # print message to track number of events already ran
                    # if (i+1) % n_events_print == 0:
                    #     logging.info(f"\t\t- finished {i+1} events...")
                logging.info(f">>>>>>>>>>> Finished calculating ground motion predictions")
        
        
    # def export_gm_pred(self, sdir=None, stype=['sparse','csv'], addl_rup_meta=None):
    # def export_gm_pred(self, sdir=None, stype=['sparse','csv']):
    def export_gm_pred(self, sdir=None, stype=['sparse']):
        """exports calculated ground motion predictions"""
        name_map = {
            'mean': 'MEAN',
            'sigma': 'ALEATORY',
            'sigma_mu': 'EPISTEMIC'
        }
        # only run if self.gm_pred is not None
        if self.gm_pred is None:
            return "Nothing to export - first run self.get_gm_pred to get predictions"
        else:
            # check output path
            if sdir is None:
                return "Must provide sdir (save directory)"
            else:
                # export
                logging.info(f"Under: {sdir}")
                # make sdir if not created
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                # loop through IMs
                for im in self.im_list:
                    logging.info(f"\t- under {os.path.basename(sdir)}\{im.upper()}:")
                    if not os.path.isdir(os.path.join(sdir,im.upper())):
                        os.mkdir(os.path.join(sdir,im.upper()))
                    for item in self.gm_pred[im]:
                        # create save name for sparse matrix
                        # save_name = r'C:\Users\barry\OneDrive\Desktop\CEC\temp\test\im\TEMP.npz'
                        save_name = os.path.join(sdir,im.upper(),f'{name_map[item]}.npz')
                        # save_name = os.path.join(sdir,'TEMP.npz')
                        # save_name = save_name.replace('TEMP',f'{im}_{item}')
                        if 'sparse' in stype:
                            # convert to sparse matrix
                            if item == 'mean':
                                coo_out = sparse.coo_matrix(np.round(self.gm_pred[im][item]+10,decimals=3))
                            else:
                                coo_out = sparse.coo_matrix(np.round(self.gm_pred[im][item],decimals=3))
                            # export data
                            sparse.save_npz(save_name, coo_out)
                            logging.info(f"\t\t- {os.path.basename(save_name)}")
                        if 'csv' in stype:
                            save_name = save_name.replace('npz','csv')
                            df_out = pd.DataFrame.from_dict(self.gm_pred[im][item])
                            df_out.to_csv(save_name, index=False, header=False)
                            # np.savetxt(save_name, np.round(self.gm_pred[im][item],decimals=3), fmt='%.3f', delimiter=r'\t')
                            logging.info(f"\t\t- {os.path.basename(save_name)}")
            # export other information
            if self.ssc_name == 'UCERF':
                self._export_site_data(sdir)
            # self._export_rupture_metadata(sdir, addl_rup_meta=addl_rup_meta)
            self._export_rupture_metadata(sdir)
    
    
    # def _export_rupture_metadata(self, sdir=None, export_rup_geom=True, addl_rup_meta=None):
    def _export_rupture_metadata(self, sdir=None, export_rup_geom=True):
        """exports rupture scenario metadata (mean annual rate and magnitude)"""
        if self.mag is None:
            return "Nothing to export - first run self.init_ssc (and self.filter_ucerf_rupture) to get ruptures"
        else:
            mag = self.mag
            if self.rate is None:
                rate = np.ones(self._n_event)
            else:
                rate = self.rate
            if self.event_id is None:
                event_id = np.arange(self._n_event)+1
            else:
                event_id = self.event_id
            event_id = np.asarray(event_id).astype(int)
            # make DataFrame
            rup_meta_out = DataFrame(
                np.vstack([event_id,mag,rate]).T,
                columns=['event_id','magnitude','annual_rate']
            )
            # also exports rupture geometry
            if export_rup_geom:
                name_map = {
                    'dip': 'Dip',
                    'rake': 'Rake',
                    'dip_dir': 'DipDir',
                    'z_tor': 'UpperDepth',
                    'z_bor': 'LowerDepth',
                    'fault_trace': 'FaultTrace'
                }
                for each in name_map:
                    # if self.ssc_name == 'UCERF':
                    #     # first look for it in df_rupture
                    #     if name_map[each] in self.source.df_rupture:
                    #         rup_meta_out[each] = self.source.df_rupture[name_map[each]].values
                    #     # if not in df_rupture, then it will be under df_section
                    #     else:
                    #         rup_meta_out[each] = self.source.df_section[name_map[each]].values
                    # else:
                    rup_meta_out[each] = self.source.df_rupture[name_map[each]].values
            # export to csv
            save_name_csv = os.path.join(sdir,'RUPTURE_METADATA.csv')
            rup_meta_out.to_csv(save_name_csv,index=False)
            # export to shp
            save_name_shp = os.path.join(sdir,'RUPTURE_METADATA.gpkg')
            geoms = []
            for i in range(rup_meta_out.shape[0]):
                # trace = np.asarray(json.loads(rup_meta.fault_trace.iloc[i]))
                trace = np.asarray(rup_meta_out.fault_trace.iloc[i])
                geoms.append(LineString(trace[:,:2]))
            rup_meta_out_gdf = GeoDataFrame(
                pd.read_csv(save_name_csv), # reread dataframe to convert fields of lists into strings
                crs=4326, geometry=geoms
                # rup_meta_out, crs=4326, geometry=geoms
            )
            rup_meta_out_gdf.to_file(save_name_shp,index=False,layer='data')
            logging.info(f"Exported rupture metadata to:")
            logging.info(f"\t- {save_name_csv}")
            logging.info(f"\t- {save_name_shp}")
            

    def _export_site_data(self, sdir=None):
        """exports site data (lat, lon, z1p0, z2p5, vs30, vs30_source)"""
        if self.site_data is None:
            return "Nothing to export - first run self.set_site_data"
        else:
            # reorder keys
            keys = list(self.site_data)
            ind_site_id = keys.index('site_id')
            keys_reorder = ['site_id']
            for key in keys:
                if not key == 'site_id':
                    keys_reorder.append(key)
            # make DataFrame and export to CSV
            site_data_out = DataFrame.from_dict(self.site_data)
            site_data_out = site_data_out[keys_reorder]
            site_data_out.to_csv(
                os.path.join(sdir,'site_data.csv'),
                index=False
            )
            logging.info(f"Exported updated site data to:")
            logging.info(f"\t- {os.path.join(sdir,'site_data.csv')}")
            
    
    def get_haz_curve(self, site_num, im, x_vals=None, fractiles=[16,50,84], n_epi_sample=1000, to_plot=False, to_export=False, export_spath=None):
        """gets hazard curves for site for target IM"""
        # checks
        if self.gm_pred is None:
            return "Mean ground motions have not been calculated - first run self.get_gm_pred to get predictions"
        im = im.lower() # convert to lower case
        if not im in self.im_list:
            return f"{im} was not specified when calculating predictions, re-run self.get_gm_pred if {im} is needed"
        # empty self.haz_curve
        self.haz_curve = None
        # get x_values if not provided
        if x_vals is None:
            if isinstance(im, str):
                if im.lower() == 'pgv':
                    x_vals = np.logspace(-1, 3, 41)
                else:
                    x_vals = np.logspace(-3, 1, 41)
            else:
                if im == -1: # PGV
                    x_vals = np.logspace(-1, 3, 41)
                else:
                    x_vals = np.logspace(-3, 1, 41)
        # calculate haz curve
        means = self.gm_pred[im]['mean'][:, site_num]
        ind_im_mean_gt_m10 = np.where(means>-10)[0] # keep means > -10 only (within max dist)
        # get reduced list
        means = means[ind_im_mean_gt_m10]
        sigmas = self.gm_pred[im]['sigma'][ind_im_mean_gt_m10, site_num]
        sigma_mus = self.gm_pred[im]['sigma_mu'][ind_im_mean_gt_m10, site_num]
        # rates = self.source.df_rupture.AnnualRate[ind_im_mean_gt_m10].values
        if self.rate is not None:
            rates = self.rate[ind_im_mean_gt_m10]
        elif self.source is not None:
            if getattr(self.source,'df_rupture',None) is not None:
                rates = self.source.df_rupture.AnnualRate[ind_im_mean_gt_m10].values
            else:
                rates = np.ones(len(ind_im_mean_gt_m10))/len(ind_im_mean_gt_m10)
        else:
            rates = np.ones(len(ind_im_mean_gt_m10))/len(ind_im_mean_gt_m10)
        # get epistemic samples
        # epi_samples = np.random.rand(100)
        self.n_epi_sample = n_epi_sample
        epi_samples = norm.rvs(size=self.n_epi_sample)
        # get mean hazard
        haz_curve_mean = np.sum(
            norm.sf(
                np.log(x_vals), # x values
                loc=np.expand_dims(means,axis=1), # mean + sigma_mu * epi_sample
                scale=np.expand_dims(sigmas,axis=1) # aleatory or sigma
            ) * np.expand_dims(rates,axis=1) # multiply by rates
        ,axis=0)
        # get hazard curves for epistemic samples
        haz_curve = np.asarray([
            np.sum(
                norm.sf(
                    np.log(x_vals), # x values
                    loc=np.expand_dims(means,axis=1) + samp*np.expand_dims(sigma_mus,axis=1), # mean + sigma_mu * epi_sample
                    scale=np.expand_dims(sigmas,axis=1) # aleatory or sigma
                ) * np.expand_dims(rates,axis=1) # multiply by rates
            ,axis=0)
            for samp in epi_samples # loop through samples
        ])
        # sort hazard curves to get fractiles
        haz_curve_sort = haz_curve[haz_curve[:,-1].argsort()] # sort by last column
        haz_curve_frac = np.asarray([
            haz_curve_sort[int(np.round(self.n_epi_sample/100*frac,decimals=0))]
            for frac in fractiles
        ])
        # store to instance
        self.haz_curve = {
            'x': x_vals,
            'mean': haz_curve_mean,
            'samples': haz_curve,
            'fractiles': haz_curve_frac,
            'fractile_vals': fractiles,
            'site_num': site_num
        }
        # plot
        if to_plot:
            self.plot_haz_curve()
            # export
            if to_export:
                self.export_plot(spath=export_spath)
        

    def plot_haz_curve(self, show=True, figsize=[12,8], fignum=None):
        """plots hazard curve"""
        if self.haz_curve is None:
            return "Nothing to export - first run self.get_haz_curve to get hazard curve"
        # if matplotlib is loaded, make plot
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot generate figure')
        else:
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
            
            # plot mean curve
            self._ax[0,0].plot(self.haz_curve['x'], self.haz_curve['mean'], 'r', zorder=5)
            lg_str = ['mean']
            # plot fractiles
            for i, frac in enumerate(self.haz_curve['fractile_vals']):
                self._ax[0,0].plot(self.haz_curve['x'], self.haz_curve['fractiles'][i], '--g', zorder=10)
                lg_str.append(f'fractile: {frac}th')
            # plot samples
            lines = [np.transpose([self.haz_curve['x'],self.haz_curve['samples'][i]]) for i in range(self.n_epi_sample)]
            line_coll = LineCollection(lines, colors='gray')
            self._ax[0,0].add_collection(line_coll)
            lg_str.append('samples')
            
            # add legend
            self._ax[0,0].legend(lg_str, loc='lower left')
            
            # other formatting operations
            self._ax[0,0].set_xlabel('IM (g)')
            self._ax[0,0].set_ylabel('Prob (im>IM)')
            self._ax[0,0].set_title(f'Hazard curve for site #{self.haz_curve["site_num"]}')
            self._ax[0,0].set_xscale('log')
            self._ax[0,0].set_yscale('log')
            self._ax[0,0].grid()
            # self._ax[0,0].set_aspect('equal')
            self._ax[0,0].autoscale()
            self._fig.tight_layout()
            
            # show or not
            if show is True:
                plt.show()
    
    
    def export_plot(self, spath=None, fmt='pdf', dpi='figure', orientation='landscape'):
        """export figure"""
        # only works if matplotlib is loaded
        if not 'matplotlib' in sys.modules:
            logging.info('The "matplotlib" module is not installed; cannot export figure')
        else:
            if spath is None:
                return "Must specify spath"
            else:
                self._fig_spath = spath
            self._fig.savefig(self._fig_spath, transparent=True, orientation=orientation, bbox_inches='tight')