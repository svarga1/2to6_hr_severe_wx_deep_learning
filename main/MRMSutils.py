import pandas as pd
import numpy as np
import netCDF4
import h5netcdf
import xarray as xr
from os.path import join
import joblib
from glob import glob
import datetime as dt
import sys
import pyresample
import itertools
from pathlib import Path

#Custom Packages
sys.path.append('/home/samuel.varga/python_packages/WoF_post') #WoF post package
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe/')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
sys.path.append('/home/samuel.varga/projects/deep_learning/')

#Filters
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter

from wofs.post.utils import (
    save_dataset,
    load_multiple_nc_files,
)
from main.dl_2to6_data_pipeline import get_files, load_dataset
from collections import ChainMap
from wofs.plotting.util import decompose_file_path


class MeshGrabber:
    """
    MESHGrabber links WoFS ensemble storm tracks with MRMS MESH objects 
    """
   
    def __init__(self, ncfile, size, grid_size =np.zeros([300,300]), mm_threshold=30, err_window = 15, return_df=False, forecast_window='2to6' , n_ens=18):
        # Get the beginning of the 30-min period for the ENSEMBLETRACK file.
        self.init_path_date = ncfile.split('/')[4]
        self.size = size
        self.ncfile = ncfile
        self.grid_size = grid_size
        self.mm_threshold=mm_threshold
        self.path_date = (pd.to_datetime(decompose_file_path(ncfile)['VALID_DATE']+decompose_file_path(ncfile)['INIT_TIME'])).strftime('%Y%m%d%H%M') #Path date (str) not including 2 hour desync
        self.path_date_dt = dt.datetime.strptime(self.path_date, '%Y%m%d%H%M')
        self.err_window = err_window
        self.forecast_window = forecast_window
        self.return_df = return_df
        self.n_ens = n_ens
        self.MRMS_PATHS = {#'2019': '/work/brian.matilla/WoFS_2020/MRMS/RAD_AZS_MSH/2019/',
              #'2020' : '/work/brian.matilla/WoFS_2020/MRMS/RAD_AZS_MSH/2020/',
              #'2021' : '/work/brian.matilla/WOFS_2021/MRMS/RAD_AZS_MSH/',
            '2018' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2018/',
            '2019' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2019/',
            '2020' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2020/',
            '2021' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2021/',  
            '2022' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2022/',
            '2023' : '/work/rt_obs/MRMS/RAD_AZS_MSH/2023/',
             }
        
        
    def __call__(self):
        mesh_grid = self.load()
        mesh_grid = self.to_boolean_grid(mesh_grid)
        mesh_grid = self.coarsen_values(mesh_grid, self.size)
        
        
        return mesh_grid
    
    def dt_rng(self):
        sdate = dt.datetime.strptime(self.path_date, '%Y%m%d%H%M')
        
        #Add Desync for lagged forecasts
        sdate+=dt.timedelta(minutes=120 if self.forecast_window=='2to6' else 0) 
        
        #End date is start of forecast + duration + window
        edate = sdate+dt.timedelta(minutes=240 if self.forecast_window=='2to6' else 180)+dt.timedelta(minutes=self.err_window)
        
        #Start time is 15 minutes before
        sdate-=dt.timedelta(minutes=self.err_window) 
    
        return sdate, edate 

    def find_mrms_files(self):
        
        sdate, edate = self.dt_rng()
        date_rng = pd.date_range(sdate, edate, freq=dt.timedelta(minutes=5))
        
        mrms_filenames = [date.strftime('wofs_MRMS_RAD_%Y%m%d_%H%M.nc') for date in date_rng]
        mrms_filepaths = [Path(self.MRMS_PATHS[str(self.path_date_dt.year)]).joinpath(self.init_path_date, f) for f in mrms_filenames 
                  if Path(self.MRMS_PATHS[str(self.path_date_dt.year)]).joinpath(self.init_path_date, f).is_file()
                 ]
    
        return mrms_filepaths 
    
    def load(self):
        files = self.find_mrms_files()
        
        # Load the file for the 30-min+ and concate along time dim. 
        if files:
            try:
                mrms_ds = xr.concat([xr.load_dataset(f, drop_variables=['lat', 'lon']) 
                         for f in files], dim='time') 
                out = mrms_ds['mesh_consv'].max(dim='time').values
            except:
                try:
                    print(f'Issues loading files: {files}')
                    print('Default MESH Read-in failed, trying alternate method')
                    mrms_ds=[xr.load_dataset(f, drop_variables=['lat','lon'])['mesh_consv'].values for f in files]
                    out=np.max(mrms_ds, axis=0)
                except:
                    print('Alternate MESH Read-in failed, returning null values')
                    out = -1*np.ones_like(self.grid_size)
        else:
            #No mesh files-- return grid of -1
            print('No MESH files, returning -1')
            out = -1*np.ones_like(self.grid_size)
        # Compute the time-max MESH and return 
        return out
    
    def to_boolean_grid(self, mesh_grid):
       
        IN_TO_MM = 25.4
        meshhold = self.mm_threshold / IN_TO_MM #Mesh Threshold in inches -- values < are 0, values >= are 1
        out = np.zeros_like(mesh_grid, dtype=int)
        out[mesh_grid >= meshhold] = 1 #Map values >= Meshold to 1, otherwise 0
        
        return out
    
    def coarsen_values(self, mesh_grid, size):
        '''Applies a maximum filter of size size to the original grid, preparing for coarsening'''
        
        return maximum_filter(mesh_grid, size)
    
    def load_comp_dz(self):
        '''Loads the composite reflectivity at the initialization time of the forecast'''
        sdate = dt.datetime.strptime(self.path_date, '%Y%m%d%H%M') #Init time of forecast
        
        mrms_filenames = [date.strftime('wofs_MRMS_RAD_%Y%m%d_%H%M.nc') for date in [sdate]]
        mrms_filepaths = [Path(self.MRMS_PATHS[str(self.path_date_dt.year)]).joinpath(self.init_path_date, f) for f in mrms_filenames 
                  if Path(self.MRMS_PATHS[str(self.path_date_dt.year)]).joinpath(self.init_path_date, f).is_file()]
        
        try:
            mrms_ds = xr.load_dataset(mrms_filepaths[0], drop_variables=['lat','lon'])
            if int(self.ncfile.split('/')[4][:4]) >= 2023:
                print('Using 2023 Naming Convention')
                out_var='refl_consv'
            else:
                out_var = 'dz_consv'
            out = mrms_ds[out_var].values
        except:
            print('Something Went Wrong Loading Composite Reflectivity')
            out = -1*np.ones_like(self.grid_size)
                 
        return out