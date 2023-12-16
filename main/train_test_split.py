# Import packages 
import pandas as pd
import numpy as np
import netCDF4
import h5netcdf
import xarray as xr
from os.path import join, exists
import joblib
from glob import glob
import datetime as dt
import pickle
import sys, os
import pyresample
import itertools
from pathlib import Path

#Filters
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter

#Custom Packages
sys.path.append('/home/samuel.varga/python_packages/WoF_post') #WoF post package
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe/')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
sys.path.append('/home/samuel.varga/projects/deep_learning/')

from wofs.post.utils import (
    save_dataset,
    load_multiple_nc_files,
)
from main.dl_2to6_data_pipeline import get_files, load_dataset
from collections import ChainMap

#Get list of Patch files - convert cases to datetime
path_base = f'/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/SummaryFiles/'
file_base = f'wofs_DL2TO6_16_16_data.feather'
meta_file_base = f'wofs_DL2TO6_16_16_meta.feather'
out_path = '/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/'


def format_metadata(meta_data_list):
    '''Reformats the metadata to appease the duplicate index errors.'''
    '''Args: meta_data_list: list of opened datasets'''
    meta = {}
    for v in meta_data_list[0].variables:
        #print(v)
        if v in ['run_date','init_time','patch_no']:
            meta[v] = np.append(np.array([]), [ x[v].values for x in meta_data_list])
        else:
            meta[v] = (['patch','NY_ind','NX_ind'],np.reshape(np.append(np.array([]), [x[v].values for x in meta_data_list]), (10*len(meta_data_list),16,16)))
        #print(np.shape(meta[v]))

    #Open NC file, add vars, save
    meta_ds = xr.Dataset(meta)
    return meta_ds
def save_rotation_nc(rot_num, train_ind, val_ind, unique_dates, path_list, date_list, out_path=out_path):
    '''rot_num: int - rotation number
        train_ind: list - list of indices for training folds - indices correspond to day in training_dates
        val_ind: list - list of indices for validation folds - indices correspond to day in training_dates
        unique_dates: list - list of unique dates in training set
        path_list: list - list of file paths of length N that contain directory info and init time
        date_list: list - list of dates of length N, with each date being YYYYmmdd for the corresponding path in path_list
    '''
    #Get list of paths for current rotation
    training_paths=list(np.array(path_list)[np.isin(np.array([date.strftime('%Y%m%d') for date in date_list]), unique_dates[train_ind])])
    validation_paths=list(np.array(path_list)[np.isin(np.array([date.strftime('%Y%m%d') for date in date_list]), unique_dates[val_ind])])
    
    #Add the filename to each of the paths
    print('Appending Filenames')
    training_file_paths = [join(path, file_base) for path in training_paths]
    training_meta_paths=[join(path, meta_file_base) for path in training_paths]
    validation_file_paths = [join(path, file_base) for path in validation_paths]
    validation_meta_paths=[join(path, meta_file_base) for path in validation_paths]
    
    
    #Create Training Data
    print(f'Saving training data for Rot {rot_num}')
    ds = [xr.open_dataset(f) for f in training_file_paths]
    ds = xr.concat(ds, dim='patch_no')

    #Save mean/variance for use in scaling 
    mean = np.array([np.nanmean(ds[v]) for v in ds.variables if 'severe' not in v])
    var = np.array([np.nanvar(ds[v]) for v in ds.variables if 'severe' not in v])
    with open(f'/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/scaling/rot_{rot_num}_scaling.pkl', 'wb') as scale_file:
        pickle.dump({'mean':mean,'var':var}, scale_file)
    
    ds.to_netcdf(join(out_path, f'wofs_dl_severe__2to6hr__rot_{rot_num}__training_data'))
    ds.close()
    
    print(f'Saving metadata for Rot {rot_num}')
    meta_ds = [xr.open_dataset(f) for f in training_meta_paths]
    meta_ds = format_metadata(meta_ds)
    meta_ds.to_netcdf(join(out_path, f'wofs_dl_severe__2to6hr__rot_{rot_num}__training_meta'))
    meta_ds.close()
    
    #Create validation data
    print(f'Saving validation data for Rot {rot_num}')
    ds = [xr.open_dataset(f) for f in validation_file_paths]
    ds = xr.concat(ds, dim='patch_no')
    ds.to_netcdf(join(out_path, f'wofs_dl_severe__2to6hr__rot_{rot_num}__validation_data'))
    ds.close()
    
    print(f'Saving metadata for Rot {rot_num}')
    meta_ds = [xr.open_dataset(f) for f in validation_meta_paths]
    meta_ds = format_metadata(meta_ds)
    meta_ds.to_netcdf(join(out_path, f'wofs_dl_severe__2to6hr__rot_{rot_num}__validation_meta'))
    meta_ds.close()
                          
    return None


dates=[d for d in os.listdir(path_base) if '.txt' not in d]

paths=[] #Valid paths for worker function
bad_paths=[]
for d in dates:
    if d[4:6] !='05': 
        continue

    times = [t for t in os.listdir(join(path_base, d)) if 'basemap' not in t] #Init time

    for t in times:
        path = join(path_base, d , t)
        if exists(join(path,file_base)):
            paths.append(path)
print(paths[0])
print(f'Num Total Paths: {len(paths)} ')

#Check files to see where bad MRMS data, drop cases from list of files
for path in paths:
    ds = xr.load_dataset(join(join(path_base, path), file_base))
    if np.any(ds['MESH_severe__4km'].values<0) or np.any(ds['MRMS_DZ'].values<0):
        print('Bad path found - Missing Data')
        bad_paths.append(path)
        paths.remove(path)
    elif np.any(ds['MRMS_DZ'].values > 10**35):
        print('Bad path found - MRMS DZ Values exceed expected range')
        bad_paths.append(path)
        paths.remove(path)
    ds.close()
print(f'Num Paths w/ usable data: {len(paths)}') 

#Convert remaining files into train/validation/test based on day
temp_paths=[path.split('/')[-2][0:8]+path.split('/')[-1] for path in paths] #Different domains on the same day are treated as identical for the purposes of T/T split
dates=[pd.to_datetime(path, format=f'%Y%m%d%H%M') for path in temp_paths]

#Split into train/test
from sklearn.model_selection import KFold as kfold, train_test_split
import random

all_dates = np.unique([date.strftime('%Y%m%d') for date in dates])
random.Random(42).shuffle(all_dates)
train_dates, test_dates = train_test_split(all_dates, test_size=0.3)
print('Training Dates:')
print(train_dates)

print('Testing Dates:')
print(test_dates)

#Split training set into 5 folds
train_folds = kfold(n_splits = 5, random_state=42, shuffle=True).split(train_dates)

with open(f'/work/samuel.varga/data/dates_split_deep_learning.pkl', 'wb') as date_file:
    pickle.dump({'train_dates':train_dates,'test_dates':test_dates}, date_file)
    
#Save training folds:
for i, (train_ind, val_ind) in enumerate(train_folds):
    save_rotation_nc(i, train_ind, val_ind, train_dates, paths, dates)
    
#Save testing set
testing_paths=list(np.array(paths)[np.isin(np.array([date.strftime('%Y%m%d') for date in dates]), test_dates)])
testing_file_paths = [join(path, file_base) for path in testing_paths]
testing_meta_paths=[join(path, meta_file_base) for path in testing_paths]


print(f'Saving testing data')
ds = [xr.open_dataset(f) for f in testing_file_paths]
ds = xr.concat(ds, dim='patch_no')
ds.to_netcdf(join(out_path, f'wofs_dl_severe__2to6hr__testing_data'))
ds.close()
    
print(f'Saving testing metadata')
meta_ds = [xr.open_dataset(f) for f in testing_meta_paths]
meta_ds = format_metadata(meta_ds)
meta_ds.to_netcdf(join(out_path, f'wofs_dl_severe__2to6hr__testing_meta'))
meta_ds.close()