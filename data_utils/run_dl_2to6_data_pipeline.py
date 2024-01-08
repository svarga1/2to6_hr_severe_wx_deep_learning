"""
    Script that creates a deep learning dataset
"""

################################
##Input and Output Directories##
################################
import os
from os.path import join, exists
from glob import glob
import pickle
import numpy as np
# Import packages 
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

#Filters
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter

#Custom Packages
sys.path.append('/home/samuel.varga/python_packages/WoF_post') #WoF post package
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe/')
sys.path.append('/home/monte.flora/python_packages/scikit-explain')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
sys.path.append('/home/samuel.varga/projects/deep_learning/')

from wofs.post.utils import (
    save_dataset,
    load_multiple_nc_files,
)
from data_utils.dl_2to6_data_pipeline import get_files, load_dataset
from collections import ChainMap
from wofs_ml_severe.data_pipeline.storm_report_loader import StormReportLoader
from data_utils.MRMSutils import MeshGrabber
from data_utils.PatchExtractor import PatchExtractor
from wofs_ml_severe.common.emailer import Emailer
from skexplain.common.multiprocessing_utils import run_parallel, to_iterator

########
##Main##
########
regen=True #If true, will regenerate the entire dataset. If false, will only create data for missing paths.

#######
##I/O##
#######

out_path_base = f'/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/'
input_path_base=f'/work/mflora/SummaryFiles'
random_state_file=f'/work/samuel.varga/data/random_state.pkl'
n_jobs=2
n_patches=15
patches=(16,16)

#####################################################
##Get List of Valid Dates and Path of Summary Files##
#####################################################

emailer = Emailer()
start_time = emailer.get_start_time()
print('Searching for valid paths')
dates=[d for d in os.listdir(input_path_base) if '.txt' not in d]

paths=[] #Valid paths for worker function

for d in dates:
    if d[4:6] !='05' or int(d[:4])<=2018: 
        continue

    times = [t for t in os.listdir(join(input_path_base, d)) if 'basemap' not in t] #Init time

    for t in times:
        path = join(input_path_base, d , t)
        
        if int(path.split('/')[4][:4]) >= 2021:
            files = glob(join(path, f'wofs_ALL_[2-7]*.nc'))
        else:
            files = glob(join(path, f'wofs_ENS_[2-7]*.nc'))
        #all_nc_files = len([f for f in files if f.endswith('.nc')])

        #if all_nc_files == len(files) and all_nc_files==53:
        if len(files) ==53:
            #print(path)
            paths.append(path)

print(paths)            
print(len(paths))

if regen:
    #Make all files
    pass
else:
    #Only make missing files
    print(f'Total Paths: {len(paths)}')
    existing_paths = [path.replace(input_path_base, join(out_path_base, f'SummaryFiles')) for path in paths]
    existing_paths = [join(path, f'wofs_DL2TO6_16_16_data.feather') for path in existing_paths]
    missing_paths=[]
    for path in existing_paths:
        if not exists(path):
            missing_paths.append(path)
            
    print(f'Total Missing Paths to be regened: {len(missing_paths)}')
    paths = [path.replace('wofs_DL2TO6_16_16_data.feather','') for path in missing_paths]
    paths=[str(path.replace(join(out_path_base, f'SummaryFiles'),input_path_base))[:-1] for path in paths]

    
    

print(f'Number of valid paths: {len(paths)}')
emailer.send_email(f'Starting patch extraction', start_time)
 
################
##Random State##
################
if exists(random_state_file):
    print('Using previous random states')
    random_states = pd.read_pickle(random_state_file)      
else:
    print('Generating Random States')
    states=np.random.RandomState(42).choice(np.arange(len(paths)), len(paths), replace=False)
    random_states = {path:states for path, states in zip(paths,states)}
    with open(random_state_file,'wb') as state_file:
        pickle.dump(random_states, state_file)
#################
##Worker Script##
#################

def worker(path, FRAMEWORK='POTVIN', TIMESCALE='2to6', patch_shape=(16,16)):
    print(path)
    random_state = random_states[path]
    #Load the files
    X_env, X_strm, ncfile, ll_grid = load_dataset(path, TIMESCALE=TIMESCALE)
    
    #Create predictors and extract patches
    patch_extractor = PatchExtractor(ncfile, ll_grid, X_env.keys(), X_strm.keys(), n_patches, patch_shape, random_state=random_state, verbose=True)
    ml_data, metadata = patch_extractor.make_dataset(X_env=X_env, X_strm=X_strm)
    
    
    #Save patch Summary Files
    out_path = path.replace(input_path_base, join(out_path_base, f'SummaryFiles'))
    if not exists(out_path):
        os.makedirs(out_path)
    out_name = join(out_path, f'wofs_DL{TIMESCALE.upper()}_{patch_shape[0]}_{patch_shape[1]}_data.feather')
    meta_name = join(out_path, f'wofs_DL{TIMESCALE.upper()}_{patch_shape[0]}_{patch_shape[1]}_meta.feather')
    
    print(f'Saving {out_name}...')
    
    ml_data.to_netcdf(out_name)
    ml_data.close()
    
    metadata.to_netcdf(meta_name)
    metadata.close()

    return None

#####################
##Run Worker Script##
#####################
emailer.send_email(f'Starting patchwork',start_time)

run_parallel( func = worker, n_jobs=n_jobs, args_iterator=to_iterator(paths))

emailer.send_email('Patches have been created', start_time)
