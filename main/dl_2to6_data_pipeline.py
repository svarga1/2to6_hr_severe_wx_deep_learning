"""
    Functions Supporting the PatchExtractor Object Type
"""

import sys
from glob import glob
from os.path import join


#Custom Packages
sys.path.append('/home/samuel.varga/python_packages/WoF_post') #WoF post package
sys.path.append('/home/samuel.varga/python_packages/wofs_ml_severe/')
sys.path.append('/home/samuel.varga/python_packages/MontePython/')
from wofs.post.utils import (
    save_dataset,
    load_multiple_nc_files,
)

##################
##Fields to Load##
##################

ml_config = { 'ENS_VARS':  ['uh_2to5_instant',
                            'uh_0to2_instant',
                            'wz_0to2_instant',
                            'comp_dz',
                            'ws_80',
                            'hailcast',
                            'w_up',
                            'okubo_weiss',
                            'ctt'
                    ],
             
              'ENV_VARS' : ['mid_level_lapse_rate', 
                            'low_level_lapse_rate', 
                           ],
             
              'SVR_VARS': ['shear_u_0to1', 
                        'shear_v_0to1', 
                        'shear_u_0to6', 
                        'shear_v_0to6',
                        'shear_u_3to6', 
                        'shear_v_3to6',
                        'srh_0to3',
                        'cape_ml', 
                        'cin_ml', 
                        'stp',
                        'scp',
                       ]
            }

##############
##File Input##
##############

def get_files(path, TIMESCALE):
    """Get the ENS, ENV, and SVR file paths for the 0-3 || 2-6 hr forecasts"""
    # Load summary files between time step 00-36 || 24-72. 
    if TIMESCALE=='0to3':
        ens_files = glob(join(path,'wofs_ENS_[0-3]*')) 
        ens_files.sort()
        ens_files = ens_files[:37] #Drops the last 4 files, so we have 0-36
    elif TIMESCALE=='2to6':
        ens_files = glob(join(path,'wofs_ENS_[2-7]*'))
        ens_files.sort()
        ens_files = ens_files[4:] #Drops the first 4 files, so we have 24-72 instead of 20-72
    
    svr_files = [f.replace('ENS', 'SVR') for f in ens_files]
    env_files = [f.replace('ENS', 'ENV') for f in ens_files]
    
    return ens_files, env_files, svr_files

def load_dataset(path, TIMESCALE='2to6'):
    """Load the 0-3|| 2-6 hr forecasts"""
    ens_files, env_files, svr_files = get_files(path, TIMESCALE)
    
    coord_vars = ["xlat", "xlon", "hgt"]
    
    X_strm, coords, _, _  = load_multiple_nc_files(
                ens_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENS_VARS'])

    X_env, _, _, _  = load_multiple_nc_files(
                env_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['ENV_VARS'])

    X_svr, _, _, _ = load_multiple_nc_files(
                svr_files, concat_dim="time", coord_vars=coord_vars,  load_vars=ml_config['SVR_VARS'])

    X_env = {**X_env, **X_svr}

    X_env = {v : X_env[v][1] for v in X_env.keys()}
    X_strm = {v : X_strm[v][1] for v in X_strm.keys()}
    
    ll_grid = (coords['xlat'][1].values, coords['xlon'][1].values)
    
    return X_env, X_strm, ens_files[0], ll_grid