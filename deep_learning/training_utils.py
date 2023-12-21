import pandas as pd
import numpy as np
import netCDF4
import h5netcdf
import xarray as xr
import pickle

def load_rotation(filepath, rotation, target_column, training=False, verbose=False):
    '''Loads the rotation file, reshapes to be (samples, y, x, channels), selects appropriate target variables,
    and returns the predictors and targets as arrays'''
    '''Arguments:
    filepath - path to nc file 
    rotation - int - rotation number
    training - boolean - if true, returns scalers as well as data '''
    
    #Load Scaling information if loading training data
    if 'train' in filepath or training:
        training=True
        print('Training path detected - loading scaling')
        scalers = pd.read_pickle(f'/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/scaling/rot_{rotation}_scaling.pkl')
        predictor_mean, predictor_variance = scalers['mean'], scalers['var']
        
    
    #Load NCDF
    ds = xr.open_dataset(filepath, engine='netcdf4')
    
    #Split predictors and targets and reshape into (samples, lat, lon, channels)
    X = np.stack([ds[v].values for v in ds.variables if 'severe' not in v], axis=-1)
    y = np.stack([ds[v].values for v in ds.variables if 'severe' in v], axis=-1)
    
    #Select specified target variable
    target_ind = np.argwhere(np.array([v for v in ds.variables if 'severe' in v])==target_column)[0][0]
    y = y[:,:,:,target_ind]
    
    #Debug
    if verbose:
        print(targ_ind)
        print(np.shape(X))
        print(np.shape(y))
        
    ds.close()
    
    if training:
        return X, y, predictor_mean, predictor_variance
    else:
        return X, y
def convert_to_tf(data_in, batch_size=None):
    '''For training/validation, data_in should be (X,y)'''
    import tensorflow as tf
    
    tf_data = tf.data.Dataset.from_tensor_slices(data_in)   
    if batch_size:
        tf_data = tf_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return tf_data    