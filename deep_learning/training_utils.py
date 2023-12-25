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
    if type(target_column) is str: #If only one column is requested
        target_ind = np.argwhere(np.array([v for v in ds.variables if 'severe' in v])==target_column)[0][0]
        y = y[:,:,:,target_ind]
    elif type(target_column) is list:
        y_dic = {}
        for targ in target_column: #if a list of target columns is supplied, return a dictionary
            target_ind = np.argwhere(np.array([v for v in ds.variables if 'severe' in v])==targ)[0][0]
            y_dic[str(targ)] = y[:,:,:,target_ind]
        y = y_dic
    
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

def resize_neural_net(model, new_input_shape):
    from tensorflow import keras
    from tensorflow.keras.layers import Activation
    import sys
    sys.path.insert(0, "/home/monte.flora/python_packages/wofs-super-resolution/dl4ds")
    import dl4ds as dds
    
    # replace input shape of first layer -- causes error
    model.layers[0]._batch_input_shape = new_input_shape
    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())
    #model.save("tmp.h5")
    #new_model = keras.models.load_model("tmp.h5",
    #                                custom_objects={"dssim_mse": dds.losses.dssim_mse,
    #                                                "Activation": Activation("relu"),
    #                                               })
    #new_model.summary()
    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))
    return new_model