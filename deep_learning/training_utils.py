import pandas as pd
import numpy as np
import netCDF4
import h5netcdf
import xarray as xr
import pickle
from os.path import join

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

def save_results(args, u_net, history, train_ds, val_ds, test_ds, predict=False, outdir=None):
    fbase = f"{args['target_column']}_Rot_{args['rotation']}_{args['shape'][0]}_{args['shape'][1]}_lrate_{args['lrate']}_spatial_dropout_{args['p_spatial_dropout']}_i_{args['i']}_filters_{args['filters']}_size_{args['size']}_pool_{args['pool']}_loss_{args['loss']}"
    results = {}
    results['args'] = args
    if predict:
        results['predict_val'] = u_net.predict(val_ds)
        results['predict_train']=u_net.predict(train_ds)
    results['predict_val_eval'] = u_net.evaluate(val_ds)

    if test_ds is not None:
        results['predict_test_eval']=u_net.evaluate(test_ds)
        if predict:
            results['predict_test']=u_net.predict(test_ds)
            print(np.max(results['predict_test']))
            print(np.mean(results['predict_test']))

    results['predict_train_eval']=u_net.evaluate(train_ds)
    results['history']=history.history
    results['fname_base']=fbase

    #Save results
    with open(join(join(outdir, 'results'), f'{fbase}_results.pkl'),'wb') as fp:
        pickle.dump(results, fp)

    #save model
    if False:
        u_net.save(join(join(outdir, 'models'), f'{fbase}_model'))
    
    print(fbase)
    return None

def read_rotation_results(directory, filebase, n_rots):
    '''Loads the results dictionaries for every rotation of a single model'''
    """Directory: location of results dictionarys
    filebase: filename including wildcard (*) in place of rotation number"""
    import fnmatch
    import os
    import pickle
    
    #Get the separate rotation files for this HP combination
    files = [filebase.split('Rot_')[0]+'Rot_'+str(i)+filebase.split('Rot_')[1][1:] for i in range(n_rots)]
    
    results = []
    
    #Load rotations
    for f in files:
        with open(join(directory, f), 'rb') as fp:
            results.append(pickle.load(fp))
    
    return results

def load_hp_opt_results(directory, combination_list, n_rots=5):
    '''Loops through possible hyperparameter combinations, loads each rotations' results, and then computes average performance'''
    """directory - directory where the results dictionarys are saved
    combination_list - list of strings, where each string is a specific HP combination
    n_rots - number of rotations"""
    
    results = {}
    for combo in combination_list:
        results[combo] = read_rotation_results(directory, combo, n_rots)
        
    return results
    