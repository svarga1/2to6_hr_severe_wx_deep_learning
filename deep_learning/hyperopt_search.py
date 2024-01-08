#Import Statements
import numpy as np
import netCDF4
import h5netcdf
import xarray as xr
import sys
import tensorflow as tf
from tensorflow import keras
import pickle
from os.path import join
sys.path.append('/home/samuel.varga/projects/deep_learning/')
from deep_learning.training_utils import load_rotation, convert_to_tf, resize_neural_net, save_results
from deep_learning.deep_networks import create_U_net_classifier_2D
from keras import backend as K
import gc
from numba import cuda
from itertools import product

##Settings
outdir='/work/samuel.varga/projects/2to6_hr_severe_wx/DEEP_LEARNING/'
batch_size=2048
target_column='any_severe__36km'

#Metrics to evaluate
thresholds=[.15]
metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.MeanSquaredError(name='Brier score'),
    tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='prc', curve='PR'),     
         tf.keras.metrics.FalseNegatives(thresholds=thresholds), tf.keras.metrics.FalsePositives(thresholds=thresholds),
         tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.TrueNegatives(thresholds=thresholds),
         tf.keras.metrics.TruePositives(thresholds=thresholds)]

#Load testing set
X_test, y_test = load_rotation(join('/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/',f'wofs_dl_severe__2to6hr__testing_data'), None, target_column)
test_ds=convert_to_tf((X_test[None,:,:,:], np.expand_dims(y_test[None,:,:,:], axis=-1)))


#Hyperparam search

for rotation, p_s, lrate, cs, i, loss in product([1, 2 ,3, 4], [0.01, 0.1, 0.25], [0.0001, 0.001, 0.01, 0.1], ([2,1,2,1],[2,2,2,2],[2,3,2,3],[4,3,2,2]), (1,2,3,4), ['binary_crossentropy']):
    
    #Load Rotation and convert to tf dataset
    X_train, y_train, mean, variance = load_rotation(join('/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/',f'wofs_dl_severe__2to6hr__rot_{rotation}__training_data'), rotation, target_column)
    X_val, y_val = load_rotation(join('/work/samuel.varga/data/2to6_hr_severe_wx/DEEP_LEARNING/',f'wofs_dl_severe__2to6hr__rot_{rotation}__validation_data'), rotation, target_column)
    train_ds = convert_to_tf((X_train,np.expand_dims(y_train, axis=-1)), batch_size)
    val_ds = convert_to_tf((X_val,np.expand_dims(y_val, axis=-1)), batch_size)
    
    #U-net architectural parameters
    conv_filters=[n*i for n in [32,64,128,256]]
    max_pool=[2,2,2,2]
    conv_layers =[{'filters': f, 'kernel_size': (s), 'pool_size': (p), 'strides': (p)} if p > 1
                       else {'filters': f, 'kernel_size': (s), 'pool_size': None, 'strides': None}
                       for s, f, p, in zip(cs, conv_filters, max_pool)]
    args={'lrate':lrate, 'loss':loss,'activation_conv':'relu','activation_out':'sigmoid',
         'p_spatial_dropout':p_s, 'filters':conv_filters, 'size':cs, 'pool':max_pool, 'shape':(16,16),
         'rotation':rotation,'target_column':target_column, 'i':i}
##Create U-net
    u_net = create_U_net_classifier_2D(image_size=args['shape'], nchannels=63, n_classes=1, conv_layers=conv_layers, p_spatial_dropout=args['p_spatial_dropout'], metrics=metrics,
                               lrate=args['lrate'], loss=args['loss'], activation_conv=args['activation_conv'], activation_out=args['activation_out'],
                                      normalization=(mean, variance))
##
    early_stopping_cb =keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True,
                                                    min_delta=0.001, monitor='val_loss')
    tensorboard_cb= keras.callbacks.TensorBoard(log_dir=join(outdir,f'logs/Rot_{rotation}_p_s_{p_s}_lrate_{lrate}_cs_{cs}_i_{i}_loss_{loss}'), histogram_freq=1)
#Train the model
    history = u_net.fit(train_ds, epochs=100, verbose=True, validation_data = val_ds,
        callbacks=[early_stopping_cb, tensorboard_cb])
#Save results
    save_results(args, u_net, history, train_ds, val_ds, test_ds, outdir=outdir)
    
    
#When finished, clear the GPU
K.clear_session()
gc.collect()
cuda.close()
