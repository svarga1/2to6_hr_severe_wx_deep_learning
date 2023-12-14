##Script for use with hw7_base.py
#Sam Varga
import tensorflow as tf
from tensorflow import keras
from keras.layers import InputLayer, Dense, BatchNormalization, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalMaxPool2D, SpatialDropout2D, Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPool1D, SpatialDropout1D, AveragePooling1D, SimpleRNN, GRU, concatenate
from keras.layers.reshaping.up_sampling2d import UpSampling2D

def create_U_net_classifier_2D(image_size=(256,256), nchannels=26, n_classes=7, conv_layers=None,p_spatial_dropout=0.1, lrate=0.001, loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy], padding='same', activation_conv='relu', activation_out='softmax', clipvalue=None, skip=True ):
  '''Creates a U net classifier with skip connections for 2D Image classification'''
  '''Arguments:
  image_size-(width, height)- shape of images
  nchannels - number of channels for each pixel
  conv_layers- list of dictionaries containing architectural information for each conv layer
  p_spatial_dropout: probability of spatial dropout
  lrate: learning rate
  loss: loss function
  metrics: additional metrics to evaluate
  padding: type of padding to use in Conv layers
  activation_conv: activation nonlinearity for convolutional layers
  activation_out: activiation nonlinearity for output layer
  clipvalue: gradient clipping value
  skip: add skip connections if true'''
  

  reg=None #There are no dense layers, so we don't need L2 reg

  in_layer = Input(shape=(image_size[0],image_size[1],nchannels), name='Input') #Create input layer
  skip_points=[] #List to use with skip connections
  
  for i, conv in enumerate(conv_layers): #Create the encoding portion of the network
    
    main = Conv2D(conv['filters'], conv['kernel_size'], padding=padding, activation=activation_conv, name=f'Encode_{i}_0')(main if i!=0 else in_layer) #Connect to the input layer for the first time, then connect to the model
    if p_spatial_dropout:
      main=SpatialDropout2D(p_spatial_dropout, name=f'En_Sp_Dr_{i}_0')(main)
    main = Conv2D(conv['filters'], conv['kernel_size'], padding=padding, activation=activation_conv,  name=f'Encode_{i}_1')(main) #Add 2 Conv2D layers with the same number of units, as proposed in Ronneberger, Fischer, and Brox's 2015 Paper
    if p_spatial_dropout: #Insert dropout layers between convolutional layers
      main=SpatialDropout2D(p_spatial_dropout, name=f'En_Sp_Dr_{i}_1')(main)
    
    skip_points.append(main) #Create a skip point before pooling

    main = MaxPooling2D(pool_size=conv['pool_size'], padding=padding, name=f'Encode_Pool_{i}')(main) #Add the MaxPooling2D layer to downscale

  main = Conv2D(conv_layers[-1]['filters']*2, (1,1), padding=padding, name='Midway')(main) #Add the Midway point using twice the number of filters as the last layer


  for i, conv in enumerate(conv_layers[::-1]): #Create the decoding portion of the network
    #Upscale layer and concatenate with skip point
    main = UpSampling2D(conv['pool_size'], name=f'Decode_Upsample_{len(conv_layers)-1-i}')(main) #Upsample
    if skip: #Add skip connections
       main = concatenate([main, skip_points.pop()], name=f'Concat{len(conv_layers)-1-i}') #Concatenate with skip point and remove skip point from list
   
   #Add 2x Conv2D layers with dropout in between
    main = Conv2D(conv['filters'], conv['kernel_size'], padding=padding, activation=activation_conv, name=f'Decode_{len(conv_layers)-1-i}_1')(main)
    if p_spatial_dropout: #Insert dropout layers between convolutional layers
      main=SpatialDropout2D(p_spatial_dropout, name=f'De_Sp_Dr_{len(conv_layers)-1-i}_1')(main)
    main = Conv2D(conv['filters'], conv['kernel_size'], padding=padding, activation=activation_conv, name=f'Decode_{len(conv_layers)-1-i}_0')(main)
    if p_spatial_dropout: #Insert dropout layers between convolutional layers
      main=SpatialDropout2D(p_spatial_dropout, name=f'De_Sp_Dr_{len(conv_layers)-1-i}_0')(main)

  main = Conv2D(n_classes, (1,1), activation=activation_out, name='Output')(main) #Add output layer after decoder
  
  #Compile model and return it
  main = keras.Model(in_layer, main)
  opt= tf.keras.optimizers.legacy.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipvalue=clipvalue)
  main.compile(loss=loss, optimizer=opt, metrics=metrics)
  print(main.summary())

  return main