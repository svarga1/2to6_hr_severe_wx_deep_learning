################
###CS5043 HW7###
################
'''Sam Varga'''
'''Create a U-Net and Sequential CNN for Semantic Segmentation of Chesepeake Bay Watershed Images'''

##Imports
import sys
import argparse
import pickle
import pandas as pd
import py3nvml
import tensorflow as tf
from tensorflow.keras.utils import plot_model
tf_tools='./'
sys.path.append(tf_tools + 'job_control')
#sys.path.append(tf_tools + 'deep_networks_hw7')
sys.path.append(tf_tools+'chesapeake_loader')
from job_control import *
from deep_networks_hw7 import *
from chesapeake_loader import *

###Set up Argparse
def create_parser():
    '''
    Create argument parser - repurposed & updated from HW3/4
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='HW6', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='Loss Function')
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/pfam', help='Data set directory')
    parser.add_argument('--Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--version',type=str, default='B', help='Version of pfam dataset to load')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=3, help='Number of training folds')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")

    # Convolutional parameters
    parser.add_argument('--conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--max_pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('--padding', type=str, default='valid', help='Padding type for convolutional layers')
    parser.add_argument('--flatten', action='store_true', default=False, help='Flatten Conv output instead of using GlobalMaxpooling')
    parser.add_argument('--GMP1D', action='store_true', default=False, help='Add a Global Max Pooling layer after convolutional layers')
    parser.add_argument('--skip', action='store_true', default=False, help='Add skip connections across the U-net')
    #Recurrent Parameters
    parser.add_argument('--recurrent', nargs='+', type=int, default=[5,5], help='Number of units in each recurrent layer')
    parser.add_argument('--n_tokens', type=int, default=3934, help='Number of possible tokens')
    parser.add_argument('--n_embeddings', type=int, default=15, help='Number of embedding dimensions for the tokens')
    parser.add_argument('--avg_pool', nargs='+', type=int, default=[0,0], help='Avg pooling size (<2=None)')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[5,5], help='No. Attention Heads per MHA layer')
    parser.add_argument('--key_dim', nargs='+', type=int, default=[5,5], help='Size of attention head for Q/K')
    parser.add_argument('--return_attention_scores','-ras', action='store_true', default=False, help='In Development - uses attention scores from MHA as additional input')
    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('--hidden_activation',type=str, default=None, help='Activation Nonlinearity for Hidden Layers')
    parser.add_argument('--conv_activation',type=str, default=None, help='Activation Nonlinearity for Conv1D Layers')
    parser.add_argument('--rec_activation',type=str, default=None, help='Activation Nonlinearity for SimpleRNN Layers')
    parser.add_argument('--output_activation',type=str, default='softmax', help='Activation Nonlinearity for Output Layer')
    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--spatial_dropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--recurrent_dropout', type=float, default=None, help='Dropout rate for recurrent layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")
    parser.add_argument('--clipvalue', type=float, default=None, help='Value for gradient clipping')
    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")
    parser.add_argument('--save_weights',action='store_true', default=True, help="Save the weights while training")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=None, help="Size of the shuffle buffer (0 = no shuffle")
    
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help="Number of training batches per epoch (must use --repeat if you are using this)")

    # Post
    parser.add_argument('--save_model', action='store_true', default=False , help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    parser.add_argument('--overwrite_output', action='store_true', default=False, help='Overwrite any fnames that already exist -- USE WITH CAUTION')
    
    return parser

###Set up job control
def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type.lower() in [None, 'unet','cnn']:
        p = {'rotation': range(0,5)} #Use the command line parameters and go through all rotations
    elif args.exp_type == 'cnn_search' or args.exp_type=='unet_search':
        p = {'rotation':range(0,2),
        'spatial_dropout':[0.05,0.1,0.15,0.2]}
    else:
        assert False, "Unrecognized exp_type"

    return p

def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

def check_args(args):
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.spatial_dropout is None or (args.spatial_dropout > 0.0 and args.spatial_dropout < 1)), "Spatial dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L1_regularization is None or (args.L1_regularization > 0.0 and args.L1_regularization < 1)), "L1_regularization must be between 0 and 1"
    assert (args.L2_regularization is None or (args.L2_regularization > 0.0 and args.L2_regularization < 1)), "L2_regularization must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    

def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if(index is None):
        return ""
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)

###Set up filename
def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Conv configuration
    if args.conv_nfilters: 
        conv_size_str = '_'.join(str(x) for x in args.conv_size)
        conv_filter_str = '_'.join(str(x) for x in args.conv_nfilters)
        maxpool_str = '_'.join(str(x) for x in args.max_pool)
    else:
        conv_size_str=''; conv_filter_str=''; maxpool_str=''
    #Recurrent Configuration
    if args.recurrent:
        rec_str = '_'.join(str(x) for x in args.recurrent) 
        n_embed_str = f'_{args.n_embeddings}'
        avgpool_str = '_'.join(str(x) for x in args.avg_pool)
    elif args.num_heads:
        rec_str='Head'+'_'.join(str(x) for x in args.num_heads)
        n_embed_str = f'_{args.n_embeddings}'
        avgpool_str = 'Key_Dim'+'_'.join(str(x) for x in args.key_dim)
    else:
        rec_str=''; n_embed_str=''; avgpool_str=''
    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_'%(args.dropout)
        
    # Spatial Dropout
    if args.spatial_dropout is None:
        sdropout_str = ''
    else:
        sdropout_str = 'sdrop_%0.3f_'%(args.spatial_dropout)

    if args.recurrent_dropout is None:
        rdropout_str=''
    else:
        rdropout_str='rdrop_%0.3f_'%(args.recurrent_dropout)
        
    # L1 regularization
    if args.L1_regularization is None:
        regularizer_l1_str = ''
    else:
        regularizer_l1_str = 'L1_%0.6f_'%(args.L1_regularization)

    # L2 regularization
    if args.L2_regularization is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.L2_regularization)


    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
        
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type

    # learning rate
    lrate_str = "LR_%0.6f_"%args.lrate
    
    # Put it all together, including # of training folds and the experiment rotation
    return "%s/chesapeake_%s%sCsize_%s_Cfilters_%s_MP_%s__Pad_%s_hidden_%s_%s%s%s%s%s%sntrain_%02d_rot_%02d"%(args.results_path,
                                                                                           experiment_type_str,
                                                                                           label_str,
                                                                                           conv_size_str,
                                                                                           conv_filter_str,
                                                                                           maxpool_str,
                                                                                           args.padding,
                                                                                           hidden_str, 
                                                                                           dropout_str,
                                                                                           sdropout_str,
                                                                                           rdropout_str,
                                                                                           regularizer_l1_str,
                                                                                           regularizer_l2_str,
                                                                                           lrate_str,
                                                                                           args.Ntraining,
                                                                                           args.rotation)

##Experiment Execution
def execute_exp(args=None, multi_gpus=False):
    '''
    Perform the training and evaluation for a single model
    :param args: argparse arguments
    :param multi_gpus: number of gpus >1
    '''

    #Check arguments
    if args is None:
        #Use default arguments established in create_parser
        parser= create_parser()
        args=parser.parse_args([])
    
    print(args.exp_index)

    #Override the command line arguments if exp index has specific arguments based on job control
    args_str = augment_args(args)

    #Scale batch size based on available GPUs
    if multi_gpus > 1:
        args.batch=args.batch*multi_gpus

    
    #Load datasets for training, validation, and testing -- We add 5 to the fold, so that exp index [0,1,2,3,4]-> folds [5,6,7,8,9]
    ds_train, ds_validation, ds_testing, num_classes = create_datasets(base_dir=args.dataset, fold=args.rotation+5, train_filt='*[012345678]', cache_dir=None, repeat_train=args.repeat, shuffle_train=args.shuffle,
                                                               batch_size=args.batch, prefetch=args.prefetch, num_parallel_calls=args.cpus_per_task)
    
    image_size=(256,256)
    nchannels=26
    
    ##Build the model
    #Create dictionaries containing architecture info
    #dense_layers=[{'units':i} for i in args.hidden]

    if args.conv_nfilters: #Use the specified CNN params - The Unet function mirrors the layers, so don't need to change this.
        conv_layers =[{'filters': f, 'kernel_size': (s), 'pool_size': (p), 'strides': (p)} if p > 1
                   else {'filters': f, 'kernel_size': (s), 'pool_size': None, 'strides': None}
                   for s, f, p, in zip(args.conv_size, args.conv_nfilters, args.max_pool)]
    else:
        conv_layers = None

    #I have chosen not to include the flag for mirrored strategy, as I only have one GPU
    model = create_U_net_classifier_2D(image_size=image_size, nchannels=nchannels, n_classes=num_classes, conv_layers=conv_layers, p_spatial_dropout=args.spatial_dropout, lrate=args.lrate, loss=args.loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], padding=args.padding, activation_conv=args.conv_activation, activation_out=args.output_activation, skip=args.skip)
        
    if args.verbose>=1:
        print(model.summary())
    print(args)


    #Output filenames
    fbase = generate_fname(args, args_str)
    fname_out = f'{fbase}_results.pkl'

    #Plot graph based representation
    #plot_model(model, to_file=f'{fbase}_model_plot.png', show_shapes=True, show_layer_names=True)
    
    ##Stop Here for Nogo
    if(args.nogo):
        print('Do not pass go')
        print(fbase)
        return
    if os.path.exists(fname_out) and not args.overwrite_output:
        print(f'File {fname_out} already exists')
        return

    #Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(save_weights_only=True, verbose=1, filepath='%s_checkpt.ckpt'%fbase, save_best_only=True, monitor=args.monitor, mode='min')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                      min_delta=args.min_delta, monitor=args.monitor)

    #Learn the model
    #Steps per epoch will be used, so the dataset has to be repeated.
    #Validation steps=None means all validation samples are used. May be changed depending on runtime
    history = model.fit(ds_train, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
    use_multiprocessing=True, verbose=True, validation_data = ds_validation,
    validation_steps=None, callbacks=[early_stopping_cb, checkpoint_cb])

    #Generate the results dictionary
    results = {}
    results['args'] = args
    results['predict_validation'] = model.predict(ds_validation)
    results['predict_validation_eval'] = model.evaluate(ds_validation)

    if ds_testing is not None:
        results['predict_testing']=model.predict(ds_testing)
        results['predict_testing_eval']=model.evaluate(ds_testing)

    results['predict_training']=model.predict(ds_train, steps=args.steps_per_epoch)
    results['predict_training_eval']=model.evaluate(ds_train, steps=args.steps_per_epoch)
    results['history']=history.history
    results['fname_base']=fbase

    #Save results
    with open(f'{fbase}_results.pkl','wb') as fp:
        pickle.dump(results, fp)

    #save model
    if args.save_model:
        model.save(f'{fbase}_model')
    print(fbase)

    return model
    

########################
###Calling the Script###
########################

if __name__=='__main__':
    #Create parser and check arguments
    parser = create_parser()
    args=parser.parse_args()
    check_args(args)

    #Turn off gpu if not using
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'

    #GPU Check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    print(physical_devices)
    if(n_physical_devices > 0):
        py3nvml.grab_gpus(num_gpus=n_physical_devices, gpu_select=range(n_physical_devices))
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')

    if(args.check):
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Set number of threads, if it is specified
        if args.cpus_per_task is not None:
            tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
            tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

        execute_exp(args, multi_gpus=n_physical_devices)