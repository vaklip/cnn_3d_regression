# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:46:29 2018

This script trains a 3D convolutional neural network with a user-defined 
architecture for regression and evaluates its performance on a validation set.
The script also performs testing on an independent dataset and the calculation 
of localization maps using Gradient-weighted Class Activation Mapping (Grad-CAM).
The particular mode (train/test/Grad-CAM) along with the relevant parameters
can be set below. The Grad-CAM method is detailed in:

    Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 
    2017. Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based 
    Localization. Presented at the Proceedings of the IEEE International 
    Conference on Computer Vision, pp. 618–626.

IMPORTANT:
During training, each data tensor may contain an arbitrary number of samples.
However, during evaluation, it is assumed that the number of samples in each
data tensor is an integer multiple of batch size. To this end, a batch size of
1 might be used for evaluation.

@author: Pál Vakli (RCNS HAS BIC)
"""

#%% ################################## Settings ###############################

#### Mode and model ID
# mode: 0 = Training & validation; 1 = Test; 2 = Perform Grad-CAM
# model_id: used for all three modes
mode = 0
model_id = 'cnn_3D_multicovar'
continue_train = False
continue_train_last_epoch = None
continue_train_tstamp = ''

#### File identifiers and rootpath 
# These identifiers are used for all three modes
data_fid = 't1'
label_fid = 't1'

#### Test mode settings
test_valid_evalind = 31
test_tstamp = 't20190507093139'
eval_training_set = True        # If true, performance on the whole training set is
                                # is evaluated alongside the evaluation of performance 
                                # on the test set
# This and the following two variables are used if you want to test on a novel 
# test set
novel_test_set = False 
novel_test_data_fid = ''
novel_test_label_fid = ''

#### Grad-CAM mode settings
rectify_heatmap = False          # If true, the resulting heatmap is passed 
                                 # through a ReLU function.

#### Training & Validation mode settings
# Optimizer
optim = 'Adam'                  # Type of optimizer to use: 'Momentum' or 'Adam'
momentum = 0.9                  # Parameter for Momentum optimizer

# A dictionary of hyperparameters specifying the network architecture. Keys:
# - input_chans   : Int specifying the number of input channels.
# - kernel_shapes : Tuple of 2-tuples specifying the shape of the convolutional
#                   kernels in each convolution block. E.g. (5, 3) specifies
#                   a spatially separated 3D convolution with kernel shapes 
#                   [5, 1, 1], [1, 5, 1], [1, 1, 5], followed by another spatially
#                   separated 3D convolution with kernel shapes [3, 1, 1], 
#                   [1, 3, 1], [1, 1, 3]. The number of elements in the kernel_shapes
#                   tuple gives the number of convolution blocks in the network
#                   (each block consisting of 2 subsequent spatially sperated 
#                   convolutions).
# - conv_paddings : Tuple of 2-tuples consisting of strings specifying the 
#                   padding applied in each spatially sperated convolution in 
#                   each convolution block - shoud have the same structure as 
#                   kernel_shapes.
# - conv_strides  : Tuple of 2-tuples specifying the stride in each 
#                   spatially sperated convolution in each convolution block -  
#                   shoud have the same structure as kernel_shapes.
# - conv_chans    : Tuple of 2-tuples specifying the number of kernels in each 
#                   spatially sperated convolution in each convolution block -  
#                   shoud have the same structure as kernel_shapes.
# - max_poolings  : Tuple of ints specifying the shape of the max pooling kernels
#                   at the end of each convolution block. E.g. a size of 3 means
#                   a max pooling kernel of shape [3, 3, 3]. The number of ints
#                   in the tuple should be the number of tuples in kernel_shapes
#                   minus 1 - the last convolution block is followed by global
#                   average pooling instead of max pooling.
# - max_poolings  : Tuple of ints specifying the strides for max pooling in each
#                   convolution block - should have the same structure as max_poolings.
# - fc_neurons    : Tuple of ints specifying the number of neurons in each fully
#                   connected layer. The number of elements in the tuple corresponds
#                   to the number of fully connected hidden layers.
# - num_covars    : Int specifying the number of covariates included in the model.
#                   These covariates are concatenated to the output of the global
#                   average pooling operation.
architecture_dict = {
                    'input_chans'   : 1,
                    'kernel_shapes' : ((5, 3), (3, 3), (3, 3), (3, 3)),
                    'conv_paddings' : (('SAME', 'SAME'), ('SAME', 'SAME'), 
                                       ('SAME', 'SAME'), ('SAME', 'SAME')),
                    'conv_strides'  : ((1, 1), (1, 1), (1, 1), (1, 1)),
                    'conv_chans'    : ((8, 16), (16, 32), (32, 64), (64, 128)),
                    'max_poolings'  : (3, 3, 3),
                    'pool_strides'  : (2, 2, 2),
                    'fc_nneurons'   : (128),
                    'num_covars'    : 2
                    }

# Hyperparameters specifying the training of the network
n_train_samples = 20        # Total number of samples in the training set.
n_epochs = 50               # Number of epochs for training.
batch_size = 2              # Batch size is always 1 for Grad-CAM mode (mode: 2).
keep_prob = 0.6             # Keeping probability for dropout in the fully connected layers.
learning_rate = 0.0005      # Learning rate for Adam or Momentum optimiziation.

# Data augmentation
do_augment = False          # If set to True, data augmentation is performed on-line.
bgval = 0                   # Value of the background of the anatomical images.
transl_range = 10           # Range of translations applied to the images.
rotate_range = 30           # Range of rotations (in degrees) applied to the images.

# If valid_eval_epochwise is set to True, validation set performance is evaluated 
# at the end of each epoch. Otherwise, validation set performance is elvauated 
# only at the end of training. Results (true and predicted labels along with 
# the subject identifiers) are saved for each validation data tensor separately.
valid_eval_epochwise = True

#%% ################################ Loading data #############################

# Importing necessary modules and packages
import tensorflow as tf
import os
from six.moves import cPickle as pickle
from datetime import datetime
from conv_net.load_and_save import gen_tensor_list, load_data_labels_multicovars, save_model_params, \
load_model_params
from conv_net.preprocess import remove_nans, normalize_tensor, randomize_tensor_covar, \
create_batch_positions, create_batch_covar
from conv_net.cnn_3d import cnn_3d_multicovar
from conv_net.evaluate import eval_multicovar

if mode == 2:
    from conv_net.grad_CAM import gradCAM_multicovar

rootpath = os.path.abspath(os.path.curdir)
tf_log_path = rootpath+'\\tf_logs\\'
tf_save_path = rootpath+'\\tf_save\\'

# Number of input channels
n_channels = architecture_dict['input_chans']

# Loading data tensors and normalization parameters for the training set in 
# Training and Validation mode (mode: 0) or in Test mode (mode: 1) if the whole 
# training set is evaluated (eval_training_set: True)
if mode == 0 or (mode == 1 and eval_training_set):
    # Loading normalization parameters
    params_fname = 'norm_params_x_train_{}.pickle'.format(data_fid)
    params_fpath = os.path.join(rootpath, 'train_set', params_fname)
    with open(params_fpath, 'rb') as f:
        norm_params = pickle.load(f)
    train_glob_mean = norm_params['glob_mean']
    train_glob_std = norm_params['glob_std']
    # Loading the list of training data tensors
    train_tensorlist = gen_tensor_list(rootpath, 'train', label_fid)

# Loading data, labels, and assembling the dictionaries guiding performance
# evaluation
### Test mode / Grad-CAM mode
if mode == 1 or mode == 2:
    # Evaluate training set performance on the whole training set in Test
    # mode (optional)
    if mode == 1 and eval_training_set:
        # Dict to store the training set set MAE after each evaluation
        train_maes = {}
        # Dictionary containing the parameters that control the evaluation of performance
        # on the training set
        train_eval_dict = {}
        train_eval_dict['tensorlist'] = train_tensorlist
        train_eval_dict['rootpath'] = rootpath
        train_eval_dict['data_fid'] = data_fid
        train_eval_dict['label_fid'] = label_fid
        train_eval_dict['glob_mean'] = train_glob_mean
        train_eval_dict['glob_std'] = train_glob_std
        train_eval_dict['batch_size'] = batch_size
        train_eval_dict['model_id'] = model_id
        train_eval_dict['tstamp'] = test_tstamp
    
    # Identifiers of test files
    if novel_test_set:
        test_data_fid = novel_test_data_fid
        test_label_fid = novel_test_label_fid
    else:
        test_data_fid = data_fid
        test_label_fid = label_fid
        
    # Loading normalization parameters for the test set
    params_fname = 'norm_params_x_test_{}.pickle'.format(test_data_fid)
    params_fpath = os.path.join(rootpath, 'test_set', params_fname)
    with open(params_fpath, 'rb') as f:
        norm_params = pickle.load(f)
    test_glob_mean = norm_params['glob_mean']
    test_glob_std = norm_params['glob_std']
    
    # Loading model parameters to retrieve architecture dict
    load_model_paramsin = (rootpath, test_label_fid, model_id, test_tstamp)
    model_params = load_model_params(*load_model_paramsin)
    architecture_dict = model_params['architecture_dict']
    
    # Test tensor list and dict to store the validation set MAE after each 
    # evaluation
    test_tensorlist = gen_tensor_list(rootpath, 'test', test_label_fid)
    test_maes = {}
    
    # Dictionary containing the parameters that control the evaluation of performance
    # on the validation set
    test_eval_dict = {}
    test_eval_dict['tensorlist'] = test_tensorlist
    test_eval_dict['rootpath'] = rootpath
    test_eval_dict['data_fid'] = test_data_fid
    test_eval_dict['label_fid'] = test_label_fid
    test_eval_dict['glob_mean'] = test_glob_mean
    test_eval_dict['glob_std'] = test_glob_std
    if mode == 2:
        test_eval_dict['batch_size'] = 1
    else:
        test_eval_dict['batch_size'] = batch_size
    test_eval_dict['model_id'] = model_id
    test_eval_dict['tstamp'] = test_tstamp
    
### Training & Validation mode    
elif mode == 0:
    # Loading normalization parameters for the validation set
    params_fname = 'norm_params_x_valid_{}.pickle'.format(data_fid)
    params_fpath = os.path.join(rootpath, 'valid_set', params_fname)
    with open(params_fpath, 'rb') as f:
        norm_params = pickle.load(f)
    valid_glob_mean = norm_params['glob_mean']
    valid_glob_std = norm_params['glob_std']
    
    # Loading the first data tensor and corresponding labels and subject IDs (eids)
    load_in = (rootpath, 'train', train_tensorlist[0], data_fid, label_fid)
    train_tensor, train_labels, train_covars, train_eids = load_data_labels_multicovars(*load_in)
    print('\nTraining tensor #{} loaded.\n'.format(train_tensorlist[0]))
    
    # Removing NaNs, normalizing and randomizing the training data      
    train_tensor = remove_nans(train_tensor)
    train_tensor = normalize_tensor(train_tensor, train_glob_mean, train_glob_std)
    random_in = (train_tensor, train_labels, train_covars, train_eids)
    train_tensor, train_labels, train_covars, train_eids = randomize_tensor_covar(*random_in)
    train_tensorind = 1
    
    # Creating an array of batch positions for the current data tensor
    train_batchpos, train_batchind = create_batch_positions(train_labels.shape[0], batch_size)
    
    # Instantiating 3D image manipulator class ImgManipAdj
    if do_augment:
        from conv_net.augment import ImgManipAdj
        manip_in = (transl_range, rotate_range, bgval, train_glob_mean, train_glob_std)
        imgManipAdj = ImgManipAdj(*manip_in)
    
    # Calculating the total number of steps
    n_steps_train = int(n_train_samples*n_epochs/batch_size)
    if continue_train:
        start_step = int(n_train_samples*continue_train_last_epoch/batch_size)+1
    else:
        start_step = 0
    
    # Log directory
    if continue_train:
        tstart = datetime.now()
        logdir = tf_log_path+"\\"+model_id+"\\"+continue_train_tstamp[1:]+"\\"
        tstamp = continue_train_tstamp
    else:
        tstart = datetime.now()
        logdir = tf_log_path+"\\"+model_id+"\\"+tstart.strftime("%Y%m%d_%H%M%S")+"\\"
        tstamp = 't'+tstart.strftime("%Y%m%d%H%M%S")
    
    # Validation tensor list and dict to store the validation set MAE after each 
    # evaluation
    valid_tensorlist = gen_tensor_list(rootpath, 'valid', label_fid)
    valid_maes = {}
    if valid_eval_epochwise:
        if continue_train:
            valid_evalind = continue_train_last_epoch+1
        else:
            valid_evalind = 0
        
    # Dictionary containing the parameters that control the evaluation of performance
    # on the validation set
    valid_eval_dict = {}
    valid_eval_dict['tensorlist'] = valid_tensorlist
    valid_eval_dict['rootpath'] = rootpath
    valid_eval_dict['data_fid'] = data_fid
    valid_eval_dict['label_fid'] = label_fid
    valid_eval_dict['glob_mean'] = valid_glob_mean
    valid_eval_dict['glob_std'] = valid_glob_std
    valid_eval_dict['batch_size'] = batch_size
    valid_eval_dict['model_id'] = model_id
    valid_eval_dict['tstamp'] = tstamp

#%% ##################### Defining the computational graph ####################
tf_batch_dataset_shape = (None, None, None, None, n_channels)
tf_batch_covars_shape = (None, architecture_dict['num_covars'])

graph = tf.Graph()

with graph.as_default():

    # Placeholders for batch data, labels, and covariates
    tf_is_train = tf.placeholder(tf.bool, name="is_train")
    tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    tf_batch_dataset = tf.placeholder(tf.float32, shape=tf_batch_dataset_shape, name='batch_data')
    tf_batch_labels = tf.placeholder(tf.float32, shape=(None, 1), name='batch_labels')
    tf_batch_covars = tf.placeholder(tf.float32, shape=tf_batch_covars_shape, name='batch_covars')

    # Predict batch labels and calculating loss (MSE) and MAE
    batch_prediction, last_rect_conv = cnn_3d_multicovar(tf_batch_dataset, tf_batch_covars, tf_is_train, tf_keep_prob, architecture_dict)
    gradients_prediction = tf.gradients(batch_prediction, last_rect_conv)
    loss = tf.losses.mean_squared_error(labels=tf_batch_labels, predictions=batch_prediction)
    mae = tf.reduce_mean(tf.abs(tf.subtract(batch_prediction, tf_batch_labels)))

    # Create a summary for the loss
    with tf.name_scope('Loss'):
        loss_summ = tf.summary.scalar('MSE', loss)
    
    # Create summary for mae
    with tf.name_scope('Accuracy'):
        mae_summ = tf.summary.scalar('MAE', mae)
        
    # Add Adam optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optim == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
        elif optim == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
    # Saver node
    saver = tf.train.Saver(max_to_keep=n_epochs)
    
#%% ##################### Running the computational graph #####################
with tf.Session(graph=graph) as session:
    
    if mode == 2:
        ### Grad-CAM mode
        # Restoring the model (in the same way as for the Test mode)
        save_name = '{}_{}_{}_eval{}.ckpt'.format(model_id, label_fid, test_tstamp, test_valid_evalind)
        saver.restore(session, os.path.join(tf_save_path, save_name))
        
        # A dictionary containing the necessary placeholder graph nodes
        tf_placeholders = {}
        tf_placeholders['tf_is_train'] = tf_is_train
        tf_placeholders['tf_keep_prob'] = tf_keep_prob
        tf_placeholders['tf_batch_dataset'] = tf_batch_dataset
        tf_placeholders['tf_batch_covars'] = tf_batch_covars
        
        # Grad-CAM mode
        grad_camin = ('test', test_valid_evalind, test_eval_dict, tf_placeholders, 
                      gradients_prediction, last_rect_conv, session, rectify_heatmap)        
        gradCAM_multicovar(*grad_camin)
        print('Grad-CAM finished.')
        
    elif mode == 1:
        #### Test mode
        # Restoring the model
        save_name = '{}_{}_{}_eval{}.ckpt'.format(model_id, label_fid, test_tstamp, test_valid_evalind)
        saver.restore(session, os.path.join(tf_save_path, save_name))
        
        # A dictionary of nodes that are used for evaluating performance on the
        # training and test set
        eval_tfnodes = {}
        eval_tfnodes['tf_is_train'] = tf_is_train
        eval_tfnodes['tf_keep_prob'] = tf_keep_prob
        eval_tfnodes['tf_batch_dataset'] = tf_batch_dataset
        eval_tfnodes['tf_batch_covars'] = tf_batch_covars
        
        # Evaluating performance on the training set (optional)
        if eval_training_set:
            train_evalin = ('train', test_valid_evalind, 0, train_maes, train_eval_dict, 
                           eval_tfnodes, batch_prediction, True)
            train_maes, _ = eval_multicovar(*train_evalin)
            print('Finished evaluating performance on the training set.')  
        
        # Evaluating performance on the test set
        test_evalin = ('test', test_valid_evalind, 0, test_maes, test_eval_dict, 
                       eval_tfnodes, batch_prediction, True)
        test_maes, _ = eval_multicovar(*test_evalin)
        print('Finished evaluating performance on the test set.')    
                
    elif mode == 0:
        ### Training&Validation mode
        if continue_train:
             save_name = '{}_{}_{}_eval{}.ckpt'.format(model_id, label_fid, continue_train_tstamp, continue_train_last_epoch)
             saver.restore(session, os.path.join(tf_save_path, save_name))
        else:
            # Initializing global variables
            tf.global_variables_initializer().run()
            print('\nGlobal variables initialized.\n')
        
        # Op to write into the log file
        write_summary = tf.summary.FileWriter(logdir, tf.get_default_graph())
        
        # A diuctionary of nodes that are used for evaluating performance on the
        # validation set
        eval_tfnodes = {}
        eval_tfnodes['tf_is_train'] = tf_is_train
        eval_tfnodes['tf_keep_prob'] = tf_keep_prob
        eval_tfnodes['tf_batch_dataset'] = tf_batch_dataset
        eval_tfnodes['tf_batch_covars'] = tf_batch_covars
        
        # Iterating over the training set
        for step in range(start_step, n_steps_train):
            
            # Printing step number
            print('Training step '+str(step)+'/'+str(n_steps_train-1))
            
            # Checking whether the current data tensor is exhausted
            if train_batchind == train_batchpos.shape[0]:                        
                
                # Checking whether the current data tensor list is finished, i.e.
                # an epoch has passed 
                if train_tensorind == len(train_tensorlist):
                    
                    # Freeing up memory
                    del train_tensor, train_labels, train_covars, train_eids
                    
                    # Saving model and evaluating performance on the validation set 
                    # when an epoch is over (optional)
                    if valid_eval_epochwise:
                        # Saving model
                        save_name = model_id+'_'+label_fid+'_'+tstamp+'_eval'+\
                                    str(valid_evalind)+'.ckpt'
                        saver.save(session, os.path.join(tf_save_path, save_name))
                        
                        # Evaluating validation set
                        valid_evalin = ('valid', valid_evalind, step, valid_maes, valid_eval_dict, 
                                        eval_tfnodes, batch_prediction, True)
                        valid_maes, valid_evalind = eval_multicovar(*valid_evalin)
                        
                    # Re-randomizing the training data tensor list before continuing
                    # training
                    train_tensorlist = gen_tensor_list(rootpath, 'train', label_fid)
                    train_tensorind = 0
                
                # Load training data tensor, labels, and subject IDs (eids)
                load_in = (rootpath, 'train', train_tensorlist[train_tensorind], data_fid, label_fid)
                train_tensor, train_labels, train_covars, train_eids = load_data_labels_multicovars(*load_in)
                print('\nTraining tensor #'+str(train_tensorlist[train_tensorind])+' loaded.\n')
                
                # Increment training data tensor index
                train_tensorind += 1
                
                # Preprocess training data tensor, labels, and subject IDs (eids)
                train_tensor = remove_nans(train_tensor)
                train_tensor = normalize_tensor(train_tensor, train_glob_mean, train_glob_std)
                random_in = (train_tensor, train_labels, train_covars, train_eids)
                train_tensor, train_labels, train_covars, train_eids = randomize_tensor_covar(*random_in)
                
                # Creating an array of batch positions for the current data tensor
                train_batchpos, train_batchind = create_batch_positions(train_labels.shape[0], batch_size)
                
            # Create batch
            cb_in = (train_tensor, train_labels, train_covars, train_batchpos, train_batchind)
            batch_data, batch_labels, batch_covars, train_batchind = create_batch_covar(*cb_in)
            
            # Data augmentation (optional)
            if do_augment:
                batch_data = imgManipAdj.rotate(batch_data)
                batch_data = imgManipAdj.translate(batch_data)
            
            # Feed batch data and labels to the placeholders and run session
            feed_dict = {tf_is_train      : True,
                         tf_keep_prob     : keep_prob,
                         tf_batch_dataset : batch_data, 
                         tf_batch_labels  : batch_labels,
                         tf_batch_covars  : batch_covars}
            list_ops = [update_ops, optimizer, loss, loss_summ, mae, mae_summ, batch_prediction]
            _, _, l, l_summary, batch_mae, mae_summary, batch_preds = session.run(list_ops, feed_dict=feed_dict)
    
            # At every 21. step write to log file and give some feedback on progress
            if (step % 20 == 0):
                write_summary.add_summary(l_summary, step)
                write_summary.add_summary(mae_summary, step)
                print('\nMinibatch loss at step %d: %f' % (step, l))
                print('Minibatch MAE: %.2f ' % batch_mae)
                now = datetime.now()
                elapsed_time = now-tstart
                tdelta_h = round(elapsed_time.total_seconds()/3600, 2)
                print('Hours passed since start: '+str(tdelta_h)+'\n')
        
        # Evaluating performance on the validation set when training is over (optional)
        if not valid_eval_epochwise:
            # Saving model
            save_name = model_id+'_'+label_fid+'_'+tstamp+'_eval'+\
                        str(valid_evalind)+'.ckpt'
            saver.save(session, os.path.join(tf_save_path, save_name))
            
            # Evaluating validation set performance
            valid_evalin = ('valid', n_epochs-1, step, valid_maes, valid_eval_dict, 
                            eval_tfnodes, batch_prediction, True)
            valid_maes, _ = eval_multicovar(*valid_evalin)
    
        # Dictionnary of model parameters (and overall validation results)
        model_params = {
                        'mode' : mode,
                        'model_id' : model_id,
                        'continue_train' : continue_train,
                        'continue_train_last_epoch' : continue_train_last_epoch,
                        'continue_train_tstamp' : continue_train_tstamp,
                        'train_data_fid' : data_fid,
                        'train_label_fid' : label_fid,
                        'rootpath' : rootpath,
                        'optimizer' : optim,
                        'n_channels' : n_channels,
                        'architecture_dict' : architecture_dict,
                        'n_train_samples' : n_train_samples,
                        'n_epochs' : n_epochs,
                        'batch_size' : batch_size,
                        'keep_prob' : keep_prob,
                        'learning_rate' : learning_rate,
                        'do_augment' : do_augment,
                        'bgval' : bgval,
                        'transl_range' : transl_range,
                        'rotate_range' : rotate_range,
                        'valid_eval_epochwise' : valid_eval_epochwise,
                        'tf_log_path' : tf_log_path,
                        'tf_save_path' : tf_save_path,
                        'train_glob_mean' : train_glob_mean,
                        'train_glob_std' : train_glob_std,
                        'n_steps_train' : n_steps_train,
                        'valid_maes' : valid_maes,
                        'logdir' : logdir,
                        'tstamp' : tstamp,
                        'save_name' : save_name
                        }
        
        if optim == 'Momentum':
            model_params['momentum'] = momentum
        
        # Saving model parameters
        save_model_params(rootpath, label_fid, model_id, tstamp, model_params)