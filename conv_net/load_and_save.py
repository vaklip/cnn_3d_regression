# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:01:26 2019

This module defines functions for loading data tensors and corresponding labels.

@author: Pál Vakli (RCNS HAS BIC)
"""

############################ Importing necessary packages #####################
import re
from random import shuffle
import numpy as np
import os
from six.moves import cPickle as pickle

############################### Function definitions ##########################


def gen_tensor_list(rootpath, set_name, train_label_fid):
    """ Generating a list of tensors numbers.
    
    Data tensors and corresponding label arrays are likely partitioned into two
    or more subarrays, each included in different files. This function collects
    the numbers of all files belonging to the ID specified by train_label_fid.
    
    Input:
        rootpath            String specifying the path to the root directory of
                            the project.
        set_name            String specifying the set ('train', 'valid', or
                            'test') to which the data tensor, eids, and labels
                            belong.
        train_data_fid      String specifying the ID of the file containing
                            the training data (file names are inferred from 
                            this).
    Output:
        tensor_list         List containing the numbers of the files specified
                            by train_label_fid.
    """
    
    # Getting the list of files in the set folder
    set_folder = set_name+'_set'
    file_list = os.listdir(os.path.join(rootpath, set_folder))
    
    # Regex
    regex_label = re.compile('y_'+set_name+'_.*_'+train_label_fid+'.npy')
    
    # List of data and label files
    label_files = [f for f in file_list if re.match(regex_label, f)]
    
    # List of tensor numbers shuffled
    tensor_list = list(range(len(label_files)))
    shuffle(tensor_list)    
        
    return tensor_list
    

def load_data_labels_multicovars(rootpath, set_name, tensor_num, train_data_fid, 
                                 train_label_fid):
    """ Loading data tensors and labels.
    
    Loading the data tensor and corresponding subject identifiers and labels
    and multiple covariates.
    
    Input:
        rootpath            String specifying the path to the root directory of
                            the project.
        set_name            String specifying the set ('train', 'valid', or
                            'test') to which the data tensor, eids, and labels
                            belong.
        tensor_num          Int specifying the number of the data tensor.
        train_data_fid      String specifying the ID of the file containing
                            the training data (file names for train, valid and
                            test are inferred from this).
        train_label_fid     String specifying the name of the file containing
                            labels for the training data (file names for train, 
                            valid, and test are inferred from this).
    Output:
        data_tensor         5D Numpy array holding the data tensor of shape:
                            (subjects*matrix_size*matrix_size*matrix_size*1)
        labels              1D Numpy array holding the subject labels.
        covars              2D Numpy array holding the subject covariates.
        eids                1D Numpy array holding the subject identifiers.
    """
    
    # Setting data tensor and label array file names and folders
    set_folder = set_name+'_set'
    data_fname = 'x_'+set_name+'_'+str(tensor_num)+'_'+train_data_fid+'.npy'
    label_fname = 'y_'+set_name+'_'+str(tensor_num)+'_'+train_label_fid+'.npy'
        
    # Loading data tensor
    data_tensor = np.load(os.path.join(rootpath, set_folder, data_fname))
    data_tensor = data_tensor.astype(np.float32)
        
    # Loading labels
    label_array = np.load(os.path.join(rootpath, set_folder, label_fname))
    eids = np.reshape(label_array[:, 0], (-1, 1))
    labels = np.reshape(label_array[:, 1].astype(np.float32), (-1, 1))
    covars = label_array[:, 2:].astype(np.float32)
    
    return (data_tensor, labels, covars, eids)
       

def save_results(rootpath, set_name, tensor_num, train_label_fid, model_id, 
                 tstamp, valid_evalind, eids, labels, preds):
    """ Saving results.
    
    Saving an array containing subject IDs (eids, 1st column), true labels (2nd
    column), and predicted labels (3rd column).
    
    Input:
        rootpath            String specifying the path to the root directory of
                            the project.
        set_name            String specifying the set ('train', 'valid', or
                            'test') to which the data tensor, eids, and labels
                            belong.
        tensor_num          Int specifying the number of the data tensor.
        train_label_fid     String specifying the ID of the file containing
                            labels for the training data. The results file name 
                            is inferred from this.
        model_id            String specifying the ID of the trained model.
        tstamp              String specifying the time stamp of training.
        valid_evalind       Int specifying the ordinal number of evaluation.
        eids                1D Numpy array holding the subject identifiers.
        labels              1D/2D Numpy array holding the (true) subject labels.
        preds               1D/2D Numpy array holding the predicted labels.
    Output:
        None
    """
    
    # Stacking subject IDs (eids), true labels (labels) and predicted labels
    # (preds) into an array of three consecutive columns
    results = np.column_stack((eids, labels, preds))
    # Setting the results file name
    set_folder = set_name+'_results'
    results_fname = 'r_'+set_name+'_'+str(tensor_num)+'_'+train_label_fid+'_'+\
                    model_id+'_'+tstamp+'_eval'+str(valid_evalind)+'.npy'
    # Saving results
    np.save(os.path.join(rootpath, set_folder, results_fname), results)
    print('\n'+set_name.capitalize()+' results saved.')
    
    return None


def save_model_params(rootpath, train_label_fid, model_id, tstamp, params):
    """ Saving model parameters.
    
    Saving the model parameters into a pickle file.
    
    Input:
        rootpath            String specifying the path to the root directory of
                            the project.
        train_label_fid     String specifying the ID of the file containing
                            labels for the training data. The params file name 
                            is inferred from this.
        model_id            String specifying the ID of the trained model.
        tstamp              String specifying the time stamp of training.
        params              Dict containing the model parameters.
    Output:
        None
    """
    
    # Setting the names of the files containing the parameters and weights
    params_fname = 'p_train_'+train_label_fid+'_'+model_id+'_'+tstamp+'.pickle'
    # Saving model parameters
    try:
        f = open(os.path.join(rootpath, 'model_parameters', params_fname), 'wb')
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print('Model parameters saved.')
    except Exception as e:
        print('Unable to save parameters to ', params_fname, ':', e)
        raise
        
    return None


def load_model_params(rootpath, label_fid, model_id, tstamp):
    """ Loading model parameters.
    
    Loading the model parameters from a pickle file.
    
    Input:
        rootpath            String specifying the path to the root directory of
                            the project.
        label_fid           String specifying the ID of the file containing
                            labels for the data. The params file name 
                            is inferred from this.
        model_id            String specifying the ID of the trained model.
        tstamp              String specifying the time stamp of training.        
    Output:
        model_params        Dict containing the model parameters.
    """
    
    # Setting the name of the file containing the model parameters
    model_params_fname = 'p_train_'+label_fid+'_'+model_id+'_'+tstamp+'.pickle'
    # Loading model parameters from the file
    model_params_fpath = os.path.join(rootpath, 'model_parameters', model_params_fname)
    with open(model_params_fpath, 'rb') as f:
        model_params = pickle.load(f)

    return model_params
