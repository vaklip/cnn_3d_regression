# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:29:39 2019

This module defines functions for evaluating model performance.

@author: Pál Vakli (RCNS HAS BIC)
"""

############################ Importing necessary packages #####################
import numpy as np
from conv_net.load_and_save import load_data_labels_multicovars, save_results
from conv_net.preprocess import remove_nans, normalize_tensor

############################### Function definitions ##########################


def eval_multicovar(eval_set, evalind, step, eval_maes, eval_dict, eval_tfnodes, evalop, save):
    """ Evaluating peformance using multiple covariates.
    
    Evaluation of performance on the training/validation/test set
    using multiple covariates.
    
    Input:
        eval_set                String specifying the name of the to-be-evaluated
                                set.
        evalind                 Int specifying the ordinal number of evaluation.
        step                    Int specifying the training iteration step.
        maes                    Dictionary containing the result (mean absolute
                                error) of each evaluation.
        eval_dict               Dictinary containing the paramters controlling
                                evaluation.
        eval_tfnodes            Dictionary containing TensorFlow placeholders
                                used or evaluating validation set performance.
        evalop                  TensorFlow object to be evaluated to get the
                                batch prediction.
        save                    Bool specifying whether the predicted labels
                                should be saved.
    Output:
        eval_maes               Updated dictionary containing the result (mean 
                                absolute error) of each evaluation.
        evalind                 Incremented int specifying the ordinal number of 
                                evaluation.
    """
    
    # Unpacking eval_dict
    tensorlist = eval_dict['tensorlist']
    rootpath = eval_dict['rootpath']
    data_fid = eval_dict['data_fid']
    label_fid = eval_dict['label_fid']
    glob_mean = eval_dict['glob_mean']
    glob_std = eval_dict['glob_std']
    batch_size = eval_dict['batch_size']
    model_id = eval_dict['model_id']
    tstamp = eval_dict['tstamp']
    
    # Unpacking eval_tfnodes
    tf_is_train = eval_tfnodes['tf_is_train']
    tf_keep_prob = eval_tfnodes['tf_keep_prob']
    tf_batch_dataset = eval_tfnodes['tf_batch_dataset']
    tf_batch_covars = eval_tfnodes['tf_batch_covars']
        
    # Evaluating performance
    print('\nEvaluating '+eval_set+' set...\n')
    
    # Iterating over the data tensors
    for i in range(len(tensorlist)):
        
        # Loading data tensor and corresponding labels and subject IDs
        load_in = (rootpath, eval_set, tensorlist[i], data_fid, label_fid)
        eval_tensor, eval_labels, eval_covars, eval_eids = load_data_labels_multicovars(*load_in)
        
        # Preprocess data tensor, labels, and subject IDs (eids)
        eval_tensor = remove_nans(eval_tensor)
        eval_tensor = normalize_tensor(eval_tensor, glob_mean, glob_std)        
        
        # Stacks for concatenating predictions and labels
        eval_preds = np.zeros((eval_labels.shape[0], 1))
        
        # Evaluating performance
        n_steps_eval = int(eval_labels.shape[0]/batch_size)
        for estep in range(n_steps_eval):
            
            # Feedback
            print(eval_set.capitalize()+' tensor '+str(i)+' step '+str(estep)+'/'+str(n_steps_eval-1))
            
            # Calculating offset
            offset = (estep*batch_size) % eval_labels.shape[0]
            
            # Create batch    
            batch_data = eval_tensor[offset:(offset + batch_size), :, :, :, :]
            batch_covars = eval_covars[offset:(offset + batch_size), :]
            
            # Feed dictionary
            feed_dict = {tf_is_train      : False,
                         tf_keep_prob     : 1,
                         tf_batch_dataset : batch_data,
                         tf_batch_covars  : batch_covars}
            
            # Running session
            batch_preds = evalop.eval(feed_dict=feed_dict)        
            
            # Storing predictions and true labels in stack
            eval_preds[offset:(offset + batch_size), :] = batch_preds
            
        # Saving the results (optional)
        if save:
            save_results(rootpath, eval_set, tensorlist[i], label_fid, 
                         model_id, tstamp, evalind, eval_eids, eval_labels, 
                         eval_preds)
        
        # Storing predictions and true labels to compute overall validation MAE
        if i == 0:
            eval_preds_all = eval_preds
            eval_labels_all = eval_labels
        else:
            eval_preds_all = np.concatenate((eval_preds_all, eval_preds), axis=0)
            eval_labels_all = np.concatenate((eval_labels_all, eval_labels), axis=0)
            
        # Freeing up memory
        del eval_tensor, eval_labels, eval_covars, eval_eids
        
    # Grand average of validation MAEs
    eval_mae = np.mean(np.abs(np.subtract(eval_preds_all[:, 0], eval_labels_all[:, 0]))) 
    eval_maes['Epoch'+str(evalind)+'GlobalStep'+str(step)] = eval_mae
    print('\n'+eval_set.capitalize()+' MAE in epoch#'+str(evalind)+': %.2f \n' % eval_mae)
    evalind += 1
    
    return eval_maes, evalind

