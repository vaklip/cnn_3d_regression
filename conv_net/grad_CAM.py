# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:47:28 2019

This module contains a function for Gradient-weighted Class Activation Mapping
(grad-CAM) based on:

Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 
2017. Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based 
Localization. Presented at the Proceedings of the IEEE International Conference 
on Computer Vision, pp. 618–626.

@author: Pál Vakli (RCNS HAS BIC)
"""

############################ Importing necessary packages #####################
import numpy as np
import os
from scipy import ndimage
from conv_net.load_and_save import load_data_labels_multicovars
from conv_net.preprocess import remove_nans, normalize_tensor

############################### Function definitions ##########################


def gradCAM_multicovar(eval_set, evalind, eval_dict, placeholders, gradients, conv_output, session, rectify):
    """ Performing Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Function for gradient-based class activation mapping (grad-CAM) for a model
    using multiple covariates.
    
    Input:
        eval_set                String specifying the name of the to-be-evaluated
                                set.
        evalind                 Int specifying the ordinal number of evaluation.
        eval_dict               Dictinary containing the paramters controlling
                                evaluation.
        placeholders            Dictionary containing TensorFlow placeholders.
        gradients               Tensorflow Tensor object containg the gradients
                                of the scalar output of the neural network w.r.t.
                                the feature maps of the last convolutional layer.                                
        conv_output             Tensorflow Tensor object containing the feature
                                maps of the last convolutional layer of the
                                neural network.
        session                 Tensorflow Session object.
        rectify                 Boolean specifying whether the resulting heatmap
                                should be passed through a ReLU.
    Output:
        None
    """
    
    # Unpacking eval_dict
    tensorlist = eval_dict['tensorlist']
    rootpath = eval_dict['rootpath']
    data_fid = eval_dict['data_fid']
    label_fid = eval_dict['label_fid']
    glob_mean = eval_dict['glob_mean']
    glob_std = eval_dict['glob_std']
    model_id = eval_dict['model_id']
    tstamp = eval_dict['tstamp']
    
    # Unpacking eval_tfnodes
    tf_is_train = placeholders['tf_is_train']
    tf_keep_prob = placeholders['tf_keep_prob']
    tf_batch_dataset = placeholders['tf_batch_dataset']
    tf_batch_covars = placeholders['tf_batch_covars']

    # Evaluating performance
    print('\nEvaluating '+eval_set+' set...\n')

    # Iterating over the data tensors
    for i in range(len(tensorlist)):
        
        # Loading data tensor and corresponding labels and subject IDs
        load_in = (rootpath, eval_set, tensorlist[i], data_fid, label_fid)
        eval_tensor, eval_labels, eval_covars, eval_eids = load_data_labels_multicovars(*load_in)
        num_samples = eval_tensor.shape[0]
        
        # Get image shape
        if i == 0:
            img_i = eval_tensor.shape[1]
            img_j = eval_tensor.shape[2]
            img_k = eval_tensor.shape[3]
                
        # Preprocess data tensor, labels, and subject IDs (eids)
        eval_tensor = remove_nans(eval_tensor)
        eval_tensor = normalize_tensor(eval_tensor, glob_mean, glob_std)        
        
        # Preallocating memory for grad-CAM array and corresponding subject 
        # identifiers (eids) and labels
        grad_cams = np.zeros((num_samples, img_i, img_j, img_k))
        grad_cam_eids_labels = np.zeros((num_samples, 2))
        
        # Iterating over samples in the current data tensor
        n_steps_eval = eval_labels.shape[0]
        for j in range(n_steps_eval):
            
            # Feedback
            print('{} tensor {} step {}/{}'.format(eval_set.capitalize(), i, j, n_steps_eval-1))

            # Select current sample and corresponding covariates
            data = eval_tensor[np.newaxis, j, :, :, :, :]
            covars = eval_covars[np.newaxis, j, :]

            # Feed dictionary
            feed_dict = {tf_is_train      : False,
                         tf_keep_prob     : 1,
                         tf_batch_dataset : data,
                         tf_batch_covars  : covars}
            
            # Running session
            grads, conv_out = session.run([gradients, conv_output], feed_dict=feed_dict)

            # Grad-CAM
            feature_map_weights = np.mean(grads[0], axis=(1, 2, 3))
            cam = np.squeeze(np.sum(feature_map_weights*conv_out, axis=4))
            zoom_factor = (img_i/conv_out.shape[1], img_j/conv_out.shape[2], img_k/conv_out.shape[3])
            cam = ndimage.zoom(cam, zoom_factor)
            if rectify:
                cam = np.maximum(cam, 0)
            
            # Assign to Grad-CAM array
            grad_cams[j, :, :, :] = cam
            grad_cam_eids_labels[j, 0] = eval_eids[j, 0]
            grad_cam_eids_labels[j, 1] = eval_labels[j, 0]
            
        # Saving Grad-CAM array
        if rectify:
            rect = 'rect'
        else:
            rect = 'norect'
        grad_cam_fname = 'gradCAM_{}_{}_{}_{}_{}_{}_eval{}'.format(rect, eval_set, i, 
                          label_fid, model_id, tstamp, evalind)
        save_name = os.path.join(rootpath, eval_set+'_results', 'gradcam', 'gradcams', grad_cam_fname)
        np.save(save_name, grad_cams)
        print('Grad-CAMs for {} tensor {}/{} saved.'.format(eval_set.capitalize(), i, len(tensorlist)-1))
        
        # Saving corresponding labels and subject IDs (eids)
        grad_cam_eids_labels_fname = 'gradCAM_{}_eids_labels_{}_{}_{}_{}_{}_eval{}'.format(rect, eval_set, i, 
                                     label_fid, model_id, tstamp, evalind)
        save_name = os.path.join(rootpath, eval_set+'_results', 'gradcam', 'gradcams', grad_cam_eids_labels_fname)
        np.save(save_name, grad_cam_eids_labels)
        print('Grad-CAM eids and labels for {} tensor {}/{} saved.\n'.format(eval_set.capitalize(), i, len(tensorlist)-1))
        
        # Freeing up memory
        del eval_tensor, eval_labels, eval_covars, eval_eids
    
    return None