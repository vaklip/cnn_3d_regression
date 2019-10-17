# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:41:40 2019

This module defines functions for preprocessing 5D data tensors.

@author: Pál Vakli (RCNS HAS BIC)
"""

############################ Importing necessary packages #####################
import numpy as np

############################### Function definitions ##########################


def remove_nans(data_tensor):
    """ Removing NaNs from data tensor.
    
    Replacing each NaN element of the data tensor with 0.
    
    Input:
        data_tensor     5D Numpy array holding the data tensor of shape:
                        (subjects*matrix_size*matrix_size*matrix_size*1)
    Output:
        data_tensor     5D Numpy array holding the data tensor with NaNs 
                        replaced with 0s. Shape is unchanged:
                        (subjects*matrix_size*matrix_size*matrix_size*1)
    """
    
    data_tensor[np.isnan(data_tensor)] = 0
    
    return data_tensor
    

def normalize_tensor(data_tensor, glob_mean, glob_std):
    """ Normalizing data tensor.
    
    Normalizing data tensor by subtracting the global mean and dividing by the
    global standard deviation elementwise. Global mean and standard deviation
    describe values in all the data tensors corresponding to the same set.
    
    Input:
        data_tensor     5D Numpy array holding the data tensor of shape:
                        (subjects*matrix_size*matrix_size*matrix_size*1)
        glob_mean       Float specifying the global mean.
        glod_std        Float specifying the global standard deviation.
    Output:
        data_tensor     5D Numpy array holding the normalized data tensor:
                        (subjects*matrix_size*matrix_size*matrix_size*1)
    """
    
    data_tensor = np.divide(np.subtract(data_tensor, glob_mean), glob_std)
    
    return data_tensor


def randomize_tensor_covar(data_tensor, labels, covars, eids):
    """ Randomizing instances and labels.
    
    Randomly reordering instances in the data tensor and the corresponding 
    labels and covariates. 
    
    Input:
        data_tensor             5D Numpy array holding the data tensor of shape:
                                (subjects*matrix_size*matrix_size*matrix_size*1)
        labels                  1D/2D Numpy array holding the labels corresponding 
                                to the data tensor.
        covars                  1D or 2D Numpy array holding the covariates.
        eids                    1D Numpy array holding the subject identifiers.
    Output:
        shuffled_data_tensor    5D Numpy array holding the data tensor with
                                instances permuted along the first (0th)
                                dimension.
        shuffled_labels         1D/2D Numpy array holding the labels of instances 
                                in the shuffled data tensor.
        shuffled_covars         1D or 2D Numpy array holding the covariates.
        shuffled_eids           1D Numpy array holding the subject identifiers
                                in the shuffled data tensor.
    """

    permutation = np.random.permutation(labels.shape[0])
    shuffled_data_tensor = data_tensor[permutation, :, :, :, :]
    shuffled_labels = labels[permutation, :]
    shuffled_covars = covars[permutation, :]
    shuffled_eids = eids[permutation, :]
    
    return (shuffled_data_tensor, shuffled_labels, shuffled_covars, shuffled_eids)


def create_batch_positions(n_samples, batch_size):
    """ Specifies the position of each batch in a data tensor.
    
    Creates an array containing the slicing indices of each batch for a data 
    tensor consisting of 'n_samples' samples. The first column specifies the 
    starting position of each batch (included), the second specifies the end 
    positions (excluded).
    
    Input:
        n_samples               Int specifying the number of samples in the data
                                tensor.
        batch_size              Int specifying the number of samples in a batch.
    Output:
        batch_pos               2D Numpy array including the starting (1st col)
                                and ending (2nd col) slice indices.
        batch_ind               Int initialized to 0 (batch index in batch_pos).
    """
    
    batch_starts = np.array(range(0, n_samples-batch_size, batch_size)).reshape(-1, 1)
    batch_ends = batch_starts+batch_size
    batch_pos = np.concatenate((batch_starts, batch_ends), axis=1)
    batch_ind = 0
    
    return (batch_pos, batch_ind)


def create_batch_covar(data_tensor, labels, covars, batch_pos, batch_ind):
    """ Creates a batch of data and labels and covariates.
    
    Input:
        data_tensor             5D Numpy array holding the data tensor of shape:
                                (subjects*matrix_size*matrix_size*matrix_size*1)
        labels                  1D/2D Numpy array holding the labels corresponding 
                                to the data tensor.
        covars                  1D Numpy array holding the covariates corresponding 
                                to the data tensor.
        batch_pos               2D Numpy array including the starting (1st col)
                                and ending (2nd col) slice indices.
        batch_ind               Int specifying the batch index in batch_pos.
    Output:
        batch_data              5D Numpy array holding the batch data of shape:
                                (subjects*matrix_size*matrix_size*matrix_size*1)
        batch_labels            1D/2D Numpy array holding the labels corresponding 
                                to the batch data tensor.
        batch_covars            1D or 2D Numpy array holding the covariates  
                                corresponding to the batch data tensor.                        
        batch_ind               Int specifying the next batch index in batch_pos 
                                (inputted index incremented by one).
    """
    
    batch_start = batch_pos[batch_ind, 0]
    batch_end = batch_pos[batch_ind, 1]
    batch_data = data_tensor[batch_start:batch_end, :, :, :, :]
    batch_labels = labels[batch_start:batch_end, :]
    batch_covars = covars[batch_start:batch_end, :]
    batch_ind += 1
    
    return batch_data, batch_labels, batch_covars, batch_ind

    