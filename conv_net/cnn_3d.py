# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 08:59:05 2019

This module defines functions for building a 3D CNN.

@author: Pál Vakli (RCNS HAS BIC)
"""

############################ Importing necessary packages #####################
import tensorflow as tf

############################### Function definitions ##########################


def conv_block(data, is_train, params_dict, max_pooling):
    """ Building a convolution block consisting of 2 subsequent spattialy 
    separated convolution blocks with ReLU and dropout and an optional max pooling
    at the end of the convolution block.
    
    """
    
    # Unpacking convolutional block paramters
    input_chans = params_dict['input_chans']
    kernel_shapes = params_dict['kernel_shapes']
    conv_paddings = params_dict['conv_paddings']
    conv_strides = params_dict['conv_strides']
    conv_chans = params_dict['conv_chans']
    if max_pooling:
        pool_shape = params_dict['pool_shape']
        pool_stride = params_dict['pool_stride']
        
    # Weights for the first spatially separated convolution
    w11 = tf.get_variable("w11", shape=[kernel_shapes[0], 1, 1, input_chans, conv_chans[0]],
                                       initializer=tf.contrib.layers.xavier_initializer())
    w12 = tf.get_variable("w12", shape=[1, kernel_shapes[0], 1, conv_chans[0], conv_chans[0]],
                                       initializer=tf.contrib.layers.xavier_initializer())
    w13 = tf.get_variable("w13", shape=[1, 1, kernel_shapes[0], conv_chans[0], conv_chans[0]],
                                       initializer=tf.contrib.layers.xavier_initializer())
    
    # Weights for the second spatially separated convolution
    w21 = tf.get_variable("w21", shape=[kernel_shapes[1], 1, 1, conv_chans[0], conv_chans[1]],
                                       initializer=tf.contrib.layers.xavier_initializer())
    w22 = tf.get_variable("w22", shape=[1, kernel_shapes[1], 1, conv_chans[1], conv_chans[1]],
                                       initializer=tf.contrib.layers.xavier_initializer())
    w23 = tf.get_variable("w23", shape=[1, 1, kernel_shapes[1], conv_chans[1], conv_chans[1]],
                                       initializer=tf.contrib.layers.xavier_initializer())
    
    # Spatially separated convolution #1 + batch normalization + ReLU
    cstride = conv_strides[0]
    padding = conv_paddings[0]
    conv = tf.nn.conv3d(data, w11, [1, cstride, cstride, cstride, 1], padding=padding)
    conv = tf.nn.conv3d(conv, w12, [1, cstride, cstride, cstride, 1], padding=padding)
    conv = tf.nn.conv3d(conv, w13, [1, cstride, cstride, cstride, 1], padding=padding)
    norm = tf.layers.batch_normalization(conv, training=is_train)
    acti = tf.nn.relu(norm)
    
    # Spatially separated convolution #2 + batch normalization + ReLU
    cstride = conv_strides[1]
    padding = conv_paddings[1]
    conv = tf.nn.conv3d(acti, w21, [1, cstride, cstride, cstride, 1], padding=padding)
    conv = tf.nn.conv3d(conv, w22, [1, cstride, cstride, cstride, 1], padding=padding)
    conv = tf.nn.conv3d(conv, w23, [1, cstride, cstride, cstride, 1], padding=padding)
    norm = tf.layers.batch_normalization(conv, training=is_train)
    acti = tf.nn.relu(norm)
    
    # Max pooling (optional)
    if max_pooling:
        return tf.layers.max_pooling3d(acti,
                                       pool_size=(pool_shape, pool_shape, pool_shape),
                                       strides=(pool_stride, pool_stride, pool_stride),
                                       padding='VALID')
    else:
        return acti
    

def fc_block(data, input_size, nneurons, keep_prob):
    """ Building a fully connected hidden layer.
    
    """
    wfc = tf.get_variable("wfc",
                          shape=[input_size, nneurons],
                          initializer=tf.contrib.layers.xavier_initializer())
    bfc = tf.get_variable("bfc", initializer=tf.add(tf.zeros([nneurons], dtype=tf.float32), 0.01))
    return tf.nn.dropout(tf.nn.relu(tf.matmul(data, wfc)+bfc), keep_prob)


def cnn_3d_multicovar(data, covars, is_train, keep_prob, architecture_dict):
    """ 3D convolutional neural network.
    
    3D convolutional neural network consisting of a chain of convolution blocks 
    and subsequent fully connected layers. Each convolution block includes 2 
    spatially separated convolutions with batch normalization and ReLU activation
    function and max pooling at the end of the block. The last convolution block
    ends in global average pooling instead of max pooling. Multiple covariates are 
    concatenated to the output of global average pooling which is then fed into 
    one or several subsequent fully connected hidden layers with ReLU and dropout 
    and a final output layer.
    
    Input:
        data                TensorFlow float32 placeholder for batch data of shape
                            [batch_size, img_size_i, img_size_j, img_size_k, n_channels].
        covars              TensorFlow float32 placheolder for batch covariates 
                            of shape [None, num_covars].
        is_train            TensorFlow boolean placeholder. True indicates that
                            the network is being trained.
        keep_prob           Tensorflow float32 placeholder specifying the keeping
                            probability for dropout in the fully connected layers.
        architecture_dict   Python dictionary containing the hyperparameters that
                            specify the architecture of the neural network.
    Output:
        data_after_conv     Value of the output of the fully connected output layer.
        data                Value of the output of last convolutional layer.
        
    """
    # Unpacking architecture parameters
    kernel_shapes = architecture_dict['kernel_shapes']
    conv_paddings = architecture_dict['conv_paddings'] 
    conv_strides = architecture_dict['conv_strides']  
    conv_chans = architecture_dict['conv_chans'] 
    max_poolings = architecture_dict['max_poolings'] 
    pool_strides = architecture_dict['pool_strides'] 
    fc_nneurons = architecture_dict['fc_nneurons'] 
    num_covars = architecture_dict['num_covars']
    
    # Building a chain of convolutional blocks
    for i in range(len(kernel_shapes)):
        
        if i == 0:
            input_chans = architecture_dict['input_chans']
        else:
            input_chans = conv_chans[i-1][1]
            
        params_dict = {
                      'input_chans'   : input_chans,
                      'kernel_shapes' : kernel_shapes[i],
                      'conv_paddings' : conv_paddings[i],
                      'conv_strides'  : conv_strides[i],
                      'conv_chans'    : conv_chans[i],
                      }
        
        if i < len(kernel_shapes)-1:
            max_pooling = True
            params_dict['pool_shape'] = max_poolings[i]
            params_dict['pool_stride'] = pool_strides[i]
        else:
            max_pooling = False
            
        with tf.variable_scope("CONV_BLOCK"+str(i+1), reuse=tf.AUTO_REUSE):
            data = conv_block(data, is_train, params_dict, max_pooling=max_pooling)
    
    # Global average pooling
    data_after_conv = tf.reduce_mean(data, axis=[1, 2, 3])
    data_after_conv = tf.concat([data_after_conv, covars], axis=1)
    
    # Building a chain of fully connected layers
    Li = i+2 # layer index
    
    if isinstance(fc_nneurons, int):
        input_size = conv_chans[-1][1]+num_covars
        n_weights_output = fc_nneurons
        with tf.variable_scope("FC"+str(Li), reuse=tf.AUTO_REUSE):
                data_after_conv = fc_block(data_after_conv, input_size, fc_nneurons, keep_prob)
    else:
        n_weights_output = fc_nneurons[-1]
        for i in range(len(fc_nneurons)):
            
            if i == 0:
                input_size = conv_chans[-1][1]+num_covars
            else:
                input_size = fc_nneurons[i-1]
            
            with tf.variable_scope("FC"+str(Li), reuse=tf.AUTO_REUSE):
                data_after_conv = fc_block(data_after_conv, input_size, fc_nneurons[i], keep_prob)
            
            Li += 1
    
    # Output layer
    with tf.variable_scope("OUTPUT", reuse=tf.AUTO_REUSE):
        wout = tf.get_variable("wout",
                               shape=[n_weights_output, 1],
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.get_variable("bout", initializer=tf.add(tf.zeros([1], dtype=tf.float32), 0.01))
        data_after_conv = tf.matmul(data_after_conv, wout) + bout
    
    return (data_after_conv, data)