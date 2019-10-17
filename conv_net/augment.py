# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:39:17 2019

This module defines the classes that are suitable for 3D image data augmentation.

@author: Pál Vakli (RCNS HAS BIC)
"""

############################ Importing necessary packages #####################
import numpy as np
from scipy import ndimage

################################ Class definitions ############################

    
class ImgManipAdj():
    """ A class used for manipulating 3D images.
    
    A class used for randomly translating and rotating a batch of 3D images. 
    The values of points outside the boundaries of the input image are adjusted 
    based on the original background value (i.e. before normalization) and the 
    global mean and std that have been used to normalize the data.
    
    Attributes:
        transl_range        Float specifying the range of random translations 
                            of the 3D image. The range of translations is 
                            [-transl_range transl_range]
        rotate_range        Float specifying the range of rotations of the 3D 
                            image in degrees. The range of rotations is
                            [-rotate_range rotate_range]
        bgval               Float specifying the value of the background
        glob_mean           Float specifying the mean across all data tensors
                            corresponding to the given set (global mean)
        glob_std            Float specifying the standard deviation across all 
                            data tensors corresponding to the given set 
                            (global standard deviation)
    Methods:
        translate           Translates the 3D image
        rotate              Rotates the 3D image
    """
    
    def __init__(self, transl_range, rotate_range, bgval, glob_mean, glob_std):
        """ Parameters
        
        transl_range        Float specifying the range of random translations 
                            of the 3D image. The range of translations is 
                            [-transl_range transl_range]
        rotate_range        Float specifying the range of rotations of the 3D 
                            image in degrees.The range of rotations is
                            [-rotate_range rotate_range]
        bgval               Float specifying the value of the background
        glob_mean           Float specifying the mean across all data tensors
                            corresponding to the given set (global mean)
        glob_std            Float specifying the standard deviation across all 
                            data tensors corresponding to the given set 
                            (global standard deviation)
        """
        
        self.transl_range = transl_range
        self.rotate_range = rotate_range
        self.bgval = bgval
        self.glob_mean = glob_mean
        self.glob_std = glob_std
        
    def translate(self, img_batch):
        """ Randomly translates the image within the given range.
        
        The magnitude of translation along each axis is specified independently
        by selecting a value randomly from: [-transl_range transl_range].
        
        Input:
            img_batch     4D Numpy array of a batch of 3D images. Shape:
                          (batch_size, length_i, length_j, length_k, 1)
        Output:
            img_trl       4D Numpy array of a batch of randomly translated 3D 
                          images. Shape:
                          (batch_size, length_i, length_j, length_k, 1)
        
        """
        
        img_trl = np.zeros((img_batch.shape))
        cval = (self.bgval-self.glob_mean)/self.glob_std
        for i in range(img_batch.shape[0]):
            shift_x = np.random.randint(-self.transl_range, self.transl_range)
            shift_y = np.random.randint(-self.transl_range, self.transl_range)
            shift_z = np.random.randint(-self.transl_range, self.transl_range)
            img_trl[i, :, :, :, 0] = ndimage.shift(img_batch[i, :, :, :, 0], 
                                                   [shift_y, shift_x, shift_z],
                                                   cval=cval)
        return img_trl
        
    def rotate(self, img_batch):
        """ Randomly rotates the image within the given range.
        
        The degree of rotation in each cardinal plane (x-y, x-z, y-z) is 
        specified independently by selecting a value randomly from: 
        [-rotate_range rotate_range].
        
        Input:
            img_batch     4D Numpy array of a batch of 3D images. Shape:
                          (batch_size, length_i, length_j, length_k, 1)
        Output:
            img_rot       4D Numpy array of a batch of randomly rotated 3D 
                          images. Shape:
                          (batch_size, length_i, length_j, length_k, 1)
        """
        
        img_rot = np.zeros((img_batch.shape))
        axes_list = [(0, 1), (0, 2), (1, 2)]
        cval = (self.bgval-self.glob_mean)/self.glob_std
        for i in range(img_batch.shape[0]):
            for j in range(3):
                deg = np.random.randint(-self.rotate_range, self.rotate_range)
                img_rot[i, :, :, :, 0] = ndimage.rotate(img_batch[i, :, :, :, 0], 
                                                        deg, 
                                                        reshape=False, 
                                                        axes=axes_list[j],
                                                        cval=cval)
        return img_rot