# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:22:30 2019

This script computes the mean (glob_mean) and standard deviation (glob_std)
across all the data tensors corresponding to the same set. The mean intensity 
of each data tensor is calculated and then these means are averaged to obtain 
glob_mean. THe variance of each data tensor is calculated and then these values
are averaged. The square root of this average is the value of glob_std.

@author: Pál Vakli (RCNS-HAS_BIS)
"""

#%% ################################# Settings ################################
set_name = 'test'
data_fid = 't1'

#%% ####################### Get normalization paramters #######################

# Importing necessary libraries
import os
import re
import numpy as np
import pickle

# Get a list of data tensors
file_list = os.listdir(set_name+'_set')
regexp = re.compile('x_'+set_name+'_.*_'+data_fid+'.npy')
tensor_list = [f for f in file_list if re.match(regexp, f)]

# Loading data tensors and computing mean and variance
means = np.zeros((len(tensor_list),))
varis = np.zeros((len(tensor_list),))
nans = np.zeros((len(tensor_list),))

for i in range(len(tensor_list)):
    # Loading data tensor
    data_tensor = np.load(os.path.join(set_name+'_set', tensor_list[i]))
    # Get mean and variance of the data tensor
    means[i] = np.nanmean(data_tensor)
    varis[i] = np.nanvar(data_tensor)
    # Get number of NaNs in the data tensor
    nans[i] = np.count_nonzero(np.isnan(data_tensor))
    # Feedback
    print('Finished with '+tensor_list[i]+'.')
    
# Compute global mean and standard error
glob_mean = np.mean(means)
glob_std = np.sqrt(np.mean(varis))

# Printing the numbers of NaNs
print('Numbers of NaNs in the data tensors:')
print(nans)

# Save
save_name = os.path.join(set_name+'_set', 'norm_params_x_'+set_name+'_'+data_fid+'.pickle')
try:
    f = open(save_name, 'wb')
    norm_params = {
                   'glob_mean': glob_mean,
                   'glob_std': glob_std
                   }
    pickle.dump(norm_params, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print('Normalization parameters saved.')
except Exception as e:
    print('Unable to save normalization parameters to ', save_name, ':', e)
    raise