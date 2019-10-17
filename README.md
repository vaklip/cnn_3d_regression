# cnn_3d_regression

This repository contains the Tensorflow 1.13 implementation of a convolutional neural network (CNN) designed to perform regression
using 3D images as input and two additional covariates.

It is assumed that the training and validation sets consist of an arbitrary number of data tensors of shape [num_samples, i, j, k, 1] 
and corresponding label arrays of shape [num_samples, labels], where 'labels' represent the columns for subject identifiers, target labels, 
and covariates in this particular order.
An example for a training data tensor file:
    train_set/x_train_0_t1.npy
where '0' is the ordinal number of the data tensor and 't1' is the data file identifier (see 'data_fid' in Settings). The corresponding 
label array file is:
    train_set/y_train_0_t1.npy
where 't1' is the label file identifier (see 'label_fid' in Settings; this is necessary because a particular data tensor may have 
several different corresponding label arrays).
For the validation and test sets, files may be named as such:
    valid_set/x_valid_0_t1.npy
    valid_set/y_valid_0_t1.npy
    test_set/x_test_0_t1.npy
    test_set/y_test_0_t1.npy

The script in main.py controls training, testing on an independent dataset and the calculation of localization maps using 
Gradient-weighted Class Activation Mapping (Grad-CAM). The particular mode (train/test/Grad-CAM) along with the relevant parameters,
including training and network architecture hyperparameters, can be set in the "Settings" cell. The Grad-CAM method is detailed in:

    Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 
    2017. Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based 
    Localization. Presented at the Proceedings of the IEEE International 
    Conference on Computer Vision, pp. 618–626.

The repository contains dummy data for demonstration purposes. The repository also contains the model architecture and learnt weights
of the model used in: 