# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 13:10:23 2017

@author: James Topor

This python script was written to provide a relatively simple method
of prepping a subset of the color images contained within the CASIA WebFace data 
set for use with Tensorflow v1.0 in python v 3.52 on a Windows 10 system. 
The CASIA data set is very large, containing close to 500,000 250x250 images of
10,575 individuals. Since most Windows users will lack sufficient
RAM resources for use of the full CASIA WF dataset, the algorithm used here
allows the user to specify the number of images to be prepped for use with Tensorflow
as well as the preferred image size to be used (implemented via scaling).

The algorithm assumes that the user has already downloaded and un-tarred the CASIA
WF data set after having received access to it from the Center for Biometrics and
Security Research in Beijing. If the data set has been properly downloaded and
un-tarred, the user should then find a local directory of directories containing
the uncompressed images. Each directory of the data set contains images pertaining
to a single individual, and the images are a mix of color and monochrome. The base
directory will also contain a 'names.txt' file that provides the names of 
the individuals pertaining to each of the numbered directories containing the
CASIA images.

Given a specified maximum number of images, the algorithm will load that number of
images in sequential order starting with the first directory. As such, if you
wish to make use of a sample of images from all 10,575 individuals, you will need to 
modify this code.  The "size" variable allows the use to specify their preferred
dimensionality for the images: since the raw CASIA images are 250x250, they may
prove to be too large for practical use on a Windows system. As such, the user can 
specify 128x128 or 64x64, etc. The algorithm will automatically rescale each image
accordingly. The algorithm will also discard all monochrome images (they are a 
minority of the CASIA data set). If monochrome images are preferred, you will
need to modify the code.

The algorithm automatically compiles the images into an (n, size(1), size(2), 3)
array and also creates an array of the corresponding labels for the images.

Finally, the user can specify whether the data are to be pre-split into training
and testing subsets via specification of the "test_size" variable. By default the
variable is set to test_size = 0.2, which sets aside 20% of the processed images
for testing. If no training/testing split is desired prior to loading the data
set into your Tensorflow application, set test_size = 0.0.

The resulting training/testing arrays are then written to disk using python's
pickle() function. Specifically, a dict object is created containing the images,
labels, and number of classes for the data set(s). The resulting file(s) can
then be easily loaded into your own Tensorflow application as follows:
    
# load contents of pickle file into a python object
with open('c:/tmp/casia/casia_training.pickle', 'rb') as handle:
    b = pickle.load(handle)

# to access images + labels + number of classes in dict object:
images = b["Images"]
labels = b["Labels"]
num_classes = b["NumClasses"]
    
"""

import numpy as np
import pandas as pd
import pickle
import glob
import time

from scipy.misc import imread
from scipy.misc import imresize

# Set the desired size of the CASIA images for your application
size = 64,64

# define a constant tuple that matches the anticipated size of the CASIA
# images after scaling. If you are using color images, be sure to include a 3
# as the third element of the tuple. If you are using B/W images, omit the 
# third element. The tuple is named RGB here since this algorithm assumes 
# that the images to be used are RGB color images
RGB = 64,64,3

# set the maximum number of images to load. The CASIA data set has more than
# 494,000 images so you may need to limit the number used depending on your 
# local hw/sw envirobnment for performance reasons. Use '-1' due to Python;s
# 0-based indexing scheme
max_images = 50000 - 1

# set % of files to use for neural network testing
test_size = 0.2

# get count of directories => this will be the number of classes
# Be sure to set your local path accordingly !!
d_count = len(glob.glob("c:/tmp/casia/Casia-webface/*") )

# open names.txt and get read into a data frame
# Be sure to set your local path accordingly !!
names_df = pd.read_csv("c:/tmp/casia/names.txt", sep = ' ', header = None,
                       names = ['DirName', 'Who'], dtype = str)

# truncate names_df at d_count - 1 rows since that's how much was downloaded
names_df = names_df[:d_count-1]

# init an empty array - will be filled with images
im_arr = []

# init empty array to store classifier ID's
class_arr = []

# initialize a counter for the number of images processed so far
total_imgs = -1

# loop through the available directories and read in images
for i in range(0, d_count-1) :
     if total_imgs == max_images :
        break
     else:
        # get next dir name from data frame
        # Be sure to set your local path accordingly !!
        f_path = "c:/tmp/casia/Casia-Webface/" + names_df["DirName"][i]
        
        # get count of files in the directory
        f_count = len(glob.glob(f_path + "/*") )
        
        # fetch the list of file names from the directory
        f_list = glob.glob(f_path + "/*")
        
        # add each file to the array of images; classifier will be whatever i currently is
        for file in range(1, f_count) :
            # check for max_images reached
            if total_imgs == max_images :
                break
            else:
                # get raw image
                i_raw = imread(f_list[file])
                
                # resize image from 250x250 to 'size' and convert to floating point
                # NOTE: floating point format is MANDATORY for Tensorflow
                i_res = imresize(i_raw, size).astype('float32')
                
                # test image to ensure it is RGB and not B/W: if B/W ignore it
                # if color, add to python images + classes arrays
                if i_res.shape == RGB :
                    total_imgs += 1
                    im_arr.append( i_res )
                    class_arr.append(i)
                
# cleanup memory: Free up memory so that remaining functions will run OK
del names_df, i_raw, i_res, f_list, f_path
# pause to allow memory cleanup to be completed
time.sleep(5)

################# Now use SKLearn to split into training + test sets ##########

# if a testing data set is desired, use train_test_split() function to divvy up data
# between training at test sets via random selection across the ordered
# images + class identifiers
if test_size != 0 :
    
    from sklearn.model_selection import train_test_split
    
    Training_imgs, Testing_imgs, Training_lbls, Testing_lbls = train_test_split(im_arr, 
                                                        class_arr, test_size = test_size,
                                                        random_state = 42)  

    # cleanup memory
    del im_arr, class_arr
    # pause to allow memory cleanup to be completed
    time.sleep(5)

    # convert testing lists to numpy arrays and store in a dict
    test_dict_obj = {"Images": np.array(Testing_imgs), 
                     "Labels": np.array(Testing_lbls),
                     "NumClasses": i}
    
    # remove arrays from memory since they are no longer needed
    del Testing_imgs, Testing_lbls
     # pause to allow memory cleanup to be completed
    time.sleep(5)

    
    #  write testing dict objects to disk
    with open('c:/tmp/casia/casia_testing.pickle', 'wb') as handle:
        pickle.dump(test_dict_obj, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    # remove test data dict object from memory
    del test_dict_obj
     # pause to allow memory cleanup to be completed
    time.sleep(5)

    # convert both training lists to  numpy arrays and store in a dict
    train_dict_obj = {"Images": np.array(Training_imgs), 
                      "Labels": np.array(Training_lbls),
                      "NumClasses": i}
    
    # remove arrays from memory since they are no longer needed
    del Training_imgs, Training_lbls
     # pause to allow memory cleanup to be completed
    time.sleep(5)

    #  write dict objects to disk
    with open('c:/tmp/casia/casia_training.pickle', 'wb') as handle:
        pickle.dump(train_dict_obj, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    del train_dict_obj
    # pause to allow memory cleanup to be completed
    time.sleep(5)


# else no testing set is required: all data will be used for training
else:
    # convert both training lists numpy arrays and store in a dict
    train_dict_obj = {"Images": np.array(im_arr), 
                      "Labels": np.array(class_arr),
                      "NumClasses": i}
    
    del im_arr, class_arr
    # pause to allow memory cleanup to be completed
    time.sleep(5)
    
    #  write dict objects to disk
    with open('c:/tmp/casia/casia_training.pickle', 'wb') as handle:
        pickle.dump(train_dict_obj, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    del train_dict_obj   


#############################################################################
''' how to load dict object from disk in your Tensorflow program

# load contents of pickle file into a python object
with open('c:/tmp/casia/casia_training.pickle', 'rb') as handle:
    b = pickle.load(handle)

# to access images + labels in dict object
images = b["Images"]
labels = b["Labels"]
num_classes = b["NumClasses"]

# get first image from array
tp = images[0,0:]

toimage(tp)

# get class of first image
tp_class = classes[0]

'''
