# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:05:24 2017

@author: James T
"""
# 03.27.2017
# Added a dropout layer after each maxpool + added a second dense layer with a 
# droput layer between dense1 and dense 2. Also added a dropout layer before
# softmax layer per various comments from CNN users
#
# This model achieves approx 93% accuracy on the training set and 75% accuracy
# on the evaluation data set after 25,300 steps
#
# The code herein is based largely on the Tensorflow TF Layers tutorial 
# accessible at https://www.tensorflow.org/tutorials/layers
# That tutorial builds a CNN to evaluate MNIST images. The approach used
# in the tutorial is modified here to enable the classification of CIFAR-10 
# color (RGB) images.
#
# To use this code, the CIFAR-10 data set must first be downloaded and unpacked 
# within your local environment. The data set is available as a TAR file here:
# https://www.cs.toronto.edu/~kriz/cifar.html
#
# On that page, scroll down to the "Download" section and select "CIFAR-10 python version"
# The download will start automatically. When download is complete, you must
# then unpack the resulting TAR file and modify the calls to the "unpickle()" function
# herein so that the unpacked files can be loaded into Python.
#
# Also, please note that the CIFAR data set when unpacked is comprised of 6 
# separate files, each of which contains 10,000 images. 5 of the files are meant 
# to be used for model training while the sixth is meant for evaluation. As you
# will see below, there are 6 separate calls to the "unpickle()" function used
# to load the various CIFAR data files.

"""Convolutional Neural Network Estimator for CIFAR-10, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import _pickle as cPickle

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###########################################################

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # CIFAR-10 images are 32x32 pixels, and have RGB color channel
  input_layer = tf.convert_to_tensor(features)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 3]
  # Output Tensor Shape: [batch_size, 32, 32, 32]
  # NOTE: tf.layers.conv2d input MUST be floating point for some reason???
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  conv2 = tf.layers.conv2d(
      inputs= conv1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 16, 16, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Add dropout operation; 0.75 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=pool1, rate=0.25, training=mode == learn.ModeKeys.TRAIN)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 32]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  conv3 = tf.layers.conv2d(
      inputs=dropout1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  conv4 = tf.layers.conv2d(
      inputs= conv3,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 8, 8, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  
  # Add dropout operation; 0.75 probability that element will be kept
  dropout2 = tf.layers.dropout(
      inputs=pool2, rate=0.25, training=mode == learn.ModeKeys.TRAIN)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 8, 8, 64]
  # Output Tensor Shape: [batch_size, 4096]
  pool2_flat = tf.reshape(dropout2, [-1, 8 * 8 * 64])

  # Dense Layer
  # Densely connected layer with 512 neurons
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 512]
  dense1 = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout3 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
  
  # second fully connected layer = use 256 features
  dense2 = tf.layers.dense(inputs=dropout3, units=256, activation=tf.nn.relu)
  
   # Add dropout operation; 0.6 probability that element will be kept
  dropout4 = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 256]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout4, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  # !!! Make sure depth variable is set properly relative to data !!!!
  # For CIFAR-10 there are 10 classes so depth = 10
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.0001,
        optimizer="Adam")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

########################################################################
# CIFAR-10 data loader

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict

#########################################################################


def main(unused_argv):
  # d = unpickle(os.path.join(os.path.expanduser('~/Dropbox/data/CIFAR-10/cifar-10-batches-py/'), file_name))
  d = unpickle('C:/SQLData/661/CIFAR10/data_batch_1')
  # now extract image data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  data = d['data']
  data = data.astype('float32')
  train_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
  train_labels = np.asarray(d['labels'], dtype=np.int32)

  # ---- add second batch to training data
  d = unpickle('C:/SQLData/661/CIFAR10/data_batch_2')
  # now extract image data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  data = d['data']
  data = data.astype('float32')
  b_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
  b_labels = np.asarray(d['labels'], dtype=np.int32)
  
  train_data = np.append(train_data, b_data, axis = 0)
  train_labels = np.append(train_labels, b_labels)
  
  # ---- add third batch to training data
  d = unpickle('C:/SQLData/661/CIFAR10/data_batch_3')
  # now extract image data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  data = d['data']
  data = data.astype('float32')
  b_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
  b_labels = np.asarray(d['labels'], dtype=np.int32)
  
  train_data = np.append(train_data, b_data, axis = 0)
  train_labels = np.append(train_labels, b_labels)

  # ---- add fourth batch to training data
  d = unpickle('C:/SQLData/661/CIFAR10/data_batch_4')
  # now extract image data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  data = d['data']
  data = data.astype('float32')
  b_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
  b_labels = np.asarray(d['labels'], dtype=np.int32)
  
  train_data = np.append(train_data, b_data, axis = 0)
  train_labels = np.append(train_labels, b_labels)
  
   # ---- add fifth batch to training data
  d = unpickle('C:/SQLData/661/CIFAR10/data_batch_5')
  # now extract image data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  data = d['data']
  data = data.astype('float32')
  b_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
  b_labels = np.asarray(d['labels'], dtype=np.int32)
  
  train_data = np.append(train_data, b_data, axis = 0)
  train_labels = np.append(train_labels, b_labels)
  
  # clean memory
  del b_data
  del b_labels
  del data
  del d
  
  # Create the Estimator
  cifar_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/cifarV3")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  # Train the model
  with tf.device('/cpu:0'):
      cifar_classifier.fit(
              x=train_data,
              y=train_labels,
              batch_size=64,
              steps=5100,
              monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          # learn.metric_spec.MetricSpec(
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # remove training data from memory
  # del train_data
  # del train_labels
   
  # --------------------------
  # now load eval data
  
  d = unpickle('C:/SQLData/661/CIFAR10/test_batch')
  # now extract image data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  eval_data = d['data']
  eval_data = eval_data.astype('float32')
  eval_data = np.transpose(np.reshape(eval_data,(-1,32,32,3), order='F'),axes=(0,2,1,3))
  eval_labels = np.asarray(d['labels'], dtype=np.int32)
  
  del d
  
  
  ###############################
  ###### Data verificatiun - comment out when running
  # Displays a handful of images in iPython console to verify proper
  # loading of data set
  #from scipy.misc.pilutil import toimage
  #from matplotlib import pyplot 
  #for i in range(0,9):
  #    pyplot.subplot(330 + 1 + i)
  #    pyplot.imshow(toimage(t_eval[0][i]))
    
  # show the plot
  #pyplot.show()
  #################################
  
  # eval a subset of the training data
  small_train =  np.array_split(train_data, 20)[1]
  small_labs =  np.array_split(train_labels, 20)[1]
  eval_results = cifar_classifier.evaluate(
      x= small_train, y=small_labs, metrics=metrics)
  print(eval_results)


### Code below ONLY for tesitng ############
  # use snall subset of eval data to avoid laptop crash
#  small_train =  np.array_split(eval_data, 10)[3]
#  small_labs =  np.array_split(eval_labels, 10)[3]
#  eval_results = cifar_classifier.evaluate(
#      x= small_train, y=small_labs, metrics=metrics)
#  print(eval_results)


####################################
  # Evaluate the model and print results
  eval_results = cifar_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()