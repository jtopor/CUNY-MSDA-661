# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:05:24 2017

@author: James Topor
"""

"""Convolutional Neural Network Estimator for CIFAR-100, built with tf.layers."""

# NOTE: This model achieves 100% accuracy on the training set and 47% accuracy
# on the eval set after 65,000 steps. Training required more than 23 hours 
# on a Windows 10 laptop via CPU only (graphics card was slower)

# This convolutional neural network is designed to train on + evaluate the 
# CIFAR-100 data set. That data set is comprised of 60,000 images spread over
# 100 possible classifications. Each classification is represented by a total 
# of 600 images.
#
# The code herein is based partially on the Tensorflow TF Layers tutorial 
# accessible at https://www.tensorflow.org/tutorials/layers
# That tutorial builds a CNN to evaluate MNIST images. The approach used
# in the tutorial is modified here to enable the classification of CIFAR-100 
# color (RGB) images.
#
# To use this code, the CIFAR-100 data set must first be downloaded and unpacked 
# within your local environment. The data set is available as a TAR file here:
# https://www.cs.toronto.edu/~kriz/cifar.html
#
# On that page, scroll down to the "Download" section and select "CIFAR-100 python version"
# The download will start automatically. When download is complete, you must
# then unpack the resulting TAR file and modify the calls to the "unpickle()" function
# herein so that the unpacked files can be loaded into Python.
#
# Also, please note that the CIFAR 100 data set when unpacked is comprised of 2
# separate files, "train" and "test". "train" contains 50,000 images while "test"
# contains 10,000 images. As you
# will see below, there are 2 separate calls to the "unpickle()" function used
# to load the two CIFAR-100 data files.


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
  # CIFAR-100 images are 32x32 pixels, and have RGB color channel
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
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 1024]
  # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout3 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
  
  # second fully connected layer = use 512 features
  dense2 = tf.layers.dense(inputs=dropout3, units=512, activation=tf.nn.relu)
  
   # Add dropout operation; 0.6 probability that element will be kept
  dropout4 = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 512]
  # Output Tensor Shape: [batch_size, 100] since CIFAR-100 has 100 possible classes
  logits = tf.layers.dense(inputs=dropout4, units=100)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  # !!! Make sure depth variable is set properly relative to data !!!!
  # For CIFAR-100 there are 100 possible classifications, so depth = 100
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=100)
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
# CIFAR-100 data loader

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict

#########################################################################

def main(unused_argv):
  # unpickle CIFAR-100 data from local directory
  d = unpickle('C:/SQLData/661/CIFAR100/train')
  # now extract image data from unpickled object. Result will be 50000 x 3072 array
  # wherein each row is 3072 unraveled image

  data = d['data']
  data = data.astype('float32')
  train_data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
  train_labels = np.asarray(d['fine_labels'], dtype=np.int32)
    
  # cleanup memory
  del d
  del data
  
  # Create the Estimator
  cifar_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/cifar100v2")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  
  # Train the model
  with tf.device('/cpu:0'):
      cifar_classifier.fit(
              x=train_data,
              y=train_labels,
              batch_size=64,
              steps=5240,
              monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          # learn.metric_spec.MetricSpec(
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

    # eval small subset of training data
  small_train =  np.array_split(train_data, 50)[7]
  small_labs =  np.array_split(train_labels, 50)[7]
  eval_results = cifar_classifier.evaluate(
      x= small_train, y=small_labs, metrics=metrics)
  print(eval_results)

  # remove training data from memory
  del train_data
  del train_labels
   
  ################## EVAL #############################
  # now load eval data
  
  d = unpickle('C:/SQLData/661/CIFAR100/test')
  # now extract testing data from unpickled object. Result will be 10000 x 3072 array
  # wherein each row is 3072 unraveled image

  eval_data = d['data']
  eval_data = eval_data.astype('float32')
  eval_data = np.transpose(np.reshape(eval_data,(-1,32,32,3), order='F'),axes=(0,2,1,3))
  eval_labels = np.asarray(d['fine_labels'], dtype=np.int32)
  
  del d
  
  # Evaluate all test data and print results
  acc = 0
  e_loss = 0
  n_batches = 20
  for i in range (0, n_batches - 1):
    small_eval =  np.array_split(eval_data, n_batches)[i]
    small_labs =  np.array_split(eval_labels, n_batches)[i]
    eval_results = cifar_classifier.evaluate(
      x=small_eval, y=small_labs, metrics=metrics)
    print(eval_results)
    acc += eval_results["accuracy"]
    e_loss += eval_results["loss"]
  
  overall_acc = acc / n_batches
  overall_loss = e_loss/ n_batches
  print("Eval Accuracy = ", overall_acc)
  print("Eval Loss = ", overall_loss)  


if __name__ == "__main__":
  tf.app.run()