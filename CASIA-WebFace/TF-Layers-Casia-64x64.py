# -*- coding: utf-8 -*-
"""
Created on Fri April 3 12:05:24 2017

@author: James Topor
"""
"""Convolutional Neural Network Estimator for CASIA WebFace, built with tf.layers."""

# NOTE: This model achieves 100% accuracy on the training set and 40.29% accuracy
# on the eval set after 7,000 steps. Training required more than 20 hours 
# on a Windows 10 laptop via CPU only (graphics card was slower). Processing
# was very slow due to the 64x64 size of the images and the relatively large
# number of possible classes (524)


# The code herein is based partly on the Tensorflow TF Layers tutorial 
# accessible at https://www.tensorflow.org/tutorials/layers
# That tutorial builds a CNN to evaluate MNIST images. The approach used
# in the tutorial is modified here to enable the classification of the
# CASIA WebFace (CWF) data set as provided by Beijing's Center for Biometrics
# and Security Research.
#
# LFW is comprised of more than 494,000 250x250 color (RGB) images of human faces.
# Here a pre-prepped subset of the CASIA data set is loaded via python's unpickle()
# function. The python code is to pre-prep the CASIA data can be found here:
#     
# https://github.com/jtopor/CUNY-MSDA-661/blob/master/CASIA-WebFace/casia-TF-dataprepper.py
#
# The size of the data set used here for training is 40,000 64x64 images representing 
# 524 individuals. The testing data set is 10,000 images spanning the same 524 
# individuals
#
# Initially, dropout layers were inserted after each pooling layer. However, the
# loss was not decreasing in any measurable manner while those dropout layers
# were present. As such, the were removed (commented out below) as was a second
# fully connected layer. Once those layers were removed the CNN's loss began to 
# decrease very slowly. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import pickle

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# define global variable for number of classes
n_classes = 0

##############################################################

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  
  # instantiate global for number of possible classifications
  global n_classes
  
  # Input Layer
  input_layer = tf.convert_to_tensor(features)

  # Convolutional Layers 1 + 2
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 3]
  # Output Tensor Shape: [batch_size, 64, 64, 32]
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
  # Input Tensor Shape: [batch_size, 64, 64, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # add dropout layer to introduce randomness / reduce possibility of overfitting
  '''
  dropout1 = tf.layers.dropout(
    inputs=pool1, rate=0.25, training=mode == learn.ModeKeys.TRAIN)
  '''
  
  # Convolutional Layers 3 + 4
  # Computes 64 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 64]
  conv3 = tf.layers.conv2d(
      inputs= pool1,
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
  # Input Tensor Shape: [batch_size,  32, 32, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  '''
  # add dropout layer to introduce randomness / reduce possibility of overfitting
  dropout2 = tf.layers.dropout(
      inputs=pool2, rate=0.25, training=mode == learn.ModeKeys.TRAIN)
  '''
  
  # Convolutional Layers 5 + 6
  # Computes 64 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 96]
  conv5 = tf.layers.conv2d(
      inputs= pool2,
      filters=96,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  conv6 = tf.layers.conv2d(
      inputs= conv5,
      filters=96,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  '''
  dropout3 = tf.layers.dropout(
      inputs=conv6, rate=0.25, training=mode == learn.ModeKeys.TRAIN)
  '''
    # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size,  16, 16, 96]
  # Output Tensor Shape: [batch_size, 8, 8, 96]
  pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
  
   
  # Convolutional Layers 7 + 8
  # Computes 64 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 8, 8, 96]
  # Output Tensor Shape: [batch_size, 8, 8, 128]
  conv7 = tf.layers.conv2d(
      inputs= pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  conv8 = tf.layers.conv2d(
      inputs= conv7,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 8, 8, 128]
  # Output Tensor Shape: [batch_size, 8192]
  pool3_flat = tf.reshape(conv8, [-1, 8 * 8 * 128])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 8192]
  # Output Tensor Shape: [batch_size, 1024]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout4 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # add a 2nd dense layer with 512 neurons
#  dense2 = tf.layers.dense(inputs= dropout4, units=512, activation=tf.nn.relu)
  
  # Add dropout operation; 0.6 probability that element will be kept
#  dropout5 = tf.layers.dropout(
#      inputs=dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)


  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, n_classes]
  logits = tf.layers.dense(inputs=dropout4, units= n_classes)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  # !!! Make sure depth variable is set properly relative to data !!!!
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth= n_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  # Use Adagrad optimizer + initial learning rate of 0.01  
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.01, 
        optimizer="Adagrad")

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

#########################################################################
#########################################################################

def main(unused_argv):

  global n_classes
  
 # how to load dict object from disk in your Tensorflow program
  with open('c:/tmp/casia/casia_training64x64.pickle', 'rb') as handle:
    b = pickle.load(handle)

  # to access images + labels in dict object
  train_data = b["Images"]
  train_labels = b["Labels"]
  n_classes = b["NumClasses"]
  
  ''' For verification of image input: single image extract + display
  from scipy.misc import toimage
  tp = train_data[100,0:]
  tl = train_labels[0]

  toimage(tp)
  '''
  
  ###############################
  ###### Data verificatiun - comment out when running
  # Displays a handful of images in iPython console to verify proper
  # loading of data set
  # t_eval = train_data
  # from scipy.misc.pilutil import toimage
  # from matplotlib import pyplot 
  # for i in range(0,9):
  #    pyplot.subplot(330 + 1 + i)
  #    pyplot.imshow(toimage(t_eval[i]))
    
  # show the plot
  # pyplot.show()
  #################################
  
  # Create the Estimator
  casia_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/casia1")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  
  # Train the model
  with tf.device('/cpu:0'):
      casia_classifier.fit(
              x=train_data,
              y=train_labels,
              batch_size=128,
              steps=7000,
              monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          # learn.metric_spec.MetricSpec(
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # eval a small subset of the training data
  small_train =  np.array_split(train_data, 200)[21]
  small_labs =  np.array_split(train_labels, 200)[21]
  eval_results = casia_classifier.evaluate(
      x= small_train, y=small_labs, metrics=metrics)
  print(eval_results)

  # remove training data from memory
  del train_data
  del train_labels

##################################### EVAL ####################################
 # load testing data
  with open('c:/tmp/casia/casia_testing64x64.pickle', 'rb') as handle:
    b = pickle.load(handle)

  # to access images + labels in dict object
  eval_data = b["Images"]
  eval_labels = b["Labels"]
  n_classes = b["NumClasses"]

  # Evaluate all test data and print results
  acc = 0
  e_loss = 0
  n_batches = 40
  for i in range (0, n_batches - 1):
    small_eval =  np.array_split(eval_data, n_batches)[i]
    small_labs =  np.array_split(eval_labels, n_batches)[i]
    eval_results = casia_classifier.evaluate(
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