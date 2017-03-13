# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:05:24 2017

@author: James T
"""

"""Convolutional Neural Network Estimator for LFW, built with tf.layers."""
# The code herein is based largely on the Tensorflow TF Layers tutorial 
# accessible at https://www.tensorflow.org/tutorials/layers
# That tutorial builds a CNN to evaluate MNIST images. The approach used
# in the tutorial is modified here to enable the classification of Labeled Faces
# In the Wild (LFW) data set as provided by the University of Massachuesetts
# at the following web link http://vis-www.cs.umass.edu/lfw/.
#
# LFW is comprised of more than 13,000 250x250 color (RGB) images of human faces.
# Here the data set is subsetted to extract images of any individual that is 
# represented in at least 30 images within the LFW data set. This subsetting 
# yields a total of 2370 images, which are then divided up into training and
# evaluation subsets, with the training set having 1777 images and 593 images
# set aside for evaluation.
#
# The data set is loaded via a built-in Python module: 
#    sklearn.datasets.fetch_lfw_people
# Each image is then centered and resized / rescaled to be 64x64 prior to activation of the
# CNN.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer

  input_layer = tf.convert_to_tensor(features)

  # if comment out above then use:
  # input_layer = features

  # Convolutional Layer #1
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

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 64]
  conv3 = tf.layers.conv2d(
      inputs=pool1,
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
  # Input Tensor Shape: [batch_size, 32, 32, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)


 # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  conv5 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  conv6 = tf.layers.conv2d(
      inputs= conv5,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 8, 8, 64]
  pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 4096]
  pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.2, training=mode == learn.ModeKeys.TRAIN)
      # inputs=dense, rate=0.4, training= True)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 34]
  logits = tf.layers.dense(inputs=dropout, units=34)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  # !!! Make sure depth variable is set properly relative to data !!!!
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=34)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001, 
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

  # this produces centered 64x64 image from orig. 250x250
  lfw_people = fetch_lfw_people(min_faces_per_person=30, 
                                slice_ = (slice(61,189),slice(61,189)),
                                resize=0.5, color = True)
  X = lfw_people.images
  y = lfw_people.target
  
  # get count of number of possible labels - need to use this as
  # number of units for dense layer in call to tf.layers.dense and
  # for defining the one-hot matrix. Here the number of possible
  # labels is 34. It will be differnet if you use a different subset
  # of LFW.
  target_names = lfw_people.target_names
  n_classes = target_names.shape[0]
  
  y = np.asarray(y, dtype=np.int32)
  
  # split into a training and testing set
  # X_train, X_test, y_train, y_test = train_test_split(
  train_data, eval_data, train_labels, eval_labels = train_test_split(
    X, y, test_size=0.25, random_state=42)
  
  ###############################
  ###### Data verificatiun - comment out when running
  # Displays a handful of images in iPython console to verify proper
  # loading of data set
# t_eval = train_data
# from scipy.misc.pilutil import toimage
# from matplotlib import pyplot 
# for i in range(0,9):
#      pyplot.subplot(330 + 1 + i)
#      pyplot.imshow(toimage(t_eval[i]))
    
  # show the plot
  #pyplot.show()
  #################################
  
  # Create the Estimator
  lfw_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/lfw4")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  # Train the model
  with tf.device('/cpu:0'):
      lfw_classifier.fit(
              x=train_data,
              y=train_labels,
              batch_size=64,
              steps=10000,
              monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          # learn.metric_spec.MetricSpec(
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # remove training data from memory
  del train_data
  del train_labels
   
####################################
  # Evaluate the model and print results
  eval_results = lfw_classifier.evaluate(
    x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()