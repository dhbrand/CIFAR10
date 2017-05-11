#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""

MODEL_NAME = "CIFAR_10_VGG3_50neuron_1pool_33_55_filters_1e-4lr_adam"

DEBUG = True

IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

INPUT_PIPELINE_THREADS = 16
#batch size * minibatches = # samples in data set or greater.
BATCH_SIZE = 1000
MINI_BATCHES = 50
EPOCHS = 500
CHECKPOINT_EPOCHS = 25
LEARNING_RATE = 1e-4
N_CLASSES = 10

mounted_basedir = '/data/cifar10/'

RecordPaths = [
    mounted_basedir + '1.tfrecords',
    mounted_basedir + '2.tfrecords',
    mounted_basedir + '3.tfrecords',
    mounted_basedir + '4.tfrecords'
]
