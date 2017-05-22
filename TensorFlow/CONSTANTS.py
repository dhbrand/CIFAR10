#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""

MODEL_NAME = "TEST-THIS-THING"

DEBUG = True

IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

INPUT_PIPELINE_THREADS = 6
#batch size * minibatches = # samples in data set or greater.
BATCH_SIZE = 500
MINI_BATCHES = 10
EPOCHS = 50
LEARNING_RATE = 1e-3
N_CLASSES = 10

r_bdir = 'C:/data/cifar_10/tfrecords/'
RecordPaths = [
    r_bdir + '1.tfrecords',
    r_bdir + '2.tfrecords',
    r_bdir + '3.tfrecords',
    r_bdir + '4.tfrecords'
]
