#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""

MODEL_NAME = "Image_Classify_m_vgg3_3conv_lr_1e-8_adam"

DEBUG = True

IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

INPUT_PIPELINE_THREADS = 6
#batch size * minibatches = # samples in data set or greater.
BATCH_SIZE = 5000
MINI_BATCHES = 10
EPOCHS = 200
CHECKPOINT_EPOCHS = 25
LEARNING_RATE = 1e-8
N_CLASSES = 10

mounted_basedir = '/data/cifar10/'
mounted_tboard_dir = '/tensorboard/imagesystems/'

RecordPaths = [
    mounted_basedir + '1.tfrecords',
    mounted_basedir + '2.tfrecords',
    mounted_basedir + '3.tfrecords',
    mounted_basedir + '4.tfrecords'
]
