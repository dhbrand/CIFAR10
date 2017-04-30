#!/usr/bin/env python3
"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""

import tensorflow as tf
import time
import Vgg9CIFAR10
import Vgg3CIFAR10
import CONSTANTS
import Inputs
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.python import debug as tf_debug

def gradients_summary(gradients):
    for grad, var in gradients:
        if grad is not None:
            tf.summary.histogram('{}/gradients'.format(var.op.name), grad)


def create_sess_ops():
    '''
    Creates and returns operations needed for running
    a tensorflow training session
    '''
    GRAPH = tf.Graph()
    with GRAPH.as_default():
        examples, labels = Inputs.read_inputs(CONSTANTS.RecordPaths,
                                              batch_size=CONSTANTS.BATCH_SIZE,
                                              img_shape=CONSTANTS.IMAGE_SHAPE,
                                              num_threads=CONSTANTS.INPUT_PIPELINE_THREADS)
        examples = tf.reshape(examples, [CONSTANTS.BATCH_SIZE, CONSTANTS.IMAGE_SHAPE[0],
                                         CONSTANTS.IMAGE_SHAPE[1], CONSTANTS.IMAGE_SHAPE[2]])
        logits = Vgg9CIFAR10.inference(examples)
        loss = Vgg9CIFAR10.loss(logits, labels)
        OPTIMIZER = tf.train.AdamOptimizer(CONSTANTS.LEARNING_RATE)
        #OPTIMIZER = tf.train.RMSPropOptimizer(CONSTANTS.LEARNING_RATE)
        gradients = OPTIMIZER.compute_gradients(loss)
        apply_gradient_op = OPTIMIZER.apply_gradients(gradients)
        gradients_summary(gradients)
        summaries_op = tf.summary.merge_all()
        return [apply_gradient_op, summaries_op, loss, logits], GRAPH

def main():
    '''
    Run and Train CIFAR 10
    '''
    print('starting...')
    ops, GRAPH = create_sess_ops()
    total_duration = 0.0
    with tf.Session(graph=GRAPH) as SESSION:
        # if CONSTANTS.DEBUG:
        #     SESSION = tf_debug.LocalCLIDebugWrapperSession(SESSION)
        #     SESSION.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        COORDINATOR = tf.train.Coordinator()
        THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)
        SESSION.run(tf.global_variables_initializer())
        SUMMARY_WRITER = tf.summary.FileWriter('Tensorboard/CIFAR_10_VGG9_50neuron_1pool_1e-4lr_adam')
        GRAPH_SAVER = tf.train.Saver()

        for EPOCH in range(CONSTANTS.EPOCHS):
            duration = 0
            error = 0.0
            start_time = time.time()
            for batch in range(CONSTANTS.MINI_BATCHES):
                _, summaries, cost_val, prediction = SESSION.run(ops)
                # print(np.where(np.isnan(prediction)))
                # print(prediction[0])
                # print(labels[0])
                # plt.imshow(examples[0])       
                # plt.show()         
                error += cost_val
            duration += time.time() - start_time
            total_duration += duration
            SUMMARY_WRITER.add_summary(summaries, EPOCH)
            print('Epoch %d: loss = %.2f (%.3f sec)' % (EPOCH, error, duration))
            if EPOCH == CONSTANTS.EPOCHS - 1 or error < 0.005:
                print(
                    'Done training for %d epochs. (%.3f sec)' % (EPOCH, total_duration)
                )
                break
        GRAPH_SAVER.save(SESSION, 'models/cifar10_vgg9_rmsprop.model')
        COORDINATOR.request_stop()
        COORDINATOR.join(THREADS)

if __name__ == "__main__":
    main()

