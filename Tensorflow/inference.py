'''
CURRENTLY NOT COMPLETE
'''
import tensorflow as tf

MODEL_PATH = 'models/cifar10_vgg9_2.model'

def run_inference():
    '''Runs inference against a loaded model'''
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(MODEL_PATH + '.meta', clear_devices=True)
        new_saver.restore(sess, MODEL_PATH)
        print(tf.get_default_graph().get_all_collection_keys())

run_inference()