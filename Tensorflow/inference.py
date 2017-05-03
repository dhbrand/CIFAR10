'''
CURRENTLY NOT COMPLETE
'''
import time
import pickle
import tensorflow as tf
import CONSTANTS
import numpy as np
from scipy import misc


MODEL_PATH = 'models/' + CONSTANTS.MODEL_NAME + '.model'
imgs_bsdir = 'C:/data/cifar_10/train/'

images = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
with open('models/' + CONSTANTS.MODEL_NAME + '.pkl', 'rb') as model_in:
    model = pickle.load(model_in)
logits = model.inference(images)

def run_inference():
    '''Runs inference against a loaded model'''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(sess, MODEL_PATH)
        print("Starting...")
        for i in range(1, 30):
            print(str(i) + '.png')
            img = misc.imread(imgs_bsdir + str(i) + '.png').astype(np.float32) / 255.0
            img = img.reshape(1, 32, 32, 3)
            pred = sess.run(logits, feed_dict={images : img})
            max_node = np.argmax(pred)
            max_pred = str(np.amax(pred))
            print(pred)
            print('predicted label: ' + str(max_node) + ' prob: ' + max_pred)
        print('done')

run_inference()
