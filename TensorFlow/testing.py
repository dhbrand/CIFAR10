
import tensorflow as tf
import CONSTANTS

reader = tf.TFRecordReader()
_, example = reader.read(CONSTANTS.RecordPaths[0])
features = tf.parse_single_example(example, features={
    'example': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
})
img_sample = tf.decode_raw(features['example'], tf.float32)
img_sample = tf.reshape(img_sample, image_shape)
label = tf.cast(features['label'], tf.int64)

print(img_sample)
