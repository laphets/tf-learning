import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v")
    print(v1)
