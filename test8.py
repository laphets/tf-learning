import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

v = tf.Variable(0, dtype=tf.float32, name="v")
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "model7/model.ckpt")
    print(sess.run(v))