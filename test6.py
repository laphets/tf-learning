import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1-1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2-1")

result = v1+v2

saver = tf.train.Saver({"v1": v1, "v2": v2})
saver.export_meta_graph("model6/1.json", as_text=True)
with tf.Session() as sess:
    saver.restore(sess, "model/model6.ckpt")
    print(sess.run(result))
