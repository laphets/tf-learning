import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE: int = 500


def get_weight_variable(shape, regularizer):
    weight = tf.get_variable("weight", shape,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None :
        tf.add_to_collection('losses', regularizer(weight))
    return weight


def inference(input_tensor, regularizer):
    # The first layout
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
    # The second layout
    with tf.variable_scope('layout2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights)+biases
    return layer2
