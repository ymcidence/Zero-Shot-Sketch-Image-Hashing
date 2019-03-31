import tensorflow as tf
from util.layer import conventional_layers as layers


def _kronecker_product(tensor_a, tensor_b):
    batch_size = tensor_a.shape.as_list()[0]
    a_size = tensor_a.shape.as_list()[1]
    b_size = tensor_b.shape.as_list()[1]

    a = tf.reshape(tensor_a, [batch_size, 1, a_size])
    b = tf.reshape(tensor_b, [batch_size, 1, b_size])
    matrix_prod = tf.transpose(a, [0, 2, 1]) @ b

    prod = tf.reshape(matrix_prod, [batch_size, -1])
    return prod


def kronecker_layer(name, tensor_a, tensor_b, out_size=256):
    with tf.variable_scope(name):
        tensor_a = layers.fc_relu_layer('fc_a', tensor_a, out_size)
        tensor_b = layers.fc_relu_layer('fc_b', tensor_b, out_size)

        return _kronecker_product(tensor_a, tensor_b)


def fused_kronecker_layer(tensor_a, tensor_b, tensor_semantic):
    fused_feat = tensor_a * tensor_b
    return kronecker_layer(fused_feat, tensor_semantic)
