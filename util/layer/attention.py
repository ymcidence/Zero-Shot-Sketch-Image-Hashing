import tensorflow as tf


def map_attention(feat_map):
    """
    Attention mechanism for feature maps
    :param feat_map: [N H W C]
    :return:
    """
    channel_size = feat_map.shape.as_list()[-1]
    weights_initializer = tf.random_normal_initializer(stddev=0.01)
    biases_initializer = tf.constant_initializer(0.)

    weights = tf.get_variable("kernel", [1, 1, channel_size, 1], initializer=weights_initializer)
    biases = tf.get_variable("bias", 1, initializer=biases_initializer)

    attention = tf.nn.conv2d(feat_map, weights, strides=[1, 1, 1, 1], padding='SAME')
    attention = tf.nn.bias_add(attention, biases)

    exp_attention = tf.exp(attention)
    sum_attention = tf.reduce_sum(exp_attention, [1, 2, 3], keep_dims=True)
    normed_attention = exp_attention / sum_attention

    att_feat = tf.reduce_sum(feat_map * normed_attention, [1, 2])

    return att_feat, normed_attention


def semantic_attention(feat_img, feat_semantic):
    """
    Fused attention wrapper
    :param feat_img: [N H W C]
    :param feat_semantic: [N C]
    :return:
    """
    fc_s = tf.expand_dims(tf.expand_dims(tf.tanh(feat_semantic), 1), 1)
    mixed_feat = feat_img * fc_s
    att_feat, normed_attention = map_attention(mixed_feat)

    return att_feat, normed_attention


if __name__ == '__main__':
    import numpy as np

    test_map = [[[[1, 2], [1, 3], [1, 4]], [[2, 2], [2, 3], [2, 1]], [[3, 3], [3, 4], [3, 5]]],
                [[[1, 2], [1, 3], [1, 4]], [[2, 2], [2, 3], [2, 1]], [[3, 3], [3, 4], [3, 9]]]]
    test_filter = [[[[1], [1]]]]
    test_map = np.asarray(test_map)
    test_filter = np.asarray(test_filter)
    a = tf.placeholder(tf.float32, [2, 3, 3, 2])
    b = tf.placeholder(tf.float32, [1, 1, 2, 1])
    c = map_attention(a)
    sess = tf.Session()

    d, e = sess.run(c, feed_dict={a: test_map, b: test_filter})
    print(d)
