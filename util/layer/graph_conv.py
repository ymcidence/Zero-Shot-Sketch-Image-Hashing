import tensorflow as tf
from util.layer import conventional_layers as layers

OVERFLOW_MARGIN = 1e-8
ADJACENCY_SCALER = 0.1


def build_adjacency(tensor_in):
    squared_sum = tf.reshape(tf.reduce_sum(tensor_in * tensor_in, 1), [-1, 1])
    distances = squared_sum - 2 * tf.matmul(tensor_in, tensor_in, transpose_b=True) + tf.transpose(squared_sum)
    adjacency = tf.exp(-1 * distances / ADJACENCY_SCALER)
    return adjacency


def graph_laplacian(adjacency):
    """
    :param adjacency: must be self-connected
    :return: 
    """
    graph_size = adjacency.shape.as_list()[0]
    a = adjacency  # + tf.eye(graph_size)
    d = a @ tf.ones([graph_size, 1])
    d_inv_sqrt = tf.pow(d + OVERFLOW_MARGIN, -0.5)
    d_inv_sqrt = tf.eye(graph_size) * d_inv_sqrt
    laplacian = d_inv_sqrt @ a @ d_inv_sqrt
    return laplacian


def spectrum_conv_layer(name, tensor_in, adjacency, out_dim):
    """
    Convolution on a graph with graph Laplacian
    :param name:
    :param tensor_in: [N D]
    :param adjacency: [N N]
    :param out_dim:
    :return:
    """
    fc_sc = layers.fc_layer(name, tensor_in, output_dim=out_dim)
    conv_sc = graph_laplacian(adjacency) @ fc_sc
    return conv_sc


def _try():
    adj = tf.constant([[0, 0.2, 0.3], [0.2, 0, 0.6], [0.3, 0.6, 0]], dtype=tf.float32)
    hehe = graph_laplacian(adj)
    sess = tf.Session()
    hehe_out = sess.run(hehe)
    print(hehe_out)


def _try2():
    import numpy as np
    import scipy.sparse as sp

    def normalize_adj(adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(adj):
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return adj_normalized.toarray()

    a = np.asarray([[0, 0.2, 0.3], [0.2, 0, 0.6], [0.3, 0.6, 0]])
    b = preprocess_adj(a)
    print(b)


if __name__ == '__main__':
    _try2()
