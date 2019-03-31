import numpy as np

import tensorflow as tf
from util.layer.conventional_layers import fc_layer

# NPY_PATH = 'D:\\Workspace\\Data\\TU\\bvlc_alexnet.npy'
NPY_PATH = '/home/liuxiaoming/data/CMZSL/TU/bvlc_alexnet.npy'


# NPY_PATH = 'E:\\WorkSpace\\Data\\bvlc_alexnet.npy'


def alex_net(tensor_in, with_fc=True):
    def conv(_tensor_in, kernel, biases, _k_h, _k_w, _c_o, _s_h, _s_w, _padding="VALID", _group=1):
        c_i = _tensor_in.get_shape()[-1]
        assert c_i % _group == 0
        assert _c_o % _group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, _s_h, _s_w, 1], padding=_padding)

        if _group == 1:
            _conv = convolve(_tensor_in, kernel)
        else:
            input_groups = tf.split(_tensor_in, _group, 3)  # tf.split(3, group, input)
            kernel_groups = tf.split(kernel, _group, 3)  # tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            _conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
        return tf.reshape(tf.nn.bias_add(_conv, biases), [-1] + _conv.get_shape().as_list()[1:])

    x = tensor_in
    weights_initializer = tf.random_normal_initializer(stddev=0.01)
    biases_initializer = tf.constant_initializer(0.)
    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11
    k_w = 11
    c_o_old = 3
    c_o = 96
    s_h = 4
    s_w = 4
    conv1w = tf.get_variable('conv_1/kernel', [k_h, k_w, c_o_old, c_o], initializer=weights_initializer)
    conv1b = tf.get_variable("conv_1/bias", c_o, initializer=biases_initializer)
    conv1_in = conv(x, conv1w, conv1b, k_h, k_w, c_o, s_h, s_w, _padding="SAME", _group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5
    k_w = 5
    group = 2
    c_o_old = c_o / group
    c_o = 256
    s_h = 1
    s_w = 1
    conv2w = tf.get_variable('conv_2/kernel', [k_h, k_w, c_o_old, c_o], initializer=weights_initializer)
    conv2b = tf.get_variable("conv_2/bias", c_o, initializer=biases_initializer)
    conv2_in = conv(maxpool1, conv2w, conv2b, k_h, k_w, c_o, s_h, s_w, _padding="SAME", _group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3
    k_w = 3
    c_o_old = c_o
    c_o = 384
    s_h = 1
    s_w = 1
    group = 1
    conv3w = tf.get_variable('conv_3/kernel', [k_h, k_w, c_o_old, c_o], initializer=weights_initializer)
    conv3b = tf.get_variable("conv_3/bias", c_o, initializer=biases_initializer)
    conv3_in = conv(maxpool2, conv3w, conv3b, k_h, k_w, c_o, s_h, s_w, _padding="SAME", _group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3
    k_w = 3
    group = 2
    c_o_old = c_o / group
    c_o = 384
    s_h = 1
    s_w = 1
    conv4w = tf.get_variable('conv_4/kernel', [k_h, k_w, c_o_old, c_o], initializer=weights_initializer)
    conv4b = tf.get_variable("conv_4/bias", c_o, initializer=biases_initializer)
    conv4_in = conv(conv3, conv4w, conv4b, k_h, k_w, c_o, s_h, s_w, _padding="SAME", _group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3
    k_w = 3
    group = 2
    c_o_old = c_o / group
    c_o = 256
    s_h = 1
    s_w = 1
    conv5w = tf.get_variable('conv_5/kernel', [k_h, k_w, c_o_old, c_o], initializer=weights_initializer)
    conv5b = tf.get_variable("conv_5/bias", c_o, initializer=biases_initializer)
    conv5_in = conv(conv4, conv5w, conv5b, k_h, k_w, c_o, s_h, s_w, _padding="SAME", _group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    if with_fc:
        # fc6
        # fc(4096, name='fc6')
        shape = maxpool5.get_shape().as_list()
        input_dim = 1
        for d in shape[1:]:
            input_dim *= d

        fc6w = tf.get_variable('fc_6/kernel', [input_dim, 4096], initializer=weights_initializer)
        fc6b = tf.get_variable("fc_6/bias", 4096, initializer=biases_initializer)

        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, input_dim]), fc6w, fc6b)

        # fc7
        # fc(4096, name='fc7')
        fc7w = tf.get_variable('fc_7/kernel', [4096, 4096], initializer=weights_initializer)
        fc7b = tf.get_variable("fc_7/bias", 4096, initializer=biases_initializer)
        fc7 = tf.nn.xw_plus_b(fc6, fc7w, fc7b)

        return fc7, maxpool5
    else:
        return maxpool5


def alex_classifier(tensor_in, out_dim):
    from util.layer.conventional_layers import fc_layer
    fc7, _ = tf.nn.relu(alex_net(tensor_in))
    fc8 = fc_layer('fc_8', fc7, out_dim)
    return fc8


class AlexNet(object):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess')
        self.log_path = kwargs.get('log_path')
        self.label_size = kwargs.get('label_size')
        self.name_scope = kwargs.get('name_scope')
        self.batch_size = kwargs.get('batch_size')
        self.batch_image = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.batch_label = tf.placeholder(tf.int32, [self.batch_size, self.label_size])
        self.g_step = tf.Variable(0, trainable=False, name='global_step')
        self.net, self.fc7 = self._build_net()
        self.loss = self._build_loss()

    def get_op(self, mode=None):
        return self.fc7

    def _build_net(self):

        with tf.variable_scope(self.name_scope):
            # fc8 = alex_classifier(self.batch_image, self.label_size)
            fc7, _ = alex_net(self.batch_image)
            fc7 = tf.nn.relu(fc7)
            fc8 = fc_layer('fc_8', fc7, self.label_size)
        return fc8, fc7

    def _build_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net, labels=self.batch_label))

        regu_list = [var for var in tf.trainable_variables() if var.name.find('conv') < 0]
        single_loss = [tf.nn.l2_loss(par) for par in regu_list]
        regu_loss = 0.0001 * tf.add_n(single_loss)

        correct_prediction = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.batch_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('regu_loss', regu_loss)
        tf.summary.scalar('batch_acc', accuracy)
        return loss + regu_loss

    def _build_opt(self):
        optimizer = tf.train.AdamOptimizer(1e-4)
        return optimizer.minimize(self.loss, global_step=self.g_step)

    def _restore(self, restore_file, var_list=None):
        if var_list is None:
            save_list = tf.trainable_variables()
        else:
            save_list = var_list
        saver = tf.train.Saver(var_list=save_list)
        saver.restore(self.sess, save_path=restore_file)
        print('Model restored.')

    def restore_np(self, file_path):
        net_param = np.load(file_path, encoding='latin1').item()
        restore_list = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'fc_6', 'fc_7']
        assigner = []
        with tf.variable_scope(self.name_scope, reuse=True):
            for this_name in restore_list:
                this_kernel = tf.get_variable(this_name + '/kernel')
                this_bias = tf.get_variable(this_name + '/bias')
                kernel_param = net_param[this_name.replace('_', '')][0]
                bias_param = net_param[this_name.replace('_', '')][1]
                kernel_assigner = tf.assign(this_kernel, kernel_param)
                bias_assigner = tf.assign(this_bias, bias_param)
                assigner.append(kernel_assigner)
                assigner.append(bias_assigner)

        self.sess.run(tf.group(*assigner))
        print('Model restored.')

    def _save(self, save_path, step):
        var_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, save_path + 'YMModel', global_step=step)
        print('Saved!')

    def train(self, max_iter, dataset, restore_file=None):
        from time import gmtime, strftime
        from six.moves import xrange
        import os
        import gc
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        ops = self._build_opt()
        initial_op = tf.global_variables_initializer()
        self.sess.run(initial_op)
        summary_path = os.path.join(self.log_path, 'log', time_string) + os.sep
        save_path = os.path.join(self.log_path, 'model') + os.sep

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if restore_file is not None:
            self._restore(restore_file)
        else:
            self.restore_np(NPY_PATH)

        writer = tf.summary.FileWriter(summary_path)
        summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for i in xrange(max_iter):
            this_batch = dataset.next_batch_train()
            print('Data obtained')
            feed_dict = {self.batch_image: this_batch['batch_image'],
                         self.batch_label: this_batch['batch_label']}
            this_loss, _, this_summary = self.sess.run([self.loss, ops, summary], feed_dict=feed_dict)
            this_step = tf.train.global_step(self.sess, self.g_step)
            writer.add_summary(this_summary, global_step=this_step)
            print('Batch ' + str(i) + '(Global Step: ' + str(this_step) + '): ' + str(this_loss))
            gc.collect()

            if i % 500 == 0 and i > 0:
                self._save(save_path, this_step)
