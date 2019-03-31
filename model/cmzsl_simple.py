import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python.framework import function
from model.net_factory import AbstractNet as ANet
from util.layer import conventional_layers as layers
from util.layer import graph_conv as gcn
from six.moves import xrange
from util.eval import eval_tools
from util.layer.kronecker_prod import kronecker_layer

D_TYPE = tf.float32


def loss_regu(par_list, weight=0.005):
    single_regu = [tf.nn.l2_loss(v) for v in par_list]
    loss = tf.add_n(single_regu) * weight
    return loss


@function.Defun(D_TYPE, D_TYPE, D_TYPE, D_TYPE)
def doubly_sn_grad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    # {-1, 1} coding
    # yout = tf.sign(prob - epsilon)

    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon


@function.Defun(D_TYPE, D_TYPE, grad_func=doubly_sn_grad)
def doubly_sn(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob


class FeatData(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size')
        self.data_path_im = kwargs.get('data_path_im')
        self.data_path_sk = kwargs.get('data_path_sk')
        self.im_feat, self.im_label, self.wv, self.sk_feat, self.sk_label = self._get_data()
        self.batch_count = 0
        self.round_count = 0

    def _shuffle(self):
        self.im_feat, self.im_label, self.wv, self.sk_feat, self.sk_label = shuffle(self.im_feat, self.im_label,
                                                                                    self.wv, self.sk_feat,
                                                                                    self.sk_label)

    @property
    def set_size(self):
        return self.im_feat.shape[0]

    @property
    def batch_num(self):
        return self.set_size // self.batch_size

    def _get_data(self):
        im = sio.loadmat(self.data_path_im)
        sk = sio.loadmat(self.data_path_sk)
        return im['feat'], im['label'], im.get('wv'), sk['feat'], sk['label']

    def next_batch_train(self):
        if self.batch_count == 0:
            self._shuffle()
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        batch_im_feat = self.im_feat[batch_start:batch_end, ...]
        batch_label = self.im_label[batch_start:batch_end, ...]
        batch_wv = self.wv[batch_start:batch_end, ...]
        batch_sk_feat = self.sk_feat[batch_start:batch_end, ...]

        batch_label_sk = self.sk_label[batch_start:batch_end, ...]
        assert np.sum(batch_label_sk - batch_label) == 0

        this_batch = dict(batch_im_feat=batch_im_feat,
                          batch_label=batch_label,
                          batch_sk_feat=batch_sk_feat,
                          batch_semantic=batch_wv)
        self.batch_count = (self.batch_count + 1) % self.batch_num
        return this_batch

    def next_batch_test(self, mode='img'):
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        batch_im_feat = self.im_feat[batch_start:batch_end, ...]
        batch_label = self.im_label[batch_start:batch_end, ...]
        batch_wv = 0
        batch_sk_feat = self.sk_feat[batch_start:batch_end, ...]

        batch_label_sk = self.sk_label[batch_start:batch_end, ...]
        if mode == 'img':
            label = batch_label
        else:
            label = batch_label_sk
        this_batch = dict(batch_im_feat=batch_im_feat,
                          batch_label=label,
                          batch_sk_feat=batch_sk_feat,
                          batch_semantic=batch_wv)
        self.batch_count = (self.batch_count + 1) % self.batch_num
        return this_batch


class SimpleModel(ANet):
    def _build_net(self):
        fc_im = layers.fc_relu_layer('fc_im', self.batch_im_feat, 1024)
        enc_im = tf.sigmoid(layers.fc_layer('enc_im', fc_im, self.code_length))

        fc_sk = layers.fc_relu_layer('fc_sk', self.batch_sk_feat, 1024)
        enc_sk = tf.sigmoid(layers.fc_layer('enc_sk', fc_sk, self.code_length))

        concat_1 = kronecker_layer('kron', self.batch_im_feat, self.batch_sk_feat)
        batch_adjacency = gcn.build_adjacency(self.batch_semantic)
        gcn_1 = tf.nn.relu(gcn.spectrum_conv_layer('gcn_1', concat_1, batch_adjacency, 1024))
        gcn_2 = gcn.spectrum_conv_layer('gcn_2', gcn_1, batch_adjacency, self.code_length)

        eps = tf.ones([self.batch_size, self.code_length], dtype=D_TYPE) * 0.5

        codes, code_prob = doubly_sn(gcn_2, eps)

        dec_2_mean = layers.fc_layer_2('dec_mean', codes, self.code_length, 300)
        dec_2_var = layers.fc_layer_2('dec_var', codes, self.code_length, 300)

        rslt = dict(
            enc_im=enc_im,
            enc_sk=enc_sk,
            code_logits=gcn_2,
            codes=codes,
            code_prob=code_prob,
            dec_mean=dec_2_mean,
            dec_var=dec_2_var
        )
        return rslt

    def _build_opt(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        return optimizer.minimize(self.loss, global_step=self.g_step)

    def _build_loss(self):

        p_xz = tf.nn.l2_loss(self.nets.get('dec_mean') - self.batch_semantic) * 0.5
        q_zx = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.nets.get('code_logits'),
                                                                      labels=self.nets.get('codes')))
        enc_sup = (tf.sign(self.nets.get('code_logits') - 0.5) + 1) / 2

        loss_im = tf.nn.l2_loss(self.nets.get('enc_im') - enc_sup)
        loss_sk = tf.nn.l2_loss(self.nets.get('enc_sk') - enc_sup)
        l2_loss = (loss_im + loss_sk) * 0.1

        regu_loss = loss_regu(tf.trainable_variables())
        loss = p_xz + q_zx + l2_loss + regu_loss

        code_im = tf.to_float(tf.greater(self.nets.get('enc_im'), 0.5))
        code_sk = tf.to_float(tf.greater(self.nets.get('enc_sk'), 0.5))
        diff_im = tf.reduce_sum(tf.abs(code_im - enc_sup))
        diff_sk = tf.reduce_sum(tf.abs(code_sk - enc_sup))

        tf.summary.scalar('prob/p_xz', p_xz)
        tf.summary.scalar('prob/q_zx', q_zx)
        tf.summary.scalar('l2/code_im', loss_im)
        tf.summary.scalar('l2/code_sk', loss_sk)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('sum_code', tf.reduce_sum(enc_sup))
        tf.summary.scalar('diff/im', diff_im)
        tf.summary.scalar('diff/sk', diff_sk)
        return loss

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_im_feat = tf.placeholder(tf.float32, [self.batch_size, 4096])
        self.batch_sk_feat = tf.placeholder(tf.float32, [self.batch_size, 4096])
        self.nets = self._build_net()
        self.loss = self._build_loss()

    def get_op(self, mode):
        if mode == 'img':
            return self.nets.get('enc_im')
        else:
            return self.nets.get('enc_sk')

    def train(self, max_iter, dataset, restore_file=None, test_im=None, test_sk=None):
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
        writer = tf.summary.FileWriter(summary_path)
        summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

        if test_im is not None:
            hook_op = tf.placeholder(tf.float32, [])
            tf.summary.scalar('hook/map', tf.reduce_sum(hook_op))
            hook_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='hook'))

        for i in xrange(max_iter):
            this_batch = dataset.next_batch_train()
            print('Data obtained')
            feed_dict = {self.batch_im_feat: this_batch['batch_im_feat'],
                         self.batch_label: this_batch['batch_label'],
                         self.batch_sk_feat: this_batch['batch_sk_feat'],
                         self.batch_semantic: this_batch['batch_semantic']}
            this_loss, _, this_summary = self.sess.run([self.loss, ops, summary], feed_dict=feed_dict)
            this_step = tf.train.global_step(self.sess, self.g_step)
            writer.add_summary(this_summary, global_step=this_step)
            print('Batch ' + str(i) + '(Global Step: ' + str(this_step) + '): ' + str(this_loss))
            gc.collect()

            if i % 500 == 0 and i > 1999:
                if test_im is not None:
                    hook_map = self._hook(test_im, test_sk)
                    # noinspection PyUnboundLocalVariable
                    test_summary = self.sess.run(hook_summary, feed_dict={hook_op: hook_map})
                    writer.add_summary(test_summary, global_step=this_step)
                self._save(save_path, this_step)

    def _hook(self, test_im, test_sk):
        feat_im_test, label_im_test = self._get_feat(test_im)
        feat_sk_test, label_sk_test = self._get_feat(test_sk, mode='sk')
        mean_ap = eval_tools.eval_cls_map(feat_sk_test, feat_im_test, label_sk_test, label_im_test)
        return mean_ap

    def _get_feat(self, test_data, mode='img'):
        encoder_op = self.get_op(mode)
        encoder_op = tf.cast(tf.greater(encoder_op, 0.5), tf.float32)
        feat_out = []
        label_out = []
        turns = test_data.set_size // self.batch_size
        for i in xrange(turns):
            print(str(i))
            this_batch = test_data.next_batch_test(mode)
            if mode == 'img':
                feed_dict = {self.batch_im_feat: this_batch['batch_im_feat']}
            else:
                feed_dict = {self.batch_sk_feat: this_batch['batch_sk_feat']}
            feats = self.sess.run([encoder_op], feed_dict=feed_dict)
            if i == 0:
                feat_out = feats[0]
                label_out = this_batch['batch_label']
            else:
                feat_out = np.append(feat_out, feats[0], axis=0)
                label_out = np.append(label_out, this_batch['batch_label'], axis=0)
        return feat_out, label_out


if __name__ == '__main__':
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    bit = 32
    conf = dict(
        sess=sess,
        batch_size=200,
        data_path_im='E:\\Workspace\\Data\\Sketchy\\Codes\\feat_img_alex.mat',
        data_path_sk='E:\\Workspace\\Data\\Sketchy\\Codes\\feat_sk.mat',
        log_path='E:\\WorkSpace\\Data\\Log\\SK' + str(bit),
        code_length=bit
    )

    conf_sk = dict(
        batch_size=200,
        data_path_im='E:\\Workspace\\Data\\Sketchy\\Codes\\feat_img_alex_test.mat',
        data_path_sk='E:\\Workspace\\Data\\Sketchy\\Codes\\feat_sk_test.mat',
        log_path='E:\\WorkSpace\\Data\\Log\\SK' + str(bit),
        code_length=bit
    )
    conf_im = dict(
        batch_size=200,
        data_path_im='E:\\Workspace\\Data\\Sketchy\\Codes\\feat_img_alex_test.mat',
        data_path_sk='E:\\Workspace\\Data\\Sketchy\\Codes\\feat_sk_test.mat',
        log_path='E:\\WorkSpace\\Data\\Log\\SK' + str(bit),
        code_length=bit
    )
    model = SimpleModel(**conf)
    data = FeatData(**conf)
    test_im = FeatData(**conf_im)
    test_sk = FeatData(**conf_sk)
    model.train(10000, data, test_im=test_im, test_sk=test_sk)
