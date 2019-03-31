from abc import ABCMeta, abstractmethod

import tensorflow as tf

MODE_FLAG_TRAIN = 'train'
MODE_FLAG_TEST = 'test'


class AbstractNet(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess')
        self.log_path = kwargs.get('log_path')
        self.is_feature = kwargs.get('is_feature')
        self.batch_size = kwargs.get('batch_size')
        self.label_size = kwargs.get('label_size')
        self.code_length = kwargs.get('code_length')
        self.semantic_size = kwargs.get('semantic_size')
        if kwargs.get('is_feature'):
            input_size = [self.batch_size, 4096]
        else:
            input_size = [self.batch_size, 227, 227, 3]

        self.batch_image = tf.placeholder(tf.float32, input_size)
        self.batch_sketch = tf.placeholder(tf.float32, input_size)
        self.batch_label = tf.placeholder(tf.int32, [self.batch_size, self.label_size])
        self.batch_semantic = tf.placeholder(tf.float32, [self.batch_size, self.semantic_size])

        self.g_step = tf.Variable(0, trainable=False, name='global_step')

    @abstractmethod
    def _build_net(self):
        pass

    @abstractmethod
    def _build_loss(self):
        pass

    @abstractmethod
    def _build_opt(self):
        pass

    def train(self, max_iter, dataset, restore_file=None):
        pass

    def _restore(self, restore_file, var_list=None):
        if var_list is None:
            save_list = tf.trainable_variables()
        else:
            save_list = var_list
        saver = tf.train.Saver(var_list=save_list)
        saver.restore(self.sess, save_path=restore_file)

    def _save(self, save_path, step, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, save_path + 'YMModel', step)
        print('Saved!')


class NetFactory(object):
    @staticmethod
    def get_net(**kwargs):
        from model.cmzsl_simple import SimpleModel
        cases = {
            'abstract': AbstractNet,
            'default': SimpleModel
        }
        model_name = kwargs.get('model', 'default')
        model = cases.get(model_name)
        return model(**kwargs)
