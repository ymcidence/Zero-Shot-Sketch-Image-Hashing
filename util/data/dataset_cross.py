import numpy as np
import scipy.io as sio
from util.data.read_data import read_image
from six.moves import xrange


class DatasetCross(object):
    def __init__(self, **kwargs):
        self.set_size = kwargs.get('set_size')
        self.label_size = kwargs.get('label_size')
        self.batch_size = kwargs.get('batch_size')
        self.data_path_img = kwargs.get('data_path_img')
        self.data_path_sk = kwargs.get('data_path_sk')
        self.rounds = kwargs.get('rounds')
        self.train_meta_img = self._preprocess(sio.loadmat(kwargs.get('train_meta_img')))
        self.train_meta_sk = self._preprocess(sio.loadmat(kwargs.get('train_meta_sk')))
        self.batch_num = self.set_size // self.batch_size
        self.batch_count = 0
        self.round_count = 0
        self.im_seq = 0
        self.sk_seq = 0

    def _preprocess(self, meta):
        this_meta = meta.copy()
        file_list = this_meta['list_train']
        new_list = []
        dense = np.zeros([this_meta['class_train'].__len__(), self.label_size])
        for i in xrange(this_meta['class_train'].__len__()):
            this_file_name = file_list[i][0][0]
            this_file_name = str(this_file_name).replace('.png', '.jpg')
            new_list.append(this_file_name)
            ind = this_meta['cls_new_train'][i, 0] - 1
            dense[i, ind] = 1

        this_meta['list_train'] = np.asarray(new_list)
        this_meta['mean'] = np.asarray(this_meta['mean'])
        this_meta['cls_dense_train'] = dense
        return this_meta

    def _shuffle(self):
        self.im_seq = self.train_meta_img.get('im_seq')[..., self.round_count] - 1
        self.sk_seq = self.train_meta_sk.get('sk_seq')[..., self.round_count] - 1
        self.round_count = (self.round_count + 1) % self.rounds

    def next_batch_train(self):
        if self.batch_count == 0:
            self._shuffle()
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        batch_im_seq = self.im_seq[batch_start:batch_end]
        batch_image = [read_image(self.data_path_img + v) for v in
                       self.train_meta_img['list_train'][batch_im_seq]]
        batch_image = np.asarray(batch_image) - self.train_meta_img['mean']
        batch_label = self.train_meta_img.get('cls_dense_train')[batch_im_seq, ...]
        batch_wv = self.train_meta_img.get('wv_train')[batch_im_seq, ...]

        batch_sk_seq = self.sk_seq[batch_start:batch_end]
        batch_sketch = [read_image(self.data_path_sk + v) for v in
                        self.train_meta_sk['list_train'][batch_sk_seq]]
        batch_sketch = np.asarray(batch_sketch) - self.train_meta_sk['mean']

        this_batch = dict(batch_image=batch_image,
                          batch_label=batch_label,
                          batch_sketch=batch_sketch,
                          batch_semantic=batch_wv)
        self.batch_count = (self.batch_count + 1) % self.batch_num
        return this_batch


class DatasetTest(object):
    def __init__(self, **kwargs):
        self.set_size = kwargs.get('set_size')
        self.label_size = kwargs.get('label_size_1')
        self.batch_size = kwargs.get('batch_size')
        self.data_path_img = kwargs.get('data_path_img')
        self.data_path_sk = kwargs.get('data_path_sk')
        self.rounds = kwargs.get('rounds')
        self.test_meta_img = self._preprocess(sio.loadmat(kwargs.get('test_meta_img')))
        self.test_meta_sk = self._preprocess(sio.loadmat(kwargs.get('test_meta_sk')))
        self.batch_num = self.set_size // self.batch_size
        self.batch_count = 0
        self.round_count = 0
        self.im_seq, self.sk_seq = self._get_seq()

    def _get_seq(self):
        return self.test_meta_img.get('im_seq')[..., 0] - 1, self.test_meta_sk.get('sk_seq')[..., 0] - 1

    def _preprocess(self, meta):
        this_meta = meta.copy()
        file_list = this_meta['list_test']
        new_list = []
        dense = np.zeros([this_meta['class_test'].__len__(), self.label_size])
        for i in xrange(this_meta['class_test'].__len__()):
            this_file_name = file_list[i][0][0]
            this_file_name = str(this_file_name).replace('.png', '.jpg')
            new_list.append(this_file_name)
            ind = this_meta['cls_new_test'][i, 0] - 1
            dense[i, ind] = 1

        this_meta['list_test'] = np.asarray(new_list)
        this_meta['mean'] = np.asarray(this_meta['mean'])
        this_meta['cls_dense_test'] = dense
        return this_meta

    def _shuffle(self):
        self.im_seq = self.test_meta_img.get('im_seq')[..., self.round_count] - 1
        self.sk_seq = self.test_meta_sk.get('sk_seq')[..., self.round_count] - 1
        self.round_count = (self.round_count + 1) % self.rounds

    def next_batch_test(self, mode):
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        if mode == 'img':
            batch_im_seq = self.im_seq[batch_start:batch_end]
            batch_image = [read_image(self.data_path_img + v) for v in
                           self.test_meta_img['list_test'][batch_im_seq]]
            batch_image = np.asarray(batch_image) - self.test_meta_img['mean']
            batch_label = self.test_meta_img.get('cls_dense_test')[batch_im_seq, ...]
            batch_wv = self.test_meta_img.get('wv_test')[batch_im_seq, ...]

            this_batch = dict(batch_image=batch_image,
                              batch_label=batch_label,
                              batch_semantic=batch_wv)
        else:
            batch_im_seq = self.sk_seq[batch_start:batch_end]
            batch_label = self.test_meta_sk.get('cls_dense_test')[batch_im_seq, ...]
            batch_wv = self.test_meta_sk.get('wv_test')[batch_im_seq, ...]

            batch_sk_seq = self.sk_seq[batch_start:batch_end]
            batch_sketch = [read_image(self.data_path_sk + v) for v in
                            self.test_meta_sk['list_test'][batch_sk_seq]]
            batch_sketch = np.asarray(batch_sketch) - self.test_meta_sk['mean']

            this_batch = dict(batch_sketch=batch_sketch,
                              batch_label=batch_label,
                              batch_semantic=batch_wv)
        self.batch_count = (self.batch_count + 1) % self.batch_num
        return this_batch


class DatasetTest1(object):
    def __init__(self, **kwargs):
        self.set_size = kwargs.get('set_size')
        self.label_size = kwargs.get('label_size_1')
        self.batch_size = kwargs.get('batch_size')
        self.data_path_img = kwargs.get('data_path_img')
        self.data_path_sk = kwargs.get('data_path_sk')
        self.rounds = kwargs.get('rounds')
        self.train_meta_img = self._preprocess(sio.loadmat(kwargs.get('test_meta_img')))
        self.train_meta_sk = self._preprocess(sio.loadmat(kwargs.get('test_meta_sk')))
        self.batch_num = self.set_size // self.batch_size
        self.batch_count = 0
        self.round_count = 0
        self.im_seq, self.sk_seq = self._get_seq()

    def _get_seq(self):
        return self.train_meta_img.get('im_seq')[..., 0] - 1, self.train_meta_sk.get('sk_seq')[..., 0] - 1

    def _preprocess(self, meta):
        this_meta = meta.copy()
        file_list = this_meta['list_train']
        new_list = []
        dense = np.zeros([this_meta['class_train'].__len__(), self.label_size])
        for i in xrange(this_meta['class_train'].__len__()):
            this_file_name = file_list[i][0][0]
            this_file_name = str(this_file_name).replace('.png', '.jpg')
            new_list.append(this_file_name)
            ind = this_meta['cls_new_train'][i, 0] - 1
            dense[i, ind] = 1

        this_meta['list_train'] = np.asarray(new_list)
        this_meta['mean'] = np.asarray(this_meta['mean'])
        this_meta['cls_dense_train'] = dense
        return this_meta

    def _shuffle(self):
        self.im_seq = self.train_meta_img.get('im_seq')[..., self.round_count] - 1
        self.sk_seq = self.train_meta_sk.get('sk_seq')[..., self.round_count] - 1
        self.round_count = (self.round_count + 1) % self.rounds

    def next_batch_test(self, mode):
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        if mode == 'img':
            batch_im_seq = self.im_seq[batch_start:batch_end]
            batch_image = [read_image(self.data_path_img + v) for v in
                           self.train_meta_img['list_train'][batch_im_seq]]
            batch_image = np.asarray(batch_image) - self.train_meta_img['mean']
            batch_label = self.train_meta_img.get('cls_dense_train')[batch_im_seq, ...]
            batch_wv = self.train_meta_img.get('wv_train')[batch_im_seq, ...]

            this_batch = dict(batch_image=batch_image,
                              batch_label=batch_label,
                              batch_semantic=batch_wv)
        else:
            batch_im_seq = self.sk_seq[batch_start:batch_end]
            batch_label = self.train_meta_sk.get('cls_dense_train')[batch_im_seq, ...]
            batch_wv = self.train_meta_sk.get('wv_train')[batch_im_seq, ...]

            batch_sk_seq = self.sk_seq[batch_start:batch_end]
            batch_sketch = [read_image(self.data_path_sk + v) for v in
                            self.train_meta_sk['list_train'][batch_sk_seq]]
            batch_sketch = np.asarray(batch_sketch) - self.train_meta_sk['mean']

            this_batch = dict(batch_sketch=batch_sketch,
                              batch_label=batch_label,
                              batch_semantic=batch_wv)
        self.batch_count = (self.batch_count + 1) % self.batch_num
        return this_batch


if __name__ == '__main__':
    config = dict(
        set_size=160862,
        label_size=200,
        batch_size=32,
        rounds=10,
        data_path_img='E:\\WorkSpace\\Data\\TU\\ImageResized\\',
        data_path_sk='E:\\WorkSpace\\Data\\TU\\SketchResized\\',
        train_meta_img='E:\\WorkSpace\\Data\\TU\\Meta\\img_train.mat',
        train_meta_sk='E:\\WorkSpace\\Data\\TU\\Meta\\sk_train.mat'
    )
    data = DatasetCross(**config)
    data.next_batch_train()
    data.next_batch_train()
    print('hehe')
