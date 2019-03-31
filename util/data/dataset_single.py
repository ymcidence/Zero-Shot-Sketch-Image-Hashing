import numpy as np
import scipy.io as sio
import time
from util.data.read_data import read_image
from six.moves import xrange


class DatasetSingle(object):
    def __init__(self, **kwargs):
        self.set_size = kwargs.get('set_size')
        self.label_size = kwargs.get('label_size')
        self.batch_size = kwargs.get('batch_size')
        self.data_path = kwargs.get('data_path')
        self.meta_file = kwargs.get('meta_file')
        self.meta = sio.loadmat(self.meta_file)
        self._preprocess()
        self.batch_num = self.set_size // self.batch_size
        self.batch_count = 0

    def _preprocess(self):
        file_list = self.meta['list_train']
        new_list = []
        dense = np.zeros([self.set_size, self.label_size])
        for i in xrange(self.set_size):
            this_file_name = file_list[i][0][0]
            this_file_name = str(this_file_name).replace('.png', '.jpg')
            new_list.append(this_file_name)
            ind = self.meta['cls_new_train'][i, 0] - 1
            dense[i, ind] = 1

        self.meta['list_train'] = np.asarray(new_list)
        self.meta['mean'] = np.asarray(self.meta['mean'])
        self.meta['cls_dense_train'] = dense

    def _shuffle(self):
        inds = np.random.choice(self.batch_num * self.batch_size, self.batch_num * self.batch_size)
        new_meta = dict(
            class_train=self.meta['class_train'][inds],
            cls_new_train=self.meta['cls_new_train'][inds],
            cls_dense_train=self.meta['cls_dense_train'][inds, ...],
            list_train=self.meta['list_train'][inds],
            wv_train=self.meta['wv_train'][inds, ...],
            mean=self.meta['mean'],
            tmp_ind=self.meta['tmp_ind']
        )
        self.meta = new_meta

    def next_batch_train(self):
        if self.batch_count == 0:
            self._shuffle()
        batch_start = self.batch_count * self.batch_size
        batch_end = self.batch_size + batch_start
        start_time = time.time()
        batch_image = [read_image(self.data_path + v) for v in self.meta['list_train'][batch_start:batch_end]]
        read_time = time.time()
        print('time for image reading: ' + str(read_time - start_time))
        batch_image = np.asarray(batch_image) - self.meta['mean']
        batch_label = self.meta.get('cls_dense_train')[batch_start:batch_end, ...]
        this_batch = dict(batch_image=batch_image,
                          batch_label=batch_label)
        self.batch_count = (self.batch_count + 1) % self.batch_num
        other_time = time.time()
        print('time for the rest: ' + str(other_time - read_time))
        return this_batch


if __name__ == '__main__':
    config = dict(
        set_size=160859,
        batch_size=96,
        data_path='E:\\WorkSpace\\Data\\TU\\ImageResized\\',
        meta_file='E:\\WorkSpace\\Data\\TU\\Meta\\img_train.mat',
        label_size=200
    )
    this_set = DatasetSingle(**config)
    this_set.next_batch_train()
    print('hehe')
