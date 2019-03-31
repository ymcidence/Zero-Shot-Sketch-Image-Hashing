import tensorflow as tf
from model.net_factory import NetFactory
from model.cmzsl_simple import FeatData

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
bit = 32
conf = dict(
    sess=sess,
    batch_size=200,
    data_path_im='path_to_your_mat_file',
    data_path_sk='path_to_your_mat_file',
    log_path='path_to_your_log_file' + str(bit),
    code_length=bit
)

conf_sk = dict(
    batch_size=200,
    data_path_im='path_to_another_mat_file',
    data_path_sk='path_to_another_mat_file',
    log_path='path_to_your_log_file' + str(bit),
    code_length=bit
)
conf_im = dict(
    batch_size=200,
    data_path_im='path_to_another_mat_file',
    data_path_sk='path_to_another_mat_file',
    log_path='path_to_your_log_file' + str(bit),
    code_length=bit
)
model = NetFactory.get_net(**conf)
data = FeatData(**conf)
test_im = FeatData(**conf_im)
test_sk = FeatData(**conf_sk)
model.train(10000, data, test_im=test_im, test_sk=test_sk)