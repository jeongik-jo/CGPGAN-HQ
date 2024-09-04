import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr


dis_opt = kr.optimizers.AdamW(learning_rate=0.003, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
cla_opt = kr.optimizers.AdamW(learning_rate=0.003, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
gen_opt = kr.optimizers.AdamW(learning_rate=0.003, weight_decay=0.0001, beta_1=0.0, beta_2=0.99,
                              use_ema=True, ema_momentum=0.999, ema_overwrite_frequency=None)

is_ffhq = False
img_res = 256
img_chn = 3
cnt_dim = 1024
ctg_dim = 10

ctg_w = 1.0
adv_reg_w = 10.0
ctg_reg_w = 10.0


decay_rate = 0.999
ctg_update_start_epoch = 50

batch_size = 8
save_img_size = batch_size

train_data_size = -1
test_data_size = -1
shuffle_test_dataset = False
epochs = 150

load_model = False

eval_model = True
epoch_per_evaluate = 2

ctg_prob = tf.Variable(tf.fill([ctg_dim], 1.0 / ctg_dim))

def cnt_dist_func(batch_size):
    return tf.random.normal([batch_size, cnt_dim])


def ctg_dist_func(batch_size):
    return tf.one_hot(tf.random.categorical(logits=[tf.math.log(ctg_prob + 1e-8)], num_samples=batch_size)[0], depth=ctg_dim)


def calc_ctg_ent():
    return tf.reduce_sum(-ctg_prob * tf.math.log(ctg_prob + 1e-8))
