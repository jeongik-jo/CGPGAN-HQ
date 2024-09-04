import tensorflow as tf
from tensorflow import keras as kr
import Layers
import os
import HyperParameters as hp
import numpy as np


class Gen(object):
    def build_model(self):
        cnt_vec = kr.Input([hp.cnt_dim])
        ctg_vec = kr.Input([hp.ctg_dim])
        return kr.Model([ctg_vec, cnt_vec], Layers.Generator()([ctg_vec, cnt_vec]))

    def __init__(self):
        self.model = self.build_model()
        self.save_cnt_vecs = hp.cnt_dist_func(hp.save_img_size)

    def save_images(self, epoch):
        if not os.path.exists('results/samples'):
            os.makedirs('results/samples')
        # --------------------------------------------------------------------------------------------------------------
        def save_fake_images():
            path = 'results/samples/fake_images'
            if not os.path.exists(path):
                os.makedirs(path)

            images = []
            for ctg in range(hp.ctg_dim):
                if hp.ctg_prob[ctg] < 0.01:
                    continue
                ctg_vecs = tf.one_hot(tf.fill([hp.save_img_size], ctg), depth=hp.ctg_dim)
                fake_images = self.model([ctg_vecs, self.save_cnt_vecs])
                images.append(np.vstack(fake_images))

            kr.preprocessing.image.save_img(path=path + '/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.hstack(images), clip_value_min=-1, clip_value_max=1))
        save_fake_images()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/gen.h5')
        np.save('models/ctg_prob.npy', hp.ctg_prob)

    def load(self):
        self.model.load_weights('models/gen.h5')
        hp.ctg_prob.assign(np.load('models/ctg_prob.npy'))

    def to_ema(self):
        self.train_w = [tf.constant(w) for w in self.model.trainable_variables]
        hp.gen_opt.finalize_variable_values(self.model.trainable_variables)

    def to_train(self):
        for ema_w, train_w in zip(self.model.trainable_variables, self.train_w):
            ema_w.assign(train_w)


class Dis(object):
    def build_model(self):
        inp_img = kr.Input([hp.img_res, hp.img_res, hp.img_chn])
        return kr.Model(inp_img, Layers.Encoder()(inp_img))

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/dis.h5')

    def load(self):
        self.model.load_weights('models/dis.h5')


class Cla(object):
    def build_model(self):
        input_image = kr.Input([hp.img_res, hp.img_res, hp.img_chn])
        return kr.Model(input_image, Layers.Encoder()(input_image))

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/cla.h5')

    def load(self):
        self.model.load_weights('models/cla.h5')
