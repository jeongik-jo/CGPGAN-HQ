import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _train_step(dis: kr.Model, cla: kr.Model, gen: kr.Model, real_imgs: tf.Tensor, update_ctg_prob):
    batch_size = real_imgs.shape[0]
    cnt_vecs = hp.cnt_dist_func(batch_size)
    ctg_vecs = hp.ctg_dist_func(batch_size)
    fake_imgs = gen([ctg_vecs, cnt_vecs])

    with tf.GradientTape(persistent=True) as dis_cla_tape:
        real_ctg_probs = tf.nn.softmax(cla(real_imgs))
        if not update_ctg_prob:
            real_ctg_probs = real_ctg_probs - tf.reduce_mean(real_ctg_probs, axis=0, keepdims=True) + 1.0 / hp.ctg_dim
        real_ctg_vecs = tf.one_hot(tf.argmax(real_ctg_probs, axis=-1), depth=hp.ctg_dim)
        real_adv_vals = tf.reduce_sum(dis(real_imgs) * real_ctg_vecs, axis=-1)

        with tf.GradientTape(persistent=True) as reg_tape:
            reg_tape.watch(fake_imgs)
            fake_ctg_logits = cla(fake_imgs)
            ctg_losses = tf.losses.categorical_crossentropy(ctg_vecs, fake_ctg_logits, from_logits=True)
            fake_adv_vals = tf.reduce_sum(dis(fake_imgs) * ctg_vecs, axis=-1)
            ctg_reg_scores = tf.square(1 - tf.reduce_sum(tf.nn.softmax(fake_ctg_logits) * ctg_vecs, axis=-1))

        adv_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(fake_adv_vals, fake_imgs)), axis=[1, 2, 3])
        ctg_reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(ctg_reg_scores, fake_imgs)), axis=[1, 2, 3])

        dis_adv_losses = tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals)

        dis_losses = dis_adv_losses + hp.adv_reg_w * adv_reg_losses
        cla_losses = hp.ctg_w * ctg_losses + hp.ctg_reg_w * ctg_reg_losses

        dis_loss = tf.reduce_mean(dis_losses)
        cla_loss = tf.reduce_mean(cla_losses)

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_cla_tape)
    hp.cla_opt.minimize(cla_loss, cla.trainable_variables, tape=dis_cla_tape)

    acc = 1 - tf.math.count_nonzero(tf.argmax(fake_ctg_logits, axis=-1) - tf.argmax(ctg_vecs, axis=-1)) / batch_size

    cnt_vecs = hp.cnt_dist_func(batch_size)
    ctg_vecs = hp.ctg_dist_func(batch_size)

    with tf.GradientTape() as gen_tape:
        fake_x = gen([ctg_vecs, cnt_vecs])
        fake_adv_vals = tf.reduce_sum(dis(fake_x) * ctg_vecs, axis=-1)
        gen_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_vals))

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, tape=gen_tape)
    hp.ctg_prob.assign(hp.ctg_prob * hp.decay_rate + tf.reduce_mean(real_ctg_probs, axis=0) * (1 - hp.decay_rate))

    results = {
        'real_adv_vals': real_adv_vals, 'fake_adv_vals': fake_adv_vals,
        'ctg_losses': ctg_losses,
        'ctg_reg_losses': ctg_reg_losses, 'adv_reg_losses': adv_reg_losses,
        'acc': [acc]
    }
    return results


def train(dis: kr.Model, cla: kr.Model, gen: kr.Model, dataset, epoch):
    results = {}
    for real_imgs in dataset:
        batch_results = _train_step(dis, cla, gen, real_imgs, hp.ctg_update_start_epoch <= epoch)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    results = temp_results

    results['ctg_ent'] = hp.calc_ctg_ent()

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())
    print('%-30s:' % 'ctg_prob', hp.ctg_prob.numpy())

    return results
