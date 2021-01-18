import tensorflow as tf


def metric_cf_ngt(indlosses):
    met = indlosses["conf_loss_nogt"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_cf_gt(indlosses):
    met = indlosses["conf_loss_gt"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_cnt(indlosses):
    met = indlosses["cent_loss"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_sz(indlosses):
    met = indlosses["size_loss"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_cl(indlosses):
    met = indlosses["class_loss"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_tp(indlosses):
    met = indlosses["TP"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_fp(indlosses):
    met = indlosses["FP"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_fn(indlosses):
    met = indlosses["FN"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_re(indlosses):
    met = indlosses["Re"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_pr(indlosses):
    met = indlosses["Pr"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_fpr(indlosses):
    met = indlosses["FPR"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_tl(ind_losses, dict_in):

    lam_coord = dict_in['lambda_coord']
    lam_noobj = dict_in['lambda_noobj']
    lam_objct = dict_in['lambda_object']
    lam_class = dict_in['lambda_class']
    conf_loss_nogt = ind_losses['conf_loss_nogt']
    pos_loss = ind_losses['pos_loss']
    conf_loss_gt = ind_losses['conf_loss_gt']
    class_loss = ind_losses['class_loss']

    met = tf.add(tf.add(tf.add(tf.multiply(lam_noobj, conf_loss_nogt), tf.multiply(lam_coord, pos_loss)),
                               tf.multiply(lam_objct, conf_loss_gt)), tf.multiply(lam_class, class_loss))

    met, update_op = tf.metrics.mean(met)

    return met, update_op


def metric_t1(indlosses):
    met = indlosses["test1"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op


def metric_t2(indlosses):
    met = indlosses["test2"]
    met, update_op = tf.metrics.mean(met)
    return met, update_op
