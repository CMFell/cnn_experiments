import numpy as np
import tensorflow as tf
from my_loss_tf import convert_pred, convert_pred_for_loss, calc_no_conf, convert_gt2, convert_gt_loss, calc_iou


def metrics_gfrc_yolo(gt1, gt2, y_pred, dict_in):
    # compares output from cnn with ground truth to calculate loss
    # only for one image at the moment
    n_bat = int(dict_in['batch_size'])
    boxsx = int(dict_in['boxs_x'])
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes
    lam_coord = dict_in['lambda_coord']
    lam_noobj = dict_in['lambda_noobj']
    lam_objct = dict_in['lambda_object']
    lam_class = dict_in['lambda_class']
    thresh = dict_in['iou_threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

    # get loss values for iou
    confs_cnn, cent_cnn_loss, size_cnn_loss, class_cnn = convert_pred_for_loss(y_pred)
    confs_cnn, cent_cnn, size_cnn, class_cnn = convert_pred(y_pred, boxsx, boxsy, anchors)

    # calculate nogt loss
    conf_loss_nogt = calc_no_conf(gt1, confs_cnn, cent_cnn, size_cnn, thresh)

    confs_true, cent_true, size_true, class_true = convert_gt2(gt2)
    iou, onesij = calc_iou(cent_true, size_true, cent_cnn, size_cnn)

    # calculate confidence losses when ground truth in cell
    conf_loss_gt = tf.multiply(tf.square(tf.subtract(1.0, confs_cnn)), onesij)
    conf_loss_gt = tf.reduce_sum(conf_loss_gt)

    cent_for_loss_true, size_for_loss_true = convert_gt_loss(cent_true, size_true, boxsx, boxsy)

    onesij = tf.expand_dims(onesij, axis=-1)
    # calculate centre losses
    cent_loss = tf.multiply(tf.square(tf.subtract(cent_for_loss_true, cent_cnn_loss)), onesij)
    cent_loss = tf.reduce_sum(cent_loss)

    # calculate size lossses
    size_loss = tf.multiply(tf.square(tf.subtract(size_for_loss_true, size_cnn_loss)), onesij)
    size_loss = tf.reduce_sum(size_loss)

    # calculate total positional losses
    pos_loss = tf.add(cent_loss, size_loss)

    # calculate class losses
    class_loss = tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), onesij)
    class_loss = tf.reduce_sum(class_loss)

    total_loss = tf.add(tf.add(tf.add(tf.multiply(lam_noobj, conf_loss_nogt), tf.multiply(lam_coord, pos_loss)),
                               tf.multiply(lam_objct, conf_loss_gt)), tf.multiply(lam_class, class_loss))

    confs_true = tf.greater(confs_true, thresh)
    confs_cnn = tf.greater(confs_cnn, thresh)
    TP = tf.reduce_sum(tf.cast(tf.logical_and(confs_true, confs_cnn), dtype=tf.float32))
    FP = tf.reduce_sum(tf.cast(tf.logical_and(confs_cnn, tf.logical_not(confs_true)), dtype=tf.float32))
    FN = tf.reduce_sum(tf.cast(tf.logical_and(confs_true, tf.logical_not(confs_cnn)), dtype=tf.float32))
    Re = TP / (TP + FN)
    Pr = TP / (TP + FP)
    FPR = FP / (TP + FP)

    dict_out = {"total_loss": total_loss, "conf_loss_nogt": conf_loss_nogt, "conf_loss_gt": conf_loss_gt,
                "cent_loss": cent_loss, "size_loss": size_loss, "class_loss": class_loss, "TP": TP, "FP": FP, "FN": FN,
                "Re": Re, "Pr": Pr, "FPR": FPR}

    return dict_out


