import numpy as np
import tensorflow as tf


def convert_pred(y_pred, boxsx, boxsy, anchors):

    # gives values in size relative to whole image used for iou and actually being interpretable
    # number of boxes in each direction used for calculations rather than sizing so x first
    size4calc = [boxsx, boxsy]

    # get top left position of cells
    rowz = np.divide(np.arange(boxsy, dtype=np.float32), boxsy)
    colz = np.divide(np.arange(boxsx, dtype=np.float32), boxsx)
    rowz = np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx))
    colz = np.reshape(np.tile(colz, boxsy), (boxsy, boxsx))
    rowz = tf.convert_to_tensor(rowz)
    colz = tf.convert_to_tensor(colz)
    rowz = tf.expand_dims(rowz, axis=0)
    colz = tf.expand_dims(colz, axis=0)
    rowz = tf.expand_dims(rowz, axis=-1)
    colz = tf.expand_dims(colz, axis=-1)
    tl_cell = tf.stack((colz, rowz), axis=4)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = tf.sigmoid(y_pred[:, :, :, :, 4])
    cent_cnn = tf.sigmoid(y_pred[:, :, :, :, 0:2])
    # divide so position is relative to whole image
    cent_cnn = tf.divide(cent_cnn, size4calc)
    # add to cent_cnn so is position in whole image
    cent_cnn = tf.add(cent_cnn, tl_cell)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # size is to power of prediction
    size_cnn = tf.exp(size_cnn)
    # adjust so size is relative to anchors
    size_cnn = tf.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = tf.divide(size_cnn, size4calc)
    # if more than one class
    # class_cnn = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    # else
    class_cnn = tf.sigmoid(y_pred[:, :, :, :, 5:])

    return confs_cnn, cent_cnn, size_cnn, class_cnn


def convert_pred_for_loss(y_pred):
    # get confidences centres sizes and class predictions from from net_output
    # values direct from predictions used for loss calculation
    cent_cnn = tf.sigmoid(y_pred[:, :, :, :, 0:2])
    size_cnn = y_pred[:, :, :, :, 2:4]

    return cent_cnn, size_cnn


def convert_gt1(gt1):
    # get confidences centres sizes and class predictions from from net_output
    # values direct from predictions used for loss calculation
    confs_true = gt1[:, :, :, :, 4, :]
    confs_true = tf.expand_dims(confs_true, axis=4)
    cent_true = gt1[:, :, :, :, 0:2, :]
    size_true = gt1[:, :, :, :, 2:4, :]
    class_true = gt1[:, :, :, :, 5:, :]

    return confs_true, cent_true, size_true, class_true


def calc_no_conf(gt1, confs_cnn, cent_cnn, size_cnn, thresh):
    confs_true, cent_true, size_true, class_true = convert_gt1(gt1)
    # add an extra dimension to output to match the extra dimension for each groundtruth
    cent_cnn = tf.expand_dims(cent_cnn, axis=-1)
    size_cnn = tf.expand_dims(size_cnn, axis=-1)
    area_cnn = tf.multiply(size_cnn[:, :, :, :, 0, :], size_cnn[:, :, :, :, 1, :])
    area_true = tf.multiply(size_true[:, :, :, :, 0, :], size_true[:, :, :, :, 1, :])
    size_cnn_half = tf.divide(size_cnn, 2)
    size_true_half = tf.divide(size_true, 2)
    min_cnn = tf.subtract(cent_cnn, size_cnn_half)
    min_true = tf.subtract(cent_true, size_true_half)
    max_cnn = tf.add(cent_cnn, size_cnn_half)
    max_true = tf.add(cent_true, size_true_half)
    inter_maxs = tf.minimum(max_cnn, max_true)
    inter_mins = tf.maximum(min_cnn, min_true)
    inter_size = tf.maximum(tf.subtract(inter_maxs, inter_mins), 0)
    inter_area = tf.multiply(inter_size[:, :, :, :, 0, :], inter_size[:, :, :, :, 1, :])
    # subtract seems to collapse
    union_area = tf.subtract(tf.add(area_true, area_cnn), inter_area)
    iou = tf.divide(inter_area, union_area)
    # check threshold will be one if less than threshold or one if less than
    # as calculating ones that don't have a good overlap
    # actually its any that are zero that we want to use, any overlap at all and we discount them
    # actually using a small value due to machine accuracy rounding etc
    over_thresh = tf.cast(tf.less(iou, 1e-3), tf.float32)
    # reduce min so it will keep zero if any of the detections are zero else it will be one.
    over_thresh = tf.reduce_min(over_thresh, axis=-1)
    # calculate confidence losses when no ground truth in cell
    conf_loss_nogt = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(0.0, confs_cnn)), over_thresh))
    return conf_loss_nogt


def calc_no_conf2(gt2, confs_cnn, cent_cnn, size_cnn, thresh):
    confs_true, cent_true, size_true, class_true = convert_gt2(gt2)
    # add an extra dimension to output to match the extra dimension for each groundtruth
    area_cnn = tf.multiply(size_cnn[:, :, :, :, 0], size_cnn[:, :, :, :, 1])
    area_true = tf.multiply(size_true[:, :, :, :, 0], size_true[:, :, :, :, 1])
    size_cnn_half = tf.divide(size_cnn, 2)
    size_true_half = tf.divide(size_true, 2)
    min_cnn = tf.subtract(cent_cnn, size_cnn_half)
    min_true = tf.subtract(cent_true, size_true_half)
    max_cnn = tf.add(cent_cnn, size_cnn_half)
    max_true = tf.add(cent_true, size_true_half)
    inter_maxs = tf.minimum(max_cnn, max_true)
    inter_mins = tf.maximum(min_cnn, min_true)
    inter_size = tf.maximum(tf.subtract(inter_maxs, inter_mins), 0)
    inter_area = tf.multiply(inter_size[:, :, :, :, 0], inter_size[:, :, :, :, 1])
    # subtract seems to collapse
    union_area = tf.subtract(tf.add(area_true, area_cnn), inter_area)
    iou = tf.divide(inter_area, union_area)
    # check threshold will be one if less than threshold or one if less than
    # as calculating ones that don't have a good overlap
    # over_thresh = tf.floor(tf.divide(iou, thresh))
    over_thresh = tf.cast(tf.less(iou, thresh), tf.float32)
    nonesij = tf.subtract(1.0, confs_true)
    # reduce min so it will keep zero if any of the detections are zero else it will be one.
    # over_thresh = tf.reduce_min(over_thresh, axis=-1)
    # calculate confidence losses when no ground truth in cell
    conf_loss_nogt = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(0.0, confs_cnn)), nonesij))
    return conf_loss_nogt


def convert_gt2(gt2):
    # get confidences centres sizes and class predictions from from net_output
    # values direct from predictions used for loss calculation
    confs_true = gt2[:, :, :, :, 4]
    cent_true = gt2[:, :, :, :, 0:2]
    size_true = gt2[:, :, :, :, 2:4]
    class_true = gt2[:, :, :, :, 5:]

    return confs_true, cent_true, size_true, class_true


def convert_gt_loss(cent_true, size_true, boxsx, boxsy):
    # gives values in size relative to whole image used for iou and actually being interpretable
    # number of boxes in each direction used for calculations rather than sizing so x first
    size4calc = [boxsx, boxsy]

    # get top left position of cells
    rowz = np.divide(np.arange(boxsy, dtype=np.float32), boxsy)
    colz = np.divide(np.arange(boxsx, dtype=np.float32), boxsx)
    rowz = np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx))
    colz = np.reshape(np.tile(colz, boxsy), (boxsy, boxsx))
    rowz = tf.convert_to_tensor(rowz)
    colz = tf.convert_to_tensor(colz)
    rowz = tf.expand_dims(rowz, axis=0)
    colz = tf.expand_dims(colz, axis=0)
    rowz = tf.expand_dims(rowz, axis=-1)
    colz = tf.expand_dims(colz, axis=-1)
    tl_cell = tf.stack((colz, rowz), axis=4)
    cent_for_loss = tf.multiply(tf.subtract(cent_true, tl_cell), size4calc)
    size_true_loss = tf.log(tf.add(size_true, 0.00000001))

    return cent_for_loss, size_true_loss


def calc_iou(cent_true, size_true, cent_cnn, size_cnn):

    # there is only one ground truth per detection box which is repeated for each anchor box
    # therefore size_true ans centre_true are the same thing in each anchor box layer
    area_cnn = tf.multiply(size_cnn[:, :, :, :, 0], size_cnn[:, :, :, :, 1])
    area_true = tf.multiply(size_true[:, :, :, :, 0], size_true[:, :, :, :, 1])
    size_cnn_half = tf.divide(size_cnn, 2)
    size_true_half = tf.divide(size_true, 2)
    min_cnn = tf.subtract(cent_cnn, size_cnn_half)
    min_true = tf.subtract(cent_true, size_true_half)
    max_cnn = tf.add(cent_cnn, size_cnn_half)
    max_true = tf.add(cent_true, size_true_half)
    inter_maxs = tf.minimum(max_cnn, max_true)
    inter_mins = tf.maximum(min_cnn, min_true)
    inter_size = tf.maximum(tf.subtract(inter_maxs, inter_mins), 0)
    inter_area = tf.multiply(inter_size[:, :, :, :, 0], inter_size[:, :, :, :, 1])
    union_area = tf.subtract(tf.add(area_true, area_cnn), inter_area)
    iou = tf.divide(inter_area, union_area)
    iou_max = tf.reduce_max(iou, axis=-1)
    iou_max = tf.expand_dims(iou_max, axis=-1)
    # this is because apparently you can't use argmax for loss functions because of differentiability for backprop
    # this will be one at the maximum iou reducing to zero if there is no overlap
    iou_max = tf.where(tf.greater(iou_max, 1e-7), iou_max, tf.fill(iou_max.get_shape(), 1e-7))
    ones_argmax = tf.floor(tf.divide(iou, iou_max))
    iou = tf.multiply(iou, ones_argmax)

    return ones_argmax


def loss_gfrc_yolo(gt1, gt2, y_pred, dict_in):
    # compares output from cnn with ground truth to calculate loss
    n_bat = int(dict_in['batch_size'])
    boxsx = int(dict_in['boxs_x'])
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes
    thresh = dict_in['iou_threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

    # get loss values for iou
    cent_cnn_loss, size_cnn_loss = convert_pred_for_loss(y_pred)
    confs_cnn, cent_cnn, size_cnn, class_cnn = convert_pred(y_pred, boxsx, boxsy, anchors)

    # calculate nogt loss
    conf_loss_nogt = calc_no_conf(gt1, confs_cnn, cent_cnn, size_cnn, thresh)

    confs_true, cent_true, size_true, class_true = convert_gt2(gt2)
    best_iou = calc_iou(cent_true, size_true, cent_cnn, size_cnn)
    confs_cnn_best_iou = tf.multiply(best_iou, confs_cnn)
    onesij = confs_true
    # onesij = tf.cast(tf.greater_equal(iou, thresh), tf.float32)
    test1 = tf.reduce_sum(onesij)

    # calculate confidence losses when ground truth in cell
    conf_loss_gt = tf.multiply(tf.square(tf.subtract(1.0, confs_cnn_best_iou)), onesij)
    conf_loss_gt = tf.reduce_sum(conf_loss_gt)

    cent_for_loss_true, size_for_loss_true = convert_gt_loss(cent_true, size_true, boxsx, boxsy)

    onesij = tf.expand_dims(onesij, axis=-1)
    # calculate centre losses
    cent_loss = tf.multiply(tf.square(tf.subtract(cent_for_loss_true, cent_cnn_loss)), onesij)
    cent_loss = tf.reduce_sum(cent_loss)

    # calculate size losses
    size_loss = tf.multiply(tf.square(tf.subtract(size_for_loss_true, size_cnn_loss)), onesij)
    # size_loss = tf.multiply(tf.square(tf.subtract(tf.sqrt(size_true), tf.sqrt(size_cnn))), onesij)
    # size_loss = tf.multiply(tf.square(tf.subtract(size_true, size_cnn)), onesij)
    size_loss = tf.reduce_sum(size_loss)

    # calculate total positional losses
    pos_loss = tf.add(cent_loss, size_loss)

    # calculate class losses
    class_loss = tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), onesij)
    class_loss = tf.reduce_sum(class_loss)

    # calculate TPs etc
    confs_true = tf.greater(confs_true, thresh)
    confs_cnn = tf.greater(confs_cnn, thresh)
    TP = tf.reduce_sum(tf.cast(tf.logical_and(confs_true, confs_cnn), dtype=tf.float32))
    FP = tf.reduce_sum(tf.cast(tf.logical_and(confs_cnn, tf.logical_not(confs_true)), dtype=tf.float32))
    FN = tf.reduce_sum(tf.cast(tf.logical_and(confs_true, tf.logical_not(confs_cnn)), dtype=tf.float32))
    Re = tf.divide(TP, tf.add(TP, FN))
    Pr = tf.divide(TP, tf.add(TP, FP))
    FPR = tf.divide(FP, tf.add(TP, FP))

    test2 = tf.reduce_sum(gt2)
    test1 = tf.reduce_sum(tf.multiply(tf.subtract(tf.exp(y_pred[:, :, :, :, 2]), gt2[:, :, :, :, 2]), gt2[:, :, :, :, 4]))

    ind_losses = {
        "conf_loss_nogt": conf_loss_nogt,
        "conf_loss_gt": conf_loss_gt,
        "cent_loss": cent_loss,
        "size_loss": size_loss,
        "pos_loss": pos_loss,
        "class_loss": class_loss,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Re": Re,
        "Pr": Pr,
        "FPR": FPR,
        "test1": test1,
        "test2": test2
    }

    return ind_losses


def loss_gfrc_yolo_ws(gt1, gt2, y_pred, dict_in):
    # compares output from cnn with ground truth to calculate loss
    n_bat = int(dict_in['batch_size'])
    boxsx = int(dict_in['boxs_x'])
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes
    thresh = dict_in['iou_threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

    # get loss values for iou
    cent_cnn_loss, size_cnn_loss = convert_pred_for_loss(y_pred)
    confs_cnn, cent_cnn, size_cnn, class_cnn = convert_pred(y_pred, boxsx, boxsy, anchors)

    # convert size cnn for warm start
    ones_cnn = tf.cast(tf.greater(confs_cnn, 0), dtype=tf.float32)
    ones_cnn = tf.expand_dims(ones_cnn, axis=-1)
    anchors_mat = tf.expand_dims(anchors, axis=0)
    anchors_mat = tf.expand_dims(anchors_mat, axis=0)
    anchors_mat = tf.expand_dims(anchors_mat, axis=0)
    size_cnn = tf.multiply(ones_cnn, anchors_mat)

    # calculate nogt loss
    conf_loss_nogt = calc_no_conf(gt1, confs_cnn, cent_cnn, size_cnn, thresh)

    confs_true, cent_true, size_true, class_true = convert_gt2(gt2)
    iou = calc_iou(cent_true, size_true, cent_cnn, size_cnn)
    onesij = confs_true

    # calculate confidence losses when ground truth in cell
    conf_loss_gt = tf.multiply(tf.square(tf.subtract(1.0, confs_cnn)), onesij)
    conf_loss_gt = tf.reduce_sum(conf_loss_gt)

    cent_for_loss_true, size_for_loss_true = convert_gt_loss(cent_true, size_true, boxsx, boxsy)

    onesij = tf.expand_dims(onesij, axis=-1)
    # calculate centre losses
    cent_loss = tf.multiply(tf.square(tf.subtract(cent_for_loss_true, cent_cnn_loss)), onesij)
    cent_loss = tf.reduce_sum(cent_loss)

    # calculate size losses
    size_loss = tf.multiply(tf.square(tf.subtract(size_for_loss_true, size_cnn_loss)), onesij)
    # size_loss = tf.multiply(tf.square(tf.subtract(tf.sqrt(size_true), tf.sqrt(size_cnn))), onesij)
    # size_loss = tf.multiply(tf.square(tf.subtract(size_true, size_cnn)), onesij)
    size_loss = tf.reduce_sum(size_loss)

    # calculate total positional losses
    pos_loss = tf.add(cent_loss, size_loss)

    # calculate class losses
    class_loss = tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), onesij)
    class_loss = tf.reduce_sum(class_loss)

    # calculate TPs etc
    poz = tf.greater_equal(iou, thresh)
    negz = tf.less(iou, thresh)
    truez = tf.greater_equal(confs_true, thresh)
    falsez = tf.less(confs_true, thresh)
    TP = tf.reduce_sum(tf.cast(tf.logical_and(poz, truez), dtype=tf.float32))
    FP = tf.reduce_sum(tf.cast(tf.logical_and(poz, falsez), dtype=tf.float32))
    FN = tf.reduce_sum(tf.cast(tf.logical_and(negz, truez), dtype=tf.float32))
    Re = TP / (TP + FN)
    Pr = TP / (TP + FP)
    FPR = FP / (TP + FP)

    test1 = tf.reduce_sum(gt1)
    test2 = tf.reduce_sum(gt2)

    ind_losses = {
        "conf_loss_nogt": conf_loss_nogt,
        "conf_loss_gt": conf_loss_gt,
        "cent_loss": cent_loss,
        "size_loss": size_loss,
        "pos_loss": pos_loss,
        "class_loss": class_loss,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Re": Re,
        "Pr": Pr,
        "FPR": FPR,
        "test1": size_loss,
        "test2": test2
    }

    return ind_losses


def total_loss_calc(ind_losses, dict_in):

    lam_coord = dict_in['lambda_coord']
    lam_noobj = dict_in['lambda_noobj']
    lam_objct = dict_in['lambda_object']
    lam_class = dict_in['lambda_class']
    lam_size = dict_in['lambda_size']
    conf_loss_nogt = ind_losses['conf_loss_nogt']
    pos_loss = ind_losses['pos_loss']
    conf_loss_gt = ind_losses['conf_loss_gt']
    class_loss = ind_losses['class_loss']
    cent_loss = ind_losses['cent_loss']
    size_loss = ind_losses['size_loss']

    ctl = tf.multiply(lam_coord, cent_loss)
    szl = tf.multiply(lam_size, size_loss)
    nol = tf.multiply(lam_noobj, conf_loss_nogt)
    obl = tf.multiply(lam_objct, conf_loss_gt)
    cll = tf.multiply(lam_class, class_loss)

    total_loss = tf.add(tf.add(tf.add(tf.add(ctl, szl), nol), obl), cll)

    return total_loss


