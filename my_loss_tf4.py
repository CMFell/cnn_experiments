import numpy as np
import tensorflow as tf

# Working version with double size pictures


def convert_pred(y_pred, boxsx, boxsy, anchors):

    # gives values in size relative to whole image used for iou and actually being interpretable
    # number of boxes in each direction used for calculations rather than sizing so x first
    size4calc = [boxsx, boxsy]

    # get top left position of cells
    # rowz = np.divide(np.arange(boxsy, dtype=np.float32), boxsy)
    rowz = np.arange(boxsy, dtype=np.float32)
    # colz = np.divide(np.arange(boxsx, dtype=np.float32), boxsx)
    colz = np.arange(boxsx, dtype=np.float32)
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
    cent_cnn = tf.add(tl_cell, cent_cnn)
    # divide so position is relative to whole image
    cent_cnn = tf.divide(cent_cnn, size4calc)

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


def pos_true_ws(n_bat, n_anchors, boxsx, boxsy, anchors):
    size4calc = [boxsx, boxsy]
    anchors_whole_image = tf.divide(anchors, size4calc)
    # rowz = np.divide(np.arange(boxsy, dtype=np.float32), boxsy)
    rowz = np.arange(boxsy, dtype=np.float32)
    # colz = np.divide(np.arange(boxsx, dtype=np.float32), boxsx)
    colz = np.arange(boxsx, dtype=np.float32)
    rowz = np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx))
    colz = np.reshape(np.tile(colz, boxsy), (boxsy, boxsx))
    rowz = tf.convert_to_tensor(rowz)
    colz = tf.convert_to_tensor(colz)
    rowz = tf.expand_dims(rowz, axis=0)
    colz = tf.expand_dims(colz, axis=0)
    rowz = tf.expand_dims(rowz, axis=-1)
    colz = tf.expand_dims(colz, axis=-1)
    tl_cell = tf.stack((colz, rowz), axis=4)
    cent_true = np.full((n_bat, boxsy, boxsx, n_anchors, 2), 0.5)
    cent_true = tf.convert_to_tensor(cent_true)
    cent_true = tf.cast(cent_true, tf.float32)
    cent_true = tf.add(cent_true, tl_cell)
    cent_true = tf.divide(cent_true, size4calc)
    tot_rep = n_bat * boxsy * boxsx
    size_true = tf.reshape(tf.tile(anchors_whole_image, [tot_rep, 1]), [n_bat, boxsy, boxsx, n_anchors, 2])
    return cent_true, size_true


def calc_true_ious(gt2, cent_cnn, size_cnn):
    cent_true_x = gt2[:, :, :, :, 0, :]
    cent_true_y = gt2[:, :, :, :, 1, :]
    size_true_x = gt2[:, :, :, :, 2, :]
    size_true_y = gt2[:, :, :, :, 3, :]
    # need to remove weird results due to blank zero sized truths
    filter_out = tf.ceil(size_true_x)

    cent_cnn_x = cent_cnn[:, :, :, :, 0]
    cent_cnn_y = cent_cnn[:, :, :, :, 1]
    size_cnn_x = size_cnn[:, :, :, :, 0]
    size_cnn_y = size_cnn[:, :, :, :, 1]

    cent_cnn_x = tf.expand_dims(cent_cnn_x, axis=-1)
    cent_cnn_y = tf.expand_dims(cent_cnn_y, axis=-1)
    size_cnn_x = tf.expand_dims(size_cnn_x, axis=-1)
    size_cnn_y = tf.expand_dims(size_cnn_y, axis=-1)
    # add an extra dimension to output to match the extra dimension for each groundtruth
    # area_cnn = tf.multiply(size_cnn[:, :, :, :, 0, :], size_cnn[:, :, :, :, 1, :])
    area_cnn = tf.multiply(size_cnn_x, size_cnn_y)
    # area_true = tf.multiply(size_true[:, :, :, :, 0, :], size_true[:, :, :, :, 1, :])
    area_true = tf.multiply(size_true_x, size_true_y)
    size_cnn_half_x = tf.divide(size_cnn_x, 2)
    size_cnn_half_y = tf.divide(size_cnn_y, 2)
    size_true_half_x = tf.divide(size_true_x, 2)
    size_true_half_y = tf.divide(size_true_y, 2)
    min_cnn_x = tf.subtract(cent_cnn_x, size_cnn_half_x)
    min_cnn_y = tf.subtract(cent_cnn_y, size_cnn_half_y)
    min_true_x = tf.subtract(cent_true_x, size_true_half_x)
    min_true_y = tf.subtract(cent_true_y, size_true_half_y)
    max_cnn_x = tf.add(cent_cnn_x, size_cnn_half_x)
    max_cnn_y = tf.add(cent_cnn_y, size_cnn_half_y)
    max_true_x = tf.add(cent_true_x, size_true_half_x)
    max_true_y = tf.add(cent_true_y, size_true_half_y)
    inter_maxs_x = tf.minimum(max_cnn_x, max_true_x)
    inter_maxs_y = tf.minimum(max_cnn_y, max_true_y)
    inter_mins_x = tf.maximum(min_cnn_x, min_true_x)
    inter_mins_y = tf.maximum(min_cnn_y, min_true_y)
    inter_size_x = tf.maximum(tf.subtract(inter_maxs_x, inter_mins_x), 0)
    inter_size_y = tf.maximum(tf.subtract(inter_maxs_y, inter_mins_y), 0)
    # inter_area = tf.multiply(inter_size[:, :, :, :, 0, :], inter_size[:, :, :, :, 1, :])
    inter_area = tf.multiply(inter_size_x, inter_size_y)
    # subtract seems to collapse
    union_area = tf.subtract(tf.add(area_true, area_cnn), inter_area)
    iou = tf.divide(inter_area, union_area)
    iou = tf.multiply(iou, filter_out)
    best_iou = tf.reduce_max(iou, axis=-1)
    return best_iou


def split_mat(mat):
    # get confidences centres sizes and class predictions from from net_output
    # values direct from predictions used for loss calculation
    confs_out = mat[:, :, :, :, 4]
    cent_out = mat[:, :, :, :, 0:2]
    size_out = mat[:, :, :, :, 2:4]
    class_out = mat[:, :, :, :, 5:]

    return confs_out, cent_out, size_out, class_out


def calc_size_cent(size_cnn_raw, cent_cnn_raw, size_true, cent_true, boxsx, boxsy, anchors):
    # set base values
    size4calc = [boxsx, boxsy]
    # get top left position of cells
    rowz = np.arange(boxsy, dtype=np.float32)
    colz = np.arange(boxsx, dtype=np.float32)
    rowz = np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx))
    colz = np.reshape(np.tile(colz, boxsy), (boxsy, boxsx))
    rowz = tf.convert_to_tensor(rowz)
    colz = tf.convert_to_tensor(colz)
    rowz = tf.expand_dims(rowz, axis=0)
    colz = tf.expand_dims(colz, axis=0)
    rowz = tf.expand_dims(rowz, axis=-1)
    colz = tf.expand_dims(colz, axis=-1)
    tl_cell = tf.stack((colz, rowz), axis=4)

    txy = tf.subtract(tf.multiply(cent_true, size4calc), tl_cell)
    # should log here but zeros cause problems
    twh = tf.divide(tf.multiply(size_true, size4calc), anchors)

    cent_loss_x = txy[:, :, :, :, 0] - tf.sigmoid(cent_cnn_raw[:, :, :, :, 0])
    cent_loss_y = txy[:, :, :, :, 1] - tf.sigmoid(cent_cnn_raw[:, :, :, :, 1])
    size_loss_x = twh[:, :, :, :, 0] - tf.exp(size_cnn_raw[:, :, :, :, 0])
    size_loss_y = twh[:, :, :, :, 1] - tf.exp(size_cnn_raw[:, :, :, :, 1])

    return cent_loss_x, cent_loss_y, size_loss_x, size_loss_y


def mag_array(mat):
    mag = tf.reduce_sum(tf.square(mat))
    return mag


def loss_gfrc_yolo(gt1, gt2, y_pred, dict_in):
    # compares output from cnn with ground truth to calculate loss
    n_bat = int(dict_in['batch_size'])
    boxsx = int(dict_in['boxs_x'])
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = int(dict_in['n_anchors'])
    # nanchors = anchors.shape[0]
    # anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    n_classes = int(dict_in['n_classes'])
    # num_out = 5 + n_classes
    num_out = int(dict_in['num_out'])
    thresh = dict_in['iou_threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

    # get predictions for iou
    confs_cnn, cent_cnn, size_cnn, class_cnn = convert_pred(y_pred, boxsx, boxsy, anchors)

    # convert ground truth
    confs_true, cent_true, size_true, class_true = split_mat(gt1)

    # split out predictions
    confs_cnn_raw, cent_cnn_raw, size_cnn_raw, class_cnn_raw = split_mat(y_pred)

    # calculate iou with ground truths
    best_iou = calc_true_ious(gt2, cent_cnn, size_cnn)

    # convert to if overlap or not
    noonesij = tf.cast(tf.less(best_iou, 0.5), tf.float32)
    onesij = confs_true

    # calculate losses when detection matches truth or not
    conf_loss_nogt = tf.multiply(tf.subtract(0.0, confs_cnn), noonesij)
    conf_loss_gt = tf.multiply(tf.subtract(1.0, confs_cnn), onesij)

    # calculate positional losses for all detections
    ctl_x, ctl_y, szl_x, szl_y = calc_size_cent(size_cnn_raw, cent_cnn_raw, size_true, cent_true, boxsx, boxsy, anchors)

    # filter positional losses only at true detections
    ctl_x = tf.multiply(ctl_x, onesij)
    ctl_y = tf.multiply(ctl_y, onesij)
    szl_x = tf.multiply(szl_x, onesij)
    szl_y = tf.multiply(szl_y, onesij)
    test1 = tf.reduce_min(ctl_x)

    # calculate class losses
    onesij = tf.expand_dims(onesij, axis=-1)
    class_loss = tf.multiply(tf.subtract(1.0, class_cnn), onesij)

    # calculate all losses single figures
    ngt_loss = mag_array(conf_loss_nogt)
    gt_loss = mag_array(conf_loss_gt)
    ct_loss = tf.add(mag_array(ctl_x), mag_array(ctl_y))
    sz_loss = tf.add(mag_array(szl_x), mag_array(szl_y))
    cl_loss = mag_array(class_loss)
    pos_loss = tf.add(ct_loss, sz_loss)

    # calculate TPs etc
    poz = tf.greater_equal(best_iou, thresh)
    negz = tf.less(best_iou, thresh)
    truez = tf.greater_equal(confs_true, thresh)
    falsez = tf.less(confs_true, thresh)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(poz, truez), dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(poz, falsez), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(negz, truez), dtype=tf.float32))
    re = tp / (tp + fn)
    pr = tp / (tp + fp)
    fpr = fp / (tp + fp)

    # test1 = tf.reduce_sum(noonesij)
    test2 = tf.reduce_sum(gt1)

    ind_losses = {
        "conf_loss_nogt": ngt_loss,
        "conf_loss_gt": gt_loss,
        "cent_loss": ct_loss,
        "size_loss": sz_loss,
        "pos_loss": pos_loss,
        "class_loss": cl_loss,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Re": re,
        "Pr": pr,
        "FPR": fpr,
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
    nanchors = int(dict_in['n_anchors'])
    #nanchors = anchors.shape[0]
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    n_classes = int(dict_in['n_classes'])
    #num_out = 5 + n_classes
    num_out = int(dict_in['num_out'])
    thresh = dict_in['iou_threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

    # get predictions for iou
    confs_cnn, cent_cnn, size_cnn, class_cnn = convert_pred(y_pred, boxsx, boxsy, anchors)

    # convert ground truth
    confs_true, cent_true, size_true, class_true = split_mat(gt1)

    # split out predictions
    confs_cnn_raw, cent_cnn_raw, size_cnn_raw, class_cnn_raw = split_mat(y_pred)

    # calculate iou with ground truths
    best_iou = calc_true_ious(gt2, cent_cnn, size_cnn)

    # convert to if overlap or not
    noonesij = tf.cast(tf.less(best_iou, 0.5), tf.float32)
    onesij = confs_true

    # calculate losses when detection matches truth or not
    conf_loss_nogt = tf.multiply(tf.subtract(0.0, confs_cnn), noonesij)
    conf_loss_gt = tf.multiply(tf.subtract(best_iou, confs_cnn), onesij)

    # set detection positions at centre of cell of size anchor box
    cent_true, size_true = pos_true_ws(n_bat, nanchors, boxsx, boxsy, anchors)

    # calculate positional losses for all detections
    ctl_x, ctl_y, szl_x, szl_y = calc_size_cent(size_cnn_raw, cent_cnn_raw, size_true, cent_true, boxsx, boxsy, anchors)
    test1 = tf.reduce_max(onesij)
    # calculate class losses
    onesij = tf.expand_dims(onesij, axis=-1)
    class_loss = tf.multiply(tf.subtract(1.0, class_cnn), onesij)

    # calculate all losses single figures
    ngt_loss = mag_array(conf_loss_nogt)
    gt_loss = mag_array(conf_loss_gt)
    ct_loss = tf.add(mag_array(ctl_x), mag_array(ctl_y))
    sz_loss = tf.add(mag_array(szl_x), mag_array(szl_y))
    cl_loss = mag_array(class_loss)
    pos_loss = tf.add(ct_loss, sz_loss)

    # calculate TPs etc
    poz = tf.greater_equal(best_iou, thresh)
    negz = tf.less(best_iou, thresh)
    truez = tf.greater_equal(confs_true, thresh)
    falsez = tf.less(confs_true, thresh)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(poz, truez), dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(poz, falsez), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(negz, truez), dtype=tf.float32))
    re = tp / (tp + fn)
    pr = tp / (tp + fp)
    fpr = fp / (tp + fp)

    test2 = tf.reduce_sum(mag_array(gt1))

    ind_losses = {
        "conf_loss_nogt": ngt_loss,
        "conf_loss_gt": gt_loss,
        "cent_loss": ct_loss,
        "size_loss": sz_loss,
        "pos_loss": pos_loss,
        "class_loss": cl_loss,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Re": re,
        "Pr": pr,
        "FPR": fpr,
        "test1": test1,
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
    # pos_loss = ind_losses['pos_loss']
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
