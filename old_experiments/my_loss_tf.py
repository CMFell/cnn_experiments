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
    cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    cent_cnn = tf.add(cent_cnn, tl_cell)
    # divide so position is relative to whole image
    cent_cnn = tf.divide(cent_cnn, size4calc)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # keep for loss
    size_cnn_in = size_cnn
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
    confs_cnn = tf.sigmoid(y_pred[:, :, :, :, 4])
    cent_cnn = tf.sigmoid(y_pred[:, :, :, :, 0:2])
    size_cnn = y_pred[:, :, :, :, 2:4]
    class_cnn = tf.sigmoid(y_pred[:, :, :, :, 5:])

    return confs_cnn, cent_cnn, size_cnn, class_cnn


def convert_gt1(gt1):
    # get confidences centres sizes and class predictions from from net_output
    # values direct from predictions used for loss calculation
    confs_true = gt1[:, :, :, :, 4, :]
    cent_true = gt1[:, :, :, :, 0:2]
    size_true = gt1[:, :, :, :, 2:4]
    class_true = gt1[:, :, :, :, 5:]

    return confs_true, cent_true, size_true, class_true


def calc_no_conf(gt1, confs_cnn, cent_cnn, size_cnn, thresh):
    # convert truths
    confs_true, cent_true, size_true, class_true = convert_gt1(gt1)
    # size_true = tf.expand_dims(size_true, axis=-1)
    area_cnn = tf.multiply(size_cnn[:, :, :, :, 0], size_cnn[:, :, :, :, 1])
    area_true = tf.multiply(size_true[:, :, :, :, 0, :], size_true[:, :, :, :, 1, :])
    size_cnn_half = tf.divide(size_cnn, 2)
    size_true_half = tf.divide(size_true, 2)
    min_cnn = tf.subtract(cent_cnn, size_cnn_half)
    min_true = tf.subtract(cent_true, size_true_half)
    max_cnn = tf.add(cent_cnn, size_cnn_half)
    max_true = tf.add(cent_true, size_true_half)
    max_cnn = tf.expand_dims(max_cnn, axis=-1)
    min_cnn = tf.expand_dims(min_cnn, axis=-1)
    area_cnn = tf.expand_dims(area_cnn, axis=-1)
    inter_maxs = tf.minimum(max_cnn, max_true)
    inter_mins = tf.maximum(min_cnn, min_true)
    inter_size = tf.maximum(tf.subtract(inter_maxs, inter_mins), 0)
    inter_area = tf.multiply(inter_size[:, :, :, :, 0, :], inter_size[:, :, :, :, 1, :])
    union_area = tf.subtract(tf.add(area_true, area_cnn), inter_area)
    iou = tf.divide(inter_area, union_area)
    # check threshold will be one if greater than threshold or zero if less than
    over_thresh = tf.floor(tf.divide(iou, thresh))
    over_thresh = tf.reduce_max(over_thresh, axis=-1)
    # calculate confidence losses when no ground truth in cell
    conf_loss_nogt = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(0.0, confs_cnn)), over_thresh))
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
    negiou = tf.subtract(iou_max, iou)
    iou_max = tf.where(tf.greater(iou_max, 1e-7), iou_max, tf.fill(iou_max.get_shape(), 1e-7))
    negiou = tf.divide(negiou, iou_max)
    onesij = tf.floor(negiou)
    iou = tf.multiply(iou, onesij)

    return iou, onesij


def loss_gfrc_yolo(gt1, gt2, y_pred, dict_in):
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

    return total_loss


