import tensorflow as tf
import numpy as np
from scipy.special import expit

def convert_pred_to_output_np(y_pred, dict_in):

    boxsx = y_pred.shape[2]
    boxsy = y_pred.shape[1]
    # gives values in size relative to whole image used for iou and actually being interpretable
    # number of boxes in each direction used for calculations rather than sizing so x first
    size4calc = [boxsx, boxsy]
    n_bat = y_pred.shape[0]
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes
    thresh = dict_in['iou_threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = np.reshape(y_pred, size1)

    # get top left position of cells
    rowz = np.arange(boxsy, dtype=np.float32)
    colz = np.arange(boxsx, dtype=np.float32)
    rowz = np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx))
    colz = np.reshape(np.tile(colz, boxsy), (boxsy, boxsx))
    rowz = np.expand_dims(rowz, axis=0)
    colz = np.expand_dims(colz, axis=0)
    rowz = np.expand_dims(rowz, axis=-1)
    colz = np.expand_dims(colz, axis=-1)
    tl_cell = np.stack((colz, rowz), axis=4)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = expit(y_pred[:, :, :, :, 4])
    cent_cnn = expit(y_pred[:, :, :, :, 0:2])
    cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    # need to multiply by size of each box, so instead dividing by number of boxes
    cent_cnn = np.add(cent_cnn, tl_cell)
    cent_cnn = np.divide(cent_cnn, size4calc)
    # divide so position is relative to whole image
    # cent_cnn = np.divide(cent_cnn, size4calc)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # keep for loss
    size_cnn_in = size_cnn
    # size is to power of prediction
    size_cnn = np.exp(size_cnn)
    # adjust so size is relative to anchors
    size_cnn = np.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = np.divide(size_cnn, size4calc)
    # if more than one class
    # class_cnn = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    # else
    class_cnn = expit(y_pred[:, :, :, :, 5:])

    # create xmin, ymin, xmax, ymax for boxes
    half_size = np.divide(size_cnn, 2)
    xmin = np.subtract(cent_cnn[:, :, :, :, 0], half_size[:, :, :, :, 0])
    xmax = np.add(cent_cnn[:, :, :, :, 0], half_size[:, :, :, :, 0])
    ymin = np.subtract(cent_cnn[:, :, :, :, 1], half_size[:, :, :, :, 1])
    ymax = np.add(cent_cnn[:, :, :, :, 1], half_size[:, :, :, :, 1])
    class_cnn = class_cnn.flatten()
    confs_cnn = confs_cnn.flatten()
    xmin = xmin.flatten()
    xmax = xmax.flatten()
    ymin = ymin.flatten()
    ymax = ymax.flatten()
    # take out zeros
    mask = np.greater(confs_cnn, thresh)
    class_cnn_out = class_cnn[mask]
    confs_cnn_out = confs_cnn[mask]
    xmin_out = xmin[mask]
    xmax_out = xmax[mask]
    ymin_out = ymin[mask]
    ymax_out = ymax[mask]
    boxes = np.stack([xmin_out, ymin_out, xmax_out, ymax_out], axis=-1)

    return class_cnn_out, confs_cnn_out, boxes



def convert_pred_to_output_tf(y_pred, dict_in):

    boxsx = int(dict_in['boxs_x'])
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    # gives values in size relative to whole image used for iou and actually being interpretable
    # number of boxes in each direction used for calculations rather than sizing so x first
    size4calc = [boxsx, boxsy]
    n_bat = int(dict_in['batch_size'])
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

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

    # create xmin, ymin, xmax, ymax for boxes
    half_size = tf.divide(size_cnn, 2)
    xmin = tf.subtract(cent_cnn[:, :, :, :, 0], half_size[:, :, :, :, 0])
    xmax = tf.add(cent_cnn[:, :, :, :, 0], half_size[:, :, :, :, 0])
    ymin = tf.subtract(cent_cnn[:, :, :, :, 1], half_size[:, :, :, :, 1])
    ymax = tf.add(cent_cnn[:, :, :, :, 1], half_size[:, :, :, :, 1])
    class_cnn = tf.layers.flatten(class_cnn)
    confs_cnn = tf.layers.flatten(confs_cnn)
    xmin = tf.layers.flatten(xmin)
    xmax = tf.layers.flatten(xmax)
    ymin = tf.layers.flatten(ymin)
    ymax = tf.layers.flatten(ymax)
    # take out zeros
    mask = tf.greater(confs_cnn, 0.001)
    class_cnn_out = tf.boolean_mask(class_cnn, mask)
    confs_cnn_out = tf.boolean_mask(confs_cnn, mask)
    xmin_out = tf.boolean_mask(xmin, mask)
    xmax_out = tf.boolean_mask(xmax, mask)
    ymin_out = tf.boolean_mask(ymin, mask)
    ymax_out = tf.boolean_mask(ymax, mask)
    boxes = tf.stack([xmin_out, ymin_out, xmax_out, ymax_out], axis=-1)

    return class_cnn_out, confs_cnn_out, boxes
