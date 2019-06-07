import numpy as np
import tensorflow as tf

from keras import backend as K

"""
# define values for calculating loss
input_shape = (307, 460, 3)
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
box_size = [16, 16]
anchor_pixel = np.multiply(anchors_in, box_size)
img_x_pix = input_shape[1]
img_y_pix = input_shape[0]
boxs_x = np.ceil(img_x_pix / 16)
boxs_y = np.ceil(img_y_pix / 16)
classes_in = 1
classes_in = 1
lambda_cl = 1.0
lambda_no = 0.5
lambda_ob = 1.0
lambda_cd = 5.0

base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
batch_size = 32
dict_in1 = {'boxs_x': boxs_x, 'boxs_y': boxs_y, 'img_x_pix': img_x_pix, 'img_y_pix': img_y_pix,
           'anchors': anchors_in, 'n_classes': classes_in, 'lambda_coord': lambda_cd, 'lambda_noobj': lambda_no,
           'base_dir': base_dir, 'batch_size': batch_size, 'lambda_class': lambda_cl, 'lambda_object': lambda_ob}
"""


def metric_TFPN(y_true, y_pred, dict_in):
    # compares output from cnn with ground truth to calculate loss
    # only for one image at the moment

    # n_bat = y_true.shape[0]
    n_bat = int(dict_in['batch_size'])
    # boxsx = y_true.shape[2]
    boxsx = int(dict_in['boxs_x'])
    # boxsy = y_true.shape[1]
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    # num_out = y_true.shape[3] / nanchors
    # n_classes = num_out - 5
    n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes
    lam_coord = dict_in['lambda_coord']
    lam_noobj = dict_in['lambda_noobj']
    thresh = 0.5

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]

    # size of output layer
    size4 = [n_bat, boxsy, boxsx, nanchors * num_out]

    # get top left position of cells
    rowz = np.divide(np.arange(boxsy), boxsy)
    colz = np.divide(np.arange(boxsx), boxsx)
    rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    tl_cell = np.stack((colno, rowno), axis=4)

    # restructure ground truth output
    y_true = K.reshape(y_true, size1)

    # get confidences, centres sizes and classes from ground truth
    conf_true = y_true[:, :, :, :, 4]
    conf_true = K.reshape(conf_true, size2)
    # compress so only looking over all anchor boxes
    # conf_true = K.greater(K.sum(conf_true, axis=3), 0)
    # conf_true = tf.cast(conf_true, dtype=tf.float32)

    # restructure net_output to same as gt_out
    y_pred = K.reshape(y_pred, size1)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = K.sigmoid(y_pred[:, :, :, :, 4])
    # confs_cnn = K.reshape(confs_cnn, size2)
    confs_cnn = K.greater(confs_cnn, thresh)
    confs_cnn = tf.cast(confs_cnn, dtype=tf.float32)
    # compress as only looking over all anchor boxes
    # confs_cnn = K.greater(K.sum(confs_cnn, axis=3), 0)
    # confs_cnn = tf.cast(confs_cnn, dtype=tf.float32)

    # calculate metrics
    confs_cnn = tf.greater(confs_cnn, 0)
    conf_true = tf.greater(conf_true, 0)
    TP = K.sum(tf.cast(tf.logical_and(conf_true, confs_cnn), dtype=tf.float32))
    FP = K.sum(tf.cast(tf.logical_and(confs_cnn, tf.logical_not(conf_true)), dtype=tf.float32))
    FN = K.sum(tf.cast(tf.logical_and(conf_true, tf.logical_not(confs_cnn)), dtype=tf.float32))
    Re = TP / (TP + FN)
    Pr = TP / (TP + FP)
    FPR = FP / (TP + FP)

    output = [TP, FP, FN, Re, Pr, FPR]

    return output


def metric_yolo_TP(dict_in):
    def metric_TP(y_true, y_pred):
        rez = metric_TFPN(y_true, y_pred, dict_in)
        return rez[0]
    return metric_TP

def metric_yolo_FP(dict_in):
    def metric_FP(y_true, y_pred):
        rez = metric_TFPN(y_true, y_pred, dict_in)
        return rez[1]
    return metric_FP

def metric_yolo_FN(dict_in):
    def metric_FN(y_true, y_pred):
        rez = metric_TFPN(y_true, y_pred, dict_in)
        return rez[2]
    return metric_FN


def loss_yolo_metric(y_true, y_pred, dict_in):
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

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]

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
    out_test6 = tl_cell[0, 7, 23, 0, 0]

    n = 3
    out_test = y_true[0, 7, 23, 17:23]
    # out_test = [0, 0, 0, 0, 0, 0]
    # restructure ground truth output
    y_true = tf.reshape(y_true, size1)

    # get confidences, centres sizes and classes from ground truth
    conf_true = y_true[:, :, :, :, 4]
    # conf_true = tf.reshape(conf_true, size2)
    centres_true = y_true[:, :, :, :, 0:2]
    # centres true are between 0 and 1 relative to image
    # require this to calculate iou
    # to calc loss need to be between 0 and 1 for cell
    centres_for_loss = tf.multiply(tf.subtract(centres_true, tl_cell), size3)
    size_true = y_true[:, :, :, :, 2:4]
    # size true is relative to cell size save for loss
    # add a small value so don't get nan error for zeros (nb later multiply by zero if zero anyway)
    size_true_loss = tf.log(tf.add(size_true, 0.00000001))
    # need to convert to relative to image size for iou
    size_true = tf.divide(size_true, size3)
    class_true = y_true[:, :, :, :, 5:]

    # restructure net_output to same as gt_out
    y_pred = tf.reshape(y_pred, size1)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = tf.sigmoid(y_pred[:, :, :, :, 4])
    cent_cnn = tf.sigmoid(y_pred[:, :, :, :, 0:2])
    cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    cent_cnn = tf.add(cent_cnn, tl_cell)
    # divide so position is relative to whole image
    cent_cnn = tf.divide(cent_cnn, size3)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # keep for loss
    size_cnn_in = size_cnn
    # size is to power of prediction
    size_cnn = tf.exp(size_cnn)
    # adjust so size is relative to anchors
    size_cnn = tf.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = tf.divide(size_cnn, size3)
    # if more than one class
    # class_cnn = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    # else
    class_cnn = tf.sigmoid(y_pred[:, :, :, :, 5:])

    # calculate cells with and without ground truths
    # only one ground truth for all anchor boxes
    onesij = conf_true
    out_test1 = tf.reduce_max(onesij)
    out_test2 = tf.reduce_min(onesij)
    noones_ij = tf.subtract(1.0, onesij)

    # there is only one ground truth per detection box which is repeated for each anchor box
    # therefore size_true ans centre_true are the same thing in each anchor box layer
    area_cnn = tf.multiply(size_cnn[:, :, :, :, 0], size_cnn[:, :, :, :, 1])
    area_true = tf.multiply(size_true[:, :, :, :, 0], size_true[:, :, :, :, 1])
    size_cnn_half = tf.divide(size_cnn, 2)
    size_true_half = tf.divide(size_true, 2)
    min_cnn = tf.subtract(cent_cnn, size_cnn_half)
    min_true = tf.subtract(centres_true, size_true_half)
    max_cnn = tf.add(cent_cnn, size_cnn_half)
    max_true = tf.add(centres_true, size_true_half)
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
    iou_max = tf.add(iou_max, 0.00001)
    negiou = tf.divide(negiou, iou_max)
    negiou = tf.subtract(1.0, negiou)
    # negiou = tf.subtract(1.0, tf.divide(tf.subtract(iou, tf.add(iou_max, 0.0000001)), iou_max))
    onesij = tf.multiply(onesij, negiou)
    noones_ij = tf.subtract(1.0, onesij)



    # calculate losses
    # calculate confidence losses when no ground truth in cell
    conf_loss_nogt = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(0.0, confs_cnn)), noones_ij))

    # calculate confidence losses when ground truth in cell
    conf_loss_gt = tf.multiply(tf.square(tf.subtract(1.0, confs_cnn)), onesij)
    conf_loss_gt = tf.reduce_sum(conf_loss_gt)

    onesij = tf.expand_dims(onesij, axis=-1)
    # calculate centre losses
    cent_loss = tf.multiply(tf.square(tf.subtract(centres_for_loss, cent_cnn_in)), onesij)
    cent_loss = tf.reduce_sum(cent_loss)

    # calculate size lossses
    size_loss = tf.multiply(tf.square(tf.subtract(size_true_loss, size_cnn_in)), onesij)
    size_loss = tf.reduce_sum(size_loss)

    # calculate total positional losses
    pos_loss = tf.add(cent_loss, size_loss)

    # calculate class losses
    class_loss = tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), onesij)
    class_loss = tf.reduce_sum(class_loss)

    total_loss = tf.add(tf.add(tf.add(tf.multiply(lam_noobj, conf_loss_nogt), tf.multiply(lam_coord, pos_loss)),
                               tf.multiply(lam_objct, conf_loss_gt)), tf.multiply(lam_class, class_loss))

    conf_loss_gt = tf.multiply(lam_objct, conf_loss_gt)
    conf_loss_nogt = tf.multiply(lam_noobj, conf_loss_nogt)
    cent_loss = tf.multiply(lam_coord, cent_loss)
    size_loss = tf.multiply(lam_coord, size_loss)
    class_loss = tf.multiply(lam_class, class_loss)

    #out_test1 = onesij[0, 7, 23, n, 0]
    #out_test2 = size_true_loss[0, 7, 23, n, 0]
    #out_test3 = size_true_loss[0, 7, 23, n, 1]
    #out_test4 = size_cnn_in[0, 7, 23, n, 0]
    #out_test5 = size_cnn_in[0, 7, 23, n, 1]

    #out_test1 = anchors[0, 1]
    #out_test2 = anchors[0, 1]
    #out_test3 = anchors[1, 0]
    #out_test4 = anchors[1, 1]
    #out_test5 = anchors[2, 0]
    #out_test6 = anchors[0, 1]
    out_test1 = tf.reduce_max(onesij)
    out_test2 = tf.reduce_min(onesij)
    out_test3 = tf.reduce_max(negiou)
    out_test4 = tf.reduce_min(negiou)
    out_test5 = tf.reduce_max(iou_max)
    out_test6 = tf.reduce_min(iou_max)




    output = [total_loss, conf_loss_nogt, conf_loss_gt, cent_loss, size_loss, class_loss, out_test1, out_test2, out_test3, out_test4, out_test5, out_test6]

    return output


def metric_yolo_tl(dict_in):
    def metric_tl(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[0]
    return metric_tl


def metric_yolo_cfno(dict_in):
    def metric_cfno(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[1]
    return metric_cfno


def metric_yolo_cf(dict_in):
    def metric_cf(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[2]
    return metric_cf


def metric_yolo_ct(dict_in):
    def metric_ct(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[3]
    return metric_ct


def metric_yolo_sz(dict_in):
    def metric_sz(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[4]
    return metric_sz


def metric_yolo_cl(dict_in):
    def metric_cl(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[5]
    return metric_cl


def metric_yolo_ot1(dict_in):
    def metric_1ot(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[6]
    return metric_1ot

def metric_yolo_ot4(dict_in):
    def metric_4ot(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[9]
    return metric_4ot

def metric_yolo_ot2(dict_in):
    def metric_2ot(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[7]
    return metric_2ot

def metric_yolo_ot3(dict_in):
    def metric_3ot(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[8]
    return metric_3ot

def metric_yolo_ot5(dict_in):
    def metric_5ot(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[10]
    return metric_5ot

def metric_yolo_ot6(dict_in):
    def metric_6ot(y_true, y_pred):
        rez = loss_yolo_metric(y_true, y_pred, dict_in)
        return rez[11]
    return metric_6ot