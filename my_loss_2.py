import numpy as np
import tensorflow as tf


def yolo_loss(dict_in):
    def loss_gfrc_yolo(y_true, y_pred):
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

        """
        # previous loss function
        # restructure ground truth output
        y_true = tf.reshape(y_true, size1)

        # get confidences, centres sizes and classes from ground truth
        conf_true = y_true[:, :, :, :, 4]
        # conf_true = tf.reshape(conf_true, size2)
        centres_true = y_true[:, :, :, :, 0:2]
        # centres true are between 0 and 1 relative to image
        # # paper says centres should be for box not image
        # # divide so position is relative to whole image size
        # # centres_true = tf.divide(centres_true, size3)
        # # add tl position to cent so is position in whole image
        # # centres_true = tf.add(centres_true, tl_cell)
        size_true = y_true[:, :, :, :, 2:4]
        # size true is relative to cell size
        # need to convert to relative to image size
        size_true = tf.divide(size_true, size3)
        class_true = y_true[:, :, :, :, 5:]

        # restructure net_output to same as gt_out
        y_pred = tf.reshape(y_pred, size1)

        # get confidences centres sizes and class predictions from from net_output
        confs_cnn = tf.sigmoid(y_pred[:, :, :, :, 4])
        cent_cnn = tf.sigmoid(y_pred[:, :, :, :, 0:2])
        # divide so position is relative to whole image
        cent_cnn = tf.divide(cent_cnn, size3)
        # add to cent_cnn so is position in whole image
        cent_cnn = tf.add(cent_cnn, tl_cell)
        size_cnn = y_pred[:, :, :, :, 2:4]
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
        iou = tf.reshape(iou, size2)
        negiou = tf.subtract(1.0, iou)

        # calculate losses
        # calculate confidence losses when no ground truth in cell
        conf_loss_nogt = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(0.0, confs_cnn)), noones_ij))
        # The rest of the losses are only calculated in cells where there are ground truths
        conf_loss_gt = tf.multiply(tf.square(tf.subtract(1.0, confs_cnn)), onesij)
        conf_loss_gt = tf.reduce_sum(conf_loss_gt)

        onesij = tf.expand_dims(onesij, axis=-1)
        cent_loss = tf.multiply(tf.square(tf.subtract(centres_true, cent_cnn)), onesij)
        cent_loss = tf.reduce_sum(cent_loss)

        size_loss = tf.multiply(tf.square(tf.subtract(tf.sqrt(size_true), tf.sqrt(size_cnn))), onesij)
        size_loss = tf.reduce_sum(size_loss)
        pos_loss = tf.add(cent_loss, size_loss)

        class_loss = tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), onesij)
        class_loss = tf.reduce_sum(class_loss)
        
        """

        total_loss = tf.add(tf.add(tf.add(tf.multiply(lam_noobj, conf_loss_nogt), tf.multiply(lam_coord, pos_loss)),
                                   tf.multiply(lam_objct, conf_loss_gt)), tf.multiply(lam_class, class_loss))

        return total_loss
    return loss_gfrc_yolo


