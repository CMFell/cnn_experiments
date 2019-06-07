import numpy as np
import tensorflow as tf

# Working version with double size pictures


def get_grid(gridw, gridh, batchsz):
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(gridw), [gridh]), (1, gridh, gridw, 1, 1)))
    # ridiculous equivalent of np.repeat
    cell_y = tf.reshape(tf.tile(tf.reshape(tf.range(gridh), [-1, 1]), [1, gridw]), [-1])
    # tile and reshape in same way as cell_x
    cell_y = tf.to_float(tf.reshape(cell_y, (1, gridh, gridw, 1, 1)))
    # combine to give grid
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batchsz, 1, 1, 5, 1])
    #cell_grid = tf.convert_to_tensor(cell_grid, dtype=tf.float32)

    return cell_grid


def adjust_pred(y_pred):
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2])

    # adjust w and h
    pred_box_wh = y_pred[..., 2:4]
    # shouldn't matter about the clip as these really high values are then multiplied by zeros later
    # pred_box_wh = tf.minimum(pred_box_wh, 100.0)
    # If I've got this right this should stop it reducing box sizes to zero
    # pred_box_wh = tf.maximum(pred_box_wh, -1.)

    # adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    # adjust class probabilities
    pred_box_class = tf.sigmoid(y_pred[..., 5])

    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class


def adjust_pred_wi(pred_box_xy, pred_box_wh, cell_grid, gridw, gridh, anchors, box):
    # new line convert to whole image
    pred_box_xy_wi = pred_box_xy + cell_grid
    pred_box_xy_wi = tf.divide(pred_box_xy_wi, [gridw, gridh])

    # this clips to prevent Inf values after exponentiating,
    # doesn't matter about the clip as these really high values are then multiplied by zeros later
    # pred_box_wh = tf.minimum(pred_box_wh, 2.)
    # new line adjust so relative to whole image
    pred_box_wh_wi = tf.exp(pred_box_wh) * tf.reshape(anchors, [1, 1, 1, box, 2])
    pred_box_wh_wi = tf.divide(pred_box_wh_wi, [gridw, gridh])

    return pred_box_xy_wi, pred_box_wh_wi


def adjust_gt(y_true, cell_grid, gridw, gridh, anchors, box):
    # adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
    # add new line give relative to whole image
    true_box_xy_wi = tf.divide(tf.add(true_box_xy, cell_grid), [gridw, gridh])

    # get w and h
    true_box_wh_wi = y_true[..., 2:4]
    # adjust w and h
    true_box_wh = tf.multiply(true_box_wh_wi, [gridw, gridh])
    true_box_wh = true_box_wh / tf.reshape(anchors, [1, 1, 1, box, 2])
    true_box_wh = tf.log(true_box_wh + 0.00001)
    # the + 0.00001 takes out zeros which can't be logged these should then be multiplied by zero again later

    # adjust confidence
    true_wh_half = true_box_wh_wi / 2.
    true_mins = true_box_xy_wi - true_wh_half
    true_maxes = true_box_xy_wi + true_wh_half
    true_areas = true_box_wh_wi[..., 0] * true_box_wh_wi[..., 1]

    return true_box_xy, true_box_wh, true_mins, true_maxes, true_areas

def ious_centre_cell(pred_box_wh_wi, pred_box_xy_wi, true_mins, true_maxes, true_areas):
    pred_wh_half = pred_box_wh_wi / 2.
    pred_mins = pred_box_xy_wi - pred_wh_half
    pred_mins = tf.maximum(pred_mins, 0.)
    pred_maxes = pred_box_xy_wi + pred_wh_half
    pred_maxes = tf.minimum(pred_maxes, 1.)

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_size = tf.subtract(pred_maxes, pred_mins)
    # pred_areas = pred_box_wh_wi[..., 0] * pred_box_wh_wi[..., 1]
    pred_areas = pred_size[..., 0] * pred_size[..., 1]

    # add a small amount to avoide divide by zero, will later be multiplied by zero
    union_areas = pred_areas + true_areas - intersect_areas + 0.00001
    iou_scores = tf.divide(intersect_areas, union_areas)
    # try to prevent it decreasing iouscores to zero instead of improving confidence
    iou_scores = tf.maximum(iou_scores, 0.05)

    return iou_scores, union_areas


def create_masks(true_boxes, cell_grid, gridw, gridh, pred_box_xy_wi, pred_box_wh_wi, no_obj_thresh):
    # confidence mask: penalize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy_wi = true_boxes[..., 0:2]
    # true_xy_wi = tf.divide(tf.add(true_xy, tf.expand_dims(cell_grid, axis=4)), [gridw, gridh])
    true_wh_wi = true_boxes[..., 2:4]

    true_wh_half2 = true_wh_wi / 2.
    true_mins2 = true_xy_wi - true_wh_half2
    true_maxes2 = true_xy_wi + true_wh_half2

    pred_xy_wi = tf.expand_dims(pred_box_xy_wi, 4)
    pred_wh_wi = tf.expand_dims(pred_box_wh_wi, 4)

    pred_wh_half2 = pred_wh_wi / 2.
    pred_mins2 = pred_xy_wi - pred_wh_half2
    pred_maxes2 = pred_xy_wi + pred_wh_half2

    intersect_mins2 = tf.maximum(pred_mins2, true_mins2)
    intersect_maxes2 = tf.minimum(pred_maxes2, true_maxes2)
    intersect_wh2 = tf.maximum(intersect_maxes2 - intersect_mins2, 0.)
    intersect_areas2 = intersect_wh2[..., 0] * intersect_wh2[..., 1]

    true_areas2 = true_wh_wi[..., 0] * true_wh_wi[..., 1]
    pred_areas2 = pred_wh_wi[..., 0] * pred_wh_wi[..., 1]

    union_areas2 = pred_areas2 + true_areas2 - intersect_areas2
    iou_scores_all = tf.truediv(intersect_areas2, union_areas2)
    best_ious = tf.reduce_max(iou_scores_all, axis=4)

    # create masks ones and no ones
    noones = tf.to_float(best_ious < no_obj_thresh)

    return noones

def warm_up_adjust(seen, mask_shape, wu_bat, coord_scale, ones):
    seen = tf.assign_add(seen, 1.)
    warm_xy = tf.fill(mask_shape, 0.5)
    warm_xy = warm_xy[..., 0:2]
    warm_wh = tf.fill(mask_shape, 0.)
    warm_wh = warm_wh[..., 2:4]
    warm_no = tf.fill(mask_shape[0:4], 1.)

    true_box_xy, true_box_wh, coord_scale, coord_mask = tf.cond(
        tf.less(seen, wu_bat),
        lambda: [warm_xy, warm_wh, 0.01, warm_no],
        lambda: [true_box_xy, true_box_wh, coord_scale, ones]
    )

    return true_box_xy, true_box_wh, coord_scale, coord_mask


def warm_up_adjust_simp(mask_shape):

    warm_xy = tf.fill(mask_shape, 0.5)
    warm_xy = warm_xy[..., 0:2]
    warm_wh = tf.fill(mask_shape, 0.)
    warm_wh = warm_wh[..., 2:4]
    coord_mask = tf.fill(mask_shape, 1.)
    coord_mask = coord_mask[..., 0:2]
    coord_scale = 0.01

    return warm_xy, warm_wh, coord_scale, coord_mask


def loss_gfrc_yolo(gt1, gt2, y_pred, bat_no, dict_in):
    # compares output from cnn with ground truth to calculate loss
    n_bat = int(dict_in['batch_size'])
    boxsx = int(dict_in['boxs_x'])
    boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = int(dict_in['n_anchors'])
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    n_classes = int(dict_in['n_classes'])
    num_out = int(dict_in['num_out'])
    thresh = dict_in['iou_threshold']
    coord_scale = dict_in['lambda_coord']
    no_obj_scale = dict_in['lambda_noobj']
    class_scale = dict_in['lambda_class']
    obj_scale = dict_in['lambda_object']
    wu_bat = dict_in['warmbat']
    # size_scale = dict_in['lambda_size']

    y_pred = tf.reshape(y_pred, [n_bat, boxsy, boxsx, nanchors, num_out])

    cellgrid = get_grid(boxsx, boxsy, n_bat)
    predxy, predwh, predconf, predclass = adjust_pred(y_pred)
    predxy_wi, predwh_wi = adjust_pred_wi(predxy, predwh, cellgrid, boxsx, boxsy, anchors, nanchors)
    truexy, truewh, true_mins, true_maxes, true_areas = adjust_gt(gt1, cellgrid, boxsx, boxsy, anchors, nanchors)
    mask_shape = tf.shape(gt1)
    iouscore, testout = ious_centre_cell(predwh_wi, predxy_wi, true_mins, true_maxes, true_areas)
    noones = create_masks(gt2, cellgrid, boxsx, boxsy, predxy_wi, predwh_wi, thresh)
    ones = gt1[..., 4]
    # noones = tf.subtract(1., ones)

    loss_conf = tf.sqrt(tf.reduce_sum(tf.square((iouscore - predconf) * ones * obj_scale)))
    loss_noconf = tf.sqrt(tf.reduce_sum(tf.square((0. - predconf) * noones * no_obj_scale)))
    loss_class = tf.sqrt(tf.reduce_sum(tf.square((1. - predclass) * ones * class_scale)))
    warmxy, warmwh, coord_scale, warmno = warm_up_adjust_simp(mask_shape)
    coord_mask = tf.expand_dims(ones, axis=-1)
    truexy, truewh, coord_scale, coord_mask = tf.cond(
        tf.less(bat_no, wu_bat),
        lambda: [warmxy, warmwh, coord_scale, warmno],
        lambda: [truexy, truewh, coord_scale, coord_mask]
    )
    # coord_mask = tf.expand_dims(coord_mask, axis=-1)
    loss_xy = tf.sqrt(tf.reduce_sum(tf.square((truexy - predxy) * coord_mask * coord_scale)))
    loss_wh = tf.sqrt(tf.reduce_sum(tf.square((truewh - predwh) * coord_mask * coord_scale)))
    pos_loss = loss_xy + loss_wh

    poz = tf.greater_equal(predconf, thresh)
    negz = tf.less(predconf, thresh)
    truez = tf.greater_equal(ones, thresh)
    falsez = tf.less(ones, thresh)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(poz, truez), dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(poz, falsez), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(negz, truez), dtype=tf.float32))
    re = tp / (tp + fn)
    pr = tp / (tp + fp)
    test = tf.square((iouscore - predconf) * ones) * obj_scale
    test = iouscore
    fpr = tf.reduce_max(predwh_wi)
    #fpr = fp / (tp + fp)


    ind_losses = {
        "conf_loss_nogt": loss_noconf,
        "conf_loss_gt": loss_conf,
        "cent_loss": loss_xy,
        "size_loss": loss_wh,
        "pos_loss": pos_loss,
        "class_loss": loss_class,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Re": re,
        "Pr": pr,
        "FPR": fpr
    }

    return ind_losses


"""

# get cell grid

seen = tf.Variable(0.)

# Adjust Predictions
    
# Adjust ground truth for just cells with a centre of a ground truth
    
# Calculate IOU with any truth - create masks

# Warm-up training

# Finalize the loss
    
"""



def total_loss_calc(ind_losses, dict_in):

    conf_loss_nogt = ind_losses['conf_loss_nogt']
    # pos_loss = ind_losses['pos_loss']
    conf_loss_gt = ind_losses['conf_loss_gt']
    class_loss = ind_losses['class_loss']
    cent_loss = ind_losses['cent_loss']
    size_loss = ind_losses['size_loss']

    ctl = cent_loss
    szl = size_loss
    nol = conf_loss_nogt
    obl = conf_loss_gt
    cll = class_loss

    # total_loss = tf.add(tf.add(tf.add(tf.add(ctl, szl), nol), obl), cll)
    total_loss = tf.add(tf.add(tf.add(ctl, szl), nol), obl)
    # total_loss = tf.square(total_loss)

    return total_loss

