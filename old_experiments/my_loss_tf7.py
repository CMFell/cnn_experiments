import numpy as np
import tensorflow as tf

# Working version with double size pictures

def stable_sigmoid(z):
    return tf.where(z >= 0., 1. / (1. + tf.exp(-z)), tf.exp(z) / (1. + tf.exp(z)))

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
    #pred_box_wh = tf.where(pred_box_wh > 2., pred_box_wh * 0.1, pred_box_wh)
    #pred_box_wh = tf.where(pred_box_wh < -2., pred_box_wh * 0.1, pred_box_wh)
    #pred_box_wh1 = tf.where(tf.greater(pred_box_wh, 0.5), tf.random_uniform(tf.shape(pred_box_wh), 0.0, 0.5), pred_box_wh)
    #pred_box_wh1 = tf.where(tf.less(pred_box_wh, -10), tf.random_uniform(tf.shape(pred_box_wh), 0, -10),pred_box_wh)
    #pred_box_wh = tf.maximum(pred_box_wh, -0.7)
    #pred_box_wh = tf.exp(pred_box_wh)
    # shouldn't matter about the clip as these really high values are then multiplied by zeros later
    # problem with clipping at 100 doesn't give gradient change to adjust
    # pred_box_wh = tf.minimum(pred_box_wh, 50.0)
    # If I've got this right this should stop it reducing box sizes to zero
    # pred_box_wh = tf.maximum(pred_box_wh, -5.)

    # adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    # adjust class probabilities
    pred_box_class = tf.sigmoid(y_pred[..., 5])

    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class


def adjust_pred_wi(pred_box_xy, pred_box_wh, cell_grid, gridw, gridh, anchors, box):
    # new line convert to whole image
    pred_box_xy_wi = tf.divide(tf.add(pred_box_xy, cell_grid), [gridw, gridh])
    #pred_box_xy_wi = tf.divide(pred_box_xy_wi, [gridw, gridh])

    # this clips to prevent Inf values after exponentiating,
    # doesn't matter about the clip as these really high values are then multiplied by zeros later
    # pred_box_wh = tf.minimum(pred_box_wh, 2.)
    # new line adjust so relative to whole image
    #pred_box_step = tf.where(tf.less(pred_box_wh, -17.7),tf.zeros(tf.shape(pred_box_wh)), tf.exp(pred_box_wh))
    pred_box_wh_wi = tf.divide(tf.multiply(tf.exp(pred_box_wh), tf.reshape(anchors, [1, 1, 1, box, 2])), [gridw, gridh])
    # pred_box_wh_wi = pred_box_wh * tf.reshape(anchors, [1, 1, 1, box, 2])
    # pred_box_wh_wi = tf.divide(pred_box_wh_wi, [gridw, gridh])
    # pred_box_wh_wi = tf.minimum(pred_box_wh_wi, 1.0)

    return pred_box_xy_wi, pred_box_wh_wi


def adjust_gt(y_true, cell_grid, gridw, gridh, anchors, box):
    # get x and y relative to whole image
    true_box_xy = y_true[..., 0:2]
    # adjust to relative to box
    true_box_xy_wi = tf.divide(tf.add(true_box_xy, cell_grid), [gridw, gridh])

    # get w and h
    true_box_wh_wi = y_true[..., 2:4]
    # adjust w and h
    true_box_wh = tf.divide(tf.multiply(true_box_wh_wi, [gridw, gridh]), tf.reshape(anchors, [1, 1, 1, box, 2]))
    true_box_wh = tf.log(tf.add(true_box_wh, 0.000001))

    #true_box_wh = true_box_wh / tf.reshape(anchors, [1, 1, 1, box, 2])
    #true_box_wh = tf.log(true_box_wh)
    # the + 0.00001 takes out zeros which can't be logged these should then be multiplied by zero again later

    # adjust confidence
    true_wh_half = tf.divide(true_box_wh_wi, 2.)
    true_mins = tf.subtract(true_box_xy_wi, true_wh_half)
    true_maxes = tf.add(true_box_xy_wi, true_wh_half)
    true_areas = tf.multiply(true_box_wh_wi[..., 0], true_box_wh_wi[..., 1])

    return true_box_xy, true_box_wh, true_mins, true_maxes, true_areas, true_box_wh_wi

def ious_centre_cell(pred_box_wh_wi, pred_box_xy_wi, true_mins, true_maxes, true_areas):
    pred_wh_half = tf.divide(pred_box_wh_wi, 2.)
    pred_mins = tf.subtract(pred_box_xy_wi, pred_wh_half)
    #pred_mins = tf.maximum(tf.subtract(pred_box_xy_wi, pred_wh_half), 0)
    #pred_mins = tf.maximum(pred_mins, 0.)
    pred_maxes = tf.add(pred_box_xy_wi, pred_wh_half)
    #pred_maxes = tf.minimum(tf.add(pred_box_xy_wi, pred_wh_half), 1.0)
    #pred_maxes = tf.minimum(pred_maxes, 1.)

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(tf.subtract(intersect_maxes, intersect_mins), 0.)
    intersect_areas = tf.multiply(intersect_wh[..., 0], intersect_wh[..., 1])

    pred_size = tf.subtract(pred_maxes, pred_mins)
    pred_areas = tf.multiply(pred_box_wh_wi[..., 0], pred_box_wh_wi[..., 1])
    #pred_areas = tf.multiply(pred_size[..., 0], pred_size[..., 1])

    # add a small amount to avoid divide by zero, will later be multiplied by zero
    union_areas = tf.add(tf.subtract(tf.add(pred_areas, true_areas), intersect_areas), 0.00001)
    iou_scores = tf.divide(intersect_areas, union_areas)
    #iou_scores = tf.where(union_areas < 0.001, tf.zeros(tf.shape(union_areas)), tf.divide(intersect_areas, union_areas))
    # try to prevent it decreasing iouscores to zero instead of improving confidence
    # iou_scores = tf.maximum(iou_scores, 0.05)

    return iou_scores, union_areas


def create_masks(true_boxes, pred_box_xy_wi, pred_box_wh_wi, no_obj_thresh):
    # confidence mask: penalize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy_wi = true_boxes[..., 0:2]
    true_wh_wi = true_boxes[..., 2:4]

    true_wh_half2 = tf.divide(true_wh_wi, 2.)
    true_mins2 = tf.subtract(true_xy_wi, true_wh_half2)
    true_maxes2 = tf.add(true_xy_wi, true_wh_half2)

    pred_xy_wi = tf.expand_dims(pred_box_xy_wi, 4)
    pred_wh_wi = tf.expand_dims(pred_box_wh_wi, 4)

    pred_wh_half2 = tf.divide(pred_wh_wi, 2.)
    pred_mins2 = tf.subtract(pred_xy_wi, pred_wh_half2)
    pred_maxes2 = tf.add(pred_xy_wi, pred_wh_half2)

    intersect_mins2 = tf.maximum(pred_mins2, true_mins2)
    intersect_maxes2 = tf.minimum(pred_maxes2, true_maxes2)
    intersect_wh2 = tf.maximum(tf.subtract(intersect_maxes2, intersect_mins2), 0.)
    intersect_areas2 = tf.multiply(intersect_wh2[..., 0], intersect_wh2[..., 1])

    true_areas2 = tf.multiply(true_wh_wi[..., 0], true_wh_wi[..., 1])
    pred_areas2 = tf.multiply(pred_wh_wi[..., 0], pred_wh_wi[..., 1])

    union_areas2 = tf.add(tf.subtract(tf.add(pred_areas2, true_areas2), intersect_areas2), 0.00001)
    iou_scores_all = tf.divide(intersect_areas2, union_areas2)
    #iou_scores_all = tf.where(union_areas2 < 0.001, tf.zeros(tf.shape(union_areas2)),
    #                          tf.divide(intersect_areas2, union_areas2))
    best_ious = tf.reduce_max(iou_scores_all, axis=4)

    # create masks ones and no ones
    noones = tf.to_float(best_ious < no_obj_thresh)

    return noones, union_areas2, intersect_areas2

def warm_up_adjust_simp(mask_shape, gridh, gridw, anchors, box):

    # warm_xy = tf.divide(tf.add(cell_grid, 0.5), [gridw, gridh])
    # warm_xy = tf.expand_dims(warm_xy, -1)
    warm_xy = tf.fill(mask_shape, 0.5)
    warm_wh = tf.fill(tf.shape(warm_xy), 0.) # everything is same size of anchor boxes so all 1 then log gives zero
    warm_wh_wi = tf.divide(tf.multiply(tf.fill(tf.shape(warm_xy), 1.), tf.reshape(anchors, [1, 1, 1, box, 2])), [gridw,gridh])
    #warm_wh2 = tf.divide(warm_wh, [gridw, gridh])
    coord_mask = tf.fill(tf.shape(warm_xy)[0:4], 1.0)
    warm_iou = tf.fill(tf.shape(warm_xy)[0:4], 0.5)
    coord_scale = 1.0

    return warm_xy, warm_wh, coord_scale, coord_mask, warm_iou, warm_wh_wi


def loss_gfrc_yolo(gt1, gt2, y_pred, bat_no, dict_in, biasdict, wtdict):
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
    truexy, truewh, true_mins, true_maxes, true_areas, truewh_wi = adjust_gt(gt1, cellgrid, boxsx, boxsy, anchors, nanchors)
    iouscore, unar1 = ious_centre_cell(predwh_wi, predxy_wi, true_mins, true_maxes, true_areas)
    #noones, unar2, ia2 = create_masks(gt2, predxy_wi, predwh_wi, thresh)
    ones = gt1[..., 4]
    noones = tf.subtract(1., ones)

    maskshape = tf.shape(predxy)
    warmxy, warmwh, warm_scale, warmno, warmiou, warmwh_wi = warm_up_adjust_simp(maskshape, boxsy, boxsx, anchors, nanchors)
    warmzero = tf.zeros(tf.shape(noones))

    truexy1 = tf.cond(
        tf.less(bat_no, wu_bat),
        lambda: [warmxy],
        lambda: [truexy]
    )
    truewh1 = tf.cond(
        tf.less(bat_no, wu_bat),
        lambda: [warmwh],
        lambda: [truewh]
    )
    coord_mask1 = tf.cond(
        tf.less(bat_no, wu_bat),
        lambda: [warmno],
        lambda: [ones]
    )
    coord_scale1 = tf.cond(
        tf.less(bat_no, wu_bat),
        lambda: [warm_scale],
        lambda: [coord_scale]
    )
    noones1 = tf.cond(
        tf.less(bat_no, wu_bat),
        lambda: [warmzero],
        lambda: [noones]
    )

    loss_conf = tf.subtract(iouscore, predconf)
    loss_conf = tf.square(loss_conf)
    loss_conf = tf.multiply(loss_conf, ones)
    loss_conf = tf.multiply(loss_conf, obj_scale)
    loss_conf = tf.reduce_sum(loss_conf)

    loss_noconf = tf.subtract(0., predconf)
    loss_noconf = tf.square(loss_noconf)
    loss_noconf = tf.multiply(loss_noconf, noones1)
    loss_noconf = tf.multiply(loss_noconf, no_obj_scale)
    loss_noconf = tf.reduce_sum(loss_noconf)

    loss_class = tf.subtract(1., predclass)
    loss_class = tf.square(loss_class)
    loss_class = tf.multiply(loss_class, ones)
    loss_class = tf.multiply(loss_class, class_scale)
    loss_class = tf.reduce_sum(loss_class)

    # position losses
    coord_mask1 = tf.expand_dims(coord_mask1, axis=-1)

    loss_xy = tf.square(tf.subtract(truexy1, predxy))
    loss_xy = tf.multiply(loss_xy, coord_mask1)
    loss_xy = tf.multiply(loss_xy, coord_scale1)
    loss_xy = tf.reduce_sum(loss_xy)

    loss_wh = tf.square(tf.subtract(truewh1, predwh))
    loss_wh = tf.multiply(loss_wh, coord_mask1)
    loss_wh = tf.multiply(loss_wh, coord_scale1)
    loss_wh = tf.reduce_sum(loss_wh)
    #pos_loss = loss_xy + loss_wh

    poz = tf.greater_equal(predconf, thresh)
    negz = tf.less(predconf, thresh)
    truez = tf.greater_equal(ones, thresh)
    falsez = tf.less(ones, thresh)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(poz, truez), dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(poz, falsez), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(negz, truez), dtype=tf.float32))
    #re = tp / (tp + fn)
    #pr = tp / (tp + fp)
    #re = tf.reduce_sum(tf.cast(tf.is_nan(predwh_wi), dtype=tf.float32))
    #pr = tf.reduce_sum(tf.cast(tf.is_nan(predxy), dtype=tf.float32))
    #test = tf.cast(tf.greater_equal(predwh, 50.), dtype=tf.float32)
    #test = tf.multiply(predwh_wi, coord_mask)
    #fpr = tf.reduce_sum(tf.cast(tf.is_nan(iouscore), dtype=tf.float32))

    #pr = tf.shape(unar2)[4]
    test1 = tf.multiply(predwh_wi, 1.)
    test2 = tf.multiply(iouscore, 1.)
    test3 = tf.multiply(predconf, 1.)
    re = tf.reduce_max(test2)
    pr = tf.reduce_max(test3)
    fpr = tf.reduce_max(test1)
    #fpr = fp / (tp + fp)


    ind_losses = {
        "conf_loss_nogt": loss_noconf,
        "conf_loss_gt": loss_conf,
        "cent_loss": loss_xy,
        "size_loss": loss_wh,
        "pos_loss": tf.constant(1.0), #pos_loss,
        "class_loss": loss_class,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Re": re,
        "Pr": pr,
        "FPR": fpr
    }

    return ind_losses


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

    total_loss = tf.sqrt(tf.add(tf.add(tf.add(tf.add(szl, ctl), cll), obl), nol))

    return total_loss

