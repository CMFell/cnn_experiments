import cv2

import numpy as np
import pandas as pd

from scipy.special import expit
from my_metric import metric_yolo_tl, metric_yolo_cfno, metric_yolo_cf, metric_yolo_ct, metric_yolo_sz, \
    metric_yolo_cl, metric_yolo_TP, metric_yolo_FP, metric_yolo_FN, metric_yolo_ot1, metric_yolo_ot2, \
    metric_yolo_ot3, metric_yolo_ot4, metric_yolo_ot5, metric_yolo_ot6
from my_loss import yolo_loss
from keras.models import load_model


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex


def loss_yolo_np2(y_true, y_pred):
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
    lam_objct = dict_in['lambda_object']
    lam_class = dict_in['lambda_class']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]

    # get top left position of cells
    rowz = np.divide(np.arange(boxsy), boxsy)
    colz = np.divide(np.arange(boxsx), boxsx)
    rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    tl_cell = np.stack((colno, rowno), axis=4)

    # restructure ground truth output
    y_true = np.reshape(y_true, size1)

    # get confidences, centres sizes and classes from ground truth
    conf_true = y_true[:, :, :, :, 4]
    conf_true = np.reshape(conf_true, size2)
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
    size_true = np.divide(size_true, size3)
    class_true = y_true[:, :, :, :, 5:]

    # restructure net_output to same as gt_out
    y_pred = np.reshape(y_pred, size1)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = expit(np.reshape(y_pred[:, :, :, :, 4], size2))
    cent_cnn = expit(y_pred[:, :, :, :, 0:2])
    # divide so position is relative to whole image
    cent_cnn = np.divide(cent_cnn, size3)
    # add to cent_cnn so is position in whole image
    cent_cnn = np.add(cent_cnn, tl_cell)
    size_cnn = y_pred[:, :, :, :, 2:4]
    # size is to power of prediction
    size_cnn = np.exp(size_cnn)
    # adjust so size is relative to anchors
    size_cnn = np.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = np.divide(size_cnn, size3)
    # if more than one class
    # class_cnn = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    # else
    class_cnn = expit(y_pred[:, :, :, :, 5:])

    # calculate cells with and without ground truths
    # only one ground truth for all anchor boxes
    onesij = conf_true
    # ones_i = tf.reduce_sum(conf_true, axis=-1)
    # noones_i = tf.subtract(1.0, ones_i)
    noones_ij = np.subtract(1.0, onesij)
    # ones_i = tf.expand_dims(ones_i, axis=-1)
    # noones_i = tf.expand_dims(noones_i, axis=-1)

    # there is only one ground truth per detection box which is repeated for each anchor box
    # therefore size_true ans centre_true are the same thing in each anchor box layer
    area_cnn = np.multiply(size_cnn[:, :, :, :, 0], size_cnn[:, :, :, :, 1])
    area_true = np.multiply(size_true[:, :, :, :, 0], size_true[:, :, :, :, 1])
    size_cnn_half = np.divide(size_cnn, 2)
    size_true_half = np.divide(size_true, 2)
    min_cnn = np.subtract(cent_cnn, size_cnn_half)
    min_true = np.subtract(centres_true, size_true_half)
    max_cnn = np.add(cent_cnn, size_cnn_half)
    max_true = np.add(centres_true, size_true_half)
    inter_maxs = np.minimum(max_cnn, max_true)
    inter_mins = np.maximum(min_cnn, min_true)
    inter_size = np.maximum(np.subtract(inter_maxs, inter_mins), 0)
    inter_area = np.multiply(inter_size[:, :, :, :, 0], inter_size[:, :, :, :, 1])
    union_area = np.subtract(np.add(area_true, area_cnn), inter_area)
    iou = np.divide(inter_area, union_area)
    iou = np.reshape(iou, size2)
    negiou = np.subtract(1.0, iou)

    # calculate losses
    # calculate confidence losses when no ground truth in cell
    conf_loss_nogt = np.sum(np.multiply(np.square(np.subtract(0.0, confs_cnn)), noones_ij))
    # The rest of the losses are only calculated in cells where there are ground truths
    # conf_loss_gt = tf.multiply(tf.multiply(tf.square(tf.subtract(1.0, confs_cnn)), iou), onesij)
    conf_loss_gt = np.multiply(np.square(np.subtract(np.multiply(1.0, negiou), np.multiply(confs_cnn, negiou))), onesij)
    conf_loss_gt = np.sum(conf_loss_gt)
    onesij = np.expand_dims(onesij, axis=-1)
    # iou = tf.expand_dims(iou, axis=-1)
    # negiou = tf.expand_dims(negiou, axis=-1)
    # ones_i = tf.expand_dims(ones_i, axis=-1)
    # cent_loss = tf.multiply(tf.multiply(tf.square(tf.subtract(centres_true, cent_cnn)), negiou), onesij)
    cent_loss = np.multiply(np.square(np.subtract(centres_true, cent_cnn)), onesij)
    cent_loss = np.sum(cent_loss)
    # size_loss = tf.multiply(tf.multiply(tf.square(tf.subtract(size_true, size_cnn)), negiou), onesij)
    size_loss = np.multiply(np.square(np.subtract(np.sqrt(size_true), np.sqrt(size_cnn))), onesij)
    size_loss = np.sum(size_loss)
    pos_loss = np.add(cent_loss, size_loss)
    # class_loss = tf.multiply(tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), negiou), ones_i)
    class_loss = np.multiply(np.square(np.subtract(class_true, class_cnn)), onesij)
    class_loss = np.sum(class_loss)

    total_loss = np.add(np.add(np.add(np.multiply(lam_noobj, conf_loss_nogt), np.multiply(lam_coord, pos_loss)),
                               np.multiply(lam_objct, conf_loss_gt)), np.multiply(lam_class, class_loss))

    conf_loss_gt = np.multiply(lam_objct, conf_loss_gt)
    conf_loss_nogt = np.multiply(lam_noobj, conf_loss_nogt)
    cent_loss = np.multiply(lam_coord, cent_loss)
    size_loss = np.multiply(lam_coord, size_loss)
    class_loss = np.multiply(lam_class, class_loss)
    # class_loss = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(class_true, class_cnn)), onesij))
    output = [total_loss, conf_loss_nogt, conf_loss_gt, cent_loss, size_loss, class_loss]

    return output


def get_gt(dict_gt, indexes):
    """Generates data containing batch_size samples"""
    # X : (n_samples, *dim, n_channels)
    # Initialization
    batch_size = dict_gt['batch_size']
    dim = dict_gt['dim']
    n_channels = dict_gt['n_channels']
    size_reduce = dict_gt['size_reduce']
    nanchors = dict_gt['nanchors']
    nclasses = dict_gt['nclasses']
    input_dir = dict_gt['input_dir']
    input_array = dict_gt['input_array']
    anchors = dict_gt['anchors_in']
    lenout = 5 + nclasses

    xx = np.empty((batch_size, dim[0], dim[1], n_channels))
    boxx = int(np.ceil(dim[1] / size_reduce[1]))
    boxy = int(np.ceil(dim[0] / size_reduce[0]))
    tot_len = int(nanchors * (5 + nclasses))
    yy = np.zeros((batch_size, boxy, boxx, tot_len))
    anks = np.multiply(anchors, size_reduce)
    pix_xy = [dim[1], dim[0]]
    anks = np.divide(anks, pix_xy)
    halfanks = np.divide(anks, 2)
    xmin = np.reshape(np.multiply(halfanks[:, 0], -1), (nanchors, 1))
    xmax = np.reshape(halfanks[:, 0], (nanchors, 1))
    ymin = np.reshape(np.multiply(halfanks[:, 1], -1), (nanchors, 1))
    ymax = np.reshape(halfanks[:, 1], (nanchors, 1))
    minz = np.hstack((xmin, ymin))
    maxz = np.hstack((xmax, ymax))
    ank_area = np.multiply(anks[:, 0], anks[:, 1])
    # Generate data
    for ii in range(batch_size):
        row_index = indexes[ii]
        # store training images
        xx_path = input_dir + input_array[row_index, 0]
        if n_channels == 1:
            imread_val = 0
        else:
            imread_val = n_channels
        xx[ii,] = cv2.imread(xx_path, imread_val)

        # store training ground truth data
        yy_path = input_dir + input_array[row_index, 1]
        yy_in = pd.read_csv(yy_path, delimiter=' ', header=None)
        yy_in = np.array(yy_in)
        # this means that if there are two ground truths in one cell only last gets saved.
        # need to rework so can save up to five
        for gt in range(yy_in.shape[0]):
            class_in = int(yy_in[gt, 0])
            cent_in = yy_in[gt, 1:3]
            size_in = yy_in[gt, 3:5]

            cellx = int(np.floor(cent_in[0] * boxx))
            celly = int(np.floor(cent_in[1] * boxy))
            # Going to have out put as x and y pos in image
            # centx = cent_in[0] * boxx - cellx
            # centy = cent_in[1] * boxy - celly
            centx = cent_in[0]
            centy = cent_in[1]
            # size is supposed to be relative to image but that doesn't work for different size images
            # give size as relative to box
            sizex = size_in[0] * boxx
            sizey = size_in[1] * boxy
            # calculate using whole image size for iou so same as anchors
            gt_area = size_in[0] * size_in[1]
            gt_xmin = size_in[0] / -2
            gt_xmax = size_in[0] / 2
            gt_ymin = size_in[1] / -2
            gt_ymax = size_in[1] / 2
            gt_min = [gt_xmin, gt_ymin]
            gt_max = [gt_xmax, gt_ymax]
            class_oh = np.zeros(nclasses)
            class_oh[class_in] = 1
            inter_maxs = np.minimum(maxz, gt_max)
            inter_mins = np.maximum(minz, gt_min)
            inter_size = np.maximum(np.subtract(inter_maxs, inter_mins), 0)
            inter_area = np.multiply(inter_size[:, 0], inter_size[:, 1])
            union_area = np.subtract(np.add(ank_area, gt_area), inter_area)
            iou = np.divide(inter_area, union_area)
            iou_idx = np.argmax(iou, axis=-1)
            out_vec = np.array([centx, centy, sizex, sizey, 0])
            out_vec = np.hstack((out_vec, class_oh))
            out_vec = np.tile(out_vec, nanchors)
            nn = lenout * iou_idx + 4
            out_vec[nn] = 1
            yy[ii, celly, cellx, :] = out_vec
    return xx, yy

input_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_out_test.h5"

# create batches and train model
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
train_file = base_dir + "gfrc_train.txt"
train_df = pd.read_csv(train_file, delimiter=',')
train_df = np.array(train_df)

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
lambda_cl = 1.0
lambda_no = 0.5
lambda_ob = 1.0
lambda_cd = 5.0
test_img_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/Z101_Img00083_48.png"
test_out = cv2.imread(test_img_path, 1)
print(test_out)
batch_size = 1
dict_in = {'boxs_x': boxs_x, 'boxs_y': boxs_y, 'img_x_pix': img_x_pix, 'img_y_pix': img_y_pix,
           'anchors': anchors_in, 'n_classes': classes_in, 'lambda_coord': lambda_cd, 'lambda_noobj': lambda_no,
           'base_dir': base_dir, 'batch_size': batch_size, 'lambda_class': lambda_cl, 'lambda_object': lambda_ob}

dict_gt = {'batch_size': 1, 'dim': (307, 460), 'n_channels': 3, 'size_reduce': (16, 16), 'nanchors': 5, 'nclasses': 1,
           'input_dir': base_dir, 'input_array': train_df, 'anchors_in': anchors_in}

custom_objects_dict = {'metric_TP': metric_yolo_TP(dict_in=met_dict), 'metric_FP': metric_yolo_FP(dict_in=met_dict),
                       'metric_FN': metric_yolo_FN(dict_in=met_dict), 'metric_tl': metric_yolo_tl(dict_in=met_dict),
                       'loss_gfrc_yolo': yolo_loss(dict_in=met_dict), 'metric_cfno': metric_yolo_cfno(dict_in=met_dict),
                       'metric_cf': metric_yolo_cf(dict_in=met_dict), 'metric_ct': metric_yolo_ct(dict_in=met_dict),
                       'metric_sz': metric_yolo_sz(dict_in=met_dict), 'metric_cl': metric_yolo_cl(dict_in=met_dict),
                       'metric_1ot': metric_yolo_ot1(dict_in=met_dict), 'metric_2ot': metric_yolo_ot2(dict_in=met_dict),
                       'metric_3ot': metric_yolo_ot3(dict_in=met_dict), 'metric_4ot': metric_yolo_ot4(dict_in=met_dict),
                       'metric_5ot': metric_yolo_ot5(dict_in=met_dict), 'metric_6ot': metric_yolo_ot6(dict_in=met_dict)}
gfrc_yolo = load_model(input_path, custom_objects=custom_objects_dict)

cent_loss = []
tot_loss = []
cfno_loss = []
cf_loss = []
sz_loss = []
cl_loss = []
#1344
for ind in range(1):
    index_in = [ind]

    print(train_df[ind, :])

    xxx, yyy = get_gt(dict_gt, index_in)
    print(xxx)

    print(yyy[0, 7, 23, :])
    print(yyy[0, 8, 25, :])
    print(yyy[0, 9, 28, :])
    print(yyy[0, 13, 22, :])

    net_output = gfrc_yolo.predict(xxx)

    loss_out = loss_yolo_np2(yyy, net_output)

    tot_loss.append(loss_out[0])
    cfno_loss.append(loss_out[1])
    cf_loss.append(loss_out[2])
    cent_loss.append(loss_out[3])
    sz_loss.append(loss_out[4])
    cl_loss.append(loss_out[5])

for i in range(len(cent_loss)):
    print(cent_loss[i])

print("sum", np.sum(cent_loss)/1344)
print("max", np.max(cent_loss))
print("min", np.min(cent_loss))
print("min", np.sum(tot_loss)/1344)
print("min", np.sum(cfno_loss)/1344)
print("min", np.sum(cf_loss)/1344)
print("min", np.sum(sz_loss)/1344)
print("min", np.sum(cl_loss)/1344)