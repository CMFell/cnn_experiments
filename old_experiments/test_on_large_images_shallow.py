from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.initializers import RandomNormal
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
from scipy.special import expit
import time

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

LABELS = ['animal']

IMAGE_PRED_H, IMAGE_PRED_W = 1664, 2496
RESIZE = 2
GRID_PRED_H, GRID_PRED_W = int(IMAGE_PRED_H * RESIZE / 16), int(IMAGE_PRED_W * RESIZE / 16)
BOX = 5
CLASS = len(LABELS)
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
OBJ_CONF_THRESHOLD = 0.3  # 0.5
ANCHORS = [2.387088, 2.985595, 1.540179, 1.654902, 3.961755, 3.936809, 2.681468, 1.803889, 5.319540, 6.116692]
TRUE_BOX_BUFFER = 15

start_time = time.time()

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(xx):
    return tf.space_to_depth(xx, block_size=2)


input_image = Input(shape=(None, None, 3))
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

# Layer 1
x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False, kernel_initializer=RandomNormal())(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False, kernel_initializer=RandomNormal())(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23', kernel_initializer=RandomNormal())(x)
output = Reshape((GRID_PRED_H, GRID_PRED_W, BOX, 4 + 1 + CLASS))(x)
# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

model.summary()

def convert2box(input_arr):
    boxes = []
    for img in range(input_arr.shape[0]):
        for ycl in range(input_arr.shape[1]):
            for xcl in range(input_arr.shape[2]):
                for anc in range(input_arr.shape[3]):
                    rez = input_arr[img, ycl, xcl, anc, :]
                    if expit(rez[4]) >= OBJ_CONF_THRESHOLD:
                        xcent = (expit(rez[0]) + xcl) / input_arr.shape[2]
                        ycent = (expit(rez[1]) + ycl) / input_arr.shape[1]
                        xsizhalf = ((np.exp(rez[2]) * ANCHORS[anc * 2]) / GRID_PRED_W) / 2
                        ysizhalf = ((np.exp(rez[3]) * ANCHORS[anc * 2 + 1]) / GRID_PRED_H) / 2
                        xmin = (xcent - xsizhalf) * (IMAGE_PRED_W * RESIZE)
                        xmax = (xcent + xsizhalf) * (IMAGE_PRED_W * RESIZE)
                        ymin = (ycent - ysizhalf) * (IMAGE_PRED_H * RESIZE)
                        ymax = (ycent + ysizhalf) * (IMAGE_PRED_H * RESIZE)
                        box = [xmin, ymin, xmax, ymax, expit(rez[4]), 0]
                        boxes += [box]
    return boxes

def basic_nms(boxes_in, thresh):
    # calculate iou

    boxes_ot = pd.DataFrame(columns=["XMIN", "XMAX", "YMIN", "YMAX"])

    while boxes_in.shape[0] > 0:
        xmins = boxes_in.XMIN
        xmins = xmins.tolist()
        xmaxs = boxes_in.XMAX
        xmaxs = xmaxs.tolist()
        ymins = boxes_in.YMIN
        ymins = ymins.tolist()
        ymaxs = boxes_in.YMAX
        ymaxs = ymaxs.tolist()

        xmn = xmins[0]
        xmns = np.array(xmins[1:])
        xmx = xmaxs[0]
        xmxs = np.array(xmaxs[1:])
        ymn = ymins[0]
        ymns = np.array(ymins[1:])
        ymx = ymaxs[0]
        ymxs = np.array(ymaxs[1:])

        ol_wid = np.minimum(xmx, xmxs) - np.maximum(xmn, xmns)
        ol_hei = np.minimum(ymx, ymxs) - np.maximum(ymn, ymns)

        ol_x = np.maximum(0, ol_wid)
        ol_y = np.maximum(0, ol_hei)

        distx = np.subtract(xmxs, xmns)
        disty = np.subtract(ymxs, ymns)
        bxx = xmx - xmn
        bxy = ymx - ymn

        ol_area = np.multiply(ol_x, ol_y)
        bx_area = bxx * bxy
        bxs_area = np.multiply(distx, disty)

        ious = np.divide(ol_area, np.subtract(np.add(bxs_area, bx_area), ol_area))
        mask_bxs = np.greater(ious, thresh)

        if np.sum(mask_bxs) > 0:
            box_ot = pd.DataFrame(index=[1], columns=["XMIN", "XMAX", "YMIN", "YMAX"])
            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]

            xmns = np.append(xmns, xmn)
            xmxs = np.append(xmxs, xmx)
            ymxs = np.append(ymxs, ymx)
            ymns = np.append(ymns, ymn)

            box_ot.XMIN.iloc[0] = np.min(xmns)
            box_ot.XMAX.iloc[0] = np.max(xmxs)
            box_ot.YMIN.iloc[0] = np.min(ymns)
            box_ot.YMAX.iloc[0] = np.max(ymxs)

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out[1:] = mask_bxs
            mask_out = np.logical_not(mask_out)
            boxes_in = boxes_in[mask_out]
        else:
            box_ot = pd.DataFrame(index=[1], columns=["XMIN", "XMAX", "YMIN", "YMAX"])
            box_ot.XMIN.iloc[0] = xmn
            box_ot.XMAX.iloc[0] = xmx
            box_ot.YMIN.iloc[0] = ymn
            box_ot.YMAX.iloc[0] = ymx
            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out = np.logical_not(mask_out)
            boxes_in = boxes_in[mask_out]
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0)

    return boxes_ot

def split_fn(filename_in):
    splits = filename_in.split("/")
    filename_out = splits[0] + "_" + splits[1]
    return filename_out


def get_gt(filen, file_l, gt_l):
    file_root = file_l[filen]
    mask_boxes = gt_l.filename_ == file_root
    filter_boxes = gt_l[mask_boxes]
    return filter_boxes


def convert_gt_to_pix(gt, img_wid, img_hei):
    wid_hlf = gt.wid / 2
    hei_hlf = gt.height / 2
    gt["XMIN"] = (gt.xc - wid_hlf) * img_wid
    gt["XMAX"] = (gt.xc + wid_hlf) * img_wid
    gt["YMIN"] = (gt.yc - hei_hlf) * img_hei
    gt["YMAX"] = (gt.yc + hei_hlf) * img_hei
    gt.XMIN = gt.XMIN.astype(int)
    gt.XMAX = gt.XMAX.astype(int)
    gt.YMIN = gt.YMIN.astype(int)
    gt.YMAX = gt.YMAX.astype(int)

    return gt


def calc_iou(bx, bxs):

    xmns = bxs.XMIN
    xmns = xmns.tolist()
    xmxs = bxs.XMAX
    xmxs = xmxs.tolist()
    ymns = bxs.YMIN
    ymns = ymns.tolist()
    ymxs = bxs.YMAX
    ymxs = ymxs.tolist()

    xmn = bx.XMIN.tolist()
    xmx = bx.XMAX.tolist()
    ymn = bx.YMIN.tolist()
    ymx = bx.YMAX.tolist()

    ol_wid = np.minimum(xmx, xmxs) - np.maximum(xmn, xmns)
    ol_hei = np.minimum(ymx, ymxs) - np.maximum(ymn, ymns)

    ol_x = np.maximum(0, ol_wid)
    ol_y = np.maximum(0, ol_hei)

    distx = np.subtract(xmxs, xmns)
    disty = np.subtract(ymxs, ymns)
    bxx = xmx - xmn
    bxy = ymx - ymn

    ol_area = np.multiply(ol_x, ol_y)
    bx_area = bxx * bxy
    bxs_area = np.multiply(distx, disty)

    ious = np.divide(ol_area, np.subtract(np.add(bxs_area, bx_area), ol_area))
    return ious


def get_tpfpfn(out_box, box_gt, thresh):
    boxes_ot = pd.DataFrame(columns=["XMIN", "XMAX", "YMIN", "YMAX", "filename", "type"])
    out_box.index = range(out_box.shape[0])
    box_gt.index = range(box_gt.shape[0])
    tp_tot = 0
    tp_inds = []
    for gt in range(box_gt.shape[0]):
        if out_box.shape[0] > 0:
            ious = calc_iou(box_gt.iloc[gt], out_box)
            max_iou_index = np.argmax(ious)
            max_iou = ious[max_iou_index]
            if max_iou >= thresh:
                tp_out = out_box.iloc[max_iou_index]
                tp_box = np.reshape([tp_out.XMIN, tp_out.XMAX, tp_out.YMIN, tp_out.YMAX, tp_out.filename, "tp"], (1, 6))
                tp_box = pd.DataFrame(tp_box, columns=["XMIN", "XMAX", "YMIN", "YMAX", "filename", "type"])
                boxes_ot = pd.concat((boxes_ot, tp_box), axis=0)
                del_in = max_iou_index + 1
                # out_box = pd.concat((out_box.iloc[0:max_iou_index, ], out_box.iloc[del_in:, ]), axis=0)
                out_box.drop(out_box.index[max_iou_index], inplace=True)
                tp_tot += 1
                tp_inds.append(gt)
    out_box["type"] = "fp"
    fp_tot = out_box.shape[0]
    boxes_ot = pd.concat((boxes_ot, out_box), axis=0)
    tn_out = pd.concat((box_gt.XMIN, box_gt.XMAX, box_gt.YMIN, box_gt.YMAX, box_gt.filename_), axis=1)
    tn_out.columns = ["XMIN", "XMAX", "YMIN", "YMAX", "filename"]
    tn_out["type"] = "fn"
    tn_out.drop(tn_out.index[tp_inds], inplace=True)
    fn_tot = tn_out.shape[0]
    boxes_ot = pd.concat((boxes_ot, tn_out))
    sumz = [tp_tot, fp_tot, fn_tot]
    return boxes_ot, sumz


# valid_full_image_folder = 'C:/Users/kryzi/OneDrive - University of St Andrews/PhD/valid_images/'
valid_full_image_folder = 'C:/Users/christina/OneDrive - University of St Andrews/PhD/valid_images/'
# wt_folder = 'C:/Users/kryzi/OneDrive - University of St Andrews/PhD/from_cnn/'
wt_folder = 'C:/Users/christina/OneDrive - University of St Andrews/PhD/from_cnn/'

model.load_weights(wt_folder + 'weights_coco.h5')

windowz = pd.read_csv(wt_folder + "windowz9ol.csv")

valid_files = os.listdir(valid_full_image_folder)

rez_all_img = pd.DataFrame(columns=["XMIN", "XMAX", "YMIN", "YMAX", "filename"])
gt_list = pd.read_csv(wt_folder + "yolo_valid_GFRC_bboxes.csv")

gt_files = gt_list.file_loc
update_files = gt_files.apply(split_fn)
gt_list["filename_"] = update_files
gt_list = convert_gt_to_pix(gt_list, 7360, 4912)

all_boxes_out = pd.DataFrame(columns=["XMIN", "XMAX", "YMIN", "YMAX", "filename", "type"])
tot_tp = 0
tot_fp = 0
tot_fn = 0

for ff in range(len(valid_files)):
# for ff in range(84,86):
    print(ff)
    image_name = valid_files[ff]
    prefix = image_name[:-4]
    boxes_all = np.empty((0,6))
    if image_name[-4:] == ".jpg":
        for wnd in range(windowz.shape[0]):
            wind = windowz.iloc[wnd]
            xmn = wind.xn
            xmx = wind.xx
            ymn = wind.yn
            ymx = wind.yx
            image_in = cv2.imread(valid_full_image_folder + image_name)
            image_in = image_in[ymn:ymx, xmn:xmx]
            height, width = image_in.shape[:2]
            image_in = cv2.resize(image_in, (RESIZE * width, RESIZE * height), interpolation=cv2.INTER_CUBIC)
            # image_in = cv2.imread(valid_image_folder + image_name)
            dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
            image_in = image_in / 255.
            image_in = image_in[:, :, ::-1]
            image_in = np.expand_dims(image_in, 0)
            netout = model.predict([image_in, dummy_array])
            boxes_pred = convert2box(netout)
            boxes_pred = np.array(boxes_pred)
            # print("Wnd", wnd, "detects", boxes_pred.shape[0])
            if len(boxes_pred) > 0:
                boxes_pred[:, 0] = np.add(np.divide(boxes_pred[:, 0], 2), wind.xn)
                boxes_pred[:, 2] = np.add(np.divide(boxes_pred[:, 2], 2), wind.xn)
                boxes_pred[:, 1] = np.add(np.divide(boxes_pred[:, 1], 2), wind.yn)
                boxes_pred[:, 3] = np.add(np.divide(boxes_pred[:, 3], 2), wind.yn)
                try:
                    boxes_all
                except NameError:
                    boxes_all = boxes_pred
                else:
                    boxes_all = np.vstack((boxes_all, boxes_pred))
        boxes_all = pd.DataFrame(boxes_all, columns=["XMIN", "YMIN", "XMAX", "YMAX", "CONF", "CLASS"])
        boxes_all = basic_nms(boxes_all, 0.5)
        boxes_all["filename"] = image_name
        boxes_all.XMIN = boxes_all.XMIN.astype(int)
        boxes_all.XMAX = boxes_all.XMAX.astype(int)
        boxes_all.YMIN = boxes_all.YMIN.astype(int)
        boxes_all.YMAX = boxes_all.YMAX.astype(int)
    print(boxes_all.shape)
    gt_box = get_gt(ff, valid_files, gt_list)
    tpfpfn_box, rez = get_tpfpfn(boxes_all, gt_box, 0.1)
    all_boxes_out = pd.concat((all_boxes_out, tpfpfn_box), axis=0)
    # rez_all_img = pd.concat((rez_all_img, boxes_all), axis=0)
    tot_tp += rez[0]
    tot_fp += rez[1]
    tot_fn += rez[2]

print("--- %s end time --- " % (time.time() - start_time))

print(all_boxes_out)
all_boxes_out.to_csv(wt_folder + "output_combined.csv")
print(tot_tp, tot_fp, tot_fn)
