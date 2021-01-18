import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
n_anchors = anchors_in.shape[0]
# input_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_v2.tfrecords"
output_path = "E:/CF_Calcs/BenchmarkSets/GFRC/TFrec/gfrc_yolo_v2.tfrecords"
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_aug/'
train_file = base_dir + "gfrc_train.txt"
input_file = pd.read_csv(train_file)
# Shuffle rows so they aren't in order
# This mean smaller buffer sizes can be used
input_file = input_file.sample(frac=1).reset_index(drop=True)
image_paths = input_file.img_name
dir_rep = np.repeat(base_dir, image_paths.shape[0])
file_dir = pd.DataFrame(dir_rep, columns=["basedir"])
file_dir = file_dir.basedir
image_paths = file_dir.str.cat(image_paths, sep=None)
image_paths = image_paths.tolist()
gt_paths = input_file.gt_details
gt_paths = file_dir.str.cat(gt_paths, sep=None)
gt_paths = gt_paths.tolist()
n_imgs = len(image_paths)
n_classes = 1
train_img_size = (384, 576)
size_reduction = (32, 32)
img_x_pix = train_img_size[1]
img_y_pix = train_img_size[0]
boxs_x = np.ceil(img_x_pix / size_reduction[1])
boxs_y = np.ceil(img_y_pix / size_reduction[0])
max_gt = 14  # max number of animals in images
maxoutsize = np.array([boxs_y, boxs_x, n_anchors, 5 + n_classes, max_gt], dtype=np.int32)
print(maxoutsize)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(listin):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=listin))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(listin):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=listin))


def read_image(imagepath):
    image = cv2.imread(imagepath)
    return image


def convert_to_gt1(boxes, out_size, anchors):
    boxy, boxx, nanc, outlen, maxgt = out_size
    gt1 = np.zeros((boxy, boxx, nanc, outlen))
    nbox = boxes.shape[0]
    for bx in range(nbox):
        box = boxes.iloc[bx]
        # Calc best box compared to anchors,
        # zero position both
        xmin_true = 0 - box.wid / 2
        xmax_true = 0 + box.wid / 2
        ymin_true = 0 - box.hei / 2
        ymax_true = 0 + box.hei / 2
        anc_xmin = np.subtract(0, np.divide(anchors[:, 0], 2))
        anc_xmax = np.add(0, np.divide(anchors[:, 0], 2))
        anc_ymin = np.subtract(0, np.divide(anchors[:, 1], 2))
        anc_ymax = np.add(0, np.divide(anchors[:, 1], 2))
        # calc intersection
        interxmax = np.minimum(anc_xmax, xmax_true)
        interxmin = np.maximum(anc_xmin, xmin_true)
        interymax = np.minimum(anc_ymax, ymax_true)
        interymin = np.maximum(anc_ymin, ymin_true)
        sizex = np.maximum(np.subtract(interxmax, interxmin), 0)
        sizey = np.maximum(np.subtract(interymax, interymin), 0)
        inter_area = np.multiply(sizex, sizey)
        # calc iou
        anc_area = np.multiply(anchors[:, 0], anchors[:, 1])
        truth_area = np.multiply(box.wid, box.hei)
        union_area = np.subtract(np.add(anc_area, truth_area), inter_area)
        iou = np.divide(inter_area, union_area)
        # get best anchor box
        best_box = np.argmax(iou)
        # calculate which cell truth belongs to
        xcell = np.int32(np.floor(box.xc * boxx))
        ycell = np.int32(np.floor(box.yc * boxy))
        # get centre position relative to cell
        centx = (box.xc * boxx) - xcell
        centy = (box.yc * boxy) - ycell
        # create output
        out_vec = np.zeros(outlen)
        out_vec[0:5] = [centx, centy, box.wid, box.hei, 1]
        class_pos = np.int32(5 + box.clazz)
        out_vec[class_pos] = 1
        # put output in correct position
        gt1[ycell, xcell, best_box, :] = out_vec

    return gt1


def convert_to_gt2(boxes, out_size):
    boxy, boxx, nanc, outlen, maxgt = out_size
    gt2 = np.zeros((1, 1, 1, 4, maxgt))
    nbox = boxes.shape[0]
    for bx in range(nbox):
        box = boxes.iloc[bx]
        # calculate which cell truth belongs to
        xcell = np.int32(np.floor(box.xc * boxx))
        ycell = np.int32(np.floor(box.yc * boxy))
        # get centre position relative to cell
        centx = (box.xc * boxx) - xcell
        centy = (box.yc * boxy) - ycell
        # create output
        # put output in correct position
        gt2[0, 0, 0, :, bx] = [box.xc, box.yc, box.wid, box.hei]
    return gt2


def paths_to_example(imagepath, gtpath, outsize, anchors):
    file = cv2.imread(imagepath)
    # need to save image shape as going to convert image to string
    # need shape to be able to reform image
    image_shape = file.shape
    # this safely converts image to a string
    # tfrecords has to save as string(bytes), int64, or float
    image_string = file.tostring()
    # save filename as well
    imagepath = imagepath.encode()

    # read in boxes and save as lists
    boxes = pd.read_csv(gtpath, sep=' ', names=["clazz", "xc", "yc", "wid", "hei"])
    gt1 = convert_to_gt1(boxes, outsize, anchors)
    gt2 = convert_to_gt2(boxes, outsize)
    gt1_out = gt1.flatten()
    gt2_out = gt2.flatten()

    feature = {
        'imheight': _int64_feature(image_shape[0]),
        'imwidth': _int64_feature(image_shape[1]),
        'imdepth': _int64_feature(image_shape[2]),
        'filename': _bytes_feature(imagepath),
        'image_raw': _bytes_feature(image_string),
        'bxy': _int64_feature(outsize[0]),
        'bxx': _int64_feature(outsize[1]),
        'nanc': _int64_feature(outsize[2]),
        'otln': _int64_feature(outsize[3]),
        'mxgt': _int64_feature(outsize[4]),
        'gt1_list': _float_list_feature(gt1_out),
        'gt2_list': _float_list_feature(gt2_out)
    }

    tfexample = tf.train.Example(features=tf.train.Features(feature=feature))
    return tfexample


writer = tf.python_io.TFRecordWriter(output_path)

for gt in range(len(gt_paths)):
    print(gt)
    image_path = image_paths[gt]
    gt_path = gt_paths[gt]
    tf_example = paths_to_example(image_path, gt_path, maxoutsize, anchors_in)
    writer.write(tf_example.SerializeToString())

writer.close()
