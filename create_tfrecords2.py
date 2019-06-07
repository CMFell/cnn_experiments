import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import base64

# output_path = "C:/Benchmark_data/GFRC/gfrc_yolo_out.h5"
input_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_v2.tfrecords"
output_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_v2.tfrecords"
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
train_file = base_dir + "gfrc_train.txt"
input_file = pd.read_csv(train_file)
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
train_img_size = (307, 460)
size_reduction = (16, 16)
img_x_pix = train_img_size[1]
img_y_pix = train_img_size[0]
boxs_x = np.ceil(img_x_pix / size_reduction[1])
boxs_y = np.ceil(img_y_pix / size_reduction[0])
max_gt = 14 # max number of animals in images
maxoutsize = np.array([boxs_y, boxs_x, 1, 5 + n_classes, max_gt], dtype=np.int32)
print(maxoutsize)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(list):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=list))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(list):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list))


def read_image(image_path):
    image = cv2.imread(image_path)
    return image


def convert_to_gt1(boxes, out_size):
    boxy, boxx, nanc, outlen, maxgt = out_size
    gt1 = np.zeros(out_size)
    nbox = boxes.shape[0]
    for bx in range(nbox):
        box = boxes.iloc[bx]
        xmin = box.xc - box.wid / 2
        xmax = box.xc + box.wid / 2
        ymin = box.yc - box.hei / 2
        ymax = box.yc + box.hei / 2
        xmincell = np.int32(np.maximum(np.floor(xmin * boxx), 0))
        xmaxcell = np.int32(np.minimum(np.ceil(xmax * boxx), boxx))
        ymincell = np.int32(np.maximum(np.floor(ymin * boxy), 0))
        ymaxcell = np.int32(np.minimum(np.ceil(ymax * boxy), boxy))
        out_vec = np.zeros(outlen)
        out_vec[0:5] = [box.xc, box.yc, box.wid, box.hei, 1]
        class_pos = np.int32(5 + box.clazz)
        out_vec[class_pos] = 1
        for yy in range(ymincell, ymaxcell):
            for xx in range(xmincell, xmaxcell):
                gt1[yy, xx, 0, :, bx] = out_vec
    return gt1


def convert_to_gt2(boxes, out_size):
    boxy, boxx, nanc, outlen, maxgt = out_size
    gt2 = np.zeros(out_size[0:4])
    nbox = boxes.shape[0]
    for bx in range(nbox):
        box = boxes.iloc[bx]
        xcell = np.int32(np.floor(box.xc * boxx))
        ycell = np.int32(np.floor(box.yc * boxy))
        out_vec = np.zeros(outlen)
        out_vec[0:5] = [box.xc, box.yc, box.wid, box.hei, 1]
        class_pos = np.int32(5 + box.clazz)
        out_vec[class_pos] = 1
        gt2[ycell, xcell, 0, :] = out_vec
    return gt2


def paths_to_example(image_path, gt_path, maxoutsize):
    file = cv2.imread(image_path)
    # print(np.dtype(file[0,0,0]).name)
    # need to save image shape as going to convert image to string
    # need shape to be able to reform image
    image_shape = file.shape
    # this safely converts image to a string
    # tfrecords has to save as string(bytes), int64, or float
    #image_string = base64.b64encode(file)
    image_string = file.tostring()
    # save filename as well
    image_path = image_path.encode()

    # read in boxes and save as lists
    boxes = pd.read_csv(gt_path, sep=' ', names=["clazz","xc","yc","wid","hei"])
    gt1 = convert_to_gt1(boxes, maxoutsize)
    gt2 = convert_to_gt2(boxes, maxoutsize)
    gt1_out = gt1.flatten()
    gt2_out = gt2.flatten()

    feature = {
        'imheight': _int64_feature(image_shape[0]),
        'imwidth': _int64_feature(image_shape[1]),
        'imdepth': _int64_feature(image_shape[2]),
        'filename': _bytes_feature(image_path),
        'image_raw': _bytes_feature(image_string),
        'bxy': _int64_feature(maxoutsize[0]),
        'bxx': _int64_feature(maxoutsize[1]),
        'nanc': _int64_feature(maxoutsize[2]),
        'otln': _int64_feature(maxoutsize[3]),
        'mxgt': _int64_feature(maxoutsize[4]),
        'gt1_list': _float_list_feature(gt1_out),
        'gt2_list': _float_list_feature(gt2_out)
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


writer = tf.python_io.TFRecordWriter(output_path)

for gt in range(len(gt_paths)):
    print(gt)
    image_path = image_paths[gt]
    gt_path = gt_paths[gt]
    tf_example = paths_to_example(image_path, gt_path, maxoutsize)
    writer.write(tf_example.SerializeToString())

writer.close()

