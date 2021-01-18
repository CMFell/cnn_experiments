import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import base64

# output_path = "C:/Benchmark_data/GFRC/gfrc_yolo_out.h5"
input_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo.tfrecords"
output_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo.tfrecords"
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


def paths_to_example(image_path, gt_path):
    file = cv2.imread(image_path)
    # need to save image shape as going to convert image to string
    # need shape to be able to reform image
    image_shape = file.shape
    # this safely converts image to a string
    # tfrecords has to save as string(bytes), int64, or float
    image_string = base64.b64encode(file)
    # save filename as well
    image_path = image_path.encode()

    # read in boxes and save as lists
    boxes = pd.read_csv(gt_path, sep=' ', names=["clazz","xc","yc","wid","hei"])
    xcs = boxes.xc
    xcs = xcs.tolist()
    ycs = boxes.yc
    ycs = ycs.tolist()
    wids = boxes.wid
    wids = wids.tolist()
    heis = boxes.hei
    heis = heis.tolist()
    clazz = boxes.clazz
    clazz = clazz.tolist()
    ngt = boxes.shape[0]

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'filename': _bytes_feature(image_path),
        'image_raw': _bytes_feature(image_string),
        'bbox_xc': _float_list_feature(xcs),
        'bbox_yc': _float_list_feature(ycs),
        'bbox_wid': _float_list_feature(wids),
        'bbox_hei': _float_list_feature(heis),
        'bbox_class': _int64_list_feature(clazz),
        'num_bbox': _int64_feature(ngt)
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


writer = tf.python_io.TFRecordWriter(output_path)

for gt in range(len(gt_paths)):
    image_path = image_paths[gt]
    gt_path = gt_paths[gt]
    print(gt_path)
    tf_example = paths_to_example(image_path, gt_path)
    writer.write(tf_example.SerializeToString())

writer.close()

