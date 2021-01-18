import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from my_gen_class_tf3 import  make_batch
from gfrc_yolo_model_tf2 import gfrc_tf_yolo_model_fn
import io
import time
import tensorboard

base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
log_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/tb_log/'
model_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/tf_mod/'
# output_path = "C:/Benchmark_data/GFRC/gfrc_yolo_out.h5"
input_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_v2.tfrecords"
# output_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_tf.h5"
# input_path = "C:/Benchmark_data/GFRC/gfrc_yolo.tfrecords"
train_file = base_dir + "gfrc_train.txt"
# placeholder for input image size
input_shape = (None, None, 3)
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
n_anchors = anchors_in.shape[0]
n_classes = 1
threshold = 0.5
n_batch = 16
train_img_size = (307, 460)
size_reduction = (16, 16)
fin_size = (5 + n_classes) * n_anchors
learning_rate = 0.00001
n_epochs = 900
# define values for calculating loss
img_x_pix = train_img_size[1]
img_y_pix = train_img_size[0]
boxs_x = np.ceil(img_x_pix / size_reduction[1])
boxs_y = np.ceil(img_y_pix / size_reduction[0])
lambda_cl = 1.0
lambda_no = 0.5
lambda_ob = 1.0
lambda_cd = 5.0
max_gt = 14 # max number of animals in images
max_out_size = [boxs_y, boxs_x, 1, 5 + n_classes, max_gt]
max_out_size = np.int32(max_out_size)
n_pix = 1344
# stepz = np.int(np.multiply(n_epochs, np.ceil(n_pix / n_batch)))
# print(stepz)
model_dict = {
    'batch_size': n_batch,
    'dim': train_img_size,
    'n_channels': input_shape[2],
    'size_reduce': size_reduction,
    'n_anchors': n_anchors,
    'n_classes': n_classes,
    'learn_rate': learning_rate,
    'final_size': fin_size,
    'boxs_x': boxs_x,
    'boxs_y': boxs_y,
    'img_x_pix': img_x_pix,
    'img_y_pix': img_y_pix,
    'anchors': anchors_in,
    'lambda_coord': lambda_cd,
    'lambda_noobj': lambda_no,
    'base_dir': base_dir,
    'lambda_class': lambda_cl,
    'lambda_object': lambda_ob,
    'epochs': n_epochs,
    'max_ground_truth': max_gt,
    'iou_threshold': threshold
}

tf.logging.set_verbosity(tf.logging.INFO)

# Build the estimator
gfrc_tf_yolo = tf.estimator.Estimator(
    model_fn = gfrc_tf_yolo_model_fn,
    params = model_dict,
    model_dir = log_dir
    # , warm_start_from = log_dir
)


# Train the estimator
gfrc_tf_yolo.train(
    input_fn=lambda : make_batch(n_batch, input_path, n_epochs, n_pix),
    steps=420
)

