import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from my_gen_class_tf import make_batch
from gfrc_yolo_model_tf import gfrc_tf_yolo_model_fn
import io
import time
import tensorboard

base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
log_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/tb_log/'
model_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/tf_mod/'
# output_path = "C:/Benchmark_data/GFRC/gfrc_yolo_out.h5"
input_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo.tfrecords"
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
n_batch = 32
train_img_size = (307, 460)
size_reduction = (16, 16)
fin_size = (5 + n_classes) * n_anchors
learning_rate = 0.00001
n_epochs = 50
# define values for calculating loss
img_x_pix = train_img_size[1]
img_y_pix = train_img_size[0]
boxs_x = np.ceil(img_x_pix / size_reduction[1])
boxs_y = np.ceil(img_y_pix / size_reduction[0])
lambda_cl = 1.0
lambda_no = 0.5
lambda_ob = 1.0
lambda_cd = 5.0
max_gt = 20 # max number of animals in images
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

#merged = tf.summary.merge_all()

#tf.global_variables_initializer()

#features, labels = make_batch(n_batch, input_path, 1344)

#train_model = gfrc_tf_yolo_model_fn_train(features, labels, model_dict)

#sess = tf.Session()
#sess.run(train_model)

# The op for initializing the variables.
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Load data
# image_paths, gt_paths = read_file_names_in(train_file, base_dir)

log_dir = 'E:/CF_Calcs/'

# Build the estimator
gfrc_tf_yolo = tf.estimator.Estimator(
    model_fn = gfrc_tf_yolo_model_fn,
    params = model_dict,
    model_dir = log_dir
)

# Train the estimator
gfrc_tf_yolo.train(
    steps=10,
    input_fn=lambda : make_batch(n_batch, input_path, 1344)
)


gfrc_tf_yolo.eval(
    steps=10,
    input_fn=lambda : make_batch(n_batch, input_path, 1344)
)


#writer = tf.summary.FileWriter(log_dir)
#writer.flush()
#writer.add_graph(tf.get_default_graph())
#writer.close()


"""
dict_deets = {'anchors': anchors_in, 'threshold': threshold}

test_img_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/Z101_Img00083_48.png"
test_image = cv2.imread(test_img_path)

test_image = np.reshape(test_image, (1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))

net_output = gfrc_yolo.predict(test_image)

rez = yolo_output_to_box(net_output, dict_deets)
print(rez[0])
#rez_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/Z10_Img06910_128.txt"
rez_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/Z101_Img00083_48.txt"
print(pd.read_csv(rez_path))

test_image = test_image[0, :, :, :]

# print(history.history)


for det in range(rez[0].shape[0]):
    xmin = (rez[0].iloc[det, 0] - rez[0].iloc[det, 2] / 2) * 460
    xmax = (rez[0].iloc[det, 0] + rez[0].iloc[det, 2] / 2) * 460
    ymin = (rez[0].iloc[det, 1] - rez[0].iloc[det, 3] / 2) * 307
    ymax = (rez[0].iloc[det, 1] + rez[0].iloc[det, 3] / 2) * 307

    cv2.rectangle(test_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

# print(net_output[0, 7, 23, :])
# print(net_output[0, 8, 25, :])
# print(net_output[0, 9, 28, :])
# print(net_output[0, 13, 22, :])

test_small = cv2.resize(test_image, (920, 614), interpolation=cv2.INTER_CUBIC)

cv2.imshow('image',test_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.ioff()

plt.plot(history.history['loss'])
plt.plot(history.history['metric_cfno'])
plt.plot(history.history['metric_cf'])
plt.plot(history.history['metric_ct'])
plt.plot(history.history['metric_sz'])
plt.plot(history.history['metric_cl'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'no object', 'object', 'centre', 'size', 'class'], loc='upper right')
plt.show()

plt.plot(history.history['metric_TP'])
plt.plot(history.history['metric_FP'])
plt.plot(history.history['metric_FN'])
plt.ylabel('detections')
plt.xlabel('epoch')
plt.legend(['TP', 'FP', 'FN'], loc='upper right')
plt.show()

"""

