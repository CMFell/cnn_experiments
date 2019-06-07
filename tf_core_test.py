import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from out_box_class_conf import convert_pred_to_output_np
from importweights3 import get_weights, process_layers
from tf_gfrc_model import gfrc_model
from post_p import post_p

test_ep = 5
test_base_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_valid384_subset/"
test_img_path = test_base_path + "gfrc_valid.txt"
test_filez = pd.read_csv(test_img_path, sep=',')
test_imgz = test_filez.img_name
test_imgz = test_imgz.tolist()
test_gt = test_filez.gt_details
test_gt = test_gt.tolist()
n_classes = 1
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
train_img_size = (384, 576)
out_dict1 = {
    'n_classes': n_classes,
    'anchors': anchors_in,
    'iou_threshold': 0.01 # threshold over which probability to keep detections
}

weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"
filt_in = [3, 32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024,
           1024, 3072, 1024]
n_filters_read = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 1024,
                  1024, 1024, 55]
n_filters = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 1024,
             1024, 1024, 30]
filtersizes = [3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 3, 3, 1]

img_out = tf.placeholder(tf.float32, shape=(None, None, None, 3))

lay_out = get_weights(weightspath, n_filters_read, filt_in, filtersizes)
biases_dict, weights_dict = process_layers(lay_out, filt_in, n_filters_read, n_filters, filtersizes)
y_pred = gfrc_model(img_out, weights_dict, biases_dict)
saver = tf.train.Saver()

TPz = 0
FNz = 0
FPz = 0

with tf.Session() as sess:
    rest_path = "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model_" + str(test_ep) + ".ckpt"
    saver.restore(sess, rest_path)
    for ff in range(len(test_imgz)):
        test_img_path = test_base_path + test_imgz[ff]
        test_image = cv2.imread(test_img_path)
        test_image = np.divide(test_image, 255.0)
        test_image = np.reshape(test_image, (1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
        net_output = sess.run(y_pred, feed_dict={img_out: test_image})
        class_out, conf_out, boxes = convert_pred_to_output_np(net_output, out_dict1)
        test_image = test_image[0, :, :, :]
        """
        rez_path = test_base_path + test_gt[ff]
        rez_gt = pd.read_csv(rez_path, sep=' ', header=None)
        rez_gt.columns = ['clazz', 'xc', 'yc', 'wid', 'hei']
        rez_xmin = np.reshape(np.subtract(rez_gt.xc, np.divide(rez_gt.wid, 2)).tolist(), (rez_gt.shape[0], 1))
        rez_xmax = np.reshape(np.add(rez_gt.xc, np.divide(rez_gt.wid, 2)).tolist(), (rez_gt.shape[0], 1))
        rez_ymin = np.reshape(np.subtract(rez_gt.yc, np.divide(rez_gt.hei, 2)).tolist(), (rez_gt.shape[0], 1))
        rez_ymax = np.reshape(np.add(rez_gt.yc, np.divide(rez_gt.hei, 2)).tolist(), (rez_gt.shape[0], 1))
        rez_box = np.hstack((rez_xmin, rez_ymin, rez_xmax, rez_ymax))
        for gt in range(rez_gt.shape[0]):
            if not np.isnan(rez_gt.xc[0]):
                xmin = rez_xmin[gt] * train_img_size[1]
                xmax = rez_xmax[gt] * train_img_size[1]
                ymin = rez_ymin[gt] * train_img_size[0]
                ymax = rez_ymax[gt] * train_img_size[0]
                cv2.rectangle(test_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        """
        for det in range(boxes.shape[0]):
            xmin = boxes[det, 0] * train_img_size[1]
            xmax = boxes[det, 2] * train_img_size[1]
            ymin = boxes[det, 1] * train_img_size[0]
            ymax = boxes[det, 3] * train_img_size[0]
            cv2.rectangle(test_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        test_small = cv2.resize(test_image, (960, 640), interpolation=cv2.INTER_CUBIC)
        test_path = test_imgz[ff]
        test_path = test_path[:-4]
        test_path = test_base_path + "out/" + test_path + "_out.png"
        cv2.imwrite(test_path, test_small)
        print(ff, boxes.shape[0])
        #tp, fn, fp = post_p(rez_box, boxes, 0.1)
        #TPz += tp
        #FNz += fn
        #FPz += fp

print(TPz, FNz, FPz)