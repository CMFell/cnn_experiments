# All stolen from https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb

# from keras.models import Sequential, Model
from keras.models import Model
# from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda, Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# from keras.optimizers import SGD, Adam, RMSprop
from keras.optimizers import Adam
from keras.layers.merge import concatenate
# import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
# import imgaug as ia
# from tqdm import tqdm
# from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
# import pickle
import os
import cv2
from start_anew_preprocessing import parse_annotation, BatchGenerator
# from start_anew_utils import WeightReader, decode_netout, draw_boxes
from start_anew_utils import WeightReader
import xml.etree.ElementTree as ET
from scipy.special import expit


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %matplotlib inline

# LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
# 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
# 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
# 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
# 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
# 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
# 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
# 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
LABELS = ['animal']

# IMAGE_H, IMAGE_W = 416, 416
# GRID_H,  GRID_W  = 13 , 13
IMAGE_H, IMAGE_W = 384, 576
GRID_H, GRID_W = 12, 18
BOX = 5
CLASS = len(LABELS)
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD = 0.3  # 0.5
NMS_THRESHOLD = 0.3  # 0.45
# ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS = [2.387088, 2.985595, 1.540179, 1.654902, 3.961755, 3.936809, 2.681468, 1.803889, 5.319540, 6.116692]

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

BATCH_SIZE = 20
WARM_UP_BATCHES = 1000
# TRUE_BOX_BUFFER = 50
TRUE_BOX_BUFFER = 15

# wt_path = 'yolov2.weights'
wt_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"
train_image_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_train_some_img/'
train_annot_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_train_some_ann_voc/'
valid_image_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_valid_subset_img/'
valid_annot_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_valid_subset_ann_voc/'
valid_full_image_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_valid_img/'
valid_full_annot_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_copy_valid_ann_voc/'

tb_log_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/tb_log/'
wt_folder = 'E:/CF_Calcs/BenchmarkSets/GFRC/copy_out/'


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(xx):
    return tf.space_to_depth(xx, block_size=2)


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
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

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

model.summary()

weight_reader = WeightReader(wt_path)

weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))

    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])

layer = model.layers[-4]  # the last convolutional layer
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:5]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    # cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    # ridiculous equivalent of np.repeat
    cell_y = tf.reshape(tf.tile(tf.reshape(tf.range(GRID_H), [-1, 1]), [1, GRID_W]), [-1])
    # tile and reshape in same way as cell_x
    cell_y = tf.to_float(tf.reshape(cell_y, (1, GRID_H, GRID_W, 1, 1)))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

    # coord_mask = tf.zeros(mask_shape)
    # conf_mask = tf.zeros(mask_shape)
    # class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    # tp = tf.Variable(0.)
    # fp = tf.Variable(0.)
    # fn = tf.Variable(0.)

    """
    Adjust prediction
    """
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2])
    # new line convert to whole image
    pred_box_xy_wi = pred_box_xy + cell_grid
    pred_box_xy_wi = tf.divide(pred_box_xy_wi, [GRID_W, GRID_H])

    # adjust w and h
    pred_box_wh = y_pred[..., 2:4]
    # new line adjust so relative to whole image
    pred_box_wh_wi = tf.exp(y_pred[..., 2:4]) * tf.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    pred_box_wh_wi = tf.divide(pred_box_wh_wi, [GRID_W, GRID_H])

    # adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    # adjust class probabilities
    # pred_box_class = y_pred[..., 5:]
    pred_box_class = tf.sigmoid(y_pred[..., 5])

    """
    Adjust ground truth
    """
    # adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
    # add new line give relative to whole image
    true_box_xy_wi = tf.divide(tf.add(true_box_xy, cell_grid), [GRID_W, GRID_H])

    # get w and h
    true_box_wh_wi = y_true[..., 2:4]
    # adjust w and h
    true_box_wh = tf.log(true_box_wh_wi / tf.reshape(ANCHORS, [1, 1, 1, BOX, 2]) + 0.001) + 3
    # the + 0.001 takes out zeros which can't be logged, adding three then removes this adjustment

    # adjust confidence
    true_wh_half = true_box_wh_wi / 2.
    true_mins = true_box_xy_wi - true_wh_half
    true_maxes = true_box_xy_wi + true_wh_half

    pred_wh_half = pred_box_wh_wi / 2.
    pred_mins = pred_box_xy_wi - pred_wh_half
    pred_maxes = pred_box_xy_wi + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh_wi[..., 0] * true_box_wh_wi[..., 1]
    pred_areas = pred_box_wh_wi[..., 0] * pred_box_wh_wi[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    best_ious = tf.reduce_max(iou_scores, axis=4)

    # true_box_conf = iou_scores * y_true[..., 4]
    true_box_conf = y_true[..., 4]

    # adjust class probabilities
    # true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    # coordinate mask: simply the position of the ground truth boxes (the predictors)
    # coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    # coord_mask = tf.expand_dims(y_true[..., 4], axis=-1)

    # confidence mask: penalize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    # conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    ones = y_true[..., 4]
    noones = tf.to_float(best_ious < 0.6)

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    # conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

    # class mask: simply the position of the ground truth boxes (the predictors)
    # class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

    """
    Warm-up training
    """
    # no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    no_boxes_mask = tf.to_float(y_true[..., 4] < 1)
    seen = tf.assign_add(seen, 1.)
    warm_xy = tf.fill(mask_shape, 0.5)
    warm_xy = warm_xy[..., 0:2]
    warm_xy = warm_xy + cell_grid
    warm_wh = tf.reshape(ANCHORS, [1, 1, 1, BOX, 2])

    true_box_xy, true_box_wh = \
        tf.cond(tf.less(seen, WARM_UP_BATCHES),
        lambda: [warm_xy, warm_wh],
        lambda: [true_box_xy, true_box_wh])

    """
    Finalize the loss
    """
    # nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    # nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    # nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    # pred_box_conf = pred_box_conf / best_ious

    loss_conf = tf.reduce_sum(tf.square(1.0 - pred_box_conf) * ones) * OBJECT_SCALE
    loss_noconf = tf.reduce_sum(tf.square(0. - pred_box_conf) * noones) * NO_OBJECT_SCALE
    # loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    # loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    loss_class = tf.reduce_sum(tf.square(1. - pred_box_class) * ones) * CLASS_SCALE
    ones = tf.expand_dims(ones, axis=-1)
    # loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * ones) * COORD_SCALE
    # loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * ones) * COORD_SCALE
    # loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.

    loss = loss_xy + loss_wh + loss_conf + loss_class + loss_noconf

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
    # gt_true = tf.equal(y_true[..., 4], 1)
    # detect_true = pred_box_conf > 0.3
    # ttpp = tf.reduce_sum(tf.to_float(tf.logical_and(gt_true, detect_true)))
    # ffpp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.logical_not(gt_true), detect_true)))
    # ffnn = tf.reduce_sum(tf.to_float(tf.logical_and(gt_true, tf.logical_not(detect_true))))

    """
    Debugging code
    """

    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall)
    # tp = tf.assign_add(tp, ttpp)
    # fp = tf.assign_add(fp, ffpp)
    # fn = tf.assign_add(fn, ffnn)

    test1 = pred_box_conf
    test2 = tf.reduce_max(test1)
    test3 = tf.reduce_min(test1)


    loss = tf.Print(loss, [test2, test3], message='\t')
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_noconf], message='Loss No Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss


generator_config = {
    'IMAGE_H': IMAGE_H,
    'IMAGE_W': IMAGE_W,
    'GRID_H': GRID_H,
    'GRID_W': GRID_W,
    'BOX': BOX,
    'LABELS': LABELS,
    'CLASS': len(LABELS),
    'ANCHORS': ANCHORS,
    'BATCH_SIZE': BATCH_SIZE,
    'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER,
}


def normalize(image):
    return image / 255.


class AllTP(Layer):
    """Stateful Metric to count the total true positives over all batches.
    Assumes predictions and targets of shape `(samples, 1)`.
    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='true_positives', **kwargs):
        super(AllTP, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        gt_pos = tf.equal(y_true[..., 4], 1.)
        pred_pos = tf.greater_equal(tf.sigmoid(y_pred[..., 4]), 0.5)
        correct_preds = tf.logical_and(gt_pos, pred_pos)
        true_pos = tf.reduce_sum(tf.to_int32(correct_preds))
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives,
                                     true_pos),
                        inputs=[y_true, y_pred])
        return current_true_pos + true_pos


class AllFP(Layer):
    """Stateful Metric to count the total true positives over all batches.
    Assumes predictions and targets of shape `(samples, 1)`.
    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='false_positives', **kwargs):
        super(AllFP, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.false_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.false_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        gt_pos = tf.equal(y_true[..., 4], 1.)
        pred_pos = tf.greater_equal(tf.sigmoid(y_pred[..., 4]), 0.5)
        correct_preds = tf.logical_and(tf.logical_not(gt_pos), pred_pos)
        false_pos = tf.reduce_sum(tf.to_int32(correct_preds))
        current_false_pos = self.false_positives * 1
        self.add_update(K.update_add(self.false_positives,
                                     false_pos),
                        inputs=[y_true, y_pred])
        return current_false_pos + false_pos


class AllFN(Layer):
    """Stateful Metric to count the total true positives over all batches.
    Assumes predictions and targets of shape `(samples, 1)`.
    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='false_negatives', **kwargs):
        super(AllFN, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.false_negatives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.false_negatives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        gt_pos = tf.equal(y_true[..., 4], 1.)
        pred_pos = tf.greater_equal(tf.sigmoid(y_pred[..., 4]), 0.5)
        correct_preds = tf.logical_and(gt_pos, tf.logical_not(pred_pos))
        false_neg = tf.reduce_sum(tf.to_int32(correct_preds))
        current_false_neg = self.false_negatives * 1
        self.add_update(K.update_add(self.false_negatives,
                                     false_neg),
                        inputs=[y_true, y_pred])
        return current_false_neg + false_neg


train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
# write parsed annotations to pickle for fast retrieval next time
# with open('train_imgs', 'wb') as fp:
#    pickle.dump(train_imgs, fp)

# read saved pickle of parsed annotations
# with open ('train_imgs', 'rb') as fp:
#    train_imgs = pickle.load(fp)
train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)

valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
# write parsed annotations to pickle for fast retrieval next time
# with open('valid_imgs', 'wb') as fp:
#    pickle.dump(valid_imgs, fp)

# read saved pickle of parsed annotations
# with open ('valid_imgs', 'rb') as fp:
#    valid_imgs = pickle.load(fp)
valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=10,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint(wt_folder + 'weights_coco.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)

tb_counter = len([log for log in os.listdir(os.path.expanduser(tb_log_folder)) if 'coco_' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser(tb_log_folder) + 'coco_' + '_' + str(tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
# optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
# conf_mat = ConfMat()

model.compile(loss=custom_loss, optimizer=optimizer, metrics=[AllTP(), AllFP(), AllFN()])


model.fit_generator(generator=train_batch,
                    steps_per_epoch=len(train_batch),
                    epochs=100,
                    verbose=1,
                    validation_data=valid_batch,
                    validation_steps=len(valid_batch),
                    callbacks=[early_stop, checkpoint, tensorboard],
                    max_queue_size=3)



"""
model.load_weights(wt_folder + 'weights_coco.h5')
model.evaluate_generator(generator=valid_batch,
                         steps=len(valid_batch),
                         verbose=1,
                         max_queue_size=3)

"""



model.load_weights(wt_folder + 'weights_coco.h5')

def convert2box(input_arr):
    boxes = []
    for img in range(input_arr.shape[0]):
        for ycl in range(input_arr.shape[1]):
            for xcl in range(input_arr.shape[2]):
                for anc  in range(input_arr.shape[3]):
                    rez = input_arr[img, ycl, xcl, anc, :]
                    if expit(rez[4]) >= 0.5:
                        xcent = (expit(rez[0]) + xcl) / input_arr.shape[2]
                        ycent = (expit(rez[1]) + ycl) / input_arr.shape[1]
                        xsizhalf = ((np.exp(rez[2]) * ANCHORS[anc * 2]) / GRID_W) / 2
                        ysizhalf = ((np.exp(rez[3]) * ANCHORS[anc * 2 + 1]) / GRID_H) / 2
                        xmin = (xcent - xsizhalf) * IMAGE_W
                        xmax = (xcent + xsizhalf) * IMAGE_W
                        ymin = (ycent - ysizhalf) * IMAGE_H
                        ymax = (ycent + ysizhalf) * IMAGE_H
                        box = [xmin, ymin, xmax, ymax]
                        boxes += [box]
    return boxes


def calc_iou(boxes, box):
    intersect_mins = np.maximum(boxes[:, 0:2], box[0:2])
    intersect_maxes = np.minimum(boxes[:, 2:4], box[2:4])
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[:, 0] * intersect_wh[:, 1]

    boxesw = boxes[:, 2] - boxes[:, 0]
    boxesh = boxes[:, 3] - boxes[:, 1]
    boxes_areas = boxesw * boxesh
    box_area = (box[2] - box[0]) * (box[3] - box[1])

    union_areas = boxes_areas + box_area - intersect_areas
    iou_scores = np.divide(intersect_areas, union_areas)
    best_iou = np.max(iou_scores)
    best_iou_ind = np.argmax(iou_scores)
    out = [best_iou, best_iou_ind]
    return best_iou, best_iou_ind


valid_files = os.listdir(valid_full_image_folder)
ttpp = 0
ffpp = 0
ffnn = 0


truep_all = pd.DataFrame(columns=['file', 'xmin', 'ymin', 'xmax', 'ymax'])
falsep_all = pd.DataFrame(columns=['file', 'xmin', 'ymin', 'xmax', 'ymax'])
falsen_all = pd.DataFrame(columns=['file', 'xmin', 'ymin', 'xmax', 'ymax'])
for ff in range(len(valid_files)):
# for ff in [213,686,856,867,956,967]:
    if ff % 1000 == 0:
        print(ff)
    image_name = valid_files[ff]
    prefix = image_name[:-4]
    if image_name[-4:] == ".png":
        image_in = cv2.imread(valid_full_image_folder + image_name)
        dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
        image_in = image_in / 255.
        image_in = image_in[:, :, ::-1]
        image_in = np.expand_dims(image_in, 0)
        netout = model.predict([image_in, dummy_array])
        # y_pred = netout[..., 4]
        # pred_pos = np.greater_equal(y_pred, 0.5)
        boxes_pred = convert2box(netout)
        annot_file = valid_full_annot_folder + prefix + ".xml"
        annots = []
        tree = ET.parse(annot_file)
        for elem in tree.iter():
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                for attr in list(elem):
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
                annots += [annot]
        true_pos = []
        if len(boxes_pred) > 0:
            boxes_pred = np.array(boxes_pred)
            if len(annots) > 0:
                annots = np.array(annots)
                false_neg = annots
                fn_ind = 0
                for gt in range(annots.shape[0]):
                    true_box = annots[gt, :]
                    if boxes_pred.shape[0] > 0:
                        iou, ind = calc_iou(boxes_pred, true_box)
                        if iou >= 0.3:
                            true_pos += [boxes_pred[ind, :]]
                            boxes_pred = np.delete(boxes_pred, ind, 0)
                            false_neg = np.delete(false_neg, fn_ind, 0)
                            fn_ind -= 1
                        fn_ind += 1
                true_pos = np.array(true_pos)
                ttpp += true_pos.shape[0]
                if true_pos.shape[0] > 0:
                    true_pos = pd.DataFrame(true_pos, columns=['xmin', 'ymin', 'xmax', 'ymax'])
                    true_pos.insert(0, 'file', prefix)
                    truep_all = truep_all.append(true_pos)
                ffpp += boxes_pred.shape[0]
                if boxes_pred.shape[0] > 0:
                    boxes_pred = pd.DataFrame(boxes_pred, columns=['xmin', 'ymin', 'xmax', 'ymax'])
                    boxes_pred.insert(0, 'file', prefix)
                    falsep_all = falsep_all.append(boxes_pred)
                ffnn += false_neg.shape[0]
                if false_neg.shape[0] > 0:
                    false_neg = pd.DataFrame(false_neg, columns=['xmin', 'ymin', 'xmax', 'ymax'])
                    false_neg.insert(0, 'file', prefix)
                    falsen_all = falsen_all.append(false_neg)
            else:
                ffpp += boxes_pred.shape[0]
                boxes_pred = pd.DataFrame(boxes_pred, columns=['xmin', 'ymin', 'xmax', 'ymax'])
                boxes_pred.insert(0, 'file', prefix)
                falsep_all = falsep_all.append(boxes_pred)
        else:
            if len(annots) > 0:
                annots = np.array(annots)
                ffnn += annots.shape[0]
                false_neg = pd.DataFrame(annots, columns=['xmin', 'ymin', 'xmax', 'ymax'])
                false_neg.insert(0, 'file', prefix)
                falsen_all = falsen_all.append(false_neg)

print("TP:", ttpp, " FP:", ffpp, " FN:", ffnn)
# truep_all = np.asarray(truep_all)
# falsep_all = np.asarray(falsep_all)
# falsen_all = np.asarray(falsen_all)
truep_all.to_csv(wt_folder + "tp.csv", index=False)
falsep_all.to_csv(wt_folder + "fp.csv", index=False)
falsen_all.to_csv(wt_folder + "fn.csv", index=False)
# np.savetxt(wt_folder + "tp.csv", truep_all, delimiter=',')
# np.savetxt(wt_folder + "fp.csv", falsep_all, delimiter=',')
# np.savetxt(wt_folder + "fn.csv", falsen_all, delimiter=',')



"""
model.load_weights("weights_coco.h5")

image = cv2.imread('images/giraffe.jpg')
dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))

plt.figure(figsize=(10, 10))

input_image = cv2.resize(image, (416, 416))
input_image = input_image / 255.
input_image = input_image[:, :, ::-1]
input_image = np.expand_dims(input_image, 0)

netout = model.predict([input_image, dummy_array])

boxes = decode_netout(netout[0],
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS,
                      nb_class=CLASS)

image = draw_boxes(image, boxes, labels=LABELS)

plt.imshow(image[:, :, ::-1]);
plt.show()

"""
