import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from my_loss_tf4 import loss_gfrc_yolo, loss_gfrc_yolo_ws, total_loss_calc
from create_dataset2 import BatchGenerator
from out_box_class_conf import convert_pred_to_output_np
from importweights2 import load_weights_from_file

tf.reset_default_graph()

weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"
# n_filters = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 1024, 1024, 1024]
# filtersizes = [3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 3, 3]
n_filters = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 1024, 1024, 64, 1024]
filtersizes = [3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 3, 3, 1, 3]
# Read in file with paths to images and groundtruths
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train384_5pc/'
train_file = base_dir + "gfrc_train.txt"
test_img_in = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_zoom/Z101_Img00083_217.png"
test_rez_out = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_zoom/Z101_Img00083_217.txt"
# set batch size and take first records as batch
n_batch = 16
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
n_anchors = anchors_in.shape[0]
n_classes = 1
out_len = 5 + n_classes
fin_size = out_len * n_anchors
max_gt = 14
train_img_size = (384, 576)
size_reduction = (16, 16)
size_reduction = (32, 32)
anchors_in_train = np.divide(np.multiply(anchors_in, size_reduction), (train_img_size[1], train_img_size[0]))
anchors_out_train = np.multiply(anchors_in, size_reduction)
learning_rate = 0.0005
n_epochs = 2
ini_ep = 2
n_ep = 0
ws = True
ini = True


# define values for calculating loss
lambda_cl = 1.0
lambda_no = 1.0
lambda_ob = 1.0
lambda_cd = 1.0
lambda_sz = 1.0
threshold = 0.5
boxy = np.int(np.ceil(train_img_size[0] / size_reduction[0]))
boxx = np.int(np.ceil(train_img_size[1] / size_reduction[1]))
size_out = [boxx, boxy]
maxoutsize = [boxy, boxx, 1, 5 + n_classes, max_gt]
model_dict = {
    'batch_size': n_batch,
    'boxs_x': boxx,
    'boxs_y': boxy,
    'anchors': anchors_in,
    'lambda_coord': lambda_cd,
    'lambda_noobj': lambda_no,
    'lambda_class': lambda_cl,
    'lambda_object': lambda_ob,
    'lambda_size': lambda_sz,
    'n_classes': n_classes,
    'iou_threshold': threshold,
    'n_anchors': n_anchors,
    'num_out': out_len
}
out_dict = {
    'n_classes': n_classes,
    'anchors': anchors_in,
    'iou_threshold': 0.5
}

# read in file names for images and labels
input_file = pd.read_csv(train_file)
image_paths = input_file.img_name
dir_rep = np.repeat(base_dir, image_paths.shape[0])
file_dir = pd.DataFrame(dir_rep, columns=["basedir"])
file_dir = file_dir.basedir
image_paths = file_dir.str.cat(image_paths, sep=None)
image_paths_list = image_paths.tolist()
gt_paths = input_file.gt_details
gt_paths = file_dir.str.cat(gt_paths, sep=None)
gt_paths_list = gt_paths.tolist()
n_pix = len(image_paths)
bat_per_epoch = np.int(np.ceil(n_pix / n_batch))
n_steps = np.int(np.multiply(bat_per_epoch, n_epochs))

paths = np.stack((image_paths, gt_paths), axis=-1)

batch_dict = {
    'BATCH_SIZE': n_batch,
    'IMAGE_H': train_img_size[0],
    'IMAGE_W': train_img_size[1],
    'TRUE_BOX_BUFFER': max_gt,
    'BOX': n_anchors,
    'N_CLASSES': n_classes,
    'GRID_H': boxy,
    'GRID_W': boxx,
    'ANCHORS': anchors_in
}

tf.logging.set_verbosity(tf.logging.INFO)
# img_out = tf.placeholder(tf.float32, shape=(None, train_img_size[0], train_img_size[1], 3))
img_out = tf.placeholder(tf.float32, shape=(None, None, None, 3))
# gt1_out = tf.placeholder(tf.float32, shape=(None, boxy, boxx, n_anchors, out_len))
gt1_out = tf.placeholder(tf.float32, shape=(None, None, None, n_anchors, out_len))
gt2_out = tf.placeholder(tf.float32, shape=(None, 1, 1, 1, 4, max_gt))


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def conv_bias(lay_input, weights_in, biases_in):
    # Creating the final output convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases_in
    return layer


def conv_bn2_lr(lay_input, weights_in):
    # Creating a default convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME')
    # Batch normalisation
    batch_mean, batch_var = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)
    scale = tf.Variable(tf.ones([1]))
    beta = tf.Variable(tf.zeros([1]))
    layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, offset=beta, scale=scale, variance_epsilon=1e-6)
    # activation function
    layer = tf.nn.leaky_relu(layer)
    return layer


def conv_bn_lr(lay_input, weights_in, batch_mean, batch_var):
    # Creating a default convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME')
    # Batch normalisation
    scale = tf.Variable(tf.ones([1]))
    beta = tf.Variable(tf.zeros([1]))
    batch_mean, batch_var = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)
    layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, offset=beta, scale=scale, variance_epsilon=1e-6)
    # activation function
    layer = tf.nn.leaky_relu(layer)
    return layer


layer_list = load_weights_from_file(weightspath, n_filters, filtersizes)
wtcn1 = tf.convert_to_tensor(layer_list[0][0][0])
wtcn2 = tf.convert_to_tensor(layer_list[1][0][0])
wtcn3 = tf.convert_to_tensor(layer_list[2][0][0])
wtcn4 = tf.convert_to_tensor(layer_list[3][0][0])
wtcn5 = tf.convert_to_tensor(layer_list[4][0][0])
wtcn6 = tf.convert_to_tensor(layer_list[5][0][0])
wtcn7 = tf.convert_to_tensor(layer_list[6][0][0])
wtcn8 = tf.convert_to_tensor(layer_list[7][0][0])
wtcn9 = tf.convert_to_tensor(layer_list[8][0][0])
wtcn10 = tf.convert_to_tensor(layer_list[9][0][0])
wtcn11 = tf.convert_to_tensor(layer_list[10][0][0])
wtcn12 = tf.convert_to_tensor(layer_list[11][0][0])
wtcn13 = tf.convert_to_tensor(layer_list[12][0][0])
wtcn14 = tf.convert_to_tensor(layer_list[13][0][0])
wtcn15 = tf.convert_to_tensor(layer_list[14][0][0])
wtcn16 = tf.convert_to_tensor(layer_list[15][0][0])
wtcn17 = tf.convert_to_tensor(layer_list[16][0][0])
wtcn18 = tf.convert_to_tensor(layer_list[17][0][0])
wtcn19 = tf.convert_to_tensor(layer_list[18][0][0])
wtcn20 = tf.convert_to_tensor(layer_list[19][0][0])
wtcn21 = tf.convert_to_tensor(layer_list[20][0][0])
wtcn22 = tf.convert_to_tensor(layer_list[21][0][0])

bn1m = tf.convert_to_tensor(layer_list[0][1][0])
bn2m = tf.convert_to_tensor(layer_list[1][1][0])
bn3m = tf.convert_to_tensor(layer_list[2][1][0])
bn4m = tf.convert_to_tensor(layer_list[3][1][0])
bn5m = tf.convert_to_tensor(layer_list[4][1][0])
bn6m = tf.convert_to_tensor(layer_list[5][1][0])
bn7m = tf.convert_to_tensor(layer_list[6][1][0])
bn8m = tf.convert_to_tensor(layer_list[7][1][0])
bn9m = tf.convert_to_tensor(layer_list[8][1][0])
bn10m = tf.convert_to_tensor(layer_list[9][1][0])
bn11m = tf.convert_to_tensor(layer_list[10][1][0])
bn12m = tf.convert_to_tensor(layer_list[11][1][0])
bn13m = tf.convert_to_tensor(layer_list[12][1][0])
bn14m = tf.convert_to_tensor(layer_list[13][1][0])
bn15m = tf.convert_to_tensor(layer_list[14][1][0])
bn16m = tf.convert_to_tensor(layer_list[15][1][0])
bn17m = tf.convert_to_tensor(layer_list[16][1][0])
bn18m = tf.convert_to_tensor(layer_list[17][1][0])
bn19m = tf.convert_to_tensor(layer_list[18][1][0])
bn20m = tf.convert_to_tensor(layer_list[19][1][0])
bn21m = tf.convert_to_tensor(layer_list[20][1][0])
bn22m = tf.convert_to_tensor(layer_list[21][1][0])
bn1v = tf.convert_to_tensor(layer_list[0][1][1])
bn2v = tf.convert_to_tensor(layer_list[1][1][1])
bn3v = tf.convert_to_tensor(layer_list[2][1][1])
bn4v = tf.convert_to_tensor(layer_list[3][1][1])
bn5v = tf.convert_to_tensor(layer_list[4][1][1])
bn6v = tf.convert_to_tensor(layer_list[5][1][1])
bn7v = tf.convert_to_tensor(layer_list[6][1][1])
bn8v = tf.convert_to_tensor(layer_list[7][1][1])
bn9v = tf.convert_to_tensor(layer_list[8][1][1])
bn10v = tf.convert_to_tensor(layer_list[9][1][1])
bn11v = tf.convert_to_tensor(layer_list[10][1][1])
bn12v = tf.convert_to_tensor(layer_list[11][1][1])
bn13v = tf.convert_to_tensor(layer_list[12][1][1])
bn14v = tf.convert_to_tensor(layer_list[13][1][1])
bn15v = tf.convert_to_tensor(layer_list[14][1][1])
bn16v = tf.convert_to_tensor(layer_list[15][1][1])
bn17v = tf.convert_to_tensor(layer_list[16][1][1])
bn18v = tf.convert_to_tensor(layer_list[17][1][1])
bn19v = tf.convert_to_tensor(layer_list[18][1][1])
bn20v = tf.convert_to_tensor(layer_list[19][1][1])
bn21v = tf.convert_to_tensor(layer_list[20][1][1])
bn22v = tf.convert_to_tensor(layer_list[21][1][1])

"""
wtcn1 = create_weights(shape=[3, 3, 3, 32])
wtcn2 = create_weights(shape=[3, 3, 32, 64])
wtcn3 = create_weights(shape=[3, 3, 64, 128])
wtcn4 = create_weights(shape=[1, 1, 128, 64])
wtcn5 = create_weights(shape=[3, 3, 64, 128])
wtcn6 = create_weights(shape=[3, 3, 128, 256])
wtcn7 = create_weights(shape=[1, 1, 256, 128])
wtcn8 = create_weights(shape=[3, 3, 128, 256])
wtcn9 = create_weights(shape=[3, 3, 256, 512])
wtcn10 = create_weights(shape=[1, 1, 512, 256])
wtcn11 = create_weights(shape=[3, 3, 256, 512])
wtcn12 = create_weights(shape=[3, 3, 512, 1024])
"""

wtcn23 = create_weights(shape=[1, 1, 1024, fin_size])
bias23 = create_biases(fin_size)



weights_dict = {"wtcn1": wtcn1, "wtcn2": wtcn2, "wtcn3": wtcn3, "wtcn4": wtcn4, "wtcn5": wtcn5, "wtcn6": wtcn6,
                "wtcn7": wtcn7, "wtcn8": wtcn8, "wtcn9": wtcn9, "wtcn10": wtcn10, "wtcn11": wtcn11, "wtcn12": wtcn12,
                "wtcn13": wtcn13, "wtcn14": wtcn14, "wtcn15": wtcn15, "wtcn16": wtcn16, "wtcn17": wtcn17,
                "wtcn18": wtcn18, "wtcn19": wtcn19, "wtcn20": wtcn20, "wtcn21": wtcn21, "wtcn22": wtcn22,
                "wtcn23": wtcn23}
biases_dict = {"bn1m": bn1m, "bn2m": bn2m, "bn3m": bn3m, "bn4m": bn4m, "bn5m":bn5m, "bn6m":bn6m, "bn7m":bn7m,
               "bn8m": bn8m, "bn9m": bn9m, "bn10m": bn10m, "bn11m": bn11m, "bn12m": bn12m, "bn13m": bn13m,
               "bn14m": bn14m, "bn15m": bn15m, "bn16m": bn16m, "bn17m": bn17m, "bn18m": bn18m, "bn19m": bn19m,
               "bn20m": bn20m, "bn21m":bn21m, "bn22m":bn22m,
               "bn1v": bn1v, "bn2v": bn2v, "bn3v": bn3v, "bn4v": bn4v, "bn5v":bn5v, "bn6v":bn6v, "bn7v":bn7v,
               "bn8v": bn8v, "bn9v": bn9v, "bn10v": bn10v, "bn11v": bn11v, "bn12v": bn12v, "bn13v": bn13v,
               "bn14v": bn14v, "bn15v": bn15v, "bn16v": bn16v, "bn17v": bn17v, "bn18v": bn18v, "bn19v":bn19v,
               "bn20v": bn20v, "bn21v": bn21v, "bn22v": bn22v, "bn23": bias23}


def gfrc_model(img_in, weights, biases):
    conv1 = conv_bn_lr(img_in, weights['wtcn1'], biases['bn1m'], biases['bn1v'])
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_bn_lr(pool1, weights['wtcn2'], biases['bn2m'], biases['bn2v'])
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = conv_bn_lr(pool2, weights['wtcn3'], biases['bn3m'], biases['bn3v'])
    conv4 = conv_bn_lr(conv3, weights['wtcn4'], biases['bn4m'], biases['bn4v'])
    conv5 = conv_bn_lr(conv4, weights['wtcn5'], biases['bn5m'], biases['bn5v'])
    pool3 = tf.nn.max_pool(value=conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv6 = conv_bn_lr(pool3, weights['wtcn6'], biases['bn6m'], biases['bn6v'])
    conv7 = conv_bn_lr(conv6, weights['wtcn7'], biases['bn7m'], biases['bn7v'])
    conv8 = conv_bn_lr(conv7, weights['wtcn8'], biases['bn8m'], biases['bn8v'])
    pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv9 = conv_bn_lr(pool4, weights['wtcn9'], biases['bn9m'], biases['bn9v'])
    conv10 = conv_bn_lr(conv9, weights['wtcn10'], biases['bn10m'], biases['bn10v'])
    conv11 = conv_bn_lr(conv10, weights['wtcn11'], biases['bn11m'], biases['bn11v'])
    conv12 = conv_bn_lr(conv11, weights['wtcn12'], biases['bn12m'], biases['bn12v'])
    conv13 = conv_bn_lr(conv12, weights['wtcn13'], biases['bn13m'], biases['bn13v'])
    conv14 = conv_bn_lr(conv13, weights['wtcn14'], biases['bn14m'], biases['bn14v'])
    conv15 = conv_bias(conv14, weights['wtcn23'], biases['bn23'])

    return conv15


def gfrc_model2(img_in, weights, biases):
    conv1 = conv_bn_lr(img_in, weights['wtcn1'], biases['bn1m'], biases['bn1v'])
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_bn_lr(pool1, weights['wtcn2'], biases['bn2m'], biases['bn2v'])
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = conv_bn_lr(pool2, weights['wtcn3'], biases['bn3m'], biases['bn3v'])
    conv4 = conv_bn_lr(conv3, weights['wtcn4'], biases['bn4m'], biases['bn4v'])
    conv5 = conv_bn_lr(conv4, weights['wtcn5'], biases['bn5m'], biases['bn5v'])
    pool3 = tf.nn.max_pool(value=conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv6 = conv_bn_lr(pool3, weights['wtcn6'], biases['bn6m'], biases['bn6v'])
    conv7 = conv_bn_lr(conv6, weights['wtcn7'], biases['bn7m'], biases['bn7v'])
    conv8 = conv_bn_lr(conv7, weights['wtcn8'], biases['bn8m'], biases['bn8v'])
    pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv9 = conv_bn_lr(pool4, weights['wtcn9'], biases['bn9m'], biases['bn9v'])
    conv10 = conv_bn_lr(conv9, weights['wtcn10'], biases['bn10m'], biases['bn10v'])
    conv11 = conv_bn_lr(conv10, weights['wtcn11'], biases['bn11m'], biases['bn11v'])
    conv12 = conv_bn_lr(conv11, weights['wtcn12'], biases['bn12m'], biases['bn12v'])
    conv13 = conv_bn_lr(conv12, weights['wtcn13'], biases['bn13m'], biases['bn13v'])
    pool5 = tf.nn.max_pool(value=conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv14 = conv_bn_lr(pool5, weights['wtcn14'], biases['bn14m'], biases['bn14v'])
    conv15 = conv_bn_lr(conv14, weights['wtcn15'], biases['bn15m'], biases['bn15v'])
    conv16 = conv_bn_lr(conv15, weights['wtcn16'], biases['bn16m'], biases['bn16v'])
    conv17 = conv_bn_lr(conv16, weights['wtcn17'], biases['bn17m'], biases['bn17v'])
    conv18 = conv_bn_lr(conv17, weights['wtcn18'], biases['bn18m'], biases['bn18v'])
    conv19 = conv_bn_lr(conv18, weights['wtcn19'], biases['bn19m'], biases['bn19v'])
    conv20 = conv_bn_lr(conv19, weights['wtcn20'], biases['bn20m'], biases['bn20v'])
    conv21 = conv_bn_lr(conv13, weights['wtcn21'], biases['bn21m'], biases['bn21v'])
    s2d1 = tf.space_to_depth(conv21, 2)
    concat1 = tf.concat([conv20, s2d1], axis=-1)
    conv22 = conv_bn_lr(concat1, weights['wtcn22'], biases['bn22m'], biases['bn22v'])
    conv23 = conv_bias(conv22, weights['wtcn23'], biases['bn23'])

    return conv23


y_pred = gfrc_model2(img_out, weights_dict, biases_dict)
individual_losses = loss_gfrc_yolo(gt1=gt1_out, gt2=gt2_out, y_pred=y_pred, dict_in=model_dict)
ws_losses = loss_gfrc_yolo_ws(gt1=gt1_out, gt2=gt2_out, y_pred=y_pred, dict_in=model_dict)
loss = total_loss_calc(individual_losses, dict_in=model_dict)
loss_ws = total_loss_calc(ws_losses, dict_in=model_dict)

if ws:
    met_cf_ngt = ws_losses["conf_loss_nogt"]
    met_cf_gt = ws_losses["conf_loss_gt"]
    met_cnt = ws_losses["cent_loss"]
    met_sz = ws_losses["size_loss"]
    met_cl = ws_losses["class_loss"]
    met_tp = ws_losses["TP"]
    met_fp = ws_losses["FP"]
    met_fn = ws_losses["FN"]
    met_re = ws_losses["Re"]
    met_pr = ws_losses["Pr"]
    met_fpr = ws_losses["FPR"]
    met_t1 = ws_losses["TP"]
    met_t2 = ws_losses["TP"]
    tf.summary.scalar('loss', loss_ws)
else:
    met_cf_ngt = individual_losses["conf_loss_nogt"]
    met_cf_gt = individual_losses["conf_loss_gt"]
    met_cnt = individual_losses["cent_loss"]
    met_sz = individual_losses["size_loss"]
    met_cl = individual_losses["class_loss"]
    met_tp = individual_losses["TP"]
    met_fp = individual_losses["FP"]
    met_fn = individual_losses["FN"]
    met_re = individual_losses["Re"]
    met_pr = individual_losses["Pr"]
    met_fpr = individual_losses["FPR"]
    met_t1 = individual_losses["TP"]
    met_t2 = individual_losses["TP"]
    tf.summary.scalar('loss', loss)


tf.summary.scalar('met_cf_ngt', met_cf_ngt)
tf.summary.scalar('met_cf_gt', met_cf_gt)
tf.summary.scalar('met_cnt', met_cnt)
tf.summary.scalar('met_sz', met_sz)
tf.summary.scalar('met_cl', met_cl)
tf.summary.scalar('met_tp', met_tp)
tf.summary.scalar('met_fp', met_fp)
tf.summary.scalar('met_fn', met_fn)
tf.summary.scalar('met_re', met_re)
tf.summary.scalar('met_pr', met_pr)
tf.summary.scalar('met_fpr', met_fpr)
tf.summary.scalar('met_t1', met_t1)
tf.summary.scalar('met_t2', met_t2)

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
train_ws = optimizer.minimize(loss_ws)

merged = tf.summary.merge_all()

if ini:
    init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    rest_path = "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model_" + str(ini_ep) + ".ckpt"
    if ini:
        sess.run(init)
    else:
        saver.restore(sess, rest_path)
    summary_writer = tf.summary.FileWriter("E:/CF_Calcs/BenchmarkSets/GFRC/tb_log/", tf.get_default_graph())
    train_generator = BatchGenerator(paths, batch_dict)
    for ep in range(n_epochs):
        cf_ngt = 0
        cf_gt = 0
        cnt_ls = 0
        sz_ls = 0
        cl_ls = 0
        tp_ls = 0
        fp_ls = 0
        fn_ls = 0
        re_ls = 0
        pr_ls = 0
        tl = 0
        tst1 = 1
        ind_in = 0
        for bt in range(bat_per_epoch):
            batch = train_generator.next_batch(ind_in)
            xx = batch[0]
            img_bat = xx['images']
            yy = batch[1]
            gt1_bat = yy['gt1']
            gt2_bat = yy['gt2']
            ind_in = yy['ind']
            if ws:
                input2sess = (train_ws, loss_ws, merged, met_cf_ngt, met_cf_gt, met_cnt, met_sz, met_cl,
                              met_tp, met_fp, met_fn, met_re, met_pr, met_t1)
            else:
                input2sess = (train, loss, merged, met_cf_ngt, met_cf_gt, met_cnt, met_sz, met_cl,
                              met_tp, met_fp, met_fn, met_re, met_pr, met_t1)
            dict4feed = {img_out: img_bat, gt1_out: gt1_bat, gt2_out: gt2_bat}
            _, loss_value, summary, outcfngt, outcfgt, outcnt, outsz, outcl, outtp, outfp, outfn, outre, outpr, test1 = sess.run(input2sess, feed_dict=dict4feed)
            summary_writer.add_summary(summary, ep * bat_per_epoch + bt)
            cf_ngt += outcfngt
            cf_gt += outcfgt
            cnt_ls += outcnt
            sz_ls += outsz
            cl_ls += outcl
            tp_ls += outtp
            fp_ls += outfp
            fn_ls += outfn
            re_ls += outre
            pr_ls += outpr
            tl += loss_value
            # test per epoch should be 1712 ish
            if test1 < tst1:
                tst1 = test1
            """
            print("Batch ", bt + 1, " - loss: ", "{0:.2f}".format(loss_value), " - no_gt: ", "{0:.2f}".format(outcfngt),
                  " - gt: ", "{0:.2f}".format(outcfgt), " - cent: ", "{0:.2f}".format(outcnt),
                  " - size: ", "{0:.2f}".format(outsz), " - class: ", "{0:.2f}".format(outcl),
                  " - TP: ", "{0:.2f}".format(outtp), " - FP: ", "{0:.2f}".format(outfp),
                  " - FN: ", "{0:.2f}".format(outfn), " - Test: ", "{0:.2f}".format(tst1))
            """
        cf_ngt = cf_ngt / bat_per_epoch
        cf_gt = cf_gt / bat_per_epoch
        cnt_ls = cnt_ls / bat_per_epoch
        sz_ls = sz_ls / bat_per_epoch
        cl_ls = cl_ls / bat_per_epoch
        tp_ls = tp_ls
        fp_ls = fp_ls
        fn_ls = fn_ls
        re_ls = re_ls / bat_per_epoch
        pr_ls = pr_ls / bat_per_epoch
        tl = tl / bat_per_epoch
        print("Epoch ", ep + 1, " - loss: ", "{0:.2f}".format(tl), " - no_gt: ", "{0:.2f}".format(cf_ngt),
              " - gt: ", "{0:.2f}".format(cf_gt), " - cent: ", "{0:.2f}".format(cnt_ls),
              " - size: ", "{0:.2f}".format(sz_ls), " - class: ", "{0:.2f}".format(cl_ls),
              " - TP: ", "{0:.2f}".format(tp_ls), " - FP: ", "{0:.2f}".format(fp_ls),
              " - FN: ", "{0:.2f}".format(fn_ls), " - Recall: ", "{0:.2f}".format(re_ls),
              " - Precision: ", "{0:.2f}".format(pr_ls), " - Test: ", "{0:.2f}".format(tst1))

        if ep % 1 == 0 and ep > 0:
            print("Saving...")
            path2save = "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model_" + str(n_ep) + ".ckpt"
            save_path = saver.save(sess, path2save)
            n_ep += 1

    """
    test_img_path = test_img_in
    test_image = cv2.imread(test_img_path)
    test_image = np.reshape(test_image, (1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
    net_output = sess.run(y_pred, feed_dict={img_out: test_image})
    """
    """
    for op in range(30):
        out_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_new/net_out" + str(op) + ".csv"
        np.savetxt(out_path, net_output[0, :, :, op])
    """
    """
    class_out, conf_out, boxes = convert_pred_to_output_np(net_output, out_dict)
    rez_path = test_rez_out
    print(pd.read_csv(rez_path, sep=' ', header=None))
    test_image = test_image[0, :, :, :]
    print(boxes.shape)
    print(boxes)
    for det in range(boxes.shape[0]):
        xmin = boxes[det, 0] * 480
        xmax = boxes[det, 2] * 480
        ymin = boxes[det, 1] * 320
        ymax = boxes[det, 3] * 320

        cv2.rectangle(test_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    test_small = cv2.resize(test_image, (920, 614), interpolation=cv2.INTER_CUBIC)

    cv2.imshow('image', test_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

