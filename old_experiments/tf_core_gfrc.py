import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

from my_loss_tf2 import loss_gfrc_yolo, total_loss_calc
from create_dataset import BatchGenerator

tf.reset_default_graph()

# Read in file with paths to images and groundtruths
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
train_file = base_dir + "gfrc_train.txt"
# set batch size and take first records as batch
n_batch = 16
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
n_anchors = anchors_in.shape[0]
n_classes = 1
out_len = 5 + n_classes
fin_size = out_len * n_anchors
max_gt = 14
train_img_size = (307, 460)
size_reduction = (16, 16)
learning_rate = 0.0001
n_epochs = 900
# define values for calculating loss
lambda_cl = 1.0
lambda_no = 0.5
lambda_ob = 1.0
lambda_cd = 5.0
threshold = 0.5
boxy = np.int(np.ceil(train_img_size[0] / size_reduction[0]))
boxx = np.int(np.ceil(train_img_size[1] / size_reduction[1]))
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
    'n_classes': n_classes,
    'iou_threshold': threshold
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

paths = np.stack((image_paths, gt_paths), axis=-1)

batch_dict = {
    'BATCH_SIZE': n_batch,
    'IMAGE_H': train_img_size[0],
    'IMAGE_W': train_img_size[1],
    'TRUE_BOX_BUFFER': max_gt,
    'BOX': n_anchors,
    'N_CLASSES': n_classes,
    'GRID_H': boxy,
    'GRID_W': boxx
}

# Assume that each row of `features` corresponds to the same row as `labels`.
assert len(image_paths_list) == len(gt_paths_list)

def read_img(img_path):
    image = tf.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.to_float(image)
    image = tf.divide(image, 255.0)
    return image

dataset_im = tf.data.Dataset.from_tensor_slices(image_paths_list)
dataset_im = dataset_im.map(read_img)

tf_record_path = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/gfrc_yolo_v2.tfrecords"

dataset_all = tf.data.TFRecordDataset(tf_record_path)

def decode(serialized_example):
    # reader = tf.TFRecordReader()

    # _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'imheight': tf.FixedLenFeature([], tf.int64),
            'imwidth': tf.FixedLenFeature([], tf.int64),
            'imdepth': tf.FixedLenFeature([], tf.int64),
            'filename': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'bxy': tf.FixedLenFeature([], tf.int64),
            'bxx': tf.FixedLenFeature([], tf.int64),
            'nanc': tf.FixedLenFeature([], tf.int64),
            'otln': tf.FixedLenFeature([], tf.int64),
            'mxgt': tf.FixedLenFeature([], tf.int64),
            'gt1_list': tf.VarLenFeature(tf.float32),
            'gt2_list': tf.VarLenFeature(tf.float32)
        })

    # Get meta data
    imheight = tf.cast(features['imheight'], tf.int32)
    imwidth = tf.cast(features['imwidth'], tf.int32)
    channels = tf.cast(features['imdepth'], tf.int32)
    boxy = tf.cast(features['bxy'], tf.int32)
    boxx = tf.cast(features['bxx'], tf.int32)
    nanc = tf.cast(features['nanc'], tf.int32)
    outlen = tf.cast(features['otln'], tf.int32)
    maxgt = tf.cast(features['mxgt'], tf.int32)


    # Get data shapes
    image_shape = tf.stack([imheight, imwidth, channels])
    gt1_shape = tf.stack([boxy, boxx, nanc, outlen, maxgt])
    gt2_shape = tf.stack([boxy, boxx, nanc, outlen])

    # Get data
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = tf.to_float(image)
    image = tf.divide(image, 255.0)
    gt1 = tf.sparse_tensor_to_dense(features['gt1_list'], default_value=0.0)
    gt1 = tf.reshape(gt1, gt1_shape)
    gt2 = tf.sparse_tensor_to_dense(features['gt2_list'], default_value=0.0)
    gt2 = tf.reshape(gt2, gt2_shape)
    labels = {'gt1': gt1, 'gt2': gt2}

    return {"image": image}, labels

dataset_all = dataset_all.map(decode)

# create one batch for testing
input_short = input_file.iloc[0:16]


# functions for converting groundtruths
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

# read in data for first file
image_path = base_dir + input_short.img_name.iloc[0]
gt_path = base_dir + input_short.gt_details.iloc[0]
img_out = cv2.imread(image_path)
img_out = np.expand_dims(img_out, axis=0)
boxes = pd.read_csv(gt_path, sep=' ', names=["clazz","xc","yc","wid","hei"])
gt1_out = convert_to_gt1(boxes, maxoutsize)
gt1_out = np.expand_dims(gt1_out, axis=0)
gt2_out = convert_to_gt2(boxes, maxoutsize)
gt2_out = np.expand_dims(gt2_out, axis=0)

for xx in range(1, n_batch):
    image_path = base_dir + input_short.img_name.iloc[xx]
    gt_path = base_dir + input_short.gt_details.iloc[xx]
    img_in = cv2.imread(image_path)
    img_in = np.expand_dims(img_in, axis=0)
    img_out = np.concatenate((img_out, img_in), axis=0)
    boxes = pd.read_csv(gt_path, sep=' ', names=["clazz", "xc", "yc", "wid", "hei"])
    gt1 = convert_to_gt1(boxes, maxoutsize)
    gt2 = convert_to_gt2(boxes, maxoutsize)
    gt1 = np.expand_dims(gt1, axis=0)
    gt2 = np.expand_dims(gt2, axis=0)
    gt1_out = np.concatenate((gt1_out, gt1), axis=0)
    gt2_out = np.concatenate((gt2_out, gt2), axis=0)

print(img_out.shape)
print(gt1_out.shape)
print(gt2_out.shape)

img_out = np.divide(np.array(img_out, dtype=np.float32), 255.0)
gt1_out = np.array(gt1_out, dtype=np.float32)
gt2_out = np.array(gt2_out, dtype=np.float32)

img_out = tf.placeholder(tf.float32, shape=(n_batch, train_img_size[0], train_img_size[1], 3))
gt1_out = tf.placeholder(tf.float32, shape=(n_batch, boxy, boxx, 1, out_len, max_gt))
gt2_out = tf.placeholder(tf.float32, shape=(n_batch, boxy, boxx, n_anchors, out_len))

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def conv_bias(input, weights, biases):

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    return layer


def conv_bn_lr(input, weights):

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    ## Batch normalisation
    batch_mean, batch_var = tf.nn.moments(layer, [0])
    scale = tf.Variable(tf.ones([1]))
    beta = tf.Variable(tf.zeros([1]))
    layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, offset=beta, scale=scale, variance_epsilon=1e-6)

    # activation function
    layer = tf.nn.leaky_relu(layer)

    return layer


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
wtcn13 = create_weights(shape=[1, 1, 1024, fin_size])


weights = {"wtcn1": wtcn1, "wtcn2": wtcn2, "wtcn3": wtcn3, "wtcn4": wtcn4, "wtcn5": wtcn5, "wtcn6":wtcn6,
           "wtcn7": wtcn7, "wtcn8":wtcn8, "wtcn9":wtcn9, "wtcn10": wtcn10, "wtcn11": wtcn11, "wtcn12": wtcn12,
           "wtcn13": wtcn13}
biases = create_biases(fin_size)


def gfrc_model(img_in, weights, biases):
    conv1 = conv_bn_lr(img_in, weights['wtcn1'])
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_bn_lr(pool1, weights['wtcn2'])
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = conv_bn_lr(pool2, weights['wtcn3'])
    conv4 = conv_bn_lr(conv3, weights['wtcn4'])
    conv5 = conv_bn_lr(conv4, weights['wtcn5'])
    pool3 = tf.nn.max_pool(value=conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv6 = conv_bn_lr(pool3, weights['wtcn6'])
    conv7 = conv_bn_lr(conv6, weights['wtcn7'])
    conv8 = conv_bn_lr(conv7, weights['wtcn8'])
    pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv9 = conv_bn_lr(pool4, weights['wtcn9'])
    conv10 = conv_bn_lr(conv9, weights['wtcn10'])
    conv11 = conv_bn_lr(conv10, weights['wtcn11'])
    conv12 = conv_bn_lr(conv11, weights['wtcn12'])
    conv13 = conv_bias(conv12, weights['wtcn13'], biases)

    return conv13


y_pred = gfrc_model(img_out, weights, biases)
individual_losses = loss_gfrc_yolo(gt1=gt1_out, gt2=gt2_out, y_pred=y_pred, dict_in=model_dict)
loss = total_loss_calc(individual_losses, dict_in=model_dict)

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # sess.run(init)
    saver.restore(sess, "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model.ckpt")
    train_generator = BatchGenerator(paths, batch_dict)
    for i in range(100):
        batch = train_generator.next_batch()
        xx = batch[0]
        img_bat = xx['images']
        yy = batch[1]
        gt1_bat = yy['gt1']
        gt2_bat = yy['gt2']
        _, loss_value = sess.run((train, loss), feed_dict={img_out: img_bat, gt1_out: gt1_bat, gt2_out: gt2_bat})
        print(loss_value)

    save_path = saver.save(sess, "E:/CF_Calcs/BenchmarkSets/GFRC/core_test/model.ckpt")

