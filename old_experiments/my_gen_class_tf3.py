import math
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2


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


def make_batch(batch_size, tf_records_in, nepochs, n_pix):

    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(tf_records_in)

    # Shuffle records create batch
    # dataset = dataset.shuffle(tot_pix).repeat(steps).batch(batch_size)
    #dataset = dataset.shuffle(buffer_size=n_pix)

    # Parse records.
    dataset = dataset.map(decode)

    dataset = dataset.shuffle(n_pix).repeat().batch(batch_size)

    #dataset = dataset.repeat(nepochs)
    #dataset = dataset.batch(batch_size)


    # Batch it up.
    # iterator = dataset.make_one_shot_iterator()
    # image_batch, label_batch = iterator.get_next()

    # return image_batch, label_batch

    return dataset

