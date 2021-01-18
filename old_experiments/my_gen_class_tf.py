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
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'filename': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'bbox_xc': tf.VarLenFeature(tf.float32),
            'bbox_yc': tf.VarLenFeature(tf.float32),
            'bbox_wid': tf.VarLenFeature(tf.float32),
            'bbox_hei': tf.VarLenFeature(tf.float32),
            'bbox_class': tf.VarLenFeature(tf.float32),
            'num_bbox': tf.FixedLenFeature([], tf.int64)
        })

    # Get meta data
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['depth'], tf.int32)
    n_bbox = tf.cast(features['num_bbox'], tf.int32)

    # Get data shapes
    image_shape = tf.parallel_stack([height, width, channels])
    bboxes_shape = tf.parallel_stack([n_bbox, 5])
    bbox_shape_sing = tf.parallel_stack([n_bbox, 1])

    # Get data
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = tf.to_float(image)
    image = tf.divide(image, 255.0)

    # BBOX data is actually dense convert it to dense tensor
    bbox_xc = tf.sparse_tensor_to_dense(features['bbox_xc'], default_value=0.0)
    #bbox_xc = tf.reshape(bbox_xc, bbox_shape_sing)
    bbox_yc = tf.sparse_tensor_to_dense(features['bbox_yc'], default_value=0.0)
    #bbox_yc = tf.reshape(bbox_yc, bbox_shape_sing)
    bbox_wid = tf.sparse_tensor_to_dense(features['bbox_wid'], default_value=0.0)
    #bbox_wid = tf.reshape(bbox_wid, bbox_shape_sing)
    bbox_hei = tf.sparse_tensor_to_dense(features['bbox_hei'], default_value=0.0)
    #bbox_hei = tf.reshape(bbox_hei, bbox_shape_sing)
    bbox_class = tf.sparse_tensor_to_dense(features['bbox_class'], default_value=0)
    #bbox_class = tf.reshape(bbox_class, bbox_shape_sing)

    bboxes = tf.stack((bbox_class, bbox_xc, bbox_yc, bbox_wid, bbox_hei), axis=-1)

    # images, annotations = tf.data.Dataset.batch([image, bboxes], batch_size=2)
    # images, annotations = tf.train.shuffle_batch([image, bboxes], batch_size=2, capacity=30, num_threads=2, min_after_dequeue=10)

    return image, bboxes


def make_batch(batch_size, tf_records_in, tot_pix):

    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(tf_records_in)

    # Parse records.
    dataset = dataset.map(decode)

    # Shuffle records create batch
    dataset = dataset.shuffle(tot_pix).repeat().batch(batch_size)

    # Batch it up.
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

