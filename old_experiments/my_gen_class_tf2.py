import numpy as np
import pandas as pd
import tensorflow as tf
import cv2





def gfrc_read_from_file(image_path, gt_path, out_size):
    out_size_sm = out_size[0:4]
    # image_in = cv2.imread(image_path)
    image_string = tf.read_file(image_path)
    image_in = tf.image.decode_jpeg(image_string)
    record_defaults = [tf.int32, tf.float32, tf.float32, tf.float32, tf.float32]   # Eight required float columns
    boxes = tf.contrib.data.CsvDataset(gt_path, record_defaults, field_delim=' ')
    print(boxes)
    # boxes = pd.read_csv(gt_path, sep=' ', names=["clazz", "xc", "yc", "wid", "hei"])
    gt1 = convert_to_gt1(boxes, out_size)
    gt2 = convert_to_gt2(boxes, out_size_sm)
    labels = {'gt1': gt1, 'gt2': gt2}
    return image_in, labels


def train_input_fn(image_paths, gt_paths, outsize, steps, batch_size):
    tot_imgs = len(image_paths)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, gt_paths))

    #dataset = dataset.map(lambda im, gt: tuple(tf.py_func(
    #        gfrc_read_from_file, [im, gt, outsize], [tf.float32, tf.float32])))

    dataset = dataset.map(lambda x, y: gfrc_read_from_file(x, y, outsize))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(tot_imgs).repeat(steps).batch(batch_size)

    # Return the dataset.
    return dataset


def read_file_names_in(train_file, base_dir):
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
    return image_paths, gt_paths
