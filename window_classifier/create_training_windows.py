import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as tf
from torchvision import transforms as transforms


class Rotate(object):
    """ Create a rotate class """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        image = tf.rotate(img, angle=self.angle)
        return image

class FlipRotate(object):
    """ Create a flip and rotate class """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """ Flips and then Rotates the image.
        
        Args:
            img (Image): The input image 

        Returns:
            image (Image): flipped and rotated imagg.

        """
        image = tf.hflip(img)
        image = tf.rotate(image, angle=self.angle)
        return image


def process_annotation_df_positive(annotations_file_path, img_size):
    annotations_in = pd.read_csv(annotations_file_path)
    pix_width, pix_height = img_size
    annotations_in['wid_pixels'] = np.array(np.multiply(annotations_in['wid'], pix_width), dtype=np.int)
    annotations_in['hei_pixels'] = np.array(np.multiply(annotations_in['height'], pix_height), dtype=np.int)
    annotations_in['square_size'] = np.maximum(annotations_in['wid_pixels'], annotations_in['hei_pixels'])
    annotations_in['xc_pix'] = np.array(np.multiply(annotations_in['xc'], pix_width), dtype=np.int)
    annotations_in['yc_pix'] = np.array(np.multiply(annotations_in['yc'], pix_height), dtype=np.int)
    annotations_in['jitter_size'] = np.array(np.multiply(annotations_in['square_size'], 1.25), np.int)
    annotations_in['xmin'] = np.array(np.minimum(np.maximum(np.subtract(annotations_in['xc_pix'], np.divide(annotations_in['jitter_size'], 2)), 0), np.subtract(pix_width, annotations_in['jitter_size'])), dtype=np.int)
    annotations_in['ymin'] = np.array(np.minimum(np.maximum(np.subtract(annotations_in['yc_pix'], np.divide(annotations_in['jitter_size'], 2)), 0), np.subtract(pix_height, annotations_in['jitter_size'])), dtype=np.int)
    annotations_in['xmax'] = np.add(annotations_in['xmin'], annotations_in['jitter_size'])
    annotations_in['ymax'] = np.add(annotations_in['ymin'], annotations_in['jitter_size'])
    annotations_in['filename'] = [strin.replace('/', '_') for strin in annotations_in.file_loc]
    return annotations_in


def process_annotation_df_negative(detections_fil, img_size):
    pix_width, pix_height = img_size
    detections_fil['wid_pixels'] = np.array(np.multiply(detections_fil['wid'], pix_width), dtype=np.int)
    detections_fil['hei_pixels'] = np.array(np.multiply(detections_fil['hei'], pix_height), dtype=np.int)
    detections_fil['square_size'] = np.maximum(detections_fil['wid_pixels'], detections_fil['hei_pixels'])
    detections_fil['xc_pix'] = np.array(np.multiply(detections_fil['xc'], pix_width), dtype=np.int)
    detections_fil['yc_pix'] = np.array(np.multiply(detections_fil['yc'], pix_height), dtype=np.int)
    detections_fil['xmin'] = np.array(np.minimum(np.maximum(np.subtract(detections_fil['xc_pix'], np.divide(detections_fil['square_size'], 2)), 0), np.subtract(pix_width, detections_fil['square_size'])), dtype=np.int)
    detections_fil['ymin'] = np.array(np.minimum(np.maximum(np.subtract(detections_fil['yc_pix'], np.divide(detections_fil['square_size'], 2)), 0), np.subtract(pix_height, detections_fil['square_size'])), dtype=np.int)
    detections_fil['xmax'] = np.add(detections_fil['xmin'], detections_fil['square_size'])
    detections_fil['ymax'] = np.add(detections_fil['ymin'], detections_fil['square_size'])
    return detections_fil


def create_positive_patches(annotations_in, pos_img_dir, outputdir, augments_list):
    for img_fn, group in annotations_in.groupby('filename'):
        whole_im = cv2.imread(pos_img_dir + img_fn)
        nn = 0
        print(img_fn)
        for row in group.itertuples():
            row_array = whole_im[row.ymin:row.ymax, row.xmin:row.xmax]
            row_array = cv2.cvtColor(row_array, cv2.COLOR_BGR2RGB)
            row_pil = Image.fromarray(row_array)
            jitter_transform = transforms.RandomCrop((row.square_size, row.square_size))
            for aug in augments_list:
                aug_im = aug(row_pil)
                cropped_im = jitter_transform(aug_im)
                animal_outputdir = outputdir + "animal" + str(row.oc) + "/"
                filename_out = row.filename[:-4] + "_" + str(nn) + ".png"
                cropped_im.save(animal_outputdir + filename_out)
                nn += 1


def create_negative_patches(annotations_in, outputdir, augments_list, naugments):
    for img_fn, group in annotations_in.groupby('file'):
        whole_im = cv2.imread(img_fn)
        nn = 0
        print(img_fn)
        just_flnm = img_fn.split('/')
        just_flnm = just_flnm[-1]
        for row in group.itertuples():
            row_array = whole_im[row.ymin:row.ymax, row.xmin:row.xmax]
            row_array = cv2.cvtColor(row_array, cv2.COLOR_BGR2RGB)
            row_pil = Image.fromarray(row_array)
            augments_fil = random.sample(augments_list, naugments)
            for aug in augments_fil:
                aug_im = aug(row_pil)
                filename_out = just_flnm[:-4] + "_" + str(nn) + ".png"
                aug_im.save(outputdir + filename_out)
                nn += 1

# training

# calculate information needed from annotations
annotations_file_path = "/data/old_home_dir/ChrissyF/GFRC/yolo_train_GFRC_bboxes_multi.csv"
img_size = [7360, 4912]
#annotations_in = process_annotation_df_positive(annotations_file_path, img_size)

# create positive patches
augments_list = [Rotate(angle=0), Rotate(angle=90), Rotate(angle=180), Rotate(angle=270), FlipRotate(angle=0), FlipRotate(angle=90), FlipRotate(angle=180), FlipRotate(angle=270)]
pos_img_dir = "/data/old_home_dir/ChrissyF/GFRC/Train/whole_images/pos/"
outputdir = "/data/old_home_dir/ChrissyF/GFRC/window_classifier/multi_class/"
#create_positive_patches(annotations_in, pos_img_dir, outputdir, augments_list)

# create negative windows from false positives of output
tile_img_size = [1856, 1258]
output_from_yolo_bin = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2/boxes_out163_full_train.csv"
outputdir_neg = "/data/old_home_dir/ChrissyF/GFRC/window_classifier/multi_class/not_animals/"
yolo_output_in = pd.read_csv(output_from_yolo_bin)
yolo_output_fil = yolo_output_in.loc[yolo_output_in.tp < 0.01, :]
# negative different filenames and does not jitter
#neg_windows_df = process_annotation_df_negative(yolo_output_fil, tile_img_size)
#create_negative_patches(neg_windows_df, outputdir_neg, augments_list, 2)

# create valid windows from yolo output subset
tile_img_size = [1856, 1248]
output_from_yolo_valid = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2/boxes_out163.csv"
outputdir_valid_pos = "/data/old_home_dir/ChrissyF/GFRC/window_classifier/valid_bin_class/animal/"
outputdir_valid_neg = "/data/old_home_dir/ChrissyF/GFRC/window_classifier/valid_bin_class/not_animal/"
yolo_output_val = pd.read_csv(output_from_yolo_valid)
yolo_output_val_pos = yolo_output_val.loc[yolo_output_val.tp >= 0.25, :]
yolo_output_val_neg = yolo_output_val.loc[yolo_output_val.tp < 0.25, :]
# negative different filenames and does not jitter
val_windows_pos = process_annotation_df_negative(yolo_output_val_pos, tile_img_size)
val_windows_neg = process_annotation_df_negative(yolo_output_val_neg, tile_img_size)

val_augments = [Rotate(angle=0)]
create_negative_patches(val_windows_pos, outputdir_valid_pos, val_augments, 1) 
create_negative_patches(val_windows_neg, outputdir_valid_neg, val_augments, 1) 

# testing

    # create windows from yolo output greater than 0.3 conf

        # pad detection to square

        # crop from image

    # classify as animal or not


