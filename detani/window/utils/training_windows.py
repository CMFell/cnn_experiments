import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

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
