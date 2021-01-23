import random

import cv2
import numpy as np
import pandas as pd

from window.utils.training_windows import create_negative_patches, create_positive_patches, process_annotation_df_positive, process_annotation_df_negative

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

