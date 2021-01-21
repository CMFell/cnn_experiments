import os
import cv2
import math
# import numpy as np
import pandas as pd
from shutil import copy
import random

def decision(probability):
    return random.random() < probability

def copy_whole_image(filez, fl, base_dir, data_list, out_path, multi=False):
    file_name = filez[fl]
    print(file_name)
    from_loc = base_dir + filez[fl]
    copy(from_loc, out_path)
    keep_list = data_list.file_loc == filez[fl]
    file_boxes = data_list[keep_list]
    out_string = ""
    for ln in range(file_boxes.shape[0]):
        line = file_boxes.iloc[ln]
        if multi:
            out_string = out_string + str(line.oc) + ' ' + str(line.xc) + ' ' + str(line.yc) + ' ' + str(line.wid) + ' ' + str(line.height) + '\n'
        else:
            out_string = out_string + str(0) + ' ' + str(line.xc) + ' ' + str(line.yc) + ' ' + str(line.wid) + ' ' + str(line.height) + '\n'
    # get rid of final line separator
    out_string = out_string[:-1]
    txt_out = filez[fl]
    txt_out = txt_out[:-4]
    txt_out = txt_out + '.txt'
    txt_path = out_path + txt_out
    with open(txt_path, "w") as text_file:
        text_file.write(out_string)


def copy_whole_image_resize(filez, fl, base_dir, data_list, out_path, multi=False):
    file_name = filez[fl]
    print(file_name)
    from_loc = base_dir + filez[fl]
    txt_out = filez[fl]
    txt_out = txt_out[:-4]
    out_loc = out_path + txt_out + '.png'
    image_in = cv2.imread(from_loc)
    rowz_in, colz_in, chan_in = image_in.shape
    resized = cv2.resize(image_in, (640, 480)) 
    cv2.imwrite(out_loc, resized)
    keep_list = data_list.file_loc == filez[fl]
    file_boxes = data_list[keep_list]
    out_string = ""
    for ln in range(file_boxes.shape[0]):
        line = file_boxes.iloc[ln]
        # out_string = out_string + str(line.oc) + ' ' + str(line.bx / colz_in) + ' ' \
        #     + str(line.by / rowz_in) + ' ' + str(line.bw / colz_in) + ' ' \
        #     + str(line.bh / rowz_in) + '\n'
        xc = (line.xmin + (line.xmax - line.xmin) / 2) / colz_in
        yc = (line.ymin + (line.ymax - line.ymin) / 2) / rowz_in
        xw = (line.xmax - line.xmin) / colz_in
        yh = (line.ymax - line.ymin)  / rowz_in
        if multi:
            out_string = out_string + str(line.oc) + ' ' + str(xc) + ' ' \
                + str(yc) + ' ' + str(xw) + ' ' + str(yh) + '\n'
        else:
            out_string = out_string + str(0) + ' ' + str(xc) + ' ' \
                + str(yc) + ' ' + str(xw) + ' ' + str(yh) + '\n'
    # get rid of final line separator
    out_string = out_string[:-1]
    txt_out = txt_out + '.txt'
    txt_path = out_path + txt_out
    with open(txt_path, "w") as text_file:
        text_file.write(out_string)


"""
### validation set
base_dir = '/data/old_home_dir/ChrissyF/VEDAI/Valid/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/VEDAI/Valid/test_yolo_VEDAI_bboxes.csv'
out_path = '/data/old_home_dir/ChrissyF/VEDAI/yolo_bin_valid/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)
print(data_list)

for ff in range(len(img_list)):
    copy_whole_image(img_list, ff, base_dir, data_list, out_path)
#    if decision(0.2):
#        copy_whole_image(img_list, ff, base_dir, data_list, out_path)


### subselect for small validation set
base_dir = '/data/old_home_dir/ChrissyF/VEDAI/Valid/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/VEDAI/Valid/test_yolo_VEDAI_bboxes.csv'
out_path = '/data/old_home_dir/ChrissyF/VEDAI/yolo_bin_valid_sm/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)
print(data_list)

for ff in range(len(img_list)):
    if decision(0.2):
       copy_whole_image(img_list, ff, base_dir, data_list, out_path)


### VEDAI details
base_dir = '/data/old_home_dir/ChrissyF/VEDAI/Train/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/VEDAI/train_yolo_VEDAI_bboxes_v2.csv'
out_path = '/data/old_home_dir/ChrissyF/VEDAI/yolo_bin_train/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)

for ff in range(len(img_list)):
    copy_whole_image(img_list, ff, base_dir, data_list, out_path)
"""

"""
### validation set
base_dir = '/data/old_home_dir/ChrissyF/VEDAI/Valid/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/VEDAI/Valid/test_yolo_VEDAI_bboxes.csv'
out_path = '/data/old_home_dir/ChrissyF/VEDAI/yolo_multi_valid/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)
print(data_list)

for ff in range(len(img_list)):
    copy_whole_image(img_list, ff, base_dir, data_list, out_path, multi=True)


### subselect for small validation set
base_dir = '/data/old_home_dir/ChrissyF/VEDAI/Valid/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/VEDAI/Valid/test_yolo_VEDAI_bboxes.csv'
out_path = '/data/old_home_dir/ChrissyF/VEDAI/yolo_multi_valid_sm/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)
print(data_list)

for ff in range(len(img_list)):
    if decision(0.2):
       copy_whole_image(img_list, ff, base_dir, data_list, out_path, multi=True)


### VEDAI details
base_dir = '/data/old_home_dir/ChrissyF/VEDAI/Train/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/VEDAI/train_yolo_VEDAI_bboxes_v2.csv'
out_path = '/data/old_home_dir/ChrissyF/VEDAI/yolo_multi_train/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)

for ff in range(len(img_list)):
    copy_whole_image(img_list, ff, base_dir, data_list, out_path, multi=True)
"""

"""
### INRIA details
base_dir = '/data/old_home_dir/ChrissyF/INRIA/Valid/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/INRIA/00INRIA_bboxes_valid_yolo.csv'
out_path = '/data/old_home_dir/ChrissyF/INRIA/yolo_valid/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)

for ff in range(len(img_list)):
    copy_whole_image_resize(img_list, ff, base_dir, data_list, out_path)
"""

"""
### subselect for small validation set
base_dir = '/data/old_home_dir/ChrissyF/INRIA/Valid/whole_images/'
data_path = '/data/old_home_dir/ChrissyF/INRIA/00INRIA_bboxes_valid_yolo.csv'
out_path = '/data/old_home_dir/ChrissyF/INRIA/yolo_valid_sm/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)

for ff in range(len(img_list)):
    if decision(0.2):
        copy_whole_image_resize(img_list, ff, base_dir, data_list, out_path)
"""

### GFRC details
base_dir = '/data/old_home_dir/ChrissyF/GFRC/Valid/whole_images_all/split/'
data_path = '/data/old_home_dir/ChrissyF/GFRC/Valid/00GFRC_bboxes.csv'
out_path = '/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_bin/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)

for ff in range(len(img_list)):
    copy_whole_image_resize(img_list, ff, base_dir, data_list, out_path)



### subselect for small validation set
out_path = '/data/old_home_dir/ChrissyF/INRIA/yolo_valid1248_bin_subset/'

img_list = os.listdir(base_dir)
data_list = pd.read_csv(data_path)

for ff in range(len(img_list)):
    if decision(0.2):
        copy_whole_image_resize(img_list, ff, base_dir, data_list, out_path)

