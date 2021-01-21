import os
from pathlib import Path

import numpy as np
import pandas as pd 

dataset_to_use = 'GFRC'
bin_yn = True
grey_tf = False
orig_size = True 
name_out = 'rgb_baseline2'
nn = 163


hnm_dir = "/home/cmf21/pytorch_save/GFRC/Bin/" + name_out + "_hnm/"

hnm_file = hnm_dir + "boxes_out" + str(nn) + "_full.csv"
hnm_results = pd.read_csv(hnm_file)

# column tp contains iou of overlap with truths

fp_results = hnm_results[hnm_results.tp < 0.001]
fp_results = fp_results[fp_results.conf > 0.3]

new_img_dir = "/data/old_home_dir/ChrissyF/GFRC/yolo_train1248_bin_hnm/"

img_list = os.listdir(new_img_dir)
img_names = [nm[:-4] for nm in img_list]

old_txt_dir = "/data/old_home_dir/ChrissyF/GFRC/yolo_train1248_bin/"

file_names = fp_results.file.tolist()
file_names = [Path(fn).stem for fn in file_names]

for img in img_names:
    print(img)
    orig_txt_path = old_txt_dir + img + '.txt'
    try:
        orig_txt_file = pd.read_csv(orig_txt_path, sep=' ', header=None)
    except:
        orig_txt_file = pd.DataFrame(columns = ['class', 'xc', 'yc', 'wid', 'hei'])
    orig_txt_file.iloc[:, 0] = orig_txt_file.iloc[:, 0].astype('int32')
    orig_txt_file.columns = ['class', 'xc', 'yc', 'wid', 'hei']
    file_out = orig_txt_file

    file_mask = np.array(file_names) == img

    filter_fp_results = fp_results[file_mask]
    filter_fp_results = filter_fp_results[['class', 'xc', 'yc', 'wid', 'hei']]
    
    new_txt_path = new_img_dir + img + '.txt'

    if filter_fp_results.shape[0] > 0:
        filter_fp_results['class'] = 1
        file_out = pd.concat((file_out, filter_fp_results), axis=0)
        
    file_out.to_csv(new_txt_path, sep=' ', index=False, header=False)

