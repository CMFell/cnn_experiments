from pathlib import Path
import numpy as np
import pandas as pd

dataset_to_use = 'GFRC'
bin_yn = True
name_out = 'rgb_baseline'
subset_to_window = 'train'

if dataset_to_use == 'GFRC':
    ### GFRC
    img_w = 1856
    img_h = 1248
    max_annotations = 14
    anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
               [5.319540, 6.116692]]
    valid_imgs = 312

    if bin_yn:
            # Bin
        if subset_to_window == 'valid':
            files_location_subset = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_bin/"
        elif subset_to_window == 'train':
            files_location_subset = "/data/old_home_dir/ChrissyF/GFRC/yolo_train1248_bin/"
        save_file = "/home/cmf21/pytorch_save/GFRC/Bin/" + subset_to_window + "_truth_for_tiles.csv"
        nclazz = 1

    else:
        # Multi
        if subset_to_window == 'valid':
            files_location_subset = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_multi/"
        elif subset_to_window == 'train':
            files_location_subset = "/data/old_home_dir/ChrissyF/GFRC/yolo_train1248_multi/"
        save_file = "/home/cmf21/pytorch_save/GFRC/Multi/" + subset_to_window + "_truth_for_tiles.csv"
        nclazz = 6

# get list of all text files
txt_files = list(Path(files_location_subset).glob('*.txt'))

df_all_truth = pd.DataFrame(columns = ['class', 'xc', 'yc', 'wid', 'hei', 'file'])

for tt in txt_files:
    bndbxs = pd.read_csv(tt, sep=' ', header=None, names=['class', 'xc', 'yc', 'wid', 'hei'])
    if bndbxs.shape[0] > 0:
        bndbxs['file'] = tt
        df_all_truth = pd.concat((df_all_truth, bndbxs), axis=0)

print(df_all_truth)
print(save_file)
df_all_truth.to_csv(save_file, index=False)


