import os
import numpy as np
import pandas as pd

from preprocess.prepare_data_functions import create_windows, create_tiled_data, create_tiled_data_blanks


# Set input directory containing downloaded GFRC data and choose to prepare valid or traindata
# Directory with files
basedir = 'E:/GFRC_data/'
# type of input valid or train
settype = 'valid'

# values below here can be varied for other data but are set as used for GFRC
# binary or multi
binmulti = 'binary'
# Set sizes of input and output images
input_shape = [4912, 7360]
size_out = [1248, 1856]
# how much should tiles overlap - a pct_overlap of 1 will give non overlapping images
pct_overlap = 1.0
overlap = 1 / pct_overlap
# resize tiles resize_mult = 1 does not resize, else resizes tiles so image is split into tiles of size_out which are then resized 
resize_mult = 1
finsize = (size_out[1] * resize_mult, size_out[0] * resize_mult)
# keep or not blank positive tiles and tiles from negative images prob_blank keeps that percentage of blanks
kb = True
keep_negs = True
prob_blank = 0.01


# image dirs
pos_folder = settype + '_images/pos/'
neg_folder = settype + '_images/neg/'
# csv containing object positions
csv_name = settype + '_images/' + settype + '_GFRC_bboxes_' + binmulti'.csv'
# output folder to save tiles
out_folder = 'yolo_' + settype + '_1248_' + binmulti + '/'
# Full path csv
csv_file = basedir + csv_name
# Read in csv file
datalist = pd.read_csv(csv_file)
# Get list of unique filenames
filez_in = np.unique(datalist.file_loc)
# Full path to save
outpath = basedir + out_folder

# create array of window locations
gfrcwindz = create_windows(size_out, input_shape, overlap)

# create input for tiling
dict4tiling = {"base_dir": basedir + pos_folder, "data_list": datalist, "out_path": outpath, "image_shape": input_shape,
               "size_out": size_out, "final_size" = finsize}
    
for ff in range(filez_in.shape[0]):
    create_tiled_data(filez_in, ff, gfrcwindz, dict4tiling, keep_blanks=kb, pblank=prob_blank)

if keep_negs:
    dict4tilingneg = {"base_dir": basedir + neg_folder, "data_list": datalist, "out_path": outpath, "image_shape": input_shape,
                      "size_out": size_out, "final_size" = finsize}
    create_tiled_datablanks(filez_in, ff, gfrcwindz, dict4tilingneg, pblank=prob_blank)
    
