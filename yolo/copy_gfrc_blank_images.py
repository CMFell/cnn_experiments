import os
from shutil import copyfile

import pandas as pd
import numpy as np
import cv2

# GFRC input
# Directory with files
basedir = '/data/old_home_dir/ChrissyF/GFRC/Valid/whole_images_all/split/'
outdir = '/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_multi/'

# get list of files in basedir
basefiles = os.listdir(basedir)

# get list of files in outdir
outfiles = os.listdir(outdir)

# check if each basedir file is in outdir
for fl in basefiles:
    pngfl = fl[:-4] + '.png'
    img_in = cv2.imread(basedir + fl)
    if pngfl not in outfiles:
        cv2.imwrite(outdir + pngfl, img_in)
        txtname = fl[:-4] + '.txt'
        print(txtname)
        open(outdir + txtname, 'a').close()


# if not copy file

# create blank txt file


