import pandas as pd
import numpy as np
import cv2

# GFRC input
# Directory with files
basedir = 'E:/'
# List of boxes
csv_name = 'CF_Calcs/BenchmarkSets/GFRC/yolo_valid_GFRC_bboxes.csv'
# Full path csv
csv_file = basedir + csv_name
# folder to save in
out_folder = 'CF_Calcs/BenchmarkSets/GFRC/Images2copy/'
outpath = basedir + out_folder
# Read in csv file
datalist = pd.read_csv(csv_file)
filez = np.unique(datalist.file_loc)

for fl in range(filez.shape[0]):
    file_name = filez[fl]
    # get root of file name
    from_loc = basedir + file_name
    image_in = cv2.imread(from_loc, -1)
    img_name = file_name[-12:]
    dir_name = file_name[:-13]
    print(dir_name, img_name)
    to_loc = basedir + out_folder + dir_name + "_" + img_name
    cv2.imwrite(to_loc, image_in)
