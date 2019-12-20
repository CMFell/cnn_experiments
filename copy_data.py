import pandas as pd
import numpy as np
import cv2

# GFRC input
# Directory with files
basedir = 'E:/'
# List of boxes
csv_name = 'CF_Calcs/BenchmarkSets/GFRC/yolo_train_GFRC_bboxes.csv'
csv_name = 'C:/Users/christina/OneDrive - University of St Andrews/PhD/SplitFLs/image_data_output.csv'
# Full path csv
csv_file = csv_name
# folder to save in
out_folder = 'CF_Calcs/BenchmarkSets/GFRC/Images2copy/test_images/'
outpath = basedir + out_folder
# Read in csv file
datalist = pd.read_csv(csv_file)
#filez = np.unique(datalist.file_loc)

"""
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
"""

filez_mask = datalist.set_imgs == 'Test -ve'
filez_cd = datalist[filez_mask]

for fl in range(filez_cd.shape[0]):
    FLine = filez_cd.FL.iloc[fl]
    ImNo = filez_cd.ImgNo.iloc[fl]
    ImNo = str(ImNo)
    ImNo = ImNo.zfill(5)
    filen = "Z" + str(FLine) + "/Img" + ImNo + ".jpg"
    print(filen)
    from_loc = basedir + filen
    image_in = cv2.imread(from_loc, -1)
    to_loc = basedir + out_folder + "Z" + str(FLine) + "_Img" + ImNo + ".jpg"
    print(to_loc)
    cv2.imwrite(to_loc, image_in)

