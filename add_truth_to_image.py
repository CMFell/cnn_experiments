import pandas as pd
import numpy as np
import cv2
import os


def split_fn(filename_in):
    splits = filename_in.split("/")
    filen = splits[-1]
    filen_split = filen.split("_")
    filename_out = filen_split[0] + "_" + filen_split[1] + ".jpg"
    return filename_out


truth_location = "C:/Users/christina/OneDrive - University of St Andrews/PhD/valid_files/"
detect_images = "C:/Users/christina/OneDrive - University of St Andrews/output_image_test/"

filenamez_in = os.listdir(detect_images)

for fn in filenamez_in:
    print(fn)
    fn_type = fn[-4:]
    if fn_type == '.jpg':
        fn_root = fn[:-10]
        dat_file = truth_location + fn_root + '.csv'
        boxes_truth = pd.read_csv(dat_file, header=None, sep=' ')
        boxes_truth.columns = ['oc', 'xc', 'yc', 'wid', 'hei']

        # convert box sizes to pixels
        xc_img = np.array(boxes_truth.xc * 7360, dtype=np.int)
        yc_img = np.array(boxes_truth.yc * 4912, dtype=np.int)
        wid_pix = np.array(boxes_truth.wid * 7360, dtype=np.int)
        hei_pix = np.array(boxes_truth.hei * 4912, dtype=np.int)

        xmin_img = np.array(xc_img - wid_pix / 2, dtype=np.int)
        xmax_img = np.array(xc_img + wid_pix / 2, dtype=np.int)
        ymin_img = np.array(yc_img - hei_pix / 2, dtype=np.int)
        ymax_img = np.array(yc_img + hei_pix / 2, dtype=np.int)

        file_path = detect_images + fn
        img_in = cv2.imread(file_path)
        for bx in range(len(xmin_img)):
            cv2.rectangle(img_in, (xmin_img[bx], ymin_img[bx]), (xmax_img[bx], ymax_img[bx]), (0, 0, 255), 1)
        image_out_path = detect_images + 'truth_' + fn_root + ".jpg"
        cv2.imwrite(image_out_path, img_in)
        print(image_out_path)



