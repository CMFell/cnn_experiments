import pandas as pd
import numpy as np
import cv2


def split_fn(filename_in):
    splits = filename_in.split("/")
    filen = splits[-1]
    filen_split = filen.split("_")
    filename_out = filen_split[0] + "_" + filen_split[1] + ".jpg"
    return filename_out


def patch_no(filename_in):
    splits = filename_in.split("/")
    filen = splits[-1]
    filen_split = filen.split("_")
    patchno = int(filen_split[2][:-4])
    return patchno


def basic_nms(xmins, xmaxs, ymins, ymaxs, confz, thresh):

    conf_ord = np.argsort(confz)
    xmins = xmins[conf_ord]
    xmaxs = xmaxs[conf_ord]
    ymins = ymins[conf_ord]
    ymaxs = ymaxs[conf_ord]

    boxes_ot = pd.DataFrame(columns=["XMIN", "XMAX", "YMIN", "YMAX"])

    xmins = np.array(xmins.tolist())
    xmaxs = np.array(xmaxs.tolist())
    ymins = np.array(ymins.tolist())
    ymaxs = np.array(ymaxs.tolist())

    while len(xmins) > 0:

        xmn = xmins[0]
        xmns = np.array(xmins[1:])
        xmx = xmaxs[0]
        xmxs = np.array(xmaxs[1:])
        ymn = ymins[0]
        ymns = np.array(ymins[1:])
        ymx = ymaxs[0]
        ymxs = np.array(ymaxs[1:])

        ol_wid = np.minimum(xmx, xmxs) - np.maximum(xmn, xmns)
        ol_hei = np.minimum(ymx, ymxs) - np.maximum(ymn, ymns)

        ol_x = np.maximum(0, ol_wid)
        ol_y = np.maximum(0, ol_hei)

        distx = np.subtract(xmxs, xmns)
        disty = np.subtract(ymxs, ymns)
        bxx = xmx - xmn
        bxy = ymx - ymn

        ol_area = np.multiply(ol_x, ol_y)
        bx_area = bxx * bxy
        bxs_area = np.multiply(distx, disty)

        ious = np.divide(ol_area, np.subtract(np.add(bxs_area, bx_area), ol_area))
        mask_bxs = np.greater(ious, thresh)

        if np.sum(mask_bxs) > 0:
            box_ot = pd.DataFrame(index=[1], columns=["XMIN", "XMAX", "YMIN", "YMAX"])
            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]

            xmns = np.append(xmns, xmn)
            xmxs = np.append(xmxs, xmx)
            ymxs = np.append(ymxs, ymx)
            ymns = np.append(ymns, ymn)

            box_ot.XMIN.iloc[0] = np.min(xmns)
            box_ot.XMAX.iloc[0] = np.max(xmxs)
            box_ot.YMIN.iloc[0] = np.min(ymns)
            box_ot.YMAX.iloc[0] = np.max(ymxs)

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out[1:] = mask_bxs
            mask_out = np.logical_not(mask_out)

            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
        else:
            box_ot = pd.DataFrame(index=[1], columns=["XMIN", "XMAX", "YMIN", "YMAX"])
            box_ot.XMIN.iloc[0] = xmn
            box_ot.XMAX.iloc[0] = xmx
            box_ot.YMIN.iloc[0] = ymn
            box_ot.YMAX.iloc[0] = ymx
            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out = np.logical_not(mask_out)
            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0)

    xmins_out = boxes_ot.XMIN.tolist()
    xmaxs_out = boxes_ot.XMAX.tolist()
    ymins_out = boxes_ot.YMIN.tolist()
    ymaxs_out = boxes_ot.YMAX.tolist()

    return xmins_out, xmaxs_out, ymins_out, ymaxs_out


file_location = "E:/CF_Calcs/BenchmarkSets/GFRC/pytorch_save/boxes_out10.csv"
image_location = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_valid384_subset/"

boxes_found = pd.read_csv(file_location)

filenamez_in = boxes_found.file
all_filez = np.unique(filenamez_in)

# convert box sizes to pixels
xc_pix = np.array(boxes_found.xc * 576, dtype=np.int)
yc_pix = np.array(boxes_found.yc * 384, dtype=np.int)
wid_pix = np.array(boxes_found.wid * 576, dtype=np.int)
hei_pix = np.array(boxes_found.hei * 384, dtype=np.int)

xmin_img = np.array(xc_pix - wid_pix / 2, dtype=np.int)
xmax_img = np.array(xc_pix + wid_pix / 2, dtype=np.int)
ymin_img = np.array(yc_pix - hei_pix / 2, dtype=np.int)
ymax_img = np.array(yc_pix + hei_pix / 2, dtype=np.int)

all_filez = np.unique(filenamez_in)

image_output_loc = "C:/Users/kryzi/OneDrive - University of St Andrews/output_image_test/"

# for fl in range(len(all_filez)):
for fl in range(50):
    file_mask = filenamez_in == all_filez[fl]
    file_path = all_filez[fl]
    print(file_path)
    print(boxes_found[file_mask])
    img_in = cv2.imread(file_path)
    xmin_file = xmin_img[file_mask]
    xmax_file = xmax_img[file_mask]
    ymin_file = ymin_img[file_mask]
    ymax_file = ymax_img[file_mask]
    conf_file = boxes_found.conf[file_mask]
    xmin_file, xmax_file, ymin_file, ymax_file = basic_nms(xmin_file, xmax_file, ymin_file, ymax_file, conf_file, 0.1)
    for bx in range(len(xmin_file)):
        cv2.rectangle(img_in, (xmin_file[bx], ymin_file[bx]), (xmax_file[bx], ymax_file[bx]), (0, 255, 0), 3)
    cv2.imshow('image', img_in)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


