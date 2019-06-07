import cv2
import os
import numpy as np
import pandas as pd


def create_tiled_data(filez, fl, base_dir, data_list, out_path, windowz, image_shape, final_size, resize=False):
    file_name = filez[fl]
    # get root of file name
    file_out = file_name[:-4]
    print(file_out)
    # get list of detections in this image
    keep_list = data_list.file_loc == file_name
    file_boxes = data_list[keep_list]
    # get min and max positions for boxes
    file_boxes['wid_half'] = np.divide(file_boxes.wid, 2)
    file_boxes['hei_half'] = np.divide(file_boxes.height, 2)
    file_boxes['xmin'] = np.subtract(file_boxes.xc, file_boxes.wid_half)
    file_boxes['xmax'] = np.add(file_boxes.xc, file_boxes.wid_half)
    file_boxes['ymin'] = np.subtract(file_boxes.yc, file_boxes.hei_half)
    file_boxes['ymax'] = np.add(file_boxes.yc, file_boxes.hei_half)
    # read in image
    from_loc = base_dir + file_name
    image_in = cv2.imread(from_loc, -1)
    # need windows as both pixels and percentage windowz is in pixels so convert to percentage of image
    wind_pct = np.array(windowz, dtype=np.float)
    wind_pct[:, 0] = np.divide(wind_pct[:, 0], image_shape[0])
    wind_pct[:, 1] = np.divide(wind_pct[:, 1], image_shape[1])
    wind_pct[:, 2] = np.divide(wind_pct[:, 2], image_shape[0])
    wind_pct[:, 3] = np.divide(wind_pct[:, 3], image_shape[1])
    # for each window
    for wnd in range(wind_pct.shape[0]):
        # set output for text file
        out_string = ""
        # set shortnames for window position
        xmin = wind_pct[wnd, 1]
        xmax = wind_pct[wnd, 3]
        ymin = wind_pct[wnd, 0]
        ymax = wind_pct[wnd, 2]
        # min pixels for width or height in ground truth is 10 pixels
        # pick 5 pixels as a minimum size of box to keep in case detection is cut in half by tiling
        min_size_x = 5.0 / 7360
        min_size_y = 5.0 / 4912
        # for each detection
        for ln in range(file_boxes.shape[0]):
            line = file_boxes.iloc[ln]
            # check if detection is in window
            if line.xmax >= xmin and line.xmin < xmax and line.ymax >= ymin and line.ymin < ymax:
                # find new position of bbox
                line.xmax = np.minimum(line.xmax, xmax)
                line.xmin = np.maximum(line.xmin, xmin)
                line.ymax = np.minimum(line.ymax, ymax)
                line.ymin = np.maximum(line.ymin, ymin)
                # get new width and height
                line.wid = line.xmax - line.xmin
                # if this makes the box too thin skip to next detection
                if line.wid < min_size_x:
                    continue
                line.height = line.ymax - line.ymin
                # if this makes the box too short skip to next detection
                if line.height < min_size_y:
                    continue
                line.wid_half = np.divide(line.wid, 2)
                line.hei_half = np.divide(line.height, 2)
                line.xc = np.add(line.xmin, line.wid_half)
                line.yc = np.add(line.ymin, line.hei_half)
                # convert position in  image to position in window
                line.xc = (line.xc - xmin) / (xmax - xmin)
                line.yc = (line.yc - ymin) / (ymax - ymin)
                line.wid = line.wid / (xmax - xmin)
                line.height = line.height / (ymax - ymin)
                # output position in window
                out_string = out_string + str(line.oc) + ' ' + str(line.xc) + ' ' + str(line.yc) + ' '
                out_string = out_string + str(line.wid) + ' ' + str(line.height) + '\n'
        if len(out_string) > 0:
            # get rid of final line separator
            out_string = out_string[:-1]
        else:
            out_string = "NA NA NA NA NA"
        # if filename contains a directory split out
        split_filename = file_out.split("/")
        if len(split_filename) > 1:
            file_out = split_filename[0] + '_'
            for splt in range(1, len(split_filename)):
                file_out = file_out + split_filename[splt] + '_'
            # remove uneccesary extra underscore
            file_out = file_out[:-1]
        # get image file name
        out_img_name = file_out + '_' + str(wnd) + '.png'
        print(out_img_name)
        # get just this window from image and write it out
        image_out = image_in[windowz[wnd, 0]:windowz[wnd, 2], windowz[wnd, 1]:windowz[wnd, 3]]
        if resize:
            image_out = cv2.resize(image_out, final_size)
        print(out_path + out_img_name)
        cv2.imwrite(out_path + out_img_name, image_out)
        # create text file name and write output to text file
        txt_out = file_out + '_' + str(wnd) + '.txt'
        txt_path = out_path + txt_out
        with open(txt_path, "w") as text_file:
            text_file.write(out_string)


# GFRC input
# Directory with files
basedir = 'E:/'
# List of boxes
csv_name = 'CF_Calcs/BenchmarkSets/GFRC/yolo_test_GFRC_bboxes.csv'
# Full path csv
csv_file = basedir + csv_name
# folder to save in
out_folder = 'CF_Calcs/BenchmarkSets/GFRC/yolo_copy_valid_img/'
outpath = basedir + out_folder
# Read in csv file
datalist = pd.read_csv(csv_file)

# Get list of unique filenames
filez_in = np.unique(datalist.file_loc)

# for debugging purposes
# filez = filez[0:20]

# windowz to cut out - start rows, start cols, end rows, end cols

input_shape = [4912, 7360]
size_out = [192, 288]

row_st = 0
row_ed = size_out[0]
gfrcrowst = []
gfrcrowed = []
while row_ed < input_shape[0]:
    gfrcrowst.append(row_st)
    gfrcrowed.append(row_ed)
    row_st = row_st + size_out[0]
    row_ed = row_ed + size_out[0]
row_ed = input_shape[0]
row_st = row_ed - size_out[0]
gfrcrowst.append(row_st)
gfrcrowed.append(row_ed)
col_st = 0
col_ed = size_out[1]
gfrccolst = []
gfrccoled = []
while col_ed < input_shape[1]:
    gfrccolst.append(col_st)
    gfrccoled.append(col_ed)
    col_st = col_st + size_out[1]
    col_ed = col_ed + size_out[1]
col_ed = input_shape[1]
col_st = col_ed - size_out[1]
gfrccolst.append(col_st)
gfrccoled.append(col_ed)
print(gfrccolst)
nrow = len(gfrcrowst)
ncol = len(gfrccolst)
gfrcrowst = np.reshape(np.tile(gfrcrowst, ncol), (nrow * ncol, 1))
gfrcrowed = np.reshape(np.tile(gfrcrowed, ncol), (nrow * ncol, 1))
gfrccolst = np.reshape(np.repeat(gfrccolst, nrow), (nrow * ncol, 1))
gfrccoled = np.reshape(np.repeat(gfrccoled, nrow), (nrow * ncol, 1))
gfrcwindz = np.hstack((gfrcrowst, gfrccolst, gfrcrowed, gfrccoled))
gfrcwindz = np.array(gfrcwindz, dtype=np.int)

gfrcwindz = np.array(gfrcwindz)
gfrcsize = [4912, 7360]
resize = True
finsize = size_out
if resize:
    finsize = (size_out[1]*2, size_out[0]*2)
print(gfrcwindz)

# copy image to folder and create text file to go with it
for ff in range(filez_in.shape[0]):
    create_tiled_data(filez_in, ff, basedir, datalist, outpath, gfrcwindz, gfrcsize, finsize, resize=resize)

# create train.txt list of images
out_folder_list = os.listdir(outpath)
train_string = "img_name,gt_details\n"
for ff in range(len(out_folder_list)):
    filename = out_folder_list[ff]
    if filename[-4:] == ".png":
        train_string = train_string + filename + ','
    if filename[-4:] == ".txt":
        train_string = train_string + filename + '\n'
# remove unecesary last line separator
train_string = train_string[:-1]
# write out file
train_txt_path = outpath + "gfrc_test.txt"
with open(train_txt_path, "w") as textfile:
    textfile.write(train_string)
