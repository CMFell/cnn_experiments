import numpy as np
import pandas as pd


def write_txt_file(out_string, out_name):
    # get rid of final line separator
    out_string = out_string[:-1]
    with open(out_name, "w") as text_file:
        text_file.write(out_string)


def get_image_data(filez, fl, data_list, out_path):
    file_name = filez[fl]
    print(file_name)
    # get root of file name
    file_out = file_name[:-4]
    # get list of detections in this image
    keep_list = data_list.file_loc == file_name
    file_boxes = data_list[keep_list]
    out_string_horiz, out_array_horiz = get_detect_in_img(file_boxes)
    out_name = get_file_name_out(file_out)
    csv_out = out_path + out_name + ".csv"
    write_txt_file(out_string_horiz, csv_out)


def get_detect_in_img(file_boxes):
    out_string = ""
    out_list = []
    n_out = 0
    for ln in range(file_boxes.shape[0]):
        line = file_boxes.iloc[ln]
        line_out = [line.oc, line.xc, line.yc, line.wid, line.height]
        out_list.extend(line_out)
        n_out += 1
        # output position
        out_string = out_string + str(line.oc) + ' ' + str(line.xc) + ' ' + str(line.yc) + ' '
        out_string = out_string + str(line.wid) + ' ' + str(line.height) + '\n'
    out_array = np.array(out_list)
    out_array = np.reshape(out_array, (n_out, 5))
    return out_string, out_array


def get_file_name_out(file_out):
    # if filename contains a directory split out
    split_filename = file_out.split("/")
    if len(split_filename) > 1:
        file_out = split_filename[0] + '_'
        for splt in range(1, len(split_filename)):
            file_out = file_out + split_filename[splt] + '_'
        # remove uneccesary extra underscore
        file_out = file_out[:-1]
    return file_out


# GFRC input
# Directory with files
basedir = 'E:/'
# type of dataset
typed = 'valid'
# List of boxes
csv_name = 'CF_Calcs/BenchmarkSets/GFRC/yolo_' + typed + '_GFRC_bboxes.csv'
# folder to save in
out_folder = 'C:/Users/christina/OneDrive - University of St Andrews/PhD/valid_files/'

# Full path csv
csv_file = basedir + csv_name
# Read in csv file
datalist = pd.read_csv(csv_file)
# Get list of unique filenames
filez_in = np.unique(datalist.file_loc)

for ff in range(filez_in.shape[0]):
    get_image_data(filez_in, ff, datalist, out_folder)
