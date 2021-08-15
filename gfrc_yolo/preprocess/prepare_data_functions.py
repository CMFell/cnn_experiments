import cv2
import numpy as np
import pandas as pd
import random


def create_windows(size_out, input_shape, overlap):
    # overlap is number of images that overlap so 2 will give 50% overlap of images
    row_st = 0
    row_ed = size_out[0]
    gfrcrowst = []
    gfrcrowed = []
    while row_ed < input_shape[0]:
        gfrcrowst.append(row_st)
        gfrcrowed.append(row_ed)
        row_st = int(row_st + size_out[0] / overlap)
        row_ed = int(row_st + size_out[0])
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
        col_st = int(col_st + size_out[1] / overlap)
        col_ed = int(col_st + size_out[1])
    col_ed = input_shape[1]
    col_st = col_ed - size_out[1]
    gfrccolst.append(col_st)
    gfrccoled.append(col_ed)
    nrow = len(gfrcrowst)
    ncol = len(gfrccolst)
    gfrcrowst = np.reshape(np.tile(gfrcrowst, ncol), (nrow * ncol, 1))
    gfrcrowed = np.reshape(np.tile(gfrcrowed, ncol), (nrow * ncol, 1))
    gfrccolst = np.reshape(np.repeat(gfrccolst, nrow), (nrow * ncol, 1))
    gfrccoled = np.reshape(np.repeat(gfrccoled, nrow), (nrow * ncol, 1))
    gfrcwindz = np.hstack((gfrcrowst, gfrccolst, gfrcrowed, gfrccoled))
    gfrcwindz = np.array(gfrcwindz, dtype=np.int)
    return gfrcwindz


def get_detect_in_wind(file_boxes, dict_in):
    xmin = dict_in["xmin"]
    xmax = dict_in["xmax"]
    ymin = dict_in["ymin"]
    ymax = dict_in["ymax"]
    min_size_x = dict_in["min_size_x"]
    min_size_y = dict_in["min_size_y"]
    min_pct = dict_in["min_pct"]
    out_string = ""
    out_list = []
    n_out = 0
    for ln in range(file_boxes.shape[0]):
        line = file_boxes.iloc[ln]
        # check if detection is in window
        if line.xmax >= xmin and line.xmin < xmax and line.ymax >= ymin and line.ymin < ymax:
            # get original size of box
            orig_wid = line.xmax - line.xmin
            orig_hei = line.ymax - line.ymin
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
            if line.wid < (orig_wid / min_pct):
                continue
            line.height = line.ymax - line.ymin
            # if this makes the box too short skip to next detection
            if line.height < min_size_y:
                continue
            if line.height < (orig_hei / min_pct):
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
            line_out = [line.oc, line.xc, line.yc, line.wid, line.height]
            out_list.extend(line_out)
            n_out += 1
            # output position in window
            out_string = out_string + str(line.oc) + ' ' + str(line.xc) + ' ' + str(line.yc) + ' '
            out_string = out_string + str(line.wid) + ' ' + str(line.height) + '\n'
    out_array = np.array(out_list)
    out_array = np.reshape(out_array, (n_out, 5))
    return out_string, out_array


def get_tile_image(image_in, options):
    xmin = options["xmin"]
    xmax = options["xmax"]
    ymin = options["ymin"]
    ymax = options["ymax"]
    # get just this window from image and write it out
    image_out = image_in[ymin:ymax, xmin:xmax]
    fsize = options["final_size"]
    osize = options["size_out"]
    resize = fsize[0] == osize[0] and fsize[1] == osize[1]
    if resize:
        final_size = options["final_size"]
        image_out = cv2.resize(image_out, final_size)
    return image_out


def get_tile_name(wnd, options):
    file_out = options["file_out"]
    out_path = options["out_path"]
    # if filename contains a directory split out
    split_filename = file_out.split("/")
    if len(split_filename) > 1:
        file_out = split_filename[0] + '_'
        for splt in range(1, len(split_filename)):
            file_out = file_out + split_filename[splt] + '_'
        # remove uneccesary extra underscore
        file_out = file_out[:-1]
    # create text file name and write output to text file
    out_name = file_out + '_' + str(wnd)
    # get just this window from image and write it out
    out_name = out_path + out_name
    return out_name


def write_txt_file(out_string, out_name):
    # get rid of final line separator
    out_string = out_string[:-1]
    txt_path = out_name + '.txt'
    with open(txt_path, "w") as text_file:
        text_file.write(out_string)


def blank_txt_file(out_name):
    txt_path = out_name + '.txt'
    with open(txt_path, "w") as text_file:
        text_file.write("")


def create_tiled_data(filez, fl, windowz, options, keep_blanks=False, pblank=1.0):
    base_dir = options["base_dir"]
    data_list = options["data_list"]
    out_path = options["out_path"]
    image_shape = options["image_shape"]
    size_out = options["size_out"]
    cols = size_out[1]
    rows = size_out[0]

    final_size = options["final_size"]
    cols = final_size[1]
    rows = final_size[0]

    file_name = filez[fl]
    print(file_name)
    # get root of file name
    file_out = file_name[:-4]
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
        aug_no = 0
        # set shortnames for window position in pct
        xminh = wind_pct[wnd, 1]
        xmaxh = wind_pct[wnd, 3]
        yminh = wind_pct[wnd, 0]
        ymaxh = wind_pct[wnd, 2]
        # set shortnames for window position in pixels
        xminh_px = windowz[wnd, 1]
        xmaxh_px = windowz[wnd, 3]
        yminh_px = windowz[wnd, 0]
        ymaxh_px = windowz[wnd, 2]
        # min pixels for width or height in ground truth is 10 pixels
        # min size to keep wil depend on size and shape of box.
        # going to assume you need at least a third of the box in either direction
        # pick 5 pixels as a minimum size of box to keep in case detection is cut in half by tiling
        min_size_x = 5.0 / image_shape[1]
        min_size_y = 5.0 / image_shape[0]
        min_pct = 4. / 3.
        # for each detection
        # set output for text file
        dict_horiz_pct = {"xmin": xminh, "xmax": xmaxh, "ymin": yminh, "ymax": ymaxh,
                          "min_size_x": min_size_x, "min_size_y": min_size_y, "min_pct": min_pct}
        dict_horiz_pix = {"xmin": xminh_px, "xmax": xmaxh_px, "ymin": yminh_px, "ymax": ymaxh_px,
                          "min_size_x": min_size_x, "min_size_y": min_size_y, "min_pct": min_pct,
                          "final_size": final_size}
        dict_out_name = {"file_out": file_out, "out_path": out_path, "aug_no": aug_no}

        out_string_horiz, out_array_horiz = get_detect_in_wind(file_boxes, dict_horiz_pct)

        image_horiz = get_tile_image(image_in, dict_horiz_pix, resize=resize)

        if len(out_string_horiz) > 0:
            tile_name = get_tile_name(wnd, dict_out_name)
            cv2.imwrite(tile_name + '.png', image_horiz)
            write_txt_file(out_string_horiz, tile_name)

        if keep_blanks:
            keep = random.random() < pblank
            if len(out_string_horiz) == 0:
                if keep:
                    tile_name = get_tile_name(wnd, dict_out_name)
                    cv2.imwrite(tile_name + '.png', image_horiz)
                    blank_txt_file(tile_name)


def create_tiled_data_blanks(filez, fl, windowz, options, pblank=1.0):
    base_dir = options["base_dir"]
    out_path = options["out_path"]
    image_shape = options["image_shape"]
    size_out = options["size_out"]
    cols = size_out[1]
    rows = size_out[0]

    final_size = options["final_size"]
    cols = final_size[1]
    rows = final_size[0]

    file_name = filez[fl]
    print(file_name)
    # get root of file name
    file_out = file_name[:-4]
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
        aug_no = 0
        # set shortnames for window position in pct
        xminh = wind_pct[wnd, 1]
        xmaxh = wind_pct[wnd, 3]
        yminh = wind_pct[wnd, 0]
        ymaxh = wind_pct[wnd, 2]
        # set shortnames for window position in pixels
        xminh_px = windowz[wnd, 1]
        xmaxh_px = windowz[wnd, 3]
        yminh_px = windowz[wnd, 0]
        ymaxh_px = windowz[wnd, 2]

        dict_horiz_pix = {"xmin": xminh_px, "xmax": xmaxh_px, "ymin": yminh_px, "ymax": ymaxh_px,
                    "final_size": final_size}

        image_horiz = get_tile_image(image_in, dict_horiz_pix)

        keep = random.random() < pblank
        if keep:
            tile_name = get_tile_name(wnd, dict_out_name)
            cv2.imwrite(tile_name + '.png', image_horiz)
            blank_txt_file(tile_name)

