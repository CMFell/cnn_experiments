import numpy as np
import pandas as pd


def int_union(box1, box2):
    # gt_box is in order xmin, ymin, xmax, ymax
    # compute overlap
    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    intersect = (w * h)

    # compute box sizes
    bbw = box1[2] - box1[0]
    bbh = box1[3] - box1[1]
    sww = box2[2] - box2[0]
    swh = box2[3] - box2[1]
    gt_area = bbw * bbh
    sw_area = sww * swh
    union = gt_area + sw_area - intersect

    # compute iou ratio
    iou = intersect / union

    return iou


def select_anchor(gt_line, anchors):
    # calculates which anchor box size is closest to the ground truths
    ab_xmin = np.multiply(np.divide(anchors[:, 0], 2), -1).values.reshape(anchors.shape[0], 1)
    ab_xmax = np.divide(anchors[:, 0], 2).values.reshape(anchors.shape[0], 1)
    ab_ymin = np.multiply(np.divide(anchors[:, 1], 2), -1).values.reshape(anchors.shape[0], 1)
    ab_ymax = np.divide(anchors[:, 1], 2).values.reshape(anchors.shape[0], 1)
    ab_all = np.hstack((ab_xmin, ab_ymin, ab_xmax, ab_ymax))
    gt_xmin = np.multiply(np.divide(gt_line.wid, 2), -1)
    gt_xmax = np.divide(gt_line.wid, 2)
    gt_ymin = np.multiply(np.divide(gt_line.height, 2), -1)
    gt_ymax = np.divide(gt_line.height, 2)
    gt = np.hstack((gt_xmin, gt_ymin, gt_xmax, gt_ymax))

    iouz = np.zeros(ab_all.shape[0])
    for ab in range(ab_all.shape[0]):
        ab_box = ab_all[ab, :]
        iouz[ab] = int_union(gt, ab_box)
    gt_iou = np.argmax(iouz)
    print(iouz)
    return gt_iou


def return_anchors(gt_img, dictdeets):

    # for ground truths assumes wid:height is given as between 0:1 for whole image
    anchors = dictdeets['anchors']
    boxsx = dictdeets['boxs_x']
    boxsy = dictdeets['boxs_y']

    # calculate gt width height on same scale as anchors
    gt_wid = np.multiply(gt_img.wid, boxsx).values.reshape(gt_img.shape[0], 1)
    gt_hei = np.multiply(gt_img.height, boxsy).values.reshape(gt_img.shape[0], 1)
    wh = np.hstack((gt_wid, gt_hei))

    # work out closest matching anchor box
    a_boxes = []
    for gt in range(gt_img.shape[0]):
        dists = np.sum(np.square(np.subtract(anchors, wh[gt, :])), axis=1)
        a_boxes.append(np.argmin(dists))

    a_boxes = np.array(a_boxes)

    dict_out = {'match_anchors': a_boxes}

    return dict_out


def get_new_coords(gt_img, dictdeets):

    # for ground truths assumes centre x,y is given as between 0:1 for whole image
    # calculates which cell the centre is in (cells),
    # where it is in that cell (centres),
    # and the size of the ground truth relative to the cell (sizes)

    boxsx = dictdeets['boxs_x']
    boxsy = dictdeets['boxs_y']
    nrow = gt_img.shape[0]

    # work out which box centre is in
    x_pos = np.divide(gt_img.xc, 1 / boxsx).values.reshape(nrow, 1)
    y_pos = np.divide(gt_img.yc, 1 / boxsy).values.reshape(nrow, 1)
    x_box = np.floor(x_pos)
    y_box = np.floor(y_pos)
    cells = np.array(np.hstack((x_box, y_box)), dtype=np.int)
    centre_img = np.hstack((gt_img.xc.values.reshape(nrow, 1), gt_img.yc.values.reshape(nrow, 1)))

    # work out centre and width height relative to box
    x_cent = x_pos - x_box
    y_cent = y_pos - y_box
    centres = np.hstack((x_cent, y_cent))
    w_out = np.multiply(gt_img.wid, boxsx).values.reshape(nrow, 1)
    h_out = np.multiply(gt_img.height, boxsy).values.reshape(nrow, 1)
    sizes = np.hstack((w_out, h_out))
    size_img = np.hstack((gt_img.wid.values.reshape(nrow, 1), gt_img.height.values.reshape(nrow, 1)))

    dict_out = {'centres': centres, 'sizes': sizes, 'cells': cells, 'centre_img': centre_img, 'size_img': size_img}

    return dict_out


def get_pixel_loc_area(gt_img, dictdeets):

    # for ground truths assumes centre x,y is given as between 0:1 for whole image
    # gets top left, bottom right and area of ground truth in pixels
    # for calculating IOU

    imgxpix = dictdeets['img_x_pix']
    imgypix = dictdeets['img_y_pix']
    nrow = gt_img.shape[0]

    wid_pxl = np.multiply(gt_img.wid, imgxpix).values.reshape(nrow, 1)
    hei_pxl = np.multiply(gt_img.height, imgypix).values.reshape(nrow, 1)
    areas = np.multiply(wid_pxl, hei_pxl)
    pxl_sizes = np.array(np.hstack((wid_pxl, hei_pxl, areas)), dtype=np.int)
    gt_xc = gt_img.xc.values.reshape(nrow, 1)
    gt_yc = gt_img.yc.values.reshape(nrow, 1)
    tl_x = np.subtract(np.multiply(gt_xc, imgxpix), np.divide(wid_pxl, 2))
    tl_y = np.subtract(np.multiply(gt_yc, imgypix), np.divide(hei_pxl, 2))
    tl = np.array(np.hstack((tl_x, tl_y)), dtype=np.int)
    br_x = np.add(np.multiply(gt_xc, imgxpix), np.divide(wid_pxl, 2))
    br_y = np.add(np.multiply(gt_yc, imgypix), np.divide(hei_pxl, 2))
    br = np.array(np.hstack((br_x, br_y)), dtype=np.int)

    dict_out = {'pixel_size': pxl_sizes, 'top_left': tl, 'bottom_right': br}

    return dict_out


def get_class_one_hot(gt_img, dictdeets):

    # output classes as one hot vectors for ground truths
    n_classes = dictdeets['n_classes']

    classes_oh = np.zeros((gt_img.shape[0], n_classes), dtype=np.int)
    print(classes_oh.shape)
    for gt in range(gt_img.shape[0]):
        classes_oh[gt, gt_img.oc.iloc[gt]] = 1

    dict_out = {'one_hot_class': classes_oh}

    return dict_out


def create_train_gt(gt_img, dictdeets):
    # create blank dictionary for ground truth details
    gt_dict = {}
    # add best matching anchor boxes for each ground truth to dictionary
    gt_dict.update(return_anchors(gt_img, dictdeets))
    # add coordinates in grid system for each ground truth to dictionary
    gt_dict.update(get_new_coords(gt_img, dictdeets))
    # add coordinates in pixels for each ground truth to dictionary
    gt_dict.update(get_pixel_loc_area(gt_img, dictdeets))
    # add one hot classes for ground truths to dictionary
    gt_dict.update(get_class_one_hot(gt_img, dictdeets))

    boxsx = int(dictdeets['boxs_x'])
    boxsy = int(dictdeets['boxs_y'])
    n_classes = dictdeets['n_classes']
    anchors = dictdeets['anchors']
    nanchors = anchors.shape[0]
    num_out = 5 + n_classes

    # create empty ground truth array for image
    gt_out = np.zeros((boxsy, boxsx, nanchors, num_out))

    # for each ground truth
    for gt in range(gt_img.shape[0]):
        x_cell = gt_dict['cells'][gt, 0]
        y_cell = gt_dict['cells'][gt, 1]
        anc = gt_dict['match_anchors'][gt]
        classes = gt_dict['one_hot_class'][gt, :]
        centre = gt_dict['centres'][gt, :]
        size_gt = gt_dict['sizes'][gt, :]
        out_vec = np.concatenate((centre, size_gt, [1], classes))
        gt_out[y_cell, x_cell, anc, :] = out_vec
        print(y_cell, x_cell, anc)

    gt_dict['gt_array'] = gt_out

    return gt_dict


def filter_gt_tile(gt_img, inputloc):
    filter_gt = pd.DataFrame(columns=['File_name', 'xc', 'yc', 'wid', 'height', 'oc'])
    new_row = 0
    for rw in range(gt_img.shape[0]):
        rw_fn = gt_img.File_name.iloc[rw]
        rw_xc = gt_img.xc.iloc[rw]
        rw_yc = gt_img.yc.iloc[rw]
        rw_wid = gt_img.wid.iloc[rw]
        rw_hei = gt_img.height.iloc[rw]
        rw_oc = gt_img.oc.iloc[rw]
        if inputloc[1] > rw_xc >= inputloc[0]:
            if inputloc[3] > rw_yc >= inputloc[2]:
                # get new x centre position and rescale to new image size
                rw_xc = (rw_xc - inputloc[0]) / (inputloc[1] - inputloc[0])
                # get new y centre position and rescale to new image size
                rw_yc = (rw_yc - inputloc[2]) / (inputloc[3] - inputloc[2])
                # change width and height to new size
                # actually need to think about more carefully how it deals with cropping at edges but leave for now
                rw_wid = rw_wid / (inputloc[1] - inputloc[0])
                rw_hei = rw_hei / (inputloc[3] - inputloc[2])
                row_out = [rw_fn, rw_xc, rw_yc, rw_wid, rw_hei, rw_oc]
                filter_gt.loc[new_row] = row_out
                new_row += 1

    return filter_gt
