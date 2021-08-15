import numpy as np
import pandas as pd
from scipy.special import expit
from trch_accuracy import calc_iou_centwh



def yolo_output_to_box_test(y_pred, conf_threshold, anchors):    
    n_bat = y_pred.shape[0]
    # n_bat = int(dict_in['batch_size'])
    boxsx = y_pred.shape[2]
    # boxsx = int(dict_in['boxs_x'])
    boxsy = y_pred.shape[1]
    # boxsy = int(dict_in['boxs_y'])
    # anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    num_out = int(y_pred.shape[3] / nanchors)
    n_classes = num_out - 5
    # n_classes = int(dict_in['n_classes'])
    # num_out = 5 + n_classes
    # thresh = dict_in['threshold']
    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]
    # get top left position of cells
    rowz = np.arange(boxsy)
    colz = np.arange(boxsx)
    # rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    rowno = np.expand_dims(np.expand_dims(np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx)), axis=0), axis=3)
    # colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.expand_dims(np.expand_dims(np.reshape(np.tile(colz, boxsy), (boxsy, boxsx)), axis=0), axis=3)
    tl_cell = np.stack((colno, rowno), axis=4)
    # restructure net_output
    y_pred = np.reshape(y_pred, size1)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = expit(np.reshape(y_pred[:, :, :, :, 4], size2))
    cent_cnn = expit(y_pred[:, :, :, :, 0:2])
    # cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    cent_cnn = np.add(cent_cnn, tl_cell)
    # divide so position is relative to whole image
    cent_cnn = np.divide(cent_cnn, size3)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # size is to power of prediction
    size_cnn = np.exp(size_cnn)
    # keep for loss
    # size_cnn_in = size_cnn
    # adjust so size is relative to anchors
    size_cnn = np.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = np.divide(size_cnn, size3)
    class_cnn = expit(y_pred[:, :, :, :, 5:])
    
    for img in range(n_bat):
        box_img = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'conf', 'class'])
        for yc in range(boxsy):
            for xc in range(boxsx):
                for ab in range(nanchors):
                    if confs_cnn[img, yc, xc, ab] > conf_threshold:
                        # scores_out.append(confs_cnn[img, yc, xc, ab])
                        class_out = np.argmax(class_cnn[img, yc, xc, ab, :])
                        # classes_out.append(class_out)
                        detect_deets = pd.DataFrame(
                            [[
                                cent_cnn[img, yc, xc, ab, 0],
                                cent_cnn[img, yc, xc, ab, 1],
                                size_cnn[img, yc, xc, ab, 0],
                                size_cnn[img, yc, xc, ab, 1],
                                confs_cnn[img, yc, xc, ab],
                                class_out
                            ]],
                            columns=['xc', 'yc', 'wid', 'hei', 'conf', 'class']
                        )
                        box_img = pd.concat((box_img, detect_deets), axis=0, sort=False)
    
    return box_img


def simple_nms(boxes_in, thresh):

    # conf_ord = np.argsort(boxes_in.conf)
    boxes_in = boxes_in.sort_values(by=['conf'], ascending=False)

    xmins = boxes_in.xc - boxes_in.wid / 2
    xmaxs = boxes_in.xc + boxes_in.wid / 2
    ymins = boxes_in.yc - boxes_in.hei / 2
    ymaxs = boxes_in.yc + boxes_in.hei / 2
    flns = boxes_in.file
    cnfs = boxes_in.conf
    clzz = boxes_in['class']
    tpz = boxes_in['tp']

    boxes_ot = pd.DataFrame(columns=["xc", "yc", "wid", "hei", "file", "conf", "class", "tp"])

    xmins = np.array(xmins.tolist())
    xmaxs = np.array(xmaxs.tolist())
    ymins = np.array(ymins.tolist())
    ymaxs = np.array(ymaxs.tolist())
    flns = np.array(flns.tolist())
    cnfs = np.array(cnfs.tolist())
    clzz = np.array(clzz.tolist())
    tpz = np.array(tpz.tolist())

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
            box_ot = pd.DataFrame(index=[1], columns=["xc", "yc", "wid", "hei", "file", "conf", "class", "tp"])
            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]

            xmns = np.append(xmns, xmn)
            xmxs = np.append(xmxs, xmx)
            ymxs = np.append(ymxs, ymx)
            ymns = np.append(ymns, ymn)

            box_ot.xc.iloc[0] = (np.min(xmns) + np.max(xmxs)) / 2
            box_ot.yc.iloc[0] = (np.min(ymns) + np.max(ymxs)) / 2
            box_ot.wid.iloc[0] = np.max(xmxs) - np.min(xmns)
            box_ot.hei.iloc[0] = np.max(ymxs) - np.min(ymns)
            box_ot.file.iloc[0] = flns[0]
            box_ot.conf.iloc[0] = cnfs[0]
            box_ot['class'].iloc[0] = clzz[0]
            box_ot['tp'].iloc[0] = tpz[0]

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out[1:] = mask_bxs
            mask_out = np.logical_not(mask_out)

            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            flns = flns[mask_out]
            cnfs = cnfs[mask_out]
            clzz = clzz[mask_out]
            tpz = tpz[mask_out]
        else:
            box_ot = pd.DataFrame(index=[1], columns=["xc", "yc", "wid", "hei", "file", "conf", "class", "tp"])
            box_ot.xc.iloc[0] = (xmn + xmx) / 2
            box_ot.yc.iloc[0] = (ymn + ymx) / 2
            box_ot.wid.iloc[0] = xmx - xmn
            box_ot.hei.iloc[0] = ymx - ymn
            box_ot.file.iloc[0] = flns[0]
            box_ot.conf.iloc[0] = cnfs[0]
            box_ot['class'].iloc[0] = clzz[0]
            box_ot['tp'].iloc[0] = tpz[0]

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out = np.logical_not(mask_out)
            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            flns = flns[mask_out]
            cnfs = cnfs[mask_out]
            clzz = clzz[mask_out]
            tpz = tpz[mask_out]
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0)

    return boxes_ot


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def yolo_output_to_box(y_pred, namez, anchors, truthz, nms_threshold, conf_threshold):
    # compares output from cnn with ground truth to calculate loss
    # only for one image at the moment

    n_bat = y_pred.shape[0]
    # n_bat = int(dict_in['batch_size'])
    boxsx = y_pred.shape[2]
    # boxsx = int(dict_in['boxs_x'])
    boxsy = y_pred.shape[1]
    # boxsy = int(dict_in['boxs_y'])
    # anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    num_out = int(y_pred.shape[3] / nanchors)
    n_classes = num_out - 5
    # n_classes = int(dict_in['n_classes'])
    # num_out = 5 + n_classes
    # thresh = dict_in['threshold']
    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]
    # get top left position of cells
    rowz = np.arange(boxsy)
    colz = np.arange(boxsx)
    # rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    rowno = np.expand_dims(np.expand_dims(np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx)), axis=0), axis=3)
    # colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.expand_dims(np.expand_dims(np.reshape(np.tile(colz, boxsy), (boxsy, boxsx)), axis=0), axis=3)
    tl_cell = np.stack((colno, rowno), axis=4)
    # restructure net_output
    y_pred = np.reshape(y_pred, size1)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = expit(np.reshape(y_pred[:, :, :, :, 4], size2))
    cent_cnn = expit(y_pred[:, :, :, :, 0:2])
    # cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    cent_cnn = np.add(cent_cnn, tl_cell)
    # divide so position is relative to whole image
    cent_cnn = np.divide(cent_cnn, size3)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # size is to power of prediction
    size_cnn = np.exp(size_cnn)
    # keep for loss
    # size_cnn_in = size_cnn
    # adjust so size is relative to anchors
    size_cnn = np.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = np.divide(size_cnn, size3)
    class_cnn = expit(y_pred[:, :, :, :, 5:])

    boxes_out = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class','tp'])
    scores_out = []
    classes_out = []
    iii = 0
    for img in range(n_bat):
        filen = namez[img]
        box_img = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class','tp'])
        for yc in range(boxsy):
            for xc in range(boxsx):
                for ab in range(nanchors):
                    if confs_cnn[img, yc, xc, ab] > conf_threshold:
                        iii += 1
                        # scores_out.append(confs_cnn[img, yc, xc, ab])
                        class_out = np.argmax(class_cnn[img, yc, xc, ab, :])
                        # classes_out.append(class_out)
                        detect_deets = pd.DataFrame(
                            [[
                                cent_cnn[img, yc, xc, ab, 0],
                                cent_cnn[img, yc, xc, ab, 1],
                                size_cnn[img, yc, xc, ab, 0],
                                size_cnn[img, yc, xc, ab, 1],
                                filen,
                                confs_cnn[img, yc, xc, ab],
                                class_out,
                                0
                            ]],
                            columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class', 'tp']
                        )
                        # calc iou with truths
                        iou = 0
                        for tr in range(truthz.shape[0]):
                            iou_new = calc_iou_centwh(detect_deets.iloc[0], truthz.iloc[tr])
                            if iou_new > iou:
                                # box_img['tp'].iloc[len(box_img) - 1] = 1
                                # box_img['tp'].iloc[len(box_img) - 1] = iou_new
                                detect_deets['tp'][0] = iou_new
                                iou = iou_new
                        box_img = box_img.append(detect_deets)
                        # tp = torch.sum(poz & truez)
        if box_img.shape[0] > 0:
            box_img_ot = simple_nms(box_img, nms_threshold)
            boxes_out = pd.concat((boxes_out, box_img_ot), axis=0)

    output = [boxes_out, scores_out, classes_out]

    return output



def yolo_output_to_box_vec(y_pred, namez, anchors, truthz, nms_threshold, conf_threshold):
    # compares output from cnn with ground truth to calculate loss
    # only for one image at the moment

    n_bat = y_pred.shape[0]

    boxsx = y_pred.shape[2]

    boxsy = y_pred.shape[1]

    nanchors = anchors.shape[0]
    num_out = int(y_pred.shape[3] / nanchors)
    n_classes = num_out - 5
    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]
    # get top left position of cells
    rowz = np.arange(boxsy)
    colz = np.arange(boxsx)
    # rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    rowno = np.expand_dims(np.expand_dims(np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx)), axis=0), axis=3)
    # colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.expand_dims(np.expand_dims(np.reshape(np.tile(colz, boxsy), (boxsy, boxsx)), axis=0), axis=3)
    tl_cell = np.stack((colno, rowno), axis=4)
    # restructure net_output
    y_pred = np.reshape(y_pred, size1)

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = expit(np.reshape(y_pred[:, :, :, :, 4], size2))
    cent_cnn = expit(y_pred[:, :, :, :, 0:2])
    # cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    cent_cnn = np.add(cent_cnn, tl_cell)
    # divide so position is relative to whole image
    cent_cnn = np.divide(cent_cnn, size3)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # size is to power of prediction
    size_cnn = np.exp(size_cnn)
    # keep for loss
    # size_cnn_in = size_cnn
    # adjust so size is relative to anchors
    size_cnn = np.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = np.divide(size_cnn, size3)
    class_cnn = expit(y_pred[:, :, :, :, 5:])

    confs_mask = np.greater(confs_cnn, conf_threshold)

    class_out = np.argmax(class_cnn, axis=4)
    cent_size = np.concatenate((cent_cnn, size_cnn), axis=4)
    cent_size = np.expand_dims(cent_size, axis=5)

    truthz_np = np.array(truthz)
    truthz_np = truthz_np[:, 1:5]
    truthz_np = np.transpose(truthz_np)
    truthz_np = np.expand_dims(truthz_np, axis=0)
    truthz_np = np.expand_dims(truthz_np, axis=0)
    truthz_np = np.expand_dims(truthz_np, axis=0)
    truthz_np = np.expand_dims(truthz_np, axis=0)

    # calc iou
    xmn1 = np.subtract(cent_size[:, :, :, :, 0, :], np.divide(cent_size[:, :, :, :, 2, :], 2))
    xmx1 = np.add(cent_size[:, :, :, :, 0, :], np.divide(cent_size[:, :, :, :, 2, :], 2))
    ymn1 = np.subtract(cent_size[:, :, :, :, 1, :], np.divide(cent_size[:, :, :, :, 3, :], 2))
    ymx1 = np.add(cent_size[:, :, :, :, 1, :], np.divide(cent_size[:, :, :, :, 3, :], 2))
    xmn2 = np.subtract(truthz_np[:, :, :, :, 0, :], np.divide(truthz_np[:, :, :, :, 2, :], 2))
    xmx2 = np.add(truthz_np[:, :, :, :, 0, :], np.divide(truthz_np[:, :, :, :, 2, :], 2))
    ymn2 = np.subtract(truthz_np[:, :, :, :, 1, :], np.divide(truthz_np[:, :, :, :, 3, :], 2))
    ymx2 = np.add(truthz_np[:, :, :, :, 1, :], np.divide(truthz_np[:, :, :, :, 3, :], 2))

    ol_xmn = np.fmax(xmn2, xmn1)
    ol_xmx = np.fmin(xmx2, xmx1)
    ol_ymn = np.fmax(ymn2, ymn1)
    ol_ymx = np.fmin(ymx2, ymx1)

    olx = np.fmax(np.subtract(ol_xmx, ol_xmn), 0)
    oly = np.fmax(np.subtract(ol_ymx, ol_ymn), 0)

    ol_area = np.multiply(olx, oly)
    bx1_area = np.multiply(cent_size[:, :, :, :, 2, :], cent_size[:, :, :, :, 3, :])
    bx2_area = np.multiply(truthz_np[:, :, :, :, 2, :], truthz_np[:, :, :, :, 3, :])

    iou = np.divide(ol_area, np.subtract(np.add(bx1_area, bx2_area), ol_area))

    # get max iou overall truths
    iou = np.nanmax(iou, axis=4)

    # set iou to zero where conf is less than thresh
    tp = np.where(confs_mask, iou, 0)

    # reshape arrays to 2d and convert to pandas
    shapez = cent_size.shape
    shape_out = shapez[0]*shapez[1]*shapez[2]*shapez[3]
    cent_size_2d = np.reshape(np.squeeze(cent_size, axis=-1), (shape_out, 4))
    conf_mask_2d = np.reshape(confs_mask, (shape_out, 1))
    conf_mask_2d = conf_mask_2d.flatten()
    cent_size_out = cent_size_2d[conf_mask_2d]
    cent_size_pd = pd.DataFrame(cent_size_out, columns=['xc', 'yc', 'wid', 'hei'])
    cnf_2d = np.reshape(confs_cnn, (shape_out, 1))
    cnf_out = cnf_2d[conf_mask_2d]
    cnf_pd = pd.DataFrame(cnf_out, columns=['conf'])
    cls_2d = np.reshape(class_out, (shape_out, 1))
    cls_out = cls_2d[conf_mask_2d]
    cls_pd = pd.DataFrame(cls_out, columns=['class'])
    tp_2d = np.reshape(tp, (shape_out, 1))
    tp_out = tp_2d[conf_mask_2d]
    tp_pd = pd.DataFrame(tp_out, columns=['tp'])
    filen_out = np.repeat(namez, shapez[1]*shapez[2]*shapez[3])
    filen_out = filen_out[conf_mask_2d]
    filen_pd = pd.DataFrame(filen_out, columns=['file'])
    filenoz = np.repeat(list(range(len(namez))), shapez[1]*shapez[2]*shapez[3])
    filenoz = filenoz[conf_mask_2d]
    filenoz = np.reshape(filenoz, (len(filenoz), 1))
    # vecorised nms per image

    # create final output
    box_img = pd.concat([cent_size_pd, filen_pd, cnf_pd, cls_pd, tp_pd], axis=1, ignore_index=True, sort=False)  
    box_img.columns = ['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class','tp']

    #boxes_out = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class','tp']) 

    #for im in range(len(namez)):
    #    box_im = box_img[box_img.file == namez[im]]   
    #    if box_img.shape[0] > 0:
    #        box_img_ot = simple_nms(box_img, nms_threshold)
    #        boxes_out = pd.concat((boxes_out, box_img_ot), axis=0)

    boxes_out = simple_nms_vec(cent_size_out, cnf_out, cls_out, tp_out, filen_out, filenoz, nms_threshold)

    return boxes_out


def simple_nms_vec(size_in, conf_in, class_in, tp_in, filen_in, filenoz_in, thresh):

    conf2 = np.hstack((conf_in, filenoz_in))
    img_sort_indices = np.argsort(conf2[:,1], kind='mergesort')
    conf2 = conf2[img_sort_indices, :]
    size_in = size_in[img_sort_indices, :]
    tp_in = tp_in[img_sort_indices, :]
    filen_in = filen_in[img_sort_indices]
    class_in = class_in[img_sort_indices]
    conf_sort_indices = np.argsort((-conf2[:, 0]), kind='mergesort')
    conf2 = conf2[conf_sort_indices, :]
    size_in = size_in[conf_sort_indices, :]
    tp_in = tp_in[conf_sort_indices, :]
    filen_in = filen_in[conf_sort_indices]
    class_in = class_in[conf_sort_indices]

    xmins = np.subtract(size_in[:, 0], np.divide(size_in[:, 2], 2))
    xmaxs = np.add(size_in[:, 0], np.divide(size_in[:, 2], 2))
    ymins = np.subtract(size_in[:, 1], np.divide(size_in[:, 3], 2))
    ymaxs = np.add(size_in[:, 1], np.divide(size_in[:, 3], 2))

    ol_xmn = np.fmax(xmins, np.transpose(xmins))
    ol_xmx = np.fmin(xmaxs, np.transpose(xmaxs))
    ol_ymn = np.fmax(ymins, np.transpose(ymins))
    ol_ymx = np.fmin(ymaxs, np.transpose(ymaxs))

    olx = np.fmax(np.subtract(ol_xmx, ol_xmn), 0)
    oly = np.fmax(np.subtract(ol_ymx, ol_ymn), 0)

    ol_area = np.multiply(olx, oly)
    bx_area = np.multiply(size_in[:, 2], size_in[:, 3])

    iou = np.divide(ol_area, np.subtract(np.add(bx_area, np.transpose(bx_area)), ol_area))

    iou_tri = np.tril(iou)
    iou_tri = np.where(iou_tri < thresh, 0, iou_tri)
    iou_mask = np.sum(iou_tri, axis=1)
    iou_mask = iou_mask > 0

    conf2 = conf2[iou_mask]
    size_pd = pd.DataFrame(size_in[iou_mask])
    tp_pd = pd.DataFrame(tp_in[iou_mask])
    filen_pd = pd.DataFrame(filen_in[iou_mask])
    conf2_pd = pd.DataFrame(conf2[:, 0:1])
    class_pd = pd.DataFrame(class_in[iou_mask])

    boxes_ot = pd.concat([size_pd, filen_pd, conf2_pd, class_pd, tp_pd], axis=1, ignore_index=True, sort=False)  
    boxes_ot.columns = ['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class','tp']

    return boxes_ot

