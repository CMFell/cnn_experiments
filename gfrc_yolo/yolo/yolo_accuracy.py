import math
import numpy as np
import pandas as pd
from scipy.special import expit
import torch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def accuracy(y_pred, y_true, thresh):
    outputs = y_pred.unsqueeze(4)
    outputs = torch.chunk(outputs, 5, dim=3)
    outputs = torch.cat(outputs, dim=4)
    outputs = outputs.transpose(4, 3)
    predconf = torch.sigmoid(outputs[..., 4])
    ones = y_true[..., 4]
    poz = torch.ge(predconf, thresh)
    negz = torch.lt(predconf, thresh)
    truez = torch.ge(ones, thresh)
    falsez = torch.lt(ones, thresh)
    tp = torch.sum(poz & truez)
    fp = torch.sum(poz & falsez)
    fn = torch.sum(negz & truez)

    return tp, fp, fn


def pred_to_box(pred_in, filenmz, ankbox, thresh):
    pred_in = pred_in.unsqueeze(4)
    pred_in = torch.chunk(pred_in, 5, dim=3)
    pred_in = torch.cat(pred_in, dim=4)
    pred_in = pred_in.transpose(3, 4)
    n_bat, boxsy, boxsx, ankz, vecsize = pred_in.shape
    nclass = vecsize - 5
    colnamez = ['filen', 'xc', 'yc', 'wid', 'hei', 'conf']
    for cl in range(nclass):
        clazz = 'class' + str(cl + 1)
        colnamez.append(clazz)
    boxes_out = pd.DataFrame(columns=colnamez)
    confz = torch.sigmoid(pred_in[..., 4])
    # print(torch.max(confz))
    for bt in range(n_bat):
        for by in range(boxsy):
            for bx in range(boxsx):
                for ak in range(ankz):
                    if confz[bt, by, bx, ak] > thresh:
                        xc_out = (expit(pred_in[bt, by, bx, ak, 0].tolist()) + bx) / boxsx
                        yc_out = (expit(pred_in[bt, by, bx, ak, 1].tolist()) + by) / boxsy
                        wid_out = np.exp(pred_in[bt, by, bx, ak, 2].tolist()) * ankbox[ak, 0] / boxsx
                        hei_out = np.exp(pred_in[bt, by, bx, ak, 3].tolist()) * ankbox[ak, 1] / boxsy
                        cnf_out = expit(pred_in[bt, by, bx, ak, 4].tolist())
                        clz_out = softmax(pred_in[bt, by, bx, ak, 5:].tolist())
                        vec_out = [xc_out, yc_out, wid_out, hei_out, cnf_out]
                        vec_out.extend(clz_out.tolist())
                        vec_out = np.reshape(vec_out, (1, vecsize))
                        vec_out = pd.DataFrame(vec_out, columns=colnamez[1:])
                        filenm_list = np.repeat(filenmz[bt], vec_out.shape[0])
                        vec_out.insert(0, 'filen', filenm_list)
                        # vec_out['filen'] = filenmz[bt]
                        boxes_out = boxes_out.append(vec_out)

    return boxes_out


def calc_iou_centwh(box1, box2):

    xmn1 = box1.xc - box1.wid / 2
    xmx1 = box1.xc + box1.wid / 2
    ymn1 = box1.yc - box1.hei / 2
    ymx1 = box1.yc + box1.hei / 2
    xmn2 = box2.xc - box2.wid / 2
    xmx2 = box2.xc + box2.wid / 2
    ymn2 = box2.yc - box2.hei / 2
    ymx2 = box2.yc + box2.hei / 2

    ol_xmn = max(xmn2, xmn1)
    ol_xmx = min(xmx2, xmx1)
    ol_ymn = max(ymn2, ymn1)
    ol_ymx = min(ymx2, ymx1)

    olx = max(ol_xmx - ol_xmn, 0)
    oly = max(ol_ymx - ol_ymn, 0)

    ol_area = olx * oly
    bx1_area = box1.wid * box1.hei
    bx2_area = box2.wid * box2.hei

    iou = ol_area / (bx1_area + bx2_area - ol_area)

    return iou


def accuracyiou(ypred, bndbxs, filenmz, ankbox, confthr, iouthr):
    # convert net output to detections with bounding boxes
    predbox = pred_to_box(ypred, filenmz, ankbox, confthr)
    # predbox is dataframe with columns [filen, xc, yc, wid, hei, conf, class1]

    # consider each image separately create df on first image then append others
    # convert truths to numpy array
    bndbxs = bndbxs.numpy()
    # get truths for just first image
    bndbxs_out = bndbxs[0, :, :]
    # convert to data frame
    bndbxs_out = pd.DataFrame(bndbxs_out, columns=["class", "xc", "yc", "wid", "hei"])
    # add file name 
    bndbxs_out['filen'] = filenmz[0]
    # repeat for rest of images in batch
    for fl in range(1, bndbxs.shape[0]):
        bndbx = bndbxs[fl, :, :]
        bndbx = pd.DataFrame(bndbx, columns=["class", "xc", "yc", "wid", "hei"])
        bndbx['filen'] = filenmz[fl]
        bndbxs_out = bndbxs_out.append(bndbx)
    # bndbxs_out is now a dataframe with truths for all images and has column headings
    # ["class", "xc", "yc", "wid", "hei", "filen"]

    iouz = []
    bbxz = []
    # for each detection
    for pb in range(predbox.shape[0]):
        iou_max = 0
        bb_ind = math.nan
        # for each truth
        for bb in range(bndbxs_out.shape[0]):
            # get bounding boxes of truth and detection
            predb = predbox.iloc[pb]
            bndb = bndbxs_out.iloc[bb]
            # if the truth is an actual box as opposed to zero padding
            if bndb.xc * bndb.yc > 0:
                # check truth and detection are same image
                if bndb.filen == predb.filen:
                    # calculate the iou of the truth and detection
                    iou = calc_iou_centwh(predb, bndb)
                    # if it is the most overlapping set new iou and index of truth
                    if iou > iou_max:
                        iou_max = iou
                        bb_ind = bb
        # add iou and bound box index for that detection
        iouz.append(iou_max)
        bbxz.append(bb_ind)
    # iouz should be the same length as number of predictions
    # iouz should be zero if no overlaping prediction
    # bbxz should be the same length as number of predictions
    # bbxz should be nan if no overlapping prediction

    # create tp all zero
    tps = np.repeat(0, len(iouz))
    # convert bbxz and iouz to numpy arrays
    bbxz = np.array(bbxz)
    iouz = np.array(iouz)
    # set total true count to zero for batch
    tot_true = 0

    for img in range(bndbxs.shape[0]):
        # find maximum number of boundboxes for that image
        bbxz_img = bndbxs_out[bndbxs_out.filen == filenmz[img]]
        bbxz_area = bbxz_img.xc * bbxz_img.yc
        tot_bbx = np.sum(bbxz_area > 0)
        tot_true += tot_bbx
        # if there are any predictions
        if predbox.shape[0] > 0:
            # create filter so only consider predictions of one image
            predz_mask = predbox.filen == filenmz[img]
            predz_mask = np.array(predz_mask)
            #print("pm", predz_mask.shape)
            # find maximum number of boundboxes for that image
            # bbxz_img = bndbxs_out[bndbxs_out.filen == filenmz[img]]
            # bbxz_area = bbxz_img.xc * bbxz_img.yc
            # tot_bbx = np.sum(bbxz_area > 0)
            # print("total truths", tot_bbx)
            # tot_true += tot_bbx
            for bb in range(tot_bbx):
                # create filter so only looking at detections that overlap this truth
                bb_mask = bbxz == bb
                # combine masks so only looking at detections for this truth and this image
                fin_mask = np.logical_and(predz_mask, bb_mask)
                # if there are detections that match this truth in this image
                if np.sum(np.array(fin_mask, dtype=np.int32)) > 0:
                    # get maximum iou for this truth and this image
                    max_iou = np.max(iouz[fin_mask])
                    # check if greater than threshold
                    if max_iou > iouthr:
                        # find which detection has the maximum iou
                        maxiou_mask = np.logical_and(np.array(iouz == max_iou, dtype=np.int), fin_mask)
                        #print("mim", maxiou_mask.shape)
                        tps += maxiou_mask

    tot_tps = np.sum(tps)
    iouz = np.array(iouz)
    tp_bool = np.array(tps, dtype=np.bool)
    predbox["iou"] = iouz
    tps = iouz > iouthr
    predbox["tp"] = tp_bool

    return predbox, tot_true, tot_tps


