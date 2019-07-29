import torch
import numpy as np
import pandas as pd
from scipy.special import expit
import math

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
                        clz_out = expit(pred_in[bt, by, bx, ak, 5:].tolist())
                        vec_out = [xc_out, yc_out, wid_out, hei_out, cnf_out, clz_out]
                        vec_out = np.reshape(vec_out, (1, vecsize))
                        vec_out = pd.DataFrame(vec_out, columns=colnamez[1:])
                        vec_out['filen'] = filenmz[bt]
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
    predbox = pred_to_box(ypred, filenmz, ankbox, confthr)
    # print("total preds", predbox.shape[0])
    bndbxs = bndbxs.numpy()
    bndbxs_out = bndbxs[0, :, :]
    bndbxs_out = pd.DataFrame(bndbxs_out, columns=["class", "xc", "yc", "wid", "hei"])
    bndbxs_out['filen'] = filenmz[0]
    for fl in range(1, bndbxs.shape[0]):
        bndbx = bndbxs[fl, :, :]
        bndbx = pd.DataFrame(bndbx, columns=["class", "xc", "yc", "wid", "hei"])
        bndbx['filen'] = filenmz[fl]
        bndbxs_out = bndbxs_out.append(bndbx)
    iouz = []
    bbxz = []
    for pb in range(predbox.shape[0]):
        iou_max = 0
        bb_ind = math.nan
        for bb in range(bndbxs_out.shape[0]):
            predb = predbox.iloc[pb]
            bndb = bndbxs_out.iloc[bb]
            if bndb.xc * bndb.yc > 0:
                iou = calc_iou_centwh(predb, bndb)
                if iou > iou_max:
                    iou_max = iou
                    bb_ind = bb
        iouz.append(iou_max)
        bbxz.append(bb_ind)
    tps = np.repeat(0, len(iouz))
    bbxz = np.array(bbxz)
    iouz = np.array(iouz)
    tot_true = 0
    for img in range(bndbxs.shape[0]):
        if predbox.shape[0] > 0:
            predz_mask = predbox.filen == filenmz[img]
            predz_mask = np.array(predz_mask)
            # find maximum number of boundboxes for that image
            bbxz_img = bndbxs_out[bndbxs_out.filen == filenmz[img]]
            bbxz_area = bbxz_img.xc * bbxz_img.yc
            tot_bbx = np.sum(bbxz_area > 0)
            # print("total truths", tot_bbx)
            tot_true += tot_bbx
            for bb in range(tot_bbx):
                bb_mask = bbxz == bb
                fin_mask = np.logical_and(predz_mask, bb_mask)
                if np.sum(np.array(fin_mask, dtype=np.int32)) > 0:
                    max_iou = np.max(iouz[fin_mask])
                    if max_iou > iouthr:
                        maxiou_mask = np.array(iouz == max_iou, dtype=np.int)
                        tps += maxiou_mask

    iouz = np.array(iouz)
    predbox["iou"] = iouz
    # tps = iouz > iouthr
    predbox["tp"] = tps
    # print("tps", tps)
    tot_tps = np.sum(tps)

    return predbox, tot_true, tot_tps


