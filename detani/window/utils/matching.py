import numpy as np
import pandas as pd


def intersection_over_union(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xa = max(box1.xmn, box2.xmn)
    xb = min(box1.xmx, box2.xmx)
    ya = max(box1.ymn, box2.ymn)
    yb = min(box1.ymx, box2.ymx)
    # compute the area of intersection rectangle
    inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    # compute the area of both the prediction and ground-truth
    box1_area = (box1.xmx - box1.xmn + 1) * (box1.ymx - box1.ymn + 1)
    box2_area = (box2.xmx - box2.xmn + 1) * (box2.ymx - box2.ymn + 1)
    # compute the intersection over union 
    iou = inter_area / float(box1_area + box2_area - inter_area)
    # return the intersection over union value
    return iou


def nms_for_fp(boxes_in, thresh):
    
    xmins = boxes_in.xmn
    xmaxs = boxes_in.xmx
    ymins = boxes_in.ymn
    ymaxs = boxes_in.ymx
    confs = boxes_in.conf
    anims = boxes_in.animal
    notas = boxes_in.not_animal

    boxes_ot = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])

    xmins = np.array(xmins.tolist())
    xmaxs = np.array(xmaxs.tolist())
    ymins = np.array(ymins.tolist())
    ymaxs = np.array(ymaxs.tolist())
    confs = np.array(confs.tolist())
    anims = np.array(anims.tolist())
    notas = np.array(notas.tolist())

    while len(xmins) > 0:

        xmn = xmins[0]
        xmns = np.array(xmins[1:])
        xmx = xmaxs[0]
        xmxs = np.array(xmaxs[1:])
        ymn = ymins[0]
        ymns = np.array(ymins[1:])
        ymx = ymaxs[0]
        ymxs = np.array(ymaxs[1:])
        cnf = confs[0]
        cnfs = np.array(confs[1:])
        ani = anims[0]
        anis = np.array(anims[1:])
        ntz = notas[0]
        nots = np.array(notas[1:])

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
            box_ot = pd.DataFrame(index=range(1), columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])

            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]
            cnfs = cnfs[mask_bxs]
            anis = anis[mask_bxs]
            nots = nots[mask_bxs]

            box_ot.loc[0, 'xmn'] = np.min(xmns)
            box_ot.loc[0, 'ymn'] = np.min(ymns)
            box_ot.loc[0, 'xmx'] = np.max(xmxs)
            box_ot.loc[0, 'ymx'] = np.max(ymxs)
            box_ot.loc[0, 'conf'] = np.mean(cnfs)
            box_ot.loc[0, 'animal'] = np.mean(anis)
            box_ot.loc[0, 'not_animal'] = np.mean(nots)

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out[1:] = mask_bxs
            mask_out = np.logical_not(mask_out)

            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            confs = confs[mask_out]
            anims = anims[mask_out]
            notas = notas[mask_out]
            
        else:
            box_ot = pd.DataFrame(index=range(1), columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])

            box_ot.loc[0, 'xmn'] = xmn
            box_ot.loc[0, 'ymn'] = ymn
            box_ot.loc[0, 'xmx'] = xmx
            box_ot.loc[0, 'ymx'] = ymx
            box_ot.loc[0, 'conf'] = cnf
            box_ot.loc[0, 'animal'] = ani
            box_ot.loc[0, 'not_animal'] = ntz

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out = np.logical_not(mask_out)
            
            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            confs = confs[mask_out]
            anims = anims[mask_out]
            notas = notas[mask_out]
            
        #box_ot = box_ot.reset_index(drop=True)
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0, sort=False)

    boxes_ot.loc[:, 'confmat'] = 'FP'
    boxes_ot.loc[:, 'tru_box'] = ''

    return boxes_ot


def match_truths(windows_out_pixels, truths_per_im, iou_threshold):

    if windows_out_pixels.shape[0] > 0:
        results_out = pd.DataFrame(columns = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])

        results_per_im = windows_out_pixels[['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal']]
        results_per_im = results_per_im.reset_index(drop=True)
        truz = [True] * truths_per_im.shape[0]
        rez = [True] * results_per_im.shape[0]
        matchz = np.array([True] * results_per_im.shape[0])
        for idx, tru in truths_per_im.iterrows():
            iouz = []
            for res_idx, result in results_per_im.iterrows():
                iou = intersection_over_union(tru, result)
                iouz.append(iou)
            iou_ind = np.argmax(iouz)
            if iouz[iou_ind] > iou_threshold:
                if rez[iou_ind]: 
                    best_iou_res = results_per_im.iloc[iou_ind:(iou_ind+1), :]
                    best_iou_res = best_iou_res.reset_index(drop=True)
                    best_iou_res.loc[:, 'confmat'] = 'TP'
                    true_box = f'xmin: {tru.xmn}; xmax:{tru.xmx}; ymin: {tru.ymn}; ymax: {tru.ymx}'
                    best_iou_res.loc[:, 'tru_box'] = true_box
                    results_out = pd.concat((results_out, best_iou_res), axis=0, ignore_index=True, sort=False)
                    truz[idx] = False
                    rez[iou_ind] = False
            # matchz removes any matches that overlap but are not the most overlapping
            match_mask = np.array(iouz) > iou_threshold
            matchz[match_mask] = False

        results_per_im = results_per_im[matchz]
        results_per_im = results_per_im.reset_index(drop=True)
        if results_per_im.shape[0] > 1:
            results_per_im = nms_for_fp(results_per_im, 0.5)
            results_per_im = results_per_im.reset_index(drop=True)
        results_out = pd.concat((results_out, results_per_im), axis=0, ignore_index=True, sort=False)  
        if np.sum(truz) > 0:
            false_negatives = truths_per_im[['xmn', 'xmx', 'ymn', 'ymx']]
            false_negatives = false_negatives[truz]
            false_negatives = false_negatives.reset_index(drop=True)
            false_negatives.loc[:, 'conf'] = 0
            false_negatives.loc[:, 'animal'] = 0
            false_negatives.loc[:, 'not_animal'] = 0
            false_negatives.loc[:, 'confmat'] = 'FN'
            false_negatives.loc[:, 'tru_box'] = ''
            results_out = pd.concat((results_out, false_negatives), axis=0, ignore_index=True, sort=False)
        results_out = results_out.reset_index(drop=True)
    else:
        results_out = truths_per_im.loc[:, ['xmn', 'xmx', 'ymn', 'ymx']]
        results_out.loc[:, 'conf'] = 0
        results_out.loc[:, 'animal'] = 0
        results_out.loc[:, 'not_animal'] = 0
        results_out.loc[:, 'confmat'] = 'FN'
        results_out.loc[:, 'tru_box'] = ''


    return results_out 


