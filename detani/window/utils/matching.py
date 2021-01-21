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
            box_ot = pd.DataFrame(index=[1], columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])
            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]
            cnfs = cnfs[mask_bxs]
            anis = anis[mask_bxs]
            nots = nots[mask_bxs]

            box_ot.xmn.iloc[0] = np.min(xmns)
            box_ot.ymn.iloc[0] = np.min(ymns)
            box_ot.xmx.iloc[0] = np.max(xmxs)
            box_ot.ymx.iloc[0] = np.max(ymxs)
            box_ot.conf.iloc[0] = np.mean(cnfs)
            box_ot.animal.iloc[0] = np.mean(anis)
            box_ot.not_animal.iloc[0] = np.mean(nots)

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
            box_ot = pd.DataFrame(index=[1], columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])
            box_ot.xmn.iloc[0] = xmn
            box_ot.ymn.iloc[0] = ymn
            box_ot.xmx.iloc[0] = xmx
            box_ot.ymx.iloc[0] = ymx
            box_ot.conf.iloc[0] = cnf
            box_ot.animal.iloc[0] = ani
            box_ot.not_animal.iloc[0] = ntz

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
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0)
        
    boxes_ot['confmat'] = 'FP'
    boxes_ot['tru_box'] = ''

    return boxes_ot


def match_truths(windows_out_pixels, truths_per_im):

    results_out = pd.DataFrame(columns = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal'])

    if truths_per_im.shape[0] > 0:
        results_per_im = windows_out_pixels[['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal']]
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
                    best_iou_res['confmat'] = 'TP'
                    true_box = f'xmin: {tru.xmn}, xmax:{tru.xmx}, ymin: {tru.ymn}, ymax: {tru.ymx}'
                    best_iou_res['tru_box'] = true_box
                    results_out = pd.concat((results_out, best_iou_res), axis=0, ignore_index=True)
                    truz[idx] = False
                    rez[iou_ind] = False
            # matchz removes any matches that overlap but are not the most overlapping
            match_mask = np.array(iouz) > iou_threshold
            matchz[match_mask] = False

        results_per_im = results_per_im[matchz]
        results_per_im['confmat'] = 'FP'
        results_per_im['tru_box'] = ''
        results_per_im = nms_for_fp(results_per_im, 0.5)
        results_out = pd.concat((results_out, results_per_im), axis=0, ignore_index=True)
        false_negatives = truths_per_im[['xmn', 'xmx', 'ymn', 'ymx']]
        false_negatives = false_negatives[truz]
        false_negatives['conf'] = 0
        false_negatives['animal'] = 0
        false_negatives['not_animal'] = 0
        false_negatives['confmat'] = 'FN'
        false_negatives['tru_box'] = ''
        results_out = pd.concat((results_out, false_negatives), axis=0, ignore_index=True)

    return results_out 


