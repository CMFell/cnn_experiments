import numpy as np
import pandas as pd

from postprocess.image_process import get_image_file_names


def remove_on_size(gfrc_windows):
    too_small = int(0.75*11)
    too_big = int(1.25*163)
    area_too_small = int(0.75*319)
    area_too_big = int(1.25*24287)

    gfrc_windows['xside'] = gfrc_windows.xmx - gfrc_windows.xmn
    gfrc_windows['yside'] = gfrc_windows.ymx - gfrc_windows.ymn
    gfrc_windows['minside'] = np.minimum(gfrc_windows.xside, gfrc_windows.yside)
    gfrc_windows['maxside'] = np.maximum(gfrc_windows.xside, gfrc_windows.yside)
    gfrc_windows['area'] = gfrc_windows.xside * gfrc_windows.yside
    print(np.max(gfrc_windows.maxside), np.min(gfrc_windows.minside))
    gfrc_windows = gfrc_windows[gfrc_windows.minside > too_small]
    gfrc_windows = gfrc_windows[gfrc_windows.maxside < too_big]
    gfrc_windows = gfrc_windows[gfrc_windows.area < area_too_big]
    gfrc_windows = gfrc_windows[gfrc_windows.area > area_too_small]
    return gfrc_windows


def nms_per_im(boxes_in, thresh, method='mean'):
    
    boxes_in = boxes_in.sort_values(by='conf', ascending=False)

    xmins = boxes_in.xmn
    xmaxs = boxes_in.xmx
    ymins = boxes_in.ymn
    ymaxs = boxes_in.ymx
    confs = boxes_in.conf
    clazs = boxes_in['class']

    boxes_ot = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'pred_class'])

    xmins = np.array(xmins.tolist())
    xmaxs = np.array(xmaxs.tolist())
    ymins = np.array(ymins.tolist())
    ymaxs = np.array(ymaxs.tolist())
    confs = np.array(confs.tolist())
    clazs = np.array(clazs.tolist())

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
        clz = clazs[0]
        clzs = np.array(clazs[1:])

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
            box_ot = pd.DataFrame(index=range(1), columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'pred_class'])

            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]
            cnfs = cnfs[mask_bxs]

            if method == 'mean':
                box_ot.loc[0, 'xmn'] = np.array(np.mean(xmns), dtype=int)
                box_ot.loc[0, 'ymn'] = np.array(np.mean(ymns), dtype=int)
                box_ot.loc[0, 'xmx'] = np.array(np.mean(xmxs), dtype=int)
                box_ot.loc[0, 'ymx'] = np.array(np.mean(ymxs), dtype=int)
                box_ot.loc[0, 'conf'] = np.mean(cnfs)
                box_ot.loc[0, 'pred_class'] = clz
            elif method == 'first':
                box_ot.loc[0, 'xmn'] = xmns[0]
                box_ot.loc[0, 'ymn'] = ymns[0]
                box_ot.loc[0, 'xmx'] = xmxs[0]
                box_ot.loc[0, 'ymx'] = ymxs[0]
                box_ot.loc[0, 'conf'] = np.max(cnfs)
                box_ot.loc[0, 'pred_class'] = clz
            else:
                box_ot.loc[0, 'xmn'] = np.min(xmns)
                box_ot.loc[0, 'ymn'] = np.min(ymns)
                box_ot.loc[0, 'xmx'] = np.max(xmxs)
                box_ot.loc[0, 'ymx'] = np.max(ymxs)
                box_ot.loc[0, 'conf'] = np.max(cnfs)
                box_ot.loc[0, 'pred_class'] = clz

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out[1:] = mask_bxs
            mask_out = np.logical_not(mask_out)

            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            confs = confs[mask_out]
            clazs = clazs[mask_out]
            
        else:
            box_ot = pd.DataFrame(index=range(1), columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'pred_class'])

            box_ot.loc[0, 'xmn'] = xmn
            box_ot.loc[0, 'ymn'] = ymn
            box_ot.loc[0, 'xmx'] = xmx
            box_ot.loc[0, 'ymx'] = ymx
            box_ot.loc[0, 'conf'] = cnf
            box_ot.loc[0, 'pred_class'] = clz

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out = np.logical_not(mask_out)
            
            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            confs = confs[mask_out]
            clazs = clazs[mask_out]
            
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0, sort=False)

    boxes_ot.loc[:, 'filename'] = boxes_in.filename.iloc[0]

    return boxes_ot


def nms_for_yolo(windows_df, nms_thresh, method):
    images = np.unique(windows_df.filename)
    windows_all_ims = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'pred_class', 'filename'])
    for im in images:
        windows_im = windows_df[windows_df.filename == im]
        windows_im = nms_per_im(windows_im, nms_thresh, method)
        windows_all_ims.append(windows_im)
        windows_all_ims = pd.concat((windows_all_ims, windows_im), axis=0, ignore_index=True, sort=False)
    return windows_all_ims


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


def match_to_truth_im(detections_df, truth_df, iou_threshold):
    
    if detections_df.shape[0] > 0:
        results_out = pd.DataFrame(columns = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'filename', 'pred_class', 
                                              'tru_class'])
        results_per_im = detections_df[['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'filename', 'pred_class']]
        results_per_im = results_per_im.reset_index(drop=True)
        truths_per_im = truth_df.reset_index(drop=True)
        # best match stores the detection with the highest iou
        best_match = [False] * results_per_im.shape[0]
        # true match stores if the truth overlaps with any detections
        true_match = [False] * truth_df.shape[0]
        # matchz stores any matches that overlap but aren't the best overlap (not double counting TP, but not adding to FP)
        matchz = np.array([True] * results_per_im.shape[0])

        for idx, tru in truths_per_im.iterrows():
            iouz = []
            for res_idx, result in results_per_im.iterrows():
                iou = intersection_over_union(tru, result)
                iouz.append(iou)
            iou_ind = np.argmax(iouz)
            if iouz[iou_ind] > iou_threshold:
                if not best_match[iou_ind]: 
                    best_iou_res = results_per_im.iloc[iou_ind:(iou_ind+1), :]
                    best_iou_res = best_iou_res.reset_index(drop=True)
                    best_iou_res.loc[:, 'confmat'] = 'TP'
                    true_box = f'xmin: {tru.xmn}; xmax:{tru.xmx}; ymin: {tru.ymn}; ymax: {tru.ymx}'
                    best_iou_res.loc[:, 'tru_box'] = true_box
                    best_iou_res.loc[:, 'tru_class'] = tru.tru_class
                    results_out = pd.concat((results_out, best_iou_res), axis=0, ignore_index=True, sort=False)
                    best_match[iou_ind] = True
                    true_match[idx] = True
            # matchz removes any matches that overlap but are not the most overlapping
            match_mask = np.array(iouz) > iou_threshold
            matchz[match_mask] = False

        # use matchz to filter results to keep only those that don't overlap with truths
        results_per_im = results_per_im[matchz]
        results_per_im = results_per_im.reset_index(drop=True)

        if results_per_im.shape[0] > 0:
            results_per_im['confmat'] = 'FP'
            results_per_im['tru_box'] = ''
            results_per_im['tru_class'] = 0
        results_out = pd.concat((results_out, results_per_im), axis=0, ignore_index=True, sort=False)  
        true_match = np.array(true_match)
        true_match = np.logical_not(true_match)
        if np.sum(true_match) > 0:
            false_negatives = truth_df[['xmn', 'xmx', 'ymn', 'ymx', 'filename', 'tru_class']]
            false_negatives = false_negatives[true_match]
            false_negatives = false_negatives.reset_index(drop=True)
            false_negatives.loc[:, 'conf'] = 1.0
            false_negatives.loc[:, 'confmat'] = 'FN'
            false_negatives.loc[:, 'tru_box'] = ''
            false_negatives.loc[:, 'pred_class'] = 0
            results_out = pd.concat((results_out, false_negatives), axis=0, ignore_index=True, sort=False)
        results_out = results_out.reset_index(drop=True)
    else:
        results_out = truth_df.loc[:, ['xmn', 'xmx', 'ymn', 'ymx', 'filename', 'tru_class']]
        results_out.columns = ['xmn', 'xmx', 'ymn', 'ymx', 'filename', 'tru_class']
        results_out.loc[:, 'conf'] = 1
        results_out.loc[:, 'confmat'] = 'FN'
        results_out.loc[:, 'tru_box'] = ''
        results_out.loc[:, 'pred_class'] = 0

    return results_out


def match_to_truth(detections_df, truth_df, valid_whole_image_dir, iou_threshold):
    image_files = get_image_file_names(valid_whole_image_dir)
    matched_results = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'filename', 'confmat', 'tru_box', 'pred_class', 'tru_class'])
    
    for im in image_files:
        detections_im = detections_df[detections_df.filename == im]
        truth_im = truth_df[truth_df.filename == im]
        if detections_im.shape[0] > 0:
            # detections and truths need to match
            if truth_im.shape[0] > 0:
                match_results_im = match_to_truth_im(detections_im, truth_im, iou_threshold)
            # detections and no truths - all detections false postive
            else:
                match_results_im = detections_im[['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'filename', 'pred_class']]
                match_results_im['confmat'] = 'FP'
                match_results_im['tru_box'] = ''
                match_results_im['tru_class'] = 0
        else:
            # no detections and truths - all truths false negatives
            if truth_im.shape[0] > 0:
                match_results_im = truth_im[['xmn', 'xmx', 'ymn', 'ymx', 'filename', 'tru_class']]
                match_results_im['conf'] = 1.0
                match_results_im['confmat'] = 'FN'
                match_results_im['tru_box'] = ''
                match_results_im['pred_class'] = 0
            else:
                match_results_im = pd.DataFrame(columns=['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'filename', 'confmat', 'tru_box', 'pred_class', 'tru_class'])
        matched_results = pd.concat((matched_results, match_results_im), axis=0, ignore_index=True, sort=False)
    
    return matched_results


def calculate_threshold_results(windows_df, truth_df, im_dir, nms_threshold=0.05, iou_threshold=0.25, method='first', start_thresh=0.01, nsteps=100):
    threshes = np.linspace(start_thresh, 1, nsteps)
    threshold_array = np.zeros((len(threshes), 8))
    image_files = get_image_file_names(im_dir)

    for idx, th in enumerate(threshes):
        print(th)
        detections_th = windows_df[windows_df.conf > th]
        detections_th = nms_for_yolo(detections_th, nms_threshold, method)
        result_th = match_to_truth(detections_th, truth_df, im_dir, iou_threshold)
        TP = np.sum(result_th.confmat=="TP")
        FP = np.sum(result_th.confmat=="FP")
        FN = np.sum(result_th.confmat=="FN")
        RE = TP / (TP + FN)
        PR = TP / (TP + FP)
        MR = 1 - RE
        FPPI = FP / len(image_files)

        list_out = [th, TP, FP, FN, RE, PR, MR, FPPI]

        threshold_array[idx, :] = list_out 
        
    threshold_metrics = pd.DataFrame(threshold_array, columns=['threshold', 'TP', 'FP', 'FN', 'RE', 'PR', 'MR', 'FPPI'])
    return threshold_metrics


def single_threshold_results(windows_df, truth_df, im_dir, thresh, nms_threshold=0.05, iou_threshold=0.25, method='first'):

    image_files = get_image_file_names(im_dir)
    detections_th = windows_df[windows_df.conf > thresh]
    detections_th['class'] = detections_th['class'].add(1)
    detections_th = nms_for_yolo(detections_th, nms_threshold, method)
    result_th = match_to_truth(detections_th, truth_df, im_dir, iou_threshold)

    return result_th


def get_image_stats(detections_df):
    unique_images = np.unique(detections_df.filename)
    tpz = []
    fpz = []
    fnz = []
    for fl in unique_images:
        results_per_im = detections_df[detections_df['filename'] == fl]
        tpz.append(np.sum(results_per_im.confmat == 'TP'))
        fpz.append(np.sum(results_per_im.confmat == 'FP'))
        fnz.append(np.sum(results_per_im.confmat == 'FN'))

    TPz = np.reshape(np.array(tpz), (len(tpz), 1))
    FPz = np.reshape(np.array(fpz), (len(fpz), 1))
    FNz = np.reshape(np.array(fnz), (len(fnz), 1))
    UIz = np.reshape(unique_images, (len(unique_images), 1))
    df_out = pd.DataFrame(np.hstack((UIz, TPz, FPz, FNz)), columns=['filename', 'TP', 'FP', 'FN'])
    
    return df_out
    