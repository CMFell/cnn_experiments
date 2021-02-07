import numpy as np
import pandas as pd

from window.utils.matching import match_truths, match_truths_yolo, nms_for_fp, nms_for_fp_yolo, match_truths_yolo_alt, match_truths_yolo_alt2

def match_results_to_truths(window_results, truths_im, iou_threshold, nms_threshold=0.5):
    columns_out = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal', 'confmat', 'tru_box']
    results_out = pd.DataFrame(columns=columns_out)
    window_reind = window_results.reset_index(drop=True)

    if truths_im.shape[0] == 0:
        results_per_im = window_reind.loc[:, ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal']]
        results_per_im['confmat'] = 'FP'
        results_per_im['tru_box'] = ''
        results_out = pd.concat((results_out, results_per_im), axis=0, sort=False)
        if results_out.shape[0] > 0:
            results_out = nms_for_fp(results_out, nms_threshold)
    else:
        results_out = match_truths(window_reind, truths_im, iou_threshold, nms_threshold)

    return results_out


def match_yolo_to_truths(window_results, truths_im, iou_threshold, nms_threshold=0.5):

    columns_out = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'confmat', 'tru_box']
    results_out = pd.DataFrame(columns=columns_out)
    window_reind = window_results.reset_index(drop=True)

    if truths_im.shape[0] == 0:
        results_per_im = window_reind.loc[:, ['xmn', 'xmx', 'ymn', 'ymx', 'conf']]
        results_per_im['confmat'] = 'FP'
        results_per_im['tru_box'] = ''
        results_out = pd.concat((results_out, results_per_im), axis=0, sort=False)
        if results_out.shape[0] > 0:
            results_out = nms_for_fp_yolo(results_out, nms_threshold)
    else:
        results_out = match_truths_yolo_alt2(window_reind, truths_im, iou_threshold, nms_threshold)

    return results_out
    