import numpy as np
import pandas as pd

from window.utils.matching import match_truths

def match_results_to_truths(window_results, truths_im, iou_threshold):
    columns_out = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal', 'confmat', 'tru_box']
    results_out = pd.DataFrame(columns=columns_out)
    window_reind = window_results.reset_index(drop=True)

    if truths_im.shape[0] == 0:
        results_per_im = window_reind.loc[:, ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal']]
        results_per_im['confmat'] = 'FP'
        results_per_im['tru_box'] = ''
        results_out = pd.concat((results_out, results_per_im), axis=0, sort=False)
    else:
        results_out = match_truths(window_reind, truths_im, iou_threshold)

    return results_out
    