import numpy as np
import pandas as pd


def match_results_to_truth(window_results, truths_im, iou_threshold):
    columns_out = ['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal', 'confmat', 'tru_box']
    results_out = pd.DataFrame(columns=columns_out)

    if truths_im.shape[0] == 0:
        results_per_im = window_results[['xmn', 'xmx', 'ymn', 'ymx', 'conf', 'animal', 'not_animal']]
        results_per_im['confmat'] = 'FP'
        results_per_im['tru_box'] = ''
        results_out = pd.concat((results_out, results_per_im), axis=0)
    else:
        results_out = match_truths(window_results, truths_per_im)

    return results_out
    