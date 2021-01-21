import copy

import cv2


def draw_results_on_image(whole_im, results_out):
    whole_im_out = copy.deepcopy(whole_im)

    # draw TP
    tp_results = results_out[results_out.confmat == 'TP']
    for idx, row in tp_results.iterrows():
        cv2.rectangle(whole_im_out,(row.xmn,row.ymn),(row.xmx,row.ymx),(0,255,0),5)
        
    # draw FP
    fp_results = results_out[results_out.confmat == 'FP']
    for idx, row in fp_results.iterrows():
        cv2.rectangle(whole_im_out,(row.xmn,row.ymn),(row.xmx,row.ymx),(255,255,0),5)
        
    # draw FN
    fn_results = results_out[results_out.confmat == 'FN']
    for idx, row in fn_results.iterrows():
        cv2.rectangle(whole_im_out,(row.xmn,row.ymn),(row.xmx,row.ymx),(255,0,0),5) 

    return whole_im_out
