import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import random


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


def draw_res(results_all_ims, image_files, whole_image_dir, image_out_dir, experiment):
    
    Path(image_out_dir).mkdir(parents=True, exist_ok=True)

    for fl in image_files:
        # Per image
        whole_im = cv2.imread(whole_image_dir + fl)
        whole_im = cv2.cvtColor(whole_im, cv2.COLOR_BGR2RGB)

        fl_png = fl

        # calculate results
        results_per_im = results_all_ims[results_all_ims.filename == fl]

        # draw results on image
        image_out = draw_results_on_image(whole_im, results_per_im)
        #image_out = cv2.resize(image_out, (1840, 1228))
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        im_name = fl[:-4] + '_' + experiment + fl[-4:]
        cv2.imwrite(str(image_out_dir + im_name), image_out)
        #cv2.imwrite(str(image_out_dir + fl), image_out)

def get_fpz(results_all_ims, valid_whole_image_dir):

    fpz_out = []
    
    fn_images = results_all_ims[results_all_ims.confmat=='FP']
    
    image_files = np.unique(fn_images.filename).tolist()
    image_files = [Path(fl) for fl in image_files]
    image_files = [img.name for img in image_files]

    for fl in image_files:
        # Per image
        whole_im = cv2.imread(valid_whole_image_dir + fl)
        whole_im = cv2.cvtColor(whole_im, cv2.COLOR_BGR2RGB)

        fl_png = fl

        # calculate results
        results_per_im = results_all_ims[results_all_ims.filename == fl]

        # create list of all false negatives
        for rw in range(results_per_im.shape[0]):
            row = results_per_im.iloc[rw, :]
            if row.confmat == 'FP':
                xmn = max(0, row.xmn - 50)
                ymn = max(0, row.ymn - 50)
                xmx = min(whole_im.shape[1], row.xmx + 50)
                ymx = min(whole_im.shape[0], row.ymx + 50)
                fp_window = whole_im[row.ymn:row.ymx, row.xmn:row.xmx]
                f_window = cv2.resize(fp_window, (fp_window.shape[1]*2, fp_window.shape[0]*2))
                fpz_out.append(fp_window)
        
    return fpz_out

def get_fnz(results_all_ims, valid_whole_image_dir):

    fnz_out = []
    
    fn_images = results_all_ims[results_all_ims.confmat=='FN']
    
    image_files = np.unique(fn_images.filename).tolist()
    image_files = [Path(fl) for fl in image_files]
    image_files = [img.name for img in image_files]

    for fl in image_files:
        # Per image
        whole_im = cv2.imread(valid_whole_image_dir + fl)
        whole_im = cv2.cvtColor(whole_im, cv2.COLOR_BGR2RGB)

        fl_png = fl

        # calculate results
        results_per_im = results_all_ims[results_all_ims.filename == fl]

        # create list of all false negatives
        for rw in range(results_per_im.shape[0]):
            row = results_per_im.iloc[rw, :]
            if row.confmat == 'FN':
                xmn = max(0, row.xmn - 50)
                ymn = max(0, row.ymn - 50)
                xmx = min(whole_im.shape[1], row.xmx + 50)
                ymx = min(whole_im.shape[0], row.ymx + 50)
                fn_window = whole_im[row.ymn:row.ymx, row.xmn:row.xmx]
                fn_window = cv2.resize(fn_window, (fn_window.shape[1]*2, fn_window.shape[0]*2))
                fnz_out.append(fn_window)
        
    return fnz_out


def create_mosaic(list_in, mosaic_tuple, image_size, multiplier=1, grey=False):
    rowz = mosaic_tuple[0]
    colz = mosaic_tuple[1]
    img_rw = image_size[0] 
    img_cl = image_size[1] 
    img_asp =  img_rw / img_cl
    channels = 3
    if grey:
        channels = 1
    combined_im = np.zeros((img_rw * rowz * multiplier, img_cl * colz * multiplier, channels), dtype=np.uint8)
    sample_list = random.sample(list_in, rowz*colz)
    for idx, im in enumerate(sample_list):
        im_shp = im.shape
        im_rw = im_shp[0]
        im_cl = im_shp[1]
        im_asp = im_rw / im_cl
        if im_asp > img_asp:
            tot_cls = im_shp[0] / img_asp
            ncls_to_add = int((tot_cls - im_shp[1]) / 2)
            cls_to_add = np.ones((im_shp[0], ncls_to_add, 3)) * 255
            border_im = np.hstack((cls_to_add, im, cls_to_add))
        else:
            tot_rws = im_shp[1] * img_asp
            nrws_to_add = int((tot_rws - im_shp[0]) / 2)
            rws_to_add = np.ones((nrws_to_add, im_shp[1], 3)) * 255
            border_im = np.vstack((rws_to_add, im, rws_to_add))

        im_reshape = cv2.resize(border_im, (img_cl * multiplier, img_rw * multiplier))
        col = idx % colz
        row = idx // colz
        x1 = col * img_cl * multiplier
        x2 = (col + 1) * img_cl * multiplier
        y1 = row * img_rw * multiplier
        y2 = (row + 1) * img_rw * multiplier
        combined_im[y1:y2, x1:x2, :] = im_reshape
        
    return combined_im


