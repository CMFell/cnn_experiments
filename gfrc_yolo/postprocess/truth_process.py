import numpy as np
import pandas as pd

def windows_truth(df_in):
    img_sz = [4912, 7360]
    df_out = df_in.reset_index(drop=True)
    df_out['filename'] = df_out['file_loc']
    xc_pix_rect = df_out['xc'].multiply(img_sz[1])
    yc_pix_rect = df_out['yc'].multiply(img_sz[0])
    wid_rect = df_out['wid'].multiply(img_sz[1])
    hei_rect = df_out['height'].multiply(img_sz[0])
    df_out['xmn'] = xc_pix_rect.subtract(wid_rect.divide(2)).astype(int)
    df_out['xmx'] = xc_pix_rect.add(wid_rect.divide(2)).astype(int)
    df_out['ymn'] = yc_pix_rect.subtract(hei_rect.divide(2)).astype(int)
    df_out['ymx'] = yc_pix_rect.add(hei_rect.divide(2)).astype(int)
    return df_out


def process_truths(truths, dataset):
    truths.loc[:, 'filename'] = [strin.replace('/', '_') for strin in truths.file_loc]
    truths = windows_truth(truths)
    truths['oc'] = truths['oc'].add(1)
    truths['tru_class'] = truths['oc']
    return truths
