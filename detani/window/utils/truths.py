import numpy as np
import pandas as pd

def windows_truth(df_in):
    img_sz = [4912, 7360]
    df_out = df_in.reset_index(drop=True)
    xc_pix_rect = df_out['xc'].multiply(img_sz[1])
    yc_pix_rect = df_out['yc'].multiply(img_sz[0])
    wid_rect = df_out['wid'].multiply(img_sz[1])
    hei_rect = df_out['height'].multiply(img_sz[0])
    df_out['xmn'] = xc_pix_rect.subtract(wid_rect.divide(2)).astype(int)
    df_out['xmx'] = xc_pix_rect.add(wid_rect.divide(2)).astype(int)
    df_out['ymn'] = yc_pix_rect.subtract(hei_rect.divide(2)).astype(int)
    df_out['ymx'] = yc_pix_rect.add(hei_rect.divide(2)).astype(int)
    return df_out
