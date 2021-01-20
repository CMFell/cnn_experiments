import numpy as np
import pandas as pd

def windows_truth(df_in, img_sz):
    df_in['xc_pix_rect'] = np.multiply(df_in['xc'], img_sz[1])
    df_in['yc_pix_rect'] = np.multiply(df_in['yc'], img_sz[0])
    df_in['wid_rect'] = np.multiply(df_in['wid'], img_sz[1])
    df_in['hei_rect'] = np.multiply(df_in['height'], img_sz[0])
    df_in['xmn'] = np.array(np.subtract(df_in['xc_pix_rect'], np.divide(df_in['wid_rect'], 2)), dtype=np.int)
    df_in['xmx'] = np.array(np.add(df_in['xc_pix_rect'], np.divide(df_in['wid_rect'], 2)), dtype=np.int)
    df_in['ymn'] = np.array(np.subtract(df_in['yc_pix_rect'], np.divide(df_in['hei_rect'], 2)), dtype=np.int)
    df_in['ymx'] = np.array(np.add(df_in['yc_pix_rect'], np.divide(df_in['hei_rect'], 2)), dtype=np.int)
    return df_in