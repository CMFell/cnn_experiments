import numpy as np

def process_annotation_df_negative(detections_fil, img_size):
    pix_width, pix_height = img_size
    detections_fil['wid_pixels'] = np.array(np.multiply(detections_fil['wid'], pix_width), dtype=np.int)
    detections_fil['hei_pixels'] = np.array(np.multiply(detections_fil['hei'], pix_height), dtype=np.int)
    detections_fil['square_size'] = np.maximum(detections_fil['wid_pixels'], detections_fil['hei_pixels'])
    detections_fil['xc_pix'] = np.array(np.multiply(detections_fil['xc'], pix_width), dtype=np.int)
    detections_fil['yc_pix'] = np.array(np.multiply(detections_fil['yc'], pix_height), dtype=np.int)
    detections_fil['xmin'] = np.array(np.minimum(np.maximum(np.subtract(detections_fil['xc_pix'], np.divide(detections_fil['square_size'], 2)), 0), np.subtract(pix_width, detections_fil['square_size'])), dtype=np.int)
    detections_fil['ymin'] = np.array(np.minimum(np.maximum(np.subtract(detections_fil['yc_pix'], np.divide(detections_fil['square_size'], 2)), 0), np.subtract(pix_height, detections_fil['square_size'])), dtype=np.int)
    detections_fil['xmax'] = np.add(detections_fil['xmin'], detections_fil['square_size'])
    detections_fil['ymax'] = np.add(detections_fil['ymin'], detections_fil['square_size'])
    detections_fil = detections_fil.reset_index()
    return detections_fil

def create_windows_from_yolo(windows_in, tile_list):
    windowz_out = []
    for index, row in windows_in.iterrows():
        tile_for_row = tile_list[row.tile]
        row_array = tile_for_row[row.ymin:row.ymax, row.xmin:row.xmax]
        row_pil = Image.fromarray(row_array)
        windowz_out.append(row_pil)
    return windowz_out

def windows_to_whole_im(df_in):
    tile_w = 1856
    tile_h = 1256
    img_w = 7360
    img_h = 4912
    tile_vals = split_locations_array()
    xmin = np.subtract(df_in['xc'], np.divide(df_in['wid'], 2))
    xmax = np.add(df_in['xc'], np.divide(df_in['wid'], 2))
    ymin = np.subtract(df_in['yc'], np.divide(df_in['hei'], 2))
    ymax = np.add(df_in['yc'], np.divide(df_in['hei'], 2))
    xmin = np.multiply(xmin, tile_w)
    xmax = np.multiply(xmax, tile_w)
    ymin = np.multiply(ymin, tile_h)
    ymax = np.multiply(ymax, tile_h)
    tileord = df_in.tile.tolist()
    tilerowst = tile_vals[tileord, 0]
    tilecolst = tile_vals[tileord, 1]
    xmin = np.add(xmin, tilecolst)
    xmax = np.add(xmax, tilecolst)
    ymin = np.add(ymin, tilerowst)
    ymax = np.add(ymax, tilerowst)
    xmin = np.maximum(xmin, 0)
    xmax = np.minimum(xmax, img_w)
    ymin = np.maximum(ymin, 0)
    ymax = np.minimum(ymax, img_h)
    df_in['xmn'] = np.array(xmin, dtype=np.int)
    df_in['xmx'] = np.array(xmax, dtype=np.int)
    df_in['ymn'] = np.array(ymin, dtype=np.int)
    df_in['ymx'] = np.array(ymax, dtype=np.int)
    return df_in