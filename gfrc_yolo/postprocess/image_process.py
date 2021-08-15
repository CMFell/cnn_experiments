import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io

def get_image_file_names(whole_image_dir):
    image_files_jpg = list(Path(whole_image_dir).glob("**/*.jpg"))
    image_files_png = list(Path(whole_image_dir).glob("**/*.png"))
    image_files = image_files_jpg + image_files_png
    image_files = [img.name for img in image_files]
    return image_files


def create_tile_list(whole_im):
    gfrcwindz = split_locations_array()
    
    # split image into tiles
    im_tiles = []
    for tl in gfrcwindz:
        tile = whole_im[tl[0]:tl[2], tl[1]:tl[3]]
        im_tiles.append(tile)

    return im_tiles


def split_locations_array():
    # create list of tile splits
    gfrccolst = [0, 1856, 3712, 5504]
    gfrccolst = np.reshape(np.tile(gfrccolst, 4), (np.square(4), 1))
    gfrccoled = [1856, 3712, 5568, 7360]
    gfrccoled = np.reshape(np.tile(gfrccoled, 4), (np.square(4), 1))

    gfrcrowst = [0, 1248, 2496, 3664]
    gfrcrowst = np.reshape(np.repeat(gfrcrowst, 4), (np.square(4), 1))
    gfrcrowed = [1248, 2496, 3744, 4912]
    gfrcrowed = np.reshape(np.repeat(gfrcrowed, 4), (np.square(4), 1))

    gfrcwindz = np.hstack((gfrcrowst, gfrccolst, gfrcrowed, gfrccoled))
    gfrcwindz = np.array(gfrcwindz, dtype=np.int)
    
    return gfrcwindz


def windows_to_whole_im(df_in):
    tile_w = 1856
    tile_h = 1256
    img_w = 7360
    img_h = 4912
    tile_vals = split_locations_array()
    xmin = np.subtract(df_in.loc[:, 'xc'], np.divide(df_in.loc[:, 'wid'], 2))
    xmax = np.add(df_in.loc[:, 'xc'], np.divide(df_in.loc[:, 'wid'], 2))
    ymin = np.subtract(df_in.loc[:, 'yc'], np.divide(df_in.loc[:, 'hei'], 2))
    ymax = np.add(df_in.loc[:, 'yc'], np.divide(df_in.loc[:, 'hei'], 2))
    xmin = np.multiply(xmin, tile_w)
    xmax = np.multiply(xmax, tile_w)
    ymin = np.multiply(ymin, tile_h)
    ymax = np.multiply(ymax, tile_h)
    tileord = df_in.tile.astype(int).tolist()
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
    df_in.loc[:, 'xmn'] = np.array(xmin, dtype=np.int)
    df_in.loc[:, 'xmx'] = np.array(xmax, dtype=np.int)
    df_in.loc[:, 'ymn'] = np.array(ymin, dtype=np.int)
    df_in.loc[:, 'ymx'] = np.array(ymax, dtype=np.int)
    return df_in


def split_filename(str_in):
    file_nm = str_in[:-4]
    file_splt = file_nm.split('_')
    file_out = file_splt[0] + '_' + file_splt[1] + '.jpg'
    tile_out = file_splt[2]
    return file_out, tile_out


def predict_on_images(image_files, whole_image_dir, yolo_model, basedir):
          
    windows_whole = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'conf', 'class', 'xmn', 'xmx', 'ymn', 'ymx', 'filename'])

    for idx, fl in enumerate(image_files):
        # Per image
        print(fl, idx + 1, "of" len(image_files))
        
        whole_im = io.imread(whole_image_dir + fl, as_gray=False)
        orig_im_size = whole_im.shape
        
        # create tiles
        tilez = create_tile_list(whole_im)

        # get predictions from yolo , 0.01 keeps confidence over 1% only
        boxes_whole_im = yolo_model.inference_on_image(tilez, 0.01, fl, basedir)
        # convert tile results into whole image results
        windows_whole_im = windows_to_whole_im(boxes_whole_im)
        windows_whole_im['filename'] = fl

        windows_whole = pd.concat((windows_whole, windows_whole_im), axis=0)

    return windows_whole
