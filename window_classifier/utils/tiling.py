import numpy as np

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

def create_tile_list(whole_im):
    gfrcwindz = split_locations_array()
    
    # split image into tiles
    im_tiles = []
    for tl in gfrcwindz:
        tile = whole_im[tl[0]:tl[2], tl[1]:tl[3]]
        im_tiles.append(tile)

    return im_tiles