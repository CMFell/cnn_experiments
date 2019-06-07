import numpy as np
import pandas as pd

from scipy.special import expit


# define values for calculating loss
input_shape = (307, 460, 3)
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
box_size = [16, 16]
anchor_pixel = np.multiply(anchors_in, box_size)
img_x_pix = input_shape[1]
img_y_pix = input_shape[0]
boxs_x = np.ceil(img_x_pix / 16)
boxs_y = np.ceil(img_y_pix / 16)
classes_in = 1
lambda_c = 5.0
lambda_no = 0.5
base_dir = 'E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train_lg/'
batch_size = 32
threshold = 0.5
dict_deets = {'boxs_x': boxs_x, 'boxs_y': boxs_y, 'img_x_pix': img_x_pix, 'img_y_pix': img_y_pix,
              'anchors': anchors_in, 'n_classes': classes_in, 'lambda_coord': lambda_c, 'lambda_noobj': lambda_no,
              'base_dir': base_dir, 'batch_size': batch_size, 'threshold': threshold}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def yolo_output_to_box(y_pred, dict_in):
    # compares output from cnn with ground truth to calculate loss
    # only for one image at the moment

    n_bat = y_pred.shape[0]
    # n_bat = int(dict_in['batch_size'])
    boxsx = y_pred.shape[2]
    # boxsx = int(dict_in['boxs_x'])
    boxsy = y_pred.shape[1]
    # boxsy = int(dict_in['boxs_y'])
    anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    num_out = int(y_pred.shape[3] / nanchors)
    n_classes = num_out - 5
    # n_classes = int(dict_in['n_classes'])
    num_out = 5 + n_classes
    thresh = dict_in['threshold']

    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]

    print("size1",size1)

    # get top left position of cells
    rowz = np.divide(np.arange(boxsy), boxsy)
    colz = np.divide(np.arange(boxsx), boxsx)
    rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    tl_cell = np.stack((colno, rowno), axis=4)

    print("y_in", y_pred[0,0,0,:])
    # restructure net_output
    y_pred = np.reshape(y_pred, size1)
    print("shape2",expit(y_pred[0,0,0,:]))

    # get confidences centres sizes and class predictions from from net_output
    confs_cnn = expit(np.reshape(y_pred[:, :, :, :, 4], size2))
    cent_cnn = expit(y_pred[:, :, :, :, 0:2])
    # cent_cnn_in = cent_cnn
    # add to cent_cnn so is position in whole image
    cent_cnn = np.add(cent_cnn, tl_cell)
    # divide so position is relative to whole image
    cent_cnn = np.divide(cent_cnn, size3)

    size_cnn = y_pred[:, :, :, :, 2:4]
    # size is to power of prediction
    size_cnn = np.exp(size_cnn)
    # keep for loss
    # size_cnn_in = size_cnn
    # adjust so size is relative to anchors
    size_cnn = np.multiply(size_cnn, anchors)
    # adjust so size is relative to whole image
    size_cnn = np.divide(size_cnn, size3)
    class_cnn = expit(y_pred[:, :, :, :, 5:])

    boxes_out = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei'])
    scores_out = []
    classes_out = []

    print("max", np.max(confs_cnn))

    for img in range(n_bat):
        for yc in range(boxsy):
            for xc in range(boxsx):
                for ab in range(nanchors):
                    #print(confs_cnn[img, yc, xc, ab])
                    if confs_cnn[img, yc, xc, ab] > thresh:
                        boxes_out.loc[len(boxes_out)] = [cent_cnn[img, yc, xc, ab, 0], cent_cnn[img, yc, xc, ab, 1],
                                                         size_cnn[img, yc, xc, ab, 0], size_cnn[img, yc, xc, ab, 1]]
                        scores_out.append(confs_cnn[img, yc, xc, ab])
                        class_out = np.argmax(class_cnn[img, yc, xc, ab, :])
                        classes_out.append(class_out)

    output = [boxes_out, scores_out, classes_out]

    return output




