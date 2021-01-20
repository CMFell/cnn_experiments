import numpy as np
from scipy.special import expit


def yolo_output_to_box_test(y_pred, conf_threshold):    
    n_bat = y_pred.shape[0]
    # n_bat = int(dict_in['batch_size'])
    boxsx = y_pred.shape[2]
    # boxsx = int(dict_in['boxs_x'])
    boxsy = y_pred.shape[1]
    # boxsy = int(dict_in['boxs_y'])
    # anchors = dict_in['anchors']
    nanchors = anchors.shape[0]
    num_out = int(y_pred.shape[3] / nanchors)
    n_classes = num_out - 5
    # n_classes = int(dict_in['n_classes'])
    # num_out = 5 + n_classes
    # thresh = dict_in['threshold']
    # size of all boxes anchors and data
    size1 = [n_bat, boxsy, boxsx, nanchors, num_out]
    # size of all boxes and anchors
    size2 = [n_bat, boxsy, boxsx, nanchors]
    # number of boxes in each direction used for calculations rather than sizing so x first
    size3 = [boxsx, boxsy]
    # get top left position of cells
    rowz = np.arange(boxsy)
    colz = np.arange(boxsx)
    # rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    rowno = np.expand_dims(np.expand_dims(np.reshape(np.repeat(rowz, boxsx), (boxsy, boxsx)), axis=0), axis=3)
    # colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.expand_dims(np.expand_dims(np.reshape(np.tile(colz, boxsy), (boxsy, boxsx)), axis=0), axis=3)
    tl_cell = np.stack((colno, rowno), axis=4)
    # restructure net_output
    y_pred = np.reshape(y_pred, size1)

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
    
    for img in range(n_bat):
        box_img = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'conf', 'class'])
        for yc in range(boxsy):
            for xc in range(boxsx):
                for ab in range(nanchors):
                    if confs_cnn[img, yc, xc, ab] > conf_threshold:
                        # scores_out.append(confs_cnn[img, yc, xc, ab])
                        class_out = np.argmax(class_cnn[img, yc, xc, ab, :])
                        # classes_out.append(class_out)
                        detect_deets = pd.DataFrame(
                            [[
                                cent_cnn[img, yc, xc, ab, 0],
                                cent_cnn[img, yc, xc, ab, 1],
                                size_cnn[img, yc, xc, ab, 0],
                                size_cnn[img, yc, xc, ab, 1],
                                confs_cnn[img, yc, xc, ab],
                                class_out
                            ]],
                            columns=['xc', 'yc', 'wid', 'hei', 'conf', 'class']
                        )
                        box_img = box_img.append(detect_deets)
    
    return box_img
