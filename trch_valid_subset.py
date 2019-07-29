import torch
from trch_yolonet import YoloNet, YoloNetSimp
from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_accuracy import accuracy, accuracyiou
from trch_weights import get_weights
import numpy as np
import pandas as pd
from scipy.special import expit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def simple_nms(boxes_in, thresh):

    # conf_ord = np.argsort(boxes_in.conf)
    boxes_in = boxes_in.sort_values(by=['conf'], ascending=False)

    xmins = boxes_in.xc - boxes_in.wid / 2
    xmaxs = boxes_in.xc + boxes_in.wid / 2
    ymins = boxes_in.yc - boxes_in.hei / 2
    ymaxs = boxes_in.yc + boxes_in.hei / 2
    flns = boxes_in.file
    cnfs = boxes_in.conf
    clzz = boxes_in['class']

    boxes_ot = pd.DataFrame(columns=["xc", "yc", "wid", "hei", "file", "conf", "class"])

    xmins = np.array(xmins.tolist())
    xmaxs = np.array(xmaxs.tolist())
    ymins = np.array(ymins.tolist())
    ymaxs = np.array(ymaxs.tolist())
    flns = np.array(flns.tolist())
    cnfs = np.array(cnfs.tolist())
    clzz = np.array(clzz.tolist())

    while len(xmins) > 0:

        xmn = xmins[0]
        xmns = np.array(xmins[1:])
        xmx = xmaxs[0]
        xmxs = np.array(xmaxs[1:])
        ymn = ymins[0]
        ymns = np.array(ymins[1:])
        ymx = ymaxs[0]
        ymxs = np.array(ymaxs[1:])

        ol_wid = np.minimum(xmx, xmxs) - np.maximum(xmn, xmns)
        ol_hei = np.minimum(ymx, ymxs) - np.maximum(ymn, ymns)

        ol_x = np.maximum(0, ol_wid)
        ol_y = np.maximum(0, ol_hei)

        distx = np.subtract(xmxs, xmns)
        disty = np.subtract(ymxs, ymns)
        bxx = xmx - xmn
        bxy = ymx - ymn

        ol_area = np.multiply(ol_x, ol_y)
        bx_area = bxx * bxy
        bxs_area = np.multiply(distx, disty)

        ious = np.divide(ol_area, np.subtract(np.add(bxs_area, bx_area), ol_area))
        mask_bxs = np.greater(ious, thresh)

        if np.sum(mask_bxs) > 0:
            box_ot = pd.DataFrame(index=[1], columns=["xc", "yc", "wid", "hei", "file", "conf", "class"])
            xmns = xmns[mask_bxs]
            xmxs = xmxs[mask_bxs]
            ymns = ymns[mask_bxs]
            ymxs = ymxs[mask_bxs]

            xmns = np.append(xmns, xmn)
            xmxs = np.append(xmxs, xmx)
            ymxs = np.append(ymxs, ymx)
            ymns = np.append(ymns, ymn)

            box_ot.xc.iloc[0] = (np.min(xmns) + np.max(xmxs)) / 2
            box_ot.yc.iloc[0] = (np.min(ymns) + np.max(ymxs)) / 2
            box_ot.wid.iloc[0] = np.max(xmxs) - np.min(xmns)
            box_ot.hei.iloc[0] = np.max(ymxs) - np.min(ymns)
            box_ot.file.iloc[0] = flns[0]
            box_ot.conf.iloc[0] = cnfs[0]
            box_ot['class'].iloc[0] = clzz[0]

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out[1:] = mask_bxs
            mask_out = np.logical_not(mask_out)

            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            flns = flns[mask_out]
            cnfs = cnfs[mask_out]
            clzz = clzz[mask_out]
        else:
            box_ot = pd.DataFrame(index=[1], columns=["xc", "yc", "wid", "hei", "file", "conf", "class"])
            box_ot.xc.iloc[0] = (xmn + xmx) / 2
            box_ot.yc.iloc[0] = (ymn + ymx) / 2
            box_ot.wid.iloc[0] = xmx - xmn
            box_ot.hei.iloc[0] = ymx - ymn
            box_ot.file.iloc[0] = flns[0]
            box_ot.conf.iloc[0] = cnfs[0]
            box_ot['class'].iloc[0] = clzz[0]

            mask_out = np.repeat(False, len(xmins))
            mask_out[0] = True
            mask_out = np.logical_not(mask_out)
            xmins = xmins[mask_out]
            xmaxs = xmaxs[mask_out]
            ymins = ymins[mask_out]
            ymaxs = ymaxs[mask_out]
            flns = flns[mask_out]
            cnfs = cnfs[mask_out]
            clzz = clzz[mask_out]
        boxes_ot = pd.concat((boxes_ot, box_ot), axis=0)

    return boxes_ot


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def yolo_output_to_box(y_pred, namez, dict_in):
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

    boxes_out = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class'])
    scores_out = []
    classes_out = []

    for img in range(n_bat):
        filen = namez[img]
        box_img = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class'])
        for yc in range(boxsy):
            for xc in range(boxsx):
                for ab in range(nanchors):
                    #print(confs_cnn[img, yc, xc, ab])
                    if confs_cnn[img, yc, xc, ab] > thresh:
                        scores_out.append(confs_cnn[img, yc, xc, ab])
                        class_out = np.argmax(class_cnn[img, yc, xc, ab, :])
                        classes_out.append(class_out)
                        box_img.loc[len(box_img)] = [cent_cnn[img, yc, xc, ab, 0], cent_cnn[img, yc, xc, ab, 1],
                                                         size_cnn[img, yc, xc, ab, 0], size_cnn[img, yc, xc, ab, 1],
                                                         filen, confs_cnn[img, yc, xc, ab], class_out]
        if box_img.shape[0] > 0:
            box_img_ot = simple_nms(box_img, 0.1)
            boxes_out = pd.concat((boxes_out, box_img_ot), axis=0)


    output = [boxes_out, scores_out, classes_out]

    return output

for xx in range(99):

    files_location_valid = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_valid1248_subset/"
    weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"

    save_dir = "E:/CF_Calcs/BenchmarkSets/GFRC/pytorch_save/size1248/"
    save_name = "testing_save_"
    save_path = save_dir + save_name + str(xx) + ".pt"

    grid_w = int(1856 / 32)
    grid_h = int(1248 / 32)
    # simplified net
    # grid_w = int(1856 / 16)
    # grid_h = int(1248 / 16)
    n_box = 5
    out_len = 6
    input_vec = [grid_w, grid_h, n_box, out_len]
    anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
                  [5.319540, 6.116692]]
    # anchors = [[0.718750, 0.890625], [0.750000, 0.515625], [0.468750, 0.562500], [1.140625, 1.156250],
    #            [0.437500, 0.328125]]
    #anchors = np.divide(anchors, 2.0)

    animal_dataset_valid_sm = AnimalBoundBoxDataset(root_dir=files_location_valid,
                                           inputvec=input_vec,
                                           anchors=anchors,
                                           transform=transforms.Compose([
                                                   MakeMat(input_vec, anchors),
                                                   ToTensor()
                                               ])
                                           )

    animalloader_valid = DataLoader(animal_dataset_valid_sm, batch_size=1, shuffle=False)

    layerlist = get_weights(weightspath)

    net = YoloNet(layerlist)
    net = net.to(device)
    net.load_state_dict(torch.load(save_path))
    net.eval()

    tptp = 0
    fpfp = 0
    fnfn = 0
    i = 0

    # define values for calculating loss
    input_shape = (1248, 1856, 3)
    anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
                  [5.319540, 6.116692]]
    # anchors_in = [[0.718750, 0.890625], [0.750000, 0.515625], [0.468750, 0.562500], [1.140625, 1.156250],
    #               [0.437500, 0.328125]]
    #anchors_in = np.divide(anchors_in, 2.0)
    anchors_in = np.array(anchors_in)
    box_size = [32, 32]
    anchor_pixel = np.multiply(anchors_in, box_size)
    img_x_pix = input_shape[1]
    img_y_pix = input_shape[0]
    boxs_x = np.ceil(img_x_pix / 32)
    boxs_y = np.ceil(img_y_pix / 32)
    # simplify net
    # boxs_x = np.ceil(img_x_pix / 16)
    # boxs_y = np.ceil(img_y_pix / 16)
    classes_in = 1
    lambda_c = 5.0
    lambda_no = 0.5
    batch_size = 8
    threshold = 0.5
    dict_deets = {'boxs_x': boxs_x, 'boxs_y': boxs_y, 'img_x_pix': img_x_pix, 'img_y_pix': img_y_pix,
                  'anchors': anchors_in, 'n_classes': classes_in, 'lambda_coord': lambda_c, 'lambda_noobj': lambda_no,
                  'base_dir': files_location_valid, 'batch_size': batch_size, 'threshold': threshold}
    boxes_out_all = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class'])
    scores_out_all = []
    classes_out_all = []

    tottrue = 0
    tottps = 0


    for data in animalloader_valid:
        #print(i)
        # if i==1000:
        #     break
        images = data["image"]
        images = images.to(device)
        bndbxs = data["bndbxs"]
        # bndbxs = bndbxs.to(device)
        y_true = data["y_true"]
        y_true = y_true.to(device)
        filen = data["name"]
        # print("epoch", epoch, "batch", i)
        y_pred = net(images)
        pboxes, tottr, tottp = accuracyiou(y_pred, bndbxs, filen, anchors_in, 0.3, 0.1)
        tottrue += tottr
        tottps += tottp
        # print(tottrue, tottps)
        tptp += np.sum(pboxes.tp)
        fpfp += pboxes.shape[0] - np.sum(pboxes.tp)
        # tptp += accz[0].data.item()
        # fpfp += accz[1].data.item()
        # fnfn += accz[2].data.item()
        y_pred_np = y_pred.data.cpu().numpy()
        output_img = yolo_output_to_box(y_pred_np, filen, dict_deets)
        boxes_out_all = boxes_out_all.append(output_img[0], ignore_index=True)
        i += 1

    print("epoch", xx, "TP", tottps, "FP", fpfp, "FN", (tottrue - tottps), "Recall", np.round(tottps / tottrue, 3), "FPPI", np.round(fpfp / 131, 2))
    output_path = save_dir + "boxes_out" + str(xx) + ".csv"
    boxes_out_all.to_csv(output_path)

    #print(boxes_out_all.shape[0])
    #print("Recall", tptp / 399)
    #print("FPPI", boxes_out_all.shape[0] / 131)






