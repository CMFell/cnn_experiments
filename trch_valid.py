import torch
from trch_yolonet import YoloNet
from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_accuracy import accuracy
from trch_weights import get_weights
import numpy as np
import pandas as pd
from scipy.special import expit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    rowz = np.divide(np.arange(boxsy), boxsy)
    colz = np.divide(np.arange(boxsx), boxsx)
    rowno = np.reshape(np.repeat(np.repeat(rowz, boxsx * nanchors), n_bat), (n_bat, boxsy, boxsx, nanchors))
    colno = np.reshape(np.repeat(np.tile(np.repeat(colz, nanchors), boxsy), n_bat), (n_bat, boxsy, boxsx, nanchors))
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
        for yc in range(boxsy):
            for xc in range(boxsx):
                for ab in range(nanchors):
                    #print(confs_cnn[img, yc, xc, ab])
                    if confs_cnn[img, yc, xc, ab] > thresh:
                        scores_out.append(confs_cnn[img, yc, xc, ab])
                        class_out = np.argmax(class_cnn[img, yc, xc, ab, :])
                        classes_out.append(class_out)
                        boxes_out.loc[len(boxes_out)] = [cent_cnn[img, yc, xc, ab, 0], cent_cnn[img, yc, xc, ab, 1],
                                                         size_cnn[img, yc, xc, ab, 0], size_cnn[img, yc, xc, ab, 1],
                                                         filen, confs_cnn[img, yc, xc, ab], class_out]

    output = [boxes_out, scores_out, classes_out]

    return output



files_location_valid = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_valid384/"
weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"

save_dir = "E:/CF_Calcs/BenchmarkSets/GFRC/pytorch_save/"
save_name = "testing_save_"
save_path = save_dir + save_name + str(35) + ".pt"

grid_w = int(576 / 32)
grid_h = int(384 / 32)
n_box = 5
out_len = 6
input_vec = [grid_w, grid_h, n_box, out_len]
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]

animal_dataset_valid_sm = AnimalBoundBoxDataset(root_dir=files_location_valid,
                                       inputvec=input_vec,
                                       anchors=anchors,
                                       transform=transforms.Compose([
                                               MakeMat(input_vec, anchors),
                                               ToTensor()
                                           ])
                                       )

animalloader_valid = DataLoader(animal_dataset_valid_sm, batch_size=8, shuffle=False)

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
input_shape = (384, 576, 3)
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = np.array(anchors_in)
box_size = [32, 32]
anchor_pixel = np.multiply(anchors_in, box_size)
img_x_pix = input_shape[1]
img_y_pix = input_shape[0]
boxs_x = np.ceil(img_x_pix / 32)
boxs_y = np.ceil(img_y_pix / 32)
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


for data in animalloader_valid:
    print(i)
    # if i==200:
    #     break
    images = data["image"]
    images = images.to(device)
    bndbxs = data["bndbxs"]
    bndbxs = bndbxs.to(device)
    y_true = data["y_true"]
    y_true = y_true.to(device)
    filen = data["name"]
    # print("epoch", epoch, "batch", i)
    y_pred = net(images)
    accz = accuracy(y_pred, y_true, 0.3)
    tptp += accz[0].data.item()
    fpfp += accz[1].data.item()
    fnfn += accz[2].data.item()
    y_pred_np = y_pred.data.cpu().numpy()
    output_img = yolo_output_to_box(y_pred_np, filen, dict_deets)
    boxes_out_all = boxes_out_all.append(output_img[0], ignore_index=True)
    i += 1

print(tptp, fpfp, fnfn)
output_path = save_dir + "boxes_out35.csv"
boxes_out_all.to_csv(output_path)








