import torch
from trch_yolonet import YoloNet, YoloNetSimp, YoloNetOrig
from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_accuracy import accuracy, calc_iou_centwh, accuracyiou
from trch_weights import get_weights
from trch_valid_utils import simple_nms, softmax, yolo_output_to_box_vec
import numpy as np
import pandas as pd
from scipy.special import expit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### VEDAI
img_w = 1024
img_h = 1024
max_annotations = 19
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
           [5.319540, 6.116692]]
valid_imgs = 242
# Multi
files_location_valid = "/data/old_home_dir/ChrissyF/VEDAI/yolo_multi_valid/"
save_dir = "/home/cmf21/pytorch_save/VEDAI/Multi/"
# nn = 42 # colour
nn = 289 # bw
nclazz = 9
# Bin
# files_location_valid = "/data/old_home_dir/ChrissyF/VEDAI/yolo_bin_valid/"
# save_dir = "/home/cmf21/pytorch_save/VEDAI/Bin/Baseline/"
# nn = 38
# nclazz = 1
# anchors = [[0.718750, 0.890625], [0.750000, 0.515625], [0.468750, 0.562500], [1.140625, 1.156250],
#            [0.437500, 0.328125]]

### INRIA
# img_w = 640
# img_h = 480
# max_annotations = 20
# valid_imgs = 741
# anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
#            [5.319540, 6.116692]]
# anchors = [[0.718750, 0.890625], [0.750000, 0.515625], [0.468750, 0.562500], [1.140625, 1.156250],
#            [0.437500, 0.328125]]
# files_location_valid = "/data/old_home_dir/ChrissyF/INRIA/yolo_valid/"
# save_dir = "/home/cmf21/pytorch_save/INRIA/"
# nclazz = 1
# nn = 43 # color
# nn = 50 # bw

### GFRC
# img_w = 1856
# img_h = 1248
# max_annotations = 14
# anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
#            [5.319540, 6.116692]]
# valid_imgs = int(2096 / 16) # 131 whole images split into 16 so 2096 tiles
# anchors = [[0.718750, 0.890625], [0.750000, 0.515625], [0.468750, 0.562500], [1.140625, 1.156250],
#            [0.437500, 0.328125]]
# Multi
# files_location_valid = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_multi/"
# save_dir = "/home/cmf21/pytorch_save/GFRC/Multi/Baseline/"
# nclazz = 6
# nn = 92
# Bin
# files_location_valid = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_bin/"
# save_dir = "/home/cmf21/pytorch_save/GFRC/Bin/"
# nclazz = 1
# nn = 70

# Original net size
box_size = [32, 32]
# Simplified net size
# box_size = [16, 16]

# colour or greyscale
# channels_in = 3
# grey_tf = False
channels_in = 1
grey_tf = True

### continue
weightspath = "/data/old_home_dir/ChrissyF/GFRC/yolov2.weights"
save_name = "testing_save_"
nms_threshold_out = 0.6
conf_threshold_summary = 0.3
iou_threshold_summary = 0.1
conf_threshold_out = 0.05
n_box = 5
colour_yn=False

grid_w = int(img_w / box_size[1])
grid_h = int(img_h / box_size[0])
save_path = save_dir + save_name + str(nn) + ".pt"
out_len = 5 + nclazz
fin_size = n_box * out_len
input_vec = [grid_w, grid_h, n_box, out_len]
anchors = np.array(anchors)
input_shape = (img_h, img_w, 3)

animal_dataset_valid = AnimalBoundBoxDataset(root_dir=files_location_valid,
                                                inputvec=input_vec,
                                                anchors=anchors,
                                                maxann=max_annotations,
                                                transform=transforms.Compose([
                                                    MakeMat(input_vec, anchors),
                                                    ToTensor()
                                                    ]),
                                                gray=grey_tf
                                                )

animalloader_valid = DataLoader(animal_dataset_valid, batch_size=1, shuffle=False)

layerlist = get_weights(weightspath)

net = YoloNetOrig(layerlist, fin_size, channels_in)
net = net.to(device)
net.load_state_dict(torch.load(save_path))
net.eval()

tptp = 0
fpfp = 0
fnfn = 0
i = 0

# define values for calculating loss
boxes_out_all = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class', 'tp'])
scores_out_all = []
classes_out_all = []

tottrue = 0
tottps = 0

for data in animalloader_valid:
    print(i)
    images = data["image"]
    images = images.to(device)
    bndbxs = data["bndbxs"]
    bndbxs_np = bndbxs.cpu().numpy()
    bndbxs_pd = pd.DataFrame(data=bndbxs_np[0,:,:], columns=['class', 'xc', 'yc', 'wid', 'hei'])
    y_true = data["y_true"]
    y_true = y_true.to(device)
    filen = data["name"]
    y_pred = net(images)
    pboxes, tottr, tottp = accuracyiou(y_pred, bndbxs, filen, anchors, conf_threshold_summary, iou_threshold_summary)
    tottrue += tottr
    tottps += tottp
    tptp += np.sum(pboxes.tp)
    fpfp += pboxes.shape[0] - np.sum(pboxes.tp)
    y_pred_np = y_pred.data.cpu().numpy()
    output_img = yolo_output_to_box_vec(y_pred_np, filen, anchors, bndbxs_pd, nms_threshold_out, conf_threshold_out)
    boxes_out_all = boxes_out_all.append(output_img, ignore_index=True)
    i += 1

print(nn, tottps, tottrue, fpfp, valid_imgs)
print("epoch", nn, "TP", tottps, "FP", fpfp, "FN", (tottrue - tottps), "Recall", tottps / tottrue, "FPPI", fpfp / valid_imgs)
output_path = save_dir + "boxes_out" + str(nn) + "_full.csv"
boxes_out_all.to_csv(output_path, index=False)

