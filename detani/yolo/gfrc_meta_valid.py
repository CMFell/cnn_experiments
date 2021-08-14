import torch
from trch_yolonet import YoloNet, YoloNetSimp, YoloNetOrig, YoloNetMeta
from trch_import import AnimalBoundBoxMetaDataset, ToTensor, MakeMat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_accuracy import accuracy, calc_iou_centwh, accuracyiou
from trch_weights import get_weights
from trch_valid_utils import simple_nms, softmax, yolo_output_to_box_vec
import numpy as np
import pandas as pd
from scipy.special import expit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_to_use = 'GFRC'
bin_yn = True
grey_tf = False
orig_size = True 
name_out = 'meta_ci_end'
metadata_end = False

### GFRC
img_w = 1856
img_h = 1248
max_annotations = 14
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
            [5.319540, 6.116692]]
valid_imgs = 312
# valid_imgs = int(2096 / 16) # 131 whole images split into 16 so 2096 tiles
# anchors = [[0.718750, 0.890625], [0.750000, 0.515625], [0.468750, 0.562500], [1.140625, 1.156250],
#            [0.437500, 0.328125]]
if bin_yn:
        # Bin
    files_location_valid = "/data/old_home_dir/ChrissyF/GFRC/yolo_valid1248_bin/"
    save_dir = "/home/cmf21/pytorch_save/GFRC/Bin/" + name_out + "/"
    nclazz = 1
    if name_out == 'meta_prediction':
        nn = 132
        col_list = ['prediction']
    elif name_out == 'meta_prediction_norm':
        nn = 127
        col_list = ['prediction_norm']
    elif name_out == 'meta_prediction_end':
        metadata_end = True
        col_list = ['prediction']
        nn = 124
    elif name_out == 'meta_prediction_ci':
        metadata_end = False
        col_list = ['lowerci', 'upperci']
        nn = 139
    elif name_out == 'meta_pred_ci_spread':
        metadata_end = False
        col_list = ['prediction', 'lowerci', 'upperci', 'spread']
        nn = 136
    elif name_out == 'meta_ci_end':
        metadata_end = True
        col_list = ['lowerci', 'upperci']
        nn = 149

# colour or greyscale
if grey_tf:
    channels_in = 1
else:
    channels_in = 3
if orig_size:
    # Original net size
    box_size = [32, 32]
else:
    # Simplified net size
    box_size = [16, 16]

# add channel for metadata
channels_in = channels_in + len(col_list)
print("ci", channels_in)

### continue
weightspath = "/data/old_home_dir/ChrissyF/GFRC/yolov2.weights"
save_name = "testing_save_"
nms_threshold_out = 0.25
conf_threshold_summary = 0.25
iou_threshold_summary = 0.25
conf_threshold_out = 0.05
n_box = 5

grid_w = int(img_w / box_size[1])
grid_h = int(img_h / box_size[0])
save_path = save_dir + save_name + str(nn) + ".pt"
out_len = 5 + nclazz
fin_size = n_box * out_len
input_vec = [grid_w, grid_h, n_box, out_len]
anchors = np.array(anchors)

animal_dataset_valid = AnimalBoundBoxMetaDataset(root_dir=files_location_valid,
                                                inputvec=input_vec,
                                                anchors=anchors,
                                                maxann=max_annotations,
                                                metacolumn=col_list,
                                                transform=transforms.Compose([
                                                    MakeMat(input_vec, anchors),
                                                    ToTensor()
                                                    ]),
                                                gray=grey_tf
                                                )

animalloader_valid = DataLoader(animal_dataset_valid, batch_size=1, shuffle=False)

layerlist = get_weights(weightspath)

if metadata_end:
    net = YoloNetMeta(layerlist, fin_size, channels_in)
else:
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

