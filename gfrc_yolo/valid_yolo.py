import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from yolo.yolo_accuracy import accuracy, calc_iou_centwh, accuracyiou
from yolo.yolo_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from yolo.yolo_net import YoloNet, YoloNetSimp, YoloNetOrig
from yolo.yolo_valid_utils import simple_nms, softmax, yolo_output_to_box_vec
from yolo.yolo_weights import get_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset_to_use = 'GFRC'
bin_yn = True
grey_tf = False
orig_size = True 
name_out = 'rgb_bin'
basedir = 'E:/GFRC_data/'
save_dir = basedir + 'output/' name_out + "/"
nepochs = 200
epochstart = 0

# colour or greyscale
if grey_tf:
    channels_in = 1
else:
    channels_in = 3

# if using metadata
use_meta = False:
if use_meta:
    name_out = 'meta_ci_end'
    colz = ['lowerci', 'upperci']
    metadata_end = True
    channels_in = channels_in + len(colz)

### GFRC
img_w = 1856
img_h = 1248
max_annotations = 14
valid_imgs = 244
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
            [5.319540, 6.116692]]
if bin_yn:
    # Bin
    files_location = basedir + 'yolo_valid_1248_bin/'
    nclazz = 1
else:
    # Multi
    files_location = basedir + 'yolo_valid_1248_multi/'
    nclazz = 6

if orig_size:
    # Original net size
    box_size = [32, 32]
else:
    # Simplified net size
    box_size = [16, 16]

### continue
weightspath = basedir + "weights/yolov2.weights"
save_name = "testing_save_"
nms_threshold_out = 0.25
conf_threshold_summary = 0.25
iou_threshold_summary = 0.25
conf_threshold_out = 0.01
n_box = 5

grid_w = int(img_w / box_size[1])
grid_h = int(img_h / box_size[0])
out_len = 5 + nclazz
fin_size = n_box * out_len
input_vec = [grid_w, grid_h, n_box, out_len]
anchors = np.array(anchors)

for xx in range(epochstart, nepochs):
    # if xx % 10 == 0:
    #     pass
    # else:
    #    continue
    save_path = save_dir + save_name + str(xx) + ".pt" 
    print(save_path)

    if use_meta:
        animal_dataset_valid = AnimalBoundBoxMetaDataset(root_dir=files_location,
                                        inputvec=input_vec,
                                        anchors=anchors,
                                        maxann=max_annotations,
                                        metacolumn=colz,
                                        transform=transforms.Compose([
                                                MakeMat(input_vec, anchors),
                                                ToTensor()
                                            ]),
                                            gray=grey_tf,
                                            based=basedir
                                        )
    else:
        animal_dataset_valid = AnimalBoundBoxDataset(root_dir=files_location,
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

    boxes_out_all = pd.DataFrame(columns=['xc', 'yc', 'wid', 'hei', 'file', 'conf', 'class','tp'])
    scores_out_all = []
    classes_out_all = []

    tottrue = 0
    tottps = 0

    for data in animalloader_valid:
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
        if (tottp > tottr):
            print(tottr, tottp, np.sum(pboxes.tp))
        tottrue += tottr
        tottps += tottp
        #print("tt", tottrue, "tp", tottps)
        tptp += np.sum(pboxes.tp)
        fpfp += pboxes.shape[0] - np.sum(pboxes.tp)
        y_pred_np = y_pred.data.cpu().numpy()
        output_img = yolo_output_to_box_vec(y_pred_np, filen, anchors, bndbxs_pd, nms_threshold_out, conf_threshold_out)
        boxes_out_all = boxes_out_all.append(output_img, ignore_index=True)
        i += 1
    print("epoch", xx, "TP", tottps, "FP", fpfp, "FN", (tottrue - tottps), "Recall", np.round(tottps / tottrue, 3), "FPPI", np.round(fpfp / valid_imgs, 2))
    output_path = save_dir + "boxes_out" + str(xx) + ".csv"
    boxes_out_all.to_csv(output_path)

