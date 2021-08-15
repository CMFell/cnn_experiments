import json
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from yolo.yolo_import import AnimalBoundBoxDataset, AnimalBoundBoxMetaDataset, ToTensor, MakeMat
from yolo.yolo_net import YoloNet, YoloNetSimp, YoloNetOrig, YoloNetMeta
from yolo.yolo_loss import YoloLoss
from yolo.yolo_accuracy import accuracy
from yolo.yolo_weights import get_weights

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_to_use = 'GFRC'
bin_yn = True
grey_tf = False
orig_size = True 
name_out = 'rgb_bin'
basedir = 'E:/GFRC_data/'
save_dir = basedir + 'output/' name_out + "/"
Path(save_dir).mkdir(parents=True, exist_ok=True)
nepochs = 200
# change to restart from an existing epoch
restartno = -1

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

# GFRC values
img_w = 1856
img_h = 1248
n_img = 6414
max_annotations = 14
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], 
            [5.319540, 6.116692]]
if bin_yn:
    # GFRC Binary values
    files_location = basedir + 'yolo_train_1248_bin/'
    nclazz = 1
else:
    # GFRC Multi values
    files_location = basedir + 'yolo_train_1248_multi/'
    nclazz = 6


if orig_size:
    # Original net size
    box_size = [32, 32]
else:
    # Simplified net size
    box_size = [16, 16]

weightspath = basedir + "weights/yolov2.weights"
save_name = "testing_save_"

conf_threshold_summary = 0.25
no_obj_threshold = 0.25
lambda_c = 5.0
lambda_no = 0.5
lambda_cl = 1
lambda_cf = 1
n_box = 5
bat_sz = 2
learn_rate = 0.0001
moment = 0.9
weight_d = 0.0005

save_dict = {'dataset_to_use': dataset_to_use, 'bin_yn': bin_yn, 'orig_size': orig_size, 'name_out': name_out,
    'img_w': img_w, 'img_h': img_h, 'n_img': n_img, 'max_annotations': max_annotations, 
    'anchors': anchors, 'files_location': files_location, 'save_dir': save_dir, 'nclazz': nclazz, 
    'box_size': box_size, 'channels_in': channels_in, 'grey_tf': grey_tf, 'weightspath': weightspath,
    'conf_threshold_summary': conf_threshold_summary, 'no_obj_threshold': no_obj_threshold,
    'lambda_c': lambda_c, 'lambda_no': lambda_no, 'lambda_cl': lambda_cl, 'lambda_cf': lambda_cf, 'n_box': n_box,
    'bat_sz': bat_sz, 'learn_rate': learn_rate, 'moment': moment, 'weight_d': weight_d}

with open(save_dir + "settings.json", 'w', encoding='utf-8') as f:
    json.dump(save_dict, f, ensure_ascii=False, indent=4)

grid_w = int(img_w / box_size[1])
grid_h = int(img_h / box_size[0])
out_len = 5 + nclazz
fin_size = n_box * out_len
input_vec = [grid_w, grid_h, n_box, out_len]

anchors_in = torch.from_numpy(np.array(anchors)).type(torch.FloatTensor)
anchors_in = anchors_in.to(device)
scalez = [lambda_c, lambda_no, lambda_cl, lambda_cf]
#scalez = scalez.to(device)
cell_x = np.reshape(np.tile(np.arange(grid_w), grid_h), (1, grid_h, grid_w, 1))
cell_y = np.reshape(np.repeat(np.arange(grid_h), grid_w), (1, grid_h, grid_w, 1))
# combine to give grid
cell_grid = np.tile(np.stack([cell_x, cell_y], -1), [1, 1, 1, 5, 1])
cell_grid = torch.from_numpy(cell_grid).type(torch.FloatTensor)
cell_grid = cell_grid.to(device)

if use_meta:
    animal_dataset = AnimalBoundBoxMetaDataset(root_dir=files_location,
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
    animal_dataset = AnimalBoundBoxDataset(root_dir=files_location,
                                        inputvec=input_vec,
                                        anchors=anchors,
                                        maxann=max_annotations,
                                        transform=transforms.Compose([
                                                MakeMat(input_vec, anchors),
                                                ToTensor()
                                            ]),
                                            gray=grey_tf
                                        )


animalloader = DataLoader(animal_dataset, batch_size=bat_sz, shuffle=True)

layerlist = get_weights(weightspath)

if metadata_end:
    net = YoloNetMeta(layerlist, fin_size, channels_in)
else:
    net = YoloNetOrig(layerlist, fin_size, channels_in)

net = net.to(device)
if epochstart > 0:
    save_path = save_dir + save_name + str(restartno) + ".pt"
net.load_state_dict(torch.load(save_path))

opt = optim.SGD(net.parameters(), lr=learn_rate, momentum=moment, weight_decay=weight_d)
i = 0
epochstart = restartno + 1

for epoch in range(epochstart, nepochs):
    tptp = 0
    fpfp = 0
    fnfn = 0
    for data in animalloader:
        tot_bat = epoch * n_img / bat_sz  + i
        images = data["image"]
        images = images.to(device)
        bndbxs = data["bndbxs"]
        bndbxs = bndbxs.to(device)
        ytrue = data["y_true"]
        ytrue = ytrue.to(device)
        
        ypred = net(images)

        criterion = YoloLoss()
        loss = criterion(ypred, bndbxs, ytrue, anchors_in, scalez, cell_grid, tot_bat, no_obj_threshold)
        loss.backward()

        if (i + 1) % 32 == 0:
            # every 2 iterations of batches of size 32
            opt.step()
            opt.zero_grad()

        accz = accuracy(ypred, ytrue, conf_threshold_summary)
        tptp += accz[0].data.item()
        fpfp += accz[1].data.item()
        fnfn += accz[2].data.item()

        i = i + 1

    print("epoch", epoch, tptp, fpfp, fnfn)
    save_path = save_dir + save_name + str(epoch) + ".pt"
    torch.save(net.cpu().state_dict(), save_path)
    net = net.to(device)

    i = 0
