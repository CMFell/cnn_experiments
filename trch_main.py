from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from trch_yolonet import YoloNet, YoloNetSimp, YoloNetOrig
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_loss import YoloLoss
from torch import optim
import torch
import numpy as np
from trch_accuracy import accuracy
from trch_weights import get_weights
import json

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_to_use = 'VEDAI'
bin_yn = True
grey_tf = True 
orig_size = True 
name_out = 'grey_baseline'

if dataset_to_use == 'GFRC':
    # GFRC values
    img_w = 1856
    img_h = 1248
    n_img = 6414
    max_annotations = 14
    anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], 
               [5.319540, 6.116692]]
    if bin_yn:
        # GFRC Binary values
        files_location = "/data/old_home_dir/ChrissyF/GFRC/yolo_train1248_bin/"
        save_dir = "/home/cmf21/pytorch_save/GFRC/Bin/" + name_out + "/"
        nclazz = 1
    else:
        # GFRC Multi values
        files_location = "/data/old_home_dir/ChrissyF/GFRC/yolo_train1248_multi/"
        save_dir = "/home/cmf21/pytorch_save/GFRC/Multi/" + name_out + "/"
        nclazz = 6
elif dataset_to_use == 'INRIA':
    # INRIA values
    img_w = 640
    img_h = 480
    n_img = 1832
    max_annotations = 8
    anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], 
               [5.319540, 6.116692]]
    # Inria only has binary values
    files_location = "/data/old_home_dir/ChrissyF/INRIA/yolo_train/"
    save_dir = "/home/cmf21/pytorch_save/INRIA/" + name_out + "/"
    nclazz = 1
elif dataset_to_use == 'VEDAI':
    # VEDAI values
    img_w = 1024
    img_h = 1024
    n_img = 726
    max_annotations = 19
    anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], 
               [5.319540, 6.116692]]
    if bin_yn:
        # VEDAI binary values
        files_location = "/data/old_home_dir/ChrissyF/VEDAI/yolo_bin_train/"
        save_dir = "/home/cmf21/pytorch_save/VEDAI/Bin/" + name_out + "/"
        nclazz = 1
    else:
        # VEDAI multi values
        files_location = "/data/old_home_dir/ChrissyF/VEDAI/yolo_multi_train/"
        save_dir = "/home/cmf21/pytorch_save/VEDAI/Multi/" + name_out + "/"
        nclazz = 9
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

weightspath = "/data/old_home_dir/ChrissyF/GFRC/yolov2.weights"
save_name = "testing_save_"

conf_threshold_summary = 0.3
no_obj_threshold = 0.1
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
# animalloader_valid = DataLoader(animal_dataset_valid, batch_size=bat_sz, shuffle=False)

layerlist = get_weights(weightspath)

net = YoloNetOrig(layerlist, fin_size, channels_in)
net = net.to(device)
save_path = save_dir + save_name + str(199) + ".pt"
net.load_state_dict(torch.load(save_path))

opt = optim.SGD(net.parameters(), lr=learn_rate, momentum=moment, weight_decay=weight_d)
i = 0

# for sampe in range(len(animalloader)):
# for i, data in enumerate(animalloader):
for epoch in range(200, 300):
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
        # print("epoch", epoch, "batch", i)
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
        #if i == 0:
            #print(loss)
            #break

        i = i + 1
        # print("epoch", epoch, "batch", i, "TP", accz[0].data.item(), "FP", accz[1].data.item(), "FN", accz[2].data.item())
        # print(" ")
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = animalloader
        #output = net(inputs)
    print("epoch", epoch, tptp, fpfp, fnfn)

    """
    opt.zero_grad()

    net.eval()
    with torch.no_grad():
        tptptp = 0
        fpfpfp = 0
        fnfnfn = 0
        valid_loss = 0
        for data in animalloader_validsm:
            images = data["image"]
            images = images.to(device)
            bndbxs = data["bndbxs"]
            bndbxs = bndbxs.to(device)
            ytrue = data["y_true"]
            ytrue = ytrue.to(device)
            # print("epoch", epoch, "batch", i)
            ypred = net(images)

            criterion = YoloLoss()
            valid_loss += criterion(ypred, bndbxs, ytrue, anchors_in, 0.3, scalez, cell_grid, tot_bat)
            accz = accuracy(ypred, ytrue, 0.3)
            tptptp += accz[0].data.item()
            fpfpfp += accz[1].data.item()
            fnfnfn += accz[2].data.item()

    val_loss_out = valid_loss / len(animalloader_validsm)
    print("epoch valid", i, "TP valid", tptptp, "FP valid", fpfpfp, "FN valid", fnfnfn,
          "loss", round(val_loss_out.data.item(), 2))
    """

    i = 0

    save_path = save_dir + save_name + str(epoch) + ".pt"
    torch.save(net.state_dict(), save_path)



"""
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
"""
    


