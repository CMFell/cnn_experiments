from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from trch_yolonet import YoloNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_loss import YoloLoss
from torch import optim
import torch
import numpy as np
from trch_accuracy import accuracy
from trch_weights import get_weights

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# files_location = "C:/Users/kryzi/OneDrive - University of St Andrews/Transfer/train_img/"
files_location = "E:/CF_Calcs/BenchmarkSets/GFRC/Other_train_sets/yolo_copy_train_img/"
files_location_valid = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_valid384/"
files_location_valid_sm = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_valid384_subset/"
weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"
save_dir = "E:/CF_Calcs/BenchmarkSets/GFRC/pytorch_save/"
save_name = "testing_save_"
grid_w = int(576 / 32)
grid_h = int(384 / 32)
n_box = 5
out_len = 6
input_vec = [grid_w, grid_h, n_box, out_len]
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = torch.from_numpy(np.array(anchors)).type(torch.FloatTensor)
anchors_in = anchors_in.to(device)
scalez = [5, 0.1, 1, 1]
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
                                       transform=transforms.Compose([
                                               MakeMat(input_vec, anchors),
                                               ToTensor()
                                           ])
                                       )
#print(animal_dataset[0]['bndbxs'])
#print(animal_dataset[0]['y_true'])

animal_dataset_valid_sm = AnimalBoundBoxDataset(root_dir=files_location_valid_sm,
                                       inputvec=input_vec,
                                       anchors=anchors,
                                       transform=transforms.Compose([
                                               MakeMat(input_vec, anchors),
                                               ToTensor()
                                           ])
                                       )

animalloader = DataLoader(animal_dataset, batch_size=16, shuffle=True)
animalloader_validsm = DataLoader(animal_dataset_valid_sm, batch_size=32, shuffle=False)

layerlist = get_weights(weightspath)

net = YoloNet(layerlist)
net = net.to(device)

opt = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
i = 0

# for sampe in range(len(animalloader)):
# for i, data in enumerate(animalloader):
for epoch in range(50):
    tptp = 0
    fpfp = 0
    fnfn = 0
    for data in animalloader:
        tot_bat = epoch * 11812 / 16 + i
        print(tot_bat)
        images = data["image"]
        images = images.to(device)
        bndbxs = data["bndbxs"]
        bndbxs = bndbxs.to(device)
        ytrue = data["y_true"]
        ytrue = ytrue.to(device)
        # print("epoch", epoch, "batch", i)
        ypred = net(images)

        criterion = YoloLoss()
        loss = criterion(ypred, bndbxs, ytrue, anchors_in, 0.3, scalez, cell_grid, tot_bat)
        loss.backward()

        if (i + 1) % 4 == 0:
            # every 2 iterations of batches of size 32
            opt.step()
            opt.zero_grad()

        accz = accuracy(ypred, ytrue, 0.3)
        tptp += accz[0].data.item()
        fpfp += accz[1].data.item()
        fnfn += accz[2].data.item()
        #if i == 0:
            #print(loss)
            #break

        i = i + 1
        print("epoch", epoch, "batch", i, "TP", accz[0].data.item(), "FP", accz[1].data.item(), "FN", accz[2].data.item())
        print(" ")
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = animalloader
        #output = net(inputs)
    print(tptp, fpfp, fnfn)

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
    


