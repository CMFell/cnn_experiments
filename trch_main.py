from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from trch_yolonet import YoloNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from trch_loss import YoloLoss
from torch import optim
import torch
import numpy as np
from trch_accuracy import accuracy

files_location = "C:/Users/kryzi/OneDrive - University of St Andrews/Transfer/train_img/"
grid_w = int(480 / 32)
grid_h = int(320 / 32)
n_box = 5
out_len = 6
input_vec = [grid_w, grid_h, n_box, out_len]
anchors_in = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889],
              [5.319540, 6.116692]]
anchors_in = torch.from_numpy(np.array(anchors_in)).type(torch.FloatTensor)
scalez = [1, 0.5, 1, 1]
cell_x = np.reshape(np.tile(np.arange(grid_w), grid_h), (1, grid_h, grid_w, 1))
cell_y = np.reshape(np.repeat(np.arange(grid_h), grid_w), (1, grid_h, grid_w, 1))
# combine to give grid
cell_grid = np.tile(np.stack([cell_x, cell_y], -1), [1, 1, 1, 5, 1])
cell_grid = torch.from_numpy(cell_grid).type(torch.FloatTensor)

animal_dataset = AnimalBoundBoxDataset(root_dir=files_location,
                                       inputvec=input_vec,
                                       anchors=anchors_in,
                                       transform=transforms.Compose([
                                               MakeMat(input_vec, anchors_in),
                                               ToTensor()
                                           ])
                                       )

animalloader = DataLoader(animal_dataset, batch_size=4, shuffle=True)

net = YoloNet()

opt = optim.SGD(net.parameters(), lr=0.00001)
i = 0

# for sampe in range(len(animalloader)):
# for i, data in enumerate(animalloader):
for data in animalloader:
    images = data["image"]
    bndbxs = data["bndbxs"]
    ytrue = data["y_true"]

    ypred = net(images)

    criterion = YoloLoss()
    loss = criterion(ypred, bndbxs, ytrue, anchors_in, 0.3, scalez, cell_grid)
    loss.backward()

    opt.step()
    opt.zero_grad()

    accz = accuracy(ypred, ytrue, 0.3)
    #if i == 0:
        #print(loss)
        #break

    i = i + 1
    print(i)
    print(loss)
    print(accz)
    # get the inputs; data is a list of [inputs, labels]
    #inputs, labels = animalloader
    #output = net(inputs)


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
    


