from trch_import import AnimalBoundBoxDataset, ToTensor, MakeMat
from trch_yolonet import YoloNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

files_location = "C:/Users/kryzi/OneDrive - University of St Andrews/Transfer/train_img/"
grid_w = int(480 / 32)
grid_h = int(320 / 32)
n_box = 5
out_len = 6
input_vec = [grid_w, grid_h, n_box, out_len]
anchors_in = [2.387088, 2.985595, 1.540179, 1.654902, 3.961755, 3.936809,
              2.681468, 1.803889, 5.319540, 6.116692]

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

# for sampe in range(len(animalloader)):
for i, data in enumerate(animalloader):
    images = data["image"]
    bndbxs = data["bndbxs"]
    ytrue = data["y_true"]
    print(ytrue.size())
    print(ytrue)
    outputs = net(images)
    if i == 0:
        print(outputs)
        break
    # get the inputs; data is a list of [inputs, labels]
    #inputs, labels = animalloader
    #output = net(inputs)



    


