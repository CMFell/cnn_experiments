from trch_import import AnimalBoundBoxDataset, ToTensor
from trch_yolonet import YoloNet
from torch.utils.data import Dataset, DataLoader

files_location = "C:/Users/kryzi/OneDrive - University of St Andrews/Transfer/train_img/"

animal_dataset = AnimalBoundBoxDataset(root_dir=files_location, transform=ToTensor())

animalloader = DataLoader(animal_dataset, batch_size=4, shuffle=True)

net = YoloNet()

# for sampe in range(len(animalloader)):
for i, data in enumerate(animalloader):
    images = data["image"]
    bndbxs = data["bndbxs"]
    print(images.size())
    print(images)
    outputs = net(images)
    if i == 0:
        print(outputs)
        break
    # get the inputs; data is a list of [inputs, labels]
    #inputs, labels = animalloader
    #output = net(inputs)



    


