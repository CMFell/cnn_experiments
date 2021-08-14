from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from window.models.window_classifier import BinaryVggClassifier
from window.models.train_model import AniWindowModelTrain


# retrain classifier from imagenet

experiment_savedir = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2_vgg/"

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# prepare our data
batch_size = 128

classifier = BinaryVggClassifier()

model = AniWindowModelTrain()
model.train_model(experiment_savedir, transform, batch_size, classifier)
