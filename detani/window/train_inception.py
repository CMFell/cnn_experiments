from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from window.models.window_classifier import BinaryWindowClassifier
from window.models.train_model import AniWindowModelTrain


# retrain classifier from imagenet

experiment_savedir = "/home/cmf21/pytorch_save/GFRC/Bin/post_processed/pos_from_yolo/rgb_baseline2_incep/"

transform = Compose([
    Resize((299, 299)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# prepare our data
batch_size = 32

classifier = BinaryWindowClassifier()

model = AniWindowModelTrain()
model.train_model(experiment_savedir, transform, batch_size, classifier)
