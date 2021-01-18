import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import inception_v3

# retrain classifier from imagenet

class BinaryWindowClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = inception_v3(num_classes=2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux1 = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        ### TODO:  check how losses are weighted
        loss = loss1 + 0.4 * loss2
        self.log("train_loss", loss)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log("val_loss", loss)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("val_accuracy", accu)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]

experiment_savedir = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2_post/"

transform = Compose([
    Resize((299, 299)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# prepare our data
batch_size = 32
train_set = ImageFolder("/data/old_home_dir/ChrissyF/GFRC/window_classifier/bin_class/", transform=transform)
valid_set = ImageFolder("/data/old_home_dir/ChrissyF/GFRC/window_classifier/valid_bin_class/", transform=transform)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8)
valid_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8)

# configure logging and checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    dirpath=experiment_savedir + "patch_model/",
    filename=f"checkpoint.ckpt",
    save_top_k=1,
    mode="max",
)

early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00,
    patience=10,
    verbose=False,
    mode='max'
)

# create a logger
csv_logger = pl_loggers.CSVLogger(experiment_savedir + 'logs/', name='patch_classifier', version=0)

# train our model
classifier = BinaryWindowClassifier()
trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=2, accelerator="ddp", max_epochs=100, 
                    logger=csv_logger, log_every_n_steps=1)
trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)
