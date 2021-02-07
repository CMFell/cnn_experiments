import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, vgg16
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


# predict on windows
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
    
    def forward(self, x):
        return self.model(x)


class BinaryVggClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        model = vgg16(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=2, bias=True)
        )
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)
