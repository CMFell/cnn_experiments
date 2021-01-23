from abc import ABC

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, vgg16
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from window.models.window_dataset import WindowTestDataset


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


# predict on windows
class BinaryVggClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = vgg16(num_classes=2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux1 = self.model(x)
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
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)


class AniWindowModel(ABC):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cp_path = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2_post/patch_model/checkpoint.ckpt.ckpt"
        classifier = BinaryWindowClassifier.load_from_checkpoint(checkpoint_path=cp_path)
        classifier.eval()
        classifier.to(self.device)
        self.classifier = classifier

    def inference_on_windows(self, windows_list, windows_whole_im):  
        window_dataset = WindowTestDataset(windows_list)
        batchsize = min(len(window_dataset), 32)
        windowloader = DataLoader(window_dataset, batch_size=batchsize, shuffle=False)

        preds_df = pd.DataFrame(columns=['animal', 'not_animal'])

        for batch in windowloader:
            batch = batch.to(self.device)
            output = self.classifier(batch)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().detach().numpy()
            pred_df = pd.DataFrame(pred_prob, columns=['animal', 'not_animal'])
            preds_df = pd.concat((preds_df, pred_df), axis=0, sort=False)

        # save list of positive windows
        preds_df = preds_df.reset_index(drop=True)
        windows_with_preds = pd.concat((windows_whole_im, preds_df), axis=1, sort=False)
        windows_filter_out = windows_with_preds[windows_with_preds.animal > 0.5]

        return windows_filter_out
