from abc import ABC

import numpy as np 
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from window.models.window_dataset import WindowTestDataset


class AniWindowModel(ABC):
    def __init__(self, classifier):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


class AniWindowModelTrain(ABC):
    def __init__(self):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_windows = "/data/old_home_dir/ChrissyF/GFRC/window_classifier/bin_class/"
        self.valid_windows = "/data/old_home_dir/ChrissyF/GFRC/window_classifier/valid_bin_class/"


    def train_model(self, experiment_savedir, transform, batch_size, classifier):
        train_set = ImageFolder(self.train_windows, transform=transform)
        valid_set = ImageFolder(self.valid_windows, transform=transform)
        print(len(train_set), len(valid_set))

        # create dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
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
        trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=2, accelerator="dp", max_epochs=100, 
                             logger=csv_logger, log_every_n_steps=11)
        trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)
