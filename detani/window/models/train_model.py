import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def train_model(experiment_savedir, transform, batch_size, classifier):
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
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=2, accelerator="ddp", max_epochs=100, 
                        logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)
