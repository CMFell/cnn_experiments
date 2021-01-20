import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torchvision.models import inception_v3
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from cnn_experiments.window_classifier.models.window_dataset import WindowTestDataset

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
        

def inference_on_windows(windows_list):  
    cp_path = "/home/cmf21/pytorch_save/GFRC/Bin/rgb_baseline2_post/patch_model/checkpoint.ckpt.ckpt"
    classifier = BinaryWindowClassifier.load_from_checkpoint(checkpoint_path=cp_path)
    classifier.eval()
    classifier.to(device)

    window_dataset = WindowTestDataset(windows_list)
    windowloader = DataLoader(window_dataset, batch_size=len(window_dataset), shuffle=False)

    for batch in windowloader:
        batch = batch.to(device)
        output = classifier(batch)
        sm = torch.nn.Softmax(1)
        output_sm = sm(output)
        pred_prob = output_sm.cpu().detach().numpy()

    # save list of positive windows
    preds_df = pd.DataFrame(pred_prob, columns=['animal', 'not_animal'])
    windows_with_preds = pd.concat((windows_whole_im, preds_df), axis=1)
    windows_filter_out = windows_with_preds[windows_with_preds.animal > 0.5]

    return windows_filter_out

