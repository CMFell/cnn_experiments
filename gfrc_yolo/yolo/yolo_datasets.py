import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

class TileImageTestDataset(Dataset):

    def __init__(self, tiles_list):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.tiles_list = tiles_list
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        image = self.tiles_list[idx]
        image = self.transform(image)

        return image


class TileImageTestDatasetMeta(Dataset):

    def __init__(self, tiles_list, metacolumn, imgname, basedir):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        
        image_data = pd.read_csv(basedir / 'preds_for_cnn.csv')
        new_channel = np.ones((tiles_list[0].shape[0], tiles_list[0].shape[1], len(metacolumn)))
        matched_row = image_data[image_data.image_name == imgname]
        fillval = matched_row[metacolumn]
        fillvals = np.expand_dims(fillval, axis=0)
        new_channel = np.multiply(new_channel, fillvals)  
        new_tiles_list = []
        for tl in tiles_list:
            tl = np.divide(tl, 255)
            tl = np.dstack((tl, new_channel))
            tl = np.array(tl, dtype=np.float32)
            new_tiles_list.append(tl)
        self.tiles_list = new_tiles_list
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        image = self.tiles_list[idx]
        image = self.transform(image)

        return image

