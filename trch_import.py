import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class AnimalBoundBoxDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.files_list = os.listdir(self.root_dir)
        for ff in range(len(self.files_list)):
            self.files_list[ff] = self.files_list[ff][:-4]
        self.files_list = np.unique(self.files_list)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + self.files_list[idx] + ".png"
        image = io.imread(img_name)
        image = np.divide(image, 255.0)
        bndbxs_name = self.root_dir + self.files_list[idx] + ".txt"
        bndbxs = pd.read_csv(bndbxs_name, sep=' ', header=None)
        bndbxs = bndbxs.astype('float')
        bndbx_pad = pd.DataFrame(np.empty((14-bndbxs.shape[0], 5)))
        bndbxs = pd.concat([bndbxs, bndbx_pad])
        sample = {'image': image, 'bndbxs': bndbxs}

        if self.transform:
            sample = self.transform(sample)

        return sample



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bndbxs = sample['image'], sample['bndbxs']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # bndbxs are already as percentage of image so doesn't change with scaling

        return {'image': img, 'bndbxs': bndbxs}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bndbxs = sample['image'], sample['bndbxs']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        # calculate position of box in original picture in pixels
        xcent = bndbxs.iloc[:, 1] * w
        ycent = bndbxs.iloc[:, 2] * h
        wid = bndbxs.iloc[:, 3] * w
        hei = bndbxs.iloc[:, 4] * h
        xmin = xcent - wid / 2
        xmax = xcent + wid / 2
        ymin = ycent - hei / 2
        ymax = ycent + hei / 2
        # find limits within new crop
        new_xmin = xmin - left
        new_xmax = xmax - left
        new_ymin = ymin - top
        new_ymax = ymax - top
        new_xmin = np.maximum(new_xmin, 0)
        new_xmax = np.minimum(new_xmax, new_w)
        new_ymin = np.maximum(new_ymin, 0)
        new_ymax = np.minimum(new_ymax, new_h)
        # check if box is in new crop
        chk_x1 = np.greater(new_xmax, 0)
        chk_y1 = np.greater(new_ymax, 0)
        chk_x2 = np.less(new_xmin, new_w)
        chk_y2 = np.less(new_ymin, new_h)
        chk = np.logical_and(np.logical_and(np.logical_and(chk_x1, chk_y1), chk_x2), chk_y2)
        chk = chk.values
        # adjust to new position
        new_xcent = (new_xmin + new_xmax) / 2
        new_ycent = (new_ymin + new_ymax) / 2
        new_wid = new_xmax - new_xmin
        new_hei = new_ymax - new_ymin
        # scale and store
        new_xcent = new_xcent / new_w
        new_ycent = new_ycent / new_h
        # height and width scaled
        new_wid = new_wid / new_w
        new_hei = new_hei / new_h

        clazz = bndbxs.iloc[:, 0]
        new_bndbxs = np.hstack((clazz, new_xcent, new_ycent, new_wid, new_hei))
        new_bndbxs = np.reshape(new_bndbxs, (-1, 5), order='F')
        new_bndbxs = pd.DataFrame(new_bndbxs)
        # get rid of rows where boxes are out of the image
        new_bndbxs = new_bndbxs[chk]

        return {'image': image, 'bndbxs': new_bndbxs}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bndbxs = sample['image'], sample['bndbxs']
        bndbxs = bndbxs.values
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        bndbxs = torch.from_numpy(bndbxs).type(torch.FloatTensor)
        output = {'image': image, 'bndbxs': bndbxs}
        return output

