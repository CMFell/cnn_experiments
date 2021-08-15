import os
import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AnimalBoundBoxDataset(Dataset):

    def __init__(self, root_dir, inputvec, anchors, maxann, transform=None, gray=False):
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
        self.maxann = maxann
        for ff in range(len(self.files_list)):
            self.files_list[ff] = self.files_list[ff][:-4]
        self.files_list = np.unique(self.files_list)
        self.gray = gray

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + self.files_list[idx] + ".png"
        image = io.imread(img_name, as_gray=self.gray)
        image = np.divide(image, 255.0)
        bndbxs_name = self.root_dir + self.files_list[idx] + ".txt"
        bndbxs = pd.read_csv(bndbxs_name, sep=' ', header=None, names=['class', 'xc', 'yc', 'wid', 'hei'])
        bndbxs = bndbxs.astype('float')
        bndbx_pad = pd.DataFrame(np.zeros((self.maxann-bndbxs.shape[0], 5)), columns=['class', 'xc', 'yc', 'wid', 'hei'])
        bndbxs = pd.concat([bndbxs, bndbx_pad])

        sample = {'image': image, 'bndbxs': bndbxs, 'name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class AnimalBoundBoxMetaDataset(Dataset):

    def __init__(self, root_dir, inputvec, anchors, maxann, metacolumn, based, transform=None, gray=False):
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
        self.maxann = maxann
        for ff in range(len(self.files_list)):
            self.files_list[ff] = self.files_list[ff][:-4]
        self.files_list = np.unique(self.files_list)
        self.gray = gray
        self.image_data = pd.read_csv(based / 'preds_for_cnn.csv')
        self.metacolumn = metacolumn

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + self.files_list[idx] + ".png"
        img_name_split = self.files_list[idx].split("_")
        img_to_match = img_name_split[0] + "_" + img_name_split[1]
        #img_to_match = self.files_list[idx][:14]
        #print(img_to_match)
        #if img_to_match[-1] == '_':
        #    img_to_match = img_to_match[:13]
        #elif img_to_match[-2] == '_':
        #    img_to_match = img_to_match[:12]
        #elif img_to_match[-3] == '_':
        #    img_to_match = img_to_match[:11]
        #elif img_to_match[-4] == '_':
        #    img_to_match = img_to_match[:10]
        img_to_match = img_to_match + '.jpg'
        matched_row = self.image_data[self.image_data.image_name == img_to_match]
        image = io.imread(img_name, as_gray=self.gray)
        image = np.divide(image, 255.0)
        new_channel = np.ones((image.shape[0], image.shape[1], len(self.metacolumn)))
        fillval = matched_row[self.metacolumn]
        fillvals = np.expand_dims(fillval, axis=0)
        new_channel = np.multiply(new_channel, fillvals)
        image = np.dstack((image, new_channel))
        bndbxs_name = self.root_dir + self.files_list[idx] + ".txt"
        bndbxs = pd.read_csv(bndbxs_name, sep=' ', header=None, names=['class', 'xc', 'yc', 'wid', 'hei'])
        bndbxs = bndbxs.astype('float')
        bndbx_pad = pd.DataFrame(np.zeros((self.maxann-bndbxs.shape[0], 5)), columns=['class', 'xc', 'yc', 'wid', 'hei'])
        bndbxs = pd.concat([bndbxs, bndbx_pad])

        sample = {'image': image, 'bndbxs': bndbxs, 'name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MakeMat(object):

    def __init__(self, inputz, anchors):
        self.gridw = inputz[0]
        self.gridh = inputz[1]
        self.nbox = inputz[2]
        self.outlen = inputz[3]
        self.anchors = np.reshape(anchors, [self.nbox, 2])


    def __call__(self, sample):
        image, bndbxs, name = sample['image'], sample['bndbxs'], sample['name']
        y_true = np.zeros((self.gridh, self.gridw, self.nbox, self.outlen))  # desired network output
        for row in range(bndbxs.shape[0]):
            obj = bndbxs.iloc[row]
            if obj[3] > 0:
                xcell = np.int32(np.floor(obj[1] * self.gridw))
                ycell = np.int32(np.floor(obj[2] * self.gridh))
                centx = (obj[1] * self.gridw) - xcell
                centy = (obj[2] * self.gridh) - ycell
                # Calc best box compared to anchors, zero position both
                xmin_true = 0 - obj[3] / 2
                xmax_true = 0 + obj[3] / 2
                ymin_true = 0 - obj[4] / 2
                ymax_true = 0 + obj[4] / 2
                anchors_wi = np.divide(self.anchors, [self.gridw, self.gridh])
                anc_xmin = np.subtract(0, np.divide(anchors_wi[:, 0], 2))
                anc_xmax = np.add(0, np.divide(anchors_wi[:, 0], 2))
                anc_ymin = np.subtract(0, np.divide(anchors_wi[:, 1], 2))
                anc_ymax = np.add(0, np.divide(anchors_wi[:, 1], 2))
                interxmax = np.minimum(anc_xmax, xmax_true)
                interxmin = np.maximum(anc_xmin, xmin_true)
                interymax = np.minimum(anc_ymax, ymax_true)
                interymin = np.maximum(anc_ymin, ymin_true)
                sizex = np.maximum(np.subtract(interxmax, interxmin), 0)
                sizey = np.maximum(np.subtract(interymax, interymin), 0)
                inter_area = np.multiply(sizex, sizey)
                anc_area = np.multiply(anchors_wi[:, 0], anchors_wi[:, 1])
                truth_area = np.multiply(obj[3], obj[4])
                union_area = np.subtract(np.add(anc_area, truth_area), inter_area)
                iou = np.divide(inter_area, union_area)
                best_box = np.argmax(iou)
                out_vec = np.zeros(self.outlen)
                # I think this should be this
                out_vec[0:5] = [centx, centy, obj[3], obj[4], 1.0]
                # out_vec[0:5] = [obj[1], obj[2], obj[3], obj[4], 1]
                class_pos = np.int32(5 + obj[0])
                out_vec[class_pos] = 1.
                y_true[ycell, xcell, best_box, :] = out_vec

        return {'image': image, 'bndbxs': bndbxs, 'y_true': y_true, 'name': name}


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
        image, bndbxs, name = sample['image'], sample['bndbxs'], sample['name']

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

        return {'image': img, 'bndbxs': bndbxs, 'name': name}


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
        image, bndbxs, name = sample['image'], sample['bndbxs'], sample['name']

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

        return {'image': image, 'bndbxs': new_bndbxs, 'name': name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bndbxs, ytrue, name = sample['image'], sample['bndbxs'], sample['y_true'], sample['name']
        bndbxs = bndbxs.values
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        else:
            image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        bndbxs = torch.from_numpy(bndbxs).type(torch.FloatTensor)
        ytrue = torch.from_numpy(ytrue).type(torch.FloatTensor)
        output = {'image': image, 'bndbxs': bndbxs, 'y_true': ytrue, 'name': name}
        return output

"""
files_location = "E:/CF_Calcs/BenchmarkSets/GFRC/Other_train_sets/yolo_copy_train_img/"
grid_w = int(576 / 32)
grid_h = int(384 / 32)
n_box = 5
out_len = 6
input_vec = [grid_w, grid_h, n_box, out_len]
anchors = [[2.387088, 2.985595], [1.540179, 1.654902], [3.961755, 3.936809], [2.681468, 1.803889], [5.319540, 6.116692]]
animal_dataset = AnimalBoundBoxDataset(root_dir=files_location, inputvec=input_vec, anchors=anchors)
print(animal_dataset[0]['y_true'])

file1 = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train384_5pc/Z10_Img06910_4_0.txt"
file2 = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train384_5pc/Z10_Img06910_379_4.txt"
file3 = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_train384_5pc/Z10_Img06911_329_4.txt"

bndbxs = pd.read_csv(file3, sep=' ', header=None, names=['class', 'xc', 'yc', 'wid', 'hei'])
print(bndbxs)
bndbxs = bndbxs.astype('float')
print(bndbxs)
bndbx_pad = pd.DataFrame(np.zeros((14-bndbxs.shape[0], 5)), columns=['class', 'xc', 'yc', 'wid', 'hei'])
print(bndbx_pad)
bndbxs = pd.concat([bndbxs, bndbx_pad])
print(bndbxs)
"""


