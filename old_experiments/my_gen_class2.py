import math
import numpy as np
import pandas as pd
import keras
import cv2

""" code taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """


class YoloGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, input_array, input_dir, anchors_in, gen_dict, shuffle=True):
        """Initialization"""
        self.dim = gen_dict['dim']
        self.batch_size = int(gen_dict['batch_size'])
        self.input_dir = input_dir
        self.input_array = input_array
        self.n_train = self.input_array.shape[0]
        self.n_channels = gen_dict['n_channels']
        self.size_reduce = gen_dict['size_reduce']
        self.anchors = anchors_in
        self.nanchors = int(self.anchors.shape[0])
        self.nclasses = int(gen_dict['n_classes'])
        self.lenout = 5 + self.nclasses
        self.indexes = np.arange(self.n_train)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_train / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        # print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        xx, yy = self.__data_generation(indexes)
        return xx, yy

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.n_train)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        xx = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        boxx = int(np.ceil(self.dim[1] / self.size_reduce[1]))
        boxy = int(np.ceil(self.dim[0] / self.size_reduce[0]))
        tot_len = int(self.nanchors * (5 + self.nclasses))
        yy = np.zeros((self.batch_size, boxy, boxx, tot_len))
        anks = np.multiply(self.anchors, self.size_reduce)
        pix_xy = [self.dim[1], self.dim[0]]
        anks = np.divide(anks, pix_xy)
        halfanks = np.divide(anks, 2)
        xmin = np.reshape(np.multiply(halfanks[:, 0], -1), (self.nanchors, 1))
        xmax = np.reshape(halfanks[:, 0], (self.nanchors, 1))
        ymin = np.reshape(np.multiply(halfanks[:, 1], -1), (self.nanchors, 1))
        ymax = np.reshape(halfanks[:, 1], (self.nanchors, 1))
        minz = np.hstack((xmin, ymin))
        maxz = np.hstack((xmax, ymax))
        ank_area = np.multiply(anks[:, 0], anks[:, 1])
        # Generate data
        for ii in range(self.batch_size):
            row_index = indexes[ii]
            # store training images
            xx_path = self.input_dir + self.input_array[row_index, 0]
            if self.n_channels == 1:
                imread_val = 0
            else:
                imread_val = self.n_channels
            xx[ii, ] = np.divide(cv2.imread(xx_path, imread_val), 255)

            # store training ground truth data
            yy_path = self.input_dir + self.input_array[row_index, 1]
            # print(yy_path)
            yy_in = pd.read_csv(yy_path, delimiter=' ', header=None)
            yy_in = np.array(yy_in)
            # this means that if there are two ground truths in one cell only last gets saved.
            # need to rework so can save up to five
            for gt in range(yy_in.shape[0]):
                class_in = int(yy_in[gt, 0])
                cent_in = yy_in[gt, 1:3]
                size_in = yy_in[gt, 3:5]
                cellx = int(np.floor(cent_in[0] * boxx))
                celly = int(np.floor(cent_in[1] * boxy))
                # Going to have out put as x and y pos in image
                # centx = cent_in[0] * boxx - cellx
                # centy = cent_in[1] * boxy - celly
                centx = cent_in[0]
                centy = cent_in[1]
                # size is supposed to be relative to image but that doesn't work for different size images
                # give size as relative to box
                sizex = size_in[0] * boxx
                sizey = size_in[1] * boxy
                # calculate using whole image size for iou so same as anchors
                gt_area = size_in[0] * size_in[1]
                gt_xmin = size_in[0] / -2
                gt_xmax = size_in[0] / 2
                gt_ymin = size_in[1] / -2
                gt_ymax = size_in[1] / 2
                gt_min = [gt_xmin, gt_ymin]
                gt_max = [gt_xmax, gt_ymax]
                class_oh = np.zeros(self.nclasses)
                class_oh[class_in] = 1
                inter_maxs = np.minimum(maxz, gt_max)
                inter_mins = np.maximum(minz, gt_min)
                inter_size = np.maximum(np.subtract(inter_maxs, inter_mins), 0)
                inter_area = np.multiply(inter_size[:, 0], inter_size[:, 1])
                union_area = np.subtract(np.add(ank_area, gt_area), inter_area)
                iou = np.divide(inter_area, union_area)
                iou_idx = np.argmax(iou, axis=-1)
                # only line changed over original at mo
                out_vec = np.array([centx, centy, sizex, sizey, 1])
                out_vec = np.hstack((out_vec, class_oh))
                out_vec = np.tile(out_vec, self.nanchors)
                nn = self.lenout * iou_idx + 4
                out_vec[nn] = 1
                yy[ii, celly, cellx, :] = out_vec

        return xx, yy
