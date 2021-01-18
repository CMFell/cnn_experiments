# https://github.com/experiencor/keras-yolo2/blob/master/preprocessing.py
import numpy as np
import cv2

# into config needs to go
# batch size as BATCH_SIZE
# input image height as IMAGE_H
# input image width as IMAGE_W
# number of boxes to store per image as TRUE_BOX_BUFFER
# number of anchor boxes as BOX
# number of classes as N_CLASSES
# output boxes height as GRID_H
# output boxes width as GRID_W


class BatchGenerator():
    def __init__(self, paths, config, shuffle=True):
        self.generator = None
        self.paths = paths
        self.config = config
        self.shuffle = shuffle
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = paths.shape[0]

    #def __len__(self):
    #    return int(np.ceil(float(len(self.paths)) / self.config['BATCH_SIZE']))

    def num_classes(self):
        return self.config['N_CLASSES']

    def size(self):
        return self.paths.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def load_annotation(self, i):
        annots = np.genfromtxt(i[1], delimiter=' ')
        annots = annots.reshape(-1, 5)
        if len(annots) == 0:
            annots = np.array([[]])
        return annots

    def load_image(self, i):
        image = cv2.imread(i[0])
        image = np.divide(np.array(image, dtype=np.float32), 255.0)
        return image

    def next_batch(self, ind):
        nn = 0
        # l_bound = self._index_in_epoch
        l_bound = ind
        r_bound = l_bound + self.config['BATCH_SIZE']
        out_len = np.int(4 + 1 + self.config['N_CLASSES'])
        max_gt = np.int(self.config['TRUE_BOX_BUFFER'])
        # self._index_in_epoch += self.config['BATCH_SIZE']
        ind += self.config['BATCH_SIZE']
        if r_bound > self._num_examples:
            r_bound = self._num_examples
            l_bound = r_bound - self.config['BATCH_SIZE']
        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # input images
        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))
        y_batch1 = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], 1, out_len, max_gt))
        y_batch2 = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'],
                             out_len))  # desired network output
        countrowz = 0

        for train_instance in self.paths[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img = self.load_image(train_instance)
            all_objs = self.load_annotation(train_instance)
            countrowz += all_objs.shape[0]

            # construct output from object's x, y, w, h
            true_box_index = 0

            for row in range(all_objs.shape[0]):
                obj = all_objs[row, :]
                xmin = obj[1] - obj[3] / 2
                xmax = obj[1] + obj[3] / 2
                ymin = obj[2] - obj[4] / 2
                ymax = obj[3] + obj[4] / 2
                xmincell = np.int32(np.maximum(np.floor(xmin * self.config['GRID_W']), 0))
                xmaxcell = np.int32(np.minimum(np.ceil(xmax * self.config['GRID_W']), self.config['GRID_W']))
                ymincell = np.int32(np.maximum(np.floor(ymin * self.config['GRID_H']), 0))
                ymaxcell = np.int32(np.minimum(np.ceil(ymax * self.config['GRID_H']), self.config['GRID_H']))
                xcell = np.int32(np.floor(obj[1] * self.config['GRID_W']))
                ycell = np.int32(np.floor(obj[2] * self.config['GRID_H']))
                out_vec = np.zeros(5 + self.config['N_CLASSES'])
                out_vec[0:5] = [obj[1], obj[2], obj[3], obj[4], 1]
                class_pos = np.int32(5 + obj[0])
                out_vec[class_pos] = 1
                for yy in range(ymincell, ymaxcell):
                    for xx in range(xmincell, xmaxcell):
                        y_batch1[instance_count, yy, xx, 0, :, true_box_index] = out_vec
                truthxmin = np.subtract(0, np.divide(obj[3],2))
                truthxmax = np.add(0, np.divide(obj[3], 2))
                truthymin = np.subtract(0, np.divide(obj[4], 2))
                truthymax = np.add(0, np.divide(obj[4], 2))
                anchors = self.config['ANCHORS']
                anc_xmin = np.subtract(0, np.divide(anchors[:,0],2))
                anc_xmax = np.add(0, np.divide(anchors[:, 0], 2))
                anc_ymin = np.subtract(0, np.divide(anchors[:, 1], 2))
                anc_ymax = np.add(0, np.divide(anchors[:, 1], 2))
                interxmax = np.minimum(anc_xmax, truthxmax)
                interxmin = np.maximum(anc_xmin, truthxmin)
                interymax = np.minimum(anc_ymax, truthymax)
                interymin = np.maximum(anc_ymin, truthymin)
                sizex = np.maximum(np.subtract(interxmax, interxmin), 0)
                sizey = np.maximum(np.subtract(interymax, interymin), 0)
                inter_area = np.multiply(sizex, sizey)
                anc_area = np.multiply(anchors[:, 0], anchors[:, 1])
                truth_area = np.multiply(obj[3], obj[4])
                union_area = np.subtract(np.add(anc_area, truth_area), inter_area)
                iou = np.divide(inter_area, union_area)
                best_box = np.argmax(iou)
                y_batch2[instance_count, ycell, xcell, best_box, :] = out_vec
                # for bb in range(self.config['BOX']):
                #     y_batch2[instance_count, ycell, xcell, bb, :] = out_vec
                """
                box_blank = 0
                while box_blank < self.config['BOX']:
                    if y_batch2[instance_count, ycell, xcell, box_blank, 4] > 0:
                        box_blank += 1
                    else:
                        y_batch2[instance_count, ycell, xcell, 0, :] = out_vec
                        box_blank = self.config['BOX']
                """
                # assign the true box to b_batch
                b_batch[instance_count, 0, 0, 0, true_box_index] = out_vec[0:4]

                true_box_index += 1
                true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            x_batch[instance_count, :, :, :] = img

            # increase instance counter in current batch
            instance_count += 1
        nn = np.sum(y_batch2[:, :, :, :, 4])
        # print(train_instance[0])
        # print(y_batch2[0, 7, 23, :, 2:4])

        if ind > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            if self.shuffle:
                np.random.shuffle(self.paths)

        return {"images": x_batch, "boxes": b_batch}, {"gt1": y_batch1, "gt2": y_batch2, "nn": nn, "ind": ind}
