# https://github.com/experiencor/keras-yolo2/blob/master/preprocessing.py
import numpy as np
import cv2
import os


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
        if os.stat(i[1]).st_size != 0:
            annots = np.genfromtxt(i[1], delimiter=' ')
            annots = annots.reshape(-1, 5)
            if len(annots) == 0:
                annots = np.empty((0,5))
        else:
            annots = np.empty((0, 5))
        return annots

    def load_image(self, i):
        image = cv2.imread(i[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.divide(np.array(image, dtype=np.float32), 255.0)
        return image

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.paths)

    def next_batch(self, ind):
        if ind == 0:
            # Finished epoch
            self._epochs_completed += 1
            print(self._epochs_completed, self.shuffle)
            if self.shuffle:
                print("end", self.shuffle)
                np.random.shuffle(self.paths)

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

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # input images
        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch1 = np.zeros((r_bound - l_bound, self.config['GRID_H'], self.config['GRID_W'], self.config['BOX'],
                             out_len))  # desired network output
        y_batch2 = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))

        zero_animal_check = True

        while zero_animal_check:
            instance_count = 0
            countrowz = 0

            for train_instance in self.paths[l_bound:r_bound]:
                # augment input image and fix object's position and size
                img = self.load_image(train_instance)
                all_objs = self.load_annotation(train_instance)
                countrowz += all_objs.shape[0]

                # construct output from object's x, y, w, h
                true_box_index = 0

                if all_objs.shape[0] > 0:
                    for row in range(all_objs.shape[0]):
                        obj = all_objs[row, :]
                        xcell = np.int32(np.floor(obj[1] * self.config['GRID_W']))
                        ycell = np.int32(np.floor(obj[2] * self.config['GRID_H']))
                        centx = (obj[1] *  self.config['GRID_W']) - xcell
                        centy = (obj[2] * self.config['GRID_H']) - ycell
                        # Calc best box compared to anchors, zero position both
                        xmin_true = 0 - obj[3] / 2
                        xmax_true = 0 + obj[3] / 2
                        ymin_true = 0 - obj[4] / 2
                        ymax_true = 0 + obj[4] / 2
                        anchors = self.config['ANCHORS']
                        anchors_wi = np.divide(anchors, [self.config['GRID_W'], self.config['GRID_H']])
                        anc_xmin = np.subtract(0, np.divide(anchors_wi[:,0], 2))
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
                        out_vec = np.zeros(5 + self.config['N_CLASSES'])
                        # I think this should be this
                        out_vec[0:5] = [centx, centy, obj[3], obj[4], 1.0]
                        # out_vec[0:5] = [obj[1], obj[2], obj[3], obj[4], 1]
                        class_pos = np.int32(5 + obj[0])
                        out_vec[class_pos] = 1.
                        y_batch1[instance_count, ycell, xcell, best_box, :] = out_vec
                        # y_batch1[instance_count, ycell, xcell, best_box, :] = [obj[1], obj[2], obj[3], obj[4], 1., 1.]
                        # assign the true box to b_batch
                        # y_batch2[instance_count, 0, 0, 0, true_box_index, :] = out_vec[0:4]
                        y_batch2[instance_count, 0, 0, 0, true_box_index, :] = [obj[1], obj[2], obj[3], obj[4]]
                        #print([obj[1], obj[2], obj[3], obj[4]])
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

                # assign input image to x_batch
                x_batch[instance_count, :, :, :] = img

                # increase instance counter in current batch
                instance_count += 1

            zero_animal_check = False

            if countrowz == 0:
                zero_animal_check = True
                l_bound = l_bound + 1
                r_bound = r_bound + 1



        return {"images": x_batch}, {"gt1": y_batch1, "gt2": y_batch2, "ind": ind}
