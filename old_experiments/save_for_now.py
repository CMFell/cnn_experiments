# model.fit(x_train, y_train, batch_size=32, epochs=10)
# score = model.evaluate(x_test, y_test, batch_size=32)

# sess = K.get_session()
# sing_image= tf.placeholder()
# yolo_model = load_model(gfrc_yolo)

"""
# gt_path = "C:/Benchmark_data/GFRC/yolo_GFRC_bboxes_241018.csv"
gt_path = "E:/CF_Calcs/BenchmarkSets/GFRC/yolo_GFRC_bboxes_241018.csv"
gt_list = pd.read_csv(gt_path)

images = np.unique(gt_list.File_name)
image_check = images[6]

gt_boxes = gt_list[gt_list.File_name == image_check]


# part of image to cut tile from - xmin, xmax, ymin, ymax
input_loc = [0.5, 1.0, 0.5, 1.0]

gt_boxes = filter_gt_tile(gt_boxes, input_loc)

out_dict = create_train_gt(gt_boxes, dict_deets)

print(out_dict['gt_array'][21, 227, 4, :])
print(out_dict['gt_array'][67, 182, 2, :])
"""

"""
# print(net_output[0, 0, 0, :])


def loss_gfrc_yolo_np(netoutput, gt_dict, dictdeets):
    # compares output from cnn with ground truth to calculate loss
    # only for one image at the moment

    boxsx = int(dictdeets['boxs_x'])
    boxsy = int(dictdeets['boxs_y'])
    n_classes = dictdeets['n_classes']
    anchors = dictdeets['anchors']
    nanchors = anchors.shape[0]
    num_out = 5 + n_classes
    lam_coord = dictdeets['lambda_coord']
    lam_noobj = dictdeets['lambda_noobj']
    gt_out = gt_dict['gt_array']

    # size of all boxes and anchors
    size1 = [boxsy, boxsx, nanchors]
    # size of all boxes anchors and data
    size2 = [boxsy, boxsx, nanchors, num_out]
    # size of all boxes
    size3 = [boxsy, boxsx]
    # size of output layer
    size4 = [boxsy, boxsx, nanchors * num_out]

    # restructure net_output to same as gt_out
    netoutput = np.reshape(netoutput, size4)
    netoutput = np.reshape(netoutput, size2)

    # ones i detects if object appears in cell
    onesi = np.array(np.greater(np.sum(np.reshape(gt_out[:, :, :, 5], size1), axis=2), 0), dtype=np.int)
    onesi = np.reshape(onesi, (onesi.shape[0], onesi.shape[1], 1))
    noonesi = np.subtract(1, onesi)

    # get confidences from net_output
    confs_cnn = expit(np.reshape(netoutput[:, :, :, 4], size1))

    # calculate confidence losses when no ground truth in cell
    conf_loss_nogt = np.sum(np.multiply(np.subtract(0, confs_cnn), noonesi))

    # The rest of the losses are only calculated in cells where there are ground truths
    # Since there are only a few ground truth per image it makes more sense to calculate per ground truth

    # read ground truth info from dictionary
    cents_gt = gt_dict['centre_img']
    sizes_gt = gt_dict['size_img']
    classes_gt = gt_dict['one_hot_class']
    cell_gt = gt_dict['cells']
    n_gt = cents_gt.shape[0]

    # create vectors to store losses for each location where there is a ground truth
    class_loss = np.zeros(n_gt)
    conf_loss_gt = np.zeros(n_gt)
    pos_loss = np.zeros(n_gt)

    for gt in range(n_gt):
        # get location of cell of ground truth
        cell_x = cell_gt[gt, 0]
        cell_y = cell_gt[gt, 1]

        # get output of cnn details
        # get confidences for all anchor boxes at ground truth cell
        conf_cnn = expit(confs_cnn[cell_y, cell_x, :])
        conf_cnn = np.reshape(conf_cnn, (nanchors, 1))
        # calculate top left cell position relative to image
        tl_cell = np.array([cell_x / boxsx, cell_y / boxsy], dtype=np.float)
        # get centre position in cell
        cent_cnn = expit(netoutput[cell_y, cell_x, :, :2])
        cent_cnn = np.reshape(cent_cnn, (nanchors, 2))
        # adjust relative to size of image
        cent_cnn = np.divide(cent_cnn, size3)
        # adjust to centre position in image
        cent_cnn = np.add(cent_cnn, tl_cell)
        # get size of detection relative to cell
        size_cnn = np.exp(np.clip(netoutput[cell_y, cell_x, :, 2:4], -50, 50))
        size_cnn = np.reshape(size_cnn, (nanchors, 2))
        # adjust so relative to anchors
        size_cnn = np.multiply(size_cnn, anchors)
        # adjust so relative to image to match ground truth
        size_cnn = np.divide(size_cnn, size3)
        # get classes predicted from cnn
        class_cnn = softmax(np.clip(netoutput[cell_y, cell_x, :, 5:], -50, 50))
        class_cnn = np.reshape(class_cnn, (nanchors, n_classes))

        # get ground truth details
        cent_gt = np.reshape(cents_gt[gt, :], (1, 2))
        size_gt = np.reshape(sizes_gt[gt, :], (1, 2))
        class_gt = np.reshape(classes_gt[gt, :], (1, n_classes))

        # get which bound box best iou
        size_cnn_half = np.divide(size_cnn, 2)
        area_cnn = np.multiply(size_cnn[:, 0], size_cnn[:, 1])
        size_gt_half = np.divide(size_gt, 2)
        area_gt = np.multiply(size_gt[:, 0], size_gt[:, 1])
        min_cnn = np.subtract(cent_cnn, size_cnn_half)
        max_cnn = np.add(cent_cnn, size_cnn_half)
        min_gt = np.subtract(cent_gt, size_gt_half)
        max_gt = np.add(cent_gt, size_gt_half)
        inter_mins = np.minimum(max_cnn, max_gt)
        inter_maxs = np.maximum(min_cnn, min_gt)
        inter_size = np.maximum(np.subtract(inter_maxs, inter_mins), 0)
        inter_area = np.multiply(inter_size[:, 0], inter_size[:, 1])
        union_area = np.subtract(np.add(area_gt, area_cnn), inter_area)
        iou = np.divide(inter_area, union_area)
        best_box = np.argmax(iou)

        # In darknet it appears only one set of confs and classes are output
        # Here we are going to calculate these losses along with the position loss for the best matching box only

        cent_cnn = cent_cnn[best_box, :]
        class_cnn = class_cnn[best_box, :]
        conf_cnn = conf_cnn[best_box, :]

        class_loss[gt] = np.sum(np.square(np.subtract(class_gt, class_cnn)))
        conf_loss_gt[gt] = np.sum(np.square(np.subtract(1, conf_cnn)))
        # Technically according to paper this should be iou minus confidence - ignore for now
        # conf_loss_gt[gt] = np.sum(np.square(np.subtract(iou[best_box], conf_cnn)))
        cent_loss = np.sum(np.square(np.subtract(cent_gt, cent_cnn)))
        size_loss = np.sum(np.square(np.subtract(np.sqrt(size_gt), np.sqrt(size_cnn))))
        pos_loss[gt] = cent_loss + size_loss

    total_loss = lam_noobj * conf_loss_nogt + lam_coord * np.sum(pos_loss) + np.sum(conf_loss_gt) + np.sum(class_loss)

    # print("total", total_loss, "no_gt", lam_noobj * conf_loss_nogt, "pos", lam_coord * np.sum(pos_loss),
    # "gt", np.sum(conf_loss_gt), "class", np.sum(class_loss))

    return total_loss


loss_gfrc_yolo_np(net_output, out_dict, dict_deets)
"""

"""
yolo_model = load_model("model_data/yolo_gfrc.h5")

for ly in range(len(yolo_model.layers)):
    print(ly)
    print(yolo_model.layers[ly].get_config())

yolo_model.summary()

"""