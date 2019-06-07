from my_loss_tf import loss_gfrc_yolo
from out_box_class_conf import convert_pred_to_output
from my_metric_tf import metrics_gfrc_yolo
import tensorflow as tf
import numpy as np


def elementwise_fn_lg(prevout, gt):
    mximg, boxy, boxx, nanc, ol, max_gt = prevout.get_shape()
    img = gt[11]
    ymn = gt[0]
    ymx = gt[1]
    ymd = gt[12]
    xmn = gt[2]
    xmx = gt[3]
    xmd = gt[13]
    gtin = gt[4]
    xc = tf.reshape(gt[5], [1])
    yc = tf.reshape(gt[6], [1])
    wid = tf.reshape(gt[7], [1])
    hei = tf.reshape(gt[8], [1])
    clz = gt[9]
    ncl = tf.cast(gt[10], dtype=tf.int32)
    conf = tf.reshape([1.0], [1])
    classes = tf.one_hot(clz, ncl)
    # cant' use assignment so need to stack / concat lots of zeros together so can add up at the end.
    gtbck = tf.cast(tf.subtract(tf.subtract(max_gt, 1), gtin), dtype=tf.int32)
    out_ten = tf.stack((xc, yc, wid, hei, conf, classes), axis=1)
    out_ten = tf.expand_dims(out_ten, axis=-1)
    zero_ten_front = tf.zeros([1, ol, gtin])
    zero_ten_back = tf.zeros([1, ol, gtbck])
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=-1)
    zero_ten_front = tf.zeros([xmn, 1, ol, max_gt])
    zero_ten_back = tf.zeros([xmx, 1, ol, max_gt])
    tile_shape = tf.reshape([xmd, 1, 1], [3])
    out_ten = tf.tile(out_ten, tile_shape)
    out_ten = tf.reshape(out_ten, [xmd, 1, ol, max_gt])
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=0)
    zero_ten_front = tf.zeros([ymn, boxx, 1, ol, max_gt])
    zero_ten_back = tf.zeros([ymx, boxx, 1, ol, max_gt])
    tile_shape = tf.reshape([ymd, 1, 1, 1], [4])
    out_ten = tf.tile(out_ten, tile_shape)
    out_ten = tf.reshape(out_ten, [ymd, boxx, 1, ol, max_gt])
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=0)
    zero_ten_front = tf.zeros([img, boxy, boxx, 1, ol, max_gt])
    imgbck = tf.cast(tf.subtract(tf.subtract(mximg, 1), img), dtype=tf.int32)
    zero_ten_back = tf.zeros([imgbck, boxy, boxx, 1, ol, max_gt])
    out_ten = tf.expand_dims(out_ten, axis=0)
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=0)
    prevout = tf.add(prevout, out_ten)
    return prevout


def elementwise_fn_sm(prevout, gt):
    mximg, boxy, boxx, nanc, ol = prevout.get_shape()
    img = gt[9]
    xcl = gt[0]
    ycl = gt[1]
    gtin = gt[2]
    xc = tf.reshape(gt[3], [1])
    yc = tf.reshape(gt[4], [1])
    wid = tf.reshape(gt[5], [1])
    hei = tf.reshape(gt[6], [1])
    clz = gt[7]
    ncl = tf.cast(gt[8], dtype=tf.int32)
    conf = tf.reshape([1.0], [1])
    classes = tf.one_hot(clz, ncl)
    # cant' use assignment so need to stack / concat lots of zeros together so can add up at the end.
    out_ten = tf.stack((xc, yc, wid, hei, conf, classes), axis=1)
    out_ten = tf.Print(out_ten, [out_ten])
    out_ten = tf.expand_dims(out_ten, axis=0)
    # out_ten = tf.expand_dims(out_ten, axis=0)
    zero_ten_front = tf.zeros([xcl, 1, ol])
    bcksz = tf.subtract(boxx, xcl)
    zero_ten_back = tf.zeros([bcksz, 1, ol])
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=0)
    out_ten = tf.expand_dims(out_ten, axis=0)
    zero_ten_front = tf.zeros([ycl, boxx, 1, ol])
    bcksz = tf.subtract(boxy, ycl)
    zero_ten_back = tf.zeros([bcksz, boxx, 1, ol])
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=0)
    out_ten = tf.expand_dims(out_ten, axis=0)
    zero_ten_front = tf.zeros([img, boxy, boxx, 1, ol])
    bcksz = tf.subtract(mximg, img)
    zero_ten_back = tf.zeros([bcksz, boxy, boxx, 1, ol])
    out_ten = tf.concat((zero_ten_front, out_ten, zero_ten_back), axis=0)
    prevout = tf.add(prevout, out_ten)
    print("po", prevout)
    return prevout


def length(sequence):
    used = tf.sign(sequence)
    length = tf.reduce_sum(used)
    length = tf.cast(length, tf.int32)
    return length


def get_labels_lg(out_mat_lg, input):
    boxy = input[1]
    boxx = input[2]
    nclasses = tf.cast(input[3], dtype=tf.int32)
    img = tf.cast(input[4], dtype=tf.int32)
    record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0]]
    bboxes = input[0]
    class_in = bboxes[:, 0]
    centx = bboxes[:, 1]
    centy = bboxes[:, 2]
    sizex = bboxes[:, 3]
    sizey = bboxes[:, 4]
    class_in = tf.to_int32(class_in)
    n_gt = tf.cast(tf.size(class_in), dtype=tf.int32)
    gtz = tf.range(n_gt)
    imgz = tf.range(n_gt)
    classz = tf.range(n_gt)
    half_size_x = tf.divide(sizex, 2)
    half_size_y = tf.divide(sizey, 2)
    xmin = tf.subtract(centx, half_size_x)
    ymin = tf.subtract(centy, half_size_y)
    xmax = tf.add(centx, half_size_x)
    ymax = tf.add(centy, half_size_y)
    xmincel = tf.cast(tf.floor(tf.multiply(xmin, boxx)), dtype=tf.int32)
    ymincel = tf.cast(tf.floor(tf.multiply(ymin, boxy)), dtype=tf.int32)
    xmaxcel = tf.cast(tf.floor(tf.multiply(xmax, boxx)), dtype=tf.int32)
    ymaxcel = tf.cast(tf.floor(tf.multiply(ymax, boxy)), dtype=tf.int32)
    xmidcel = tf.subtract(xmaxcel, xmincel)
    xmaxcel = tf.subtract(tf.cast(boxx, dtype=tf.int32), xmaxcel)
    ymidcel = tf.subtract(ymaxcel, ymincel)
    ymaxcel = tf.subtract(tf.cast(boxy, dtype=tf.int32), ymaxcel)
    sizex = tf.multiply(sizex, boxx)
    sizey = tf.multiply(sizey, boxy)
    out_mat_new = tf.scan(
        elementwise_fn_lg,
        (ymincel, ymaxcel, xmincel, xmaxcel, gtz, centx, centy, sizex, sizey, class_in, classz, imgz, ymidcel, xmidcel),
        out_mat_lg
    )
    out_mat_new = tf.reduce_sum(out_mat_new, axis=0)
    return out_mat_new


def get_labels_sm(out_mat_lg, input):
    boxy = input[1]
    boxx = input[2]
    nclasses = tf.cast(input[3], dtype=tf.int32)
    bboxes = input[0]
    class_in = bboxes[:, 0]
    centx = bboxes[:, 1]
    centy = bboxes[:, 2]
    sizex = bboxes[:, 3]
    sizey = bboxes[:, 4]
    class_in = tf.to_int32(class_in)
    n_gt = tf.cast(tf.size(class_in), dtype=tf.int32)
    gtz = tf.range(n_gt)
    imgz = tf.range(n_gt)
    classz = tf.range(n_gt)
    cellx = tf.cast(tf.floor(tf.multiply(centx, boxx)), dtype=tf.int32)
    celly = tf.cast(tf.floor(tf.multiply(centy, boxy)), dtype=tf.int32)
    sizex = tf.multiply(sizex, boxx)
    sizey = tf.multiply(sizey, boxy)
    out_mat_new = tf.scan(
        elementwise_fn_sm,
        (cellx, celly, gtz, centx, centy, sizex, sizey, class_in, classz, imgz),
        out_mat_lg
    )
    out_mat_new = tf.reduce_sum(out_mat_new, axis=0)
    return out_mat_new


def convert_labels_to_matrix(labels, params):
    max_gt = params["max_ground_truth"]
    nclasses = params["n_classes"]
    nanchors = params["n_anchors"]
    boxx = params["boxs_x"]
    boxy = params["boxs_y"]
    nbat = params["batch_size"]
    out_len = 5 + nclasses
    boxyz = tf.convert_to_tensor(np.repeat(boxy, nbat), dtype=tf.float32)
    boxxz = tf.convert_to_tensor(np.repeat(boxx, nbat), dtype=tf.float32)
    out_lenz = tf.convert_to_tensor(np.repeat(out_len, nbat), dtype=tf.float32)
    nclassez = tf.convert_to_tensor(np.repeat(nclasses, nbat), dtype=tf.float32)
    maxgtz = tf.convert_to_tensor(np.repeat(max_gt, nbat), dtype=tf.float32)
    imgz = tf.range(nbat)
    out_mat_lg = tf.zeros([nbat, boxy, boxx, 1, out_len, max_gt], tf.float32)
    out_mat_sm = tf.zeros([nbat, boxy, boxx, 1, out_len], tf.float32)
    # out_mat1 = tf.scan(get_labels, (labels, boxy, boxx, out_len, nclasses, max_gt), out_mat_lg)
    out_mat1 = tf.scan(get_labels_lg, (labels, boxyz, boxxz, nclassez, imgz), out_mat_lg)
    out_mat1 = tf.reduce_sum(out_mat1, axis=0)
    out_mat2 = tf.scan(get_labels_sm, (labels, boxyz, boxxz, nclassez, imgz), out_mat_sm)
    out_mat2 = tf.reduce_sum(out_mat2, axis=0)
    return out_mat1, out_mat2


def gfrc_tf_yolo_model_fn(features, labels, mode, params):
    """ Model in tensorflow rather than keras """

    # Input Layer

    labels = labels
    final_size = params["final_size"]
    learn_rate = params["learn_rate"]
    img_x_size = params["img_x_pix"]
    img_y_size = params["img_y_pix"]
    img_channels = params["n_channels"]
    bat_size = params["batch_size"]
    input_size = [bat_size, img_y_size, img_x_size, img_channels]
    #input_layer = [tf.feature_column.numeric_column(features, shape=input_size)]
    #print(input_layer)
    input_layer = tf.reshape(features, input_size)

    # For now use leaky relu before batch norm (mostly cos at the moment I can't figure out how to do it after!)
    # Convolutional Layer #1
    # leave out initializer and it will use a uniform initializer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn1 = tf.layers.batch_normalization(inputs = conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=bn1,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn2 = tf.layers.batch_normalization(inputs = conv2)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=bn2,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn3 = tf.layers.batch_normalization(inputs = conv3)
    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=bn3,
        filters=64,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn4 = tf.layers.batch_normalization(inputs=conv4)
    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=bn4,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn5 = tf.layers.batch_normalization(inputs=conv5)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(
        inputs=bn5,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn6 = tf.layers.batch_normalization(inputs = conv6)
    # Convolutional Layer #7
    conv7 = tf.layers.conv2d(
        inputs=bn6,
        filters=128,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn7 = tf.layers.batch_normalization(inputs=conv7)
    # Convolutional Layer #8
    conv8 = tf.layers.conv2d(
        inputs=bn7,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn8 = tf.layers.batch_normalization(inputs=conv8)

    # Pooling Layer #4
    pool4 = tf.layers.max_pooling2d(
        inputs=bn8,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #9
    conv9 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn9 = tf.layers.batch_normalization(inputs=conv9)
    # Convolutional Layer #10
    conv10 = tf.layers.conv2d(
        inputs=bn9,
        filters=256,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn10 = tf.layers.batch_normalization(inputs=conv10)
    # Convolutional Layer #11
    conv11 = tf.layers.conv2d(
        inputs=bn10,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn11 = tf.layers.batch_normalization(inputs=conv11)

    # Convolutional Layer #12
    conv12 = tf.layers.conv2d(
        inputs=bn11,
        filters=1024,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn12 = tf.layers.batch_normalization(inputs=conv12)
    # Convolutional Layer #13
    conv13 = tf.layers.conv2d(
        inputs=bn12,
        filters=final_size,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=None, #linear
        use_bias=True,
        bias_initializer=tf.zeros_initializer())

    # Calculate Loss (for both TRAIN and EVAL modes)
    gt1_batch, gt2_batch = convert_labels_to_matrix(labels, params)
    loss = loss_gfrc_yolo(gt1=gt1_batch, gt2=gt2_batch, y_pred=conv13, dict_in=params)
    class_out, conf_out, box_out = convert_pred_to_output(conv13, dict_in=params)
    metrics_out = metrics_gfrc_yolo(gt1=gt1_batch, gt2=gt2_batch, y_pred=conv13, dict_in=params)

    predictions = {
        # Generate predictions for PREDICT mode
        "classes": class_out,
        "probabilities": conf_out,
        "boxes": box_out,
    }

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "total_loss": metrics_out["total_loss"],
        "conf_loss_nogt": metrics_out["conf_loss_nogt"],
        "conf_loss_gt": metrics_out["conf_loss_gt"],
        "cent_loss": metrics_out["cent_loss"],
        "size_loss": metrics_out["size_loss"],
        "class_loss": metrics_out["class_loss"],
        "TP": metrics_out["TP"],
        "FP": metrics_out["FP"],
        "FN": metrics_out["FN"],
        "Re": metrics_out["Re"],
        "Pr": metrics_out["Pr"],
        "FPR": metrics_out["FPR"]
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def gfrc_tf_yolo_model_fn_train(features, labels, params):
    """ Model in tensorflow rather than keras """

    # Input Layer
    input_images = features
    print(input_images)
    labels = labels
    final_size = params["final_size"]
    learn_rate = params["learn_rate"]
    img_x_size = params["img_x_pix"]
    img_y_size = params["img_y_pix"]
    img_channels = params["n_channels"]
    bat_size = params["batch_size"]
    input_size = (bat_size, img_y_size, img_x_size, img_channels)
    input_layer = tf.reshape(input_images, input_size)

    # For now use leaky relu before batch norm (mostly cos at the moment I can't figure out how to do it after!)
    # Convolutional Layer #1
    # leave out initializer and it will use a uniform initializer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn1 = tf.layers.batch_normalization(inputs = conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=bn1,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn2 = tf.layers.batch_normalization(inputs = conv2)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=bn2,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn3 = tf.layers.batch_normalization(inputs = conv3)
    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=bn3,
        filters=64,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn4 = tf.layers.batch_normalization(inputs=conv4)
    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=bn4,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn5 = tf.layers.batch_normalization(inputs=conv5)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(
        inputs=bn5,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        strides = (1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn6 = tf.layers.batch_normalization(inputs = conv6)
    # Convolutional Layer #7
    conv7 = tf.layers.conv2d(
        inputs=bn6,
        filters=128,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn7 = tf.layers.batch_normalization(inputs=conv7)
    # Convolutional Layer #8
    conv8 = tf.layers.conv2d(
        inputs=bn7,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn8 = tf.layers.batch_normalization(inputs=conv8)

    # Pooling Layer #4
    pool4 = tf.layers.max_pooling2d(
        inputs=bn8,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='same')

    # Convolutional Layer #9
    conv9 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn9 = tf.layers.batch_normalization(inputs=conv9)
    # Convolutional Layer #10
    conv10 = tf.layers.conv2d(
        inputs=bn9,
        filters=256,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn10 = tf.layers.batch_normalization(inputs=conv10)
    # Convolutional Layer #11
    conv11 = tf.layers.conv2d(
        inputs=bn10,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn11 = tf.layers.batch_normalization(inputs=conv11)

    # Convolutional Layer #12
    conv12 = tf.layers.conv2d(
        inputs=bn11,
        filters=1024,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.leaky_relu,
        # kernel_regularizer=l2(0.0005),
        use_bias=False)
    bn12 = tf.layers.batch_normalization(inputs=conv12)
    # Convolutional Layer #13
    conv13 = tf.layers.conv2d(
        inputs=bn12,
        filters=final_size,
        kernel_size=[1, 1],
        strides=(1, 1),
        padding="same",
        activation=None, #linear
        use_bias=True,
        bias_initializer=tf.zeros_initializer())

    # Calculate Loss (for both TRAIN and EVAL modes)
    gt1_batch, gt2_batch = convert_labels_to_matrix(labels, params)
    loss = loss_gfrc_yolo(gt1=gt1_batch, gt2=gt2_batch, y_pred=conv13, dict_in=params)
    class_out, conf_out, box_out = convert_pred_to_output(conv13, dict_in=params)
    metrics_out = metrics_gfrc_yolo(gt1=gt1_batch, gt2=gt2_batch, y_pred=conv13, dict_in=params)

    predictions = {
        # Generate predictions for PREDICT mode
        "classes": class_out,
        "probabilities": conf_out,
        "boxes": box_out,
    }

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    # Configure the Training Op (for TRAIN mode)
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    #    train_op = optimizer.minimize(
    #        loss=loss,
    #        global_step=tf.train.get_global_step())
    #    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "total_loss": metrics_out["total_loss"],
        "conf_loss_nogt": metrics_out["conf_loss_nogt"],
        "conf_loss_gt": metrics_out["conf_loss_gt"],
        "cent_loss": metrics_out["cent_loss"],
        "size_loss": metrics_out["size_loss"],
        "class_loss": metrics_out["class_loss"],
        "TP": metrics_out["TP"],
        "FP": metrics_out["FP"],
        "FN": metrics_out["FN"],
        "Re": metrics_out["Re"],
        "Pr": metrics_out["Pr"],
        "FPR": metrics_out["FPR"]
    }
    # return tf.estimator.EstimatorSpec(
    #    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return train_op