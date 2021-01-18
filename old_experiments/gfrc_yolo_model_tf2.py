from my_loss_tf2 import loss_gfrc_yolo, total_loss_calc
from out_box_class_conf import convert_pred_to_output
from my_metric_tf2 import metric_cf_ngt, metric_cf_gt, metric_cnt, metric_sz, metric_cl, \
    metric_tp, metric_fp, metric_fn, metric_re, metric_pr, metric_fpr, metric_tl, metric_t1, metric_t2
import tensorflow as tf


def gfrc_tf_yolo_model_fn(features, labels, mode, params):

    gt1_batch = labels['gt1']
    gt2_batch = labels['gt2']
    final_size = params["final_size"]
    learn_rate = params["learn_rate"]
    img_x_size = params["img_x_pix"]
    img_y_size = params["img_y_pix"]
    img_channels = params["n_channels"]
    bat_size = params["batch_size"]
    input_size = [bat_size, img_y_size, img_x_size, img_channels]
    #input_layer = [tf.feature_column.numeric_column(features, shape=input_size)]
    #print(input_layer)
    input_layer = tf.reshape(features["image"], input_size)

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

    class_out, conf_out, box_out = convert_pred_to_output(conv13, dict_in=params)
    predictions = {
        # Generate predictions for PREDICT mode
        "classes": class_out,
        "probabilities": conf_out,
        "boxes": box_out,
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    individual_losses = loss_gfrc_yolo(gt1=gt1_batch, gt2=gt2_batch, y_pred=conv13, dict_in=params)
    loss = total_loss_calc(individual_losses, dict_in=params)

    met_tl = metric_tl(individual_losses, dict_in=params)
    met_cf_ngt = metric_cf_ngt(individual_losses)
    met_cf_gt = metric_cf_gt(individual_losses)
    met_cnt = metric_cnt(individual_losses)
    met_sz = metric_sz(individual_losses)
    met_cl = metric_cl(individual_losses)
    met_tp = metric_tp(individual_losses)
    met_fp = metric_fp(individual_losses)
    met_fn = metric_fn(individual_losses)
    met_re = metric_re(individual_losses)
    met_pr = metric_pr(individual_losses)
    met_fpr = metric_fpr(individual_losses)
    met_t1 = metric_t1(individual_losses)
    met_t2 = metric_t2(individual_losses)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "total_loss": met_tl,
        "conf_loss_nogt": met_cf_ngt,
        "conf_loss_gt": met_cf_gt,
        "cent_loss": met_cnt,
        "size_loss": met_sz,
        "class_loss": met_cl,
        "TP": met_tp,
        "FP": met_fp,
        "FN": met_fn,
        "Re": met_re,
        "Pr": met_pr,
        "FPR": met_fpr,
        "t1": met_t1,
        "t2": met_t2
    }

    tf.summary.scalar("total_loss", met_tl[1])
    tf.summary.scalar("conf_loss_nogt", met_cf_ngt[1])
    tf.summary.scalar("conf_loss_gt", met_cf_gt[1])
    tf.summary.scalar("cent_loss", met_cnt[1])
    tf.summary.scalar("size_loss", met_sz[1])
    tf.summary.scalar("class_loss", met_cl[1])
    tf.summary.scalar("TP", met_tp[1])
    tf.summary.scalar("FP", met_fp[1])
    tf.summary.scalar("FN", met_fn[1])
    tf.summary.scalar("Re", met_re[1])
    tf.summary.scalar("Pr", met_pr[1])
    tf.summary.scalar("FPR", met_fpr[1])

    logging_hook = tf.train.LoggingTensorHook(
        {
            "nogt": met_cf_ngt[1],
            "gt": met_cf_gt[1],
            "cnt": met_cnt[1],
            "sz": met_sz[1],
            "cl": met_cl[1],
            "TP": met_tp[1],
            "FP": met_fp[1],
            "FN": met_fn[1],
            "TL": met_tl[1],
            "T1": met_t1[1],
            "T2": met_t2[1]
         },
        every_n_iter=100
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Configure the Training Op (for TRAIN mode)
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])





