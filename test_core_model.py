import tensorflow as tf
from importweights2 import load_weights_from_file


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def conv_bias(lay_input, weights_in, biases_in):
    # Creating the final output convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases_in
    return layer


def conv_bn2_lr(lay_input, weights_in):
    # Creating a default convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME')
    # Batch normalisation
    batch_mean, batch_var = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)
    scale = tf.Variable(tf.ones([1]))
    beta = tf.Variable(tf.zeros([1]))
    layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, offset=beta, scale=scale, variance_epsilon=1e-6)
    # activation function
    layer = tf.nn.leaky_relu(layer)
    return layer


def conv_bn_lr(lay_input, weights_in, batch_mean, batch_var):
    # Creating a default convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME')
    # Batch normalisation
    scale = tf.Variable(tf.ones([1]))
    beta = tf.Variable(tf.zeros([1]))
    batch_mean, batch_var = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)
    layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, offset=beta, scale=scale, variance_epsilon=1e-6)
    # activation function
    layer = tf.nn.leaky_relu(layer)
    return layer


layer_list = load_weights_from_file(weightspath, n_filters, filtersizes)
wtcn1 = tf.convert_to_tensor(layer_list[0][0][0])
wtcn2 = tf.convert_to_tensor(layer_list[1][0][0])
wtcn3 = tf.convert_to_tensor(layer_list[2][0][0])
wtcn4 = tf.convert_to_tensor(layer_list[3][0][0])
wtcn5 = tf.convert_to_tensor(layer_list[4][0][0])
wtcn6 = tf.convert_to_tensor(layer_list[5][0][0])
wtcn7 = tf.convert_to_tensor(layer_list[6][0][0])
wtcn8 = tf.convert_to_tensor(layer_list[7][0][0])
wtcn9 = tf.convert_to_tensor(layer_list[8][0][0])
wtcn10 = tf.convert_to_tensor(layer_list[9][0][0])
wtcn11 = tf.convert_to_tensor(layer_list[10][0][0])
wtcn12 = tf.convert_to_tensor(layer_list[11][0][0])
wtcn13 = tf.convert_to_tensor(layer_list[12][0][0])
wtcn14 = tf.convert_to_tensor(layer_list[13][0][0])
wtcn15 = tf.convert_to_tensor(layer_list[14][0][0])
wtcn16 = tf.convert_to_tensor(layer_list[15][0][0])
wtcn17 = tf.convert_to_tensor(layer_list[16][0][0])
wtcn18 = tf.convert_to_tensor(layer_list[17][0][0])
wtcn19 = tf.convert_to_tensor(layer_list[18][0][0])
wtcn20 = tf.convert_to_tensor(layer_list[19][0][0])
wtcn21 = tf.convert_to_tensor(layer_list[20][0][0])
wtcn22 = tf.convert_to_tensor(layer_list[21][0][0])

bn1m = tf.convert_to_tensor(layer_list[0][1][0])
bn2m = tf.convert_to_tensor(layer_list[1][1][0])
bn3m = tf.convert_to_tensor(layer_list[2][1][0])
bn4m = tf.convert_to_tensor(layer_list[3][1][0])
bn5m = tf.convert_to_tensor(layer_list[4][1][0])
bn6m = tf.convert_to_tensor(layer_list[5][1][0])
bn7m = tf.convert_to_tensor(layer_list[6][1][0])
bn8m = tf.convert_to_tensor(layer_list[7][1][0])
bn9m = tf.convert_to_tensor(layer_list[8][1][0])
bn10m = tf.convert_to_tensor(layer_list[9][1][0])
bn11m = tf.convert_to_tensor(layer_list[10][1][0])
bn12m = tf.convert_to_tensor(layer_list[11][1][0])
bn13m = tf.convert_to_tensor(layer_list[12][1][0])
bn14m = tf.convert_to_tensor(layer_list[13][1][0])
bn15m = tf.convert_to_tensor(layer_list[14][1][0])
bn16m = tf.convert_to_tensor(layer_list[15][1][0])
bn17m = tf.convert_to_tensor(layer_list[16][1][0])
bn18m = tf.convert_to_tensor(layer_list[17][1][0])
bn19m = tf.convert_to_tensor(layer_list[18][1][0])
bn20m = tf.convert_to_tensor(layer_list[19][1][0])
bn21m = tf.convert_to_tensor(layer_list[20][1][0])
bn22m = tf.convert_to_tensor(layer_list[21][1][0])
bn1v = tf.convert_to_tensor(layer_list[0][1][1])
bn2v = tf.convert_to_tensor(layer_list[1][1][1])
bn3v = tf.convert_to_tensor(layer_list[2][1][1])
bn4v = tf.convert_to_tensor(layer_list[3][1][1])
bn5v = tf.convert_to_tensor(layer_list[4][1][1])
bn6v = tf.convert_to_tensor(layer_list[5][1][1])
bn7v = tf.convert_to_tensor(layer_list[6][1][1])
bn8v = tf.convert_to_tensor(layer_list[7][1][1])
bn9v = tf.convert_to_tensor(layer_list[8][1][1])
bn10v = tf.convert_to_tensor(layer_list[9][1][1])
bn11v = tf.convert_to_tensor(layer_list[10][1][1])
bn12v = tf.convert_to_tensor(layer_list[11][1][1])
bn13v = tf.convert_to_tensor(layer_list[12][1][1])
bn14v = tf.convert_to_tensor(layer_list[13][1][1])
bn15v = tf.convert_to_tensor(layer_list[14][1][1])
bn16v = tf.convert_to_tensor(layer_list[15][1][1])
bn17v = tf.convert_to_tensor(layer_list[16][1][1])
bn18v = tf.convert_to_tensor(layer_list[17][1][1])
bn19v = tf.convert_to_tensor(layer_list[18][1][1])
bn20v = tf.convert_to_tensor(layer_list[19][1][1])
bn21v = tf.convert_to_tensor(layer_list[20][1][1])
bn22v = tf.convert_to_tensor(layer_list[21][1][1])

"""
wtcn1 = create_weights(shape=[3, 3, 3, 32])
wtcn2 = create_weights(shape=[3, 3, 32, 64])
wtcn3 = create_weights(shape=[3, 3, 64, 128])
wtcn4 = create_weights(shape=[1, 1, 128, 64])
wtcn5 = create_weights(shape=[3, 3, 64, 128])
wtcn6 = create_weights(shape=[3, 3, 128, 256])
wtcn7 = create_weights(shape=[1, 1, 256, 128])
wtcn8 = create_weights(shape=[3, 3, 128, 256])
wtcn9 = create_weights(shape=[3, 3, 256, 512])
wtcn10 = create_weights(shape=[1, 1, 512, 256])
wtcn11 = create_weights(shape=[3, 3, 256, 512])
wtcn12 = create_weights(shape=[3, 3, 512, 1024])
"""

wtcn23 = create_weights(shape=[1, 1, 1024, fin_size])
bias23 = create_biases(fin_size)



weights_dict = {"wtcn1": wtcn1, "wtcn2": wtcn2, "wtcn3": wtcn3, "wtcn4": wtcn4, "wtcn5": wtcn5, "wtcn6": wtcn6,
                "wtcn7": wtcn7, "wtcn8": wtcn8, "wtcn9": wtcn9, "wtcn10": wtcn10, "wtcn11": wtcn11, "wtcn12": wtcn12,
                "wtcn13": wtcn13, "wtcn14": wtcn14, "wtcn15": wtcn15, "wtcn16": wtcn16, "wtcn17": wtcn17,
                "wtcn18": wtcn18, "wtcn19": wtcn19, "wtcn20": wtcn20, "wtcn21": wtcn21, "wtcn22": wtcn22,
                "wtcn23": wtcn23}
biases_dict = {"bn1m": bn1m, "bn2m": bn2m, "bn3m": bn3m, "bn4m": bn4m, "bn5m":bn5m, "bn6m":bn6m, "bn7m":bn7m,
               "bn8m": bn8m, "bn9m": bn9m, "bn10m": bn10m, "bn11m": bn11m, "bn12m": bn12m, "bn13m": bn13m,
               "bn14m": bn14m, "bn15m": bn15m, "bn16m": bn16m, "bn17m": bn17m, "bn18m": bn18m, "bn19m": bn19m,
               "bn20m": bn20m, "bn21m":bn21m, "bn22m":bn22m,
               "bn1v": bn1v, "bn2v": bn2v, "bn3v": bn3v, "bn4v": bn4v, "bn5v":bn5v, "bn6v":bn6v, "bn7v":bn7v,
               "bn8v": bn8v, "bn9v": bn9v, "bn10v": bn10v, "bn11v": bn11v, "bn12v": bn12v, "bn13v": bn13v,
               "bn14v": bn14v, "bn15v": bn15v, "bn16v": bn16v, "bn17v": bn17v, "bn18v": bn18v, "bn19v":bn19v,
               "bn20v": bn20v, "bn21v": bn21v, "bn22v": bn22v, "bn23": bias23}


def gfrc_model(img_in, weights, biases):
    conv1 = conv_bn_lr(img_in, weights['wtcn1'], biases['bn1m'], biases['bn1v'])
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_bn_lr(pool1, weights['wtcn2'], biases['bn2m'], biases['bn2v'])
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = conv_bn_lr(pool2, weights['wtcn3'], biases['bn3m'], biases['bn3v'])
    conv4 = conv_bn_lr(conv3, weights['wtcn4'], biases['bn4m'], biases['bn4v'])
    conv5 = conv_bn_lr(conv4, weights['wtcn5'], biases['bn5m'], biases['bn5v'])
    pool3 = tf.nn.max_pool(value=conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv6 = conv_bn_lr(pool3, weights['wtcn6'], biases['bn6m'], biases['bn6v'])
    conv7 = conv_bn_lr(conv6, weights['wtcn7'], biases['bn7m'], biases['bn7v'])
    conv8 = conv_bn_lr(conv7, weights['wtcn8'], biases['bn8m'], biases['bn8v'])
    pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv9 = conv_bn_lr(pool4, weights['wtcn9'], biases['bn9m'], biases['bn9v'])
    conv10 = conv_bn_lr(conv9, weights['wtcn10'], biases['bn10m'], biases['bn10v'])
    conv11 = conv_bn_lr(conv10, weights['wtcn11'], biases['bn11m'], biases['bn11v'])
    conv12 = conv_bn_lr(conv11, weights['wtcn12'], biases['bn12m'], biases['bn12v'])
    conv13 = conv_bn_lr(conv12, weights['wtcn13'], biases['bn13m'], biases['bn13v'])
    conv14 = conv_bn_lr(conv13, weights['wtcn14'], biases['bn14m'], biases['bn14v'])
    conv15 = conv_bias(conv14, weights['wtcn23'], biases['bn23'])

    return conv15


def gfrc_model2(img_in, weights, biases):
    conv1 = conv_bn_lr(img_in, weights['wtcn1'], biases['bn1m'], biases['bn1v'])
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_bn_lr(pool1, weights['wtcn2'], biases['bn2m'], biases['bn2v'])
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = conv_bn_lr(pool2, weights['wtcn3'], biases['bn3m'], biases['bn3v'])
    conv4 = conv_bn_lr(conv3, weights['wtcn4'], biases['bn4m'], biases['bn4v'])
    conv5 = conv_bn_lr(conv4, weights['wtcn5'], biases['bn5m'], biases['bn5v'])
    pool3 = tf.nn.max_pool(value=conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv6 = conv_bn_lr(pool3, weights['wtcn6'], biases['bn6m'], biases['bn6v'])
    conv7 = conv_bn_lr(conv6, weights['wtcn7'], biases['bn7m'], biases['bn7v'])
    conv8 = conv_bn_lr(conv7, weights['wtcn8'], biases['bn8m'], biases['bn8v'])
    pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv9 = conv_bn_lr(pool4, weights['wtcn9'], biases['bn9m'], biases['bn9v'])
    conv10 = conv_bn_lr(conv9, weights['wtcn10'], biases['bn10m'], biases['bn10v'])
    conv11 = conv_bn_lr(conv10, weights['wtcn11'], biases['bn11m'], biases['bn11v'])
    conv12 = conv_bn_lr(conv11, weights['wtcn12'], biases['bn12m'], biases['bn12v'])
    conv13 = conv_bn_lr(conv12, weights['wtcn13'], biases['bn13m'], biases['bn13v'])
    pool5 = tf.nn.max_pool(value=conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv14 = conv_bn_lr(pool5, weights['wtcn14'], biases['bn14m'], biases['bn14v'])
    conv15 = conv_bn_lr(conv14, weights['wtcn15'], biases['bn15m'], biases['bn15v'])
    conv16 = conv_bn_lr(conv15, weights['wtcn16'], biases['bn16m'], biases['bn16v'])
    conv17 = conv_bn_lr(conv16, weights['wtcn17'], biases['bn17m'], biases['bn17v'])
    conv18 = conv_bn_lr(conv17, weights['wtcn18'], biases['bn18m'], biases['bn18v'])
    conv19 = conv_bn_lr(conv18, weights['wtcn19'], biases['bn19m'], biases['bn19v'])
    conv20 = conv_bn_lr(conv19, weights['wtcn20'], biases['bn20m'], biases['bn20v'])
    conv21 = conv_bn_lr(conv13, weights['wtcn21'], biases['bn21m'], biases['bn21v'])
    s2d1 = tf.space_to_depth(conv21, 2)
    concat1 = tf.concat([conv20, s2d1], axis=-1)
    conv22 = conv_bn_lr(concat1, weights['wtcn22'], biases['bn22m'], biases['bn22v'])
    conv23 = conv_bias(conv22, weights['wtcn23'], biases['bn23'])

    return conv23
