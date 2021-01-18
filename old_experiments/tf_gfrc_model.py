import tensorflow as tf


def conv_bias(lay_input, weights_in, biases_in, nam_in):
    # Creating the final output convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME', name=nam_in)
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


def conv_bn_lr(lay_input, weights_in, bias, scale, batch_mean, batch_var, nam_in):
    # Creating a default convolutional layer
    layer = tf.nn.conv2d(input=lay_input, filter=weights_in, strides=[1, 1, 1, 1], padding='SAME', name=nam_in)
    # Batch normalisation
    #scale = tf.Variable(tf.ones([1]))
    #beta = tf.Variable(tf.zeros([1]))
    #batch_mean, batch_var = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)
    layer = tf.nn.batch_normalization(layer, batch_mean, batch_var, offset=bias, scale=scale, variance_epsilon=1e-6)
    # activation function
    layer = tf.nn.leaky_relu(layer)
    return layer


def gfrc_model(img_in, weights, biases):
    conv1 = conv_bn_lr(img_in, weights['wt1'], biases['bs1'], biases['sc1'], biases['rm1'], biases['rv1'], "conv1")
    pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpl1')
    conv2 = conv_bn_lr(pool1, weights['wt2'], biases['bs2'], biases['sc2'], biases['rm2'], biases['rv2'], "conv2")
    pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpl2')
    conv3 = conv_bn_lr(pool2, weights['wt3'], biases['bs3'], biases['sc3'], biases['rm3'], biases['rv3'], "conv3")
    conv4 = conv_bn_lr(conv3, weights['wt4'], biases['bs4'], biases['sc4'], biases['rm4'], biases['rv4'], "conv4")
    conv5 = conv_bn_lr(conv4, weights['wt5'], biases['bs5'], biases['sc5'], biases['rm5'], biases['rv5'], "conv5")
    pool3 = tf.nn.max_pool(value=conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpl3')
    conv6 = conv_bn_lr(pool3, weights['wt6'], biases['bs6'], biases['sc6'], biases['rm6'], biases['rv6'], "conv6")
    conv7 = conv_bn_lr(conv6, weights['wt7'], biases['bs7'], biases['sc7'], biases['rm7'], biases['rv7'], "conv7")
    conv8 = conv_bn_lr(conv7, weights['wt8'], biases['bs8'], biases['sc8'], biases['rm8'], biases['rv8'], "conv8")
    pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpl4')
    conv9 = conv_bn_lr(pool4, weights['wt9'], biases['bs9'], biases['sc9'], biases['rm9'], biases['rv9'], "conv9")
    conv10 = conv_bn_lr(conv9, weights['wt10'], biases['bs10'], biases['sc10'], biases['rm10'], biases['rv10'], "conv10")
    conv11 = conv_bn_lr(conv10, weights['wt11'], biases['bs11'], biases['sc11'], biases['rm11'], biases['rv11'], "conv11")
    conv12 = conv_bn_lr(conv11, weights['wt12'], biases['bs12'], biases['sc12'], biases['rm12'], biases['rv12'], "conv12")
    conv13 = conv_bn_lr(conv12, weights['wt13'], biases['bs13'], biases['sc13'], biases['rm13'], biases['rv13'], "conv13")
    pool5 = tf.nn.max_pool(value=conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpl5')
    conv14 = conv_bn_lr(pool5, weights['wt14'], biases['bs14'], biases['sc14'], biases['rm14'], biases['rv14'], "conv14")
    conv15 = conv_bn_lr(conv14, weights['wt15'], biases['bs15'], biases['sc15'], biases['rm15'], biases['rv15'], "conv15")
    conv16 = conv_bn_lr(conv15, weights['wt16'], biases['bs16'], biases['sc16'], biases['rm16'], biases['rv16'], "conv16")
    conv17 = conv_bn_lr(conv16, weights['wt17'], biases['bs17'], biases['sc17'], biases['rm17'], biases['rv17'], "conv17")
    conv18 = conv_bn_lr(conv17, weights['wt18'], biases['bs18'], biases['sc18'], biases['rm18'], biases['rv18'], "conv18")
    conv19 = conv_bn_lr(conv18, weights['wt19'], biases['bs19'], biases['sc19'], biases['rm19'], biases['rv19'], "conv19")
    conv20 = conv_bn_lr(conv19, weights['wt20'], biases['bs20'], biases['sc20'], biases['rm20'], biases['rv20'], "conv20")
    s2d1 = tf.space_to_depth(conv13, 2, name="s2d")
    concat1 = tf.concat([conv20, s2d1], axis=-1, name="cnct")
    conv21 = conv_bn_lr(concat1, weights['wt21'], biases['bs21'], biases['sc21'], biases['rm21'], biases['rv21'], "conv21")
    conv22 = conv_bias(conv21, weights['wt22'], biases['bs22'], "conv22")

    return conv22
