from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from keras.regularizers import l2
from my_loss_tf import loss_gfrc_yolo
import tensorflow as tf


def create_model(input_shape, n_classes, n_anchors):

    final_out = (5 + n_classes) * n_anchors

    gfrc_yolo = Sequential()
    # layer 0
    gfrc_yolo.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape, use_bias=False,
                         kernel_regularizer=l2(0.0005), kernel_initializer='random_uniform'))
    # layer 1
    gfrc_yolo.add(BatchNormalization())
    # layer 2
    gfrc_yolo.add(LeakyReLU())

    # layer 3
    gfrc_yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # layer 4
    gfrc_yolo.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 5
    gfrc_yolo.add(BatchNormalization())
    # layer 6
    gfrc_yolo.add(LeakyReLU())

    # layer 7
    gfrc_yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # layer 8  kernel_regularizer=l2(0.0005)
    gfrc_yolo.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 9
    gfrc_yolo.add(BatchNormalization())
    # layer 10
    gfrc_yolo.add(LeakyReLU())
    # layer 11
    gfrc_yolo.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 12
    gfrc_yolo.add(BatchNormalization())
    # layer 13
    gfrc_yolo.add(LeakyReLU())
    # layer 14
    gfrc_yolo.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 15
    gfrc_yolo.add(BatchNormalization())
    # layer 16
    gfrc_yolo.add(LeakyReLU())

    # layer 17
    gfrc_yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # layer 18
    gfrc_yolo.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 19
    gfrc_yolo.add(BatchNormalization())
    # layer 20
    gfrc_yolo.add(LeakyReLU())
    # layer 21
    gfrc_yolo.add(Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 22
    gfrc_yolo.add(BatchNormalization())
    # layer 23
    gfrc_yolo.add(LeakyReLU())
    # layer 24
    gfrc_yolo.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 25
    gfrc_yolo.add(BatchNormalization())
    # layer 26
    gfrc_yolo.add(LeakyReLU())

    # layer 27
    gfrc_yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # layer 28
    gfrc_yolo.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 29
    gfrc_yolo.add(BatchNormalization())
    # layer 30
    gfrc_yolo.add(LeakyReLU())
    # layer 31
    gfrc_yolo.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 32
    gfrc_yolo.add(BatchNormalization())
    # layer 33
    gfrc_yolo.add(LeakyReLU())
    # layer 34
    gfrc_yolo.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 35
    gfrc_yolo.add(BatchNormalization())
    # layer 36
    gfrc_yolo.add(LeakyReLU())
    # layer 37
    gfrc_yolo.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 38
    gfrc_yolo.add(BatchNormalization())
    # layer 39
    gfrc_yolo.add(LeakyReLU())
    # layer 40
    gfrc_yolo.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 41
    gfrc_yolo.add(BatchNormalization())
    # layer 42
    gfrc_yolo.add(LeakyReLU())  # 42

    # layer 43
    gfrc_yolo.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='random_uniform'))
    # layer 44
    gfrc_yolo.add(BatchNormalization())
    # layer 45
    gfrc_yolo.add(LeakyReLU())
    # layer 46
    gfrc_yolo.add(Conv2D(final_out, (1, 1), strides=(1, 1), padding='same', use_bias=True, activation='linear',
                         kernel_initializer='random_uniform', bias_initializer='zeros'))

    return gfrc_yolo


def import_weights(gfrc_yolo, layer_list, ignore_last=False):
    gfrc_yolo.layers[0].set_weights(layer_list[0][0])
    gfrc_yolo.layers[1].set_weights(layer_list[0][1])
    gfrc_yolo.layers[4].set_weights(layer_list[1][0])
    gfrc_yolo.layers[5].set_weights(layer_list[1][1])

    gfrc_yolo.layers[8].set_weights(layer_list[2][0])
    gfrc_yolo.layers[9].set_weights(layer_list[2][1])
    gfrc_yolo.layers[11].set_weights(layer_list[3][0])
    gfrc_yolo.layers[12].set_weights(layer_list[3][1])
    gfrc_yolo.layers[14].set_weights(layer_list[4][0])
    gfrc_yolo.layers[15].set_weights(layer_list[4][1])

    gfrc_yolo.layers[18].set_weights(layer_list[5][0])
    gfrc_yolo.layers[19].set_weights(layer_list[5][1])
    gfrc_yolo.layers[21].set_weights(layer_list[6][0])
    gfrc_yolo.layers[22].set_weights(layer_list[6][1])
    gfrc_yolo.layers[24].set_weights(layer_list[7][0])
    gfrc_yolo.layers[25].set_weights(layer_list[7][1])

    gfrc_yolo.layers[28].set_weights(layer_list[8][0])
    gfrc_yolo.layers[29].set_weights(layer_list[8][1])
    gfrc_yolo.layers[31].set_weights(layer_list[9][0])
    gfrc_yolo.layers[32].set_weights(layer_list[9][1])
    gfrc_yolo.layers[34].set_weights(layer_list[10][0])
    gfrc_yolo.layers[35].set_weights(layer_list[10][1])
    gfrc_yolo.layers[37].set_weights(layer_list[11][0])
    gfrc_yolo.layers[38].set_weights(layer_list[11][1])
    gfrc_yolo.layers[40].set_weights(layer_list[12][0])
    gfrc_yolo.layers[41].set_weights(layer_list[12][1])

    gfrc_yolo.layers[43].set_weights(layer_list[13][0])
    gfrc_yolo.layers[44].set_weights(layer_list[13][1])
    if ignore_last is False:
         gfrc_yolo.layers[46].set_weights(layer_list[21])

    return gfrc_yolo
