import numpy as np


def load_weights_from_file(weights_path, nfilters, filter_sizes):

    # Load weights and config.
    # print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    weights_header = np.ndarray(shape=(4,), dtype='int32', buffer=weights_file.read(16))
    print('Weights Header: ', weights_header)

    layer_list = []

    # adapted from yad2k
    for ly in range(len(filter_sizes)):
        filters = nfilters[ly]
        conv_bias = np.ndarray(shape=(filters,), dtype='float32', buffer=weights_file.read(filters * 4))

        bn_weights = np.ndarray(shape=(3, filters), dtype='float32', buffer=weights_file.read(filters * 12))

        bn_weight_list = [bn_weights[0],  # scale gamma
                          conv_bias,  # shift beta
                          bn_weights[1],  # running mean
                          bn_weights[2]]  # running var

        size = filter_sizes[ly]
        if 0 == ly:
            prev_layer = 3
        elif 20 == ly:
            prev_layer = 512
        elif 21 == ly:
            prev_layer = 1280
        else:
            prev_layer = nfilters[ly - 1]

        print("ly=", ly+1, " fil in=", prev_layer, " fil out=", nfilters[ly])

        layer_shape = (filters, prev_layer, size, size)
        weights_size = np.product(layer_shape)

        conv_weights = np.ndarray(shape=layer_shape, dtype='float32', buffer=weights_file.read(weights_size * 4))
        conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

        conv_layer = [conv_weights]

        layer_save = [conv_layer, bn_weight_list]

        layer_list.append(layer_save)

    conv_bias = np.ndarray(shape=(55,), dtype='float32', buffer=weights_file.read(55 * 4))
    layer_shape = (55, 1024, 1, 1)
    weights_size = np.product(layer_shape)

    conv_weights = np.ndarray(shape=layer_shape, dtype='float32', buffer=weights_file.read(weights_size * 4))
    conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

    conv_layer = [conv_weights, conv_bias]

    layer_list.append(conv_layer)

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    print("remaining_weights = ", remaining_weights)

    return layer_list


"""
# store unused
layer_weights = [864, 128, 18432, 256, 73728, 512, 8192, 256, 73728, 512, 294912, 1024, 32768, 512, 294912, 1024,
                 1179648, 2048, 131072, 1024, 1179648, 2048, 131072, 1024, 1179648, 2048, 4718592, 4096, 524288, 2048,
                 4718592, 4096, 524288, 2048, 4718592, 4096, 9437184, 4096, 9437184, 4096, 32768, 256, 11796480, 4096,
                 9295]

conv_sizes = [864, 18432, 73728, 8192, 73728, 294912, 32768, 294912, 1179648, 131072, 1179648, 131072, 1179648,
              4718592, 524288, 4718592, 524288, 4718592, 9437184, 9437184, 32768, 11796480, 56320]

"""