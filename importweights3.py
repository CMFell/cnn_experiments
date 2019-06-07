import numpy as np
import tensorflow as tf
import math

def get_weights(wt_path, nfilters, filter_in, filter_sz):
    run_tot = 0
    weight_read = np.fromfile(wt_path, dtype='float32')
    print("new weight", len(weight_read))
    weights_header = weight_read[:4]
    run_tot += 4

    nb_conv = len(nfilters)
    conv_weightz = np.multiply(np.multiply(np.square(filter_sz), filter_in), nfilters)

    layerlist = {}

    for i in range(nb_conv-1):
        size = nfilters[i]
        gamma = weight_read[run_tot:(run_tot+size)]
        #print(beta[0])
        run_tot += size
        beta = weight_read[run_tot:(run_tot+size)]
        run_tot += size
        mean = weight_read[run_tot:(run_tot+size)]
        run_tot += size
        var = weight_read[run_tot:(run_tot+size)]
        run_tot += size
        norm_lst = {"beta": beta, "gamma":gamma, "mean": mean, "var":var}
        lay_name = "norm_" + str(i + 1)
        layerlist[lay_name] = norm_lst
        kernel = weight_read[run_tot:(run_tot+conv_weightz[i])]
        #print(kernel[0])
        run_tot += conv_weightz[i]
        # kernel = kernel.transpose([2, 3, 1, 0])
        conv_name = "conv_" + str(i + 1)
        lay_out = {lay_name: norm_lst, conv_name: kernel}
        #layerlist.append(lay_out)
        layerlist[conv_name] = kernel

    size = nfilters[nb_conv - 1]
    beta = weight_read[run_tot:(run_tot + size)]
    run_tot += size
    norm_lst = {"beta": beta}
    lay_name = "norm_" + str(nb_conv)
    layerlist[lay_name] = norm_lst
    kernel = weight_read[run_tot:(run_tot + conv_weightz[nb_conv - 1])]
    run_tot += conv_weightz[nb_conv - 1]
    # kernel = kernel.transpose([2, 3, 1, 0])
    conv_name = "conv_" + str(nb_conv)
    lay_out = {lay_name: norm_lst, conv_name: kernel}
    #layerlist.append(lay_out)
    layerlist[conv_name] = kernel

    print("remaining_weights = ", len(weight_read) - run_tot)

    return(layerlist)

def create_weights(shape):
    scale = tf.sqrt(2./1024)
    #scale = 1.
    # return tf.Variable(tf.random_uniform(shape, -0.1, 0.08))
    return tf.Variable(tf.truncated_normal(shape, stddev=scale))


def create_biases(size):
    scale = 0.001
    return tf.Variable(tf.constant(0.0, shape=[size]))
    #return tf.Variable(tf.random_uniform([size], -0.08, 0.01))


def process_layers(layer_list, channels, filters_in, filters_out, sizes):
    layers = list(layer_list.keys())
    nlayers = len(channels)
    bias_dict = {}
    conv_dict = {}
    for ll in range(len(layers)):
        lname = layers[ll]
        ltype, lno = lname.split('_')
        if ltype == 'norm':
            biasn = 'bs' + str(lno)
            bias_dict[biasn] = tf.Variable(tf.convert_to_tensor(layer_list[lname]['beta']))
            norm_keys = layer_list[lname].keys()
            #print(biasn, layer_list[lname]['beta'][0], bias_dict[biasn][0])
            if len(norm_keys) > 1:
                sclen = 'sc' + str(lno)
                bias_dict[sclen] = tf.Variable(tf.convert_to_tensor(layer_list[lname]['gamma']))
                rolmn = 'rm' + str(lno)
                bias_dict[rolmn] = tf.Variable(tf.convert_to_tensor(layer_list[lname]['mean']))
                rolvn = 'rv' + str(lno)
                bias_dict[rolvn] = tf.Variable(tf.convert_to_tensor(layer_list[lname]['var']))
        if ltype == 'conv':
            wghtn = 'wt' + str(lno)
            weightz = layer_list[lname]
            lind = int(lno) - 1
            conv_shape = (filters_in[lind], channels[lind], sizes[lind], sizes[lind])
            #conv_shape = (filters_in[lind], sizes[lind], sizes[lind], channels[lind])
            weightz = np.reshape(weightz, conv_shape)
            weightz = weightz.transpose([2, 3, 1, 0])
            #weightz = weightz.transpose([1, 2, 3, 0])
            conv_dict[wghtn] = tf.Variable(tf.convert_to_tensor(weightz))
            #print(wghtn, layer_list[lname][0], weightz[0,0,0,0])
    pencno = 'wt' + str(nlayers -1)
    penbno = 'bs' + str(nlayers -1)
    pensno = 'sc' + str(nlayers -1)
    penmno = 'rm' + str(nlayers -1)
    penvno = 'rv' + str(nlayers -1)
    bias_dict[penbno] = tf.multiply(bias_dict[penbno], 1.0)
    bias_dict[pensno] = tf.multiply(bias_dict[pensno], 1.0)
    bias_dict[penmno] = tf.multiply(bias_dict[penmno], 1.0)
    bias_dict[penvno] = tf.multiply(bias_dict[penvno], 1.0)
    conv_dict[pencno] = tf.multiply(conv_dict[pencno], 1.0)
    lastcno = 'wt' + str(nlayers)
    lastbno = 'bs' + str(nlayers)
    bias_dict[lastbno] = create_biases(filters_out[nlayers - 1])
    conv_dict[lastcno] = create_weights(shape=[1, 1, channels[nlayers - 1], filters_out[nlayers - 1]])
    bias_dict[lastbno] = tf.Variable(bias_dict[lastbno][0:filters_out[nlayers-1]])
    conv_dict[lastcno] = tf.multiply(tf.Variable(conv_dict[lastcno][..., 0:filters_out[nlayers-1]]),1.0)

    return bias_dict, conv_dict


def random_layers(channels, filters_out, sizes):
    nlayers = len(channels)
    bias_dict = {}
    conv_dict = {}
    for ll in range(nlayers):
        biasn = 'bs' + str(ll + 1)
        bias_dict[biasn] = create_biases(filters_out[ll])
        sclen = 'sc' + str(ll + 1)
        bias_dict[sclen] = create_biases(filters_out[ll])
        rolmn = 'rm' + str(ll + 1)
        bias_dict[rolmn] = create_biases(filters_out[ll])
        rolvn = 'rv' + str(ll + 1)
        bias_dict[rolvn] = create_biases(filters_out[ll])
        wghtn = 'wt' + str(ll + 1)
        conv_dict[wghtn] = create_weights(shape=[sizes[ll], sizes[ll], channels[ll], filters_out[ll]])

    return bias_dict, conv_dict






