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

