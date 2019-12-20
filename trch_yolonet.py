import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trch_weights import get_weights

class YoloNet(nn.Module):

    def __init__(self, layerz, finsize):
        super(YoloNet, self).__init__()
        # kernel
        def processweights(weightz, flin, chnl, size):
            conv_shape = (flin, chnl, size, size)
            # conv_shape = (filters_in[lind], sizes[lind], sizes[lind], channels[lind])
            weightz = np.reshape(weightz, conv_shape)
            weightz = torch.from_numpy(weightz)
            # weightz = weightz.transpose(0, 1)
            # weightz = weightz.transpose(1, 2)
            return weightz
        def processbias(weightz):
            biaz = torch.from_numpy(weightz)
            return biaz
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv1.weight.data = processweights(layerz["conv_1"], 32, 3, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        # self.conv1_bn.weight.data = processbias(layerz["norm_1"]["gamma"])
        # self.conv1_bn.bias.data = processbias(layerz["norm_1"]["beta"])
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2.weight.data = processweights(layerz["conv_2"], 64, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        # self.conv2_bn.weight.data = processbias(layerz["norm_2"]["gamma"])
        # self.conv2_bn.bias.data = processbias(layerz["norm_2"]["beta"])
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3.weight.data = processweights(layerz["conv_3"], 128, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        # self.conv3_bn.weight.data = processbias(layerz["norm_3"]["gamma"])
        # self.conv3_bn.bias.data = processbias(layerz["norm_3"]["beta"])
        self.conv4 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv4.weight.data = processweights(layerz["conv_4"], 64, 128, 1)
        self.conv4_bn = nn.BatchNorm2d(64)
        # self.conv4_bn.weight.data = processbias(layerz["norm_4"]["gamma"])
        # self.conv4_bn.bias.data = processbias(layerz["norm_4"]["beta"])
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5.weight.data = processweights(layerz["conv_5"], 128, 64, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        # self.conv5_bn.weight.data = processbias(layerz["norm_5"]["gamma"])
        # self.conv5_bn.bias.data = processbias(layerz["norm_5"]["beta"])
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6.weight.data = processweights(layerz["conv_6"], 256, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(256)
        # self.conv6_bn.weight.data = processbias(layerz["norm_6"]["gamma"])
        # self.conv6_bn.bias.data = processbias(layerz["norm_6"]["beta"])
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0)
        self.conv7.weight.data = processweights(layerz["conv_7"], 128, 256, 1)
        self.conv7_bn = nn.BatchNorm2d(128)
        # self.conv7_bn.weight.data = processbias(layerz["norm_7"]["gamma"])
        # self.conv7_bn.bias.data = processbias(layerz["norm_7"]["beta"])
        self.conv8 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv8.weight.data = processweights(layerz["conv_8"], 256, 128, 3)
        self.conv8_bn = nn.BatchNorm2d(256)
        # self.conv8_bn.weight.data = processbias(layerz["norm_8"]["gamma"])
        # self.conv8_bn.bias.data = processbias(layerz["norm_8"]["beta"])
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9.weight.data = processweights(layerz["conv_9"], 512, 256, 3)
        self.conv9_bn = nn.BatchNorm2d(512)
        # self.conv9_bn.weight.data = processbias(layerz["norm_9"]["gamma"])
        # self.conv9_bn.bias.data = processbias(layerz["norm_9"]["beta"])
        self.conv10 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv10.weight.data = processweights(layerz["conv_10"], 256, 512, 1)
        self.conv10_bn = nn.BatchNorm2d(256)
        # self.conv10_bn.weight.data = processbias(layerz["norm_10"]["gamma"])
        # self.conv10_bn.bias.data = processbias(layerz["norm_10"]["beta"])
        self.conv11 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv11.weight.data = processweights(layerz["conv_11"], 512, 256, 3)
        self.conv11_bn = nn.BatchNorm2d(512)
        # self.conv11_bn.weight.data = processbias(layerz["norm_11"]["gamma"])
        # self.conv11_bn.bias.data = processbias(layerz["norm_11"]["beta"])
        self.conv12 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv12.weight.data = processweights(layerz["conv_12"], 256, 512, 1)
        self.conv12_bn = nn.BatchNorm2d(256)
        # self.conv12_bn.weight.data = processbias(layerz["norm_12"]["gamma"])
        # self.conv12_bn.bias.data = processbias(layerz["norm_12"]["beta"])
        self.conv13 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv13.weight.data = processweights(layerz["conv_13"], 512, 256, 3)
        self.conv13_bn = nn.BatchNorm2d(512)
        # self.conv13_bn.weight.data = processbias(layerz["norm_13"]["gamma"])
        # self.conv13_bn.bias.data = processbias(layerz["norm_13"]["beta"])
        self.conv14 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv14.weight.data = processweights(layerz["conv_14"], 1024, 512, 3)
        self.conv14_bn = nn.BatchNorm2d(1024)
        # self.conv14_bn.weight.data = processbias(layerz["norm_14"]["gamma"])
        # self.conv14_bn.bias.data = processbias(layerz["norm_14"]["beta"])
        self.conv15 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv15.weight.data = processweights(layerz["conv_15"], 512, 1024, 1)
        self.conv15_bn = nn.BatchNorm2d(512)
        # self.conv15_bn.weight.data = processbias(layerz["norm_15"]["gamma"])
        # self.conv15_bn.bias.data = processbias(layerz["norm_15"]["beta"])
        self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv16.weight.data = processweights(layerz["conv_16"], 1024, 512, 3)
        self.conv16_bn = nn.BatchNorm2d(1024)
        # self.conv16_bn.weight.data = processbias(layerz["norm_16"]["gamma"])
        # self.conv16_bn.bias.data = processbias(layerz["norm_16"]["beta"])
        self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv17.weight.data = processweights(layerz["conv_17"], 512, 1024, 1)
        self.conv17_bn = nn.BatchNorm2d(512)
        # self.conv17_bn.weight.data = processbias(layerz["norm_17"]["gamma"])
        # self.conv17_bn.bias.data = processbias(layerz["norm_17"]["beta"])
        self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv18.weight.data = processweights(layerz["conv_18"], 1024, 512, 3)
        self.conv18_bn = nn.BatchNorm2d(1024)
        # self.conv18_bn.weight.data = processbias(layerz["norm_18"]["gamma"])
        # self.conv18_bn.bias.data = processbias(layerz["norm_18"]["beta"])
        self.conv19 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv19.weight.data = processweights(layerz["conv_19"], 1024, 1024, 3)
        self.conv19_bn = nn.BatchNorm2d(1024)
        # self.conv19_bn.weight.data = processbias(layerz["norm_19"]["gamma"])
        # self.conv19_bn.bias.data = processbias(layerz["norm_19"]["beta"])
        self.conv20 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv20.weight.data = processweights(layerz["conv_20"], 1024, 1024, 3)
        self.conv20_bn = nn.BatchNorm2d(1024)
        # self.conv20_bn.weight.data = processbias(layerz["norm_20"]["gamma"])
        # self.conv20_bn.bias.data = processbias(layerz["norm_20"]["beta"])
        self.conv21 = nn.Conv2d(3072, 1024, 3, 1, 1)
        self.conv21.weight.data = processweights(layerz["conv_21"], 1024, 3072, 3)
        self.conv21_bn = nn.BatchNorm2d(1024)
        # self.conv21_bn.weight.data = processbias(layerz["norm_21"]["gamma"])
        # self.conv21_bn.bias.data = processbias(layerz["norm_21"]["beta"])
        self.conv22 = nn.Conv2d(1024, finsize, 1, 1, 0)
        # self.conv22.weight.data = processweights(layerz["conv_22"], 30, 1024, 1)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = F.max_pool2d(F.leaky_relu(self.conv1_bn(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv2_bn(self.conv2(x))), (2, 2))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(F.leaky_relu(self.conv5_bn(self.conv5(x))), (2, 2))
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)))
        x = F.max_pool2d(F.leaky_relu(self.conv8_bn(self.conv8(x))), (2, 2))
        x = F.leaky_relu(self.conv9_bn(self.conv9(x)))
        x = F.leaky_relu(self.conv10_bn(self.conv10(x)))
        x = F.leaky_relu(self.conv11_bn(self.conv11(x)))
        x = F.leaky_relu(self.conv12_bn(self.conv12(x)))
        x = F.leaky_relu(self.conv13_bn(self.conv13(x)))
        skip = x
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv14_bn(self.conv14(x)))
        x = F.leaky_relu(self.conv15_bn(self.conv15(x)))
        x = F.leaky_relu(self.conv16_bn(self.conv16(x)))
        x = F.leaky_relu(self.conv17_bn(self.conv17(x)))
        x = F.leaky_relu(self.conv18_bn(self.conv18(x)))
        x = F.leaky_relu(self.conv19_bn(self.conv19(x)))
        x = F.leaky_relu(self.conv20_bn(self.conv20(x)))
        #print(skip.size)
        # outest = skip.size()
        skip = torch.chunk(skip, 2, dim=2)
        skip = torch.cat(skip, dim=1)
        skip = torch.chunk(skip, 2, dim=3)
        skip = torch.cat(skip, dim=1)
        x = torch.cat((x, skip), dim=1)
        x = F.leaky_relu(self.conv21_bn(self.conv21(x)))
        x = F.leaky_relu(self.conv22(x))
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        return x


class YoloNetSimp(nn.Module):

    def __init__(self, layerz):
        super(YoloNetSimp, self).__init__()
        # kernel
        def processweights(weightz, flin, chnl, size):
            conv_shape = (flin, chnl, size, size)
            # conv_shape = (filters_in[lind], sizes[lind], sizes[lind], channels[lind])
            weightz = np.reshape(weightz, conv_shape)
            weightz = torch.from_numpy(weightz)
            # weightz = weightz.transpose(0, 1)
            # weightz = weightz.transpose(1, 2)
            return weightz
        def processbias(weightz):
            biaz = torch.from_numpy(weightz)
            return biaz
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv1.weight.data = processweights(layerz["conv_1"], 32, 3, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_bn.weight.data = processbias(layerz["norm_1"]["gamma"])
        self.conv1_bn.bias.data = processbias(layerz["norm_1"]["beta"])
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2.weight.data = processweights(layerz["conv_2"], 64, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_bn.weight.data = processbias(layerz["norm_2"]["gamma"])
        self.conv2_bn.bias.data = processbias(layerz["norm_2"]["beta"])
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3.weight.data = processweights(layerz["conv_3"], 128, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_bn.weight.data = processbias(layerz["norm_3"]["gamma"])
        self.conv3_bn.bias.data = processbias(layerz["norm_3"]["beta"])
        self.conv4 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv4.weight.data = processweights(layerz["conv_4"], 64, 128, 1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv4_bn.weight.data = processbias(layerz["norm_4"]["gamma"])
        self.conv4_bn.bias.data = processbias(layerz["norm_4"]["beta"])
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5.weight.data = processweights(layerz["conv_5"], 128, 64, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv5_bn.weight.data = processbias(layerz["norm_5"]["gamma"])
        self.conv5_bn.bias.data = processbias(layerz["norm_5"]["beta"])
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6.weight.data = processweights(layerz["conv_6"], 256, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_bn.weight.data = processbias(layerz["norm_6"]["gamma"])
        self.conv6_bn.bias.data = processbias(layerz["norm_6"]["beta"])
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0)
        self.conv7.weight.data = processweights(layerz["conv_7"], 128, 256, 1)
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv7_bn.weight.data = processbias(layerz["norm_7"]["gamma"])
        self.conv7_bn.bias.data = processbias(layerz["norm_7"]["beta"])
        self.conv8 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv8.weight.data = processweights(layerz["conv_8"], 256, 128, 3)
        self.conv8_bn = nn.BatchNorm2d(256)
        self.conv8_bn.weight.data = processbias(layerz["norm_8"]["gamma"])
        self.conv8_bn.bias.data = processbias(layerz["norm_8"]["beta"])
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9.weight.data = processweights(layerz["conv_9"], 512, 256, 3)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv9_bn.weight.data = processbias(layerz["norm_9"]["gamma"])
        self.conv9_bn.bias.data = processbias(layerz["norm_9"]["beta"])
        self.conv10 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv10.weight.data = processweights(layerz["conv_10"], 256, 512, 1)
        self.conv10_bn = nn.BatchNorm2d(256)
        self.conv10_bn.weight.data = processbias(layerz["norm_10"]["gamma"])
        self.conv10_bn.bias.data = processbias(layerz["norm_10"]["beta"])
        self.conv11 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv11.weight.data = processweights(layerz["conv_11"], 512, 256, 3)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv11_bn.weight.data = processbias(layerz["norm_11"]["gamma"])
        self.conv11_bn.bias.data = processbias(layerz["norm_11"]["beta"])
        self.conv12 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv12.weight.data = processweights(layerz["conv_12"], 256, 512, 1)
        self.conv12_bn = nn.BatchNorm2d(256)
        self.conv12_bn.weight.data = processbias(layerz["norm_12"]["gamma"])
        self.conv12_bn.bias.data = processbias(layerz["norm_12"]["beta"])
        self.conv13 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv13.weight.data = processweights(layerz["conv_13"], 512, 256, 3)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.conv13_bn.weight.data = processbias(layerz["norm_13"]["gamma"])
        self.conv13_bn.bias.data = processbias(layerz["norm_13"]["beta"])
        self.conv14 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv14.weight.data = processweights(layerz["conv_14"], 1024, 512, 3)
        self.conv14_bn = nn.BatchNorm2d(1024)
        self.conv14_bn.weight.data = processbias(layerz["norm_14"]["gamma"])
        self.conv14_bn.bias.data = processbias(layerz["norm_14"]["beta"])
        self.conv15 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv15.weight.data = processweights(layerz["conv_15"], 512, 1024, 1)
        self.conv15_bn = nn.BatchNorm2d(512)
        self.conv15_bn.weight.data = processbias(layerz["norm_15"]["gamma"])
        self.conv15_bn.bias.data = processbias(layerz["norm_15"]["beta"])
        # self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1)
        # self.conv16.weight.data = processweights(layerz["conv_16"], 1024, 512, 3)
        # self.conv16_bn = nn.BatchNorm2d(1024)
        # self.conv16_bn.weight.data = processbias(layerz["norm_16"]["gamma"])
        # self.conv16_bn.bias.data = processbias(layerz["norm_16"]["beta"])
        # self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0)
        # self.conv17.weight.data = processweights(layerz["conv_17"], 512, 1024, 1)
        # self.conv17_bn = nn.BatchNorm2d(512)
        # self.conv17_bn.weight.data = processbias(layerz["norm_17"]["gamma"])
        # self.conv17_bn.bias.data = processbias(layerz["norm_17"]["beta"])
        # self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1)
        # self.conv18.weight.data = processweights(layerz["conv_18"], 1024, 512, 3)
        # self.conv18_bn = nn.BatchNorm2d(1024)
        # self.conv18_bn.weight.data = processbias(layerz["norm_18"]["gamma"])
        # self.conv18_bn.bias.data = processbias(layerz["norm_18"]["beta"])
        # self.conv19 = nn.Conv2d(1024, 1024, 3, 1, 1)
        # self.conv19.weight.data = processweights(layerz["conv_19"], 1024, 1024, 3)
        # self.conv19_bn = nn.BatchNorm2d(1024)
        # self.conv19_bn.weight.data = processbias(layerz["norm_19"]["gamma"])
        # self.conv19_bn.bias.data = processbias(layerz["norm_19"]["beta"])
        # self.conv20 = nn.Conv2d(1024, 1024, 3, 1, 1)
        # self.conv20.weight.data = processweights(layerz["conv_20"], 1024, 1024, 3)
        # self.conv20_bn = nn.BatchNorm2d(1024)
        # self.conv20_bn.weight.data = processbias(layerz["norm_20"]["gamma"])
        # self.conv20_bn.bias.data = processbias(layerz["norm_20"]["beta"])
        self.conv21 = nn.Conv2d(1536, 1024, 3, 1, 1)
        #self.conv21.weight.data = processweights(layerz["conv_21"], 1024, 3072, 3)
        self.conv21_bn = nn.BatchNorm2d(1024)
        # self.conv21_bn.weight.data = processbias(layerz["norm_21"]["gamma"])
        # self.conv21_bn.bias.data = processbias(layerz["norm_21"]["beta"])
        self.conv22 = nn.Conv2d(1024, 30, 1, 1, 0)
        # self.conv22.weight.data = processweights(layerz["conv_22"], 30, 1024, 1)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = F.max_pool2d(F.leaky_relu(self.conv1_bn(x)), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv2_bn(self.conv2(x))), (2, 2))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = F.max_pool2d(F.leaky_relu(self.conv5_bn(self.conv5(x))), (2, 2))
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)))
        x = F.leaky_relu(self.conv8_bn(self.conv8(x)))
        skip = x
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv9_bn(self.conv9(x)))
        x = F.leaky_relu(self.conv10_bn(self.conv10(x)))
        x = F.leaky_relu(self.conv11_bn(self.conv11(x)))
        x = F.leaky_relu(self.conv12_bn(self.conv12(x)))
        x = F.leaky_relu(self.conv13_bn(self.conv13(x)))
        # skip = x
        # x = F.max_pool2d(x, (2, 2))
        # x = F.leaky_relu(self.conv14_bn(self.conv14(x)))
        # x = F.leaky_relu(self.conv15_bn(self.conv15(x)))
        # x = F.leaky_relu(self.conv16_bn(self.conv16(x)))
        # x = F.leaky_relu(self.conv17_bn(self.conv17(x)))
        # x = F.leaky_relu(self.conv18_bn(self.conv18(x)))
        # x = F.leaky_relu(self.conv19_bn(self.conv19(x)))
        # x = F.leaky_relu(self.conv20_bn(self.conv20(x)))
        # outest = skip.size()
        skip = torch.chunk(skip, 2, dim=2)
        skip = torch.cat(skip, dim=1)
        skip = torch.chunk(skip, 2, dim=3)
        skip = torch.cat(skip, dim=1)
        # print(skip.size())
        x = torch.cat((x, skip), dim=1)
        x = F.leaky_relu(self.conv21_bn(self.conv21(x)))
        x = F.leaky_relu(self.conv22(x))
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        return x

"""
weightspath = "E:/CF_Calcs/BenchmarkSets/GFRC/ToUse/Train/yolo-gfrc_6600.weights"
layerlist = get_weights(weightspath)

net = YoloNetSimp(layerlist)
#print(net)

input = torch.randn(4, 3, 320, 480)
print(input)

print(input.size())

output = net(input)

print(output.size())

"""