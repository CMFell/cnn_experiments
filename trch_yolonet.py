import torch
import torch.nn as nn
import torch.nn.functional as F



class YoloNet(nn.Module):

    def __init__(self):
        super(YoloNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, 1, 1, 0)
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv8_bn = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv10_bn = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv12_bn = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv14_bn = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv15_bn = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv16_bn = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv17_bn = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv18_bn = nn.BatchNorm2d(1024)
        self.conv19 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv19_bn = nn.BatchNorm2d(1024)
        self.conv20 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv20_bn = nn.BatchNorm2d(1024)
        self.conv21 = nn.Conv2d(3072, 1024, 3, 1, 1)
        self.conv21_bn = nn.BatchNorm2d(1024)
        self.conv22 = nn.Conv2d(1024, 30, 1, 1, 0)


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
        return x

"""
net = YoloNet()
#print(net)

input = torch.randn(4, 3, 320, 480)
print(input)

print(input.size())

output = net(input)

print(output.size())
"""
