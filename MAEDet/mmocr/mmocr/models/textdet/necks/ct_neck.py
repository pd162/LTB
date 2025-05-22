import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from mmocr.models.builder import NECKS


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


@NECKS.register_module()
class FPEM_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v1, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def fpem(self, f1, f2, f3, f4):
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))

        f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
        f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
        f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))
        return f1, f2, f3, f4

    def forward(self, x):
        f1 = self.reduce_layer1(x[0])
        f2 = self.reduce_layer2(x[1])
        f3 = self.reduce_layer3(x[2])
        f4 = self.reduce_layer4(x[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        return f


