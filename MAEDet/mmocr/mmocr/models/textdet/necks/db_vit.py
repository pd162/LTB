import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, Sequential, auto_fp16

from mmocr.models.builder import NECKS


@NECKS.register_module()
class ViTNeck(BaseModule):
    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv'),
                     dict(
                         type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        # self.deconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=4, stride=2, padding=1)

    @auto_fp16()
    def forward(self, inputs):
        out = self.conv(inputs)
        return out