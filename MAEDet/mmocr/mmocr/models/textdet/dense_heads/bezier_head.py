import warnings

import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.core import multi_apply

from mmocr.models.builder import HEADS
from ..postprocess.utils import poly_nms
from .head_mixin import HeadMixin


@HEADS.register_module()
class BezierHead(HeadMixin, BaseModule):

    def __init__(self,
                 in_channels,
                 scales,
                 num_control_points=8,
                 nms_thr=0.1,
                 num_convs=4,
                 only_p3=False,
                 loss=dict(type='BezierLoss', num_sample=20),
                 postprocessor=dict(
                     type='BezierPostprocessor',
                     text_repr_type='poly',
                     num_reconstr_points=20,
                     alpha=1.0,
                     beta=2.0,
                     score_thr=0.3),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     mean=0,
                     std=0.01,
                     override=[
                         dict(name='out_conv_cls'),
                         dict(name='out_conv_reg')
                     ]),
                 **kwargs):

        BaseModule.__init__(self, init_cfg=init_cfg)
        loss['num_control_points'] = num_control_points
        postprocessor['num_control_points'] = num_control_points
        postprocessor['nms_thr'] = nms_thr
        HeadMixin.__init__(self, loss, postprocessor)

        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.num_control_points = num_control_points
        self.num_convs = num_convs
        self.nms_thr = nms_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg



        p=2
        self.out_channels_cls = 2 * p
        self.out_channels_reg = (self.num_control_points) * 2
        self.only_p3 = only_p3
        if self.num_convs > 0:
            # cls_convs = [
            #     nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1) for i in range(num_convs)
            # ]
            # cls_convs.append(nn.ReLU())
            # reg_convs = [
            #     nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1) for i in
            #     range(num_convs)
            # ]
            # reg_convs.append(nn.ReLU())
            cls_convs = []
            reg_convs = []
            for i in range(self.num_convs):
                cls_convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1))
                cls_convs.append(nn.ReLU())
                reg_convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1))
                reg_convs.append(nn.ReLU())
            self.cls_convs = nn.Sequential(*cls_convs)
            self.reg_convs = nn.Sequential(*reg_convs)

        self.out_conv_cls = nn.Conv2d(
            self.in_channels,
            self.out_channels_cls,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)


    def forward(self, feats):
        if self.only_p3:
            feats = [feats[0]]
        cls_res, reg_res = multi_apply(self.forward_single, feats)
        level_num = len(cls_res)
        preds = [[cls_res[i], reg_res[i]] for i in range(level_num)]
        return preds

    def forward_single(self, x):
        if self.num_convs > 0:
            x_cls = self.cls_convs(x)
            x_reg = self.reg_convs(x)
        else:
            x_cls = x
            x_reg = x
        cls_predict = self.out_conv_cls(x_cls)
        reg_predict = self.out_conv_reg(x_reg)
        return cls_predict, reg_predict

    def get_boundary(self, score_maps, img_metas, rescale):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(
                score_map, scale)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def _get_boundary_single(self, score_map, scale):
        assert len(score_map) == 2
        assert score_map[1].shape[1] == 2*self.num_control_points

        return self.postprocessor(score_map, scale)