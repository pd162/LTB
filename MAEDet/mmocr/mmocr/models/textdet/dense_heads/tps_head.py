import torch.nn as nn
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textdet.postprocess import decode
from .head_mixin import HeadMixin
from ..postprocess.polygon_decoder import TPS_Decoder, poly_nms
import numpy as np


@HEADS.register_module()
class TPSHead(HeadMixin, BaseModule):

    def __init__(self,
                 in_channels,
                 scales,
                 num_fiducial=14,
                 num_fiducial_gt=14,
                 num_sample=50,
                 num_reconstr_points=20,
                 decoding_type='tps',
                 loss=dict(type='TPSLoss'),
                 tps_size=(1,1),
                 with_direction=False,
                 score_thr=0.3,
                 nms_thr=0.1,
                 alpha=1.0,
                 beta=1.0,
                 num_convs=0,
                 use_sigmod=False,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.num_fiducial = num_fiducial
        self.sample_num = num_sample
        self.tps_size = tps_size
        self.with_direction=with_direction
        self.num_reconstr_points = num_reconstr_points
        loss['num_fiducial'] = num_fiducial
        loss['num_fiducial_gt'] = num_fiducial_gt
        loss['num_sample'] = num_sample
        loss['tps_size'] = tps_size
        loss['with_direction'] = with_direction
        self.use_sigmod = use_sigmod
        if self.use_sigmod:
            loss['use_sigmod'] = use_sigmod
        self.decoding_type = decoding_type
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_convs = num_convs


        if self.use_sigmod:
            p = 1
        else:
            p=2
        self.out_channels_cls = 2 * p
        self.out_channels_reg = (self.num_fiducial + 3) * 2
        if self.with_direction:
            self.out_channels_reg = self.out_channels_reg * 2
            self.out_channels_cls += 2
        self.decoder = TPS_Decoder(self.num_fiducial, self.num_reconstr_points, self.tps_size, test_cfg)

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
            # norm = dict(type='GN', num_groups=32, requires_grad=True)
            norm = None
            for i in range(self.num_convs):
                # cls_convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1))
                # cls_convs.append(nn.BatchNorm2d(self.in_channels))
                # cls_convs.append(nn.ReLU())
                # reg_convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1))
                # reg_convs.append(nn.BatchNorm2d(self.in_channels))
                # reg_convs.append(nn.ReLU())
                cls_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm, act_cfg=dict(type='ReLU')))
                reg_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, norm_cfg=norm, act_cfg=dict(type='ReLU')))
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

        self.init_weights()

    def init_weights(self):
        normal_init(self.out_conv_cls, mean=0, std=0.01)
        normal_init(self.out_conv_reg, mean=0, std=0.01)

    def forward(self, feats):
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

    def resize_grid(self, grids, scale_factor):
        assert isinstance(scale_factor, np.ndarray)
        assert scale_factor.shape[0] == 4
        assert grids[0].shape[-1] == 2

        for g in grids:
            sz = g.shape[0]
            g[:] = g[:] * np.tile(scale_factor[:2], (sz,1))
        return grids

    def get_boundary(self, *args, **kwargs):
        if getattr(self.test_cfg, 'e2e', False):
            return self.get_boundary_for_e2e_test(*args, **kwargs)
        if getattr(self.test_cfg,'with_grid',False):
            return self.get_boundary_with_grids(*args, **kwargs)
        else:
            return self._get_boundary(*args, **kwargs)

    def _get_boundary(self, score_maps, img_metas, rescale, gt_vis=False):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries = boundaries + self._get_boundary_single(
                score_map, scale, gt_vis)

        # nms
        boundaries = poly_nms(boundaries, self.nms_thr)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries)
        return results

    def get_boundary_with_grids(self, score_maps, img_metas, rescale, gt_vis=False):
        assert len(score_maps) == len(self.scales)

        boundaries = []
        grids = []
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundary, grid = self._get_boundary_single(
                score_map, scale, gt_vis)
            boundaries = boundaries + boundary
            if len(grid) > 0:
                grids = grids + [grid]

        # nms
        boundaries, keep_index = poly_nms(boundaries, self.nms_thr, with_index=True)
        if len(grids) > 0:
            grids = np.concatenate(grids, axis=0)[keep_index]

            if rescale:
                boundaries = self.resize_boundary(
                    boundaries, 1.0 / img_metas[0]['scale_factor'])
                grids = self.resize_grid(grids, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries, grids_result=grids)
        return results


    def get_boundary_for_e2e_test(self, score_maps, img_metas, rescale, gt_vis=False):
        assert len(score_maps) == len(self.scales)

        boundaries = [None] * len(self.scales)
        grids = [None] * len(self.scales)
        for idx, score_map in enumerate(score_maps):
            scale = self.scales[idx]
            boundaries[idx], grids[idx] = self._get_boundary_single(
                score_map, scale, False
            )

        results = dict(boundary_results=boundaries, grids_results=grids, scales=self.scales)
        return results




    def _get_boundary_single(self, score_map, scale, gt_vis=False):
        assert len(score_map) == 2
        # if not gt_vis:
        #     assert score_map[1].shape[1] == 2 * (self.num_fiducial + 3)

        return decode(
            decoding_type=self.decoding_type,
            preds=score_map,
            # fourier_degree=self.fourier_degree,
            decoder=self.decoder,
            scale=scale,
            alpha=self.alpha,
            beta=self.beta,
            text_repr_type='poly',
            score_thr=self.score_thr,
            nms_thr=self.nms_thr,
            gt_val=gt_vis,
            with_direction=self.with_direction,
            test_cfg=self.test_cfg
        )
