# Copyright (c) OpenMMLab. All rights reserved.
import math

import lanms
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from mmcv.runner import BaseModule
from torchvision import transforms

from mmocr.models.builder import HEADS, build_loss
from .head_mixin import HeadMixin

# head 有很大问题

@HEADS.register_module()
class EASTHead(HeadMixin, BaseModule):
    """The class for implementing EAST text detector
    EAST(CVPR2017): EAST: An Efficient and Accurate Scene Text Detector
    """

    def __init__(self,
                 in_channels=32,
                 loss=dict(type='EASTLoss'),
                 postprocessor=dict(
                     type='EASTPostprocessor',
                     score_thr=0.3
                 ),
                 train_cfg=None,
                 test_cfg=None):
        # super(EASTHead, self).__init__()
        BaseModule.__init__(self)
        HeadMixin.__init__(self, loss, postprocessor)
        self.loss_module = build_loss(loss)
        self.downsample_ratio = 0.25
        # self.postprocessor = postprocessor

        self.conv1 = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.softmax1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = 512  # 放大倍数 后处理
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 需要完善参数

    def forward(self, x):
        score = self.conv1(x)
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        return [score, geo]

    def is_valid_poly(res, score_shape, scale):
        '''check if the poly in image scope
        Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
        Output:
            True if valid
        '''
        cnt = 0
        for i in range(res.shape[1]):
            if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                    res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
                cnt += 1
        return True if cnt <= 1 else False

    def get_rotate_mat(self, theta):
        '''positive theta value means rotate clockwise'''
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    def restore_polys(self, valid_pos, valid_geo, score_shape, scale=4):
        '''restore polys from feature maps in given positions
        Input:
            valid_pos  : potential text positions <numpy.ndarray, (n,2)>
            valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
            score_shape: shape of score map
            scale      : image / feature map
        Output:
            restored polys <numpy.ndarray, (n,8)>, index
        '''
        polys = []
        index = []
        valid_pos *= scale
        d = valid_geo[:4, :]  # 4 x N
        angle = valid_geo[4, :]  # N,

        for i in range(valid_pos.shape[0]):
            x = valid_pos[i, 0]
            y = valid_pos[i, 1]
            y_min = y - d[0, i]
            y_max = y + d[1, i]
            x_min = x - d[2, i]
            x_max = x + d[3, i]
            rotate_mat = self.get_rotate_mat(-angle[i])

            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += x
            res[1, :] += y

            if self.is_valid_poly(res, score_shape, scale):
                index.append(i)
                polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
        return np.array(polys), index

    def get_boxes(self, score, geo, score_thresh=0.7, nms_thresh=0.3):
        '''get boxes from feature map
        Input:
            score       : score map from model <numpy.ndarray, (1,row,col)>
            geo         : geo map from model <numpy.ndarray, (5,row,col)>
            score_thresh: threshold to segment score map
            nms_thresh  : threshold in nms
        Output:
            boxes       : final polys <numpy.ndarray, (n,9)>
        '''
        score = score[1, :, :]
        sigmoid = nn.Sigmoid()
        score = np.array(sigmoid(torch.from_numpy(score)))
        xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]

        if xy_text.size == 0:
            return None

        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
        valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n

        # 这里耗时长 主要是因为候选框太多了
        polys_restored, index = self.restore_polys(valid_pos, valid_geo, score.shape)
        if polys_restored.size == 0:
            return None

        boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = polys_restored
        boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
        return boxes

    def is_valid_poly(self, res, score_shape, scale):
        '''check if the poly in image scope
        Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
        Output:
            True if valid
        '''
        cnt = 0
        for i in range(res.shape[1]):
            if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                    res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
                cnt += 1
        return True if cnt <= 1 else False

    def adjust_ratio(self, boxes, ratio_w, ratio_h):
        '''refine boxes
        Input:
            boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
        Output:
            refined boxes
        '''
        if boxes is None or boxes.size == 0:
            return None
        boxes[:, [0, 2, 4, 6]] /= ratio_w
        boxes[:, [1, 3, 5, 7]] /= ratio_h
        return np.around(boxes)

    def get_boundary(self, outs, img_metas, rescale):
        score = outs[0]
        geo = outs[1]

        # get_boxes 耗时过长
        boxes = self.get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())

        # 缩放后的后处理
        ratio_h = img_metas[0]['pad_shape'][0] / img_metas[0]['ori_shape'][0]
        ratio_w = img_metas[0]['pad_shape'][1] / img_metas[0]['ori_shape'][1]
        # boxes = self.adjust_ratio(boxes, ratio_w, ratio_h)

        return self.postprocessor(boxes, ratio_w, ratio_h)


