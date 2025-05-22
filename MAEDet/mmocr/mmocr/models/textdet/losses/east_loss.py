# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply
from torch import nn

from mmocr.models.builder import LOSSES
from ....models.common.losses import DiceLoss, FocalLoss


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)


@LOSSES.register_module()
class EASTLoss(nn.Module):
    """The class for implementing EAST text loss
    EAST(CVPR2017): EAST: An Efficient and Accurate Scene Text Detector
    """

    def __init__(self, weight_angle=10, ohem_ratio=3.):
        super().__init__()
        self.weight_angle = weight_angle
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.ohem_ratio = ohem_ratio

    def ohem(self, predict, target, train_mask):
        device = train_mask.device
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(
                predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = min(
                int(neg.float().sum().item()),
                int(self.ohem_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.).to(device)
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, pred_maps, _, gt_score, gt_geo, ignored_map):
        # gt_score = results['gt_score']
        # gt_geo = results['gt_geo']
        # ignored_map = results['ignored_map']

        # list:batchsize (Tensor:(1, 128, 128)) -> Tensor:(batchsize, 1, 128, 128)
        device = pred_maps[0].device
        res_score = torch.zeros(len(gt_score), gt_score[0].size()[0],
                                gt_score[0].size()[1], gt_score[0].size()[2], device=device)
        for idx, score in enumerate(gt_score):
            temp = torch.unsqueeze(gt_score[idx], 0).to(device)
            res_score = torch.cat((res_score, temp), dim=0)

        res_geo = torch.zeros(len(gt_geo), gt_geo[0].size()[0],
                              gt_geo[0].size()[1], gt_geo[0].size()[2], device=device)
        for idx, geo in enumerate(gt_geo):
            temp = torch.unsqueeze(gt_geo[idx], 0).to(device)
            res_geo = torch.cat((res_geo, temp), dim=0)

        res_ignored = torch.zeros(len(ignored_map), ignored_map[0].size()[0],
                                  ignored_map[0].size()[1], ignored_map[0].size()[2], device=device)
        for idx, ignore in enumerate(ignored_map):
            temp = torch.unsqueeze(ignored_map[idx], 0).to(device)
            res_ignored = torch.cat((res_ignored, temp), dim=0)

        gt_score = res_score[len(gt_score):, :, :, :]
        gt_geo = res_geo[len(gt_geo):, :, :, :]
        ignored_map = res_ignored[len(ignored_map):, :, :, :]

        pred_score = pred_maps[0]
        pred_geo = pred_maps[1]

        if torch.sum(gt_score) < 1:
            results = dict(
                loss_cls=torch.tensor(0., device=device, requires_grad=True).float(),
                loss_geo=torch.tensor(0., device=device, requires_grad=True).float()
            )
            return results
        # compute classification loss
        # classify_loss = get_dice_loss(gt_score, pred_score * (1 - ignored_map))
        # gt_score_test = torch.squeeze(gt_score, dim=1).view(-1, 512 * 512)
        # pred_score_test = torch.squeeze(pred_score, dim=1)
        # ignored_map_test = torch.squeeze(ignored_map, dim=1)
        # classify_loss = F.cross_entropy((pred_score_test * (1 - ignored_map_test)).view(-1, 512 * 512), gt_score_test.long())
        # classify_loss = FocalLoss(gt_score, pred_score)
        pred = pred_score.permute(0, 2, 3, 1).contiguous()
        score = gt_score.permute(0, 2, 3, 1).contiguous()
        ignored = (1 - ignored_map).permute(0, 2, 3, 1).contiguous()
        pred = pred.view(-1, 2)
        score = score[:, :, :, 0].view(-1,)
        ignored = ignored[:, :, :, 0].view(-1,)

        classify_loss = self.ohem(pred, score.long(), ignored.long())

        # compute geo loss
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)
        angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss

        # results['loss_cls'] = classify_loss
        # results['loss_geo'] = geo_loss
        total_loss = geo_loss + classify_loss
        results = dict(
            loss_cls=classify_loss,
            loss_geo=geo_loss
        )
        return results

