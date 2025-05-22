# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function, Variable
from mmocr.models.builder import LOSSES

EPS = 1e-6

def ohem_single(score, gt_text, training_mask):
    pos_num = int(torch.sum(gt_text > 0.5)) - int(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_num = int(torch.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks


def iou_single(a, b, mask, n_class):
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()

        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.size(0)

    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    iou = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = torch.mean(iou)
    return iou

def _upsample(x, size, scale=1):
    H, W = size
    return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')


@LOSSES.register_module()
class CTLoss(nn.Module):
    def __init__(self,
                 kernel_weight=1.0,
                 loc_weight=0.05,
                 beta=0.1):
        super().__init__()
        self.kernel_weight = kernel_weight
        self.loc_weight = loc_weight
        self.beta = beta
        self.smoothL1Loss = SmoothL1Loss(beta=self.beta, loss_weight=self.loc_weight)
        self.diceLoss = DiceLoss(loss_weight=self.kernel_weight)

    def forward(self, out, downsample_ratio,
                gt_kernels, training_masks,
                gt_instances, gt_kernel_instances,
                training_mask_distances, gt_distances):
        # output
        _, _, H, W = out.shape
        new_H = int(H / downsample_ratio)
        new_W = int(W / downsample_ratio)
        out = _upsample(out, (new_H, new_W))

        kernels = out[:, 0, :, :]
        distances = out[:, 1:, :, :]

        device = out.device
        gt_instances = torch.stack(gt_instances).to(device)
        gt_kernels = torch.stack(gt_kernels).to(device)
        gt_distances = torch.stack(gt_distances).to(device)
        gt_kernel_instances = torch.stack(gt_kernel_instances).to(device)
        training_mask_distances = torch.stack(training_mask_distances).to(device)
        training_masks = torch.stack(training_masks).to(device)

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)
        loss_kernel = self.diceLoss(kernels, gt_kernels, selected_masks, reduce=False)
        iou_kernel = iou((kernels > 0).long(), gt_kernels, training_masks, reduce=False)

        # loc loss
        loss_loc, iou_text = self.smoothL1Loss(distances, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances, reduce=False)

        results = dict(
            loss_loc=loss_kernel,
            loss_kernels=loss_loc
        )
        return results


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        self.coord = nn.Parameter(torch.zeros([640, 640, 2]).long(), requires_grad=False)
        for i in range(640):
            for j in range(640):
                self.coord[i, j, 0] = j
                self.coord[i, j, 1] = i
        self.coord.data = self.coord.view(-1, 2) # (h*w, 2)

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        batch_size = input.size(0)

        diff = torch.abs(input - target) * mask.unsqueeze(1)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                           diff - 0.5 * beta)
        loss = loss.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()
        loss = torch.sum(loss, dim=-1)
        loss = loss / (mask.sum(dim=-1) + eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance, training_mask):

        with torch.no_grad():
            off_points = (self.coord.float() + 10 * distance[:, self.coord[:, 1], self.coord[:, 0]].transpose(1, 0)).long() # (h*w, 2)
            off_points = torch.clamp(off_points, 0, distance.size(-1) - 1)
            selected_mask = (gt_instance[self.coord[:, 1], self.coord[:, 0]] != gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            selected_mask = selected_mask.contiguous().view(1, -1, distance.shape[-1]).long()
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(self, distances, gt_instances, gt_kernel_instances, training_masks, gt_distances, reduce=True):

        selected_training_masks = []
        for i in range(distances.size(0)):
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                   gt_kernel_instances[i, :, :], training_masks[i, :, :])
            )
        selected_training_masks = torch.cat(selected_training_masks, 0).float()

        loss = self.forward_single(distances, gt_distances, selected_training_masks, self.beta)
        loss = self.loss_weight * loss

        with torch.no_grad():
            batch_size = distances.size(0)
            false_num = selected_training_masks.contiguous().view(batch_size, -1)
            false_num = false_num.sum(dim=-1)
            total_num = training_masks.contiguous().view(batch_size, -1).float()
            total_num = total_num.sum(dim=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        if reduce:
            loss = torch.mean(loss)

        return loss, iou_text