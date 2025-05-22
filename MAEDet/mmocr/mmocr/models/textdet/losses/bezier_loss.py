import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mmdet.core import multi_apply
from mmocr.models.builder import LOSSES
#from mmocr.models.textdet.losses.chamfer_loss import ChamferLoss2D, BoderLoss2D
# from scipy.spatial import Delaunay
# import neural_renderer as nr
import time

Pi = np.pi


@LOSSES.register_module()
class BezierLoss(nn.Module):

    def __init__(self, num_control_points, num_control_points_gt, num_sample, ohem_ratio=3.,
                 with_set=True, with_p2p=False,p2p_norm=False,
                  chamfer=False, with_p6=False,with_weight=False,
                with_area_weight=False,only_p3=False
                 ):
        super().__init__()
        self.eps = 1e-6
        self.num_control_points = num_control_points
        self.num_control_points_gt = num_control_points_gt
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio
        self.chamfer = chamfer
        self.with_set = with_set
        self.with_p2p = with_p2p
        self.p2p_norm = p2p_norm
        self.with_center_weight = with_weight
        self.with_area_weight = with_area_weight
        self.num_sample_gt = num_sample if chamfer else num_sample * 2
        self.only_p3 = only_p3

        if self.with_set:
            if self.chamfer:
                self.boder_loss = ChamferLoss2D()
            else:
                self.boder_loss = BoderLoss2D()
        self.with_p6 = with_p6


    def forward(self, preds, _, p3_maps, p4_maps=None, p5_maps=None,p6_maps=None, polygons_area=None,**kwargs):
        assert isinstance(preds, list)
        assert p3_maps[0].shape[0] == 2 * (self.num_control_points_gt) + 3 + 1

        device = preds[0][0].device
        # to tensor
        if self.only_p3:
            gts = [p3_maps]
            down_sample_rates = [8]
            k = 1
        else:
            gts = [p3_maps, p4_maps, p5_maps]
            down_sample_rates = [8, 16, 32]
            k = 3
        pad_polygon_maps = None
        pad_boxes = None
        if self.with_area_weight:
            assert polygons_area is not None
            max_num_polygon = max([len(p) for p in polygons_area])
            pad_polygon_areas = torch.zeros(len(polygons_area), max_num_polygon, device=device)
            for bi, po in enumerate(polygons_area):
                if len(po) == 0:
                    continue
                pad_polygon_areas[bi, :len(po)] = torch.from_numpy(polygons_area[bi]).to(device)
        else:
            pad_polygon_areas = None
        gt_polygon_maps = [pad_polygon_maps] * k
        gt_polygons_boxes = [pad_boxes] * k
        gt_polygons_areas = [pad_polygon_areas] * k

        if self.with_p6:
            assert p6_maps is not None
            gts.append(p6_maps)
            gt_polygon_maps.append(pad_polygon_maps)
            gt_polygons_boxes.append(pad_boxes)
            gt_polygons_areas.append(pad_polygon_areas)
            down_sample_rates.append(64)
            # gts.append()
        for idx, maps in enumerate(gts):
            gts[idx] = torch.from_numpy(np.stack(maps)).float().to(device)

        # losses = multi_apply(self.forward_single, preds, gts, gt_polygon_maps, down_sample_rates, gt_polygon_positions)

        losses = multi_apply(self.forward_single, preds, gts,  down_sample_rates, gt_polygons_areas)

        loss_tr = torch.tensor(0., device=device,requires_grad=True).float()
        loss_tcl = torch.tensor(0., device=device,requires_grad=True).float()
        loss_reg_x = torch.tensor(0., device=device,requires_grad=True).float()
        loss_reg_y = torch.tensor(0., device=device,requires_grad=True).float()
        loss_render_dice = torch.tensor(0.0, device=device,requires_grad=True).float()
        loss_boder = torch.tensor(0.0, device=device,requires_grad=True).float()
        loss_const = torch.tensor(0., device=device,requires_grad=True).float()

        for idx, loss in enumerate(losses):
            # if sum(loss ) > 1e3:
            #     loss = loss * 0
            if idx == 0:
                loss_tr = loss_tr + sum(loss)
            elif idx == 1:
                loss_tcl = loss_tcl + sum(loss)
            elif idx == 2:
                loss_reg_x = loss_reg_x + sum(loss)
            elif idx == 3:
                loss_reg_y = loss_reg_y + sum(loss)
            elif idx == 4:
                loss_render_dice = loss_render_dice + sum(loss)
            elif idx == 5:
                loss_boder = loss_boder + sum(loss)
            elif idx == 6:
                loss_const = loss_const + sum(loss)

        results = dict(
            loss_text=loss_tr,
            loss_center=loss_tcl,
            loss_reg_x=loss_reg_x,
            loss_reg_y=loss_reg_y,
            # loss_constraint=loss_constraint
            loss_render_dice=loss_render_dice,
            loss_boder=loss_boder,
        )
        return results

    # def forward_single(self, pred, gt, polygon_maps, downsample_rate,polygon_positions):
    def forward_single(self, pred, gt, downsample_rate=None, areas=None):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        gt = gt.permute(0, 2, 3, 1).contiguous()

        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:4].view(-1, 2)
        # direction_pred = cls_pred[:,:,:,4:].view(-1, 2)
        # x_pred = reg_pred[:, :, :, 0:k].view(-1, k)
        # y_pred = reg_pred[:, :, :, k:2 * k].view(-1, k)
        bezier_pred = reg_pred[:, :, :, :].view(-1, (self.num_control_points) * 2)

        if self.with_area_weight:
            tr_mask_idx = gt[:, :, :, :1].long()
            tr_mask = (tr_mask_idx != 0).view(-1)
            batch_size, H, W, _ = tr_mask_idx.shape
            ys, xs = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
            mesh_grid = torch.stack((xs, ys), dim=-1).long().to(cls_pred.device)
            batch_idx = torch.arange(0, batch_size)[:, None, None].repeat(1, H, W).to(cls_pred.device)
            batch_idx = batch_idx.view(-1)
            tr_mask_idx = tr_mask_idx.view(-1) - 1
        else:
            tr_mask_idx = gt[:, :, :, :1].long()
            tr_mask = (tr_mask_idx != 0).view(-1)

        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        # x_map = gt[:, :, :, 3:3 + k].view(-1, k)
        # y_map = gt[:, :, :, 3 + k:].view(-1, k)
        direction_map = gt[:, :, :, 3].view(-1)
        bezier_map = gt[:, :, :, 4:].view(-1, (self.num_control_points_gt) * 2)


        tr_train_mask = ((train_mask * tr_mask) > 0).float()
        device = bezier_pred.device
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # num_pos = tr_train_mask.sum().item()
        pos_idx = torch.where(tr_train_mask > 0)[0]
        # tcl loss
        loss_tcl = torch.tensor(0.).float().to(device)
        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(
                tcl_pred[pos_idx],
                tcl_mask[pos_idx].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()],
                                           tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg


        # regression loss
        loss_reg_x = torch.tensor(0.,device=device,requires_grad=True).float()
        loss_reg_y = torch.tensor(0.,device=device,requires_grad=True).float()
        render_loss = torch.tensor(0.,device=device,requires_grad=True).float()
        boder_loss = torch.tensor(0.,device=device,requires_grad=True).float()
        const_loss = torch.tensor(0.,device=device,requires_grad=True).float()
        # loss_w = torch.tensor(0.).float().to(device)
        # loss_xw = torch.tensor(0.).float().to(device)
        num_pos = tr_train_mask.sum().item()
        if num_pos > 0:
            bezier_map = bezier_map[pos_idx]
            bezier_pred = bezier_pred[pos_idx]
            direction_map = direction_map[pos_idx]
            if self.with_area_weight:
                batch_idx = batch_idx[pos_idx]
                tr_mask_idx = tr_mask_idx[pos_idx]
            # random sample
            # if num_pos > 128:
            #     select_pos_idx = torch.randperm(pos_idx.shape[0])[:512]
            #     pos_idx = pos_idx[select_pos_idx]
            #     # tr_train_mask[:] = 0
            #     # tr_train_mask[select_pos_idx] = 1
            if self.with_center_weight:
                weight = (tr_mask[pos_idx].float() +
                          tcl_mask[pos_idx].float()) / 2
                weight = weight.contiguous()
            else:
                weight = torch.ones(int(num_pos),dtype=torch.float32, device=bezier_map.device)

            if self.with_area_weight:
                pos_area = areas[batch_idx, tr_mask_idx] / downsample_rate**2
                if torch.any(pos_area<=1):
                    pos_area[pos_area <=1] = 100000000
                num_instance = torch.sum(areas > 0).float()
                if num_instance == 0:
                    return loss_tr, loss_tcl, loss_reg_x, loss_reg_y, render_loss, boder_loss, const_loss
                area_weight = 1.0/pos_area
                weight = weight * area_weight * (1.0/num_instance)
            else:
                weight = weight * 1.0/pos_idx.shape[0]


            p_gt = self.bezier2boder(bezier_map, is_gt=True, direction=direction_map)  # N * (n_sample*4) * 2
            p_pre = self.bezier2boder(bezier_pred, is_gt=False, direction=direction_map)

            boder_gt = p_gt[:,:]
            boder_pre = p_pre[:,:]


            if self.with_set:

                boder_gt = torch.cat(boder_gt.split(self.num_sample_gt, dim=1), dim=0)  # (N*4) * n_sample * 2
                boder_pre = torch.cat(boder_pre.split(self.num_sample, dim=1), dim=0)  # (N*4) * n_sample * 2
                boder_loss = self.boder_loss(boder_pre, boder_gt) # (N*4)
                weight = weight.repeat(2)/2.0
                # boder_loss = torch.mean(weight * boder_loss)
                boder_loss = torch.sum(weight * boder_loss)



            if self.with_p2p:
                if self.p2p_norm:
                    # weight = weight.view(-1, 1)
                    # loss_reg_x = torch.mean(weight * torch.norm(boder_pre[:,:]- boder_gt[:,:], p=2, dim=-1))
                    loss_reg_x = torch.sum(weight * torch.norm(boder_pre[:,:]- boder_gt[:,:], p=2, dim=-1).mean(dim=-1))
                else:
                    weight = weight.view(-1)
                    ft_x, ft_y =boder_gt[:,:,0], boder_gt[:,:,1]
                    ft_x_pre, ft_y_pre = boder_pre[:,:,0], boder_pre[:,:,1]
                    loss_reg_x = torch.sum( weight*F.smooth_l1_loss(
                        ft_x_pre[:, :],
                        ft_x[:, :],
                        reduction='none').mean(-1))
                    loss_reg_y = torch.sum( weight* F.smooth_l1_loss(
                        ft_y_pre[:, :],
                        ft_y[:, :],
                        reduction='none').mean(-1))


        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y, render_loss, boder_loss, const_loss



    def ohem(self, predict, target, train_mask):
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
            loss_pos = torch.tensor(0.)
            loss_neg = F.cross_entropy(
                predict[neg], target[neg], reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()


    def bezier2boder(self, cpts, is_gt=False, direction=None):
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """
        device = cpts.device
        batch_T = cpts.view(-1, self.num_control_points * 2)
        batch_size = batch_T.shape[0]

        t = torch.from_numpy(np.linspace(0, 1, self.num_sample, dtype='float32')).to(device)
        t = t[None].repeat(batch_size, 1)
        top = self.beizer_to_poly(cpts[:, :self.num_control_points], t)
        # t = torch.from_numpy(np.linspace(1, 0, self.num_sample)).to(device)
        # t = t[None].repeat(batch_size, 1)
        bot = self.beizer_to_poly(cpts[:, self.num_control_points:], t)
        batch_xy = torch.cat([top, bot], dim=1)
        return batch_xy


    def beizer_to_poly(self,cpts, t):
        # t = torch.from_numpy(np.linspace(0,1,self.n)).to(device)
        x0, y0, x1, y1, x2, y2, x3, y3 = torch.split(cpts,1,dim=1)
        bezier_x = (1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
                (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3))
        bezier_y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * (
                (1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
        # return np.stack([bezier_x, bezier_y], axis=1)
        return torch.stack([bezier_x, bezier_y], dim=-1)
