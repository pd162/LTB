import torch
from torch import nn
import numpy as np


class TPS(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multiplying T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size (height, width) of the rectified image.
    """

    def __init__(self, num_fiducial, rectified_img_size=(1,1),grid_size=(32,100),head_tail=True, with_center=False):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        self.num_fiducial = num_fiducial
        self.head_tail = head_tail
        self.with_center = with_center
        C = self._build_C(self.num_fiducial)  # num_fiducial x 2
        if self.with_center:
            C = self._build_C_v2(self.num_fiducial)
        self.C = np.stack([C, C[:,[1,0]]])
        P = self._build_P(40)
        self.P = np.stack([P, P[:,[1,0]]])
        # for multi-gpu, you need register buffer

        if self.head_tail:
            self.num_fiducial += 6
        if self.with_center:
            self.num_fiducial -= 2
        inv_delta_C_0 = torch.tensor(self._build_inv_delta_C(self.num_fiducial,self.C[0])).float()
        inv_delta_C_1 = torch.tensor(self._build_inv_delta_C(self.num_fiducial, self.C[1])).float()# num_fiducial+3 x num_fiducial+3
        self.register_buffer(
            'inv_delta_C',
            torch.stack([inv_delta_C_0, inv_delta_C_1],dim=0))
        P_hat_0 =  torch.tensor(self._build_P_hat(self.num_fiducial, self.C[0],
                                     self.P[0])).float()  # n x num_fiducial+3
        P_hat_1 = torch.tensor(self._build_P_hat(self.num_fiducial, self.C[1],
                                                 self.P[1])).float()  # n x num_fiducial+3
        self.P_hat = torch.stack([P_hat_0,P_hat_1])  # n x num_fiducial+3
        self.grid_size = grid_size
        P_grid = self._build_P_grid(*grid_size)
        P_hat_grid = torch.tensor(self._build_P_hat(self.num_fiducial, self.C[0],
                                     P_grid)).float()
        self.P_hat_grid = P_hat_grid
        # for fine-tuning with different image width,
        # you may use below instead of self.register_buffer
        # self.inv_delta_C = torch.tensor(
        #     self._build_inv_delta_C(
        #         self.num_fiducial,
        #         self.C)).float().cuda()  # num_fiducial+3 x num_fiducial+3
        # self.P_hat = torch.tensor(
        #     self._build_P_hat(self.num_fiducial, self.C,
        #                       self.P)).float().cuda()  # n x num_fiducial+3

    def _build_C(self, num_fiducial):
        """Return coordinates of fiducial points in rectified_img; C."""
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))*self.rectified_img_width
        ctrl_pts_y_top = -1 * np.ones(int(num_fiducial / 2)) * self.rectified_img_height
        ctrl_pts_y_bottom = np.ones(int(num_fiducial / 2)) * self.rectified_img_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        if not self.head_tail:
            C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            return C  # num_fiducial x 2
        else:
        #
            ctrl_pts_y = (np.linspace(-0.5, 0.5, 3)) * self.rectified_img_height
            ctrl_pts_x_left = -1 * np.ones(ctrl_pts_y.shape[0]) * self.rectified_img_width
            ctrl_pts_x_right = np.ones(ctrl_pts_y.shape[0]) * self.rectified_img_width
            ctrl_pts_left = np.stack([ctrl_pts_x_left, ctrl_pts_y], axis=1)
            ctrl_pts_right = np.stack([ctrl_pts_x_right, ctrl_pts_y], axis=1)
            C = np.concatenate([ctrl_pts_top, ctrl_pts_right, ctrl_pts_bottom, ctrl_pts_left], axis=0)
            return C.reshape([
                -1, 2
            ])


    def _build_C_v2(self, num_fiducial):
        assert num_fiducial == 10
        ctrl_pts_x = np.linspace(-1.0, 1.0, 3) * self.rectified_img_width
        ctrl_pts_y_top = -1 * np.ones(3) * self.rectified_img_height
        ctrl_pts_y_bottom = np.ones(3) * self.rectified_img_height
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_pts_x_center = np.linspace(-1.0,1.0,5)[[1,3]]*self.rectified_img_width
        ctrl_pts_y_center = np.zeros(2)
        ctrl_pts_center = np.stack([ctrl_pts_x_center, ctrl_pts_y_center], axis=1)
        if not self.head_tail:
            C = np.concatenate([ctrl_pts_top,ctrl_pts_center, ctrl_pts_bottom], axis=0)
            return C  # num_fiducial x 2
        else:
            raise NotImplementedError

    def _build_P_grid(self, h, w):
        rectified_img_grid_x = np.linspace(-1, 1, w) * self.rectified_img_width
        rectified_img_grid_y = np.linspace(-1, 1, h) * self.rectified_img_height
        P = np.stack(  # self.rectified_img_w x self.rectified_img_h x 2
            np.meshgrid(rectified_img_grid_x, rectified_img_grid_y),
            axis=2)
        return P.reshape([
            -1, 2
        ])

    def _build_inv_delta_C(self, num_fiducial, C):
        """Return inv_delta_C which is needed to calculate T."""
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # num_fiducial+3 x num_fiducial+3
            [
                np.concatenate([np.ones((num_fiducial, 1)), C, hat_C],
                               axis=1),  # num_fiducial x num_fiducial+3
                np.concatenate([np.zeros(
                    (2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                np.concatenate([np.zeros(
                    (1, 3)), np.ones((1, num_fiducial))],
                               axis=1)  # 1 x num_fiducial+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3

    def _build_P(self, num_fiducial):
        rectified_img_grid_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))*self.rectified_img_width
        # rectified_img_grid_y = (
        #     np.arange(-rectified_img_height, rectified_img_height, 2) +
        #     1.0) / rectified_img_height  # self.rectified_img_height
        # P = np.stack(  # self.rectified_img_w x self.rectified_img_h x 2
        #     np.meshgrid(rectified_img_grid_x, rectified_img_grid_y),
        #     axis=2)
        ctrl_pts_y_top = -1 * np.ones(rectified_img_grid_x.shape[0])*self.rectified_img_height
        ctrl_pts_y_bottom = np.ones(rectified_img_grid_x.shape[0])*self.rectified_img_height
        ctrl_pts_top = np.stack([rectified_img_grid_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([rectified_img_grid_x, ctrl_pts_y_bottom], axis=1)
        P = np.concatenate([ctrl_pts_top, ctrl_pts_bottom[::-1]], axis=0)
        return P.reshape([
            -1, 2
        ])  # n (= self.rectified_img_width x self.rectified_img_height) x 2

    def _build_P_hat(self, num_fiducial, C, P):
        n = P.shape[
            0]  # n (= self.rectified_img_width x self.rectified_img_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, num_fiducial,
                          1))  # n x 2 -> n x 1 x 2 -> n x num_fiducial x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x num_fiducial x 2
        P_diff = P_tile - C_tile  # n x num_fiducial x 2
        rbf_norm = np.linalg.norm(
            P_diff, ord=2, axis=2, keepdims=False)  # n x num_fiducial
        rbf = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # n x num_fiducial
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x num_fiducial+3

    def build_P_prime(self, batch_C_prime, device='cpu', direction=0):
        """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C[direction].repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat[direction].repeat(batch_size, 1, 1)
        batch_P_hat_grid = self.P_hat_grid.repeat(batch_size,1,1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        batch_P_boder = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        batch_P_grid = torch.bmm(batch_P_hat_grid, batch_T)
        return batch_T, batch_P_boder, batch_P_grid  # batch_size x n x 2

    def build_P_prime_p(self, batch_C_prime, P, device='cpu', direction=0):
        batch_size = batch_C_prime.size(0)
        # batch_inv_delta_C = self.inv_delta_C[direction].repeat(batch_size, 1, 1)
        inv_p = torch.from_numpy(self.build_inv_P(self.num_fiducial, P, self.C[0]))[None].repeat(batch_size, 1, 1).float()
        batch_P_hat = self.P_hat[direction].repeat(batch_size, 1, 1)
        batch_P_hat_grid = self.P_hat_grid.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2
        batch_T = torch.bmm(
            inv_p,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        batch_P_boder = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        batch_P_grid = torch.bmm(batch_P_hat_grid, batch_T)
        return batch_T, batch_P_boder, batch_P_grid  # batch_size x n x 2


    def build_inv_P(self,num_fiducial, P, C):
        p_hat = self._build_P_hat(num_fiducial, C, P) # n x (num_fiducial +3)
        p_hat = np.concatenate([p_hat,
                                np.concatenate([np.zeros(
                                    (2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                                np.concatenate([np.zeros(
                                    (1, 3)), np.ones((1, num_fiducial))],
                                    axis=1)  # 1 x num_fiducial+3
                                ])
        inv_p_hat = np.linalg.pinv(p_hat) #(num_fiducial +3) x (n +3)
        return inv_p_hat

