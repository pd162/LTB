from .tps_targets_mask import TPSTargetsMask
import numpy as np
import cv2
import torch
import mmocr.utils.check_argument as check_argument
from mmdet.datasets.builder import PIPELINES
from scipy.interpolate import splprep, splev

from scipy.special import comb as n_over_k

Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]


def bezier_fit(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2) ** 0.5
    if dt.sum() == 0:
        raise ValueError("bezier fitting failed")
    t = dt / dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))
    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :].flatten().tolist()  # x1 y1, x2 y2
    # return medi_ctp
    cpts = [x[0], y[0]] + medi_ctp + [x[-1], y[-1]]
    return cpts


def beizer_to_poly(cpts, n):
    t = np.linspace(0, 1, n)
    x0, y0, x1, y1, x2, y2, x3, y3 = cpts.flatten().tolist()
    bezier_x = (1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
            (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3))
    bezier_y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * (
            (1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
    return np.stack([bezier_x, bezier_y], axis=1)


@PIPELINES.register_module()
class BezierTargets(TPSTargetsMask):

    def __init__(self,
                 num_fiducial=14,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0)),
                 tps_size=(1, 1),  # h,w
                 with_direction=False,
                 rotate_rect=False,
                 with_mask=True,
                 with_area=True,
                 clock=True,
                 short_range=False,
                 ):

        super().__init__(num_fiducial=num_fiducial,
                         resample_step=resample_step,
                         center_region_shrink_ratio=center_region_shrink_ratio,
                         level_size_divisors=level_size_divisors,
                         level_proportion_range=level_proportion_range,
                         tps_size=tps_size,  # h,w
                         with_direction=with_direction,
                         with_mask=with_mask,
                         with_area=with_area,
                         head_tail=False,
                         short_range=short_range)
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        # self.fourier_degree = fourier_degree
        self.with_mask = with_mask
        self.with_area = with_area
        self.num_fiducial = num_fiducial
        self.resample_step = resample_step
        self.tps_size = tps_size
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range
        self.with_direction = with_direction
        self.mask_out = (100, 100)
        self.clock = clock
        # self.TPSGenerator = TPS(num_fiducial, tps_size)
        # self.num_fiducial += 6

    def resample_polygon(self, top_line, bot_line):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        n = 8
        # num_points = top_bot_l2r.shape[0]
        # assert num_points%2 == 0
        # top_line = top_bot_l2r[:num_points//2]
        # bot_line = top_bot_l2r[num_points//2:]
        resample_line = []
        for polygon in [top_line, bot_line]:
            length = []
            if polygon.shape[0] >= 3 and self.interp:
                x, y = polygon[:, 0], polygon[:, 1]
                tck, u = splprep([x, y], k=3, s=0)
                u = np.linspace(0, 1, num=20, endpoint=True)
                out = splev(u, tck)
                polygon = np.stack(out, axis=1).astype('float32')
            for i in range(len(polygon) - 1):
                p1 = polygon[i]
                if i == len(polygon) - 1:
                    p2 = polygon[0]
                else:
                    p2 = polygon[i + 1]
                length.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)

            total_length = sum(length)
            # n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
            # n_on_each_line = n_on_each_line.astype(np.int32)
            l_per_line = total_length / (n - 1)
            new_polygon = []

            p = polygon[0]
            new_polygon.append(p)
            j = 0
            l_pre = 0
            while j < polygon.shape[0] - 1:
                p = polygon[j]
                pn = polygon[j + 1]
                dxy = pn - p
                l = np.sqrt(sum(dxy ** 2))
                if l == 0:
                    # print(p,pn)
                    j += 1
                    continue
                s = l_per_line / l
                if l_pre < 0:
                    point = p + dxy * (-l_pre / l)
                    l += l_pre
                    if l >= 0:
                        new_polygon.append(point)
                        p = point
                    else:
                        j += 1
                        l_pre = l
                        continue
                l -= l_per_line
                while l >= 0:
                    point = p + dxy * s
                    p = point
                    new_polygon.append(point)
                    l -= l_per_line
                j += 1
                l_pre = l
            if len(new_polygon) < n:
                new_polygon.append(polygon[-1])
            if len(new_polygon) < n:
                print(polygon)
            resample_line.append(np.array(new_polygon))

        # resample_line = np.concatenate(resample_line)

        return resample_line  # top line, bot line

    def cal_tps_signature(self, top_line, bot_line, direction):
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
import matplotlib.pyplot as plt
plt.plot(polygon[:,0], polygon[:,1])
plt.plot(resampled_polygon[:,0], resampled_polygon[:,1])
plt.scatter(resampled_polygon[:,0], resampled_polygon[:,1])
plt.show()
        """
        top_line, bot_line = self.resample_polygon(top_line, bot_line)
        top_cpts = bezier_fit(top_line[:, 0], top_line[:, 1])
        bot_cpts = bezier_fit(bot_line[:, 0], bot_line[:, 1])
        # top_cpts = np.array(top_cpts)
        # bot_cpts = np.array(bot_cpts)
        coeff = np.array(top_cpts + bot_cpts)
        build_top = beizer_to_poly(coeff[:8], 20)
        build_bot = beizer_to_poly(coeff[8:], 20)
        build_p = np.concatenate([build_top, build_bot], axis=0)

        return torch.from_numpy(coeff.astype('float32')), build_p, None

    def generate_tps_maps(self, img_size, text_polys, text_polys_idx=None, img=None, level_size=None):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        # k = self.fourier_degree
        # real_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        # imag_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        coeff_maps = np.zeros((2 * self.num_fiducial, h, w), dtype=np.float32)
        direction_maps = np.zeros((1, h, w), dtype=np.int32)
        tps_coeffs = []
        for poly, poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            head_edge, tail_edge, top_sideline, bot_sideline = self.reorder_poly_edge(polygon[0])
            if self.clock:
                top_sideline, bot_sideline, direction = self.clockwise(head_edge, tail_edge, top_sideline, bot_sideline)
            else:
                direction = 0
            # if direction == 1:
            #     print('vertical')
            # top_bot_l2r = np.concatenate([top_sideline, bot_sideline])
            cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
            tps_coeff, build_P_hat, batch_inv_delta_C = self.cal_tps_signature(top_sideline, bot_sideline, direction)
            tps_coeffs.append(np.insert(tps_coeff.view(-1), 0, poly_idx))

            yx = np.argwhere(mask > 0.5)
            y, x = yx[:, 0], yx[:, 1]
            batch_T = torch.zeros((h, w, self.num_fiducial, 2))
            batch_T[y, x, :, :] = tps_coeff.view(-1, 2)
            batch_T[y, x, :, :] = batch_T[y, x, :, :] - yx[:, None, [1, 0]].astype(np.float32)
            batch_T = batch_T.view(h, w, -1).permute(2, 0, 1)
            coeff_maps[:, y, x] = batch_T[:, y, x]
            direction_maps[:, y, x] = direction

        if len(tps_coeffs) > 0:
            tps_coeffs = np.stack(tps_coeffs, 0)
        else:
            tps_coeffs = np.array([])
        return coeff_maps, direction_maps, tps_coeffs
