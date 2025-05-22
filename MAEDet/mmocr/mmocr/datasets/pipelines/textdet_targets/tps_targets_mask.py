import Polygon
import cv2
import numpy as np
from numpy.fft import fft
from numpy.linalg import norm
import torch
from torch import nn
import torch.nn.functional as F
import mmocr.utils.check_argument as check_argument
from mmdet.datasets.builder import PIPELINES
from .textsnake_targets import TextSnakeTargets
from mmdet.core.mask.structures import PolygonMasks
from scipy.interpolate import splprep, splev

PI = 3.1415926
from .TPS_warp import TPS


@PIPELINES.register_module()
class TPSTargetsMask(TextSnakeTargets):

    def __init__(self,
                 num_fiducial=14,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0)),
                 tps_size=(1, 1),  # h,w
                 with_direction=False,
                 with_mask=True,
                 with_area=True,
                 interp=False,
                 head_tail=True,
                 short_range=False
                 ):

        super().__init__()
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
        self.head_tail = head_tail
        self.TPSGenerator = TPS(num_fiducial, tps_size, head_tail=self.head_tail)
        if self.head_tail:
            self.num_fiducial += 6
        self.interp = interp
        self.short_range = short_range

    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                                     head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                                     head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                        resampled_top_line[i + 1] -
                        center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                        resampled_bot_line[i + 1] -
                        center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def resample_polygon(self, top_line, bot_line):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        if self.head_tail:
            n = (self.num_fiducial - 6) // 2
        else:
            n = self.num_fiducial // 2
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
                    print(p, pn)
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

    def normalize_polygon(self, polygon):

        temp_polygon = polygon - polygon.mean(axis=0)

        return temp_polygon / 32

    def poly2T(self, polygon, direction):
        """Convert polygon to tps cofficients

        Args:
            polygon (ndarray): An input polygon.
            center_point (tuple(int, int)): centerpoint of default box.
            side (float): side length of default box
        Returns:
            c (ndarray): Tps coefficients.
        """
        C_prime = polygon.reshape((1, -1, 2))
        T, build_P_prime, batch_inver_delta_C = self.TPSGenerator.build_P_prime(
            torch.from_numpy(C_prime), direction=0
        )  # batch_size x n (= rectified_img_width x rectified_img_height) x 2
        # build_P_prime_reshape = build_P_prime.reshape([
        #     build_P_prime.size(0), self.rectified_img_size[0],
        #     self.rectified_img_size[1], 2
        # ])
        return T, build_P_prime, batch_inver_delta_C

    def poly2rotate_rect(self, polygon):
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        return box

    def clockwise(self, head_edge, tail_edge, top_sideline, bot_sideline):
        # if self.with_direction:
        hc = head_edge.mean(axis=0)
        tc = tail_edge.mean(axis=0)
        d = (((hc - tc) ** 2).sum()) ** 0.5 + 0.1
        dx = np.abs(hc[0] - tc[0])
        if not dx / d <= 1:
            print(dx / d)
        angle = np.arccos(dx / d)
        direction = 0 if angle <= PI / 4 else 1  # 0 horizontal, 1 vertical
        # else:
        #     direction = 0
        # if direction == 1:
        #     print('vertical')
        # points left to right or top to bottom
        if top_sideline[0, direction] > top_sideline[-1, direction]:
            top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
        else:
            top_indx = np.arange(0, top_sideline.shape[0])
        top_sideline = top_sideline[top_indx]
        if not self.with_direction and direction == 1 and top_sideline[0, direction] < top_sideline[-1, direction]:
            top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
            top_sideline = top_sideline[top_indx]

        if bot_sideline[0, direction] > bot_sideline[-1, direction]:
            bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
        else:
            bot_indx = np.arange(0, bot_sideline.shape[0])
        bot_sideline = bot_sideline[bot_indx]
        if not self.with_direction and direction == 1 and bot_sideline[0, direction] < bot_sideline[-1, direction]:
            bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
            bot_sideline = bot_sideline[bot_indx]

        # edge: left < right, top < bottom
        if top_sideline[:, 1 - direction].mean() > bot_sideline[:, 1 - direction].mean():
            top_sideline, bot_sideline = bot_sideline, top_sideline

        if not self.with_direction:
            direction = 0
        return top_sideline, bot_sideline, direction

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
        resample_top_line, resample_bot_line = self.resample_polygon(top_line, bot_line)
        head_line = np.linspace(resample_top_line[0], resample_bot_line[0], 5)[1:-1]
        tail_line = np.linspace(resample_top_line[-1], resample_bot_line[-1], 5)[1:-1]
        # resampled_polygon = np.concatenate([resample_top_line, tail_line, resample_bot_line, head_line])
        if self.head_tail:
            resampled_polygon = np.concatenate([resample_top_line, tail_line, resample_bot_line, head_line])
        else:
            resampled_polygon = np.concatenate([resample_top_line, resample_bot_line])
        # resampled_polygon = self.normalize_polygon(polygon)
        assert resampled_polygon.shape[0] == self.num_fiducial, "resample failed"
        # resampled_polygon = polygon
        # import warnings
        # warnings.warn('resample failed')
        # import matplotlib.pyplot as plt
        # plt.plot(polygon[:, 0], polygon[:, 1])
        # plt.plot(resampled_polygon[:, 0], resampled_polygon[:, 1])
        # plt.scatter(resampled_polygon[:, 0], resampled_polygon[:, 1])
        # plt.show()
        tps_coeff, build_P_prime, batch_inver_delta_C = self.poly2T(resampled_polygon, direction)
        # fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
        # fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)
        #
        # real_part = np.real(fourier_coeff).reshape((-1, 1))
        # image_part = np.imag(fourier_coeff).reshape((-1, 1))
        # fourier_signature = np.hstack([real_part, image_part])

        return tps_coeff.view(-1, 1), build_P_prime, batch_inver_delta_C

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4, "The number of points should larger than 4"
        assert points.shape[1] == 2
        assert points.shape[0] % 2 == 0, "The data number should be 2 times"
        # lh = points.shape[0]
        # lhc2 = int(lh / 2)
        # top_sideline = points[:lhc2]
        # bot_sideline = points[lhc2:][::-1]
        # head_edge = np.stack((top_sideline[0], bot_sideline[0]),0)
        # tail_edge = np.stack((top_sideline[-1], bot_sideline[-1]),0)
        head_edge, tail_edge, top_sideline, bot_sideline = super(TPSTargetsMask, self).reorder_poly_edge(points)
        return head_edge, tail_edge, top_sideline, bot_sideline

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
        coeff_maps = np.zeros((2 * self.num_fiducial + 6, h, w), dtype=np.float32)
        direction_maps = np.zeros((1, h, w), dtype=np.int32)
        tps_coeffs = []
        for poly, poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            head_edge, tail_edge, top_sideline, bot_sideline = self.reorder_poly_edge(polygon[0])
            top_sideline, bot_sideline, direction = self.clockwise(head_edge, tail_edge, top_sideline, bot_sideline)
            # direction = 0
            # if direction == 1:
            #     print('vertical')
            # top_bot_l2r = np.concatenate([top_sideline, bot_sideline])
            cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
            tps_coeff, build_P_hat, batch_inv_delta_C = self.cal_tps_signature(top_sideline, bot_sideline, direction)
            tps_coeffs.append(np.insert(tps_coeff.view(-1), 0, poly_idx))

            yx = np.argwhere(mask > 0.5)
            y, x = yx[:, 0], yx[:, 1]
            batch_T = torch.zeros(h, w, self.num_fiducial + 3, 2)
            batch_T[y, x, :, :] = tps_coeff.view(-1, 2)
            batch_T[y, x, 0, :] = batch_T[y, x, 0, :] - yx[:, [1, 0]].astype(np.float32)
            batch_T = batch_T.view(h, w, -1).permute(2, 0, 1)
            coeff_maps[:, y, x] = batch_T[:, y, x]
            direction_maps[:, y, x] = direction

        if len(tps_coeffs) > 0:
            tps_coeffs = np.stack(tps_coeffs, 0)
        else:
            tps_coeffs = np.array([])
        return coeff_maps, direction_maps, tps_coeffs

    def generate_text_region_mask(self, img_size, text_polys, text_polys_idx, polygons_mask):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)
        big_masks = []

        for poly, poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(
                np.round(text_instance), dtype=np.int32).reshape((1, -1, 2))
            if self.with_mask or self.with_area:
                cv2.fillPoly(text_region_mask, polygon, poly_idx)
            else:
                cv2.fillPoly(text_region_mask, polygon, 1)
        return text_region_mask

    def generate_level_targets(self, img_size, text_polys, ignore_polys, img=None):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
            :param img:
        """
        import time
        # t = time.time()
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_text_polys_idx = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        polygons_mask = []
        polygons_box = []
        polygons_area = []
        zeros_mask = np.zeros((h, w), dtype=np.float32)
        level_maps = []
        lv_tps_coeffs = [[] for i in range(len(lv_size_divs))]
        for poly_idx, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int32).reshape((1, -1, 2))
            tl_x, tl_y, box_w, box_h = cv2.boundingRect(polygon)
            # assert box_w <= 200 or box_h <= 200, 'Box out of range'
            # max_l = max(box_h, box_w)
            if self.short_range:
                proportion = min(box_h, box_w) / (h + 1e-8)
            else:
                proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append([poly[0] / lv_size_divs[ind]])
                    lv_text_polys_idx[ind].append(poly_idx + 1)

            # polygons_mask.append(cv2.fillPoly(zeros_mask.copy(), [polygon], 1))
            # polygons_box.append(np.array([tl_x-5, tl_y-5, box_w+tl_x+5, box_h+tl_y+5]))
            # polygons_mask.append([polygon.flatten()])
            if self.with_area:
                polygon_area = Polygon.Polygon(poly[0].reshape(-1, 2)).area()
                polygons_area.append(polygon_area)
            if self.with_mask:
                tl_x, tl_y, box_w, box_h = tl_x - 2, tl_y - 2, box_w + 4, box_h + 4
                h_scale = self.mask_out[0] / max(box_h, 0.1)  # avoid too large scale
                w_scale = self.mask_out[1] / max(box_w, 0.1)
                resize_crop_polygon = poly[0].copy()

                resize_crop_polygon[0::2] -= tl_x
                resize_crop_polygon[1::2] -= tl_y
                resize_crop_polygon[0::2] *= w_scale
                resize_crop_polygon[1::2] *= h_scale
                polygons_mask.append([resize_crop_polygon])
                polygons_box.append(np.array([tl_x, tl_y, w_scale, h_scale]))

            # polygons_position.append(positions)
            # polygons_mask.append(img[:,:,0])
        # print(time.time()-t)
        # t = time.time()

        for ignore_poly in ignore_polys:
            assert len(ignore_poly) == 1
            text_instance = [[ignore_poly[0][i], ignore_poly[0][i + 1]]
                             for i in range(0, len(ignore_poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append(
                        [ignore_poly[0] / lv_size_divs[ind]])
        # print(time.time() - t)
        # t = time.time()
        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)
            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind], lv_text_polys_idx[ind], polygons_mask)[None]
            current_level_maps.append(text_region)
            # print('text_region: ' + str(time.time() - t))
            # t = time.time()
            center_region = self.generate_center_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)
            # print('center region: '+ str(time.time() - t))
            # t = time.time()
            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)
            # print('effective mask:' + str(time.time() - t))
            # t = time.time()
            tps_coeff_maps, direction_maps, tps_coeffs = self.generate_tps_maps(
                level_img_size, lv_text_polys[ind], lv_text_polys_idx[ind])
            current_level_maps.append(direction_maps)
            current_level_maps.append(tps_coeff_maps)
            lv_tps_coeffs[ind] = tps_coeffs
            # current_level_maps.append(fourier_image_maps)
            # print('tps map' + str(time.time() - t))
            # t = time.time()
            level_maps.append(np.concatenate(current_level_maps))
        # if len(polygons_mask) > 0:
        # polygons_mask = np.stack(polygons_mask, axis=0)
        # polygons_position = np.stack(polygons_position, axis=0)
        # lv_text_polys_idx
        # else:
        #     polygons_mask = np.array([])
        #     polygons_position = np.array([])
        if self.with_mask:
            polygons_mask = PolygonMasks(polygons_mask, *self.mask_out)
        else:
            polygons_mask = None
        if len(polygons_box) > 0:
            polygons_box = np.stack(polygons_box)
        else:
            polygons_box = np.array([])
        if len(polygons_area) > 0:
            polygons_area = np.array(polygons_area)
        else:
            polygons_area = np.array([])
        # polygons_mask = polygons_mask.crop_and_resize(polygons_box, (100,100),range(len(polygon)))
        lv_text_polys_idx = [np.array(l) for l in lv_text_polys_idx]
        # print(time.time()-t)
        return level_maps, polygons_mask, lv_text_polys_idx, polygons_box, polygons_area, lv_tps_coeffs

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)

        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        h, w, _ = results['img_shape']

        level_maps, all_polygons_map, lv_text_polys_idx, polygons_boxes, polygons_area, lv_tps_coeffs = self.generate_level_targets(
            (h, w), polygon_masks,
            polygon_masks_ignore, results['img'])

        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        if len(self.level_size_divisors) == 1:
            mapping = {
                'p3_maps': level_maps[0],
                'polygon_maps': all_polygons_map,
                'lv_text_polys_idx': lv_text_polys_idx,
                'polygons_boxes': polygons_boxes,
                'polygons_area': polygons_area,
                'lv_tps_coeffs': lv_tps_coeffs
                # 'ori_polygons': np.stack(polygon_masks)
                # 'polygon_positions':polygon_positions,
            }
        else:
            mapping = {
                'p3_maps': level_maps[0],
                'p4_maps': level_maps[1],
                'p5_maps': level_maps[2],
                'polygon_maps': all_polygons_map,
                'lv_text_polys_idx': lv_text_polys_idx,
                'polygons_boxes': polygons_boxes,
                'polygons_area': polygons_area,
                'lv_tps_coeffs': lv_tps_coeffs
                # 'ori_polygons': np.stack(polygon_masks)
                # 'polygon_positions':polygon_positions,
            }
        if len(self.level_size_divisors) == 4:
            mapping['p6_maps'] = level_maps[3]
        for key, value in mapping.items():
            results[key] = value

        return results
