# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import fill_hole, fourier2poly, poly_nms


@POSTPROCESSOR.register_module()
class EASTPostprocessor(BasePostprocessor):

    def __init__(self,
                 score_thr=0.3,
                 **kwargs):
        super().__init__()

    def __call__(self, boxes, ratio_w, ratio_h):
        post_res = self.adjust_ratio(boxes, ratio_w, ratio_h)
        if post_res is not None:
            res = []
            for idx, box in enumerate(post_res):
                res.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], box[8]])
            results = dict(boundary_result=res)
            return results
        else:
            res = np.zeros((1, 9)).tolist()
            results = dict(boundary_result=res)
            return results
        # return boundaries

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
