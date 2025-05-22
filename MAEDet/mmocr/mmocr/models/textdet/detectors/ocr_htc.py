import torch

from mmdet.models.detectors import HybridTaskCascade

from mmocr.core import seg2boundary
from mmocr.models.builder import DETECTORS
from .text_detector_mixin import TextDetectorMixin
import os
import numpy as np

@DETECTORS.register_module()
class OCRHTC(TextDetectorMixin, HybridTaskCascade):
    """HTC tailored for OCR."""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 text_repr_type='quad',
                 pretrained=None,
                 init_cfg=None):
        super(HybridTaskCascade, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        assert text_repr_type in ['quad', 'poly']
        self.text_repr_type = text_repr_type

    def get_boundary(self, results):
        """Convert segmentation into text boundaries.

        Args:
           results (tuple): The result tuple. The first element is
               segmentation while the second is its scores.
        Returns:
           dict: A result dict containing 'boundary_result'.
        """

        assert isinstance(results, tuple)

        instance_num = len(results[1][0])
        boundaries = []
        # seg_all = results[1][0][0]
        for i in range(instance_num):
            seg = results[1][0][i]
            score = results[0][0][i][-1]
            boundary = seg2boundary(seg, self.text_repr_type, score)
            # seg_all = seg_all + float(score) * seg
            if boundary is not None:
                boundaries.append(boundary)
        results = dict(boundary_result=boundaries)
        return results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        results = super().simple_test(img, img_metas, proposals, rescale)
        boundaries = self.get_boundary(results[0])
        boundaries = boundaries if isinstance(boundaries, list) else [boundaries]
        return boundaries
