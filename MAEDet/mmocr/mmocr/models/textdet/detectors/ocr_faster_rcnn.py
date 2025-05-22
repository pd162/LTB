# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.detectors import FasterRCNN

from mmocr.models.builder import DETECTORS
from mmocr.models.textdet.detectors.text_detector_mixin import \
    TextDetectorMixin


@DETECTORS.register_module()
class OCRFasterRCNN(TextDetectorMixin, FasterRCNN):
    """Faster RCNN tailored for OCR."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 text_repr_type='quad',
                 show_score=False,
                 init_cfg=None):
        TextDetectorMixin.__init__(self, show_score)
        FasterRCNN.__init__(
            self,
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

    def simple_test(self, img, img_metas, proposals=None, rescale=False):

        results = super().simple_test(img, img_metas, proposals, rescale)
        boundaries = self.get_boundary(results[0])
        boundaries = boundaries if isinstance(boundaries, list) else [boundaries]
        return boundaries

    def get_boundary(self, results):
        instance_num = len(results[0])
        boundaries = []
        for i in range(instance_num):
            boundary = self.xy2boundary(results[0][i])
            if boundary is not None:
                boundaries.append(boundary)
        results = dict(boundary_result=boundaries)
        return results

    def xy2boundary(self, res):
        '''

        :param res: np.ndarray(1, 5)
        :return: boundary: list[9]
        '''
        x1 = res[0]
        y1 = res[1]
        x2 = res[2]
        y2 = res[3]
        score = res[-1]
        # 考虑最简单的 axis-rectangle 情况
        boundary = [x1, y1, x2, y1, x2, y2, x1, y2, score]
        return boundary
