# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import box_score_fast, unclip


@POSTPROCESSOR.register_module()
class CTPostprocessor(BasePostprocessor):


    def __init__(self,
                 text_repr_type='poly',
                 **kwargs):
        super().__init__(text_repr_type)

    def __call__(self, preds):
        return preds
