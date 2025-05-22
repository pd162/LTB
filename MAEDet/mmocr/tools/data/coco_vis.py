import cv2
from mmdet.datasets.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import os

def mkdir(path):
    root = os.path.split(path)[0]
    if not os.path.exists(root):
        mkdir(root)
    if not os.path.exists(path):
        os.mkdir(path)


# coco_anns = COCO('/data/ww/mmocr/data/17mlt/17mlt_val.json')
coco_anns = COCO('/data2/ww/mmocr_04/data/worst_image_annotation.json')
img_ids = coco_anns.get_img_ids()
for i in range(len(img_ids)):
    try:
        img_name = coco_anns.load_imgs([img_ids[i]])[0]['file_name']
        # img = cv2.imread('/data/ww/mmocr/data/17mlt/val_image/'+img_name)
        # img = cv2.imread('/data2/ww/mmocr_04/data/text_detection/'+img_name)
        img = cv2.imread('/data2/ww/mmocr_04/data/worst_image_v0/'+img_name)
        ann_ids = coco_anns.get_ann_ids(img_ids[i])
        _anns = coco_anns.load_anns(ann_ids)
        # polygons = []
        img = img.copy()
        for ann in _anns:
            pts = ann['segmentation']
            if ann.get('is_crowd', 0) == 1:
                continue
            pts = np.array(pts[0]).reshape(-1,2).astype('int')
            # polygons.append(pts)
            # for pt in pts:
            #     cv2.circle(img, pt, 1, (0, 255, 0), 4)
            cv2.polylines(img, [pts.astype('int')], 0, (255, 0, 0), 2)
            # cv2.fillPoly(img, [pts.astype('int')],(255, 255, 255))

        out_path = os.path.split('vis_root/worst/' + img_name)[0]
        mkdir(out_path)
        cv2.imwrite('vis_root/worst/' + img_name, img)
        # plt.imshow(img)
        # plt.show()
    except Exception as e:
        print(e)
        continue

