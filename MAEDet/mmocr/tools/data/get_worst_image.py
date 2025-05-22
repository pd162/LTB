import json
import shutil
import os
import Polygon
import numpy as np
from PIL import Image

root_path = '/data2/ww/mmocr_04/data/text_detection'
out_path = '/data2/ww/mmocr_04/data/worst_image_v0'

n = 1000

datasets = {
    "AR":80,
    "da":40,
    "IC":80,
    "LS":480,
    "ML":32,
    "RC":128,
    "Re":159
}


def main():
    result = json.load(open('/data2/ww/mmocr_04/work_dirs/all_chinese_test_rank.json'))
    annotations = []
    images = []
    annotation_idx = 0
    image_idx = 0
    for i in range(len(result)):
        re = result[i]
        image_name = re['file_name']
        da = image_name[:2]
        annotation = re['annotation']
        if len(annotation['labels']) == 0:
            continue
        if datasets[da] > 0:
            datasets[da] -= 1
        else:
            continue
        name = os.path.split(image_name)[-1]
        name = da + '_'+name
        out_img = os.path.join(out_path, name)
        # if os.path.exists(out_img):
        #     raise ValueError("duplicate name")
        shutil.copy(os.path.join(root_path, image_name),os.path.join(out_path, name))
        for j in range(len(annotation['bboxes'])):
            bbox = annotation['bboxes'][j]
            annotations.append({
                'image_id':image_idx,
                'bboxes': [bbox[0],bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                'segmentation' :annotation['masks'][j],
                'is_crowd':0,
                "category_id": 1,
                'id':annotation_idx,
                'area': Polygon.Polygon(np.array(annotation['masks'][j][0]).reshape(-1,2)).area()
            }
            )
            annotation_idx += 1
        for j in range(len(annotation['bboxes_ignore'])):
            bbox = annotation['bboxes_ignore'][j]
            annotations.append({
                'image_id': image_idx,
                'bboxes': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'segmentation': annotation['masks_ignore'][j],
                'is_crowd': 1,
                "category_id": 1,
                'id': annotation_idx,
                'area': Polygon.Polygon(np.array(annotation['masks_ignore'][j]).reshape(-1, 2)).area()
            }
            )
            annotation_idx += 1
        # image = re['image']
        img = Image.open(out_img)
        images.append({
            'file_name':name,
            'width':img.width,
            'height':img.height,
            'id':image_idx
        })
        image_idx += 1
    out_all = {
        "images":images,
        "annotations":annotations,
        "categories": [
            {
                "supercategory": "text",
                "id": 1,
                "name": "text"
            }
        ],

    }
    with open('/data2/ww/mmocr_04/data/worst_image_annotation.json','w') as f:
        json.dump(out_all, f)



main()