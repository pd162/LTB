from mmdet.datasets.coco import COCO
import json


dataset_type = 'IcdarDataset'

mlt19_train = dict(
    type=dataset_type,
    ann_file='data/text_detection/MLT19/19mlt_english_chinese_train.json',
    img_prefix='data/text_detection/MLT19/train_english_chinese_images',
    pipeline=None)

mlt19_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/MLT19/19mlt_english_chinese_test.json',
    img_prefix='MLT19/test_english_chinese_images',
    pipeline=None)

icdar13_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ICDAR13/icdar_train.json',
    img_prefix='ICDAR13/train_images',
    pipeline=None)

icdar13_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ICDAR13/icdar_test.json',
    img_prefix='ICDAR13/test_images',
    pipeline=None)

icdar15_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ICDAR15/icdar15_train.json',
    img_prefix='ICDAR15/ch4_training_images',
    pipeline=None)

icdar15_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ICDAR15/icdar15_test.json',
    img_prefix='ICDAR15/ch4_test_images',
    pipeline=None)

ART_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ART/art_split_train.json',
    img_prefix='ART/art_train_images',
    pipeline=None)

ART_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ART/art_split_test.json',
    img_prefix='ART/art_test_images',
    pipeline=None)

rctw_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/RCTW/rctw_split_train.json',
    img_prefix='RCTW/train_images',
    pipeline=None)

rctw_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/RCTW/rctw_split_test.json',
    img_prefix='RCTW/test_images',
    pipeline=None)

icpr_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ICPR/icpr_train.json',
    img_prefix='ICPR/image_9000',
    pipeline=None)

icpr_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ICPR/icpr_test.json',
    img_prefix='ICPR/train_1000/image_1000',
    pipeline=None)

lsvt_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/LSVT/lsvt_split_train.json',
    img_prefix='LSVT/train_images',
    pipeline=None)

lsvt_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/LSVT/lsvt_split_test.json',
    img_prefix='LSVT/test_images',
    pipeline=None)

rects_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ReCTs/rects_train.json',
    img_prefix='ReCTs/img',
    pipeline=None)

rects_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/ReCTs/rects_val.json',
    img_prefix='ReCTs/img',
    pipeline=None)

dast1500_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/dast1500/dast1500_train.json',
    img_prefix='dast1500/train_image_and_gt/image',
    pipeline=None)

dast1500_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/dast1500/dast1500_test.json',
    img_prefix='dast1500/test_image_and_gt/image',
    pipeline=None)

cocotext_train = dict(
    type=dataset_type,
    ann_file='./data/text_detection/coco-text/coco_text_train.json',
    img_prefix='coco-text/train_images',
    pipeline=None)

cocotext_test = dict(
    type=dataset_type,
    ann_file='./data/text_detection/coco-text/coco_text_test.json',
    img_prefix='coco-text/test_images',
    pipeline=None)

# train_list = [icpr_train]

train_list = [mlt19_train, icdar13_train, icdar15_train, ART_train, rctw_train,
              icpr_train, lsvt_train, rects_train, dast1500_train, cocotext_train]

# test_list = [mlt19_test, icdar13_test, icdar15_test, ART_test, rctw_test,
#             icpr_test, lsvt_test, rects_test, dast1500_test, cocotext_test]
test_list = [mlt19_test, ART_test, rctw_test,
            icpr_test, lsvt_test, rects_test, dast1500_test]
# test_list = [ART_test, icpr_test,mlt19_test,lsvt_test,rctw_test, rects_test]
# test_list = [icpr_test]


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False


def main():
    all_image = []
    all_annotation = []
    all_img_id = 0
    all_annotations_id = 0
    for test in test_list:
        # file = json.load(open(test['ann_file'], 'r'))
        coco_anns = COCO(test['ann_file'])
        image_ids = coco_anns.get_img_ids()

        prefix = test['img_prefix']
        # image = file['images']
        # anns = file['annotations']
        print(len(image_ids))
        for idx in image_ids:
            im = coco_anns.load_imgs([idx])[0]
            ann_ids = coco_anns.get_ann_ids(idx)
            for ann_id in ann_ids:
                ann = coco_anns.load_anns(ann_id)[0]
                ann['image_id'] = all_img_id
                ann['id'] = all_annotations_id
                all_annotations_id += 1

                # if not check_contain_chinese(ann['transcription']):
                #     ann['is_crowd'] = 1
                all_annotation.append(ann)
            im['id'] = all_img_id
            im['file_name'] = prefix + '/' + im['file_name']
            all_img_id += 1
            all_image.append(im)

    results = {
        'images' : all_image,
        'annotations': all_annotation,
        'categories' :[
        {
            "supercategory": "text",
            "id": 1,
            "name": "text"
        }
    ]
    }
    print(len(all_image))
    # with open('data/only_chinese_test.json','w') as f:
    #     json.dump(results,f)

main()