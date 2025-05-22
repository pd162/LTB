dataset_type = 'OCRDataset'
data_root = 'path/to/LTB'

icdar2013_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2013_train.json',
    data_prefix=dict(img_path='icdar2013/imgs/training'),
    pipeline=None)

icdar2013_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2013_test.json',
    data_prefix=dict(img_path='icdar2013/imgs/test'),
    test_mode=True,
    pipeline=None)

icdar2015_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2015_train.json',
    data_prefix=dict(img_path='icdar2015/imgs/training'),
    pipeline=None)

icdar2015_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2015_test.json',
    data_prefix=dict(img_path='icdar2015/imgs/test'),
    test_mode=True,
    pipeline=None)

totaltext_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/totaltext_train.json',
    data_prefix=dict(img_path='totaltext/imgs/training'),
    pipeline=None)

totaltext_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/totaltext_test.json',
    data_prefix=dict(img_path='totaltext/imgs/test'),
    test_mode=True,
    pipeline=None)

cocotext_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/cocotext_train.json',
    data_prefix=dict(img_path='train2014'),
    pipeline=None)

cocotext_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/cocotext_test.json',
    data_prefix=dict(img_path='train2014'),
    test_mode=True,
    pipeline=None)

icdar2019_art_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2019_art_train.json',
    data_prefix=dict(img_path='ART/train_images'),
    pipeline=None)

icdar2019_art_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2019_art_test.json',
    data_prefix=dict(img_path='ART/train_images'),
    test_mode=True,
    pipeline=None)

icdar2019_lsvt_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2019_lsvt_train.json',
    data_prefix=dict(img_path='LSVT/train_full_images/train_images/'),
    pipeline=None)

icdar2019_lsvt_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2019_lsvt_test.json',
    data_prefix=dict(img_path='LSVT/train_full_images/train_images/'),
    test_mode=True,
    pipeline=None)

textocr_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/textocr_train.json',
    data_prefix=dict(img_path='TextOCR'),
    pipeline=None)

textocr_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/textocr_test.json',
    data_prefix=dict(img_path='TextOCR'),
    test_mode=True,
    pipeline=None)

icdar2017_mlt_en_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2017_mlt_en_train.json',
    data_prefix=dict(img_path='MLT17'),
    pipeline=None)

icdar2017_mlt_en_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2017_mlt_en_test.json',
    data_prefix=dict(img_path='MLT17'),
    test_mode=True,
    pipeline=None)

icdar2019_mlt_en_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2019_mlt_en_train.json',
    data_prefix=dict(img_path='MLT19/train_image'),
    pipeline=None)

icdar2019_mlt_en_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/icdar2019_mlt_en_test.json',
    data_prefix=dict(img_path='MLT19/train_image'),
    test_mode=True,
    pipeline=None)

challenging_blurry_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_blurry.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_artistic_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_artistic.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_single_char_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_single_char.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_glass_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_glass.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_distorted_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_distorted.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_delimited_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_delimited.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_inverse_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_inverse.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_dense_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_dense.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_overlapped_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_overlapped.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_occluded_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_occluded.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_low_contrast_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_low_contrast.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_complex_bg_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_complex_bg.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_others_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_others.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_hard_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_hard.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)

challenging_norm_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='anno/mmocr1.x_format/challenging_norm.json',
    data_prefix=dict(img_path='anno/hard/images/'),
    test_mode=True,
    pipeline=None)


train_list = [icdar2013_train,
              icdar2015_train,
              totaltext_train,
              cocotext_train,
              textocr_train,
              icdar2019_art_train,
              icdar2019_lsvt_train,
              icdar2017_mlt_en_train,
              icdar2019_mlt_en_train, ]

train_dict = dict(
    icdar2013_train=icdar2013_train,
    icdar2015_train=icdar2015_train,
    totaltext_train=totaltext_train,
    cocotext_train=cocotext_train,
    textocr_train=textocr_train,
    icdar2019_art_train=icdar2019_art_train,
    icdar2019_lsvt_train=icdar2019_lsvt_train,
    icdar2017_mlt_en_train=icdar2017_mlt_en_train,
    icdar2019_mlt_en_train=icdar2019_mlt_en_train,
)

test_list = [icdar2013_test,
             icdar2015_test,
             totaltext_test,
             cocotext_test,
             textocr_test,
             icdar2019_art_test,
             icdar2019_lsvt_test,
             icdar2017_mlt_en_test,
             icdar2019_mlt_en_test, ]

test_dict = dict(
    icdar2013_test=icdar2013_test,
    icdar2015_test=icdar2015_test,
    totaltext_test=totaltext_test,
    cocotext_test=cocotext_test,
    textocr_test=textocr_test,
    icdar2019_art_test=icdar2019_art_test,
    icdar2019_lsvt_test=icdar2019_lsvt_test,
    icdar2017_mlt_en_test=icdar2017_mlt_en_test,
    icdar2019_mlt_en_test=icdar2019_mlt_en_test,
)

challenging_text_list = [
    challenging_blurry_test,
    challenging_artistic_test,
    challenging_glass_test,
    challenging_single_char_test,
    challenging_distorted_test,
    challenging_inverse_test,
    challenging_delimited_test,
    challenging_dense_test,
    challenging_overlapped_test,
    challenging_occluded_test,
    challenging_low_contrast_test,
    challenging_complex_bg_test,
    challenging_others_test,
    challenging_hard_test,
    challenging_norm_test,
]

challenging_dict = dict(
    challenging_blurry_test=challenging_blurry_test,
    challenging_artistic_test=challenging_artistic_test,
    challenging_glass_test=challenging_glass_test,
    challenging_single_char_test=challenging_single_char_test,
    challenging_distorted_test=challenging_distorted_test,
    challenging_inverse_test=challenging_inverse_test,
    challenging_delimited_test=challenging_delimited_test,
    challenging_dense_test=challenging_dense_test,
    challenging_overlapped_test=challenging_overlapped_test,
    challenging_occluded_test=challenging_occluded_test,
    challenging_low_contrast_test=challenging_low_contrast_test,
    challenging_complex_bg_test=challenging_complex_bg_test,
    challenging_others_test=challenging_others_test,
    challenging_hard_test=challenging_hard_test,
    challenging_norm_test=challenging_norm_test,
)

icdar2013_train_list = [icdar2013_train]
icdar2013_test_list = [icdar2013_test]

icdar2015_train_list = [icdar2015_train]
icdar2015_test_list = [icdar2015_test]

totaltext_train_list = [totaltext_train]
totaltext_test_list = [totaltext_test]

cocotext_train_list = [cocotext_train]
cocotext_test_list = [cocotext_test]

textocr_train_list = [textocr_train]
textocr_test_list = [textocr_test]

icdar2019_art_train_list = [icdar2019_art_train]
icdar2019_art_test_list = [icdar2019_art_test]

icdar2019_lsvt_train_list = [icdar2019_lsvt_train]
icdar2019_lsvt_test_list = [icdar2019_lsvt_test]

icdar2017_mlt_en_train_list = [icdar2017_mlt_en_train]
icdar2017_mlt_en_test_list = [icdar2017_mlt_en_test]

icdar2019_mlt_en_train_list = [icdar2019_mlt_en_train]
icdar2019_mlt_en_test_list = [icdar2019_mlt_en_test]

all_test_list = [
    icdar2013_test,
    icdar2015_test,
    totaltext_test,
    cocotext_test,
    textocr_test,
    icdar2019_art_test,
    icdar2019_lsvt_test,
    icdar2017_mlt_en_test,
    icdar2019_mlt_en_test,
    challenging_blurry_test,
    challenging_artistic_test,
    challenging_glass_test,
    challenging_single_char_test,
    challenging_inverse_test,
    challenging_distorted_test,
    challenging_delimited_test,
    challenging_dense_test,
    challenging_overlapped_test,
    challenging_occluded_test,
    challenging_low_contrast_test,
    challenging_complex_bg_test,
    challenging_others_test,
    challenging_hard_test,
    challenging_norm_test,
]



"""
# https://mmocr.readthedocs.io/en/dev-1.x/basic_concepts/datasets.html#ocrdataset

{
    "metainfo":
    {
      "dataset_type": "TextDetDataset",  # Options: TextDetDataset/TextRecogDataset/TextSpotterDataset
      "task_name": "textdet",  #  Options: textdet/textspotter/textrecog
      "category": [{"id": 0, "name": "text"}]  # Used in textdet/textspotter
    },
    "data_list":
    [
      {
        "img_path": "test_img.jpg",
        "height": 604,
        "width": 640,
        "instances":  # multiple instances in one image
        [
          {
            "bbox": [0, 0, 10, 20],  # in textdet/textspotter, [x1, y1, x2, y2].
            "bbox_label": 0,  # The object category, always 0 (text) in MMOCR
            "polygon": [0, 0, 0, 10, 10, 20, 20, 0], # in textdet/textspotter. [x1, y1, x2, y2, ....]
            "text": "mmocr",  # in textspotter/textrecog
            "ignore": False # in textspotter/textdet. Whether to ignore this sample during training
          },
          #...
        ],
      }
      #... multiple images
    ]
}

"""