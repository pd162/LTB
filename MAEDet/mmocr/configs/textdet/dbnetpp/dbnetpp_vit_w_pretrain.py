_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dbnetpp_vit_fpnc.py',
    # '../../_base_/det_models/dbnetpp_r50dcnv2_fpnc.py',
    # '../../_base_/det_datasets/english_scene_text.py',
    # '../../_base_/det_datasets/totaltext.py',
    '../../_base_/det_datasets/english_scene_text_mmlab.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]

img_norm_cfg_r50dcnv2 = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}


test_pipeline_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),  # used by Resize
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
# test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}

# load_from = '/data1/ljh/code/contest/mmocrdev/work_dirs/dbnetpp_proj_pretrain/iter_130000.pth'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))

evaluation = dict(interval=100, metric='hmean-iou')
checkpoint_config = dict(interval=10)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=100)
resume_from = None
find_unused_parameters=True