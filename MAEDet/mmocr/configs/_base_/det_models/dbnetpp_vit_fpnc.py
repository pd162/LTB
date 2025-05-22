norm_cfg = dict(type='LN2d', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DBNet',
    backbone=dict(
        type='VisionTransformer',
        img_size=(1280, 1280),
        patch_size=(32, 32),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        pretrained="checkpoint-200.pth"
        # pretrained=None
    ),
    neck=dict(
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(
            type='DBPostprocessor', text_repr_type='poly',
            epsilon_ratio=0.002)),
    train_cfg=None,
    test_cfg=None)
