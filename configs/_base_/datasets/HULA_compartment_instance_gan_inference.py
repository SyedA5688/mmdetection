# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/syed/'
img_scale = (2048, 2048)
CLASSES = ("Glomerulus", "Arteriole", "Artery")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_generated_1k.json',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_generated_1k.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=24,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_generated_1k.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=24,
        classes=CLASSES))
evaluation = dict(metric=['bbox', 'segm'])  # , classwise=True, classwise_log=True
