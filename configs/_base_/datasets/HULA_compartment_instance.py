# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/syed/'
CLASSES = ("Glomerulus", "Arteriole", "Artery")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_tile_train_folds_1-4.json',
        img_prefix='',
        pipeline=train_pipeline,
        classes=CLASSES),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_tile_validation_fold0_fixed.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=80,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_tma_tile_validation_fold0_fixed.json',
        img_prefix='',
        pipeline=test_pipeline,
        samples_per_gpu=80,
        classes=CLASSES))
evaluation = dict(metric=['bbox', 'segm'], classwise=True, classwise_log=True)