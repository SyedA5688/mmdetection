
"""
This swin configuration file is intended for training swin transformers from scratch. This may be a regular
 experiment training a swin transformer on real labels, or it may be a fine tuning experiment where we load
 a checkpoint trained on GAN pseudo labels, and then fine tune on real data.

Use this config if you want to train on real labels from scratch.

Other customized configs needed:
- HULA_compartment_instance2048x_autoaug.py: Autoaugment TMA compartment dataset, real training labels.
- schedule_1x_HULA_swin.py: AdamW optimizer + scheduling for swin transformer
"""


_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py',
    '../_base_/datasets/HULA_compartment_instance2048x_autoaug.py',
    '../_base_/schedules/schedule_1x_HULA_swin.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

work_dir = "/data/syed/mmdet/run19_swin_autoaug_finetuning/"  # ToDo: Change experiment name
gpu_ids = range(4, 8)
seed = 0

# Adding in load_from checkpoint model
# ToDo: If finetuning after pseudo-annotation training, change this path to checkpoint
load_from = '/data/syed/mmdet/run17_swin_gan_img_train_autoaug/epoch_15.pth'
