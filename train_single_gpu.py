from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
import mmcv

import os.path as osp

cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_HULA_compartment.py")

datasets = [build_dataset(cfg.data.train)]

model = build_detector(
    cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg")
)

model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

train_detector(model, datasets, cfg, distributed=False, validate=True)

