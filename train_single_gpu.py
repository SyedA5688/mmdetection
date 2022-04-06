from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
import mmcv

import os.path as osp

"""
Distributed training:
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh /home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_instaboostaug.py 2

/home/cougarnet.uh.edu/srizvi7/anaconda3/envs/openmmlab_03212022_2/bin/python3.7 

"""


# cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_HULA_compartment.py")
# cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py")
# cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_mosiac.py")
cfg = Config.fromfile("/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment_instaboostaug.py")

datasets = [build_dataset(cfg.data.train)]

model = build_detector(
    cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg")
)

model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

train_detector(model, datasets, cfg, distributed=False, validate=True)

