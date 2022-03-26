_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_HULA_compartment.py',
    '../_base_/datasets/HULA_compartment_instance.py',
    '../_base_/schedules/schedule_1x_HULA_maskrcnn.py',
    '../_base_/default_runtime.py'
]
work_dir = "/data/syed/mmdet/run7_maskrcnn_full_trainset/"
gpu_ids = range(5, 6)
seed = 0
