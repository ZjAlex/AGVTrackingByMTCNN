from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 60
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6, 14, 20]

# when generating data, use big values for more data, when testing, using small values
config.net1_in_nms_iou_threshold = 0.5
config.net1_out_nms_iou_threshold = 0.5
config.net2_nms_iou_threshold = 0.5
config.net3_nms_iou_threshold = 0.5