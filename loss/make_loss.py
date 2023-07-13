# -------------------------
# Author: xin jiang
# @Time : 2022/6/26 14:49
# -------------------------
import logging
import torch.nn as nn
from .softmax_loss import CrossEntropyLabelSmooth
import torch.nn.functional as F
import torch


def make_loss(cfg, num_class):
    logger = logging.getLogger('huawei_baseline.train')
    if 'softmax' in cfg.MODEL.METRIC_LOSS_TYPE and cfg.MODEL.IF_LABELSMOOTH == 'on':
        cls_loss_func = CrossEntropyLabelSmooth(num_classes=num_class)
        logger.info('label smooth on, num_classes: {}'.format(num_class))
    else:
        weight = torch.Tensor(cfg.MODEL.ID_LOSS_WEIGHT)
        cls_loss_func = nn.CrossEntropyLoss(weight=weight)

    def loss_func(score, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'softmax':
            cross_entropy = cls_loss_func(score, target)
            return cross_entropy
        else:
            cross_entropy = cls_loss_func(score, target)
            return cross_entropy

    return loss_func
