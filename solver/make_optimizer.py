# -------------------------
# Author: xin jiang
# @Time : 2022/6/26 15:16
# -------------------------
import logging
import torch


def make_optimizer(cfg, model):
    logger = logging.getLogger("huawei_baseline.train")
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.FC_LR_TIMES > 1:
            if 'classifier' in key or 'arcface' in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.FC_LR_TIMES
                logger.info('Using {} times learning rate for fc'.format(cfg.SOLVER.FC_LR_TIMES))
        if 'gap' in key:
            lr = cfg.SOLVER.BASE_LR * 10
            weight_decay = 0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        #print("need grad params ", params)
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    return optimizer


