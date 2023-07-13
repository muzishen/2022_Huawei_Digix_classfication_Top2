# -------------------------
# Author: xin jiang
# @Time : 2022/6/22 14:30
# -------------------------
import os.path
import torch
from config import cfg
import numpy as np
import random
from torch.backends import cudnn
import datetime
import os
import logging
from utils.logger import set_logger
from model.make_model import make_model
import torch.nn as nn
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from processor.processor import do_train
from loss import make_loss
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ChainedScheduler, LinearLR, CosineAnnealingLR


# from apex import amp


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class FocalLoss(nn.Module):
    def __init__(self, class_num=4, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 4)
        len_ids = len(ids)
        class_mask = targets
        # class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # alpha = self.alpha[ids.data.view(-1)]
        target_len = torch.ones(len_ids, 1, dtype=torch.long)
        alpha = self.alpha[target_len.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train(train_loader=None, val_loader=None, num_classes=1, deploy_flag=False, output_dir=None):
    set_seed(cfg.SOLVER.SEED)
    # output_dir = cfg.OUTPUT_DIR + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    logger = logging.getLogger("huawei_baseline.train")
    logger.info("Saving model in the path: {}".format(cfg.OUTPUT_DIR))
    # logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        model = make_model(cfg, num_classes=num_classes, deploy_flag=deploy_flag)
        # model.load_param_finetune(cfg.MODEL.PRETRAIN_PATH)
        print('Loading pretrained model for finetune......')
    else:
        model = make_model(cfg, num_classes=num_classes, deploy_flag=deploy_flag)
    # loss_func = make_loss(cfg, num_class=num_classes)
    # loss_func = FocalLoss().cuda()
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]))
    # logger = logging.getLogger("huawei_baseline.train")
    logger.info('pos_weight=6.0')
    # loss_func = nn.BCEWithLogitsLoss()
    loss_func.to('cuda')
    model.to('cuda')
    # optimizer = make_optimizer(cfg, model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.MAX_LR, )
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.MAX_LR, )
    scheduler1 = LinearLR(optimizer, start_factor=0.5, total_iters=3)
    scheduler2 = ExponentialLR(optimizer, gamma=0.8)
    scheduler = ChainedScheduler([scheduler1, scheduler2])

    # scheduler = WarmupMultiStepLR(optimizer= optimizer, milestones= [10,20], gamma = 0.1, warmup_factor= 0.3333, warmup_iters = 0, warmup_method='linear')
    # for i in range(40):
    #    print(scheduler1.get_last_lr())
    #    scheduler1.step()
    step = int(len(train_loader) * 0.8) // cfg.SOLVER.IMS_PER_BATCH
    # print(step)
    # scheduler = MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.5)
    # scheduler1 = LinearLR(optimizer, start_factor=0.5, total_iters=5)
    # scheduler2 = CosineAnnealingLR(optimizer, T_max=30)
    # scheduler = ChainedScheduler([scheduler1, scheduler2])
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, total_steps=cfg.SOLVER.MAX_EPOCHS * step, )
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # if cfg.SOLVER.LR_NAME == 'WarmupMultiStepLR':
    #    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
    #                                   cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    #    logger.info("use WarmupMultiStepLR, delay_step: {}".format(cfg.SOLVER.STEPS))
    # elif cfg.SOLVER.LR_NAME == 'WarmupCosineAnnealingLR':
    #    scheduler = WarmupCosineAnnealingLR(optimizer, cfg.SOLVER.MAX_EPOCHS, cfg.SOLVER.DELAY_ITERS,
    #                                         cfg.SOLVER.ETA_MIN_LR,
    #                                         cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
    #                                        cfg.SOLVER.WARMUP_METHOD)
    #    logger.info("use WarmupCosineAnnealingLR, delay_step:{}".format(cfg.SOLVER.DELAY_ITERS))
    # scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    # shceduler1 = LinearLR(optimizer, start_factor=0.5, total_iters=4)
    # shceduler2 = ExponentialLR(optimizer, gamma=0.9, last_epoch=30)
    # scheduler = ChainedScheduler([shceduler1, shceduler2])
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_classes, output_dir)


if __name__ == '__main__':
    train()
