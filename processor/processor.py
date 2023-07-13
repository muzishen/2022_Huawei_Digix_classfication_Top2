# -------------------------
# Author: xin jiang
# @Time : 2022/6/26 22:43
# -------------------------
import logging
import os.path
import time

import torch.cuda
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda.amp import autocast as autocast, GradScaler
import numpy as np
# from apex import amp
from torch.utils.data import DataLoader
from config import cfg
import random
from datasets.bases import ImageDataset
import torchvision.transforms as T
from numpy import mean
from sklearn.metrics import roc_auc_score, auc
import albumentations as A
from albumentations import OneOf
from albumentations.pytorch import ToTensorV2
from datasets.preprocessing import RandomErasing
from datasets.autoaugment import ImageNetPolicy


def train_collate_fn(batch):
    """
        # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
        """
    images, label, _ = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(images, dim=0), label


def val_collate_fn(batch):
    images, label, img_name = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(images, dim=0), label, img_name


def _worker_init_fn(worker_id):
    seed = cfg.SOLVER.SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)




def get_k_data_lastnoaug(i, k=5, train_loader=None):

    train_transforms = A.Compose([
        A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
        # A.RandomResizedCrop(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1], scale=(0.8, 1.0)),
        # A.RandomRotate90(always_apply=False, p=0.5),
        A.Flip(p=0.5),
        A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
        A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])
    logger = logging.getLogger("huawei_baseline.train")
    logger.info(train_transforms)
    # print('============================================')
    # print(train_transforms)
    # print('============================================')
    # print(val_transforms)
    # print('============================================')
    fold_size = len(train_loader) // k
    data_train = []
    for j in range(k):
        data_part = train_loader[j * fold_size: (j + 1) * fold_size]
        if j == i:  # 第i折作test
            data_test = data_part
        else:
            data_train.extend(data_part)
    print("using {} fold as test data".format(i))
    num_workers = 8
    train_set = ImageDataset(data_train, read_flag=cfg.INPUT.USE_AUG, transform=train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn, pin_memory=True, worker_init_fn=_worker_init_fn, drop_last=True
    )
    val_set = ImageDataset(data_test, read_flag=cfg.INPUT.USE_AUG, transform=val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader

def get_k_data(i, k=5, train_loader=None):

    train_transforms = A.Compose([
        A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
        # A.RandomResizedCrop(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1], scale=(0.8, 1.0)),
        # A.RandomRotate90(always_apply=False, p=0.5),
        A.Flip(p=0.5),
        # A.ChannelShuffle(p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        # OneOf([
        #    A.MotionBlur(blur_limit=11),
        #    A.MedianBlur(blur_limit=11),
        #    A.GaussianBlur(blur_limit=11)
        # ], p=0.5),
        #
        # OneOf([
        #    A.MotionBlur(p=.2),
        #    A.MedianBlur(blur_limit=3, p=.1),
        #    A.Blur(blur_limit=3, p=.1),
        # ], p=0.5),
        # A.GaussNoise(always_apply=False, p=0.5, var_limit=(10.0, 50.0), per_channel=True, mean=0),
        OneOf([
           A.OpticalDistortion(p=0.3),
           A.GridDistortion(p=0.1),
           A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        # OneOf([
        #    A.CLAHE(clip_limit=2),
        #    A.Emboss(),
        #    A.Sharpen(),
        #    A.RandomBrightnessContrast(),
        # ], p=0.3),
        # ToGray(p=0.05),
        # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        # A.CoarseDropout(max_holes=8, max_height=8, max_width=8,
        #                 min_holes=None, min_height=None, min_width=None, fill_value=0, always_apply=False, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                           value=None, mask_value=None, always_apply=False, p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=60, p=1),
        # OneOf([
        #    A.GridDistortion(distort_limit=0.1, p=0.2),
        #    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2)], p=1),
        # A.ColorJitter(0.3, 0.3, 0.3),
        # ImageNetPolicy(),
        # A.RandomGridShuffle(grid=(2, 2), p=0.5),
        A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
        A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])
    logger = logging.getLogger("huawei_baseline.train")
    logger.info(train_transforms)

# 正常图像：0    瑕疵是1
    pos_sample = train_loader[0:3397]
    neg_sample = train_loader[3397:]
    choice_neg_sample = random.sample(neg_sample, 3397)
    train_loader = pos_sample + choice_neg_sample
    random.shuffle(train_loader)
    
    fold_size = len(train_loader) // k
    data_train = []
    for j in range(k):
        data_part = train_loader[j * fold_size: (j + 1) * fold_size]
        if j == i:  # 第i折作test
            data_test = data_part
        else:
            data_train.extend(data_part)
    print("using {} fold as test data".format(i))
    num_workers = 8
    train_set = ImageDataset(data_train, read_flag=cfg.INPUT.USE_AUG, transform=train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn, pin_memory=True, worker_init_fn=_worker_init_fn, drop_last=True
    )
    val_set = ImageDataset(data_test, read_flag=cfg.INPUT.USE_AUG, transform=val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_classes, output_dir):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    # loss_func = nn.CrossEntropyLoss()

    logger = logging.getLogger("huawei_baseline.train")
    logger.info('start training')
    logger.info('train crop')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUS for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()

    auc_meter = AverageMeter()

    if cfg.SOLVER.MIXED_PRECISION:
        scaler = GradScaler()

    best_model_wts = model.state_dict()
    best_auc = 0.0
    best_epoch = 0

    # random.shuffle(train_loader)
    acc_5 = 0.0
    auc_5 = 0.0
    log_auc = 0.0
    model_list = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        start_time = time.time()
        # if (epoch-1) % 5 == 0:
        #     pos_sample = train_loader[0:3397]
        #     neg_sample = train_loader[3397:]
        #     print(neg_sample[0:10])
        #     choice_neg_sample = random.sample(neg_sample, 3397)
        #     train_loader = pos_sample + choice_neg_sample
        #     random.shuffle(train_loader)
        train_data, test_data = get_k_data((epoch - 1) % 5, train_loader=train_loader)
        loss_meter.reset()
        auc_meter.reset()
        model.train()
        for n_iter, (images, labels) in enumerate(train_data):
            optimizer.zero_grad()
            images = images.to(device)
            # [32, 4]
            labels = labels.to(device)
            loss_label = labels.clone().detach()
            loss_label = loss_label.to(torch.float32)

            if cfg.SOLVER.MIXED_PRECISION:
                with autocast():
                    score = model(images)
                    loss = loss_func(score, loss_label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
            else:
                # [32, 4]
                score, feat_val = model(images)
                loss = loss_func(score, loss_label)
                #
                loss.backward()
                optimizer.step()
            loss_label = loss_label.cpu()
            loss_label = loss_label.detach().numpy()
            score = score.cpu()
            score = score.detach().numpy()
            if loss_label.sum() < 1.0:
                Auc = 0
            else:
                Auc = roc_auc_score(loss_label, score)
                auc_meter.update(Auc, 1)
            loss_meter.update(loss.item(), images.shape[0])
            # auc_meter.update(auc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Auc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_data), loss_meter.avg, auc_meter.avg,
                            scheduler.get_last_lr()[0]))
        #
        logger.info(
            "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Auc: {:.3f}, Base Lr: {:.2e}"
            .format(epoch, (n_iter + 1), len(train_data), loss_meter.avg, auc_meter.avg, scheduler.get_last_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_data.batch_size / time_per_batch))
        print("Testing------")

        model.eval()

        pred = None
        label_list = None
        # for n_iter, (images, labels) in enumerate(test_data):
        for n_iter, (images, labels, val_path) in enumerate(test_data):
            with torch.no_grad():
                labels = torch.from_numpy(np.array(labels).astype(int))
                image_val, labels = images.to(device), labels.to(device)
                feat_val = model(image_val)
                if pred is None:
                    pred = feat_val
                    label_list = labels
                else:
                    pred = torch.cat((pred, feat_val))
                    label_list = torch.cat((label_list, labels))
        # with test
        label_list = label_list.cpu()
        pred = pred.cpu()
        Auc = roc_auc_score(label_list, pred)
        if Auc > best_auc:
            best_auc = Auc
            best_epoch = epoch
            best_model_wts = model.state_dict()
        logger.info('best_auc {}'.format(best_auc))
        logger.info('test_auc {}'.format(Auc))

        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                        os.path.join(output_dir, 'epoch_{}.pth'.format(epoch)))