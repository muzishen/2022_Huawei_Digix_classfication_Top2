# -------------------------
# Author: xin jiang
# @Time : 2022/6/24 15:09
# -------------------------
import random
import logging
import torch
import torchvision.transforms as T
import albumentations as A
from config import cfg
import numpy as np
from albumentations import OneOf
from albumentations.pytorch import ToTensorV2
from timm.data import create_transform
from .autoaugment import ImageNetPolicy
from .jinyu import JinYu
from .bases import ImageDataset
from torch.utils.data import DataLoader
from .sampler import RandomIdentitySampler
from .preprocessing import RandomErasing
from .digix import DIGIX

__factory = {
    'DIGIX': DIGIX
}


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


def make_dataloader(cfg):
    if cfg.INPUT.USE_AUG:
        train_transforms = A.Compose([
            A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
            A.Flip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            # OneOf([
            #    A.MotionBlur(blur_limit=11),
            #    A.MedianBlur(blur_limit=11),
            #    A.GaussianBlur(blur_limit=11)
            # ], p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=60, p=1),
            # OneOf([
            #    A.GridDistortion(distort_limit=0.1, p=0.2),
            #    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2)], p=1),
            # A.RandomGridShuffle(grid=(2, 2), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                               value=None, mask_value=None, always_apply=False, p=0.5),
            A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
            A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ])
    elif cfg.INPUT.USE_TIMM:
        resize_im = cfg.INPUT.SIZE_TRAIN[0] > 32
        # train_transforms = create_transform(
        #     input_size=cfg.INPUT.SIZE_TRAIN,
        #     is_training=True,
        #     color_jitter=0.4,
        #     auto_augment='rand-m9-mstd0.5-inc1',
        #     interpolation='bicubic',
        #     re_prob=0.25,
        #     re_mode='pixel',
        #     re_count=1,
        #     mean=cfg.INPUT.PIXEL_MEAN,
        #     std=cfg.INPUT.PIXEL_STD,
        # )
        train_transforms = T.Compose([
            # T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN, scale=(0.8, 1.0), interpolation=3),
            # T.RandomRotation(degrees=10),
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomRotation(degrees=10),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.RandomVerticalFlip(),
            T.RandomVerticalFlip(p=cfg.INPUT.PROB),
            # ImageNetPolicy(),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        # train_transforms = A.Compose([
        # A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
        # A.Flip(p=0.5),
        #   A.RandomResizedCrop(height=cfg.INPUT.SIZE_TRAIN[0], width=cfg.INPUT.SIZE_TRAIN[1], scale=(0.8, 1.0)),
        #   A.RandomRotate90(always_apply=False, p=0.5),
        #   A.Flip(p=0.5),
        #   A.HorizontalFlip(p=0.5),
        #     A.ChannelShuffle(p=0.5),
        #   A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
        #   ToTensorV2()
        # ])
        # train_transforms = T.Compose([
        #      T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN, scale=(0.8, 1.0)),
        #       T.RandomRotation(degrees=10),
        #        T.Resize(cfg.INPUT.SIZE_TRAIN),
        #         T.RandomHorizontalFlip(),
        #          T.RandomVerticalFlip(),
        # A.ChannelShuffle(p=cfg.INPUT.PROB),
        # T.RandomVerticalFlip(),
        #           T.ToTensor(),
        #           T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        # 随机擦除
        #            RandomErasing(probability=0.5, mean=cfg.INPUT.PIXEL_MEAN)
        #       ])

        if not resize_im:
            train_transforms.transforms[0] = T.RandomCrop(127, padding=4)

        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # T.Resize(256),
            # T.CenterCrop(224),
            # T.ToTensor(),
            # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    #    val_transforms = T.Compose([
    # T.Resize([640, 384]),
    # T.ToTensor(),
    # T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    #         A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1]),
    #  A.ChannelShuffle(p=0.5),
    #          A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
    #           ToTensorV2()
    #        ])

    else:
        train_transforms = T.Compose([
            T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN, scale=(0.8, 1.0), interpolation=3),
            T.RandomRotation(degrees=10),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            ImageNetPolicy(),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

    # print('============================================')
    # print(train_transforms)
    # print('============================================')
    # print(val_transforms)
    logger = logging.getLogger("huawei_baseline.train")
    logger.info(train_transforms)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_trains = dataset.num_train_img
    num_tests = 0

    train_set = dataset.train
    # train_set = ImageDataset(dataset.train, read_flag=cfg.INPUT.USE_AUG, transform=train_transforms)
    # len_train = int(0.8 * len(train_set))
    # train_sets = train_set[:len_train]
    # val_set = train_set[len_train:]
    # print('length train/test : {}/{}'.format(len_train, len(train_set) - len_train))
    # train_set, val_set = torch.utils.data.random_split(train_set, [len_train, len(train_set) - len_train])
    # print("len train : {}, len val : {}".format(len_train, len(val_set)))
    # print("Using softmax sampler")
    # collate_fn定义每次dataloader迭代返回的格式
    train_loader = train_set
    # train_set = ImageDataset(train_set, read_flag=cfg.INPUT.USE_AUG, transform=train_transforms)
    # train_loader = DataLoader(
    #     train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
    #     collate_fn=train_collate_fn, pin_memory=True, worker_init_fn=_worker_init_fn, drop_last=True
    # )

    # val_set = ImageDataset(val_set, read_flag=cfg.INPUT.USE_AUG, transform=val_transforms)
    # val_loader = DataLoader(
    #    val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
    #    collate_fn=val_collate_fn
    # )
    val_loader = None
    return train_loader, val_loader, num_trains, num_tests
