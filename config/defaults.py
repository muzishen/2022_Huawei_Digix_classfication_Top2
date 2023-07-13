# -------------------------
# Author: xin jiang
# @Time : 2022/6/25 15:54
# -------------------------
from yacs.config import CfgNode as CN
import os

_C = CN()

_C.MODEL = CN()
_C.MODEL.USE_CHECKPOINT = True
# Model's Mode
_C.MODEL.MODE = 'train'
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0,1,2,3'
# Name of backbone
_C.MODEL.NAME = 'convnext_small'

# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = '/opt/data/private/code/yujian/convnext_small_384.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune', 'original'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.METRIC_LOSS_TYPE = 'softmax'
# # If train with soft triplet loss, options: 'True', 'False'
# _C.MODEL.NO_MARGIN = True
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# Frozen layers of backbone
_C.MODEL.FROZEN = 4
# Frozen layers of backbone
_C.MODEL.POOLING_METHOD = 'GeM'
_C.MODEL.ID_LOSS_WEIGHT = [1.0, 0.5, 0.8, 4.0]
_C.MODEL.TRIPLET_LOSS_WEIGHT = 0.0
_C.MODEL.CHANNEL_EXPANSION = [1, 1, 1, 1]
_C.MODEL.BLOCK_TYPE = 'base'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [640, 384]
# Size of the image during test
_C.INPUT.SIZE_TEST = [640, 384]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5

# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.76404387, 0.7610951, 0.722468]
# [0.764, 0.761, 0.722]
# [0.764, 0.761, 0.722] crop #
# [0.714, 0.711, 0.678] total  #
# [0.7635201, 0.7976389, 0.7427222]  crop total #  [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.27856505, 0.29630744, 0.2451125]
# [0.273, 0.293, 0.242]
#  [0.273, 0.293, 0.242] crop # [0.322, 0.336, 0.294] total
#  [0.27595693, 0.27708337, 0.23753342] crop total #  [0.229, 0.224, 0.225]

# Value of padding size
_C.INPUT.PADDING = 10
_C.INPUT.USE_AUG = True
_C.INPUT.USE_TIMM = False
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('DIGIX')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/opt/data/private/Data/digix/digix_data')
# 注意key value 中间不能有空格
_C.DATASETS.PID2LABEL = '{0:0, 1:1}'
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 32
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 20

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = 'Adam'
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 40
# Base learning rate
_C.SOLVER.BASE_LR = 1e-4

# the time learning rate of fc layer
_C.SOLVER.FC_LR_TIMES = 2
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss sampler method, option: batch_hard, batch_soft
_C.SOLVER.HARD_EXAMPLE_MINING_METHOD = 'batch_hard'
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# lr_scheduler
# lr_scheduler method, option WarmupMultiStepLR, WarmupCosineAnnealingLR
_C.SOLVER.LR_NAME = 'WarmupCosineAnnealingLR'
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = [40, 70]

# Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 30
_C.SOLVER.ETA_MIN_LR = 1e-7
_C.SOLVER.MAX_LR = 1e-7
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.1
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = _C.SOLVER.MAX_EPOCHS
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 50
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = _C.SOLVER.MAX_EPOCHS
_C.SOLVER.MIXED_PRECISION = True
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 32
_C.SOLVER.SEED = 42

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 32
# Whether using fliped feature for testing, option: 'on', 'off'
_C.TEST.FLIP_FEATS = 'on'
# Path to trained model
# _C.TEST.WEIGHT = _C.SOLVER.MAX_EPOCHS
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.RE_RANKING = False
# K1, K2, LAMBDA
_C.TEST.RE_RANKING_PARAMETER = [60, 10, 0.3]
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./log"
if not os.path.isdir(_C.OUTPUT_DIR):
    os.makedirs(_C.OUTPUT_DIR)
