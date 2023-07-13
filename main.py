# -------------------------
# Author: xin jiang
# @Time : 2022/6/26 10:01
# -------------------------
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from config import cfg
from datasets.make_dataloader import make_dataloader
import argparse
from tool.train import train
from torchvision.models import resnet50
import datetime
from utils.logger import set_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify Baseline Training")
    parser.add_argument("--config_file", help="path to config file", type=str)
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR + '/'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = set_logger("huawei_baseline", output_dir, if_train=True)
    logger.info(cfg)
    logger.info("Saving model in the path: {}".format(cfg.OUTPUT_DIR))
    train_loader, val_loader, num_trains, num_tests = make_dataloader(cfg)
    if cfg.MODEL.MODE == 'train':
        train(train_loader, val_loader, num_classes=1, deploy_flag=False, output_dir=output_dir)
