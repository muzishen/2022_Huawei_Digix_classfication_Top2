# -------------------------
# Author: xin jiang
# @Time : 2022/7/16 15:04
# -------------------------
import glob
import os.path as osp
from model.make_model import make_model
from config import cfg
import torchvision.transforms as T
import albumentations as A
import cv2
import torch
import csv
from PIL import Image
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description="Classify Baseline Training")
parser.add_argument("--config_file", default='./configs/convnext_base.yaml', help="path to config file", type=str)
args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cfg.freeze()
print(cfg)

val_transformer1 = A.Compose([
    A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1],interpolation=1),
    A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
    ToTensorV2()
])

val_transformer2 = A.Compose([
    A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1],interpolation=1),
    A.HorizontalFlip(p=1.0),
    A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
    ToTensorV2()

])

val_transformer3 = A.Compose([
    A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1],interpolation=1),
    A.VerticalFlip(p=1.0),
    A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
    ToTensorV2()

])

val_transformer4 = A.Compose([
    A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1],interpolation=1),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
    ToTensorV2()

])

val_transformer5 = A.Compose([
    A.Resize(height=cfg.INPUT.SIZE_TEST[0], width=cfg.INPUT.SIZE_TEST[1],interpolation=1),
    A.VerticalFlip(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, max_pixel_value=255.0, p=1.0),
    ToTensorV2()

])


transformers = [val_transformer1, val_transformer2, val_transformer3,val_transformer4, val_transformer5]
# transformers = [val_transformer1]

def val_collate_fn(batch):
    """
        # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
        """
    images, img_path = zip(*batch)
    
    # label = torch.tensor(label, dtype=torch.int64)

    return torch.stack(images, dim=0), img_path


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, read_flag=False, transform=None):
        self.dataset = dataset
        self.read_flag = read_flag
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]

        if self.read_flag:
            img = np.array(read_image(img_path))
            if self.transform is not None:
                img = self.transform(image=img)['image']
        else:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)

        return img, img_path.split('/')[-1]

if __name__ == '__main__':
    model = make_model(cfg, num_classes=1, deploy_flag=True)
    model = model.cuda()
    model.load_param_test('/opt/data/private/Logs/classfication/digix/jx_base_fix_balance/epoch_40.pth')
    model.eval()
    pred = None
    img_paths = glob.glob(osp.join('/opt/data/private/Data/digix/digix_data/jx_crop/', '*'))
    img_paths.sort()
    val_paths = []
    for i in range(len(transformers)): 
        test_set = ImageDataset(img_paths, read_flag=cfg.INPUT.USE_AUG, transform=transformers[i])
        test_loader = DataLoader(
            test_set, batch_size=32, shuffle=False, num_workers=4, collate_fn=val_collate_fn, pin_memory=True, drop_last=False
        )
        for n_iter, (images, val_path) in enumerate(test_loader):
            print(n_iter)
            with torch.no_grad():
                image_val = images.cuda()
                val_paths.extend(val_path)
                feat_val = model(image_val)
                # print(feat_val)
                if pred is None:
                    pred = feat_val
                else:
                    pred = torch.cat((pred, feat_val))

    preds = pred.cpu().reshape(5, 10000)
    sum_preds = preds.sum(dim=0,keepdim=False)
    print(sum_preds.shape)
    with open(osp.join('/opt/data/private/Logs/classfication/digix/jx_base_fix_balance',"submission.csv"), "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["imagename", "defect_prob"])
        for img_name, pred in zip(val_paths[0:10000], sum_preds):
            template = [img_name, pred.item()]
            writer.writerow(template)
    csvfile.close()

