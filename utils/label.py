# -------------------------
# Author: xin jiang
# @Time : 2022/8/2 9:03
# -------------------------
import csv
import glob
import os.path as osp

import torch
import torchvision.transforms as T
from config import cfg
from PIL import Image

from model.make_model import make_model


def label_data(model):
    data_path = '/opt/data/private/huwei/train'
    img_names = glob.glob(osp.join(data_path, '*'))
    img_names.sort()
    val_transforms = T.Compose([
        T.Resize([640, 384]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    with open("relabel.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for img_name in img_names:
            temp = img_names[-18:]
            if 'unlabel' in temp:
                img = Image.open(img_name)
                img = img.convert('RGB')
                post_image = val_transforms(img)
                post_image = post_image.cuda()
                a, b, c = post_image.size()
                post_image = post_image.view(1, a, b, c)
                pred = model(post_image)
                pred = torch.sigmoid(pred)
                if pred.data[0][0].item() > 0.9:
                    label = 1
                    template = [temp, label]
                    writer.writerow(template)
                elif pred.data[0][0].item() < 0.1:
                    label = 0
                    writer.writerow(template)
                print('finish test {}'.format(img_name))
    csvfile.close()


if __name__ == '__main__':
    label_model = make_model(cfg, num_classes=1, deploy_flag=True)
    label_model = label_model.cuda()
    label_model.load_param_test('/opt/data/private/convnext_small_10_0.9781478549842063.pth')
    label_model.eval()
    label_data(label_model)
