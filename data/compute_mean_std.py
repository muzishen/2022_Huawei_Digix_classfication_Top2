# -------------------------
# Author: xin jiang
# @Time : 2022/8/18 17:18
# -------------------------
import glob
import os

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torchvision.transforms as T


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class Dataloader:
    def __init__(self, data_root='/opt/data/private/huawei/digix_data/crop'):
        self.data_root = data_root
        self.means = [0.0, 0.0, 0.0]
        self.std = [0.0, 0.0, 0.0]
        self.transformer = T.Compose([
            T.ToTensor()
        ])
        # self.transformer = A.Compose([
        #     A.Resize(height=1280, width=768),
        #     ToTensorV2()
        # ])

    def get_mean_std(self):
        img_paths = glob.glob(os.path.join(self.data_root, '*'))
        img_paths.sort()
        num_images = 0
        for img_path in img_paths:
            # img = np.array(read_image(img_path))
            # print(img.shape)
            # img = cv2.imread(img_path)
            img = Image.open(img_path)
            img = img.convert('RGB')
            # print("computer mean std for : {}".format(img_path))
            num_images += 1
            # post_image = self.transformer(image=img)['image']
            post_image = self.transformer(img)
            # print(post_image.shape)
            for i in range(3):
                self.means[i] += post_image[i, :, :].mean()
                self.std[i] += post_image[i, :, :].std()
            print(self.means)
        self.means = np.asarray(self.means) / num_images
        self.std = np.asarray(self.std) / num_images
        print("means :", self.means)
        print("std :", self.std)


if __name__ == '__main__':
    dataloader = Dataloader()
    dataloader.get_mean_std()
