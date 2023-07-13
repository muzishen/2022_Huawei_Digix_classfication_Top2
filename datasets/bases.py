# -------------------------
# Author: xin jiang
# @Time : 2022/6/24 15:12
# -------------------------

from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        # imgs = []
        # for img in data:
        #     imgs += img
        # # 创建不重复元素集
        # imgs = set(imgs)
        num_img = len(data)
        return num_img

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, test):
        num_train_img = self.get_imagedata_info(train)
        num_test_img = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # images")
        print("  ----------------------------------------")
        print("  train    | {:5d}".format(num_train_img))
        print("  ----------------------------------------")
        print("  test    | {:5d}".format(num_test_img))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, read_flag=False, transform=None):
        self.dataset = dataset
        self.read_flag = read_flag
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]

        if self.read_flag:
            img = np.array(read_image(img_path))
            if self.transform is not None:
                img = self.transform(image=img)['image']
        else:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)

        return img, label, img_path.split('/')[-1]