# -------------------------
# Author: xin jiang
# @Time : 2022/7/16 10:03
# -------------------------

import os

import torch

from datasets.bases import BaseImageDataset
import os.path as osp
import glob
import csv
import random


class DIGIX(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, root='/opt/data/private/huawei/digix', verbose=True, **kwargs):
        super(DIGIX, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_crop/')
        self.labels = osp.join(root, 'train_label/train_label.csv')

        self.label = []
        self._check_before_run()

        train = self._process_dir(self.train_dir)

        if verbose:
            print("=> Image loaded")
            self.print_dataset_statistics(train, train)
        # 二元list
        self.train = train

        self.num_train_img = self.get_imagedata_info(self.train)

    def _process_dir(self, dir_path):
        dataset = []
        img_paths = glob.glob(osp.join(dir_path, '*'))
        img_paths.sort()
        labels = csv.reader(open(self.labels))
        
        total_0 = 0
        total_1 = 0
        # add unlabel data
        #for (img, label) in labels:
        #    img_path = os.path.join(dir_path, img)
        #    sub_label = [0.0]
        #    if float(label) > 0.0:
        #        sub_label = [1.0]
        #        total_1 += 1
        #    total += 1
        #    dataset.append((img_path, sub_label))
        #random.shuffle(dataset)

        # official data
        #采样1
        for img_path, label in zip(img_paths, labels):
            sub_label = [0.0]
            if int(label[1]) > 0:
                sub_label = [1.0]
                total_1 += 1
            else:
                continue

            dataset.append((img_path, sub_label))
        print("1: ",total_1,)

        #采样0
        labels_copy = csv.reader(open(self.labels))
        for img_path, label in zip(img_paths, labels_copy):
            sub_label = [0.0]
            if int(label[1]) == 0:
                sub_label = [0.0]
                total_0 += 1
            else:
                continue

            dataset.append((img_path, sub_label))
        print("0: ",total_0)
        return dataset

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))


if __name__ == '__main__':
    # temp = JinYu()
    # dirs = os.listdir('D:/huawei/data')
    # dataset = []
    # sub_index = 0
    # for img_dir in dirs:
    #     print(img_dir)
    loss = torch.nn.BCELoss()
    pred = torch.Tensor([0.5, 0.5, 0.4])
    la = torch.Tensor([1, 0, 1])
    print(loss(pred, la))
