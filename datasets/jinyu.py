# -------------------------
# Author: xin jiang
# @Time : 2022/6/24 15:44
# -------------------------
import os

import torch

from datasets.bases import BaseImageDataset
import os.path as osp
import glob


class JinYu(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, root='D:/huawei/data/', verbose=True, **kwargs):
        super(JinYu, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train/')
        self.test_dir = osp.join(self.dataset_dir, 'val/')

        self.label = []
        self._check_before_run()

        train = self._process_dir(self.train_dir)
        test = self._process_dir(self.test_dir)

        if verbose:
            print("=> Huawei loaded")
            self.print_dataset_statistics(train, test)
        # 二元list
        self.train = train
        self.test = test

        self.num_train_img = self.get_imagedata_info(self.train)
        self.num_test_img = self.get_imagedata_info(self.test)

    def _process_dir(self, dir_path):
        dirs = os.listdir(dir_path)
        dataset = []
        sub_index = 0
        index_again = [1, 0, 3, 2]
        for img_dir in dirs:
            signal_dir_path = dir_path + img_dir
            img_paths = glob.glob(osp.join(signal_dir_path, '*'))
            img_paths.sort()
            sub_label = [0.0, 0.0, 0.0, 0.0]
            sub_label[index_again[sub_index]] = 1.0
            # print(img_dir, sub_label)
            for img_path in img_paths:
                dataset.append((img_path, sub_label))
            sub_index += 1
        return dataset

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


if __name__ == '__main__':
    temp = JinYu()
