import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Tumour(BaseImageDataset):

    dataset_dir = ''

    def __init__(self, root='', pid2label = '', verbose=True, **kwargs):
        super(Tumour, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'new_train')
        self.test_dir = osp.join(self.dataset_dir, 'new_test')
        self.pid2label = eval(pid2label)
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        test = self._process_dir(self.test_dir, relabel=True)

        if verbose:
            print("=> Tumour loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs = self.get_imagedata_info(self.test)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*'))
        img_paths.sort()
        pattern = re.compile(r'([-\d]+)_')
        
        dataset = []
        for img_path in img_paths:
            if relabel:
                pid = int(pattern.search(img_path).groups()[0])
                real_label = self.pid2label[pid]
                #print(img_path, real_label)
                dataset.append((img_path, real_label))
                
            else:
                dataset.append((img_path, 0))
        return dataset

