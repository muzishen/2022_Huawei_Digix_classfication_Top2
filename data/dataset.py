# -------------------------
# Author: xin jiang
# @Time : 2022/6/22 14:18
# -------------------------
import torch.utils.data
from torchvision import datasets, transforms
import random
from torch.utils.data import DataLoader

torch.manual_seed(123456)


def dataset(data_path):
    data_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data_set = datasets.ImageFolder(data_path, data_trans)
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True, num_workers=16)

    return data_loader
