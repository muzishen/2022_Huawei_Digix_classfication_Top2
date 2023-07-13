# -------------------------
# Author: xin jiang
# @Time : 2022/6/22 14:39
# -------------------------
from torch import optim

from backbone.resnet import resnet50
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        self.backbone = backbone
        self.loss_func = nn.CrossEntropyLoss()
        self.param = list(self.backbone.fc.parameters())
        self.optimizer = optim.SGD(self.param, lr=0.001, momentum=0.9)

    def forward(self, x):
        out = self.backbone(x)
        return out

