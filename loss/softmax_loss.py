# -------------------------
# Author: xin jiang
# @Time : 2022/6/26 14:52
# -------------------------
import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs).cuda()
        targets = targets.cuda()
        base_loss = self.base_loss(log_probs, targets).cuda()
        base_mul = (1 - self.epsilon) + self.epsilon / self.num_classes
        loss = base_mul * base_loss
        #log_probs = self.log_softmax(inputs)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        #if self.use_gpu:
        #    targets = targets.cuda()
        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        #loss = (- targets * log_probs).mean(0).sum()
        return loss
