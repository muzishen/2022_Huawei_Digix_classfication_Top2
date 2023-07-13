# -------------------------
# Author: xin jiang
# @Time : 2022/6/26 11:02
# -------------------------
import torch.nn as nn
import torch.nn.functional

from .backbone.build_backbone import build_backbone
from abc import ABC


class GeneralizedMeanPooling(nn.Module, ABC):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of
        several input planes.
        The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
            - At p = infinity, one gets Max Pooling
            - At p = 1, one gets Average Pooling
        The output is of size H x W, for any input size.
        The number of output features is equal to the number of input planes.
        Args:
            output_size: the target output size of the image of the form H x W.
                         Can be a tuple (H, W) or a single H for a square image H x H
                         H and W can be either a ``int``, or ``None`` which means the size
                         will be the same as that of the input.
        """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)

    def __repr__(self):
        return (
                self.__class__.__name__ + "(" + str(self.p) + ", output_size=" + str(self.output_size) + ")"
        )


class GeneralizedMeanPooling2P(GeneralizedMeanPooling, ABC):
    """ Same, but norm is trainable
        """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6):
        super(GeneralizedMeanPooling2P, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, num_bottleneck=256):
        super().__init__()
        add_block1 = [
            nn.BatchNorm1d(input_dim), nn.ReLU(inplace=True),
            nn.Linear(input_dim, num_bottleneck, bias=False),
            nn.BatchNorm1d(num_bottleneck)
        ]
        add_block = nn.Sequential(*add_block1)
        add_block.apply(weights_init_kaiming)

        classifier = nn.Linear(num_bottleneck, class_num, bias=False)
        classifier.apply(weight_init_classifier)
        self.dropout = nn.Dropout(p=0.0)
        self.add_block = add_block
        self.classifier = classifier
        #nn.Sequential(
        #    nn.Linear(768, 1, bias=True),
        #   nn.BatchNorm1d(1, ),
        #)
        #self.classifier.apply(weight_init_classifier)

    def forward(self, x):
        x = self.add_block(x)
        y = self.classifier(x)
        return y


class Backbone(nn.Module):
    def __init__(self, num_class, cfg, deploy_flag):
        super().__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE

        self.model_name = cfg.MODEL.NAME
        self.use_checkpoint = cfg.MODEL.USE_CHECKPOINT
        self.in_planes, self.base = build_backbone(self.model_name, last_stride, cfg, deploy_flag)
        self.num_class = num_class
        self.classifier = ClassBlock(self.in_planes, self.num_class, num_bottleneck=16)

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM Pooling')
            self.gap = GeneralizedMeanPooling2P()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        if not deploy_flag:
            if pretrain_choice == 'imagenet':
                self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
                print('Loading pretrained ImageNet model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))
            elif pretrain_choice == 'NO':
                self.base.random_init()
                print("No using pretrained model")

    def forward(self, x):
        x = self.base(x)
        if 'swin' in self.model_name:
            # global_feat = x
            global_feat = self.gap(x)
        else:
            global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # batch_size 2048
        y = self.classifier(global_feat)

        if self.training:
            return y
        else:
            return y

    def load_param_test(self, trained_path):
        param_dict = {k: v for k, v in torch.load(trained_path).items()}
        

        #model_dict = {k: v for k, v in self.state_dict().items()}
        #print(len(param_dict))
        #print(len(self.state_dict()))
        #print(model_dict[0])
        # print(self.state_dict())
        for i in param_dict:
            t = i[7:]
            self.state_dict()[t].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            print(i)
            self.state_dict()[t].copy_(param_dict[i])
        print('Loading pretrained model for finetune from {}'.format(model_path))

    def fc_load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'classifier' in i:
                self.state_dict()[i].copy_(param_dict[i])


def make_model(cfg, num_classes, deploy_flag=False):
    model = Backbone(num_classes, cfg, deploy_flag)
    return model
