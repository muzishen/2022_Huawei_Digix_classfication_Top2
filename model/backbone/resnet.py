# -------------------------
# Author: xin jiang
# @Time : 2022/6/22 13:09
# -------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, down_sample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, dilate=None, num_class=1000, frozen_stage=-1):
        if dilate is None:
            dilate = [False, False, False]
        self.in_channels = 64
        super(ResNet, self).__init__()
        self.dilation = 1
        self.frozen_stage = frozen_stage
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=dilate[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=dilate[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=dilate[2])

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        down_sample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation = self.dilation * stride
            stride = 1
        if stride != 1 or self.in_channels != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_channels, planes, stride, down_sample, dilation=previous_dilation)]
        self.in_channels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes, dilation=previous_dilation))

        return nn.Sequential(*layers)

    def _frozen_stages(self):
        if self.frozen_stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stage + 1):
            m = getattr(self, 'layer{}'.format(i))
            print("layer{}".format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print x.size()
        x = self.maxPool(x)
        # print x.size()

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        # p3 = torch.cat([p2, p3], 1)

        p4 = self.layer4(p3)

        # result = self.avgPool(p4)
        # result = torch.flatten(result, 1)
        # result = self.fc(result)

        return p4

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet50(pretrained=False, frozen_stage=-1, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], frozen_stage=frozen_stage, **kwargs)
    if pretrained:
        model = None
    return model


if __name__ == '__main__':
    net = resnet50()
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 224, 224).cuda()
    var = Variable(var)

    print(net(var))
    # print('*************')
    # var = torch.FloatTensor(1, 3, 255, 255).cuda()
    # var = Variable(var)
    #
    # net(var)
