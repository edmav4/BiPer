import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from modules import *

BN = None

__all__ = ['mobilenet_1w1a','mobilenet']


def conv3x3Binary(in_planes, out_planes, stride, groups):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, groups=groups, bias=False)

def conv3x3Binary_g(in_planes, out_planes, stride):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, groups=int(in_planes/16), bias=False)
def conv1x1Binary(in_planes, out_planes):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=1,
                          padding=0, bias=False)


class DWBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, downsample=None):
        super(DWBlock, self).__init__()
        self.conv1 = conv3x3Binary_g(inp, inp, stride)
        self.bn1 = nn.BatchNorm2d(inp)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = conv1x1Binary(inp, oup)
        self.bn2 = nn.BatchNorm2d(oup)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.conv3 = conv1x1Binary(oup, oup)
        self.bn3 = nn.BatchNorm2d(oup)
        self.nonlinear3 = nn.Hardtanh(inplace=True)
        self.stride = stride
        self.downsample = nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.stride!=1:
            residual = self.downsample(x)
        out += residual
        
        out = self.nonlinear1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.nonlinear2(out)
        residual = out

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.nonlinear3(out)

        return out
class MobileNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardtanh(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                conv3x3Binary_g(inp,inp,stride),
                nn.BatchNorm2d(inp),
                nn.Hardtanh(inplace=True),

                # pw
                conv1x1Binary(inp,oup),
                nn.BatchNorm2d(oup),
                nn.Hardtanh(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            DWBlock(32, 64, 1),
            DWBlock(64, 128, 2),
            DWBlock(128, 128, 1),
            DWBlock(128, 256, 2),
            DWBlock(256, 256, 1),
            DWBlock(256, 512, 2),
            DWBlock(512, 512, 1),
            DWBlock(512, 512, 1),
            DWBlock(512, 512, 1),
            DWBlock(512, 512, 1),
            DWBlock(512, 512, 1),
            DWBlock(512, 1024, 2),
            DWBlock(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class MobileNetV1_32(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1_32, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
def mobilenet_1w1a(**kwargs):
    """Constructs a ResNet-18 model. """
    model = MobileNetV1(**kwargs)
    return model

def mobilenet(**kwargs):
    """Constructs a ResNet-18 model. """
    model = MobileNetV1_32(**kwargs)
    return model






def resnet18_1w1a(**kwargs):
    """Constructs a ResNet-18 model. """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(
        filter(lambda name: 'conv' in name or 'fc' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
