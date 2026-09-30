"""CIFAR-style backbones: ResNet-18, VGG-16-BN and MobileNetV2.

All three are the usual 32x32 variants (3x3 stem, no initial max-pool), so
FLOPs/params numbers are comparable with the CIFAR pruning literature.
The same networks are used unchanged on 64x64 Tiny ImageNet (global average
pooling before the classifier), which costs ~4x the FLOPs of a 32x32 input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# --------------------------------------------------------------------------- VGG
VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes=10):
        super().__init__()
        layers, c = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)
        self.linear = nn.Linear(c, num_classes)

    def forward(self, x):
        # 1x1 after the last pool at 32x32; the adaptive pool only matters for larger inputs (Tiny ImageNet)
        return self.linear(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


# --------------------------------------------------------------------------- MobileNetV2
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = None
        if stride == 1:
            self.shortcut = nn.Sequential()
            if in_planes != out_planes:
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                              nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.shortcut is not None else out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride); stride 1 in stage 2 for CIFAR
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers, c = [], 32
        for e, o, n, s in self.CFG:
            for st in [s] + [1] * (n - 1):
                layers.append(InvertedResidual(c, o, e, st))
                c = o
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu6(self.bn2(self.conv2(out)))
        return self.linear(F.adaptive_avg_pool2d(out, 1).flatten(1))


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == 'resnet18':
        return ResNet([2, 2, 2, 2], num_classes)
    if name == 'vgg16':
        return VGG(VGG16_CFG, num_classes)
    if name == 'mobilenetv2':
        return MobileNetV2(num_classes)
    raise ValueError(f'unknown model {name}')
