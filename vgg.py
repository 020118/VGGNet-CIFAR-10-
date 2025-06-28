from typing import Union, cast
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class vggnet(nn.Module):
    def __init__(self, features, dropout: float = 0.5):
        super(vggnet, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,  4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 10)
        )

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                n = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2./n))
                i.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x


def make_layer(cfg: list[Union[str, int]],) -> nn.Sequential:
    layers = []
    in_channel = 3
    for i in cfg:
        if i == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            i = cast(int, i)
            layers += [nn.Conv2d(in_channel, i, kernel_size=3, padding=1), nn.ReLU(True)]
            in_channel = i
    return nn.Sequential(*layers)


cfgs: dict[str, Union[str, int]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg11():
    return vggnet(make_layer(cfgs["A"]))

def vgg13():
    return vggnet(make_layer(cfgs["B"]))

def vgg16():
    return vggnet(make_layer(cfgs["D"]))

def vgg19():
    return vggnet(make_layer(cfgs["E"]))


