import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VGG(nn.Module):
    def __init__(self,
                 struct_cfgs: list,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 batch_norm: bool = False,
                 init_weights: bool = True):

        super(VGG, self).__init__()
        self.in_channels = in_channels    # single channel imgs

        self.features = self.__makeConvLayers(self.in_channels, struct_cfgs, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self.__initialize_weights()

    def __makeConvLayers(self, in_channels: int, struct_cfgs: list, batch_norm: bool):
        modules: List[nn.Module] = []
        for struct in struct_cfgs:
            #maxpool layer
            if struct == 'M':
                modules = modules + [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, struct, kernel_size=3, padding=1)
                # with batchnorm
                if batch_norm:
                    modules = modules + [conv2d, nn.BatchNorm2d(struct), nn.ReLU()]
                else:
                    modules = modules + [conv2d, nn.ReLU()]
                in_channels = struct
        return nn.Sequential(*modules)

    def forward(self, input):
        # features
        out = self.features(input)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        # classifier
        out = self.classifier(out)
        return out

    def __initialize_weights(self):
        for m in self.modules():
            # 卷积层权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                # 卷积层偏置初始化，默认初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # batchNorm层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # 全连接层
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
