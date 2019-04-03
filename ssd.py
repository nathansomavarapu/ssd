import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16, resnet50

import numpy as np


class Features(nn.Module):

    def __init__(self, model_type):
        super(Features, self).__init__()

        self.model_type = model_type

        if model_type == 'vgg':
            new_layers = list(vgg16(pretrained=True).features)
            new_layers[16] = nn.MaxPool2d(2, ceil_mode=True)
            new_layers[-1] = nn.MaxPool2d(3, 1, padding=1)

            self.f0 = nn.Sequential(*new_layers[:23])

            self.bn0 = nn.BatchNorm2d(512)

            self.f1 = nn.Sequential(*new_layers[23:])
        elif model_type == 'resnet':
            resenet = resnet50(pretrained=True, )
            list(list(resenet.layer1.children())[0].downsample.children())[0] = nn.Conv2d(64, 256, 1, padding=1, bias=False)

            self.f0 = nn.Sequential(*list(resenet.children())[:6])

            self.f1 = resenet.layer3

            self._init_weights()

        elif model_type == 'mobilenet':
            pass
        else:
            raise NotImplementedError()
    
    def forward(self, x):

        
        x0 = self.f0(x)
        if self.model_type == 'vgg':
            x1 = self.bn0(x0)
        else:
            x1 = x0
        x1 = self.f1(x1)

        return x0, x1
    
    def _init_weights(self, init_base=False):

        if self.model_type == 'vgg':
            self.apply(_weight_init)
        elif self.model_type == 'resnet':
            self.apply(_weight_init)


def _weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

def rm_bn(layers):
    rem_list = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.BatchNorm2d):
            rem_list.append(i)
    
    for i, val in enumerate(rem_list):
        del layers[(val - i)]
    
    return layers


class ssd(nn.Module):
    def __init__(self, num_cl, init_base=False, init_extra=True, bn=False, base='vgg'):
        super(ssd, self).__init__()

        self.num_cl = num_cl
        self.layers = []
        self.size = (300, 300)
        
        # NOTE: dilation of 6 requires a padding of 6
        feature_output_channels = 512 if base == 'vgg' else 1024
        layers = [
            nn.Conv2d(feature_output_channels, 1024, 3, dilation=3, padding=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        ]
        if not bn:
            layers = rm_bn(layers)
        f2 = nn.Sequential(*layers)

        layers = [
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ]
        if not bn:
            layers = rm_bn(layers)
        f3 = nn.Sequential(*layers)

        layers = [
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        if not bn:
            layers = rm_bn(layers)
        f4 = nn.Sequential(*layers)

        layers = [
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        if not bn:
            layers = rm_bn(layers)
        f5 = nn.Sequential(*layers)

        layers = [
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        ]
        if not bn:
            layers = rm_bn(layers)
        f6 = nn.Sequential(*layers)

        self.features = Features(base)

        self.extra_layers = nn.ModuleList([
            f2,
            f3,
            f4,
            f5,
            f6
        ])

        self.cl = nn.ModuleList([
            nn.Conv2d(512, 4 * self.num_cl, 3, padding=1),
            nn.Conv2d(1024, 6 * self.num_cl, 3, padding=1),
            nn.Conv2d(512, 6 * self.num_cl, 3, padding=1),
            nn.Conv2d(256, 6 * self.num_cl, 3, padding=1),
            nn.Conv2d(256, 4 * self.num_cl, 3, padding=1),
            nn.Conv2d(256, 4 * self.num_cl, 3, padding=1)
        ])

        self.bbx = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, 3, padding=1),
            nn.Conv2d(1024, 6 * 4, 3, padding=1),
            nn.Conv2d(512, 6 * 4, 3, padding=1),
            nn.Conv2d(256, 6 * 4, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        ])

        if init_base:
            self.features._init_weights()

        if init_extra:
            self._init_weights()
        
    def forward(self, x):
        out_cl = []
        out_bbx = []

        feats = []
        
        x0, x1 = self.features(x)

        feats.append(x0)

        prev = x1
        for f in self.extra_layers:
            x_curr = f(prev)
            feats.append(x_curr)
            prev = x_curr 
        
        for i, cl in enumerate(self.cl):
            cl_curr = cl(feats[i])
            out_cl.append(cl_curr)
        
        for i, bbx in enumerate(self.bbx):
            bbx_curr = bbx(feats[i])
            out_bbx.append(bbx_curr)

        for i in range(len(out_cl)):
            out_cl[i] = out_cl[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, self.num_cl)
            out_bbx[i] = out_bbx[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, 4)

        return torch.cat(out_cl, 1), torch.cat(out_bbx, 1)
    
    
    def _init_weights(self, vgg_16_init=False):
        self.extra_layers.apply(_weight_init)
        
        self.cl.apply(_weight_init)
        
        self.bbx.apply(_weight_init)

# model = Features('resnet')
# model = ssd(10, base='vgg')

# x = torch.rand(1,3,300,300)
# out = model(x)

# print(out[0].size())
# print(out[1].size())
