import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16

import itertools

import numpy as np

class Net(nn.Module):
    
    def __init__(self, num_cl, init_weights=True):
        super(Net, self).__init__()

        self.num_cl = num_cl + 1

        new_layers = list(vgg16(pretrained=True).features)
        new_layers[16] = nn.MaxPool2d(2, ceil_mode=True)
        new_layers[-1] = nn.MaxPool2d(3, 1, padding=1)

        self.f1 = nn.Sequential(*new_layers[:23])

        self.cl1 = nn.Sequential(
            nn.Conv2d(512, 4 * self.num_cl, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.bbx1 = nn.Sequential(
            nn.Conv2d(512, 4 * 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        for param in self.f1.parameters():
            param.requires_grad = False
        
        if init_weights:
            self._init_weights()
    
    def forward(self, x):
        
        x = self.f1(x)
        x_cl = self.cl1(x)
        x_bbx = self.bbx1(x)

        x_cl = x_cl.permute(0,2,3,1)
        x_cl = x_cl.contiguous().view(x_cl.size(0), -1)
        x_cl = x_cl.view(x_cl.size(0), -1, self.num_cl)

        x_bbx = x_bbx.permute(0,2,3,1)
        x_bbx = x_bbx.contiguous().view(x_bbx.size(0), -1)
        x_bbx = x_bbx.view(x_bbx.size(0), -1, 4)

        return x_cl, x_bbx
    
    def _init_weights(self):

        for module in [self.cl1, self.bbx1]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)


img = torch.randint(0, 255, (1,3, 300,300))
model = Net(1)

cent = nn.CrossEntropyLoss()
opt = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001, momentum=0.9, weight_decay=0.00005)

target = torch.ones((model(img)[0][0].size(0)), dtype=torch.long)

for i in range(10):
    pred = model(img)[0][0]

    print(torch.max(pred, 1)[1].sum())

    loss = cent(pred, target)

    print(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()


    
