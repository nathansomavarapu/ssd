import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16

import itertools

import numpy as np


class ssd(nn.Module):
    def __init__(self, num_cl):
        super(ssd, self).__init__()

        self.num_cl = num_cl

        # TODO: Need to add batchnorm for all layers
        new_layers = list(vgg16(pretrained=True).features)
        new_layers[16] = nn.MaxPool2d(2, ceil_mode=True)
        new_layers[-1] = nn.MaxPool2d(3, 1, padding=1)

        self.f1 = nn.Sequential(*new_layers[:23])


        self.cl1 = nn.Sequential(
            nn.Conv2d(512, 4*(num_cl + 4), 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.base1 = nn.Sequential(*new_layers[23:])

        # The refrence code uses a dilation of 6 which requires a padding of 6
        self.f2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True)
        )

        self.cl2 = nn.Sequential(
            nn.Conv2d(1024, 6*(num_cl + 4), 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.f3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # This padding is likely wrong
            nn.ReLU(inplace=True)
        )

        self.cl3 = nn.Sequential(
            nn.Conv2d(512, 6*(num_cl + 4), 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.f4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # This padding is likely wrong
            nn.ReLU(inplace=True)
        )

        self.cl4 = nn.Sequential(
            nn.Conv2d(256, 6*(num_cl + 4), 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.f5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )

        self.cl5 = nn.Sequential(
            nn.Conv2d(256, 4*(num_cl + 4), 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.f6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )

        self.cl6 = nn.Sequential(
            nn.Conv2d(256, 4*(num_cl + 4), 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):

        x1 = self.f1(x)
        x1_2 = self.cl1(x1)

        x1_2 = x1_2.view(x1_2.size(0), -1, (self.num_cl + 4))

        x1 = self.base1(x1)
        
        x2 = self.f2(x1)
        x2_2 = self.cl2(x2)

        x2_2 = x2_2.view(x2_2.size(0), -1, (self.num_cl + 4))

        x3 = self.f3(x2)
        x3_2 = self.cl3(x3)

        x3_2 = x3_2.view(x3_2.size(0), -1, (self.num_cl + 4))

        x4 = self.f4(x3)
        x4_2 = self.cl4(x4)

        x4_2 = x4_2.view(x4_2.size(0), -1, (self.num_cl + 4))

        x5 = self.f5(x4)
        x5_2 = self.cl5(x5)

        x5_2 = x5_2.view(x5_2.size(0), -1, (self.num_cl + 4))

        x6 = self.f6(x5)
        x6_2 = self.cl6(x6)

        x6_2 = x6_2.view(x6_2.size(0), -1, (self.num_cl + 4))

        preds = torch.cat([x1_2, x2_2, x3_2, x4_2, x5_2, x6_2], 1)

        return preds[:,:,:self.num_cl], preds[:,:,self.num_cl:]
    
    def _get_pboxes(self, smin=0.2, smax=0.9, ars=[1, 2, (1/2.0), 3, (1/3.0)], fks=[38, 19, 10, 5, 3, 1], bmasks=[3, 5, 5, 5, 3, 3]):
        sks = [round(smin + (((smax-smin)/(len(fks)-1)) * (k-1)), 2) for k in range(1, len(fks) + 1)]
        sks = list(reversed(sks))

        boxes = []
        for k in range(len(fks)):
            fk = fks[k]
            for i, j in itertools.product(range(fk), range(fk)):
                cx = (i + 0.5)/fk
                cy = (j + 0.5)/fk

                wk_prime = hk_prime = np.sqrt(sks[k] * sks[min(k+1, len(sks) - 1)])

                boxes.append([cx, cy, wk_prime, hk_prime])

                for ar in ars[:bmasks[k]]:
                    wk = sks[k] * np.sqrt(ar)
                    hk = sks[k] / np.sqrt(ar)

                    boxes.append([cx, cy, wk, hk])
        
        boxes = np.array(boxes)
        return boxes
        

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ssd(10)
model = model.to(device)

print(model._get_pboxes().shape)

x = torch.zeros((1, 3, 300, 300))
x = x.to(device)

for out in model(x):
    print(out.size())
