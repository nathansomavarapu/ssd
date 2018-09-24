import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16


class ssd(nn.Module):
    def __init__(self, num_cl):
        super(ssd, self).__init__()

        # TODO: Need to add batchnorm for all layers
        new_layers = list(vgg16(pretrained=True).features)
        new_layers[-1] = nn.MaxPool2d(3, 1)

        self.f1 = nn.Sequential(*new_layers[:23])

        self.base1 = nn.Sequential(*new_layers[23:])

        # The refrence code uses  a dilation of 6 which requires a padding of 6
        self.f2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, dilation=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(1024, 1024, 1, stride=0), 
            nn.ReLU(inplace=True)
        )

        self.f3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # This padding is likely wrong
            nn.ReLU(inplace=True)
        )

        self.f4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.f5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.cl1 = nn.Sequential(
            nn.Conv2d(512, 4*(num_cl + 4), 3),
            nn.ReLU(inplace=True)
        )
        
        self.cl2 = nn.Sequential(
            nn.Conv2d(1024, 6*(num_cl + 4), 3),
            nn.ReLU(inplace=True)
        )

        self.cl5 = nn.Sequential(
            nn.Conv2d(512, 4*(num_cl + 4), 3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        pass


model = ssd(10)
