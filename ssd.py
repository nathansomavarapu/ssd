import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16


class ssd(nn.Module):
    def __init__(self, num_cl):
        super(ssd, self).__init__()

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
        # print(x1.size())
        x1_2s = x1_2.size()
        x1_2 = x1_2.view(x1_2s[0],x1_2s[2] * x1_2s[3], 4, -1)
        # print(x1_2.size())

        x1 = self.base1(x1)
        
        x2 = self.f2(x1)
        x2_2 = self.cl2(x2)
        # print(x2.size())
        x2_2s = x2_2.size()
        x2_2 = x2_2.view(x2_2s[0], x2_2s[2] * x2_2s[3], 6, -1)
        # print(x2_2.size())

        x3 = self.f3(x2)
        x3_2 = self.cl3(x3)
        # print(x3.size())
        x3_2s = x3_2.size()
        x3_2 = x3_2.view(x3_2s[0], x3_2s[2] * x3_2s[3], 6, -1)
        # print(x3_2.size())

        x4 = self.f4(x3)
        x4_2 = self.cl4(x4)
        # print(x4.size())
        x4_2s = x4_2.size()
        x4_2 = x4_2.view(x4_2s[0], x4_2s[2] * x4_2s[3], 6, -1)
        # print(x4_2.size())

        x5 = self.f5(x4)
        x5_2 = self.cl5(x5)
        # print(x5.size())
        x5_2s = x5_2.size()
        x5_2 = x5_2.view(x5_2s[0], x5_2s[2] * x5_2s[3], 4, -1)
        # print(x5_2.size())

        x6 = self.f6(x5)
        x6_2 = self.cl6(x6)
        # print(x6.size())
        x6_2s = x6_2.size()
        x6_2 = x6_2.view(x6_2s[0], x6_2s[2] * x6_2s[3], 4, -1)
        # print(x6_2.size())

        return torch.cat([x1_2, x5_2, x6_2], dim=1), torch.cat([x2_2, x3_2, x4_2], dim=1)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = ssd(10)
# model = model.to(device)

# x = torch.zeros((1, 3, 300, 300))
# x = x.to(device)

# print(model(x))
