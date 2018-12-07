import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16

import itertools

import numpy as np

class ssd(nn.Module):
    def __init__(self, num_cl, init_weights=True):
        super(ssd, self).__init__()

        self.num_cl = num_cl + 1
        self.layers = []


        new_layers = list(vgg16(pretrained=True).features)
        new_layers[16] = nn.MaxPool2d(2, ceil_mode=True)
        new_layers[-1] = nn.MaxPool2d(3, 1, padding=1)

        self.f1 = nn.Sequential(*new_layers[:23])
        self.bn1 = nn.BatchNorm2d(512)


        self.cl1 = nn.Sequential(
            nn.Conv2d(512, 4*self.num_cl, 3, padding=1)
        )
        self.layers.append(self.cl1)

        self.bbx1 = nn.Sequential(
            nn.Conv2d(512, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx1)

        self.base1 = nn.Sequential(*new_layers[23:])
        # self.layers.append(self.base1)

        for param in self.f1.parameters():
            param.requires_grad = False
        
        for param in self.base1.parameters():
            param.requires_grad = False

        # The refrence code uses a dilation of 6 which requires a padding of 6
        self.f2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f2)

        self.cl2 = nn.Sequential(
            nn.Conv2d(1024, 6 * self.num_cl, 3, padding=1)
        )
        self.layers.append(self.cl2)

        self.bbx2 = nn.Sequential(
            nn.Conv2d(1024, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx2)

        self.f3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f3)

        self.cl3 = nn.Sequential(
            nn.Conv2d(512, 6 * self.num_cl, 3, padding=1)
        )
        self.layers.append(self.cl3)

        self.bbx3 = nn.Sequential(
            nn.Conv2d(512, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx3)

        self.f4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f4)

        self.cl4 = nn.Sequential(
            nn.Conv2d(256, 6 * self.num_cl, 3, padding=1)
        )
        self.layers.append(self.cl4)

        self.bbx4 = nn.Sequential(
            nn.Conv2d(256, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx4)

        self.f5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f5)

        self.cl5 = nn.Sequential(
            nn.Conv2d(256, 4 * self.num_cl, 3, padding=1)
        )
        self.layers.append(self.cl5)

        self.bbx5 = nn.Sequential(
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx5)

        self.f6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f6)

        self.cl6 = nn.Sequential(
            nn.Conv2d(256, 4 * self.num_cl, 3, padding=1)
        )
        self.layers.append(self.cl6)

        self.bbx6 = nn.Sequential(
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx6)

        if init_weights:
            self._init_weights()
        
    def forward(self, x):

        out_cl = []
        out_bbx = []
        
        x1 = self.f1(x)
        x1 = self.bn1(x1)
        
        out_cl.append(self.cl1(x1))
        out_bbx.append(self.bbx1(x1))

        x1 = self.base1(x1)
        
        x2 = self.f2(x1)

        out_cl.append(self.cl2(x2))
        out_bbx.append(self.bbx2(x2))

        x3 = self.f3(x2)

        out_cl.append(self.cl3(x3))
        out_bbx.append(self.bbx3(x3))

        x4 = self.f4(x3)

        out_cl.append(self.cl4(x4))
        out_bbx.append(self.bbx4(x4))

        x5 = self.f5(x4)
        
        out_cl.append(self.cl5(x5))
        out_bbx.append(self.bbx5(x5))

        x6 = self.f6(x5)

        out_cl.append(self.cl6(x6))
        out_bbx.append(self.bbx6(x6))

        for i in range(len(out_cl)):
            out_cl[i] = out_cl[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, self.num_cl)
            out_bbx[i] = out_bbx[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, 4)

        return torch.cat(out_cl, 1), torch.cat(out_bbx, 1)
    
    def _get_pboxes(self, smin=0.1, smax=0.9, ars=[1, 2, (1/2.0), 3, (1/3.0)], fks=[38, 19, 10, 5, 3, 1], bmasks=[3, 5, 5, 5, 3, 3]):
        sks = [round(smin + (((smax-smin)/(len(fks)-1)) * (k-1)), 2) for k in range(1, len(fks) + 1)]

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
            
        
        boxes = torch.tensor(np.array(boxes)).float()
        return torch.clamp(boxes, max=1.0)
    
    def _init_weights(self):

        for module in self.layers:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

# import numpy as np  

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# img = torch.zeros((1, 3, 300, 300)).to(device)

# model = ssd(1)
# # model = Net()
# model = model.to(device)

# opt = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=1.0)

# cl_loss = nn.CrossEntropyLoss()

# for i in range(500):

#     opt.zero_grad()

#     pred = model(img)
#     pred = pred[0][0]

#     print((torch.max(pred,1)[1] == 1).sum())

#     loss = cl_loss(pred, torch.ones((pred.size(0)), dtype=torch.long).to(device))
#     print(loss)
#     loss.backward()
#     opt.step()


# np.savetxt('pred.txt', pred.detach().cpu().numpy())



