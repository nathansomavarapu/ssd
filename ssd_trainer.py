import torch

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData
from ssd import ssd

def main():

    trainset = LocData('/home/shared/workspace/coco_full/annotations/instances_train2017.json', '/home/shared/workspace/coco_full/train2017', 'COCO')
    trainloader = DataLoader(LocData, batch_size=4, shuffle=True)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ssd(10)
    model.to(device)

    for i, data in enumerate(trainloader):
        img, anns = data
        
        img = img.to(device)
        print(img.size())
        # model(img)

    


if __name__ == '__main__':
    main()