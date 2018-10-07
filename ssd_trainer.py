import torch

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData
from ssd import ssd

def iou(bbx1, bbx2):

    assert len(bbx1) == len(bbx2) == 4

    b1x1, b1x2, b1y1, b1y2 = tuple(bbx1)
    b2x1, b2x2, b2y1, b2y2 = tuple(bbx2)

    bbx1_a = (float(b1x2) - b1x1) * (b1y2 - b1y2)
    bbx2_a = (float(b2x2) - b2x1) * (b2y2 - b2y1)

    i_x1 = max(float(b1x1), b2x1)
    i_x2 = min(float(b1x2), b2x2)
    i_y1 = max(float(b1y1), b2y1)
    i_y2 = min(float(b1y2), b2y2)

    if (i_x2 - i_x1) <= 0 or (i_y2 - i_y1) <= 0:
        return 0
    
    intersection = (i_y2 - i_y1) * (i_x2 - i_x1)

    return intersection/(bbx1_a + bbx2_a - intersection)



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