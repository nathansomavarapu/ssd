import torch

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

# TODO: Change this to perform tensor operations.
def iou(bbx1, bbx2):

    cx1, cy1, w1, h1 = tuple(bbx1)
    cx2, cy2, w2, h2 = tuple(bbx2)

    b1x1 = float(int(cx1 - w1))
    b1x2 = float(int(cx1 + w1))
    b1y1 = float(int(cy1 - h1))
    b1y2 = float(int(cy1 + h1))

    b2x1 = float(int(cx2 - w2))
    b2x2 = float(int(cx2 + w2))
    b2y1 = float(int(cy2 - h2))
    b2y2 = float(int(cy2 + h2))

    bbx1_a = (b1x2 - b1x1) * (b1y2 - b1y2)
    bbx2_a = (b2x2 - b2x1) * (b2y2 - b2y1)

    i_x1 = max(b1x1, b2x1)
    i_x2 = min(b1x2, b2x2)
    i_y1 = max(b1y1, b2y1)
    i_y2 = min(b1y2, b2y2)

    if (i_x2 - i_x1) <= 0 or (i_y2 - i_y1) <= 0:
        return 0
    
    intersection = (i_y2 - i_y1) * (i_x2 - i_x1)

    return intersection/(bbx1_a + bbx2_a - intersection)

def main():

    trainset = LocData('/home/shared/workspace/coco_full/annotations/instances_train2017.json', '/home/shared/workspace/coco_full/train2017', 'COCO')
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn_cust)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ssd(10)
    model = model.to(device)

    default_boxes = model._get_pboxes()

    for i, data in enumerate(trainloader):
        img, anns_gt, lens = data

        img = img.to(device)
        anns_gt = anns_gt.to(device)
        lens = lens.to(device)

        out_pred = model(img)
        gen_loss(out_pred, anns_gt, lens)
if __name__ == '__main__':
    main()