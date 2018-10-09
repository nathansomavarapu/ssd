import torch

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

def iou(bbx1, bbx2):
    b1x, b1y, b1w, b1h = tuple(bbx1)
    b2x, b2y, b2w, b2h = tuple(bbx2)

    # print(b1x, b1y, b1w, b1h)

    b1x1 = float(int(b1x - b1w/2.0))
    b1x2 = float(int(b1x + b1w/2.0))
    b1y1 = float(int(b1y - b1h/2.0))
    b1y2 = float(int(b1y + b1h/2.0))

    b2x1 = float(int(b2x - b2w/2.0))
    b2x2 = float(int(b2x + b2w/2.0))
    b2y1 = float(int(b2y - b2h/2.0))
    b2y2 = float(int(b2y + b2h/2.0))

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

def gen_loss(pred_maps, anns_gt, lens, alpha=1.0, iou_thresh=0.5, sk=[0.9, 0.76, 0.62, 0.48, .34, 0.2], ar=[1, 1, 2, (1/2.0), 3, (1/3.0)]):
    pass
    
    



def main():

    trainset = LocData('/home/shared/workspace/coco_full/annotations/instances_train2017.json', '/home/shared/workspace/coco_full/train2017', 'COCO')
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collate_fn_cust)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ssd(10)
    model = model.to(device)  

    for i, data in enumerate(trainloader):
        img, anns_gt, lens = data

        img = img.to(device)
        anns_gt = anns_gt.to(device)
        lens = lens.to(device)

        out_pred = model(img)
        gen_loss(out_pred, anns_gt, lens)
if __name__ == '__main__':
    main()