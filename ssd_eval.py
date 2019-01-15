import torch
import torch.nn as nn
import torch.functional as F

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

import torch.optim as optim

import cv2
import numpy as np

import os

def nms():
    pass

if __name__ == "__main__":
    testset = LocData('../data/annotations2014/instances_val2014.json', '../data/val2014', 'COCO')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_cats = len(testset.get_categories())

    model = ssd(num_cats, init_weights=True)
    if os.path.exists('ssd.pt'):
        model.load_state_dict(torch.load('ssd.pt'))
    model = model.to(device)

    default_boxes = model._get_pboxes()
    default_boxes = default_boxes.to(device)

    img, anns = testset[1]
    img = img.to(device).unsqueeze(0)
    anns = anns.to(device).unsqueeze(0)
    with torch.no_grad():
        preds = model(img)

        print(preds[0].size(), preds[1].size())

        _, all_cl = torch.max(preds[0][0], 1)
        all_offsets = preds[1][0]

        img = img[0].data.cpu().numpy()
        img = np.transpose(img, (1,2,0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (img * 255).astype(np.uint8)
        img_pred = img.copy()
        gts = anns[0]
        non_background_inds = (all_cl != 0)

        nnb = torch.sum(non_background_inds).item()
        nnb_cls = torch.unique(all_cl[non_background_inds]).cpu().numpy()

        if all_offsets.size(0) > 0 and nnb > 0:				
            bbx_centers = (all_offsets[:,:2] * default_boxes[:,2:].float() * 0.1) + default_boxes[:,:2].float()
            bbx_widths = torch.exp(all_offsets[:,2:] * 0.2) * default_boxes[:,2:].float()

            bbx_cf = torch.cat([bbx_centers, bbx_widths], 1)
            bbx_cf = bbx_cf[non_background_inds]

            pred_min = torch.clamp(bbx_cf[:,:2] - bbx_cf[:,2:]/2.0, min=0)
            pred_max = torch.clamp(bbx_cf[:,:2] + bbx_cf[:,2:]/2.0, max=1.0)

            bbxs_pred = torch.cat([pred_min, pred_max], 1)
            
            for i in range(bbxs_pred.size(0)):
                bbx = bbxs_pred[i,:]
                lp = (int(bbx[0].item() * img.shape[1]) , int(bbx[1].item() * img.shape[0]))
                rp = (int(bbx[2].item() * img.shape[1]), int(bbx[3].item() * img.shape[0]))

                cv2.rectangle(img_pred, lp, rp, (0,0,255))

        if gts.size(0) > 0:
            gts_cl = gts[:,0].unsqueeze(1)
            gts_min = gts[:,1:3] - gts[:,3:]/2.0
            gts_max = gts[:,1:3] + gts[:,3:]/2.0

            gts = torch.cat([gts_cl, gts_min, gts_max], 1)
            for i in range(gts.size(0)):
                ann_box = gts[i,1:5]
                lp = (int(ann_box[0].item() * img.shape[1]) , int(ann_box[1].item() * img.shape[0]))
                rp = (int(ann_box[2].item() * img.shape[1]), int(ann_box[3].item() * img.shape[0]))
                cv2.rectangle(img, lp, rp, (0,255,0))
                cv2.rectangle(img_pred, lp, rp, (0,255,0))
        
        cv2.imwrite('predicted.png', img_pred)
