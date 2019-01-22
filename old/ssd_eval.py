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
import glob

topk = 100

def convert_np_img(img, size=(300,300)):

    max_side = max(img.shape[:2])
    
    y_pad = int((max_side - img.shape[0])/2)
    x_pad = int((max_side - img.shape[1])/2)

    img = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)
    img_pad_s = img.shape
    img = cv2.resize(img, size)
    img_old = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = img.transpose(2,0,1)

    return torch.from_numpy(img).float()


# Did this in numpy previously, torch verion adapted from Fransisco Massa's orginal nms code.
def nms(boxes, sorted_ind, nms_thresh):

    keep = torch.zeros((boxes.size(0)), dtype=torch.long)
    keep_ctr = 0

    while len(sorted_ind) > 0:

        keep[keep_ctr] = sorted_ind[0]
        keep_ctr += 1
        
        curr_bbx = boxes[sorted_ind[0]]
        curr_boxes = boxes[sorted_ind]
        curr_bbx = curr_bbx.expand_as(curr_boxes)
        
        inter_maxs = torch.min(curr_bbx[:,2:], curr_boxes[:,2:])
        inter_mins = torch.max(curr_bbx[:,:2], curr_boxes[:,:2])

        diff = (inter_maxs - inter_mins).clamp(min=0.0)
        intersect = diff[:,0] * diff[:,1]

        area_a = curr_bbx[:,2:] - curr_bbx[:,:2]
        area_a = area_a[:,0] * area_a[:,1]

        area_b = curr_boxes[:,2:] - curr_boxes[:,:2]
        area_b = area_b[:,0] * area_b[:,1]

        iou = intersect / (area_a + area_b - intersect)

        sorted_ind = sorted_ind[iou <= nms_thresh]

    return keep, keep_ctr

if __name__ == "__main__":
    testset = LocData('../data/annotations2014/instances_val2014.json', '../data/val2014', 'COCO')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_cats = len(testset.get_categories())

    model = ssd(num_cats, init_weights=True)
    if os.path.exists('ssd.pt'):
        model.load_state_dict(torch.load('ssd.pt'))
    model = model.to(device)
    model.eval()

    default_boxes = model._get_pboxes()
    default_boxes = default_boxes.to(device)

    with torch.no_grad():
        for i, img_f in enumerate(glob.glob('./samples/*.jpg')):
            img = convert_np_img(cv2.imread(img_f)).to(device).unsqueeze(0)
            preds = model(img)

            pred_exp = torch.exp(preds[0])

            pred_exp_norm = pred_exp/pred_exp.sum(dim=2, keepdim=True)
            
            scores, all_cl = torch.max(pred_exp_norm[0], 1)
            all_offsets = preds[1][0]

            img = img[0].data.cpu().numpy()
            img = np.transpose(img, (1,2,0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = (img * 255).astype(np.uint8)
            img_pred = img.copy()

            
            non_background_inds = all_cl.nonzero().squeeze()

            if non_background_inds.size(0) > 0:

                nnb_cls = torch.unique(all_cl[non_background_inds])

                all_cl = all_cl[non_background_inds]
                scores = scores[non_background_inds]

                bbx_centers = (all_offsets[:,:2] * default_boxes[:,2:].float() * 0.1) + default_boxes[:,:2].float()
                bbx_widths = torch.exp(all_offsets[:,2:] * 0.2) * default_boxes[:,2:].float()

                bbx_cf = torch.cat([bbx_centers, bbx_widths], 1)
                bbx_cf = bbx_cf[non_background_inds]

                pred_min = torch.clamp(bbx_cf[:,:2] - bbx_cf[:,2:]/2.0, min=0)
                pred_max = torch.clamp(bbx_cf[:,:2] + bbx_cf[:,2:]/2.0, max=1.0)

                bbxs_pred = torch.cat([pred_min, pred_max], 1)

                for cl in nnb_cls:
                    curr_cls_inds = (all_cl == cl).nonzero().squeeze(1)
                    
                    curr_scores = scores[curr_cls_inds]
                    curr_boxes = bbxs_pred[curr_cls_inds]

                    _, s_ordered_cl_inds = torch.sort(curr_scores, descending=True)
                    s_ordered_cl_inds = s_ordered_cl_inds[:topk]

                    keep_inds, keep_num = nms(curr_boxes, s_ordered_cl_inds, 0.6)
                    bbxs_pred_nms = curr_boxes[keep_inds[:keep_num]]

                    for bbx in bbxs_pred_nms:
                        lp = (int(bbx[0].item() * img.shape[1]) , int(bbx[1].item() * img.shape[0]))
                        rp = (int(bbx[2].item() * img.shape[1]), int(bbx[3].item() * img.shape[0]))

                        cv2.rectangle(img_pred, lp, rp, (0,0,255))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img_pred, testset.coco_cats[cl], (lp[0] - 5, lp[1] - 5), font, 1, (244,66,143), 2)
        
            cv2.imwrite('predicted' + str(i) + '.png', img_pred)
