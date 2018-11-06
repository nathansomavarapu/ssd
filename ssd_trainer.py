import torch

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

# TODO: Change this to perform tensor operations.
def gen_loss(def_bxs, ann_bxs, lens, thresh=0.5):

#     print(ann_bxs.size())
    ann_cls = ann_bxs[:,:,0]
    ann_cords = ann_bxs[:,:,1:5]

    def_cx = def_bxs[:,:,0]
    def_cy = def_bxs[:,:,1]
    def_w = def_bxs[:,:,2]
    def_h = def_bxs[:,:,3]

    defx1 = torch.clamp(def_cx - def_w, min=0)
    defx2 = torch.clamp(def_cx + def_w, max=1)
    defy1 = torch.clamp(def_cy - def_h, min=0)
    defy2 = torch.clamp(def_cy + def_h, max=1)

    ann_cx = ann_cords[:,:,0]
    ann_cy = ann_cords[:,:,1]
    ann_w = ann_cords[:,:,2]
    ann_h = ann_cords[:,:,3]

    annx1 = torch.clamp(ann_cx - ann_w, min=0)
    annx2 = torch.clamp(ann_cx + ann_w, max=1)
    anny1 = torch.clamp(ann_cy - ann_h, min=0)
    anny2 = torch.clamp(ann_cy + ann_h, max=1)

    def_a = (defx2 - defx1) * (defy2 - defy1)
    ann_a = (annx2 - annx1) * (anny2 - anny1)

    expanded_anns_a = ann_a.expand(def_a.size(1), ann_a.size(1)).unsqueeze(0)
    def_a = def_a.unsqueeze(2).expand_as(expanded_anns_a)

    annx1_ex = annx1.expand(defx1.size(1), annx1.size(1)).unsqueeze(0)
    i_x1 = torch.max(defx1.unsqueeze(2).expand_as(annx1_ex), annx1_ex)
    annx2_ex = annx2.expand(defx2.size(1), annx2.size(1)).unsqueeze(0)
    i_x2 = torch.min(defx2.unsqueeze(2).expand_as(annx2_ex), annx2_ex)
    anny1_ex = anny1.expand(defy1.size(1), anny1.size(1)).unsqueeze(0)
    i_y1 = torch.max(defy1.unsqueeze(2).expand_as(anny1_ex), anny1_ex)
    anny2_ex = anny2.expand(defy2.size(1), anny2.size(1)).unsqueeze(0)
    i_y2 = torch.min(defy2.unsqueeze(2).expand_as(anny2_ex), anny2_ex)

    i_diff_x = torch.clamp(i_x2 - i_x1, min=0)
    i_diff_y = torch.clamp(i_y2 - i_y1, min=0)
    
    intersection = i_diff_x * i_diff_y

    ious = intersection/(expanded_anns_a + def_a - intersection)
    max_ious, max_inds = torch.max(ious, dim=1)

    thresh_inds = ious > thresh

    print(max_ious)
#     print(thresh_ious > 0.5)

    print(max_ious.size())
    print(max_inds)
#     print(thresh_ious.size())

    print(thresh_inds.size())
    print(thresh_inds.nonzero().size())
#     thresh_inds = thresh_inds.squeeze().squeeze()
#     print(thresh_inds)

    # TODO: Join the two sets of boxes, implement hard negative mining, compute loss, add augmentations

def main():

#     trainset = LocData('/home/shared/workspace/coco_full/annotations/instances_train2017.json', '/home/shared/workspace/coco_full/train2017', 'COCO')
    trainset = LocData('/Users/nathan/Documents/Projects/data/annotations/instances_train2017.json', '/Users/nathan/Documents/Projects/data/train2017', 'COCO')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_fn_cust)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ssd(80)
    model = model.to(device)

    default_boxes = model._get_pboxes()
    default_boxes = default_boxes.to(device)

    for i, data in enumerate(trainloader):
        img, anns_gt, lens = data

        img = img.to(device)
        anns_gt = anns_gt.to(device)
        lens = lens.to(device)

        out_pred = model(img)

        gen_loss(default_boxes, anns_gt, lens)
        break
if __name__ == '__main__':
    main()