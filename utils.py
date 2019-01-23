import torch
import torch.nn as nn
import torch.functional as F

import numpy as np
import cv2
import itertools

def iou(tens1, tens2):

    assert tens1.size() == tens2.size()

    squeeze = False
    if tens1.dim == 2 and tens2.dim == 2:
        squeeze = True
        tens1 = tens1.unsqueeze(0)
        tens2 = tens2.unsqueeze(0)

    assert tens1.dim() == 3 
    assert tens1.size(-1) == 4 and tens2.size(-1) == 4

    maxs = torch.max(tens1[:,:,:2], tens2[:,:,:2])
    mins = torch.min(tens1[:,:,2:], tens2[:,:,2:])

    diff = torch.clamp(mins - maxs, min=0.0)

    intersection = diff[:,:,0] * diff[:,:,1]

    diff1 = torch.clamp(tens1[:,:,2:] - tens1[:,:,:2], min=0.0)
    area1 = diff1[:,:,0] * diff1[:,:,1]

    diff2 = torch.clamp(tens2[:,:,2:] - tens2[:,:,:2], min=0.0)
    area2 = diff2[:,:,0] * diff2[:,:,1]

    iou = intersection/(area1 + area2 - intersection)

    if squeeze:
        iou.squeeze(0)

    return iou

def center_to_points(center_tens):

    assert center_tens.dim() == 2 
    assert center_tens.size(1) == 4 

    lp = torch.clamp(center_tens[:,:2] - center_tens[:,2:]/2.0, min=0.0)
    rp = torch.clamp(center_tens[:,:2] + center_tens[:,2:]/2.0, max=1.0)

    points = torch.cat([lp, rp], 1)

    return points

def points_to_center(points_tens):

    assert points_tens.dim() == 2 
    assert points_tens.size(1) == 4

    widths = torch.clamp(points_tens[:,2:] - points_tens[:,:2], max=1.0)
    centers = torch.clamp(points_tens[:,:2] + widths/2.0, max=1.0)

    center = torch.cat([centers, widths], 1)

    return center

def draw_bbx(img, bbxs, color):
    
    assert type(img) == type(np.zeros(1))

    h, w, _ = img.shape

    for bbx in bbxs:
        lp = tuple((bbx[:2] * torch.tensor([w, h], dtype=torch.float).to(bbxs.device)).long().tolist())
        rp = tuple((bbx[2:] * torch.tensor([w, h], dtype=torch.float).to(bbxs.device)).long().tolist())
        img = cv2.rectangle(img, lp, rp, color)
    
    return img
    
def convert_to_tensor(img, size):

    assert type(img) == type(np.zeros(1))

    max_side = max(img.shape[:2])

    y_pad = int((max_side - img.shape[0])/2)
    x_pad = int((max_side - img.shape[1])/2)

    img = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)
    img_pad_s = img.shape
    img = cv2.resize(img, size)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0

    ratio_x = img.shape[1]/float(img_pad_s[1])
    ratio_y = img.shape[0]/float(img_pad_s[0])

    img = torch.from_numpy(img.transpose(2,0,1)).float()

    return img, (x_pad, y_pad), (ratio_x, ratio_y)

def convert_to_np(img, padding=None, orig_size=None):

    assert type(img) == type(torch.ones(1))

    img = img.data.cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    h, w, _ = img.shape

    if padding is not None:
        img = img[padding[1]:(h - padding[1]), padding[0]:(w - padding[0])]
    
    if orig_size is not None:
        img = cv2.resize(img, orig_size)
    
    return img

def get_dboxes(smin=0.07, smax=0.9, ars=[1, 2, (1/2.0), 3, (1/3.0)], fks=[38, 19, 10, 5, 3, 1], num_boxes=[3, 5, 5, 5, 3, 3]):
    m = len(fks)
    sks = [round(smin + (((smax-smin)/(m-1)) * (k-1)), 2) for k in range(1, m + 1)]

    boxes = []
    for k, feat_k in enumerate(fks):
        for i, j in itertools.product(range(feat_k), range(feat_k)):

            cx = (i + 0.5)/feat_k
            cy = (j + 0.5)/feat_k

            w = h = np.sqrt(sks[k] * sks[min(k+1, len(sks) - 1)])

            boxes.append([cx, cy, w, h])

            sk = sks[k]
            for ar in ars[:num_boxes[k]]:
                w = sk * np.sqrt(ar)
                h = sk / np.sqrt(ar)
                boxes.append([cx, cy, w, h])

    boxes = torch.tensor(boxes).float()
    return torch.clamp(boxes, max=1.0)

def expand_defaults_and_annotations(default_boxes, annotations_boxes):

    num_annotations = annotations_boxes.size(0)

    default_boxes = default_boxes.unsqueeze(0)
    default_boxes = default_boxes.expand(num_annotations, -1, -1)

    annotations_boxes = annotations_boxes.unsqueeze(1)
    annotations_boxes = annotations_boxes.expand_as(default_boxes)

    return default_boxes, annotations_boxes

def match(default_boxes, annotations_boxes, match_thresh):

    num_annotations = annotations_boxes.size(0)

    default_boxes_pt = center_to_points(default_boxes)
    annotations_boxes_pt = center_to_points(annotations_boxes)

    default_boxes_pt, annotations_boxes_pt = expand_defaults_and_annotations(default_boxes_pt, annotations_boxes_pt)

    ious = iou(default_boxes_pt, annotations_boxes_pt)

    _, annotation_with_box = torch.max(ious, 1)
    annotation_inds = torch.arange(num_annotations, dtype=torch.long).to(annotation_with_box.device)
    
    ious_max, box_with_annotation = torch.max(ious, 0)
    matched_boxes_bin = (ious_max >= match_thresh)
    matched_boxes_bin[annotation_with_box] = 1
    box_with_annotation[annotation_with_box] = annotation_inds
    
    return box_with_annotation, matched_boxes_bin

def compute_offsets(default_boxes, annotations_boxes, box_with_annotation_idx, use_variance=True):

    matched_boxes = annotations_boxes[box_with_annotation_idx]

    offset_cx = (matched_boxes[:,:2] - default_boxes[:,:2])

    if use_variance:
        offset_cx /= (default_boxes[:,2:] * 0.1)
    else:
        offset_cx /= default_boxes[:,2:]

    offset_wh = torch.log(matched_boxes[:,2:]/default_boxes[:,2:])

    if use_variance:
        offset_wh /= 0.2
    
    return torch.cat([offset_cx, offset_wh], 1)

def undo_offsets(default_boxes, predicted_offsets, use_variance=True):
    
    offset1_mult = default_boxes[:,2:]
    offset2_mult = 1
    if use_variance:
        offset1_mult *= 0.1
        offset2_mult *= 0.2
    
    cx = (offset1_mult * predicted_offsets[:,:2]) + default_boxes[:,:2]
    wh = torch.exp(default_boxes[:,2:] * predicted_offsets[:,2:]) * offset2_mult

    return torch.cat([cx, wh], 1)

def compute_loss(default_boxes, annotations_classes, annotations_boxes, predicted_classes, predicted_offsets, match_thresh=0.5, duplciate_checking=True, neg_ratio=3):
    
    annotations_classes = annotations_classes.long()
    box_with_annotation_idx, matched_box_bin = match(default_boxes, annotations_boxes, match_thresh)

    matched_box_idxs = (matched_box_bin.nonzero()).squeeze(1)
    non_matched_idxs = (matched_box_bin == 0).nonzero().squeeze(1)
    N = matched_box_idxs.size(0)

    true_offsets = compute_offsets(default_boxes, annotations_boxes, box_with_annotation_idx)

    regression_loss_criterion = nn.SmoothL1Loss(reduction='none')
    regression_loss = regression_loss_criterion(predicted_offsets[matched_box_idxs], true_offsets[matched_box_idxs])

    true_classifications = torch.zeros(predicted_classes.size(0), dtype=torch.long).to(predicted_classes.device)
    true_classifications[matched_box_idxs] = annotations_classes[box_with_annotation_idx[matched_box_idxs]]

    classifications_loss_criterion = nn.CrossEntropyLoss(reduction='none')
    classifications_loss_total = classifications_loss_criterion(predicted_classes, true_classifications)

    positive_classifications = classifications_loss_total[matched_box_idxs]
    negative_classifications = classifications_loss_total[non_matched_idxs]

    _, hard_negative_idxs = torch.sort(classifications_loss_total[non_matched_idxs], descending=True)
    hard_negative_idxs = hard_negative_idxs.squeeze()[:N * neg_ratio]

    classifications_loss = (positive_classifications.sum() + negative_classifications[hard_negative_idxs].sum())/N
    regression_loss = regression_loss.sum()/N

    return classifications_loss, regression_loss, matched_box_idxs