import torch
import torch.nn as nn
import torch.functional as F

import torchvision
import torchvision.transforms.functional as TF

from PIL import Image
from PIL import ImageDraw

import cv2

import numpy as np
import itertools

# Takes in two tensors with boxes in point form and computes iou between them.
def iou(tens1, tens2):

    assert tens1.size() == tens2.size()

    squeeze = False
    if tens1.dim() == 2 and tens2.dim() == 2:
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
        iou = iou.squeeze(0)
    
    return iou

def center_to_points(center_tens):

    if center_tens.size(0) == 0:
        return center_tens
    
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

def draw_bbx(img, bbxs, color, classes=None, pil=True):
    
    if pil:

        assert img.mode == "RGB"

        img = img.copy()
        w, h = img.size

        draw = ImageDraw.Draw(img)

        for i, bbx in enumerate(bbxs):
            lp = tuple((bbx[:2] * torch.tensor([w, h], dtype=torch.float).to(bbxs.device)).round().long().tolist())
            rp = tuple((bbx[2:] * torch.tensor([w, h], dtype=torch.float).to(bbxs.device)).round().long().tolist())
            draw.rectangle([lp, rp], outline=color)
            if classes is not None:
                text_lp = (max(lp[0] - 10, 0), max(lp[1] - 10, 0))
                # TODO: This is not going to work, figure out a good way to load in font
                img = draw.text(text_lp, classes[i])
    else:

        assert type(img) == type(np.ones(1))

        img = img.copy()
        h, w = img.shape[:2]

        for i, bbx in enumerate(bbxs):
            lp = tuple((bbx[:2] * torch.tensor([w, h], dtype=torch.float).to(bbxs.device)).round().long().tolist())
            rp = tuple((bbx[2:] * torch.tensor([w, h], dtype=torch.float).to(bbxs.device)).round().long().tolist())
            img = cv2.rectangle(img, lp, rp, color)
            if classes is not None:
                text_lp = (max(lp[0] - 10, 0), max(lp[1] - 10, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.putText(img, classes[i], text_lp, font)
    
    return img

def pad(img):

    w, h = img.size
    m = max(w, h)

    diffx = m - w
    diffy = m - h

    new_img = Image.new('RGB', (m, m))
    new_img.paste(img, (diffx//2, diffy//2))

    return new_img, diffx//2, diffy//2
    
def convert_pil_tensor(img, size, pad=True):

    assert img.mode == "RGB"

    x_pad = y_pad = 0
    if pad:
        img, x_pad, y_pad = pad(img)
    w, h = img.size

    img = TF.resize(img, size)
    w_new, h_new = img.size

    ratio_x = w_new/float(w)
    ratio_y = h_new/float(h)

    img = TF.to_tensor(img)

    return img, (x_pad, y_pad), (ratio_x, ratio_y)

def convert_tens_pil(img, padding=None, orig_size=None):

    assert type(img) == type(torch.ones(1))

    img = TF.to_pil_image(img.cpu())

    w,h = img.size
    

    if padding is not None:
        img = img[padding[1]:(h - padding[1]), padding[0]:(w - padding[0])]
    
    if orig_size is not None:
        img = img.resize(orig_size)
    
    return img

def convert_cv2_pil(img):

    assert type(img) == type(np.ones(1))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil

def convert_pil_cv2(img):
    
    assert img.mode == "RGB"
    
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
        offset_cx = offset_cx / (default_boxes[:,2:] * 0.1)
    else:
        offset_cx = offset_cx / default_boxes[:,2:]

    offset_wh = torch.log(matched_boxes[:,2:]/default_boxes[:,2:])

    if use_variance:
        offset_wh = offset_wh / 0.2
    
    return torch.cat([offset_cx, offset_wh], 1)

def undo_offsets(default_boxes, predicted_offsets, use_variance=True):
    
    offset1_mult = default_boxes[:,2:]
    offset2_mult = 1
    if use_variance:
        offset1_mult = offset1_mult * 0.1
        offset2_mult = offset2_mult * 0.2
    
    cx = (offset1_mult * predicted_offsets[:,:2]) + default_boxes[:,:2]
    wh = torch.exp(predicted_offsets[:,2:] * offset2_mult) * default_boxes[:,2:]

    return torch.cat([cx, wh], 1)

def compute_loss(default_boxes, annotations_classes, annotations_boxes, predicted_classes, predicted_offsets, match_thresh=0.5, duplciate_checking=True, neg_ratio=3):
    
    if annotations_classes.size(0) > 0:
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
    
    else:
        matched_box_idxs = torch.LongTensor([])
        non_matched_idxs = torch.arange(default_boxes.size(0))
        N = 1

        regression_loss = torch.tensor([0.0]).to(predicted_classes.device)

        true_classifications = torch.zeros(predicted_classes.size(0), dtype=torch.long).to(predicted_classes.device)
            
    classifications_loss_criterion = nn.CrossEntropyLoss(reduction='none')
    classifications_loss_total = classifications_loss_criterion(predicted_classes, true_classifications)

    positive_classifications = classifications_loss_total[matched_box_idxs]
    negative_classifications = classifications_loss_total[non_matched_idxs]

    _, hard_negative_idxs = torch.sort(classifications_loss_total[non_matched_idxs], descending=True)
    hard_negative_idxs = hard_negative_idxs.squeeze()[:N * neg_ratio]

    classifications_loss = (positive_classifications.sum() + negative_classifications[hard_negative_idxs].sum())/N
    regression_loss = regression_loss.sum()/N

    return classifications_loss, regression_loss, matched_box_idxs

def get_nonzero_classes(predicted_classes, norm=False):

    if norm:
        pred_exp = torch.exp(predicted_classes)
        predicted_classes = pred_exp/pred_exp.sum(dim=1, keepdim=True)

    scores, classes = torch.max(predicted_classes, 1)

    non_zero_pred_idxs = (classes != 0).nonzero()

    if non_zero_pred_idxs.dim() > 1:
        non_zero_pred_idxs = non_zero_pred_idxs.squeeze(1)
    
    return classes, non_zero_pred_idxs, scores

# Did this in numpy previously, torch verion adapted from Fransisco Massa's orginal nms code.
def nms_and_thresh(classes_unique, scores, all_classes, predicted_boxes, nms_thresh, class_thresh, topk=100):

    boxes = torch.zeros((classes_unique.size(0) * topk, 6))
    box_ctr = 0
    
    for cl in classes_unique:
        
        current_class_idxs = (all_classes == cl).nonzero()

        if current_class_idxs.dim() > 1:
            current_class_idxs = current_class_idxs.squeeze(1)
                
        current_class_scores = scores[current_class_idxs]
        current_boxes = predicted_boxes[current_class_idxs]

        sorted_scores, class_idxs_by_score = torch.sort(current_class_scores, descending=True)
        class_idxs_by_score = class_idxs_by_score[sorted_scores >= class_thresh]
        class_idxs_by_score = class_idxs_by_score[:topk]

        sorted_scores = sorted_scores[sorted_scores >= class_thresh]
        sorted_scores = sorted_scores[:topk]


        while len(class_idxs_by_score) > 0:

            curr_bbx = current_boxes[class_idxs_by_score[0]]
            boxes[box_ctr,0] = cl
            boxes[box_ctr,1] = sorted_scores[0]
            boxes[box_ctr,2:6] = curr_bbx
            box_ctr += 1

            other_bbxs = current_boxes[class_idxs_by_score]
            curr_bbx = curr_bbx.expand_as(other_bbxs)

            ious = iou(curr_bbx, other_bbxs)
            
            class_idxs_by_score = class_idxs_by_score[ious <= nms_thresh]
            sorted_scores = sorted_scores[ious <= nms_thresh]
        
    return boxes, box_ctr
