import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F

import torchvision
from torch.utils.data import DataLoader

from ssd import ssd
import utils

import cv2
import numpy as np
import os
import glob
import time


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_cats = 21
    weights_path = os.path.join('weights/', 'ssd_weights_voc_250.pt')
    samples_path = 'samples/test_imgs'

    model = ssd(num_cats)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()

    default_boxes = utils.get_dboxes()
    default_boxes = default_boxes.to(device)

    with torch.no_grad():
        for i, img_f in enumerate(glob.glob(os.path.join(samples_path, '*.jpg'))):
            img = cv2.imread(img_f)
            img_t, _, _ = utils.convert_to_tensor(img, model.size)

            img_t = img_t.unsqueeze(0).to(device)

            s = time.time()
            img = utils.convert_to_np(img_t[0])
            e = time.time()

            print(abs(e - s))

            predicted_classes, predicted_boxes = model(img_t)
            predicted_classes = Variable(predicted_classes[0].data)
            predicted_boxes = Variable(predicted_boxes[0].data)

            predicted_boxes = utils.undo_offsets(default_boxes, predicted_boxes)
            predicted_boxes = utils.center_to_points(predicted_boxes)

            classes, nonzero_pred_idxs, scores = utils.get_nonzero_classes(predicted_classes, norm=True)
            classes_unique = torch.unique(classes)
            classes_unique = classes_unique[classes_unique != 0]

            nms_boxes, num_boxes = utils.nms_and_thresh(classes_unique, scores, classes, predicted_boxes, 0.5, 0.65)

            img_pred_nms = utils.draw_bbx(img, nms_boxes[:num_boxes], [0, 0, 255])
            cv2.imwrite('pred_' + str(i) + '.png', img_pred_nms)