import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F

import torchvision
from torch.utils.data import DataLoader

from ssd import ssd
import utils

import numpy as np
import os
import glob
import time

import cv2
import utils
import random

from PIL import Image

class ModelRunner():

    def __init__(self, num_cats=2):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        num_cats = 21
        weights_path = os.path.join('weights/', 'ssd_weights_voc.pt')

        model = ssd(num_cats)

        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
        self.model = model.to(device)
        self.model.eval()

        default_boxes = utils.get_dboxes()
        self.default_boxes = default_boxes.to(device)

        self.device = device
    
    def run_inference(self, img, convert=True):

        with torch.no_grad():
            if convert:
                img = utils.convert_cv2_pil(img)
            
            img_t, _, _ = utils.convert_pil_tensor(img, self.model.size, pad=False)
            img_t = img_t.unsqueeze(0).to(self.device)

            s = time.time()
            predicted_classes, predicted_boxes = self.model(img_t)
            predicted_classes = Variable(predicted_classes[0].data)
            predicted_boxes = Variable(predicted_boxes[0].data)

            predicted_boxes = utils.undo_offsets(self.default_boxes, predicted_boxes)
            predicted_boxes = utils.center_to_points(predicted_boxes)

            classes, _, scores = utils.get_nonzero_classes(predicted_classes, norm=True)
            classes_unique = torch.unique(classes)
            classes_unique = classes_unique[classes_unique != 0]

            nms_boxes, num_boxes = utils.nms_and_thresh(classes_unique, scores, classes, predicted_boxes, 0.5, 0.65)
            e = time.time()

            # print('Time for whole inference process: ' + str(e-s))
            
            return nms_boxes[:num_boxes]

if __name__ == "__main__":

    # cam = cv2.VideoCapture(0)
    runner = ModelRunner()

    # while(cam.isOpened()):
    #     ret, frame = cam.read()

    #     if ret == True:
            
    #         preds = runner.run_inference(frame, convert=True)

    #         frame = utils.draw_bbx(frame, preds, [0,0,255], pil=False)

    #         cv2.imwrite('face3.jpg', frame)

    #         cv2.imshow('Demo', frame)

    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    samples_path = '../data/VOC2007/test/JPEGImages/'

    int_to_cl = []

    with open('../data/VOC2007/train/classes.txt', 'r') as f:
        for line in f:
            int_to_cl.append(line.strip())
    
    base = 'eval'


    for i, img_f in enumerate(glob.glob(os.path.join(samples_path, '*.jpg'))):
        img = Image.open(img_f)
        w,h = img.size
        ann_f = img_f.replace('jpg', 'xml')
        preds = runner.run_inference(img, convert=False)

        img_num = img_f.split('/')[-1].replace('.jpg', '')

        for pred in preds:
            outfile = os.path.join(base, int_to_cl[int(pred[0]) - 1] + '.txt')

            line = '%s %.3f %.2f %.2f %.2f %.2f' % (img_num, pred[1], pred[2].item() * w, pred[3].item() * h, pred[4].item() * w, pred[5].item() * h)
            # line = '%s %.3f %.2f %.2f %.2f %.2f' % (img_num, pred[1], random.random() * w, random.random() * h, random.random() * w, random.random() * h)  

        with open(outfile, 'a+') as f:
            f.write(line + '\n')

        # img_pred_nms = utils.draw_bbx(img, preds, 'red')
        # img_pred_nms.save(os.path.join('samples', 'pred_images', 'pred_' + str(i) + '.png'))

            