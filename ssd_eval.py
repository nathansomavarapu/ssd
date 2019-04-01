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

from PIL import Image

class ModelRunner():

    def __init__(self, num_cats=2):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_cats = 2
        weights_path = os.path.join('weights/', 'ssd_weights_499.pt')

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

            print('Time for whole inference process:' + str(e-s))
            
            return nms_boxes[:num_boxes]

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    runner = ModelRunner()

    while(cam.isOpened()):
        ret, frame = cam.read()

        if ret == True:
            
            preds = runner.run_inference(frame, convert=True)

            frame = utils.draw_bbx(frame, preds, [0,0,255], pil=False)

            cv2.imwrite('face3.jpg', frame)

            cv2.imshow('Demo', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # samples_path = 'samples/test_imgs'
    # for i, img_f in enumerate(glob.glob(os.path.join(samples_path, '*.jpg'))):
    #     img = Image.open(img_f)
    #     preds = runner.run_inference(img, convert=False)

    #     img_pred_nms = utils.draw_bbx(img, preds, 'red')
    #     img_pred_nms.save('pred_' + str(i) + '.png')

            