import torch
import os
import time

import utils
from torch.autograd import Variable

from torch.nn.functional import softmax

from ssd import ssd

class ModelRunner():

    def __init__(self, weights_path, base, bn, num_cats=2, verbose=False):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        model = ssd(num_cats, bn=bn, base=base)

        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
        self.model = model.to(device)
        self.model.eval()

        default_boxes = utils.get_dboxes()

        self.default_boxes = default_boxes.to(device)

        self.device = device

        self.verbose = False
    
    def run_inference(self, img, convert=True):

        with torch.no_grad():
            if convert:
                img = utils.convert_cv2_pil(img)
            
            img_t, _, _ = utils.convert_pil_tensor(img, (300,300), pad=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_t = img_t.unsqueeze(0).to(self.device)
            s = time.time()

            predicted_classes, predicted_boxes = self.model(img_t)

            # print(torch.max(softmax(predicted_classes, dim=2), dim=2)[1].unique())
            # print(predicted_boxes)

            predicted_classes = Variable(predicted_classes[0].data)
            predicted_boxes = Variable(predicted_boxes[0].data)

            predicted_boxes = utils.undo_offsets(self.default_boxes, predicted_boxes)
            predicted_boxes = utils.center_to_points(predicted_boxes)

            classes, _, scores = utils.get_nonzero_classes(predicted_classes, norm=True)
            classes_unique = torch.unique(classes)
            classes_unique = classes_unique[classes_unique != 0]

            nms_boxes, num_boxes = utils.nms_and_thresh(classes_unique, scores, classes, predicted_boxes, 0.5, 0.1)
            e = time.time()

            if self.verbose:
                print('Time for whole inference process: ' + str(e-s))
            
            return nms_boxes[:num_boxes]