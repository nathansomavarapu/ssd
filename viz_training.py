import cv2
import numpy as np
import torch
from visdom import Visdom

import utils


class VisdomTrainer():
    def __init__(self, port, hostname):
        self.viz = Visdom(port=port, server=hostname)
        self.win_loc = None
        self.win_cl = None
        self.win_match = None
        self.win_ann = None
        self.win_pred = None
        self.text_window_tr = None
        self.text_window_pred = None
        self.viz_counter = 0


    def update_viz(self, cl_loss_val, loc_loss_val, img, default_boxes, match_idxs, ann_cl, ann_boxes, predicted_classes, predicted_offsets):

        x_axis = np.array([self.viz_counter])
        loc_data = np.array([loc_loss_val])
        cl_data = np.array([cl_loss_val])

        classes, non_zero_pred_idxs, _ = utils.get_nonzero_classes(predicted_classes)

        ann_classes = torch.unique(ann_cl).long().tolist()
        pred_classes = torch.unique(classes).tolist()

        match_img = self.get_match_img(img, default_boxes, match_idxs, ann_boxes)
        pred_img = self.get_pred_img(img, default_boxes, predicted_offsets, non_zero_pred_idxs)

        true_cl_str = 'True Classes: ' + str(ann_classes)
        pred_cl_str = 'Predicted Classes: ' + str(pred_classes)

        if self.win_cl is None or self.win_loc is None or self.win_match is None:
            self.win_cl = self.viz.line(X=x_axis, Y=cl_data, opts={'linecolor': np.array([[0, 0, 255],]), 'title': 'Classification Loss'})
            self.win_loc = self.viz.line(X=x_axis, Y=loc_data, opts={'linecolor': np.array([[255, 165, 0],]), 'title': 'Localization Loss'})
            self.win_match = self.viz.image(match_img, opts={'title':'Match Image'})
            self.win_pred = self.viz.image(pred_img, opts={'title':'Predictions Image'})
            self.text_window_tr = self.viz.text(true_cl_str)
            self.text_window_pred = self.viz.text(pred_cl_str)

        self.viz.line(X=x_axis, Y=cl_data, win=self.win_cl, update='append')
        self.viz.line(X=x_axis, Y=loc_data, win=self.win_loc, update='append')

        self.viz.image(match_img, win=self.win_match, opts={'title':'Match Image'})
        self.viz.image(pred_img, win=self.win_pred, opts={'title':'Predictions Image'})

        self.viz.text(true_cl_str, win=self.text_window_tr)
        self.viz.text(pred_cl_str, win=self.text_window_pred)

        self.viz_counter += 1

    def get_match_img(self, img, default_boxes, match_idxs, ann_boxes):

        img = img.copy()

        match_boxes = default_boxes[match_idxs]
        match_boxes_pt = utils.center_to_points(match_boxes)

        ann_boxes_pt = utils.center_to_points(ann_boxes)

        drawn_match_img = utils.draw_bbx(img, match_boxes_pt, [255, 0, 0])
        drawn_match_img = utils.draw_bbx(img, ann_boxes_pt, [0, 255, 0])

        return utils.convert_cvimg_pilimg(drawn_match_img)

    def get_pred_img(self, img, default_boxes, predicted_offsets, non_zero_pred_idxs):

        img = img.copy()

        predicted_boxes = utils.undo_offsets(default_boxes, predicted_offsets)
        predicted_boxes_pt = utils.center_to_points(predicted_boxes)
        predicted_boxes_pt = predicted_boxes_pt[non_zero_pred_idxs]

        pred_img = utils.draw_bbx(img, predicted_boxes_pt, [0, 0, 255])
        pred_img = utils.convert_cvimg_pilimg(pred_img)

        return pred_img
