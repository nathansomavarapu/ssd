import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

import torch.optim as optim

import cv2

# TODO: Change this to perform tensor operations.
def gen_loss(def_bxs, ann_bxs, pred, lens, device, thresh=0.5):

	pred_cl = pred[0]
	pred_reg = pred[1]
	# Add lens here for accuracy
	ann_cl = ann_bxs[:,:,0]
	ann_cords = ann_bxs[:,:,1:5]

	batch_size = ann_bxs.size(0)
	num_dbx = def_bxs.size(1)
	num_ann = ann_cords.size(1)

	def_mins = def_bxs[:,:2] - def_bxs[:,2:]/2.0
	def_maxs = def_bxs[:,:2] + def_bxs[:,2:]/2.0

	def_pts = torch.cat([def_mins, def_maxs], 1)

	ann_mins = ann_cords[:,:,:2] - ann_cords[:,:,2:]/2.0
	ann_maxs = ann_cords[:,:,:2] + ann_cords[:,:,2:]/2.0

	ann_pts = torch.cat([ann_mins, ann_maxs], 2)

	def_pts = def_pts.unsqueeze(0).unsqueeze(0)
	ann_pts = ann_pts.unsqueeze(2)

	def_pts = def_pts.expand(batch_size, num_ann, -1, -1)
	ann_pts = ann_pts.expand_as(def_pts)
	
	mins = torch.min(def_pts[:,:,:,2:], ann_pts[:,:,:,2:])
	maxs = torch.max(def_pts[:,:,:,:2], ann_pts[:,:,:,:2])

	diff = torch.clamp(mins - maxs, min=0)
	intersect = diff[:,:,:,0] * diff[:,:,:,1]

	def_diff = def_pts[:,:,:,2:] - def_pts[:,:,:,:2]
	def_area = def_diff[:,:,:,0] * def_diff[:,:,:,1]

	ann_diff = ann_pts[:,:,:,2:] - ann_pts[:,:,:,:2]
	ann_area = ann_diff[:,:,:,0] * ann_diff[:,:,:,1]

	ious = intersect / (def_area + ann_area - intersect)

	max_anns, max_ann_inds = torch.max(ious, 1)
	max_defs, max_def_inds = torch.max(ious, 2)


	print(num_ann)
	print(max_anns.size())
	print(max_defs.size())


	return None, None

def main():

	trainset = LocData('../data/annotations/instances_train2017.json', '../data/train2017', 'COCO', testing=True)
	trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_fn_cust)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = ssd(len(trainset.get_categories()))
	model = model.to(device)

	default_boxes = model._get_pboxes()
	default_boxes = default_boxes.to(device)

	print(default_boxes.size())

	# opt = optim.SGD(model.parameters(), lr=0.001)

	for i, data in enumerate(trainloader):
		img, anns_gt, lens = data

		# cv2.imwrite('curr_debug.png', img_old)

		img = img.to(device)
		anns_gt = anns_gt.to(device)
		lens = lens.to(device)

		pred = model(img)

		# print(anns_gt)
		# print(pred[0].size(), pred[1].size())

		loss, diagnostics = gen_loss(default_boxes, anns_gt, pred, lens, device)
	# 	print(loss)
	# 	for bbx in default_boxes[diagnostics.byte()]:
	# 		img_old = cv2.rectangle(img_old, ((bbx[0] - bbx[2]) * img_old.shape[1], (bbx[1] - bbx[3]) * img_old.shape[0]), ((bbx[0] + bbx[2]) * img_old.shape[1], (bbx[1] + bbx[3]) * img_old.shape[0]), (255,0,0))
		
	# 	for ann in anns_gt[0]:
	# 		bbx = ann[1:]
	# 		img_old = cv2.rectangle(img_old, ((bbx[0] - bbx[2]) * img_old.shape[1], (bbx[1] - bbx[3]) * img_old.shape[0]), ((bbx[0] + bbx[2]) * img_old.shape[1], (bbx[1] + bbx[3]) * img_old.shape[0]), (0,255,0))

	# 	cv2.imwrite('img_w_def_anns.png', img_old)
	# 	opt.zero_grad()
	# 	loss.backward()
	# 	opt.step()


		break
if __name__ == '__main__':
	main()