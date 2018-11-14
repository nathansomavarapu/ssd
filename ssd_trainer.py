import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

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

	def_a = def_a.view(batch_size, num_dbx, 1)
	expanded_def_a = def_a.expand(batch_size, num_dbx, num_ann)
	expanded_ann_a = ann_a.expand_as(expanded_def_a)

	defx1 = defx1.view(batch_size, num_dbx, 1)
	defx1 = defx1.expand(batch_size, num_dbx, num_ann)
	defx2 = defx2.view(batch_size, num_dbx, 1)
	defx2 = defx2.expand(batch_size, num_dbx, num_ann)
	defy1 = defy1.view(batch_size, num_dbx, 1)
	defy1 = defy1.expand(batch_size, num_dbx, num_ann)
	defy2 = defy2.view(batch_size, num_dbx, 1)
	defy2 = defy2.expand(batch_size, num_dbx, num_ann)

	annx1 = annx1.expand_as(defx1)
	annx2 = annx2.expand_as(defx2)
	anny1 = anny1.expand_as(defy1)
	anny2 = anny2.expand_as(defy2)

	x1_intersect = torch.max(defx1, annx1)
	x2_intersect = torch.min(defx2, annx2)
	y1_intersect = torch.max(defy1, anny1)
	y2_intersect = torch.min(defy2, anny2)

	x_intersect = torch.clamp(x2_intersect - x1_intersect, min=0)
	y_intersect = torch.clamp(y2_intersect - y1_intersect, min=0)

	intersect = x_intersect * y_intersect

	ious = intersect/(expanded_ann_a + expanded_def_a - intersect)

	thresh_matches = ious > thresh
	_, max_inds = torch.max(ious, dim=1)

	max_matches = torch.zeros(ious.size()).to(device).scatter_(1, max_inds.unsqueeze(0), torch.ones(max_inds.size()).unsqueeze(0).to(device))

	match_inds = torch.clamp(thresh_matches.long() + max_matches.long(), max=2)
	N = match_inds.sum(2).sum(1)
	neg_N = 3 * N
	# print(N)

	if N == 0:
		return 0
	
	ann_cx = ann_cx.view(batch_size, 1, num_ann)
	ann_cx = ann_cx.expand(batch_size, num_dbx, num_ann)
	def_cx = def_cx.view(batch_size, num_dbx, -1).expand_as(ann_cx)
	def_w = def_w.view(batch_size, num_dbx, -1).expand_as(ann_cx)
	ann_w = ann_w.view(batch_size, 1, num_ann)
	ann_w = ann_w.expand(batch_size, num_dbx, num_ann)
	gcx = (ann_cx - def_cx)/def_w

	ann_cy = ann_cy.view(batch_size, 1, num_ann)
	ann_cy = ann_cy.expand(batch_size, num_dbx, num_ann)
	def_cy = def_cy.view(batch_size, num_dbx, -1).expand_as(ann_cy)
	def_h = def_h.view(batch_size, num_dbx, -1).expand_as(ann_cy)
	ann_h = ann_h.view(batch_size, 1, num_ann)
	ann_h = ann_h.expand(batch_size, num_dbx, num_ann)
	gcy = (ann_cy - def_cy)/def_h

	gw = torch.log(ann_w/def_w)
	gh = torch.log(ann_h/def_h)

	glist = [gcx, gcy, gw, gh]

	reg_crit = nn.SmoothL1Loss(reduction='none')

	total_loc_loss = 0
	for i in range(4):
		pi = pred_reg[:,:,i].view(batch_size, num_dbx, 1)
		pi = pi.expand_as(glist[i])
		total_loc_loss += reg_crit(pi.double() * match_inds.double(), glist[i] * match_inds.double())
	
	cl_crit = nn.CrossEntropyLoss(reduction='none')

	pred_cl = torch.transpose(pred[0],2,1)
	pred_cl = pred_cl.unsqueeze(2)
	pred_cl = pred_cl.expand(batch_size, pred_cl.size(1), num_ann, num_dbx)
	ann_cl = ann_cl.view(batch_size, num_ann).unsqueeze(2).expand(batch_size, num_ann, num_dbx)

	ann_cl = ann_cl * torch.transpose(match_inds,2,1).double()
	ann_cl[ann_cl == 0] = pred_cl.size(1) - 1

	cl_losses = cl_crit(pred_cl, ann_cl.long())

	cl_loss_pos = cl_losses[torch.transpose(match_inds,2,1).byte()]
	cl_loss_neg = cl_losses *  torch.abs(1 - torch.transpose(match_inds,2,1).float())
	cl_loss_neg = torch.sort(cl_loss_neg.view(batch_size, -1), descending=True)[0][:,:neg_N]

	cl_losses = torch.cat([cl_loss_pos.unsqueeze(0), cl_loss_neg], 1)

	print(torch.sum(cl_losses))
	print(torch.sum(total_loc_loss.float()))
	print(torch.sum(cl_losses) + torch.sum(total_loc_loss.float()))
	print(1/N.item())
	
	total_loss = 1/N.item() * (torch.sum(cl_losses) + torch.sum(total_loc_loss.float()))

def main():

	trainset = LocData('../data/annotations/instances_train2017.json', '../data/train2017', 'COCO')
	trainloader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_fn_cust)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = ssd(len(trainset.get_categories()))
	model = model.to(device)

	default_boxes = model._get_pboxes()
	default_boxes = default_boxes.to(device)

	for i, data in enumerate(trainloader):
		img, anns_gt, lens = data

		img = img.to(device)
		anns_gt = anns_gt.to(device)
		lens = lens.to(device)

		pred = model(img)

		gen_loss(default_boxes, anns_gt, pred, lens, device)
		break
if __name__ == '__main__':
	main()