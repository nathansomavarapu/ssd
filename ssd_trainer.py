import torch
import torch.nn as nn
import torch.functional as F

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

import torch.optim as optim

import cv2

# TODO: Change this to perform tensor operations.
# This is going to need to be done one image at a time, batches wont work, uneven num_objs
def gen_loss(def_bxs, ann_bxs, pred, lens, device, num_cats, thresh=0.5):

	pred_cl = pred[0]
	pred_reg = pred[1]
	ann_cl = ann_bxs[:,0]
	ann_cords = ann_bxs[:,1:5]

	num_dbx = def_bxs.size(0)
	num_ann = ann_cords.size(0)

	def_mins = def_bxs[:,:2] - def_bxs[:,2:]/2.0
	def_maxs = def_bxs[:,:2] + def_bxs[:,2:]/2.0

	def_pts = torch.cat([def_mins, def_maxs], 1)

	ann_mins = ann_cords[:,:2] - ann_cords[:,2:]/2.0
	ann_maxs = ann_cords[:,:2] + ann_cords[:,2:]/2.0

	ann_pts = torch.cat([ann_mins, ann_maxs], 1)

	def_pts = def_pts.unsqueeze(0)
	ann_pts = ann_pts.unsqueeze(1)

	def_pts = def_pts.expand(num_ann, -1, -1)
	ann_pts = ann_pts.expand_as(def_pts)
	
	mins = torch.min(def_pts[:,:,2:], ann_pts[:,:,2:])
	maxs = torch.max(def_pts[:,:,:2], ann_pts[:,:,:2])

	diff = torch.clamp(mins - maxs, min=0)
	intersect = diff[:,:,0] * diff[:,:,1]

	def_diff = def_pts[:,:,2:] - def_pts[:,:,:2]
	def_area = def_diff[:,:,0] * def_diff[:,:,1]

	ann_diff = ann_pts[:,:,2:] - ann_pts[:,:,:2]
	ann_area = ann_diff[:,:,0] * ann_diff[:,:,1]

	ious = intersect / (def_area + ann_area - intersect)

	max_defs, max_def_inds = torch.max(ious, 0)
	max_anns, max_ann_inds = torch.max(ious, 1)
	cl_nums = torch.arange(max_anns.size(0), dtype=torch.long).to(device)

	g = torch.ones((num_dbx, 4), dtype=torch.float).to(device)

	max_def_inds[max_ann_inds] = cl_nums

	match_inds = max_defs >= thresh
	match_inds[max_ann_inds] = 1

	g[match_inds] = ann_cords[max_def_inds[match_inds]].float()

	N = torch.sum(match_inds)

	diff_c = (g[:,:2] - def_bxs.float()[:,:2])/def_bxs.float()[:,2:]
	diff_wh = torch.log(g[:,2:]/def_bxs.float()[:,2:])
	gt = torch.cat([diff_c, diff_wh], 1)

	loc_criterion = nn.SmoothL1Loss(reduce=False)
	loc_loss = loc_criterion(pred_reg[match_inds], gt[match_inds])
	loc_loss = loc_loss.sum(1)
	loc_loss = loc_loss.sum(0)

	cl_criterion = nn.CrossEntropyLoss(reduce=False)
	cl = torch.ones((num_dbx), dtype=torch.long).to(device) * num_cats
	cl[match_inds] = ann_cl[max_def_inds[match_inds]].long()

	cl_loss = cl_criterion(pred_cl, cl)
	cl_pos = cl_loss[cl != num_cats]
	cl_neg = cl_loss[cl == num_cats]
	cl_neg = torch.topk(cl_neg, N.item() * 3)[0]
	cl_loss = torch.cat([cl_pos, cl_neg], 0).sum(0)
	
	total_loss = loc_loss + cl_loss
	
	return (total_loss)/N.item(), def_bxs[match_inds]

def main():
	batch_size = 4
	epochs = 1
	trainset = LocData('../data/annotations/instances_train2017.json', '../data/train2017', 'COCO', testing=False)
	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cust)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	num_cats = len(trainset.get_categories())
	model = ssd(num_cats)
	model = model.to(device)

	default_boxes = model._get_pboxes()
	default_boxes = default_boxes.to(device)

	opt = optim.SGD(model.parameters(), lr=0.001)

	for e in range(epochs):
		for i, data in enumerate(trainloader):
			imgs, anns_gt, lens = data

			imgs = imgs.to(device)
			anns_gt = anns_gt.to(device)
			lens = lens.to(device)

			preds = model(imgs)

			batch_loss = 0
			for i in range(anns_gt.size(0)):
				ann_gt = anns_gt[i][:lens[i]]
				pred = (preds[0][i], preds[1][i])
				loss, match_boxes = gen_loss(default_boxes, ann_gt, pred, lens, device, num_cats)
				batch_loss += loss
			
			batch_loss /= batch_size
			opt.zero_grad()
			batch_loss.backward()
			opt.step()

			# if i % 100 == 0:
			print('Epoch [%d/%d], Image [%d/%d], Total Loss %f' % (e, epochs, i, len(trainloader), batch_loss.item()))
			
			_, all_cl = torch.max(preds[0][0], 1)
			all_offsets = preds[1][0]

			print(all_cl.size(), all_offsets.size())
			img = imgs[0].data.cpu().numpy()
			gts = anns_gt[0]
			non_background_inds = (all_cl != num_cats)
							
			bbx_centers = (all_offsets[:,:2] * default_boxes[:,2:].float()) + default_boxes[:,:2].float()
			bbx_widths = torch.exp(all_offsets[:,2:]) * default_boxes[:,2:].float()

			bbx_cf = torch.cat([bbx_centers, bbx_widths], 1)
			bbx_cf = bbx_cf[non_background_inds]

			bbxs_pred = torch.cat([bbx_cf[:,:2] - bbx_cf[:,2:], bbx_cf[:,:2] + bbx_cf[:,2:]], 1)
			
			for i in range(bbxs_pred.size(0)):
				bbx = bbxs_pred[i,:]
				print(bbx)
				cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (255,0,0))
			
			mb_min = match_boxes[:,:2] - match_boxes[:,2:]
			mb_max = match_boxes[:,:2] + match_boxes[:,2:]

			match_boxes = torch.cat([mb_min, mb_max], 1)
			for i in range(match_boxes.size(0)):
				mb = match_boxes[i,:]
				cv2.rectangle(img, (mb[0], mb[1]), (mb[2], mb[3]), (0,0,255))
			
			for i in range(gts.size(0)):
				ann_box = gts[i,1:5]
				cv2.rectangle(img, (ann_box[0], ann_box[1]), (ann_box[2], ann_box[3]), (0,0,255))

if __name__ == '__main__':
	main()