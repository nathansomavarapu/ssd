import torch
import torch.nn as nn
import torch.functional as F

import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData, collate_fn_cust
from ssd import ssd

import torch.optim as optim

import cv2
import numpy as np

import os

# TODO: Change this to perform tensor operations.
# This is going to need to be done one image at a time, batches wont work, uneven num_objs
def gen_loss(def_bxs, ann_bxs, pred, device, num_cats, thresh=0.5):

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

	diff_c = (g[:,:2] - def_bxs.float()[:,:2])/(def_bxs.float()[:,2:] * 0.1)
	diff_wh = torch.log(g[:,2:]/def_bxs[:,2:])/(0.2)
	gt = torch.cat([diff_c, diff_wh], 1)

	loc_criterion = nn.SmoothL1Loss(reduce=False)
	loc_loss = loc_criterion(pred_reg[match_inds], gt[match_inds]).sum().sum()

	cl_criterion = nn.CrossEntropyLoss(reduce=False)
	cl = torch.zeros((num_dbx), dtype=torch.long).to(device)
	cl[match_inds] = ann_cl[max_def_inds[match_inds]].long()
	
	cl_loss = cl_criterion(pred_cl, cl)
	cl_pos = cl_loss[cl != 0]
	cl_neg = cl_loss[cl == 0]

	cl_neg = torch.topk(cl_neg, N.item() * 3)[0]
	cl_loss = torch.cat([cl_pos, cl_neg], 0).sum(0)

	loc_loss = loc_loss/N.item()
	cl_loss = cl_loss/N.item()
	
	return loc_loss, cl_loss, def_bxs[match_inds]

def main():
	batch_size = 2
	epochs = 200
	# trainset = LocData('../data/annotations2017/instances_train2017.json', '../data/train2017', 'COCO')
	trainset = LocData('../data/annotations2014/instances_train2014.json', '../data/train2014', 'COCO')
	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cust)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	num_cats = len(trainset.get_categories())
	model = ssd(num_cats, init_weights=True)
	if os.path.exists('ssd.pt'):
		model.load_state_dict(torch.load('ssd.pt'))
	model = model.to(device)

	default_boxes = model._get_pboxes()
	default_boxes = default_boxes.to(device)

	opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
	scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[360000, 400000, 440000], gamma=0.1)
	# opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


	for e in range(epochs):
		for k, data in enumerate(trainloader):

			scheduler.step()

			imgs, anns_gt, lens = data
			imgs = imgs.to(device)
			anns_gt = anns_gt.to(device)
			opt.zero_grad()

			preds = model(imgs)

			batch_loss = 0
			total_loc_loss = 0
			total_conf_loss = 0
			curr_mb = torch.tensor([])
			for i in range(anns_gt.size(0)):
				if anns_gt[:lens[i],:].size(0) != 0:
					ann_gt = anns_gt[i][:lens[i],:]
					pred = (preds[0][i], preds[1][i])
					loc_loss, cl_loss, match_boxes = gen_loss(default_boxes, ann_gt, pred, device, num_cats)
					total_loc_loss += loc_loss 
					total_conf_loss += cl_loss
					if i == 0:
						curr_mb = match_boxes
			
			batch_loss = (total_loc_loss + total_conf_loss) / anns_gt.size(0)
			
			batch_loss.backward()
			opt.step()
			
			if k % 100 == 0:
				
				_, all_cl = torch.max(preds[0][0], 1)
				all_offsets = preds[1][0]

				img = imgs[0].data.cpu().numpy()
				img = np.transpose(img, (1,2,0))
				img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				img = (img * 255).astype(np.uint8)
				img_pred = img.copy()
				gts = anns_gt[0][:lens[0],:]
				non_background_inds = (all_cl != 0)

				nnb = torch.sum(non_background_inds).item()
				nnb_cls = torch.unique(all_cl[non_background_inds]).cpu().numpy()

				print('')
				print('Epoch [%d/%d], Image [%d/%d]' % (e, epochs, k, len(trainloader)))
				print('Loc. Loss: %f, Conf. Loss: %f, Total Loss %f, ' % (total_loc_loss, total_conf_loss, batch_loss.item()))
				print('Number of class boxes: %d, Number of Annotations: %d, Number of Match Boxes: %d' % (nnb, gts.size(0), curr_mb.size(0)))
				print('Classes Predicted: ')
				print(nnb_cls)
				print('')

				if all_offsets.size(0) > 0 and nnb > 0:				
					bbx_centers = (all_offsets[:,:2] * default_boxes[:,2:].float() * 0.1) + default_boxes[:,:2].float()
					bbx_widths = torch.exp(all_offsets[:,2:] * 0.2) * default_boxes[:,2:].float()

					bbx_cf = torch.cat([bbx_centers, bbx_widths], 1)
					bbx_cf = bbx_cf[non_background_inds]

					pred_min = torch.clamp(bbx_cf[:,:2] - bbx_cf[:,2:]/2.0, min=0)
					pred_max = torch.clamp(bbx_cf[:,:2] + bbx_cf[:,2:]/2.0, max=1.0)

					bbxs_pred = torch.cat([pred_min, pred_max], 1)
					
					for i in range(bbxs_pred.size(0)):
						bbx = bbxs_pred[i,:]
						lp = (int(bbx[0].item() * img.shape[1]) , int(bbx[1].item() * img.shape[0]))
						rp = (int(bbx[2].item() * img.shape[1]), int(bbx[3].item() * img.shape[0]))

						cv2.rectangle(img_pred, lp, rp, (0,0,255))
				
				if curr_mb.size(0) > 0: 
					mb_min = curr_mb[:,:2] - curr_mb[:,2:]/2.0
					mb_max = curr_mb[:,:2] + curr_mb[:,2:]/2.0


					match_boxes = torch.cat([mb_min, mb_max], 1)
					for i in range(match_boxes.size(0)):
						mb = match_boxes[i,:]
						lp = (int(mb[0].item() * img.shape[1]) , int(mb[1].item() * img.shape[0]))
						rp = (int(mb[2].item() * img.shape[1]), int(mb[3].item() * img.shape[0]))

						cv2.rectangle(img, lp, rp, (255,0,0))

				if gts.size(0) > 0:
					gts_cl = gts[:,0].unsqueeze(1)
					gts_min = gts[:,1:3] - gts[:,3:]/2.0
					gts_max = gts[:,1:3] + gts[:,3:]/2.0

					gts = torch.cat([gts_cl, gts_min, gts_max], 1)
					for i in range(gts.size(0)):
						ann_box = gts[i,1:5]
						lp = (int(ann_box[0].item() * img.shape[1]) , int(ann_box[1].item() * img.shape[0]))
						rp = (int(ann_box[2].item() * img.shape[1]), int(ann_box[3].item() * img.shape[0]))
						cv2.rectangle(img, lp, rp, (0,255,0))
						cv2.rectangle(img_pred, lp, rp, (0,255,0))
				
				cv2.imwrite('anns_def.png', img)
				cv2.imwrite('anns_pred.png', img_pred)

				torch.save(model.state_dict(), 'ssd.pt')
				break
		break			

if __name__ == '__main__':
	main()