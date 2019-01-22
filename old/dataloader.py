import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from lxml import etree
import json
import numpy as np
import cv2
import sys

sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO

class LocData(Dataset):

	def __init__(self, ann_path, img_path, data_type, name_path=None, size=(300,300), testing=False):

		self.data_type = data_type
		self.data = []
		self.size = size
		self.testing = testing
		if self.data_type == 'VOC':
			assert name_path is not None

			ann_paths = glob.glob(os.path.join(ann_path, '*'))
			imgs = glob.glob(os.path.join(img_path, '*'))

			img_ext = imgs[0].split('.')[-1]
			for ann in ann_paths:
				ann_ext = ann.split('.')[-1]
				check_img = ann.split('/')[-1]
				check_img = check_img.replace(ann_ext, img_ext)
				check_img = os.path.join(img_path, check_img)

				if os.path.exists(check_img):
					self.data.append((ann, check_img))
		
			self.nametoint = {}
			nf = open(name_path, 'r')
			for i, name in enumerate(nf.readlines()):
				self.nametoint[name.strip()] = int(i)
			nf.close()
		elif self.data_type == 'COCO':
			self.coco=COCO(ann_path)
			self.imgs = sorted(self.coco.getImgIds())
			# self.imgs = sorted(self.coco.getImgIds())[3:4]
			self.img_path = img_path
			self.coco_cats = []
			self.cat_renum_dict = {}

			self.coco_cats.append('None')
			for i, cat_dict in enumerate(self.coco.dataset['categories']):
				self.coco_cats.append(cat_dict['name'])
				self.cat_renum_dict[cat_dict['id']] = i + 1


		elif self.data_type == 'YOLO':
			raise NotImplementedError()

	def __len__(self):
		if self.data_type == 'VOC':
			return len(self.data)
		elif self.data_type == 'COCO':
			return len(self.imgs)
	
	def __getitem__(self, index):
		ann_repr = []
		img = None
		# Approx VOC format
		if self.data_type == 'VOC':
			curr_ann, curr_img = self.data[index]
			xml_f = open(curr_ann, 'r')
			root = etree.XML(xml_f.read())
			xml_f.close()

			objs = root.findall('object')
			for obj in objs:
				cl = self.nametoint[obj.find('name').text]
				_bbx = obj.find('bndbox')
				x1 = int(_bbx.find('xmin'))
				x2 = int(_bbx.find('xmax'))
				y1 = int(_bbx.find('ymin'))
				y2 = int(_bbx.find('ymax'))

				w = x2 - x1
				h = y2 - y1

				assert w > 0 and h > 0

				cx = x1 + w/2.0
				cy = y1 + h/2.0

				ann_repr.append([cl, cx, cy, w, h])
			img = cv2.imread(curr_img)
		elif self.data_type == 'COCO':
			curr_img_id = self.imgs[index]
			ann_ids = self.coco.getAnnIds(imgIds=curr_img_id)
			anns = self.coco.loadAnns(ann_ids)
			img_f = self.coco.loadImgs(curr_img_id)

			assert len(img_f) == 1

			img_f = img_f[0]['file_name']
			for ann in anns:
				cl = ann['category_id']
				_bbx = ann['bbox'] # Check what format this is in

				x1, y1, w, h = tuple(_bbx)

				_bbx = [x1 + w/2.0, y1 + h/2.0, w, h]
				
				ann_repr.append([self.cat_renum_dict[cl]] + _bbx)

			img = cv2.imread(os.path.join(self.img_path, img_f))
			
		elif self.data_type == 'YOLO': 
			raise NotImplementedError()
		
		max_side = max(img.shape[:2])

		y_pad = int((max_side - img.shape[0])/2)
		x_pad = int((max_side - img.shape[1])/2)

		img = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0,0)), mode='constant', constant_values=0)
		img_pad_s = img.shape
		img = cv2.resize(img, self.size)
		img_old = img.copy()
		
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.float32)/255.0

		ratio_x = img.shape[1]/float(img_pad_s[1])
		ratio_y = img.shape[0]/float(img_pad_s[0])

		# Perform any anotation corrections
		for ann in ann_repr:
			ann[1] += x_pad
			ann[2] += y_pad

			ann[1] *= ratio_x
			ann[2] *= ratio_y
			ann[3] *= ratio_x
			ann[4] *= ratio_y

			ann[1] /= float(self.size[0])
			ann[2] /= float(self.size[1])

			ann[3] /= float(self.size[0])
			ann[4] /= float(self.size[1])

		img = torch.from_numpy(img.transpose(2,0,1)).float()
		
		if self.testing:
			img_h, img_w = img_old.shape[:2]
			for ann in ann_repr:
				point1 = (int((ann[1] - ann[3]/2.0) * img_w), int((ann[2] - ann[4]/2.0) * img_h))
				point2 = (int((ann[1] + ann[3]/2.0) * img_w), int((ann[2] + ann[4]/2.0) * img_h))
				cv2.rectangle(img_old, point1, point2, (0,255,0))
				cv2.putText(img_old, self.coco_cats[int(ann[0])], (point1[0],  point1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)

			cv2.imwrite('img_w_anns.png', img_old)

		ann_repr = torch.tensor(ann_repr, dtype=torch.float)
		return (img, ann_repr)
	
	def get_categories(self):
		if self.data_type == 'COCO':
			return self.coco_cats

def collate_fn_cust(data):
	imgs, anns = zip(*data)
	imgs = torch.stack(imgs, 0)

	ann_lens = [ann.size(0) for ann in anns]
	per_ann_lens = [cann.size(1) if cann.size(0) > 0 else 0 for cann in anns]
	per_ann_len = max(per_ann_lens)
	max_ann_len = max(ann_lens)

	anns_t = torch.zeros((len(data), max_ann_len, per_ann_len), dtype=torch.float)

	for ann_ind in range(len(anns)):
		if ann_lens[ann_ind] != 0:
			anns_t[ann_ind,:ann_lens[ann_ind],:] = anns[ann_ind]
	
	lengths = torch.from_numpy(np.array(ann_lens))

	return imgs, anns_t, lengths

