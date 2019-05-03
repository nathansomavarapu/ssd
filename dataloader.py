import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from lxml import etree
import numpy as np
from PIL import Image
import sys
import utils

sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO

class LocData(Dataset):

	def __init__(self, ann_path, img_path, data_format, name_path=None, size=(300,300), testing=False, transform=None, tensor_transforms=None):

		self.data_format = data_format
		self.data = []
		self.size = size
		self.testing = testing
		self.transform = transform
		self.tensor_transforms = tensor_transforms
		if self.data_format == 'VOC':
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

			# self.data = self.data[0:1]
			self.nametoint = {}
			self.nametoint['None'] = 0
			self.inttoname = []
			self.inttoname.append('None')
			nf = open(name_path, 'r')
			for i, name in enumerate(nf.readlines()):
				self.nametoint[name.strip()] = int(i) + 1
				self.inttoname.append(name.strip())
			nf.close()
			
		elif self.data_format == 'COCO':
			self.coco=COCO(ann_path)
			self.imgs = sorted(self.coco.getImgIds())
			# self.imgs = sorted(self.coco.getImgIds())[3:4]
			self.img_path = img_path
			
			self.cat_renum_dict = {}
			self.inttoname = []
			self.inttoname.append('None')

			for i, cat_dict in enumerate(self.coco.dataset['categories']):
				self.inttoname.append(cat_dict['name'])
				self.cat_renum_dict[cat_dict['id']] = i + 1

		else:
			raise NotImplementedError()

	def __len__(self):
		if self.data_format == 'VOC':
			return len(self.data)
		elif self.data_format == 'COCO':
			return len(self.imgs)
	
	def __getitem__(self, index):
		ann_repr = []
		img = None
		# Approx VOC format
		if self.data_format == 'VOC':
			curr_ann, curr_img = self.data[index]
			xml_f = open(curr_ann, 'r')
			root = etree.XML(xml_f.read())
			xml_f.close()

			objs = root.findall('object')
			for obj in objs:
				cl = self.nametoint[obj.find('name').text.strip()]
				_bbx = obj.find('bndbox')
				x1 = int(float(_bbx.find('xmin').text))
				x2 = int(float(_bbx.find('xmax').text))
				y1 = int(float(_bbx.find('ymin').text))
				y2 = int(float(_bbx.find('ymax').text))

				w = x2 - x1
				h = y2 - y1

				assert w > 0 and h > 0

				cx = x1 + w/2.0
				cy = y1 + h/2.0
				
				ann_repr.append([cl, cx, cy, w, h])
			img = Image.open(curr_img)
		elif self.data_format == 'COCO':
			curr_img_id = self.imgs[index]
			ann_ids = self.coco.getAnnIds(imgIds=curr_img_id)
			anns = self.coco.loadAnns(ann_ids)
			img_f = self.coco.loadImgs(curr_img_id)

			assert len(img_f) == 1

			img_f = img_f[0]['file_name']
			for ann in anns:
				cl = ann['category_id']
				_bbx = ann['bbox']

				x1, y1, w, h = tuple(_bbx)

				_bbx = [x1 + w/2.0, y1 + h/2.0, w, h]
				
				ann_repr.append([self.cat_renum_dict[cl]] + _bbx)

			img = Image.open(os.path.join(self.img_path, img_f))
			
		elif self.data_format == 'YOLO': 
			raise NotImplementedError()
		
		img = img.convert('RGB')
		
		if self.transform is not None:
			img, ann_repr = self.transform((img, ann_repr))
		
		img, (x_pad, y_pad), (ratio_x, ratio_y) = utils.convert_pil_tensor(img, self.size, pad=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

		ann_repr = torch.tensor(ann_repr, dtype=torch.float)
		return (img, ann_repr)
	
	def get_categories(self):
		return self.inttoname

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