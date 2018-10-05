import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from lxml import etree
import json
import numpy as np
import cv2

from pycocotools.coco import COCO


class LocData(Dataset):

    def __init__(self, ann_path, img_path, data_type, name_path=None,):

        self.data_type = data_type
        self.data = []
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
            self.imgs = self.coco.getImgIds()
            self.img_path = img_path
        elif self.data_type == 'YOLO':
            pass

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

                ann_repr.append([cl, x1, y1, w, h])
            img = cv2.imread(curr_img)
        # TODO: Use the COCO API to get these
        elif self.data_type == 'COCO':
            curr_img_id = self.imgs[index]
            ann_ids = self.coco.getAnnIds(imgIds=curr_img_id)
            anns = self.coco.loadAnns(ann_ids)
            img_f = self.coco.loadImgs(curr_img_id)

            assert len(img_f) == 1

            img_f = img_f[0]['file_name']
            for ann in anns:
                cl = ann['category_id']
                bbx = ann['bbox'] # Check what format this is in
                
                ann_repr.append([cl] + bbx)

            img = cv2.imread(os.path.join(self.img_path, img_f))
            
        elif self.data_type == 'YOLO': 
            pass
        
        return (ann_repr, img)

# traindata = LocData('/home/shared/workspace/coco_full/annotations/instances_train2017.json', '/home/shared/workspace/coco_full/train2017', 'COCO')
# ind = np.random.randint(len(traindata))
# annotations, image = traindata[ind]

# cat_dict = {}
# for cat in traindata.coco.dataset['categories']:
#     cat_dict[int(cat['id'])] = cat['name']

# for ann in annotations:
#     point1 = (int(ann[1]), int(ann[2]))
#     point2 = (int(ann[1]) + int(ann[3]), int(ann[2]) + int(ann[4]))
#     cv2.rectangle(image, point1, point2, (0,255,0), 4)
#     cv2.putText(image, cat_dict[int(ann[0])], (point1[0],  point1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# cv2.imwrite('img_w_anns.png', image)

