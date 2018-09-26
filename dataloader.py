import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from lxml import etree
import json

class cocoloader(Dataset):

    def __init__(self, ann_path, img_path, name_path):
        ann_paths = glob.glob(os.path.join(ann_path, '*'))
        imgs = glob.glob(os.path.join(img_path, '*'))

        self.data_type = 'VOC' if ann_paths[0].split('.')[-1] == 'xml' else 'COCO' if ann_paths[0].split('.')[-1] == 'json' else 'YOLO'

        self.data = []
        if self.data_type == 'VOC':
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
            pass
        elif self.data_type == 'YOLO':
            pass

    def __len__(self):
            return len(self.data)
    
    def __getitem__(self, index):
        ann_repr = []
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
        # TODO: Use the COCO API to get these
        elif self.data_type == 'COCO':
            json_f = open(curr_ann, 'r')
            ann_j = json.load(json_f)
            json_f.close()

            [json_f['annotations'][0]['id']] + json_f['annotations'][0]['bbox']




                

        
            
