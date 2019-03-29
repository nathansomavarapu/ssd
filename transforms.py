import torchvision
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF

import numpy as np

class PhotometricDistortions():
    
    def __init__(self, p=0.2):
        self.p = p
        self.jitter = ColorJitter(brightness=0.45, saturation=0.35, hue=0.2)
    
    def __call__(self, sample):

        img, anns = sample
        if np.random.rand() <= self.p:
            img = self.jitter(img)

        return (img, anns)

class Flips():

    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, sample):
        img, anns = sample

        w, h = img.size

        if np.random.rand() <= self.p:
            c = np.random.randint(0, high=2)
            if c == 0:
                img = TF.hflip(img)
                for ann in anns:
                    x, _, _, _ = ann[1:5]

                    x_new = w - x
                    ann[1] = x_new
            else:
                img = TF.vflip(img)
                for ann in anns:
                    _, y, _, _ = ann[1:5]

                    y_new = h - y
                    ann[2] = y_new

        return (img, anns)

class SamplePatch():
    
    def __init__(self, p=0.1, crop_range=(0.5, 1.0)):
        self.p = p

        assert crop_range[0] < crop_range[1]
        self.crop_range = crop_range

    
    # NOTE: Taken from PyTorch transforms
    def get_params(self, img, output_size):

        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, high=h - th)
        j = np.random.randint(0, high=w - tw)
        return i, j, th, tw

    
    def __call__(self, sample):

        img, anns = sample

        wo,ho = img.size

        if np.random.rand() <= self.p:
            crop_percentage_w = np.random.rand() * (self.crop_range[1] - self.crop_range[0] - 0.001) + self.crop_range[0]
            ar = [1/2.0, 1, 2][np.random.randint(0,3)]

            crop_percentage_h = min(ar * crop_percentage_w, 0.99)

            crop_val_w = int(crop_percentage_w * wo)
            crop_val_h = int(crop_percentage_h * ho)

            i,j,h,w = self.get_params(img, (crop_val_h, crop_val_w))

            img = TF.crop(img, i, j, h, w)
            
            del_anns = []
            for ann in anns:
                x,y,wa,ha = ann[1:5]

                if y >= i and x >= j and x < j + w and y < i + h:
                    x = x - j
                    y = y - i

                    wh = min(x + wa/2.0, w)
                    wl = max(0, x - wa/2.0)

                    hh = min(y + ha/2.0, h)
                    hl = max(0, y - ha/2.0)

                    wa =  int(wh - wl)
                    ha = int(hh - hl)

                    x = int(wl + wa/2)
                    y = int(hl + ha/2)

                    ann[1:5] = (x,y,wa,ha)
                else:
                    del_anns.append(ann)


            for ann in del_anns:
                anns.remove(ann)
        
        return (img, anns)