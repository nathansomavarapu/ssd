import torch
from torch.autograd import Variable
import torch.optim as optim

import torch.nn.init as init
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataloader import LocData
from dataloader import collate_fn_cust

import utils
from viz_training import VisdomTrainer

from ssd import ssd

from transforms import PhotometricDistortions, Flips, SamplePatch

import os


def main():

    epochs = 800
    batch_size_target = 32
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    enable_viz = True
    pick_up = True

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    port = 8097
    hostname = 'http://localhost'
    weights_dir = 'weights'
    final_weights_path = os.path.join(weights_dir, 'ssd_weights_' + str(epochs-1) + '.pt')

    vis = None
    if enable_viz:
        vis = VisdomTrainer(port, hostname)
    
    transforms_train = Compose([
        PhotometricDistortions(),
        Flips()
        # SamplePatch()
    ])

    # trainset = LocData('../data/annotations2014/instances_train2014.json', '../data/train2014', 'COCO', transform=transforms_train)
    trainset = LocData('../data/VOC/train/Annotations', '../data/VOC/train/JPEGImages', 'VOC', name_path='../data/VOC/train/classes.txt', transform=transforms_train)
    # trainset = LocData('../data/face/face_annotations', '../data/face/face_images', 'VOC', name_path='../data/face/classes.txt', transform=transforms_train)
    # trainset = LocData('../data/FDDB_2010/Annotations/', '../data/FDDB_2010/JPEGImages/', 'VOC', name_path='../data/FDDB_2010/classes.txt')
    dataloader = DataLoader(trainset, batch_size=batch_size_target,
                            shuffle=True, collate_fn=collate_fn_cust)

    print(len(trainset.get_categories()))
    print(len(trainset))
    model = ssd(len(trainset.get_categories()), bn=False, base='vgg')
    if pick_up and os.path.exists(final_weights_path):
        print('Picking up Training')
        model.load_state_dict(torch.load(final_weights_path))
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    sched = optim.lr_scheduler.MultiStepLR(optimizer, [80000, 100000, 120000], gamma=0.1)

    default_boxes = utils.get_dboxes()
    default_boxes = default_boxes.to(device)

    iter_count = 0

    for e in range(epochs):
        for i, data in enumerate(dataloader):

            model.train()

            images, annotations, lens = data
            images = images.to(device)
            annotations = annotations.to(device)
            lens = lens.to(device)

            predicted_classes, predicted_offsets = model(images)

            assert predicted_classes.size(0) == predicted_offsets.size(0)

            batch_size = predicted_classes.size(0)

            classification_loss = 0
            localization_loss = 0
            match_idx_viz = None

            for j in range(batch_size):
                current_classes = predicted_classes[j]
                current_offsets = predicted_offsets[j]

                if lens[j].item() != 0:
                    annotations_classes = annotations[j][:lens[j]][:, 0] 
                    annotations_boxes = annotations[j][:lens[j]][:, 1:5]

                    curr_cl_loss, curr_loc_loss, _mi = utils.compute_loss(
                        default_boxes, annotations_classes, annotations_boxes, current_classes, current_offsets)
                    classification_loss += curr_cl_loss
                    localization_loss += curr_loc_loss

                if j == 0 and lens[j].item() != 0:
                    match_idx_viz = _mi
                elif j == 0:
                    match_idx_viz = torch.LongTensor([])

            localization_loss = localization_loss / batch_size
            classification_loss = classification_loss / batch_size
            total_loss = localization_loss + classification_loss
            
            if type(total_loss) == torch.Tensor:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                sched.step()

                iter_count += 1

            if i % 100 == 0:

                classification_loss_val = classification_loss.item() if type(classification_loss) == torch.Tensor else classification_loss
                localization_loss_val = localization_loss.item() if type(localization_loss) == torch.Tensor else localization_loss

                annotations_classes_viz = annotations[0][:lens[0]][:, 0] if lens[0].item() != 0 else torch.Tensor([])
                annotations_boxes_viz = annotations[0][:lens[0]][:, 1:5] if lens[0].item() != 0 else torch.Tensor([])

                predicted_classes_viz = Variable(predicted_classes[0].data)
                predicted_offsets_viz = Variable(predicted_offsets[0].data)

                mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
                std = torch.as_tensor(std, dtype=torch.float32, device=device)

                images[0].mul_(std[:,None,None]).add_(mean[:,None,None])

                img = utils.convert_tens_pil(images[0])

                vis.update_viz(classification_loss_val, localization_loss_val, img, default_boxes, match_idx_viz,
                               annotations_classes_viz, annotations_boxes_viz, predicted_classes_viz, predicted_offsets_viz)
                
                print(iter_count)
        
        torch.save(model.state_dict(), os.path.join(weights_dir, 'ssd_weights_' + str(e) + '.pt'))

if __name__ == "__main__":
    main()
