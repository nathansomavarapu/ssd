import torch
from torch.autograd import Variable
import torch.optim as optim

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

    epochs = 450
    batch_size_target = 8
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    enable_viz = True
    pick_up = True

    port = 8097
    hostname = 'http://localhost'
    weights_dir = 'weights'
    final_weights_path = os.path.join(weights_dir, 'ssd_weights_voc_' + str(epochs) + '.pt')

    vis = None
    if enable_viz:
        vis = VisdomTrainer(port, hostname)
    
    transforms_train = Compose([
        PhotometricDistortions(),
        Flips(),
        SamplePatch()
    ])

    trainset = LocData('../data/annotations2014/instances_train2014.json', '../data/train2014', 'COCO', transform=transforms_train)
    # trainset = LocData('../data/VOC2007/Annotations', '../data/VOC2007/JPEGImages', 'VOC', name_path='../data/VOC2007/classes.txt', transform=transforms_train)
    # trainset = LocData('../data/FDDB_2010/Annotations', '../data/FDDB_2010/JPEGImages',
    #                    'VOC', name_path='../data/FDDB_2010/classes.txt', transform=transforms_train)
    dataloader = DataLoader(trainset, batch_size=batch_size_target,
                            shuffle=True, collate_fn=collate_fn_cust)

    model = ssd(len(trainset.get_categories()))
    if pick_up and os.path.exists(final_weights_path):
        model.load_state_dict(torch.load(final_weights_path))
    model = model.to(device)

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    sched = optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.1)

    default_boxes = utils.get_dboxes()
    default_boxes = default_boxes.to(device)

    

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
                
                annotations_classes = annotations[j][:lens[j]][:, 0] if lens[j].item() != 0 else torch.Tensor([])
                annotations_boxes = annotations[j][:lens[j]][:, 1:5] if lens[j].item() != 0 else torch.Tensor([])

                curr_cl_loss, curr_loc_loss, _mi = utils.compute_loss(
                    default_boxes, annotations_classes, annotations_boxes, current_classes, current_offsets)
                classification_loss += curr_cl_loss
                localization_loss += curr_loc_loss

                if j == 0:
                    match_idx_viz = _mi

            localization_loss = localization_loss / batch_size
            classification_loss = classification_loss / batch_size
            total_loss = localization_loss + classification_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:

                classification_loss_val = classification_loss.item()
                localization_loss_val = localization_loss.item()

                annotations_classes_viz = annotations[0][:lens[0]][:, 0] if lens[0].item() != 0 else torch.Tensor([])
                annotations_boxes_viz = annotations[0][:lens[0]][:, 1:5] if lens[0].item() != 0 else torch.Tensor([])

                predicted_classes_viz = Variable(predicted_classes[0].data)
                predicted_offsets_viz = Variable(predicted_offsets[0].data)

                img = utils.convert_to_pil(images[0])

                vis.update_viz(classification_loss_val, localization_loss_val, img, default_boxes, match_idx_viz,
                               annotations_classes_viz, annotations_boxes_viz, predicted_classes_viz, predicted_offsets_viz)

        torch.save(model.state_dict(), os.path.join(weights_dir, 'ssd_weights_' + str(e) + '.pt'))
        sched.step()


if __name__ == "__main__":
    main()
