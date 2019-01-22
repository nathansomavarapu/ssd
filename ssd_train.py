import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from dataloader import LocData
from dataloader import collate_fn_cust

import utils

from ssd import ssd


def main():

    epochs = 250
    batch_size = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainset = LocData('../data/VOC2007/Annotations', '../data/VOC2007/JPEGImages',
                       'VOC', name_path='../data/VOC2007/classes.txt')
    dataloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn_cust)

    model = ssd(len(trainset.get_categories()))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())

    default_boxes = utils.get_dboxes()
    default_boxes = default_boxes.to(device)

    for e in range(epochs):
        for i, data in enumerate(dataloader):

            images, annotations, lens = data
            images = images.to(device)
            annotations = annotations.to(device)
            lens = lens.to(device)

            predicted_classes, predicted_offsets = model(images)

            classification_loss = 0
            localization_loss = 0
            num_predictions = predicted_classes.size(0)

            for j in range(num_predictions):
                current_classes = predicted_classes[j]
                current_offsets = predicted_offsets[j]
                annotations_classes = annotations[j][:lens[j]][:, 0]
                annotations_boxes = annotations[j][:lens[j]][:, 1:5]

                curr_cl_loss, curr_loc_loss = utils.compute_loss(
                    default_boxes, annotations_classes, annotations_boxes, current_classes, current_offsets, device=device)
                classification_loss += curr_cl_loss
                localization_loss += curr_loc_loss

            classification_loss, localization_loss = utils.compute_loss(
                default_boxes, annotations, predicted_classes, predicted_offsets, device=device)

            # localization_loss = localization_loss / num_predictions
            # classification_loss = classification_loss / num_predictions
            total_loss = localization_loss + classification_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
