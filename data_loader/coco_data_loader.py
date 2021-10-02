import torch
import torchvision.transforms as transforms
# import torchvision.datasets.CocoDetection as CocoDetection
import torch.utils.data as data
import os
import os.path
from os import path
import pickle
import numpy as np
import nltk
# from data_loader.objcat import obj_cat
from PIL import Image
from pycocotools.coco import COCO


class CocoData(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        # print("")
        # print(coco_annotation)

        if coco_annotation:
            index = coco_annotation[0]['category_id']
        # print(index)
        if index > 90:
            index = 90
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = self.transforms(image)
        num_objs = 90
        labels = torch.zeros((num_objs,), dtype=torch.int64)
        labels[index-1] = 1
        img_id = torch.tensor([img_id])
        return image, labels

    def __len__(self):
        return len(self.ids)


def loadCocoData(numImagesPerClass, batch_size=4, unsup_batch_size=12, fullyBalanced=True, useNewUnsupervised=True, unsupDatasetSize=None, shuffle=False, num_workers=4):

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        # transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transformations_valid = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Create datasets
    dataset_train_orig = CocoData(root="./data_dir/coco/images/train2017",
                                  annotation="./data_dir/coco/annotations/instances_train2017.json",
                                  transforms=transformations)
    print(dataset_train_orig[5])

    dataset_val_orig = CocoData(root="./data_dir/coco/images/val2017",
                                annotation="./data_dir/coco/annotations/instances_val2017.json",
                                transforms=transformations_valid)

    dataset_train, large_unsup = balancedMiniDataset(
        dataset_train_orig, numImagesPerClass, len(dataset_train_orig), fullyBalanced=fullyBalanced)

    # New unsupervised
    if not useNewUnsupervised:
        unsup_train = dataset_train
    else:
        unsup_train = large_unsup

    # Unsup dataset size
    if unsupDatasetSize is not None:
        unsup_train = torch.utils.data.Subset(
            unsup_train, list(range(unsupDatasetSize)))

    # Validation dataset
    dataset_valid_new = torch.utils.data.Subset(
        dataset_val_orig, list(range(0, 500)))

    # Data Loader
    train_loader = data.DataLoader(
        dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    unsup_loader = data.DataLoader(
        unsup_train, batch_size=unsup_batch_size, num_workers=4, shuffle=True)
    valid_loader = data.DataLoader(
        dataset_valid_new, batch_size=batch_size, num_workers=4)

    return train_loader, unsup_loader, valid_loader


def balancedMiniDataset(trainset, size, limit, fullyBalanced=True):
    counter = np.zeros(len(trainset[0][1]))
    iterating = True
    step = 500
    subsetToInclude = []
    subsetToNotInclude = []
    #subsetToNotInclude += list(range(step))
    while iterating and step < limit:
        label = trainset[step][1].numpy()
        if np.all(counter + label <= size) and (not fullyBalanced or np.sum(label).item() == 1):
            counter += label
            print(counter, step)
            subsetToInclude.append(step)
        else:
            subsetToNotInclude.append(step)
        if np.sum(counter) >= size * len(trainset[0][1]):
            print("Completely Balanced Dataset")
            iterating = False
        step += 1
    subsetToNotInclude += list(range(step, len(trainset)))
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude)
