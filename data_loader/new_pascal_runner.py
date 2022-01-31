# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:10:06 2021

@author: alimi
"""

import torchvision.datasets.voc as voc
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os


class PascalVOC_Dataset(voc.VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):

        super().__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return len(self.images)


def encode_labels(target):
    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

    ls = target['annotation']['object']
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))

    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))

    k = np.zeros(len(object_categories))
    k[j] = 1

    return torch.from_numpy(k)

#Parameters:
# download_data: set to True only the first time running on new system to download the Pascal VOC dataset, otherwise set to False
# batch_size: the batch size for supervised, validation, and test dataset
# unsup_batch size: the batch size for unsupervised dataset
# fullyBalanced: set to True if you want exactly numClasses * number of Instances per class images in the supervised dataset (can go up to 14 per class)
#                if set to False, because this is multilabel it will include multilabel images in the supervised dataset as well (can go up to 150 per class)
# useNewUnsupervised: if set to True, will only include images not in supervised set, if False only uses images in supervised set
# unsupDatasetSize: if not None, sets the size of the unsupervised dataset
def loadPascalData(numImagesPerClass, data_dir='../data/', download_data=False, batch_size=4, unsup_batch_size=12, fullyBalanced=True, useNewUnsupervised=True, unsupDatasetSize=None):
    resNetOrInception = 0
    if resNetOrInception == 0:
        imgSize = 256
    else:
        imgSize = 299

    transformations = transforms.Compose([
        #transforms.Resize((299,299)),
        #transforms.RandomRotation(10),
        #transforms.RandomHorizontalFlip(0.5),
        #transforms.RandomVerticalFlip(0.5),
	transforms.RandomResizedCrop(imgSize),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transformations_valid = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])


    dataset_train_orig = PascalVOC_Dataset(data_dir,
                                           year='2012',
                                           image_set='train',
                                           download=download_data,
                                           transform=transformations,
                                           target_transform=encode_labels)
    
    dataset_val_orig = PascalVOC_Dataset(data_dir,
                                           year='2012',
                                           image_set='train',
                                           download=download_data,
                                           transform=transformations_valid,
                                           target_transform=encode_labels)
 

    dataset_train, large_unsup = balancedMiniDataset(dataset_train_orig, numImagesPerClass, len(dataset_train_orig), fullyBalanced=fullyBalanced)
    

    ### useNewUnsupervised determines if the unsupservised data is the same as supervised data or the unused parts of trainset
    if not useNewUnsupervised:
        unsup_train = dataset_train
    else:
        unsup_train = large_unsup


    # If not None, unsupDatasetSize selects subset of the entire usable unsupervised dataset
    if unsupDatasetSize is not None:
        unsup_train = torch.utils.data.Subset(unsup_train, list(range(unsupDatasetSize)))



    dataset_test = PascalVOC_Dataset(data_dir,
                                      year='2012',
                                      image_set='val',
                                      download=download_data,
                                      transform=transformations_valid,
                                      target_transform=encode_labels)

    # valid_dataset_split = 500
    dataset_valid_new = torch.utils.data.Subset(dataset_train_orig, list(range(0, 500)))
    # dataset_test = torch.utils.data.Subset(dataset_valid, list(range(valid_dataset_split,len(dataset_valid))))

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    unsup_loader = DataLoader(
        unsup_train, batch_size=unsup_batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(
        dataset_valid_new, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        dataset_test, batch_size=1, num_workers=4)

    return train_loader, unsup_loader, valid_loader, test_loader


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