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


def loadPascalData(data_dir='../data/', download_data=False, batch_size=32):

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transformations_valid = transformations

    dataset_train_orig = PascalVOC_Dataset(data_dir,
                                           year='2012',
                                           image_set='train',
                                           download=download_data,
                                           transform=transformations,
                                           target_transform=encode_labels)

    dataset_train = torch.utils.data.Subset(dataset_train_orig, list(range(0,50))) 
    unsup_train = torch.utils.data.Subset(dataset_train_orig, list(range(500,1500))) #len(dataset_train_orig))))
    
    dataset_valid = PascalVOC_Dataset(data_dir,
                                      year='2012',
                                      image_set='val',
                                      download=download_data,
                                      transform=transformations_valid,
                                      target_transform=encode_labels)

    dataset_valid = torch.utils.data.Subset(dataset_valid, list(range(0, 500)))
    dataset_test = torch.utils.data.Subset(dataset_valid, list(range(500,len(dataset_valid))))

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    unsup_loader = DataLoader(
        unsup_train, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=4)

    return train_loader, unsup_loader, valid_loader, test_loader
