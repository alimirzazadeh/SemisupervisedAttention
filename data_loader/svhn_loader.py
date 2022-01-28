# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:29:04 2021

@author: alimi
"""
import torch
import torchvision
from torchvision import transforms
import numpy as np
import random
import os

from pdb import set_trace as bp


def loadSVHNdata(numImagesPerClass, data_dir="./data", download_data=True, batch_size=4, unsup_batch_size=12, fullyBalanced=True, useNewUnsupervised=True, unsupDatasetSize=None):
    VALSET_SIZE = 500

    # Data augmentation
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

    # Create train
    trainset = torchvision.datasets.SVHN(
        root=data_dir, split="train", download=download_data, transform=transformations)
    sup_train, unsupervisedTrainSet = balancedMiniDataset(
        trainset, numImagesPerClass, len(trainset), fullyBalanced=fullyBalanced)
    unsup_train = unsupervisedTrainSet if useNewUnsupervised else sup_train

    if unsupDatasetSize is not None:
        unsup_train = torch.utils.data.Subset(
            unsup_train, list(range(unsupDatasetSize)))

    # Create validation and test
    random.seed(10)
    dataset_test = torchvision.datasets.CIFAR10(
        root=data_dir, split="test", download=download_data, transform=transformations_valid)
    randomlist = random.sample(range(0, len(dataset_test)), VALSET_SIZE)
    dataset_valid = torch.utils.data.Subset(dataset_test, randomlist)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(
        sup_train, batch_size=batch_size, shuffle=True, num_workers=4)
    unsup_loader = torch.utils.data.DataLoader(
        unsup_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, unsup_loader, valid_loader, test_loader


def balancedMiniDataset(trainset, size, limit, fullyBalanced=True):
    NUM_CLASSES = 10
    counter = np.zeros(NUM_CLASSES)
    iterating = True
    step = 0
    subsetToInclude = []
    subsetToNotInclude = []
    while iterating and step < limit:
        label = trainset[step][1]
        if counter[label] + 1 <= size:
            counter[label] += 1
            print(counter, step)
            subsetToInclude.append(step)
        else:
            subsetToNotInclude.append(step)
        # again, change too whatever shape of trainset is 10 so its generarlizable
        if np.sum(counter) >= size * NUM_CLASSES:
            print("Completely Balanced Dataset")
            iterating = False
        step += 1
    print(subsetToInclude)
    print(subsetToNotInclude)
    subsetToNotInclude += list(range(step, len(trainset)))
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude)


def getLabelDigit(digit):
    return digit+1
