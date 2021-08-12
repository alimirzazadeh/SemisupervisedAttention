# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:29:04 2021

@author: alimi
"""
import torch
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import numpy as np
import random

from pdb import set_trace as bp


def loadCifarData(batch_size=4, num_workers=2,shuffle=True):
    VALSET_SIZE = 500
    # transform = transforms.ToTensor()
    transform = Compose([
        Resize(256),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    random.seed(10)
    randomlist = random.sample(range(0, len(trainset)), VALSET_SIZE)
    validset = torch.utils.data.Subset(trainset, randomlist)

    print("Finished balanced mini dataset")
    supervisedTrainSet, unsupervisedTrainSet = balancedMiniDataset(trainset, 2, int(len(trainset)))
    suptrainloader = torch.utils.data.DataLoader(supervisedTrainSet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    unsuptrainloader = torch.utils.data.DataLoader(unsupervisedTrainSet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  
    return suptrainloader, unsuptrainloader, validloader, testloader

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
        if np.sum(counter) >= size * NUM_CLASSES: ##again, change too whatever shape of trainset is 10 so its generarlizable
            print("Completely Balanced Dataset")
            iterating = False
        step += 1
    print(subsetToInclude)
    print(subsetToNotInclude)
    subsetToNotInclude += list(range(step, len(trainset)))
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude)
    
def getLabelWord(arr):
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    arrList = arr.cpu().numpy().astype(int)
    arrList = list(arrList)
    return [classes[a] for a in arrList]  