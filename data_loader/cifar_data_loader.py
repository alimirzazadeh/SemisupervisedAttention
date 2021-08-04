# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:29:04 2021

@author: alimi
"""
import torch
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import numpy as np

def loadCifarData(batch_size=4, num_workers=2,shuffle=True):
    # transform = transforms.ToTensor()
    transform = Compose([
        Resize(256),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)

    print(int(len(trainset)))
    supervisedTrainSet = trainset
    unsupervisedTrainSet  = torch.utils.data.Subset(trainset, list(range( 250 )))

    suptrainloader = torch.utils.data.DataLoader(supervisedTrainSet, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)

    unsuptrainloader = torch.utils.data.DataLoader(unsupervisedTrainSet, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return suptrainloader, unsuptrainloader, testloader

def balancedMiniDataset(trainset, size):
    counter = np.zeros(10)
    subsetToInclude = []
    iterating = True
    step = 0
    while iterating:
        label = trainset[step][1]
        if counter[label] < size:
            subsetToInclude.append(step)
            counter[label] += 1
        
        if np.sum(counter) == 100:
            iterating = False
        elif np.sum(counter) > 100:
            print("error, too many data")
        step += 1
    return torch.utils.data.Subset(trainset, subsetToInclude)
    
def getLabelWord(arr):
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    arrList = arr.cpu().numpy().astype(int)
    arrList = list(arrList)
    return [classes[a] for a in arrList]