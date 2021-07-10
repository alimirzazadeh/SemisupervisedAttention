# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 22:10:36 2021

@author: alimi
"""
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import numpy as np
import os

from data_loader.pascal_data_loader import PascalDataset
def loadPascalData(batch_size=4, num_workers=2,shuffle=True):
    # transform = transforms.ToTensor()
    transform = Compose([
        Resize((256,256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    if os.path.isdir("C:\\Users\\alimi\\Documents\\Stanford\\VOC2012\\JPEGImages"):
        trainset = PascalDataset("C:\\Users\\alimi\\Documents\\Stanford\\VOC2012\\JPEGImages","C:\\Users\\alimi\\Documents\\Stanford\\VOC2012\\allLabels.npy", transform=transform)
    else:
        trainset = PascalDataset("/scratch/users/alimirz1/VOC2012/JPEGImages/","/scratch/users/alimirz1/VOC2012/allLabels.npy", transform=transform)
    
    suptrainset = torch.utils.data.Subset(trainset, list(range(0,1000)))
    # suptrainset = balancedMiniDataset(trainset, 10)
    unsuptrainset = torch.utils.data.Subset(trainset, list(range(3000,len(trainset))))
    # suptrainset = unsuptrainset
    testset = torch.utils.data.Subset(trainset, list(range(1000,3000)))
    
    
    
    suptrainloader = torch.utils.data.DataLoader(suptrainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    unsuptrainloader = torch.utils.data.DataLoader(unsuptrainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    
    
    return suptrainloader, unsuptrainloader, testloader

def balancedMiniDataset(trainset, size):
    counter = np.zeros(len(trainset[0][1]))
    iterating = True
    step = 0
    subsetToInclude = []
    while iterating and step < 1000:
        label = trainset[step][1]
        if np.all(counter + label <= size):
            counter += label
            subsetToInclude.append(step)
        
        if np.sum(counter) >= size * len(trainset[0][1]):
            iterating = False
        step += 1
    return torch.utils.data.Subset(trainset, subsetToInclude)
    
