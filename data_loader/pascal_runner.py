# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 22:10:36 2021

@author: alimi
"""
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
import numpy as np

from data_loader.pascal_data_loader import PascalDataset
def loadPascalData(batch_size=4, num_workers=2,shuffle=True):
    # transform = transforms.ToTensor()
    transform = Compose([
        Resize((256,256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 4
    trainset = PascalDataset("C:/Users/alimi/Documents/Stanford/pascal-context/JPEGImages/","C:/Users/alimi/Documents/Stanford/pascal-context/33_context_labels/33_context_labels/", transform=transform)
    
    suptrainset = torch.utils.data.Subset(trainset, list(range(0,100)))
    unsuptrainset = torch.utils.data.Subset(trainset, list(range(2100,len(trainset))))
    testset = torch.utils.data.Subset(trainset, list(range(100,2100)))
    
    
    
    suptrainloader = torch.utils.data.DataLoader(suptrainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    unsuptrainloader = torch.utils.data.DataLoader(unsuptrainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    
    
    return suptrainloader, unsuptrainloader, testloader
