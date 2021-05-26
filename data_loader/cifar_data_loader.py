# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:29:04 2021

@author: alimi
"""
import torch
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

def loadCifarData(batch_size=4, num_workers=2,shuffle=True):
    # transform = transforms.ToTensor()
    transform = Compose([
        Resize(256),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return trainloader, testloader

def getLabelWord(arr):
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    arrList = arr.numpy().astype(int)
    arrList = list(arrList)
    return [classes[a] for a in arrList]