# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 11:04:18 2021

@author: alimi
"""

import torch
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
#from skimage import io, transform
#PascalDataset("C:/Users/alimi/Documents/Stanford/pascal-context/JPEGImages/","C:/Users/alimi/Documents/Stanford/pascal-context/33_context_labels/33_context_labels/")

class PascalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, npy_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # np.loadtxt(label_file)
        # self.landmarks_frame = pd.read_csv(csv_file)
        # self.label_dir = csv_file
        self.all_imageNames = os.listdir(root_dir)
        self.all_labels = np.load(npy_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.all_imageNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.all_imageNames[idx])
        image = Image.open(img_name)
        if self.transform:
            # print('yes')
            image = self.transform(image)
        labelLogit = self.all_labels[idx,:]
        sample = image, labelLogit
        return sample


    # supervisedRatio = 0.3
    # supervisedTrainSet = torch.utils.data.Subset(trainset, list(range(int(len(trainset)*supervisedRatio))))
    # unsupervisedTrainSet = torch.utils.data.Subset(trainset, list(range(int(len(trainset)*supervisedRatio),50000)))

    # suptrainloader = torch.utils.data.DataLoader(supervisedTrainSet, batch_size=batch_size,
    #                                           shuffle=shuffle, num_workers=num_workers)

    # unsuptrainloader = torch.utils.data.DataLoader(unsupervisedTrainSet, batch_size=batch_size,
    #                                           shuffle=shuffle, num_workers=num_workers)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=num_workers)
    # return suptrainloader, unsuptrainloader, testloader

# def loadPascalData(batch_size=4, num_workers=2,shuffle=True):
#     transform = Compose([
#         Resize(256),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
    
# def balancedMiniDataset(trainset, size):
#     counter = np.zeros(10)
#     subsetToInclude = []
#     iterating = True
#     step = 0
#     while iterating:
#         label = trainset[step][1]
#         if counter[label] < size:
#             subsetToInclude.append(step)
#             counter[label] += 1
        
#         if np.sum(counter) == 100:
#             iterating = False
#         elif np.sum(counter) > 100:
#             print("error, too many data")
#         step += 1
#     return torch.utils.data.Subset(trainset, subsetToInclude)

