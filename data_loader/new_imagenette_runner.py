import torchvision.datasets as datasets
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from ipdb import set_trace as bp
import torch.utils.data as data
from PIL import Image
import os.path
import torch.nn.functional as F
import random

def loadImagenetteData(numImagesPerClass, batch_size=4, unsup_batch_size=12, fullyBalanced=True, useNewUnsupervised=True, unsupDatasetSize=None):
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

    datapath = "/scratch/groups/rubin/alimirz1/"
    dataset_train_orig = datasets.ImageFolder(datapath + 'imagenette2/train', transform=transformations)
    dataset_test = datasets.ImageFolder(datapath + 'imagenette2/val',transform=transformations_valid)

    #classes = ('tench', 'springer', 'casette_player', 'chain_saw',
    #           'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute')




    dataset_train, large_unsup = balancedMiniDataset(dataset_train_orig, numImagesPerClass, len(dataset_train_orig), fullyBalanced=fullyBalanced)
    
    unsup_train = dataset_train

    # If not None, unsupDatasetSize selects subset of the entire usable unsupervised dataset
    if unsupDatasetSize is not None:
        unsup_train = torch.utils.data.Subset(unsup_train, list(range(unsupDatasetSize)))


    #valid_dataset_split = 500
    #dataset_valid_new = torch.utils.data.Subset(dataset_valid, list(range(0, 500)))
    #dataset_test = dataset_valid_new
    #dataset_test = torch.utils.data.Subset(dataset_valid, list(range(0,len(dataset_valid))))
    #dataset_test = dataset_valid




    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    unsup_loader = DataLoader(
        unsup_train, batch_size=unsup_batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(
        large_unsup, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        dataset_test, batch_size=1, num_workers=4)

    return train_loader, unsup_loader, valid_loader, test_loader


def balancedMiniDataset(trainset, size, limit, fullyBalanced=True):
    counter = np.zeros(10)
    iterating = True
    step = 500
    subsetToInclude = []
    subsetToNotInclude = []
    #subsetToNotInclude += list(range(step))
    wholeRange = list(range(limit))

    random.Random(1234).shuffle(wholeRange)
    subsetToNotInclude = wholeRange[:500]


    while iterating:
        label = trainset[wholeRange[step]][1]
        
        if counter[label] < size:
            counter[label] += 1
            print(counter, step)
            subsetToInclude.append(step)
        # else:
        #     subsetToNotInclude.append(step)
        if np.min(counter) >= size:
            print("Completely Balanced Dataset")
            iterating = False
        if step%1000 == 0:
            print(step)
        step += 1
    # subsetToNotInclude += list(range(step, len(trainset)))
    
    
    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/2imgclassloosebalanced.out', np.array(subsetToInclude), delimiter=',')
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude) 
