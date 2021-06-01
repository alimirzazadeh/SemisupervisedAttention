import sys
sys.path.append("./")


import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn


import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from data_loader.cifar_data_loader import loadCifarData
from visualizer.visualizer import visualizeImageBatch, show_cam_on_image
from metrics.UnsupervisedMetrics import visualizeLossPerformance
# from model.loss import calculateLoss
from train import train

if __name__ == '__main__':
    ## Load the CIFAR Dataset
    trainloader, testloader = loadCifarData()


    CHECK_FOLDER = os.path.isdir("saved_figs")
    if not CHECK_FOLDER:
        os.makedirs("saved_figs")
        print("Made Saved_Figs folder")
    
    CHECK_FOLDER = os.path.isdir("saved_checkpoints")
    if not CHECK_FOLDER:
        os.makedirs("saved_checkpoints")
        print("Made Saved_Checkpoints folder")

    # replace the classifier layer with CAM Image Generation

    

    model = models.resnet50(pretrained = True)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    
    all_checkpoints = os.listdir('saved_checkpoints')
    epoch = 0
    
    if len(all_checkpoints) > 0:
        
        PATH = 'saved_checkpoints/' + all_checkpoints[-1]
        print('Loading Saved Model', PATH)
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    target_layer = model.layer4[-1] ##this is the layer before the pooling


    # load a few images from CIFAR and save
    dataiter = iter(testloader)
    for i in range(3):
        images, labels = dataiter.next()
        imgTitle = "epoch_" + str(epoch) + "_batchNum_" + str(i)
        visualizeLossPerformance(model, target_layer, images, imgTitle)
    
    # visualizeImageBatch(images, labels)

    use_cuda = torch.cuda.is_available()
    
    target_category = None
    
    #need to set params?
    
    
    numEpochs = 200
    
    print("done")
    
    train(model, numEpochs, trainloader, optimizer, target_layer, target_category, use_cuda)
    
    
    
    

