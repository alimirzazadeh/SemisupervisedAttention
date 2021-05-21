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
from model.loss import calculateLoss

if __name__ == '__main__':
    ## Load the CIFAR Dataset
    trainloader, testloader = loadCifarData()


    CHECK_FOLDER = os.path.isdir("saved_figs")
    if not CHECK_FOLDER:
        os.makedirs("saved_figs")
        print("Made Saved_Figs folder")

    # replace the classifier layer with CAM Image Generation

    model = models.resnet50(pretrained = True)

    target_layer = model.layer4[-1] ##this is the layer before the pooling


    # load a few images from CIFAR and save

    dataiter = iter(trainloader)

    images, labels = dataiter.next()

    visualizeImageBatch(images, labels)

    print("Labels: ", labels)
    
    input_tensor = images

    use_cuda = torch.cuda.is_available()
    
    target_category = None
    
    l1, l2 = calculateLoss(input_tensor, model, target_layer, target_category, use_cuda=use_cuda, visualize=False)


