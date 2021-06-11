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

    learning_rate = 0.000001
    

    model = models.resnet50(pretrained = True)
    model.fc = nn.Linear(int(model.fc.in_features), 10)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    all_checkpoints = os.listdir('saved_checkpoints')
    epoch = 0
    
    if sys.argv[1] == 'loadCheckpoint':
        whichCheckpoint = int(input('Which Checkpoint to Load: \n'))
        if len(all_checkpoints) > 0:
            
            PATH = 'saved_checkpoints/' + all_checkpoints[whichCheckpoint]
            print('Loading Saved Model', PATH)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            # loss = checkpoint['loss']
        
    target_layer = model.layer4[-1] ##this is the layer before the pooling


    model.conv1.padding_mode = 'reflect'
    for x in model.layer1:
        x.conv2.padding_mode = 'reflect'
    for x in model.layer2:
        x.conv2.padding_mode = 'reflect'
    for x in model.layer3:
        x.conv2.padding_mode = 'reflect'
    for x in model.layer4:
        x.conv2.padding_mode = 'reflect'


    use_cuda = torch.cuda.is_available()
    # load a few images from CIFAR and save
    if sys.argv[2] == 'visualLoss':
        from model.loss import CAMLoss
        CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
        dataiter = iter(testloader)

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.eval()
        import json
        # f = open("imagenet_class_index.json",)
        # class_idx = json.load(f)
        # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        idx2label = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

        for i in range(3):
            images, labels = dataiter.next()
            images = images.cuda()
            labels = labels.cuda()
            # images.to("cpu")
            # model.to(device)
            with torch.no_grad():
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.tolist()
                # print(predicted)
                predictedNames = [idx2label[p] for p in predicted]
                # print(predictedNames)
            imgTitle = "epoch_" + str(epoch) + "_batchNum_" + str(i)
            visualizeLossPerformance(CAMLossInstance, images, labels=labels, imgTitle=imgTitle, imgLabels=predictedNames)
        
    # visualizeImageBatch(images, labels)

    
    target_category = None
    
    #need to set params?
    
    
    numEpochs = 5
    # model.fc = nn.Linear(int(model.fc.in_features), 10)
    
    print("done")

    whichTraining = sys.argv[5]
    if whichTraining not in ['supervised', 'unsupervised', 'alternating']:
        print('invalid Training. will alternate')
        whichTraining = 'alternating'
    if sys.argv[3] == 'train':
        trackLoss = sys.argv[4] == 'trackLoss'
        print(trackLoss)
        train(model, numEpochs, trainloader, testloader, optimizer, target_layer, target_category, use_cuda, trackLoss=trackLoss, training=whichTraining)
    
    
    
    

