# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:27:39 2021

@author: alimi
"""

# import argparse
# import collections
import torch
import numpy as np
import torchvision.models as models
from model.loss import CAMLoss
import pandas as pd

from metrics.SupervisedMetrics import evaluateModelPerformance
from metrics.UnsupervisedMetrics import visualizeLossPerformance

def train(model, numEpochs, trainloader, testloader, optimizer, target_layer, target_category, use_cuda, trackLoss=False):
    CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    
    criteron = torch.nn.CrossEntropyLoss()

    if trackLoss:
        imgPath = 'saved_figs/track_lossImg.npy'
        lossPath = 'saved_figs/track_lossNum.npy'
        np.save(imgPath, np.zeros((1,512,1024)))
        np.save(lossPath, np.zeros((1,4)))
        allLossNum = np.load(lossPath)
        allLossImg = np.load(imgPath)
    
    # evaluateModelPerformance(model, testloader, criteron)
    
    for epoch in range(numEpochs):
        
        
        
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        
        running_corrects = 0
        running_loss = 0.0
        
        model.train()
        
        datasetSize = len(trainloader.dataset)
        print("Total Dataset: ", datasetSize)
        
        if epoch % 2 == 1:
            saveCheckpoint(epoch, l1.mean(), model, optimizer)
            print("saved checkpoint successfully")
        
        counter = 0

        if trackLoss:
            np.save(imgPath, allLossImg)
            np.save(lossPath, allLossNum)
        
        for i, data in enumerate(trainloader, 0):
            
            if trackLoss and counter % 100 == 0:
                dataiter = iter(testloader)
                images, labels = dataiter.next()
                thisLoss, thisFig = visualizeLossPerformance(CAMLossInstance, images,use_cuda=use_cuda, saveFig=False)

                allLossImg = np.append(allLossImg,np.expand_dims(thisFig,0).astype(int), axis=0)
                allLossNum = np.append(allLossNum,np.expand_dims(thisLoss,0), axis=0)

                print('saved')

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs) 
                
                l1 = criteron(outputs, labels)
                # l1 = CAMLossInstance(inputs, target_category)
                l1.backward()
                optimizer.step()
            
            ## calculate running loss and running corrects
            running_loss += l1.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # print(l1.item(), running_loss)
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                print('[%d, %5d] accuracy: %.3f' %
                      (epoch + 1, i + 1, running_corrects / 200))
                running_corrects = 0
            
            counter += 1
            
        evaluateModelPerformance(model, testloader, criteron)

def saveCheckpoint(EPOCH, LOSS, net, optimizer):
    PATH = "saved_checkpoints/model_"+str(EPOCH)+".pt"
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, PATH)
def loadCheckpoint(model, optimizer, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
