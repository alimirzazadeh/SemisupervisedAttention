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

from metrics.UnsupervisedMetrics import visualizeLossPerformance

def train(model, numEpochs, trainloader, testloader, optimizer, target_layer, target_category, use_cuda, trackLoss=False):
    CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
            
        running_loss = 0.0
        
        model.train()
        print("Total Dataset: ", len(trainloader.dataset))
        if epoch % 50 == 49:
            saveCheckpoint(epoch, l1.mean(), model, optimizer)
            print("saved checkpoint successfully")
        
        counter = 0
        
        if trackLoss:
            df = pd.DataFrame({'loss': [],'img': []})
            df.to_csv('saved_figs/track_loss.csv', header=False)
        
        for i, data in enumerate(trainloader, 0):
            
            if trackLoss and counter % 10 == 0:
                dataiter = iter(testloader)
                images, labels = dataiter.next()
                thisLoss, thisFig = visualizeLossPerformance(model, target_layer, images, saveFig=False)
                df = pd.DataFrame({'loss': thisLoss,'img': thisFig})
                df.to_csv('saved_figs/track_loss.csv', mode='a', header=False)
                
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # l1 = CAMLoss()(2,1,1)
                l1 = CAMLossInstance(inputs, target_category)
                # l1 = calculateLoss
                l1.backward()
                optimizer.step()
    
            running_loss += l1.item()
            # print(l1.item(), running_loss)
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            
            counter += 1

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
