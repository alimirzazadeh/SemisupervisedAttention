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
from model.loss import calculateLoss

def train(model, numEpochs, trainloader, optimizer, target_layer, target_category, use_cuda):
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
            
        running_loss = 0.0
        
        model.train()
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs = inputs.to(device)
            # labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                l1, l2 = calculateLoss(inputs, model, target_layer, target_category, use_cuda=use_cuda, visualize=False)
                l1.mean().backward()
                optimizer.step()
    
            running_loss += l1.mean().item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if epoch % 2 == 1:
            saveCheckpoint(epoch, l1.mean(), model, optimizer)
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