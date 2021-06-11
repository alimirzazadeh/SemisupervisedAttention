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

def train(model, numEpochs, trainloader, testloader, optimizer, target_layer, target_category, use_cuda, trackLoss=False, training='alternating'):
    CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    criteron = torch.nn.CrossEntropyLoss()

    if trackLoss:
        imgPath = 'saved_figs/track_lossImg.npy'
        lossPath = 'saved_figs/track_lossNum.npy'
        np.save(imgPath, np.zeros((1,512,1024)))
        np.save(lossPath, np.zeros((1,4)))
        allLossNum = np.load(lossPath)
        allLossImg = np.load(imgPath)
    
    print('evaluating')
    # evaluateModelPerformance(model, testloader, criteron, device, optimizer)
    print('finished evaluating')

    for epoch in range(numEpochs):
        
        
        
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        
        running_corrects = 0
        running_loss = 0.0
        
        model.train()
        
        datasetSize = len(trainloader.dataset)
        print("Total Dataset: ", datasetSize)
        
        if epoch % 2 == 1:
            saveCheckpoint(epoch, model, optimizer)
            print("saved checkpoint successfully")
        
        counter = 0

        if trackLoss:
            np.save(imgPath, allLossImg)
            np.save(lossPath, allLossNum)
        
        if training == 'supervised':
            supervised = True
            alternating = False
        elif training == 'unsupervised':
            supervised = False
            alternating = False
        elif training == 'alternating':
            alternating = True

        for i, data in enumerate(trainloader, 0):

            if alternating:
                if i % 2 == 0:
                    supervised = True
                else:
                    supervised = False

            if trackLoss and counter % 100 == 0:
                dataiter = iter(testloader)
                images, labels = dataiter.next()
                CAMLossInstance.cam_model.activations_and_grads.register_hooks()
                thisLoss, thisFig = visualizeLossPerformance(CAMLossInstance, images,use_cuda=use_cuda, saveFig=False)

                allLossImg = np.append(allLossImg,np.expand_dims(thisFig,0).astype(int), axis=0)
                allLossNum = np.append(allLossNum,np.expand_dims(thisLoss,0), axis=0)

                print('saved')

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):

                ####FOR SUPERVISED OR UNSUPERVISED
                if supervised:
                    # print('supervised')
                    CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs) 
                    l1 = criteron(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                else:
                    # print('unsupervised')
                    CAMLossInstance.cam_model.activations_and_grads.register_hooks()
                    l1 = CAMLossInstance(inputs, target_category)


                l1.backward()
                optimizer.step()
                
                running_loss += l1.item()


            
            if i % 200 == 199:    # print every 200 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                      # (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                # print('[%d, %5d] accuracy: %.3f' %
                      # (epoch + 1, i + 1, running_corrects / 200))
                running_corrects = 0
            
            counter += 1
            
        CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
        evaluateModelPerformance(model, testloader, criteron, device, optimizer)

def saveCheckpoint(EPOCH, net, optimizer):
    PATH = "saved_checkpoints/model_"+str(EPOCH)+".pt"
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, PATH)
def loadCheckpoint(model, optimizer, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
