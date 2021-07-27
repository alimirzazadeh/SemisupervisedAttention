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
import random
from torch import nn

from metrics.SupervisedMetrics import Evaluator
from metrics.UnsupervisedMetrics import visualizeLossPerformance

def customTrain(model):
    def _freeze_norm_stats(net):
        try:
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    
        except ValueError:  
            print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
            return
    model.train()
    model.apply(_freeze_norm_stats)
            

def train(model, numEpochs, suptrainloader, unsuptrainloader, testloader, optimizer, target_layer, target_category, use_cuda, resolutionMatch, similarityMetric, alpha, trackLoss=False, training='alternating', batchDirectory='', scheduler=None, batch_size=4):
    print('alpha: ', alpha)
    CAMLossInstance = CAMLoss(model, target_layer, use_cuda, resolutionMatch, similarityMetric)
    LossEvaluator = Evaluator()
    CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    
    
    # def criteron(pred_label, target_label):
    #     m = nn.Softmax(dim=1)
    #     pred_label = m(pred_label)
    #     return (-pred_label.log() * target_label).sum(dim=1).mean()
    # def criteron(pred_label, target_label):
    #     m = nn.BCEWithLogitsLoss()
    #     return m(pred_label, target_label)
    weight = torch.tensor([0.87713311, 1.05761317, 0.73638968, 1.11496746, 0.78593272, 1.33506494,
                           0.4732965, 0.514, 0.47548566, 1.9469697, 0.97348485, 0.43670348,
                           1.15765766, 1.06639004, 0.13186249, 1.05544148, 1.71906355, 1.04684318,
                           1.028, 0.93624772])
    weight = weight.to(device)
    # criteron = nn.MultiLabelSoftMarginLoss(weight=weight)
        
    
    # criteron = torch.nn.CrossEntropyLoss()
    criteron = nn.BCEWithLogitsLoss()

    if trackLoss:
        imgPath = batchDirectory + 'saved_figs/track_lossImg.npy'
        lossPath = batchDirectory + 'saved_figs/track_lossNum.npy'
        np.save(imgPath, np.zeros((1,512,1024)))
        np.save(lossPath, np.zeros((1,4)))
        allLossNum = np.load(lossPath)
        allLossImg = np.load(imgPath)
    
    print('evaluating')
    model.eval()
    LossEvaluator.evaluateUpdateLosses(model, testloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=batchDirectory) #unsupervised=training!='supervised')
    LossEvaluator.plotLosses(batchDirectory=batchDirectory)
    print('finished evaluating')
        
    supdatasetSize = len(suptrainloader.dataset)
    print("\n\nTotal Supervised Dataset: ", supdatasetSize)
    unsupdatasetSize = len(unsuptrainloader.dataset)
    print("Total Unsupervised Dataset: ", unsupdatasetSize)

    if training == 'supervised':
        totalDatasetSize = int(supdatasetSize / batch_size)
    elif training == 'unsupervised':
        totalDatasetSize = int(unsupdatasetSize / batch_size)
    elif training == 'alternating':
        totalDatasetSize = int((alpha * supdatasetSize + unsupdatasetSize) / batch_size)
        trainingRatio = alpha * (supdatasetSize / (alpha * supdatasetSize + unsupdatasetSize))
    
    print("Total Dataset: ", totalDatasetSize)
    
    customTrain(model)
    
    
    for epoch in range(numEpochs):
        
        # if scheduler:
        #     scheduler.step()        
        running_corrects = 0
        running_loss = 0.0

        supiter = iter(suptrainloader)
        unsupiter = iter(unsuptrainloader)
        supiter_reloaded = 0
        unsupiter_reloaded = 0
        

        #if epoch % 20 == 19:
        #    saveCheckpoint(epoch, model, optimizer, batchDirectory=batchDirectory)
        #    print("saved checkpoint successfully")
        
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

        # for i, data in enumerate(trainloader, 0):
        #print('starting iterations...')
        for i in range(totalDatasetSize):
            #if i % 100 == 50:
            #    print('Epoch: ', epoch, 'Batch: ', i)
            #    model.eval()
            #    optimizer.zero_grad()
            #    LossEvaluator.evaluateUpdateLosses(model, testloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True) #training!='supervised')
            #    LossEvaluator.plotLosses(batchDirectory=batchDirectory)
            
            
            
            if alternating:
                # if i % 2 == 0:
                if random.random() <= trainingRatio:
                    try:
                        data = supiter.next()
                        supervised = True
                        # print(str(i),' s')
                    except StopIteration:
                        supiter = iter(suptrainloader)
                        supiter_reloaded += 1
                        data = supiter.next()
                        supervised = True  
                        # print(str(i),' -s')                  
                else:
                    try:
                        data = unsupiter.next()
                        supervised = False
                        # print(str(i),' u')
                    except StopIteration:
                        unsupiter = iter(unsuptrainloader)
                        unsupiter_reloaded += 1
                        data = unsupiter.next()
                        supervised = False
                        # print(str(i),' -u')
            elif supervised:
                data = supiter.next()
                #print('s')
            else:
                data = unsupiter.next()
                #print('u')

            if trackLoss and counter % 100 == 0:
                dataiter = iter(testloader)
                images, labels = dataiter.next()
                CAMLossInstance.cam_model.activations_and_grads.register_hooks()
                thisLoss, thisFig = visualizeLossPerformance(CAMLossInstance, images,use_cuda=use_cuda, saveFig=False, batchDirectory=batchDirectory)

                allLossImg = np.append(allLossImg,np.expand_dims(thisFig,0).astype(int), axis=0)
                allLossNum = np.append(allLossNum,np.expand_dims(thisLoss,0), axis=0)

                print('saved')

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            
            with torch.set_grad_enabled(True):

                ####FOR SUPERVISED OR UNSUPERVISED
                if supervised:
                    model.train()
                    optimizer.zero_grad()
                    # print('supervised')
                    CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # print(labelLogit)
                    outputs = model(inputs) 
                    l1 = criteron(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    # print(labels)
                    for pred in range(preds.shape[0]):
                        running_corrects += labels[pred, int(preds[pred])]
                    # running_corrects += torch.sum(preds == labels.data)
                else:
                    customTrain(model)
                    optimizer.zero_grad()
                    # print('unsupervised')
                    CAMLossInstance.cam_model.activations_and_grads.register_hooks()
                    l1 = CAMLossInstance(inputs, target_category)
                    
                optimizer.zero_grad()
                l1.backward()
                optimizer.step()
                
                # running_loss += l1.item()

            #if i % 200 == 199:    # print every 200 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch + 1, i + 1, running_loss / 200))
            #    running_loss = 0.0
            #    print('[%d, %5d] accuracy: %.3f' %
            #          (epoch + 1, i + 1, running_corrects / 200))
            #    running_corrects = 0            
            counter += 1
            
        # CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
        if True: #epoch % 10 == 5:
            print('Epoch {}/{}'.format(epoch, numEpochs - 1))
            model.eval()
            optimizer.zero_grad()
            LossEvaluator.evaluateUpdateLosses(model, testloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=batchDirectory) #training!='supervised')
            LossEvaluator.plotLosses(batchDirectory=batchDirectory)
            print('Unsup Iter Reloaded: ', unsupiter_reloaded)
            print('Sup Iter Reloaded: ', supiter_reloaded)
    saveCheckpoint(epoch, model, optimizer, batchDirectory=batchDirectory)

def saveCheckpoint(EPOCH, net, optimizer, batchDirectory=''):
    PATH = batchDirectory+"saved_checkpoints/"+"model_"+str(EPOCH)+".pt"
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
