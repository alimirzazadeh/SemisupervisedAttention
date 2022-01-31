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
from ipdb import set_trace as bp

#from metrics.SupervisedMetricsImageNette import Evaluator
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
            

def train(model, numEpochs, suptrainloader, unsuptrainloader, validloader, optimizer, 
    target_layer, target_category, use_cuda, resolutionMatch, similarityMetric, alpha, 
    training='alternating', batchDirectory='', scheduler=None, batch_size=4, 
    unsup_batch_size=12, perBatchEval=None, saveRecurringCheckpoint=None, maskIntensity=8):
    print('alpha: ', alpha)

    CAMLossInstance = CAMLoss(model, target_layer, use_cuda, resolutionMatch, similarityMetric, maskIntensity)
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
    # weight = torch.tensor([0.87713311, 1.05761317, 0.73638968, 1.11496746, 0.78593272, 1.33506494,
    #                        0.4732965, 0.514, 0.47548566, 1.9469697, 0.97348485, 0.43670348,
    #                        1.15765766, 1.06639004, 0.13186249, 1.05544148, 1.71906355, 1.04684318,
    #                        1.028, 0.93624772])
    # weight = weight.to(device)
    # criteron = nn.MultiLabelSoftMarginLoss(weight=weight)
        
    
    #criteron = torch.nn.CrossEntropyLoss()
    criteron = nn.BCEWithLogitsLoss()
    print('pretraining evaluation...')
    model.eval()
    LossEvaluator.evaluateUpdateLosses(model, validloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=batchDirectory) #unsupervised=training!='supervised')
    LossEvaluator.plotLosses(batchDirectory=batchDirectory)
    print('finished evaluating')
        
    supdatasetSize = len(suptrainloader.dataset)
    print("\n\nTotal Supervised Dataset: ", supdatasetSize)
    unsupdatasetSize = len(unsuptrainloader.dataset)
    print("Total Unsupervised Dataset: ", unsupdatasetSize)

    if training == 'supervised':
        totalDatasetSize = int(supdatasetSize / batch_size)
    elif training == 'unsupervised':
        totalDatasetSize = int(unsupdatasetSize / unsup_batch_size)
    elif training == 'combining':
        totalDatasetSize = int(supdatasetSize / batch_size)
        # trainingRatio = alpha * (supdatasetSize / (alpha * supdatasetSize + unsupdatasetSize))
    elif training == 'alternating':
        totalDatasetSize = int(alpha * supdatasetSize / batch_size + unsupdatasetSize / unsup_batch_size)
        trainingRatio = alpha * (supdatasetSize / (alpha * supdatasetSize + unsupdatasetSize))

    
    print("Total Dataset: ", totalDatasetSize)
    

    ##Custom model.train that freezes the batch norm layers and only keeps others in train mode
    customTrain(model)
    
    
    for epoch in range(numEpochs):
        
        # if scheduler:
        #     scheduler.step()        
        # running_corrects = 0
        # running_loss = 0.0

        supiter = iter(suptrainloader)
        unsupiter = iter(unsuptrainloader)
        supiter_reloaded = 0
        unsupiter_reloaded = 0
        

        if saveRecurringCheckpoint is not None and epoch % saveRecurringCheckpoint == saveRecurringCheckpoint - 1:
           saveCheckpoint(epoch, model, optimizer, batchDirectory=batchDirectory)
           print("saved checkpoint successfully")
        
        counter = 0


        if training == 'supervised':
            supervised = True
            alternating = False
            combining = False
        elif training == 'unsupervised':
            supervised = False
            alternating = False
            combining = False
        elif training == 'alternating':
            alternating = True
            combining = False
        elif training == 'combining':
            alternating = False
            combining = True

        # for i, data in enumerate(trainloader, 0):
        #print('starting iterations...')
        for i in range(totalDatasetSize):

            if alternating:
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
            elif combining:
                data = supiter.next()
                try:
                    data_u = unsupiter.next()
                except StopIteration:
                    unsupiter = iter(unsuptrainloader)
                    unsupiter_reloaded += 1
                    data_u = unsupiter.next()
                inputs_u, labels_u = data_u
            elif supervised:
                data = supiter.next()
                #print('s')
            elif not supervised:
                data = unsupiter.next()


            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
    
            # zero the parameter gradients
            
            with torch.set_grad_enabled(True):
                if combining or supervised:
                    model.train()
                    optimizer.zero_grad()
                    CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs) 
                    try: 
                        l1 = criteron(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    except:
                        l1 = criteron(outputs.logits, labels)
                        _, preds = torch.max(outputs.logits, 1)
                    
                    # for pred in range(preds.shape[0]):
                    #     running_corrects += labels[pred, int(preds[pred])]
                if combining or not supervised:
                    customTrain(model)
                    optimizer.zero_grad()
                    # print('unsupervised')
                    CAMLossInstance.cam_model.activations_and_grads.register_hooks()
                    if combining:
                        l2 = CAMLossInstance(inputs_u, target_category)
                    else:
                        l1 = CAMLossInstance(inputs, target_category)
                    
                optimizer.zero_grad()
                if combining:
                    l1 = l1 + alpha * l2
                #print(l1)
                l1.backward()
                optimizer.step()
            counter += 1
            
            if perBatchEval != None and counter % perBatchEval == perBatchEval - 1:
                print('Epoch {} counter {}'.format(epoch, counter))
                model.eval()
                optimizer.zero_grad()
                LossEvaluator.evaluateUpdateLosses(model, validloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=batchDirectory) #training!='supervised')
                LossEvaluator.plotLosses(batchDirectory=batchDirectory)
        if perBatchEval == None:
            print('Epoch {} of {}'.format(epoch, numEpochs))
            model.eval()
            optimizer.zero_grad()
            LossEvaluator.evaluateUpdateLosses(model, validloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=batchDirectory) #training!='supervised')
            LossEvaluator.plotLosses(batchDirectory=batchDirectory)

    print('\n \n BEST SUP LOSS OVERALL: ', LossEvaluator.bestSupSum, '\n\n')
    print('\n \n BEST MAP OVERALL: ', LossEvaluator.bestmAP, '\n\n')

    #save a final checkpoint
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