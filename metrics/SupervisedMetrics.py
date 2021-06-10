# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:28:10 2021

@author: alimi
"""
import torch

def evaluateModelPerformance(model, testloader, criteron):
    model.eval()
    running_corrects = 0
    running_loss = 0.0
    datasetSize = len(testloader.dataset)
    with torch.set_grad_enabled(False):
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = model(inputs) 
            l1 = criteron(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += l1.item()
            running_corrects += torch.sum(preds == labels.data)
    print('\n Test Model Accuracy: %.3f' % running_corrects / datasetSize)
    print('\n Test Model Loss: %.3f' % running_loss / datasetSize)