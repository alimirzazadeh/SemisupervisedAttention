# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:28:10 2021

@author: alimi
"""
import torch

def evaluateModelPerformance(model, testloader, criteron, device, optimizer):
    model.eval()
    running_corrects = 0
    running_loss = 0.0
    datasetSize = len(testloader.dataset)
    print('.')
    with torch.set_grad_enabled(False):
        for i, data in enumerate(testloader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) 
            l1 = criteron(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += l1.item()
            running_corrects += torch.sum(preds == labels.data)
            # print(running_corrects.item())
            del l1, inputs, labels, outputs, preds
        print('\n Test Model Accuracy: %.3f' % float(running_corrects.item() / datasetSize))
        print('\n Test Model Loss: %.3f' % float(running_loss / datasetSize))
    print('..')