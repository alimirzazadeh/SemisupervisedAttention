# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:28:10 2021

@author: alimi
"""
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from model.layer_attention_loss import isLayerWiseAttention
from model.transformer_loss import isTransformer
from ipdb import set_trace as bp

def removeHooks(lossInstance, attentionMethod):
    if isLayerWiseAttention(attentionMethod):
        lossInstance.outer_gradcam.activations_and_grads.remove_hooks() 
        lossInstance.inner_gradcam.activations_and_grads.remove_hooks() 
    elif isTransformer(attentionMethod):
        pass
    else:
        lossInstance.cam_model.activations_and_grads.remove_hooks()

def registerHooks(lossInstance, attentionMethod):
    if isLayerWiseAttention(attentionMethod):
        lossInstance.outer_gradcam.activations_and_grads.register_hooks() 
        lossInstance.inner_gradcam.activations_and_grads.register_hooks()
    elif isTransformer(attentionMethod):
        pass 
    else:
        lossInstance.cam_model.activations_and_grads.register_hooks()

def calculateLoss(lossInstance, inputs, target_category, labels, attentionMethod, visualize=False):
    if isLayerWiseAttention(attentionMethod):
        return lossInstance(inputs, labels, visualize=visualize)
    return lossInstance(inputs, target_category, visualize=visualize)

class Evaluator:
    def __init__(self):
        self.supervised_losses = []
        self.unsupervised_losses = []
        self.mAPs = []
        self.bestmAP = 0
        self.bestSupSum = 999999
        self.counter = 0
    def evaluateModelSupervisedPerformance(self, model, testloader, criteron, device, optimizer, storeLoss = False, batchDirectory=''):
        running_loss = 0.0
        firstTime = True
        allTrueLabels = None
        allPredLabels = None
        
        datasetSize = len(testloader.dataset)
        


        with torch.set_grad_enabled(False):
            m = nn.Sigmoid()
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs) 
                l1 = criteron(outputs, labels)
                
                running_loss += l1.item()
                
                if firstTime:
                    allTrueLabels = labels.cpu().detach().numpy()
                    allPredLabels = m(outputs).cpu().detach().numpy()
                    firstTime = False
                else:
                    allTrueLabels = np.append(allTrueLabels, labels.cpu().detach().numpy(), axis=0)
                    allPredLabels = np.append(allPredLabels, outputs.cpu().detach().numpy(), axis=0)

            supervised_loss = float(running_loss / datasetSize)
            print('\n Test Model Supervised Loss: %.3f' % supervised_loss)
            mAP = average_precision_score(allTrueLabels,allPredLabels,average='weighted')
            print('\n Test Model mAP: %.3f' % mAP)

            self.counter += 1
            
            if mAP > self.bestmAP:
                self.bestmAP = mAP
                print("\n Best mAP so far: ", self.bestmAP)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, f1orsup=0)
            if supervised_loss < self.bestSupSum:
                self.bestSupSum = supervised_loss
                print("\n Best Sup Loss so far: ", self.bestSupSum)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, f1orsup=1)
        
        
            
            if storeLoss:
                self.supervised_losses.append(supervised_loss)
                self.mAPs.append(mAP)

    def evaluateModelUnsupervisedPerformance(self, model, testloader, lossInstance, device, optimizer, attentionMethod, target_category=None, storeLoss = False):
        running_loss = 0.0
        datasetSize = len(testloader.dataset)
        with torch.set_grad_enabled(True):
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                l1 = calculateLoss(lossInstance, inputs, target_category, torch.argmax(labels, dim=1), attentionMethod)
                running_loss += l1.item()
        print('\n Test Model Unsupervised Loss: %.3f' % float(running_loss / datasetSize))
        if storeLoss:
            self.unsupervised_losses.append(float(running_loss / datasetSize))
    def evaluateUpdateLosses(self, model, testloader, criteron, lossInstance, device, optimizer, attentionMethod, unsupervised=True, batchDirectory=''):
        if unsupervised:
            registerHooks(lossInstance, attentionMethod)
            self.evaluateModelUnsupervisedPerformance(model, testloader, lossInstance, device, optimizer, attentionMethod, storeLoss = True)
        removeHooks(lossInstance, attentionMethod)
        self.evaluateModelSupervisedPerformance(model, testloader, criteron, device, optimizer, storeLoss = True, batchDirectory= batchDirectory)
        results = pd.DataFrame()              
        results['Supervised Loss'] = self.supervised_losses     
        results['Unsupervised Loss'] = self.unsupervised_losses     
        results['mAPs'] = self.mAPs      
        results.to_csv(batchDirectory+'saved_figs/results.csv', header=True)

    def plotLosses(self, batchDirectory=''):
        plt.clf()
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.supervised_losses, label="Supervised Loss")
        axs[0, 0].set_title('Supervised Loss')
        axs[0, 1].plot(self.unsupervised_losses, label="Unsupervised Loss")
        axs[0, 1].set_title('Unsupervised Loss')
        axs[1, 0].plot(self.mAPs, label="mAPs")
        axs[1, 0].set_title('mAPs')
        plt.tight_layout()
        plt.savefig(batchDirectory+'saved_figs/AllPlots.png')
        plt.close()

    def calculateF1score(self, tp, fp, fn):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * (recall * precision) / (recall + precision)
        
    def saveCheckpoint(self, net, optimizer, batchDirectory='',f1orsup=1):
        if f1orsup == 1:
            PATH = batchDirectory+"saved_checkpoints/model_best.pt"
        else:
            PATH = batchDirectory+"saved_checkpoints/model_best_mAP.pt"
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)