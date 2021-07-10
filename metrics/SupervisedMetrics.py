# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:28:10 2021

@author: alimi
"""
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

class Evaluator:
    def __init__(self):
        self.supervised_losses = []
        self.accuracies = []
        self.unsupervised_losses = []
        self.total_loss = []
    def evaluateModelSupervisedPerformance(self, model, testloader, criteron, device, optimizer, storeLoss = False):
        model.eval()
        running_corrects = 0
        running_loss = 0.0
        
        tp = None
        fp = None
        fn = None
        
        datasetSize = len(testloader.dataset)
        print('.')
        with torch.set_grad_enabled(False):
            for i, data in enumerate(testloader, 0):
                # print(i)
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs) 
                l1 = criteron(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                running_loss += l1.item()
                # running_corrects += torch.sum(preds == labels.data)
                for pred in range(preds.shape[0]):
                    running_corrects += labels[pred, int(preds[pred])]
                    m = nn.Sigmoid()
                    pred_probability = m(outputs[pred])
                    pred_logits = (pred_probability > 0.5).int()
                    
                    if tp == None:
                        tp = (pred_logits + labels[pred] > 1).int()
                        fp = (torch.subtract(pred_logits, labels[pred]) > 0).int()
                        fn = (torch.subtract(pred_logits, labels[pred]) < 0).int()
                    else:
                        tp += (pred_logits + labels[pred] > 1).int()
                        fp += (torch.subtract(pred_logits, labels[pred]) > 0).int()
                        fn += (torch.subtract(pred_logits, labels[pred]) < 0).int()
                    
                    # if labels[pred, int(preds[pred])] == 1:
                    #     tp += 1
                    # else:
                    #     fp += 1
                    # fn += 
                    # print(labels[pred, int(preds[pred])])
                # print(running_corrects.item())
                # del l1, inputs, labels, outputs, preds
            print('\n Test Model Accuracy: %.3f' % float(running_corrects.item() / datasetSize))
            print('\n Test Model Supervised Loss: %.3f' % float(running_loss / datasetSize))
            f1_score = self.calculateF1score(tp, fp, fn)
            print('\n F1 Score: \n', f1_score.data.cpu().numpy())
            
            
            if storeLoss:
                self.supervised_losses.append(float(running_loss / datasetSize))
                self.accuracies.append(float(running_corrects.item() / datasetSize))
        print('..')
    def evaluateModelUnsupervisedPerformance(self, model, testloader, CAMLossInstance, device, optimizer, target_category=None, storeLoss = False):
        model.eval()
        running_loss = 0.0
        datasetSize = len(testloader.dataset)
        print('.')
        with torch.set_grad_enabled(True):
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                l1 = CAMLossInstance(inputs, target_category, visualize=False)
                running_loss += l1.item()
        print('\n Test Model Unsupervised Loss: %.3f' % float(running_loss / datasetSize))
        if storeLoss:
            self.unsupervised_losses.append(float(running_loss / datasetSize))
    def evaluateUpdateLosses(self, model, testloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True):
        if unsupervised:
            print('evaluating unsupervised performance')
            CAMLossInstance.cam_model.activations_and_grads.register_hooks()
            self.evaluateModelUnsupervisedPerformance(model, testloader, CAMLossInstance, device, optimizer, storeLoss = True)
        print('evaluating supervised performance')
        CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
        self.evaluateModelSupervisedPerformance(model, testloader, criteron, device, optimizer, storeLoss = True)
        self.total_loss = [a + b for a, b in zip(self.supervised_losses, self.unsupervised_losses)]
    def plotLosses(self, batchDirectory=''):
        plt.clf()
        plt.plot(self.supervised_losses, label="Supervised Loss")
        plt.savefig(batchDirectory+'saved_figs/SupervisedLossPlot.png')
        plt.clf()
        plt.plot(self.unsupervised_losses, label="Unsupervised Loss")
        plt.savefig(batchDirectory+'saved_figs/UnsupervisedLossPlot.png')
        plt.clf()
        plt.plot(self.total_loss, label="Total Loss")
        plt.savefig(batchDirectory+'saved_figs/TotalLossPlot.png')
        plt.clf()
        plt.plot(self.accuracies, label="Accuracy")
        plt.savefig(batchDirectory+'saved_figs/AccuracyPlot.png')
        # plt.legend()
    def calculateF1score(self, tp, fp, fn):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * (recall * precision) / (recall + precision)
        
