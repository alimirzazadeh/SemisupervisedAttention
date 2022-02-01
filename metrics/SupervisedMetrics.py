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
from ipdb import set_trace as bp

class Evaluator:
    def __init__(self):
        self.supervised_losses = []
        # self.accuracies = []
        self.unsupervised_losses = []
        self.mAPs = []
        self.bestmAP = 0
        self.bestSupSum = 999999
        self.counter = 0




    def evaluateModelSupervisedPerformance(self, model, testloader, criteron, device, optimizer, storeLoss = False, batchDirectory=''):
        #model.eval()
        # running_corrects = 0
        running_loss = 0.0
        
        # tp = None
        # fp = None
        # fn = None
        firstTime = True
        allTrueLabels = None
        allPredLabels = None
        
        datasetSize = len(testloader.dataset)
        

        with torch.set_grad_enabled(False):
            m = nn.Sigmoid()
            for i, data in enumerate(testloader, 0):
                #bp()
                # print(i)
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs) 
                #print(outputs)
                #print(labels)
                #print(outputs.shape)
                #print(labels.shape)
                #bp()
                l1 = criteron(outputs, labels)
                
                # _, preds = torch.max(outputs, 1)
                
                running_loss += l1.item()
                #bp()
                if firstTime:
                    allTrueLabels = labels.cpu().detach().numpy()
                    allPredLabels = m(torch.cat((outputs[2],outputs[3]),0)).cpu().detach().numpy()
                    firstTime = False
                else:
                    allTrueLabels = np.append(allTrueLabels, labels.cpu().detach().numpy(), axis=0)
                    
                    allPredLabels = np.append(allPredLabels, torch.cat((outputs[2],outputs[3]),0).cpu().detach().numpy(), axis=0)

                    
                    
                # running_corrects += torch.sum(preds == labels.data)

                
                # for pred in range(preds.shape[0]):
                #     running_corrects += labels[pred, int(preds[pred])]
                #     m = nn.Sigmoid()
                #     pred_probability = m(outputs[pred])
                #     pred_logits = (pred_probability > 0.5).int()
                    
                #     if tp == None:
                #         tp = (pred_logits + labels[pred] > 1).int()
                #         fp = (torch.subtract(pred_logits, labels[pred]) > 0).int()
                #         fn = (torch.subtract(pred_logits, labels[pred]) < 0).int()
                #     else:
                #         tp += (pred_logits + labels[pred] > 1).int()
                #         fp += (torch.subtract(pred_logits, labels[pred]) > 0).int()
                #         fn += (torch.subtract(pred_logits, labels[pred]) < 0).int()
                    
                    # if labels[pred, int(preds[pred])] == 1:
                    #     tp += 1
                    # else:
                    #     fp += 1
                    # fn += 
                    # print(labels[pred, int(preds[pred])])
                # print(running_corrects.item())
                # del l1, inputs, labels, outputs, preds
            # print('\n Test Model Accuracy: %.3f' % float(running_corrects.item() / datasetSize))
            supervised_loss = float(running_loss / datasetSize)
            print('\n Test Model Supervised Loss: %.3f' % supervised_loss)
            mAP = average_precision_score(allTrueLabels,allPredLabels,average='weighted')
            print('\n Test Model mAP: %.3f' % mAP)
            # f1_score = self.calculateF1score(tp, fp, fn)
            
            # try:
            #     pd.DataFrame(dict(enumerate(f1_score.data.cpu().numpy())),index=[self.counter]).to_csv(batchDirectory+'saved_figs/f1_scores.csv', mode='a', header=False)
            # except:
            #     pd.DataFrame(dict(enumerate(f1_score.data.cpu().numpy())),index=[self.counter]).to_csv(batchDirectory+'saved_figs/f1_scores.csv', header=False)
            self.counter += 1
            
            # f1_sum = np.nansum(f1_score.data.cpu().numpy())
            
            if mAP > self.bestmAP:
                self.bestmAP = mAP
                print("\n Best mAP so far: ", self.bestmAP)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, f1orsup=0)
                # self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory)
            
            # print('\n F1 Score: ', f1_score.data.cpu().numpy())
            # print('\n F1 Score Sum: ', f1_sum)
            
            if supervised_loss < self.bestSupSum:
                self.bestSupSum = supervised_loss
                print("\n Best Sup Loss so far: ", self.bestSupSum)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, f1orsup=1)
        
        
            
            if storeLoss:
                self.supervised_losses.append(supervised_loss)
                # self.accuracies.append(float(running_corrects.item() / datasetSize))
                self.mAPs.append(mAP)
        #print('..')
    def evaluateModelUnsupervisedPerformance(self, model, testloader, CAMLossInstance, device, optimizer, target_category=None, storeLoss = False):
        # model.eval()
        running_loss = 0.0
        datasetSize = len(testloader.dataset)
        #print('.')
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
    def evaluateUpdateLosses(self, model, testloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=''):
        if False:
            #print('evaluating unsupervised performance')
            CAMLossInstance.cam_model.activations_and_grads.register_hooks()
            self.evaluateModelUnsupervisedPerformance(model, testloader, CAMLossInstance, device, optimizer, storeLoss = True)
        #print('evaluating supervised performance')
        #CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
        self.evaluateModelSupervisedPerformance(model, testloader, criteron, device, optimizer, storeLoss = True, batchDirectory= batchDirectory)
        results = pd.DataFrame()        
        # results['Accuracy'] = self.accuracies       
        results['Supervised Loss'] = self.supervised_losses     
        #results['Unsupervised Loss'] = self.unsupervised_losses     
        results['mAPs'] = self.mAPs      
        results.to_csv(batchDirectory+'saved_figs/results.csv', header=True)
    def plotLosses(self, batchDirectory=''):
        plt.clf()
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.supervised_losses, label="Supervised Loss")
        axs[0, 0].set_title('Supervised Loss')
        #plt.savefig(batchDirectory+'saved_figs/SupervisedLossPlot.png')
        #plt.clf()
        axs[0, 1].plot(self.unsupervised_losses, label="Unsupervised Loss")
        axs[0, 1].set_title('Unsupervised Loss')
        #plt.savefig(batchDirectory+'saved_figs/UnsupervisedLossPlot.png')
        #plt.clf()
        axs[1, 0].plot(self.mAPs, label="mAPs")
        axs[1, 0].set_title('mAPs')
        #plt.savefig(batchDirectory+'saved_figs/TotalLossPlot.png')
        #plt.clf()
        # axs[1, 1].plot(self.accuracies, label="Accuracy")
        # axs[1, 1].set_title('Accuracy')
        plt.tight_layout()
        plt.savefig(batchDirectory+'saved_figs/AllPlots.png')
        plt.close()
        # plt.legend()
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