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
import sklearn
from sklearn.preprocessing import OneHotEncoder

class Evaluator:
    def __init__(self):
        self.supervised_losses = []
        self.accuracies = []
        self.unsupervised_losses = []
        self.f1_scoresum = []
        self.bestF1Sum = 0
        self.bestSupSum = 999999
        self.counter = 0

    def evaluateModelSupervisedPerformance(self, model, testloader, criteron, device, optimizer, storeLoss=True, batchDirectory=''):
        running_corrects = 0
        running_loss = 0.0
        tp = None
        fp = None
        fn = None
        datasetSize = len(testloader.dataset)
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

                numClasses = outputs.shape[1]
                if tp == None:
                    tp = torch.zeros(numClasses)
                    fp = torch.zeros(numClasses)
                    fn = torch.zeros(numClasses)
                for i in range(outputs.shape[0]):
                    if preds[i] == labels[i]:
                        tp[preds[i].item()] += 1
                    else:
                        fp[preds[i].item()] += 1
                        fn[labels[i].item()] += 1

            acc = float(running_corrects.item() / datasetSize)
            print('\n Test Model Accuracy: %.3f' % acc)
            supervised_loss = float(running_loss / datasetSize)
            print('\n Test Model Supervised Loss: %.3f' % supervised_loss)
            f1_score = self.calculateF1score(tp, fp, fn)
            self.counter += 1
            f1_sum = np.nansum(f1_score.data.cpu().numpy()) / numClasses

            if f1_sum > self.bestF1Sum:
                self.bestF1Sum = f1_sum
                print("\n Best F1 Score so far: ", self.bestF1Sum)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, f1orsup=0)

            if supervised_loss < self.bestSupSum:
                self.bestSupSum = supervised_loss
                print("\n Best Sup Loss so far: ", self.bestSupSum)
                self.saveCheckpoint(model, optimizer, batchDirectory = batchDirectory, f1orsup=1)

            if storeLoss:
                self.supervised_losses.append(round(supervised_loss, 5))
                self.accuracies.append(round(acc, 5))
                self.f1_scoresum.append(round(f1_sum, 5))

    def evaluateModelUnsupervisedPerformance(self, model, testloader, CAMLossInstance, device, optimizer, target_category=None, storeLoss=True):
        # model.eval()
        running_loss = 0.0
        datasetSize = len(testloader.dataset)
        with torch.set_grad_enabled(True):
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                inputs = inputs.to(device)
                l1 = CAMLossInstance(inputs, target_category, visualize=False)
                running_loss += l1.item()
        unsupervised_loss = float(running_loss / datasetSize)
        print('\n Test Model Unsupervised Loss: %.3f' % unsupervised_loss)
        if storeLoss:
            self.unsupervised_losses.append(round(unsupervised_loss, 5))

    def evaluateUpdateLosses(self, model, testloader, criteron, CAMLossInstance, device, optimizer, unsupervised=True, batchDirectory=''):
        if unsupervised:
            CAMLossInstance.cam_model.activations_and_grads.register_hooks()
            self.evaluateModelUnsupervisedPerformance(model, testloader, CAMLossInstance, device, optimizer, storeLoss=True)
        CAMLossInstance.cam_model.activations_and_grads.remove_hooks()
        self.evaluateModelSupervisedPerformance(model, testloader, criteron, device, optimizer, storeLoss=True, batchDirectory=batchDirectory)
        results = pd.DataFrame()
        results['Accuracy'] = self.accuracies
        results['Supervised Loss'] = self.supervised_losses
        results['Unsupervised Loss'] = self.unsupervised_losses
        results['F1 score'] = self.f1_scoresum
        results.to_csv(batchDirectory+'saved_figs/results.csv', header=True)

    def plotLosses(self, batchDirectory=''):
        plt.clf()
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.supervised_losses, label="Supervised Loss")
        axs[0, 0].set_title('Supervised Loss')
        # plt.savefig(batchDirectory+'saved_figs/SupervisedLossPlot.png')
        # plt.clf()
        axs[0, 1].plot(self.unsupervised_losses, label="Unsupervised Loss")
        axs[0, 1].set_title('Unsupervised Loss')
        # plt.savefig(batchDirectory+'saved_figs/UnsupervisedLossPlot.png')
        # plt.clf()
        axs[1, 0].plot(self.f1_scoresum, label="F1 Score Sum")
        axs[1, 0].set_title('F1 Score Sum')
        # plt.savefig(batchDirectory+'saved_figs/TotalLossPlot.png')
        # plt.clf()
        axs[1, 1].plot(self.accuracies, label="Accuracy")
        axs[1, 1].set_title('Accuracy')
        plt.savefig(batchDirectory+'saved_figs/AllPlots.png')
        plt.close()
        # plt.legend()
        plt.close()

    def calculateF1score(self, tp, fp, fn):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * (recall * precision) / (recall + precision)
        
    def saveCheckpoint(self, net, optimizer, batchDirectory='', f1orsup=1):
        if f1orsup == 1:
            PATH = batchDirectory+"saved_checkpoints/model_best.pt"
        else:
            PATH = batchDirectory+"saved_checkpoints/model_best_f1.pt"
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)