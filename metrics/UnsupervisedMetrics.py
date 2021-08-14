# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:04:39 2021

@author: alimi
"""

from model.loss import CAMLoss
import matplotlib.pyplot as plt
import numpy as np

from visualizer.visualizer import visualizeImageBatch


def visualizeLossPerformance(CAMLossInstance, inputs, labels=['sentinel'], imgLabels=[], imgTitle="epoch_0_batchNum_0", use_cuda=False, target_category=None, saveFig=True, batchDirectory=''):
    # CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    l1, fig = CAMLossInstance(inputs, target_category, visualize=True)
    def arrToStr(s):
            str1 = " " 
            # return string  
            return (str1.join(s))
    if saveFig:
        plt.clf()
        plt.imshow(fig)
        l1 = [str(round(x,3)) for x in l1]
        plt.title(arrToStr(l1))
        # print('losses, ',l1)
        print('saving to: ' + batchDirectory+'saved_figs/checkpointvis_'+imgTitle+'.png')
        classes = str(labels)
        titleString = ' '.join('%5s' % classes)
        plt.title(titleString + '\n' + arrToStr(imgLabels))
        plt.savefig(batchDirectory+'saved_figs/checkpointvis_'+imgTitle+'.png')
        # plt.clf()
        # visualizeImageBatch(inputs, labels, resnetLabels=arrToStr(imgLabels))
        # plt.savefig('./'+batchDirectory+'saved_figs/checkpointvis_'+imgTitle+'_orig.png')

        return l1
    else:
        return l1, fig