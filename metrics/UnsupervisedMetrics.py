# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:04:39 2021

@author: alimi
"""

from model.loss import CAMLoss
import matplotlib.pyplot as plt
import numpy as np
import cv2

from visualizer.visualizer import visualizeImageBatch
from metrics.SupervisedMetrics import calculateLoss


def visualizeLossPerformance(lossInstance, inputs, attentionMethod, labels=['sentinel'], imgLabels=[], imgTitle="epoch_0_batchNum_0", use_cuda=False, target_category=None, saveFig=True, batchDirectory=''):
    # l1, fig = lossInstance(inputs, target_category, visualize=True)
    l1, final_img, final_firsts, final_correlates, final_mask, final_maskimg, final_seconds = \
        lossInstance(inputs, target_category, visualize=True)
    # l1, final_img, final_firsts, final_correlates, final_mask, final_maskimg, final_seconds = \
    #     calculateLoss(lossInstance, inputs, target_category, labels, attentionMethod, visualize=True)
    fig = np.array(cv2.vconcat([final_img.astype(
        'float64'), final_firsts.astype('float64'), final_correlates.astype('float64')
        , final_mask.astype('float64'), final_maskimg.astype('float64'), final_seconds.astype('float64')]))
    def arrToStr(s):
            str1 = " "  
            return (str1.join(s))
    if saveFig:
        plt.clf()
        plt.imshow(fig)
        l1 = [str(round(x,3)) for x in l1]
        plt.title(arrToStr(l1))
        print('saving to: ' + batchDirectory+'saved_figs/checkpointvis_'+imgTitle+'.png')
        classes = str(labels)
        titleString = ' '.join('%5s' % classes)
        plt.title(titleString + '\n' + arrToStr(imgLabels))
        plt.savefig(batchDirectory+'saved_figs/checkpointvis_'+imgTitle+'.png')

        return l1
    else:
        return l1, fig