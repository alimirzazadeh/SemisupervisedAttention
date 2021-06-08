# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:04:39 2021

@author: alimi
"""

from model.loss import CAMLoss
import matplotlib.pyplot as plt
import numpy as np

from visualizer.visualizer import visualizeImageBatch


def visualizeLossPerformance(CAMLossInstance, inputs, labels=['sentinel'], imgTitle="epoch_0_batchNum_0", use_cuda=False, target_category=None, saveFig=True):
    # CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    l1, fig = CAMLossInstance(inputs, target_category, visualize=True)
    if saveFig:
        plt.clf()
        plt.imshow(fig)
        l1 = [round(x,3) for x in l1]
        plt.title(l1)
        # print('losses, ',l1)
        plt.savefig('./saved_figs/unsupervised_viz_'+imgTitle+'.png')
        plt.clf()
        visualizeImageBatch(inputs, labels)
        plt.savefig('./saved_figs/unsupervised_viz_'+imgTitle+'_orig.png')

        return l1
    else:
        return l1, fig