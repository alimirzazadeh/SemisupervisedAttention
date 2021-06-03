# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:04:39 2021

@author: alimi
"""

from model.loss import CAMLoss
import matplotlib.pyplot as plt
import numpy as np


def visualizeLossPerformance(model, target_layer, inputs, imgTitle="epoch_0_batchNum_0", use_cuda=False, target_category=None, saveFig=True):
    CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    l1, fig = CAMLossInstance(inputs, target_category, visualize=True)
    if saveFig:
        fig.savefig('./saved_figs/unsupervised_viz_'+imgTitle+'.png')
        return l1
    else:
        return l1, fig