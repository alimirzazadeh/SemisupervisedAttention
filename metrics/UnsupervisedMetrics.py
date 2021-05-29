# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:04:39 2021

@author: alimi
"""

from model.loss import CAMLoss

def visualizeLossPerformance(model, target_layer, inputs, imgTitle, use_cuda=False, target_category=None):
    CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
    l1 = CAMLossInstance(inputs, target_category, visualize=True, imgTitle=imgTitle)
    return l1