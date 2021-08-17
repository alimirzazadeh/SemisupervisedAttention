# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:19:13 2021

@author: alimi
"""
import torch
import pandas as pd
from torch import nn
import numpy as np
from metrics.ConfidenceIntervals import boostrapping_CI

def evaluate(model, testloader, device, batchDirectory = ''):
    datasetSize = len(testloader.dataset)
    df = pd.DataFrame()
    
    with torch.set_grad_enabled(False):
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) 
            m = nn.Sigmoid()
            pred_logits = m(outputs)
            df = df.append(pd.DataFrame(torch.cat((labels, pred_logits),axis=1)).astype("float"))
        df.to_csv(batchDirectory + 'saved_figs/testLabelLogits.csv', index=False)
        boostrapping_CI(df)
    print("Finished Evaluation")