# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:22:20 2021

@author: alimi
"""

import pandas as pd
import numpy as np
import os

#RUN WITH PYTHON 3

#Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.
#Bootstrap hypothesis testing


def custom_metric(data_methodx):
    def calculateF1score(tp, fp, fn):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return 2 * (recall * precision) / (recall + precision)
    
    tp = None
    fp = None
    fn = None
    
    for preds in data_methodx.to_numpy():
        
        labels = preds[:int(len(preds) / 2)].astype("int")
        pred_logits = preds[int(len(preds) / 2):]
        if tp == None:
            tp = np.sum((pred_logits + labels > 1))
            fp = np.sum((np.subtract(pred_logits, labels) > 0))
            fn = np.sum((np.subtract(pred_logits, labels) < 0))
        else:
            tp += np.sum((pred_logits + labels > 1))
            fp += np.sum((np.subtract(pred_logits, labels) > 0))
            fn += np.sum((np.subtract(pred_logits, labels) < 0))
    f1_score_sum = calculateF1score(tp,fp,fn)
    # print(f1_score_sum)
    return f1_score_sum


def boostrapping_CI(data,nbr_runs=1000):
    #Confidence Interval Estimation of an ROC Curve: An Application of Generalized Half Normal and Weibull Distributions
    
    nbr_scans = len(data.index)
    
    list_metric = []
    #compute mean
    for r in range(nbr_runs):
        #sample random indexes
        ind = np.random.randint(nbr_scans,size=nbr_scans)
        
        #select random subset
        data_bootstrapped = data.iloc[ind]
        
        #compute metrics
        metric = custom_metric(data_bootstrapped)
        list_metric.append(metric)
        
    #store variable in dictionary
    metric_stats = {}
    metric_stats['avg_metric'] = np.average(list_metric) 
    metric_stats['metric_ci_lb'] = np.percentile(list_metric,5)
    metric_stats['metric_ci_ub'] = np.percentile(list_metric,95)

    print(metric_stats)
    return metric_stats


def boostrapping_hypothesisTesting(data_method1,data_method2,nbr_runs=100):
    
    n = len(data_method1.index)
    m = len(data_method2.index)
    total = n+m

    #compute the metric for both method    
    metric_method1 = custom_metric(data_method1)
    metric_method2 = custom_metric(data_method2)
    
    #compute statistic t
    t = abs(metric_method1 - metric_method2)
    
    #merge data from both methods
    data = pd.concat([data_method1,data_method2])
    
    #compute bootstrap statistic
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        #sample random indexes with replacement
        ind = np.random.randint(total,size=total)
        
        #select random subset with replacement
        data_bootstrapped = data.iloc[ind]
        
        #split into two groups
        data_bootstrapped_x = data_bootstrapped[:n]
        data_bootstrapped_y = data_bootstrapped[n:]

        #compute metric for both groups
        metric_x = custom_metric(data_bootstrapped_x)
        metric_y = custom_metric(data_bootstrapped_y)
        
        #compute bootstrap statistic
        t_boot = abs(metric_x - metric_y)
        
        #compare statistics
        if t_boot > t:
            nbr_cases_higher += 1
    
    
    pvalue = nbr_cases_higher*1./nbr_runs
    print(nbr_cases_higher)
    print(pvalue)
    
    return pvalue



if __name__ == '__main__':
    
    #You need to:
    #1.implement your own custom_metric function
    #2.change to code to load your data
    #3.check that your estimates (CI bounds and p-value) are stable over several runs of the bootstrapping method. If it is not, increase nbr_runs. 
    if os.path.isdir('/scratch/'):
        PATH1 = '/scratch/groups/rubin/alimirz1/saved_batches/pascal_sup/saved_checkpoints/model_best.pt'
        PATH2 = '/scratch/groups/rubin/alimirz1/saved_batches/pascal_sup/saved_checkpoints/model_best_f1.pt'
    else:
        PATH1 = '../saved_figs/8_3_21_comparison/testLabelLogits_sup.csv'
        PATH2 = '../saved_figs/8_3_21_comparison/testLabelLogits_c4.csv'
        
    #load data
    data_method1 = pd.read_csv(PATH1) #CHANGE
    data_method2 = pd.read_csv(PATH2) #CHANGE
    
    #compute CI
    metric_stats_method1 = boostrapping_CI(data_method1)
    metric_stats_method2 = boostrapping_CI(data_method2)
    
    #compare method 1 and 2
    pvalue = boostrapping_hypothesisTesting(data_method1,data_method2) 
    


