# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:32:41 2021

@author: alimi
"""
from libs.pytorch_grad_cam.grad_cam import GradCAM
from libs.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from libs.pytorch_grad_cam.utils.image import deprocess_image, preprocess_image
from visualizer.visualizer import visualizeImageBatch, show_cam_on_image

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch


def calculateLoss(input_tensor, model, target_layer, target_category, use_cuda=False, logs=False, visualize=False):
    cam_model = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)


    assert(len(input_tensor.shape) > 3)
    
    
    # correlation_pearson = torch.zeros(input_tensor.shape[0])
    # correlation_pearson.requires_grad=True
    correlation_pearson = torch.zeros(1)
    correlation_pearson.requires_grad = True
    # correlation_cross = tor.zeros(input_tensor.shape[0])
    
    if visualize:
        fig, axs = plt.subplots(3, input_tensor.shape[0])
        gbimgs = []
        imgs = []
        hmps = []
        
    
    for i in range(input_tensor.shape[0]):


        # thisImg = cv2.resize(thisImg, (256, 256))
        
        thisImgTensor = input_tensor[i,:,:,:]
        thisImgPreprocessed = thisImgTensor.unsqueeze(0)
            
        ##we have to keep the gradients that are calculated, from the cam_model forward method
        ####could either rerun the model or use the already calculated values i think
        ###post processing has to be done with torch operations
        # thisImgPreprocessed = preprocess_image(thisImg, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        grayscale_cam = cam_model(input_tensor=thisImgPreprocessed, target_category=target_category)
        # print(len(grayscale_cam))
        cam_result = grayscale_cam[0]  #####WHY IS THERE 2? and we're taking the first one?
        # print(cam_result.shape)
        gb_result = gb_model(thisImgPreprocessed, target_category=target_category)


        
        hmp_correlate = cam_result
        hmp_correlate = (hmp_correlate - torch.mean(hmp_correlate)) / torch.std(hmp_correlate)
        
        gb_correlate = gb_result
        gb_correlate = (gb_correlate - torch.mean(gb_correlate)) / torch.std(gb_correlate)
        gb_correlate = torch.abs(gb_correlate)
        gb_correlate = torch.sum(gb_correlate, axis = 2)
        # print(gb_correlate.shape)
        # print(hmp_correlate.shape)
        
        #calculate pearson's
        vx = gb_correlate - torch.mean(gb_correlate)
        vy = hmp_correlate - torch.mean(hmp_correlate)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        correlation_pearson2 = correlation_pearson.clone() 
        correlation_pearson = correlation_pearson2 + cost * -1
        # correlation_pearson[i] = np.corrcoef(hmp_correlate.flatten(), gb_correlate.flatten())[0,1]
        
        # correlation_cross[i] = np.correlate(hmp_correlate.flatten(), gb_correlate.flatten())[0]
        
        if logs:
            print("The Pearson output loss is: ", correlation_pearson[i])
            # print("The Cross Corr output loss is: ", correlation_cross[i])
            
        
        if visualize:
            rgb_img = np.float32(input_tensor.numpy()) 
            thisImg = rgb_img[i,:,:,:]
            thisImg = np.moveaxis(thisImg, 0, -1)
            axs[0,i].imshow(thisImg)
            axs[0,i].axis('off')    
            hmp, visualization = show_cam_on_image(thisImg, cam_result)
            imgs.append(visualization)
            hmps.append(hmp)
            gb_visualization = deprocess_image(gb_result)
            gbimgs.append(gb_visualization)
            axs[1,i].imshow(hmp_correlate)
            axs[1,i].set_title("Grad CAM",fontsize=6)
            axs[2,i].imshow(gb_correlate)
            axs[2,i].set_title("Backprop",fontsize=6)
            axs[1,i].axis('off')
            axs[2,i].axis('off')
            axs[0,i].set_title("Pearson Corr: " + str(round(correlation_pearson[i],3)) + "\n Cross Corr: " + str(round(correlation_cross[i])),fontsize=8)
    
    if visualize:
        final_gb_frame = cv2.hconcat(gbimgs)
        cv2.imwrite('./saved_figs/sampleImage_GuidedBackprop.jpg', final_gb_frame)
        final_frame = cv2.hconcat(imgs)
        cv2.imwrite('./saved_figs/sampleImage_GradCAM.jpg', final_frame)
        final_hmp_frame = cv2.hconcat(hmps)
        cv2.imwrite('./saved_figs/sampleImage_GradCAM_hmp.jpg', final_hmp_frame)
    # aa = torch.Tensor(correlation_pearson)
    # aa.requires_grad=True
    # aab = torch.Tensor(correlation_cross)
    # aab.requires_grad=True
    # print(correlation_pearson)
    return correlation_pearson #/ input_tensor.shape[0]