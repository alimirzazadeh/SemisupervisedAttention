# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:32:41 2021

@author: alimi
"""
from libs.pytorch_grad_cam.grad_cam import GradCAM
from libs.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from libs.pytorch_grad_cam.utils.image import deprocess_image, preprocess_image
from visualizer.visualizer import visualizeImageBatch, show_cam_on_image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch

class CAMLoss(nn.Module):
    def __init__(self, model, target_layer, use_cuda):
        super(CAMLoss, self).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.cam_model = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
        self.gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        
    def forward(self, predict, target):
        print("hey")
    def forward(self, input_tensor,  target_category, logs=False, visualize=False):
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
            correlations = []
            
        
        for i in range(input_tensor.shape[0]):
    
    
            # thisImg = cv2.resize(thisImg, (256, 256))
            
            thisImgTensor = input_tensor[i,:,:,:]
            thisImgPreprocessed = thisImgTensor.unsqueeze(0)
                
            ##we have to keep the gradients that are calculated, from the cam_model forward method
            ####could either rerun the model or use the already calculated values i think
            ###post processing has to be done with torch operations
            # thisImgPreprocessed = preprocess_image(thisImg, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            grayscale_cam = self.cam_model(input_tensor=thisImgPreprocessed, target_category=target_category)
            # print(len(grayscale_cam))
            cam_result = grayscale_cam[0]  #####WHY IS THERE 2? and we're taking the first one?
            # print(cam_result.shape)
            gb_result = self.gb_model(thisImgPreprocessed, target_category=target_category)
    
    
            
            hmp_correlate = cam_result
            hmp_correlate_std = torch.std(hmp_correlate)
            if (hmp_correlate_std > 0):
                hmp_correlate = (hmp_correlate - torch.mean(hmp_correlate)) / hmp_correlate_std
            
            gb_correlate = gb_result
            gb_correlate_std = torch.std(gb_correlate)
            if (gb_correlate_std > 0):
                gb_correlate = (gb_correlate - torch.mean(gb_correlate)) / gb_correlate_std
            gb_correlate = torch.abs(gb_correlate)
            gb_correlate = torch.sum(gb_correlate, axis = 2)
            # print(torch.sum(gb_correlate))
            # print(gb_correlate.shape)
            # print(hmp_correlate.shape)
            
            #calculate pearson's
            vx = gb_correlate - torch.mean(gb_correlate)
            vy = hmp_correlate - torch.mean(hmp_correlate)
            denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
            if denominator > 0:
                cost = torch.sum(vx * vy) / denominator
            else:
                cost = torch.zeros(1)
            correlation_pearson2 = correlation_pearson.clone() 
            correlation_pearson = correlation_pearson2 + cost * -1

            # print(torch.sum(hmp_correlate), torch.sum(gb_correlate), cost)
            # correlation_pearson[i] = np.corrcoef(hmp_correlate.flatten(), gb_correlate.flatten())[0,1]
            
            # correlation_cross[i] = np.correlate(hmp_correlate.flatten(), gb_correlate.flatten())[0]
            
            if logs:
                print("The Pearson output loss is: ", correlation_pearson[i])
                # print("The Cross Corr output loss is: ", correlation_cross[i])
                
            
            if visualize:
                correlations.append(cost)
                rgb_img = np.float32(input_tensor.numpy()) 
                thisImg = rgb_img[i,:,:,:]
                print(np.max(thisImg))
                thisImg = thisImg / np.max(thisImg)
                print(np.max(thisImg))
                thisImg = np.moveaxis(thisImg, 0, -1)
                axs[0,i].imshow(thisImg)
                axs[0,i].axis('off')    
                hmp, visualization = show_cam_on_image(thisImg, cam_result)
                imgs.append(visualization)
                hmps.append(hmp)
                print(gb_result.shape)
                gb_visualization = deprocess_image(gb_result.numpy())
                gbimgs.append(gb_visualization)
                axs[1,i].imshow(hmp_correlate)
                axs[1,i].set_title("Grad CAM",fontsize=6)
                axs[2,i].imshow(gb_correlate)
                axs[2,i].set_title("Backprop",fontsize=6)
                axs[1,i].axis('off')
                axs[2,i].axis('off')
                axs[0,i].set_title("Pearson Corr: " + str(round(cost.item(),3)),fontsize=8)
        
        if visualize:
            return fig, correlations
            # final_gb_frame = cv2.hconcat(gbimgs)
            # cv2.imwrite('./saved_figs/sampleImage_GuidedBackprop.jpg', final_gb_frame)
            # final_frame = cv2.hconcat(imgs)
            # cv2.imwrite('./saved_figs/sampleImage_GradCAM.jpg', final_frame)
            # final_hmp_frame = cv2.hconcat(hmps)
            # cv2.imwrite('./saved_figs/sampleImage_GradCAM_hmp.jpg', final_hmp_frame)
            
        # aa = torch.Tensor(correlation_pearson)
        # aa.requires_grad=True
        # aab = torch.Tensor(correlation_cross)
        # aab.requires_grad=True
        # print(correlation_pearson)

        # print(correlation_pearson)
        return correlation_pearson #/ input_tensor.shape[0]