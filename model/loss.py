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
import torch.nn.functional as F

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
            grayscale_cam, topClass = self.cam_model(input_tensor=thisImgPreprocessed, target_category=target_category, returnTarget=True)
            
            # print(len(grayscale_cam))
            cam_result = grayscale_cam[0]  #####WHY IS THERE 2? and we're taking the first one?
            # hmp_correlate = cam_result
            # hmp_correlate_std = torch.std(hmp_correlate)
            # if (hmp_correlate_std > 0):
            #     hmp_correlate = (hmp_correlate - torch.mean(hmp_correlate)) / hmp_correlate_std
            
            if visualize:
                hmps.append(cam_result.numpy())

            # print(cam_result.shape)
            gb_correlate = self.gb_model(thisImgPreprocessed, target_category=target_category)
            def processGB(gb_correlate):
                gb_correlate_std = torch.std(gb_correlate)
                if (gb_correlate_std > 0):
                    gb_correlate = (gb_correlate - torch.mean(gb_correlate)) / gb_correlate_std
    
                ####now zeroing negatives
                # gb_correlate[gb_correlate > 0] = 0
                # gb_correlate = -1 * gb_correlate
                gb_correlate = torch.abs(gb_correlate) ################################################################################# 
                gb_correlate = torch.sum(gb_correlate, axis = 2)
                return gb_correlate
            gb_correlate = processGB(gb_correlate)
            
            CAMtoGB = False
            def standardize(arr):
                arr = (arr - torch.mean(arr))/torch.std(arr)
                return arr.unsqueeze(0).unsqueeze(0)
            
            if CAMtoGB:
                ww = -8
                sigma = torch.mean(gb_correlate) + torch.std(gb_correlate) / 2
                # print(sigma)
                TAc = 1/ (1 + torch.exp(ww * (gb_correlate - sigma)))
                TAc = TAc.unsqueeze(0)
                TAc = torch.repeat_interleave(TAc, 3, dim=0)
                # print(TAc.shape)
                newImgTensor = TAc * thisImgTensor
                newImgPreprocessed = newImgTensor.unsqueeze(0)
                new_grayscale_cam = self.cam_model(input_tensor=newImgPreprocessed, target_category=int(topClass[0]))
                new_cam_result = new_grayscale_cam[0]
                
                if visualize:
                    gbimgs.append(new_cam_result.numpy())
                
                cost = -1 * torch.abs(F.conv2d(standardize(cam_result), standardize(new_cam_result)))
            else:
                ww = -32
                sigma = torch.mean(cam_result) + torch.std(cam_result) / 2
                TAc = 1/ (1 + torch.exp(ww * (cam_result - sigma)))
                TAc = TAc.unsqueeze(0)
                TAc = torch.repeat_interleave(TAc, 3, dim=0)
                # print(TAc.shape)
                newImgTensor = TAc * thisImgTensor
                newImgPreprocessed = newImgTensor.unsqueeze(0)
                newImgPreprocessed.type(torch.DoubleTensor) 
                new_gb = self.gb_model(newImgPreprocessed.float(), target_category=int(topClass[0]))
                new_gb = processGB(new_gb)
                
                if visualize:
                    gbimgs.append(new_gb.numpy())
                cost = -1 * torch.abs(F.conv2d(standardize(gb_correlate), standardize(new_gb)))
                
            correlation_pearson2 = correlation_pearson.clone() 
            correlation_pearson = correlation_pearson2 + cost
            
            
            if logs:
                print("The Pearson output loss is: ", correlation_pearson[i])
                # print("The Cross Corr output loss is: ", correlation_cross[i])
                
            
            if visualize:
                costLoss = cost
                correlations.append(costLoss.item())
                print('.')
        
        if visualize:
            # fig.canvas.draw()

            final_gb_frame = cv2.hconcat(gbimgs)
            # cv2.imwrite('./saved_figs/sampleImage_GuidedBackprop.jpg', final_gb_frame)
            # final_frame = cv2.hconcat(imgs)
            # cv2.imwrite('./saved_figs/sampleImage_GradCAM.jpg', final_frame)
            final_hmp_frame = cv2.hconcat(hmps)


            # print(np.min(final_hmp_frame), np.max(final_hmp_frame), np.median(final_hmp_frame))
            # print(np.min(final_gb_frame), np.max(final_gb_frame), np.median(final_gb_frame))
            # print(final_gb_frame.dtype, final_hmp_frame.dtype)


            def normalize(arr):
                # arr = arr / np.linalg.norm(arr)
                arr -= np.min(arr)
                arr -= np.median(arr) - 0.5
                arr_threshold = np.mean(arr) + 2*np.std(arr)
                cont = np.greater(arr, arr_threshold)
                arr[cont] = arr_threshold
                arr[arr < 0] = 0
                arr = arr / np.max(arr)
                return arr

            def gb_normalize(arr):
                im_max = np.percentile(arr, 99)
                im_min = np.min(arr)
                arr = (np.clip((arr - im_min) / (im_max - im_min), 0, 1))
                arr = arr / np.max(arr)
                # arr = np.expand_dims(arr, axis=0)
                return arr

            final_gb_frame = gb_normalize(final_gb_frame)
            final_hmp_frame = normalize(final_hmp_frame)

            # print(np.min(final_hmp_frame), np.max(final_hmp_frame), np.median(final_hmp_frame))
            # print(np.min(final_gb_frame), np.max(final_gb_frame), np.median(final_gb_frame))
            # print(final_gb_frame.dtype, final_hmp_frame.dtype)

            data = np.array(cv2.vconcat([final_hmp_frame, final_gb_frame.astype('float64')]))
            return correlations, data
            # cv2.imwrite('./saved_figs/sampleImage_GradCAM_hmp.jpg', final_hmp_frame)
            
        return correlation_pearson #/ input_tensor.shape[0]