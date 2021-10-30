# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:32:41 2021

@author: alimi
"""
from libs.pytorch_grad_cam.grad_cam import GradCAM
from libs.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from libs.pytorch_grad_cam.smooth_grad import VanillaGrad, SmoothGrad
from libs.pytorch_grad_cam.integrated_gradients import IntegratedGradientsModel
from libs.pytorch_grad_cam.utils.image import deprocess_image, preprocess_image
import libs.pytorch_ssim as pytorch_ssim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F

import torch


class LayerAttentionLoss(nn.Module):
    sigma_factor = 0.55

    def __init__(self, model, target_layers, use_cuda, maskIntensity, theta):
        super(LayerAttentionLoss, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.model = model
        assert(len(target_layers) >= 2)
        self.outer_gradcam = GradCAM(
            model=model, target_layer=target_layers[0], use_cuda=use_cuda)
        self.inner_gradcam = GradCAM(
            model=model, target_layer=target_layers[1], use_cuda=use_cuda)
        self.maskIntensity = maskIntensity
        self.theta = theta


    def forward(self, input_tensor,  labels, logs=False, visualize=False):
        assert(len(input_tensor.shape) > 3)
        
        maskIntensity = self.maskIntensity
        theta = self.theta
        l_ac = 0
        if visualize:
            outers = []
            inners = []
            imgs = []

        for i in range(input_tensor.shape[0]):
            label = int(labels[i])

            thisImgTensor = input_tensor[i, :, :, :]
            thisImgTensor = thisImgTensor.to(self.device)
            thisImgPreprocessed = thisImgTensor.unsqueeze(0)

            def standardize(arr):
                return (arr - torch.mean(arr))/torch.std(arr)
        
            outer_cam_result = self.outer_gradcam(
                input_tensor=thisImgPreprocessed, target_category=label, upSample=False)
            inner_cam_result = self.inner_gradcam(
                input_tensor=thisImgPreprocessed, target_category=label, upSample=False)
            ww = -1 * maskIntensity
            sigma = self.sigma_factor * torch.max(outer_cam_result)
            # sigma = torch.mean(outer_cam_result) + \
            #             torch.std(outer_cam_result) / 2
            TAc = 1 / (1 + torch.exp(ww * (outer_cam_result - sigma)))
            l_ac += theta - torch.sum(inner_cam_result * TAc) / torch.sum(inner_cam_result)
        
            if visualize:
                def reshapeNormalize(arr):
                    arr -= np.min(arr)
                    arr /= np.max(arr)
                    arr = np.repeat(np.expand_dims(arr,axis=0),3,axis=0)
                    return np.moveaxis(arr,0,-1)
                def normalize(arr):
                    arr -= np.min(arr)
                    arr /= np.max(arr)
                    return np.moveaxis(arr,0,-1)
                outers.append(reshapeNormalize(cv2.resize(outer_cam_result.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                inners.append(reshapeNormalize(cv2.resize(inner_cam_result.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                imgs.append(normalize(thisImgTensor.detach().cpu().numpy()))

        l_ac /= input_tensor.shape[0]

        if logs:
            print("The Layer Attention Consistency Loss (Lac) output loss of the batch is: ", l_ac )
        
        if visualize:
            final_outers = cv2.hconcat(outers) # Last layer gradcam
            final_inners = cv2.hconcat(inners) # second last layer gradcam
            final_img = cv2.hconcat(imgs) # Original input image
            

            def normalize(arr):
                arr -= np.min(arr)
                arr -= np.median(arr) - 0.5
                arr_threshold = np.mean(arr) + 2*np.std(arr)
                cont = np.greater(arr, arr_threshold)
                arr[cont] = arr_threshold
                arr[arr < 0] = 0
                arr = arr / np.max(arr)
                return arr

            final_inners = normalize(final_inners)
            final_outers = normalize(final_outers)
            data = np.array(cv2.vconcat([final_img.astype(
                'float64'), final_outers.astype('float64'), final_inners.astype('float64')]))
            return correlations, data

        return l_ac
