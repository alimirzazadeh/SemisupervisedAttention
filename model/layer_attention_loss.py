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

def isLayerWiseAttention(attentionMethod):
    return attentionMethod == 3 or \
        attentionMethod == 4 or \
        attentionMethod == 5 or \
        attentionMethod == 6

class LayerAttentionLoss(nn.Module):
    sigma_factor = 0.55

    def __init__(self, model, target_layers, use_cuda, maskIntensity, theta, attentionMethod):
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
        self.attentionMethod = attentionMethod


    def forward(self, input_tensor,  labels, logs=False, visualize=False):
        assert(len(input_tensor.shape) > 3)
        assert(visualize == False)
        maskIntensity = self.maskIntensity
        theta = self.theta
        loss = 0.0
        if visualize:
            outers = []
            inners = []
            imgs = []
            losses = []
        
        for i in range(input_tensor.shape[0]):
            label = int(labels[i])

            thisImgTensor = input_tensor[i, :, :, :]
            thisImgTensor = thisImgTensor.to(self.device)
            thisImgPreprocessed = thisImgTensor.unsqueeze(0)

            def standardize(arr):
                return (arr - torch.mean(arr))/torch.std(arr)
        
            AT, topClass, targetWeight = self.outer_gradcam(
                input_tensor=thisImgPreprocessed, target_category=label, upSample=False, returnTarget=True)
            Ain = self.inner_gradcam(
                input_tensor=thisImgPreprocessed, target_category=label, upSample=False)
            # apply ReLU
            AT_relu = torch.nn.ReLU()(AT)
            Ain_relu = torch.nn.ReLU()(Ain)
            # compute mask
            ww = -1 * maskIntensity

            At_min = AT_relu.min().detach()
            At_max = AT_relu.max().detach()
            Ain_min = Ain_relu.min().detach()
            Ain_max = Ain_relu.max().detach()
            scaled_At = (AT_relu - At_min + torch.finfo(torch.float32).eps)/(At_max - At_min + torch.finfo(torch.float32).eps)
            scaled_Ain = (Ain_relu - Ain_min + torch.finfo(torch.float32).eps)/(Ain_max - Ain_min + torch.finfo(torch.float32).eps)
            sigma = self.sigma_factor * At_max
            # sigma = torch.mean(outer_cam_result) + \
            #             torch.std(outer_cam_result) / 2
            mask = torch.finfo(torch.float32).eps + (1 / (1 + torch.exp(ww * (scaled_At - sigma))))
            """
            attentionMethod
            3: florians consistency loss
            4: original layer attention consistency loss
            5: original layer attention consistency loss plus layer separation loss
            6: florians consistency loss plus layer separation loss
            """
            currLoss = 0.0
            if self.attentionMethod == 3 or self.attentionMethod == 6:
                # florian version
                currLoss = - 2*(torch.sum(scaled_Ain * mask) + torch.finfo(torch.float32).eps) / (torch.sum(scaled_Ain) + torch.sum(mask) + torch.finfo(torch.float32).eps)
            elif self.attentionMethod == 4 or self.attentionMethod == 5:
                currLoss = theta - (torch.sum(scaled_Ain * mask) + torch.finfo(torch.float32).eps) / (torch.sum(scaled_Ain) + torch.finfo(torch.float32).eps)
            
            # separation loss
            if self.attentionMethod == 5 or self.attentionMethod == 6: 
                conf_ind = int(topClass[0]) if int(topClass[0]) != label else int(topClass[1])
                Aconf = self.outer_gradcam(
                    input_tensor=thisImgPreprocessed, target_category=conf_ind, upSample=False)
                minimum = torch.minimum(AT, Aconf) # Should this AT?
                numerator = torch.sum(minimum * mask) + torch.finfo(torch.float32).eps
                denominator = torch.sum(Aconf + AT) + torch.finfo(torch.float32).eps
                currLoss += 2 * numerator / denominator
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
                outers.append(reshapeNormalize(cv2.resize(AT.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                inners.append(reshapeNormalize(cv2.resize(Ain.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                imgs.append(normalize(thisImgTensor.detach().cpu().numpy()))
                losses.append(currLoss.item())
                
            loss = loss + currLoss

        loss /= input_tensor.shape[0]

        if logs:
            print("The Layer Attention output loss of the batch is: ", loss )
        
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
            return losses, data
        
        return loss
