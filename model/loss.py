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


class CAMLoss(nn.Module):
    def __init__(self, model, target_layer, use_cuda, resolutionMatch, similarityMetric, maskIntensity, attentionMethod=0):
        super(CAMLoss, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.model = model
        self.cam_model = GradCAM(
            model=model, target_layer=target_layer, use_cuda=use_cuda)
        self.gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        self.vg_model = VanillaGrad(model=model, use_cuda=use_cuda)
        self.sg_model = SmoothGrad(model=model, use_cuda=use_cuda)
        self.ig_model = IntegratedGradientsModel(model=model, use_cuda=use_cuda)

        self.resolutionMatch = resolutionMatch
        self.similarityMetric = similarityMetric
        self.maskIntensity = maskIntensity
        self.attentionMethod = attentionMethod

    def forward(self, input_tensor,  target_category, logs=False, visualize=False):
        assert(len(input_tensor.shape) > 3)
        torch.cuda.empty_cache()
        resolutionMatch = self.resolutionMatch  # Deprecated on Max branch
        similarityMetric = self.similarityMetric  # Pearson, cross corr, SSIM
        attentionMethod = self.attentionMethod # GB/GC, GB/IG, GC/IG
        topHowMany = 1

        correlation_pearson = torch.zeros(1)
        correlation_pearson.requires_grad = True
        correlation_pearson = correlation_pearson.to(self.device)

        if visualize:
            fig, axs = plt.subplots(3, input_tensor.shape[0])
            firsts = []
            seconds = []
            imgs = []
            maskimgs = []
            correlates = []
            correlations = []
        print('Begin: GPU Usage: ', torch.cuda.memory_allocated(0) / 1e9)
        # print(torch.torch.cuda.memory_summary())
        for i in range(input_tensor.shape[0]):
            target_category = None
            for whichTargetCategory in range(topHowMany):
                thisImgTensor = input_tensor[i, :, :, :]
                thisImgTensor = thisImgTensor.to(self.device)
                thisImgPreprocessed = thisImgTensor.unsqueeze(0)
                
                if target_category != None:
                    target_category = int(topClass[whichTargetCategory])

                def processGB(gb_correlate):
                    gb_correlate = torch.abs(gb_correlate)
                    gb_correlate = torch.sum(gb_correlate, axis=2)
                    return gb_correlate

                def standardize(arr):
                    return (arr - torch.mean(arr))/torch.std(arr)

                def sigmoidIt(arr):
                    m = nn.Sigmoid()
                    return m(arr)

                def reshaper(arr):
                    return arr.unsqueeze(0).unsqueeze(0).float()

                if attentionMethod == 0:
                    print('Before: GPU Usage in CAM: ', torch.cuda.memory_allocated(0) / 1e9)
                    cam_result, topClass, targetWeight = self.cam_model(
                        input_tensor=thisImgPreprocessed, target_category=target_category, returnTarget=True, upSample=False)
                    print('After: GPU Usage in CAM: ', torch.cuda.memory_allocated(0) / 1e9)
                    if target_category == None:
                        target_category = int(topClass[whichTargetCategory])
                    print('Before: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
                    correlate = self.gb_model(
                        thisImgPreprocessed, target_category=target_category)
                    print('After: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
                    correlate = processGB(correlate)
                    ww = -1 * self.maskIntensity
                    sigma = torch.mean(correlate) + \
                        torch.std(correlate) / 2
                    TAc = 1 / (1 + torch.exp(ww * (correlate - sigma)))
                    TAc = TAc.to(self.device)
                    TAc = TAc.unsqueeze(0)
                    TAc = torch.repeat_interleave(TAc, 3, dim=0)
                    newImgTensor = TAc * thisImgTensor
                    newImgPreprocessed = newImgTensor.unsqueeze(0)
                    new_cam_result = self.cam_model(
                        input_tensor=newImgPreprocessed, target_category=target_category, upSample=False)
                    firstCompare = standardize(cam_result)
                    secondCompare = standardize(new_cam_result)
                elif attentionMethod == 1:
                    print('Before: GPU Usage in IG: ', torch.cuda.memory_allocated(0) / 1e9)
                    ig_correlate, topClass, targetWeight = self.ig_model(
                        thisImgPreprocessed, target_category=target_category)
                    print('After: GPU Usage in IG: ', torch.cuda.memory_allocated(0) / 1e9) 
                    if target_category == None:
                        target_category = int(topClass[whichTargetCategory])
                    print('Before: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
                    gb_correlate = self.gb_model(
                        thisImgPreprocessed, target_category=target_category)
                    print('After: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
                    gb_correlate = processGB(gb_correlate)
                    ig_correlate = processGB(ig_correlate)
                    newImgTensor = thisImgTensor
                    firstCompare = standardize(gb_correlate)
                    secondCompare = standardize(ig_correlate)
                    correlate = torch.zeros_like(gb_correlate)
                elif attentionMethod == 2:
                    cam_result, topClass, targetWeight = self.cam_model(
                        input_tensor=thisImgPreprocessed, target_category=target_category, returnTarget=True, upSample=False)
                    if target_category == None:
                        target_category = int(topClass[whichTargetCategory])
                    correlate, _, _ = self.ig_model(
                        thisImgPreprocessed, target_category=target_category)
                    correlate = processGB(correlate)
                    ww = -1 * self.maskIntensity
                    sigma = torch.mean(correlate) + \
                        torch.std(correlate) / 2
                    TAc = 1 / (1 + torch.exp(ww * (correlate - sigma)))
                    TAc = TAc.to(self.device)
                    TAc = TAc.unsqueeze(0)
                    TAc = torch.repeat_interleave(TAc, 3, dim=0)
                    newImgTensor = TAc * thisImgTensor
                    newImgPreprocessed = newImgTensor.unsqueeze(0)
                    new_cam_result = self.cam_model(
                        input_tensor=newImgPreprocessed, target_category=target_category, upSample=False)
                    firstCompare = standardize(cam_result)
                    secondCompare = standardize(new_cam_result)

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
                    firsts.append(reshapeNormalize(cv2.resize(firstCompare.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                    seconds.append(reshapeNormalize(cv2.resize(secondCompare.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                    imgs.append(normalize(thisImgTensor.detach().cpu().numpy()))
                    maskimgs.append(normalize(newImgTensor.detach().cpu().numpy()))
                    correlates.append(4 * reshapeNormalize(correlate.detach().cpu().numpy()))

                if similarityMetric == 0:
                    firstCompare = sigmoidIt(firstCompare)
                    secondCompare = sigmoidIt(secondCompare)
                    vx = firstCompare - torch.mean(firstCompare)
                    vy = secondCompare - torch.mean(secondCompare)
                    denominator = torch.sqrt(
                        torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
                    if denominator > 0:
                        cost = -1 * torch.sum(vx * vy) / denominator
                    else:
                        cost = torch.zeros(1).cuda()
                elif similarityMetric == 1:
                    firstCompare = sigmoidIt(firstCompare)
                    secondCompare = sigmoidIt(secondCompare)
                    cost = -1 * \
                        torch.abs(F.conv2d(reshaper(firstCompare),
                                           reshaper(secondCompare)))
                    cost = cost.squeeze()
                elif similarityMetric == 2:
                    ssim_loss = pytorch_ssim.SSIM()
                    cost = 1 - ssim_loss(reshaper(firstCompare),
                                         reshaper(secondCompare))
                correlation_pearson2 = correlation_pearson.clone()
                correlation_pearson = correlation_pearson2 + cost  # * targetWeight

            if logs:
                print("The Pearson output loss is: ", correlation_pearson[i])

            if visualize:
                costLoss = cost
                correlations.append(costLoss.item())
                print('.')
        
        print('End: GPU Usage: ', torch.cuda.memory_allocated(0) / 1e9)

        if visualize:
            final_firsts_frame = cv2.hconcat(firsts) # Normal Grad Cam image
            final_seconds_frame = cv2.hconcat(seconds) # Grad Cam with Guided Backprop/Integrated Gradients masked image
            final_img_frame = cv2.hconcat(imgs) # Original input image
            final_newimg_frame = cv2.hconcat(maskimgs) # Guided backprop/Integrated Gradient masked image
            final_correlates_frame = cv2.hconcat(correlates) # Guided backprop/Integrated Gradient
            

            def normalize(arr):
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
                return arr

            final_seconds_frame = gb_normalize(final_seconds_frame)
            final_firsts_frame = normalize(final_firsts_frame)
            data = np.array(cv2.vconcat([final_img_frame.astype(
                'float64'), final_firsts_frame.astype('float64'), final_correlates_frame.astype('float64')
                , final_newimg_frame.astype('float64'), final_seconds_frame.astype('float64')]))
            return correlations, data

        return correlation_pearson  / input_tensor.shape[0]
