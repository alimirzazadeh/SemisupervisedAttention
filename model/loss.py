# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:32:41 2021

@author: alimi
"""
from libs.pytorch_grad_cam.grad_cam import GradCAM
from libs.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from libs.pytorch_grad_cam.smooth_grad import VanillaGrad, SmoothGrad
from libs.pytorch_grad_cam.utils.image import deprocess_image, preprocess_image
import libs.pytorch_ssim as pytorch_ssim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F

import torch


class CAMLoss(nn.Module):
    def __init__(self, model, target_layer, use_cuda, resolutionMatch, similarityMetric, maskIntensity):
        super(CAMLoss, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.model = model
        self.cam_model = GradCAM(
            model=model, target_layer=target_layer, use_cuda=use_cuda)
        self.gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        self.vg_model = VanillaGrad(model=model, use_cuda=use_cuda)
        self.sg_model = SmoothGrad(model=model, use_cuda=use_cuda)

        self.resolutionMatch = resolutionMatch
        self.similarityMetric = similarityMetric
        self.maskIntensity = maskIntensity

    def forward(self, predict, target):
        print("hey")

    def forward(self, input_tensor,  target_category, logs=False, visualize=False):
        assert(len(input_tensor.shape) > 3)

        
        resolutionMatch = self.resolutionMatch  # Upsample CAM, Downsample GB, GB mask, Hmp Mask
        similarityMetric = self.similarityMetric  # Pearson, cross corr, SSIM
        topHowMany = 1

        correlation_pearson = torch.zeros(1)
        correlation_pearson.requires_grad = True
        correlation_pearson = correlation_pearson.to(self.device)

        if visualize:
            fig, axs = plt.subplots(3, input_tensor.shape[0])
            gbimgs = []
            imgs = []
            maskimgs = []
            gbitself = []
            hmps = []
            correlations = []

        for i in range(input_tensor.shape[0]):
            target_category = None
            for whichTargetCategory in range(topHowMany):

                thisImgTensor = input_tensor[i, :, :, :]
                thisImgTensor = thisImgTensor.to(self.device)
                thisImgPreprocessed = thisImgTensor.unsqueeze(0)

                if resolutionMatch == 1 or resolutionMatch == 2:
                    upSample = False
                else:
                    upSample = True

                if target_category != None:
                    target_category = int(topClass[whichTargetCategory])

                cam_result, topClass, targetWeight = self.cam_model(
                    input_tensor=thisImgPreprocessed, target_category=target_category, returnTarget=True, upSample=upSample)

                if target_category == None:
                    target_category = int(topClass[whichTargetCategory])

                # doesn't automatically zero gradients, so should be feeding in new img every time
                if resolutionMatch != 4:
                    gb_correlate = self.gb_model(
                        thisImgPreprocessed, target_category=target_category)
                else:
                    gb_correlate = self.sg_model(x=thisImgPreprocessed)

                def processGB(gb_correlate):
                    # gb_correlate_std = torch.std(gb_correlate)
                    # if (gb_correlate_std > 0):
                    #     gb_correlate = (gb_correlate - torch.mean(gb_correlate)) / gb_correlate_std

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
                gb_correlate = processGB(gb_correlate)
                cam_result = cam_result

                if resolutionMatch == 0:
                    firstCompare = standardize(cam_result)
                    secondCompare = standardize(gb_correlate)
                elif resolutionMatch == 1:
                    m = nn.AvgPool2d(32)
                    gb_correlate = m(reshaper(gb_correlate))[0, 0, :, :]
                    firstCompare = standardize(cam_result)
                    secondCompare = standardize(gb_correlate)
                elif resolutionMatch == 2:
                    ww = -1 * self.maskIntensity
                    sigma = torch.mean(gb_correlate) + \
                        torch.std(gb_correlate) / 2
                    TAc = 1 / (1 + torch.exp(ww * (gb_correlate - sigma)))
                    TAc = TAc.to(self.device)
                    TAc = TAc.unsqueeze(0)
                    TAc = torch.repeat_interleave(TAc, 3, dim=0)
                    newImgTensor = TAc * thisImgTensor
                    newImgPreprocessed = newImgTensor.unsqueeze(0)
                    new_cam_result = self.cam_model(
                        input_tensor=newImgPreprocessed, target_category=target_category, upSample=upSample)
                    #new_cam_result = new_grayscale_cam[0]
                    firstCompare = standardize(cam_result)
                    secondCompare = standardize(new_cam_result)
                elif resolutionMatch == 3:
                    ww = -32
                    sigma = torch.mean(cam_result) + torch.std(cam_result) / 2
                    TAc = 1 / (1 + torch.exp(ww * (cam_result - sigma)))
                    TAc = TAc.unsqueeze(0)
                    TAc = torch.repeat_interleave(TAc, 3, dim=0)
                    TAc = TAc.to(self.device)
                    newImgTensor = TAc * thisImgTensor
                    newImgPreprocessed = newImgTensor.unsqueeze(0)
                    newImgPreprocessed.type(torch.DoubleTensor)
                    new_gb = self.gb_model(
                        newImgPreprocessed.float(), target_category=target_category)
                    new_gb = processGB(new_gb)
                    firstCompare = standardize(gb_correlate)
                    secondCompare = standardize(new_gb)
                elif resolutionMatch == 4:   
                ##Uncomment this to visualize single image while testing unsup loss convergence
                # if True:
                #     plt.imshow(firstCompare.detach().numpy())
                #     plt.show()
                #     plt.imshow(secondCompare.detach().numpy())
                #     plt.show()
                #     toShow = thisImgTensor.moveaxis(0,-1).numpy()
                #     toShow -= np.min(toShow)
                #     toShow /= np.max(toShow)
                #     plt.imshow(toShow)
                #     plt.show()
                #    #print(self.model(thisImgPreprocessed))
                ###################################3


#                     print(gb_correlate.shape)
#                     print(gb_correlate.requires_grad)
                    firstCompare = standardize(cam_result)
                    secondCompare = standardize(gb_correlate)
                    # smooth_cam, _ = smooth_grad(image)

                if visualize:
                    # print(firstCompare.shape, secondCompare.shape)
                    # print(firstCompare.dtype, secondCompare.dtype)
                    def reshapeNormalize(arr):
                        arr -= np.min(arr)
                        arr /= np.max(arr)
                        arr = np.repeat(np.expand_dims(arr,axis=0),3,axis=0)
                        return np.moveaxis(arr,0,-1)
                    def normalize(arr):
                        arr -= np.min(arr)
                        arr /= np.max(arr)
                        return np.moveaxis(arr,0,-1)
                    hmps.append(reshapeNormalize(cv2.resize(firstCompare.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                    gbimgs.append(reshapeNormalize(cv2.resize(secondCompare.detach().cpu().numpy(),(256,256),interpolation=cv2.INTER_NEAREST)))
                    imgs.append(normalize(thisImgTensor.detach().cpu().numpy()))
                    maskimgs.append(normalize(newImgTensor.detach().cpu().numpy()))
                    gbitself.append(4 * reshapeNormalize(gb_correlate.detach().cpu().numpy()))

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
                        cost = torch.zeros(1)
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

                # if targetWeight < 0:
                #    targetWeight = 0
                #    print('target less than 0!')
                correlation_pearson2 = correlation_pearson.clone()
                correlation_pearson = correlation_pearson2.to(self.device) + cost.to(self.device)  # * targetWeight

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
            final_img_frame = cv2.hconcat(imgs)
            final_newimg_frame = cv2.hconcat(maskimgs)
            final_gbitself_frame = cv2.hconcat(gbitself)
            print(final_gb_frame.shape)
            print(final_hmp_frame.shape)
            print(final_img_frame.shape)
            print(final_newimg_frame.shape)
            print(final_gbitself_frame.shape)
            

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

            data = np.array(cv2.vconcat([final_img_frame.astype(
                'float64'), final_hmp_frame.astype('float64'), final_gbitself_frame.astype('float64')
                , final_newimg_frame.astype('float64'), final_gb_frame.astype('float64')]))
            return correlations, data
            # cv2.imwrite('./saved_figs/sampleImage_GradCAM_hmp.jpg', final_hmp_frame)

        return correlation_pearson  / input_tensor.shape[0]
