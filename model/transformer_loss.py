# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from libs.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel

def isTransformer(attentionMethod):
    return attentionMethod == 7

class TransformerLoss(nn.Module):
    def __init__(self, model, use_cuda):
        super(TransformerLoss, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.model = model
        self.gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)

    def forward(self, input_tensor, target_category, logs=False, visualize=False):
        for i in range(input_tensor.shape[0]):
            thisImgTensor = input_tensor[i, :, :, :]
            thisImgTensor = thisImgTensor.to(self.device)
            thisImgPreprocessed = thisImgTensor.unsqueeze(0)

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

            # Use the guided backprop class, but this is is equivalent to backprop since no relus in ViT
            backprop = self.gb_model(thisImgPreprocessed)
            backprop = standardize(processGB(backprop))

        return torch.zeros(1)