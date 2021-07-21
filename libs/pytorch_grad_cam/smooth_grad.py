import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class VanillaGrad:

    def __init__(self, model=None, use_cuda=False):
        self.pretrained_model = model
        self.cuda = use_cuda
        # self.pretrained_model.eval()

    def __call__(self, x, index=None):
        output = self.pretrained_model(x)

        if index is None:
            index = torch.argmax(output)

        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
        one_hot[0][index] = 1
        one_hot = torch.sum(one_hot * output)
        grad = torch.autograd.grad(one_hot, x, create_graph=True)
        grad = grad[0][0, :, :, :]
        grad = torch.moveaxis(grad, 0, 2)
        return grad


class SmoothGrad(VanillaGrad):

    def __init__(self, model, use_cuda=False, stdev_spread=0.15,
                 n_samples=25, magnitude=True):
        super(SmoothGrad, self).__init__(model, use_cuda)
        self.model = model
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

    def __call__(self, x, index=None):
        stdev = self.stdev_spread * (torch.max(x) - torch.min(x))
        total_gradients = torch.zeros_like(x)
        for i in range(self.n_samples):
            noise = torch.normal(0, stdev)
            x_plus_noise = x + noise
            output = self.model(x_plus_noise)

            if index is None:
                index = torch.argmax(output)

            one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
            one_hot[0][index] = 1
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            # one_hot.backward(retain_variables=True)
            # grad = x_plus_noise.grad
            grad = torch.autograd.grad(
                one_hot, x_plus_noise, create_graph=True)

            # print(grad[0][0, :, :, :])
            if self.magnitude:
                total_gradients += (grad[0] * grad[0])
            else:
                total_gradients += grad
            # if self.visdom:

        avg_gradients = total_gradients[0, :, :, :] / self.n_samples
        avg_gradients = torch.moveaxis(avg_gradients, 0, 2)

        return avg_gradients


def show_as_gray_image(img, percentile=99):
    img_2d = torch.sum(img, axis=2)
    span = abs(torch.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = torch.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    return torch.uint8(img_2d*255)
