import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class SmoothGrad(VanillaGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=25, magnitude=True):
        super(SmoothGrad, self).__init__(pretrained_model, cuda)
        """
        self.pretrained_model = pretrained_model
        self.features = pretrained_model.features
        self.cuda = cuda
        self.pretrained_model.eval()
        """
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

    def __call__(self, x, index=None):
        x = x.data.cpu().numpy()
        stdev = self.stdev_spread * (torch.max(x) - torch.min(x))
        total_gradients = torch.zeros_like(x)
        for i in range(self.n_samples):
            noise = torch.random.normal(
                0, stdev, x.shape).astype(torch.float32)
            x_plus_noise = x + noise
            if self.cuda:
                x_plus_noise = Variable(torch.from_numpy(
                    x_plus_noise).cuda(), requires_grad=True)
            else:
                x_plus_noise = Variable(torch.from_numpy(
                    x_plus_noise), requires_grad=True)
            output = self.pretrained_model(x_plus_noise)

            if index is None:
                index = torch.argmax(output.data.cpu().numpy())

            one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
            one_hot[0][index] = 1
            if self.cuda:
                one_hot = Variable(torch.from_numpy(
                    one_hot).cuda(), requires_grad=True)
            else:
                one_hot = Variable(torch.from_numpy(
                    one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward(retain_variables=True)

            grad = x_plus_noise.grad.data.cpu().numpy()

            if self.magnitutde:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
            # if self.visdom:

        avg_gradients = total_gradients[0, :, :, :] / self.n_samples

        return avg_gradients
