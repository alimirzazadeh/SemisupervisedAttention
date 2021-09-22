import cv2
import numpy as np
import torch
import ttach as tta
from libs.pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from libs.pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import json
from ipdb import set_trace as bp

class BaseCAM:
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model#.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)
        # f = open("Data/imagenet_class_index.json",)
        # class_idx = json.load(f)
        # self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            try:
                loss = loss + output[i, target_category[i]]
            except:
                loss = loss + output.logits[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = torch.sum(weighted_activations, axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False, returnTarget=False, upSample=True):

        if self.cuda:
            input_tensor = input_tensor.cuda()


        input_tensor = input_tensor.requires_grad_(True)
        
        
        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            # print("Category shapes: ", output.cpu().data.numpy().shape)
            try:
                out_output = output.data.cpu().numpy()
            except:
                out_output = output.logits.cpu().detach().numpy()
            target_category = np.argmax(out_output, axis=-1)
            # print("Target Category: ", target_category)
            # print(target_category)
            # print("Labels: ", self.idx2label[target_category[0]])
            # print(target_category)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        
        
        torch.autograd.grad(loss,input_tensor,create_graph=True)
        # loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1]
        grads = self.activations_and_grads.gradients[-1]

        cam = self.get_cam_image(input_tensor, target_category, 
            activations, grads, eigen_smooth)
        # print("cam shape!")
        # print(cam)
        cam = torch.max(cam, 0)
        # print(cam)
        result = []
        img = cam[0]
        # print(input_tensor.shape[-2:][::-1])
        # img = cv2.resize(img, input_tensor.shape[-2:][::-1])
        # print(img.shape)
        img = img.reshape((1,1,img.shape[0],img.shape[1]))
        if upSample:
            img = torch.nn.functional.upsample_bilinear(img.double(),size=list(input_tensor.shape[-2:][::-1]))
        # print(img.shape)
        # img = img - torch.min(img)
        # img = img / torch.max(img)
        result = img[0,0,:,:]
        # result = np.float32(result)
        if returnTarget:
            target_categories = out_output.argsort()[0][-3:][::-1]
            target_weight = out_output[:,target_category].squeeze()
            target_weight[target_weight < 0] = 0
            return result, target_categories, target_weight
        else:
            return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False,
                 returnTarget=False,
                 upSample=True):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                target_category, eigen_smooth)

        return self.forward(input_tensor,
            target_category, eigen_smooth, returnTarget=returnTarget, upSample=upSample)