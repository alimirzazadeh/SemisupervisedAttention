import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

class IntegratedGradientsModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()
    
    def forward(self, input_img):
        inputs = input_img.cuda() if self.cuda else input_img
        # print('Before: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
        logits = self.model(inputs)
        # print('After: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
        return logits

    def interpolate_image(self, baseline, input_img, alphas):
        baselines = baseline.repeat(alphas.shape[0], 1, 1, 1)
        input_imgs = input_img.repeat(alphas.shape[0], 1, 1, 1)
        deltas = input_imgs - baselines
        return baselines +  alphas.view(alphas.shape[0], 1, 1, 1) * deltas

    def compute_gradients(self, input_imgs, target_category):
        input_imgs = input_imgs.requires_grad_(True)
        logits = self.forward(input_imgs)
        self.model.zero_grad()
        out_output = logits.data.cpu().numpy()
        if target_category is None:
            target_category = logits.mean(dim=1).argmax()
        probs = F.softmax(logits, dim=0)[:, target_category]
        output = torch.autograd.grad(outputs=probs, inputs=input_imgs, create_graph=True, grad_outputs=torch.ones_like(probs))
        target_categories = out_output.argsort()[0][-3:][::-1]
        target_weight = out_output[:,target_category].squeeze()
        target_weight[target_weight < 0] = 0
        return output[0], target_categories, target_weight

    def __call__(self, input_img, target_category=None, m_steps=5, returnTarget=False):
        input_img = input_img.cpu()
        # 1. Generate alphas.
        alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps+1)
        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        baseline = torch.zeros(*(input_img.shape), dtype=torch.float32)
        # 2. Generate interpolated inputs between baseline and input.
        interpolated_images = self.interpolate_image(baseline, input_img, alphas)
        # 3. Compute gradients between model outputs and interpolated inputs.
        gradients, target_categories, target_weight = self.compute_gradients(interpolated_images, target_category)
        # 4. Integral approximation through averaging gradients.
        avg_gradient = torch.mean(gradients, dim=0)
        # 5. Scale integrated gradients with respect to input.
        integrated_gradient = (input_img - baseline) * avg_gradient
        integrated_gradient = integrated_gradient.squeeze(0).permute(1, 2, 0)
        if self.cuda:
            integrated_gradient = integrated_gradient.cuda()
        if returnTarget:  
            return integrated_gradient, target_categories, target_weight
        return integrated_gradient

# import torch
# import torch.nn.functional as F
# import numpy as np
# from torchvision.utils import save_image

# class IntegratedGradientsModel:
#     def __init__(self, model, use_cuda):
#         self.model = model
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = self.model.cuda()
    
#     def forward(self, input_img):
#         inputs = input_img.cuda() if self.cuda else input_img
#         print('Before: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
#         logits = self.model(inputs)
#         print('After: GPU Usage in GB: ', torch.cuda.memory_allocated(0) / 1e9)
#         return logits

#     def interpolate_image(self, baseline, input_img, alphas):
#         baselines = baseline.repeat(alphas.shape[0], 1, 1, 1)
#         input_imgs = input_img.repeat(alphas.shape[0], 1, 1, 1)
#         deltas = input_imgs - baselines
#         return baselines +  alphas.view(alphas.shape[0], 1, 1, 1) * deltas

#     def compute_gradients(self, input_imgs, target_category):
#         input_imgs = input_imgs.requires_grad_(True)
#         logits = self.forward(input_imgs)
#         self.model.zero_grad()
#         if target_category is None:
#             out_output = logits.data.cpu().numpy()
#             target_category = logits.mean(dim=1).argmax()
#         probs = F.softmax(logits, dim=0)[:, target_category]
#         output = torch.autograd.grad(outputs=probs, inputs=input_imgs, create_graph=True, grad_outputs=torch.ones_like(probs))
#         target_categories = out_output.argsort()[0][-3:][::-1]
#         target_weight = out_output[:,target_category].squeeze()
#         target_weight[target_weight < 0] = 0
#         return output[0], target_categories, target_weight

#     def __call__(self, input_img, target_category=None, m_steps=5):
#         if self.cuda:
#             input_img = input_img.cuda()
#         # 1. Generate alphas.
#         alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps+1, device=torch.device('cuda:0'))
#         # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
#         baseline = torch.zeros(*(input_img.shape), dtype=torch.float32, device=torch.device('cuda:0'))
#         # 2. Generate interpolated inputs between baseline and input.
#         interpolated_images = self.interpolate_image(baseline, input_img, alphas)
#         # 3. Compute gradients between model outputs and interpolated inputs.
#         gradients, target_categories, target_weight = self.compute_gradients(interpolated_images, target_category)
#         # 4. Integral approximation through averaging gradients.
#         avg_gradient = torch.mean(gradients, dim=0)
#         # 5. Scale integrated gradients with respect to input.
#         integrated_gradient = (input_img - baseline) * avg_gradient
#         integrated_gradient = integrated_gradient.squeeze(0).permute(1, 2, 0)
#         if self.cuda:
#             integrated_gradient = integrated_gradient.cuda()
#         return integrated_gradient, target_categories, target_weight

