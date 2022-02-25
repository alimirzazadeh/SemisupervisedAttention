import cv2
import numpy as np
import torch

class MaskedAttention:
    def __init__(self, 
                 model, 
                 use_cuda=False):
        self.model = model#.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def forward(self, input_tensor, target_category=None, returnTarget=False, patch_size=28):
        
        num_patches_x = input_tensor.shape[-2] // patch_size
        num_patches_y = input_tensor.shape[-1] // patch_size
        masked = torch.zeros(num_patches_x * num_patches_y, *(input_tensor.shape[1:]))

        if input_tensor.shape[0] > 1:
            print("Warning: Batch size is greater than 1 for masking")

        for i in range(0, num_patches_x * patch_size, patch_size):
            for j in range(0, num_patches_y * patch_size, patch_size):
                masked[i * num_patches_y // patch_size + j // patch_size,:,i:i+patch_size,j:j+patch_size] = input_tensor[0,:,i:i+patch_size,j:j+patch_size]
        
        masked = masked.requires_grad_(True)

        if self.cuda:
            masked = masked.cuda()

        logits = self.model(masked)

        # if type(target_category) is int:
        #     target_category = [target_category] * input_tensor.size(0)
        logits = logits.data.cpu()
        if target_category is None:
            target_category = int(torch.argmax(torch.mean(logits, dim=-2), dim=-1))
        assert(isinstance(target_category, int))
        # else:
        #     assert(len(target_category) == input_tensor.size(0))

        attn_map = torch.exp(logits[:,target_category])

        if returnTarget:
            return attn_map, target_category
        else:
            return attn_map

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 returnTarget=False, 
                 patch_size=28):
        return self.forward(input_tensor, target_category, returnTarget=returnTarget, patch_size=patch_size)