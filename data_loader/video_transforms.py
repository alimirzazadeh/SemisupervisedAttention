'''
From https://github.com/pytorch/vision/blob/master/references/video_classification/transforms.py

Reference code for video classification transformations.
'''

import torch
import torchvision
import random
from scipy import ndimage
import numpy as np
import PIL

def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]

def cutout(vid, i, j, h, w):
    vid[..., i:(i + h), j:(j + w)] = 0
    return vid

def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid):
    return vid.flip(dims=(-1,))


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


def pad(vid, padding, fill=0, padding_mode="constant"):
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


# Class interface

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        if th < 1 or tw < 1:
            th = int(h * th)
            tw = int(w * tw)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)

class RandomCutOut(object):
    def __init__(self, size_range):
        self.min_size = size_range[0]
        self.max_size = size_range[1]

    @staticmethod
    def get_params(vid, min_size, max_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        size = random.uniform(min_size, max_size)
        th = size
        tw = size
        
        th = int(h * th)
        tw = int(w * tw)
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.min_size, self.max_size)
        return cutout(vid, i, j, h, w)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return hflip(vid)
        return vid


class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return pad(vid, self.padding, self.fill)
    
class TwoCropTransform:
    """Create two crops of the same image.
    
    From https://github.com/HobbitLong/SupContrast/blob/master/util.py
    """
    def __init__(self, transform, transform2 = None):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, x):
        if self.transform2 is None:
            return [self.transform(x), self.transform(x)]
        else:
            return [self.transform(x), self.transform2(x)]
    
# Transforms for masking
def ignore_mask(vid):
    return vid[..., :-1]

def apply_mask(vid):
    vid[vid[..., -1] < 128, :] = 0
    return vid[..., :-1]

def mask_only(vid):
    vid_shape = vid.shape
    repeat_shape = [1 for i in range(len(vid_shape))]
    repeat_shape[-1] = vid_shape[-1] - 1
    return vid[..., -1:].repeat(tuple(repeat_shape))

class IgnoreMask(object):
    def __call__(self, vid):
        return ignore_mask(vid)

class ApplyMask(object):
    def __call__(self, vid):
        return apply_mask(vid)
    
class MaskOnly(object):
    def __call__(self, vid):
        return mask_only(vid)
    
class RandomApplyMask(object):
    def __init__(self, p = 0.5):
        self.p = p
    
    def __call__(self, vid):
        if random.random() < self.p:
            return apply_mask(vid)
        else:
            return ignore_mask(vid)
        
class GroupRandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, tensor):
        angle = random.uniform(*self.degrees)
        return torch.tensor(ndimage.rotate(tensor, -1 * angle, axes = (-1, -2), reshape = False))
    
class GroupColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    
    From https://github.com/hassony2/torch_videovision/blob/master/torchvideotransforms/video_transforms.py
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        clip_converted = [
            PIL.Image.fromarray(img.numpy())
            for img in clip
        ]
        clip = clip_converted
        if isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            
            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)
        
        return torch.from_numpy(np.array([
            np.array(img)
            for img in jittered_clip
        ]))

class IgnoreFramesOfInterest(object):
    def __call__(self, vid_and_frame):
        return vid_and_frame[0]
    
class RepeatFramesOfInterest(object):
    def __init__(self, num_repeats):
        self.num_repeats = num_repeats
    
    def __call__(self, vid_and_frame):
        vid, frame = vid_and_frame
        if frame == -1:
            return vid
        num_frames = len(vid)
        
        start = int(frame) - int(self.num_repeats / 2)
        stop = start + self.num_repeats
        
        if start < 0:
            start = 0
            stop = self.num_repeats
        if stop > num_frames:
            stop = num_frames
            start = stop - self.num_repeats
        
        for i in range(start, stop):
            vid[i] = vid[frame]
        
        return vid
    
class RepeatClipHalf(object):
    def __call__(self, vid_and_frame):
        vid, frame = vid_and_frame
        if frame == -1:
            return vid
        num_frames = len(vid)
        
        if frame < num_frames / 2:
            vid[int(num_frames / 2):] = vid[:int(num_frames / 2)]
        else:
            vid[:int(num_frames / 2)] = vid[int(num_frames / 2):]
        
        return vid