import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import os
import pims
from PIL import Image
from data_loader.video_transforms import *
import numpy as np
import warnings
import random

video_transform = transforms.Compose([
    ToFloatTensorInZeroOne(),
    Resize((112, 112)),
    Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
])

class VideoClipDataset(Dataset):
    def __init__(self, id_to_video_files, segments, labels,
                frames_path = None, transform = video_transform,
                crop_list = None, include_mask = False, masks_path = None,
                frames_of_interest = None,
                transform_frames_of_interest = False,
                extra_frames = 0,
                extra_tasks = [],
                extra_task_transform = video_transform,
                extra_args = {},):
        self.id_to_video_files = id_to_video_files
        self.segments = segments
        self.labels = labels
        self.transform = transform
        self.frames_path = frames_path
        self.crop_list = crop_list # left, top, right, bottom in percent
        self.include_mask = include_mask
        self.masks_path = masks_path
        self.frames_of_interest = frames_of_interest
        self.transform_frames_of_interest = transform_frames_of_interest
        self.extra_frames = extra_frames
        self.extra_tasks = extra_tasks
        self.extra_task_transform = extra_task_transform
        self.extra_args = extra_args
        
        if self.include_mask and self.masks_path is None:
            raise ValueError('If include_mask is True, must specify masks_path')
        
    def __len__(self):
        return len(self.segments)

    def get_images(self, video_id, frame_list, crop):
        images = []
        
        height = 0
        width = 0
        
        pims_video = None

        for i, f in enumerate(frame_list):
                frame_path = os.path.join(self.frames_path.format(video_id), f'{f}.jpg')
                if os.path.exists(frame_path):
                    try:
                        img = Image.open(frame_path)
                    except:
                        frame_path = os.path.join('/self-supervised/sampling/self-supervised-video/tmp_data', f'{f}.jpg')
                        print(frame_path + " corrupted using adhoc path")
                        img = Image.open(frame_path)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if pims_video is None:
                            pims_video = pims.Video(self.id_to_video_files[video_id])
                        img = Image.fromarray(pims_video[f])
                cur_height, cur_width = img.height, img.width
                if self.include_mask:
                    mask_path = os.path.join(self.masks_path.format(video_id), f'{f}.jpg')
                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path)
                    else:
                        mask = Image.fromarray(np.full((cur_height, cur_width), 255, np.uint8))
                if crop is not None:
                    if len(crop) == 4 and type(crop[0]) != list:
                        cur_crop = crop
                    else:
                        cur_crop = crop[i]
                    left, top, right, bottom = cur_crop
                    img = img.crop(
                        (left * cur_width, top * cur_height, right * cur_width, bottom * cur_height)
                    )
                    if self.include_mask:
                        mask = mask.crop(
                            (left * cur_width, top * cur_height, right * cur_width, bottom * cur_height)
                        )
                if height == 0 or width == 0:
                    height, width = img.height, img.width
                else:
                    img = img.resize((width, height))
                    if self.include_mask:
                        mask = mask.resize((width, height))
                if self.include_mask:
                    images.append(np.dstack([np.array(img), np.array(mask)]))
                else:
                    images.append(np.array(img))

        if pims_video is not None:
            pims_video.close()
        
        return images, height, width
    
    def __getitem__(self, idx):
        video_id, segment = self.segments[idx]
        label = self.labels[idx]
        if self.crop_list is not None:
            crop = self.crop_list[idx]
        else:
            crop = None
        
        if len(segment) == 2:
            frame_list = list(range(segment[0] - self.extra_frames, segment[1] + self.extra_frames))
        else:
            frame_list = list(range(segment[0] - self.extra_frames * segment[2],
                                    segment[1] + self.extra_frames * segment[2],
                                    segment[2]))
        
        images, height, width = self.get_images(
            video_id,
            frame_list,
            crop
        )
        
        if self.transform_frames_of_interest:
            ret = [self.transform([
                torch.from_numpy(np.array(images)), self.frames_of_interest[idx]
            ]), label]
        else:
            ret = [self.transform(torch.from_numpy(np.array(images))), label]
        for task in self.extra_tasks:
            if task == 'shuffle':
                if torch.rand(1)[0] > 0.5:
                    random.shuffle(images)
                    label = 1
                else:
                    label = 0
                ret.append(self.extra_task_transform(torch.from_numpy(np.array(images))))
                ret.append(label)
            if task == 'speed':
                rates = [0.5, 1, 2, 4]
                rate_idx = random.randrange(len(rates))
                rate = rates[rate_idx]
                if rate == 1:
                    images = np.array(images)
                elif rate > 1:
                    length = segment[1] - segment[0] + 2 * self.extra_frames
                    images, _, __ = self.get_images(
                        video_id,
                        range(segment[0] - self.extra_frames,
                              segment[0] - self.extra_frames + rate * length,
                              rate),
                        crop
                    )
                    images = np.array(images)
                else:
                    length = segment[1] - segment[0] + 2 * self.extra_frames
                    images, _, __ = self.get_images(
                        video_id,
                        range(segment[0] - self.extra_frames,
                              int(segment[0] - self.extra_frames + rate * length)),
                        crop
                    )
                    images = np.array([
                        img
                        for img in images
                        for i in range(2)
                    ])
                ret.append(self.extra_task_transform(torch.from_numpy(images)))
                ret.append(rate_idx)
            if task == 'rotate':
                degrees = [0, 90, 180, 270]
                degree_idx = random.randrange(len(degrees))
                degree = degrees[degree_idx]
                images, _, __ = self.get_images(
                    video_id,
                    range(segment[0] - self.extra_frames, segment[1] + self.extra_frames),
                    crop,
                )
                images = np.array(images)
                
                transform = transforms.Compose([
                    self.extra_task_transform,
                    GroupRandomRotation((degree - 3, degree + 3))
                ])
                
                images_torch = transform(torch.from_numpy(images))
                ret.append(images_torch)
                ret.append(degree_idx)
            if task.startswith('top_bottom_crop'):
                load_other_clip = random.random() > 0.5
                if load_other_clip:
                    other_clip_idx = idx + (random.randrange(100) + 50) * (
                        1 if random.random() > 0.5 else -1
                    )
                    other_clip_idx = min(other_clip_idx, len(self.segments) - 1)
                    other_clip_idx = max(other_clip_idx, 0)
#                     print(idx, other_clip_idx)
                    other_video_id, other_seg = self.segments[other_clip_idx]
                    other_clip_images, _, __ = self.get_images(
                        other_video_id,
                        range(other_seg[0] - self.extra_frames, other_seg[1] + self.extra_frames),
                        crop,
                    )
                    other_clip_images = np.array(other_clip_images)
                    images = np.array(images)
                    image_height = images.shape[1]
                    if random.random() > 0.5:
                        images[
                            :, :int(image_height / 2), ...
                        ] = other_clip_images[
                            :, :int(image_height / 2), ...
                        ]
                    else:
                        images[
                            :, int(image_height / 2):, ...
                        ] = other_clip_images[
                            :, int(image_height / 2):, ...
                        ]
                else:
                    images = np.array(images)
                
                class BlackOutEdges(object):
                    def __call__(self, tensor):
                        height = tensor.shape[2]
                        width = tensor.shape[3]
                        
                        tensor[..., :int(width * .1)] = 0
                        tensor[..., int(width * .9):] = 0
                        tensor[..., int(height * .4):int(height * .6), :] = 0
                        
                        return tensor

                transform = transforms.Compose([
                    self.extra_task_transform,
                    BlackOutEdges()
                ])
                if task == 'top_bottom_crop':
                    ret.append(transform(torch.from_numpy(images)))
                elif task == 'top_bottom_crop_no_bars':
                    ret.append(self.extra_task_transform(torch.from_numpy(images)))
                ret.append(int(load_other_clip))
            if task == 'ball_detection':
                ret.append(self.extra_task_transform(torch.from_numpy(np.array(images))))
                ball_idx = idx + self.extra_args['ball_detection_offset']
                ball_idx = min(ball_idx, len(self.segments) - 1)
                ball_idx = max(ball_idx, 0)
                ret.append(self.extra_args['ball_quads'][ball_idx] + 1)
            if task == 'contact_point':
                ret += [self.extra_task_transform(torch.from_numpy(np.array(images))), label]
        
        return tuple(ret)
#         return torch.from_numpy(np.array(images))