# '''
# Train a video model.
# '''

import torch
from torch import nn
import torchvision
from torchvision import models, transforms
from ipdb import set_trace as bp
# import mayanshell
# from mayanshell.model.deep_model import DeepModel
# from mayanshell.validation import compute_metrics

import json
# import pickle

# import sys
import os
# sys.path.append('../py')
# from video_dataset import VideoClipDataset
from data_loader.video_model import VideoModel
# from fc_model import FCModel
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from data_loader.video_transforms import *

def loadVideoData():
    data_path='/home/alimirz1/babul/fdubost/experiments/256/'
    output_folder='/home/alimirz1/'
    frame_path='/home/alimirz1/babul/fdubost/experiments/258/frames/'
    train_split = '/home/alimirz1/babul/fdubost/experiments/285/model_splits.json'
    stride = 1
    frame_size = 80
        
    with open(data_path+'video_table.json', 'r') as f:
        video_table = json.load(f)
    video_folder = data_path+'videos'
    
    os.makedirs(output_folder, exist_ok=True)
            
    log_file = os.path.join(output_folder, 'log.txt')
        
    id_to_video_files = {
        'vid{}'.format(vid['id']): os.path.join(video_folder, vid['name'])
        for vid in video_table
    }
        
    frames_path = frame_path+'{}'
    
    masks_path = 'masks/{}'
    
        
    with open(train_split, 'r') as f:
        splits = json.load(f)
        
        
    #     metadata = {}
    #     metadata.update(vars(args))
    #     metadata.update({
    #         'id_to_video_files': id_to_video_files,
    #         'frames_path': frames_path
    #     })
        
    #     print(metadata)
        
    indices = list(range(len(splits['train']['segments'])))
    
    # indices = indices[::stride] ################################################################
    #indices = indices * 2
    #splitNumber = int(len(indices) / 2)
    #sup_indices = indices[splitNumber:]
    #bp()
    unsup_indices = indices[::2]
    sup_indices = indices[1::2]
        
    if 'crop_list' not in splits['train']:
        crop_list = None
    else:
        crop_list = splits['train']['crop_list'][stride]
            
    
    
    transform_list = [
        transforms.RandomApply([
            GroupColorJitter(0.8, 0.8, 0.8, 0.4)
        ], p = .8),
        ToFloatTensorInZeroOne(),
        transforms.RandomApply([
            RandomCutOut((0.1, 0.4))
        ], p = .5),
        RandomCrop((0.9, 0.9)),
        RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #     ], p = 0.8),
    #     transforms.RandomGrayscale(p = 0.2),
        Resize((frame_size, frame_size)),
        Normalize(*VideoModel.default_dataloader_options[
            'norm_vals'
        ])
    ]
    
    foi = None
        
           
    train_transform = transforms.Compose(transform_list)
    to_shuffle = True
    dl_sup = VideoModel.default_dataloader(
        id_to_video_files,
        [splits['train']['segments'][i] for i in sup_indices],
        [splits['train']['labels'][i] for i in sup_indices],
        frames_path = frames_path,
        training = to_shuffle, # do not shuffle if using temporal batches
        transform = train_transform,
        extra_frames = 0,
        crop_list = crop_list,
        frames_of_interest = foi,
        transform_frames_of_interest = False,
        options = {
            'batch_size': 12,
            'norm_vals': VideoModel.default_dataloader_options[
                'norm_vals'
            ],
            'balanced_sampling': True,
        }
    )
    dl_unsup = VideoModel.default_dataloader(
        id_to_video_files,
        [splits['train']['segments'][i] for i in unsup_indices],
        [splits['train']['labels'][i] for i in unsup_indices],
        frames_path = frames_path,
        training = to_shuffle, # do not shuffle if using temporal batches
        transform = train_transform,
        extra_frames = 0,
        crop_list = crop_list,
        frames_of_interest = foi,
        transform_frames_of_interest = False,
        options = {
            'batch_size': 12,
            'norm_vals': VideoModel.default_dataloader_options[
                'norm_vals'
            ],
            'balanced_sampling': True,
        }
    )
    
    
    if 'crop_list' not in splits['dev']:
        crop_list_dev = None
    else:
        crop_list_dev = splits['dev']['crop_list']
    
    dev_dl = VideoModel.default_dataloader(
        id_to_video_files,
        splits['dev']['segments'][::1],
        splits['dev']['labels'][::1],
        frames_path = frames_path,
        training = False,
        extra_frames = 0,
        crop_list = crop_list_dev,
        options = {
            'batch_size': 12,
        }
    )
    
    test_dl = VideoModel.default_dataloader(
        id_to_video_files,
        splits['test']['segments'][::1],
        splits['test']['labels'][::1],
        frames_path = frames_path,
        training = False,
        extra_frames = 0,
        crop_list = crop_list_dev,
        options = {
            'batch_size': 12,
        }
    )

    # bp()
    print('done!')
    return dl_sup, dl_unsup, dev_dl, test_dl
