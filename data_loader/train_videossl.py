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
from data_loader.video_transforms import ToFloatTensorInZeroOne, GroupColorJitter, RandomCutOut, RandomCrop, RandomHorizontalFlip, Resize, Normalize, CenterCrop
import json

def loadVideoData(batch_size=1, unsup_batch_size=12):
    with open('./zaman_launch.json') as f:
        data = json.load(f)
    data_path= data['data_path']
    output_folder= data['output_folder']
    frame_path= data['frame_path']
    train_split = data['train_split']
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
        
    sup_indices = list(range(len(splits['train']['segments'])))
    
    # indices = indices[::stride] ################################################################
    #indices = indices * 2
    #splitNumber = int(len(indices) / 2)
    #sup_indices = indices[splitNumber:]
    #bp()
    
    ###############################################
    # unsup_indices = indices[::2]
    # sup_indices = indices[1::2]
    unsup_indices = list(range(len(splits['unlabeled']['segments'])))

    crop_list = None
            
    
    
    transform_list = [
        transforms.RandomApply([
            GroupColorJitter(0.8, 0.8, 0.8, 0.4)
        ], p = .8),
        ToFloatTensorInZeroOne(),
   #     transforms.RandomApply([
   #         RandomCutOut((0.1, 0.4))
   #     ], p = .5),
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
            'batch_size': batch_size,
            'norm_vals': VideoModel.default_dataloader_options[
                'norm_vals'
            ],
            'balanced_sampling': True,
        }
    )
    
    
    
    dl_unsup = VideoModel.default_dataloader(
        id_to_video_files,
        [splits['unlabeled']['segments'][i] for i in unsup_indices],
        [splits['unlabeled']['labels'][i] for i in unsup_indices],
        frames_path = frames_path,
        training = to_shuffle, # do not shuffle if using temporal batches
        transform = train_transform,
        extra_frames = 0,
        crop_list = crop_list,
        frames_of_interest = foi,
        transform_frames_of_interest = False,
        options = {
            'batch_size': unsup_batch_size,
            'norm_vals': VideoModel.default_dataloader_options[
                'norm_vals'
            ]
        }
    )
    
    
    if 'crop_list' not in splits['dev']:
        crop_list_dev = None
    else:
        crop_list_dev = splits['dev']['crop_list']
       
        
    precrop = 112
    dev_transform_list = [
        ToFloatTensorInZeroOne(),
        Resize((frame_size, frame_size)),
        # CenterCrop((80,80)),
        Normalize(*VideoModel.default_dataloader_options[
            'norm_vals'
        ])
    ]
    dev_transform = transforms.Compose(dev_transform_list)
    
    dev_dl = VideoModel.default_dataloader(
        id_to_video_files,
        splits['dev']['segments'][::1],
        splits['dev']['labels'][::1],
        frames_path = frames_path,
        training = False,
        transform = dev_transform,
        extra_frames = 0,
        crop_list = crop_list_dev,
        options = {
            'batch_size': batch_size,
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
            'batch_size': batch_size,
        }
    )

    # bp()
    print('done!')
    return dl_sup, dl_unsup, dev_dl, test_dl
