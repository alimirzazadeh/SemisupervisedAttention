'''
Wrapper around models for video classification.

Currently supports 3D ResNet.
'''

import mayanshell
from data_loader.deep_model import DeepModel
import torch
from torch import nn
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms
from tqdm.auto import tqdm
import numpy as np
import warnings
import random
import time
import datetime
import copy
from PIL import Image
# from rubiksnet.models import RubiksNet

from data_loader.video_dataset import *

def make_weights_for_balanced_classes(labels):
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    nclasses = len(set(labels))
    count = [0] * nclasses
    for label in labels:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(labels)
    for idx, label in enumerate(labels):
        weight[idx] = weight_per_class[label]
    return weight

def load_valid_weights_only(model, state_dict_path):
    '''Load params from state dict path, but only the ones that are in model.'''
    existing_state_dict = model.state_dict()
    state_dict = torch.load(state_dict_path)
    new_state_dict = {
        k: v 
        for k, v in state_dict.items()
        if k in existing_state_dict and v.shape == existing_state_dict[k].shape
    }

    for k, v in existing_state_dict.items():
        if k not in new_state_dict:
            new_state_dict[k] = v
            print('Warning: {} not found in new state dict'.format(k))
    model.load_state_dict(new_state_dict)
    return model

class VideoModel(DeepModel):
    default_dataloader_options = {
        'precrop': 112,
        'crop': 112,
        'norm_vals': ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
        'rubiks_norm_vals': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'batch_size': 16,
        'num_workers': 8,
        'pin_memory': False,
        'balanced_sampling': False, # training must be True
    }
    
    default_training_options = {
        'loss': 'crossentropyloss',
        'epochs': 25,
        'scheduler': 'step',
        'warmup': 0,
        'optimizer': 'sgd',
        'lr': .001,
        'sgd_momentum': 0.9,
    }
    
    def __init__(
        self,
        model_type = 'r3d-18',
        from_pretrained = 'pytorch',
        model_weights = None,
        device = 'cpu',
        num_classes = 1,
        num_frames = 4,
    ):
        '''For model type rubiks3d-large, if from_pretrained == "svl" then model_weights should
        point to the path to the SVL weights.'''
        default_training_options = DeepModel.default_training_options.copy()
        default_training_options.update(VideoModel.default_training_options)
        
        self.default_training_options = default_training_options
        
        self.model_type = model_type
        self.from_pretrained = from_pretrained
        self.model_weights = model_weights
        self.device = device
        self.num_classes = num_classes
        
        if model_type in ['r3d-18', 'r2plus1d-18']:
            if model_type == 'r3d-18':
                model = tv_models.video.r3d_18(pretrained=from_pretrained == 'pytorch')
            else:
                model = tv_models.video.r2plus1d_18(pretrained=from_pretrained == 'pytorch')
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
            if model_weights is not None:
                model.load_state_dict(torch.load(model_weights))
        elif model_type in ['rubiks3d-large']:
            if from_pretrained == 'svl':
                ckpt = torch.load(
                    os.path.expanduser(model_weights),
                    map_location="cpu")
            
                model = RubiksNet(
                    tier = 'large', num_classes = ckpt['num_classes'],
                    num_frames = num_frames
                )
                
                model.load_state_dict(ckpt['model'])
                
                num_ftrs = model.new_fc.in_featurs
                model.new_fc = nn.Linear(num_ftrs, num_classes)
            else:
                model = RubiksNet(
                    tier = 'large', num_classes = num_classes,
                    num_frames = num_frames
                )
            
                if model_weights is not None:
                    model.load_state_dict(torch.load(model_weights))
        else:
            raise NotImplementedError('Model {} not supported'.format(model_type))
                
        model = model.to(device)
            
        self.model = model
        
    def save_weights(self, path: str) -> None:
        '''
        Save the weights of the model into path.
        
        Note that this does NOT save this whole class (for now).
        '''
        torch.save(self.model.state_dict(), path)
        
    def load_weights(self, path: str) -> None:
        '''
        Load the weights of the model from path.
        
        Note that this does NOT load this whole class (for now).
        '''
        self.model.load_state_dict(torch.load(path))
        
    def default_dataloader(id_to_video_files, segments, labels = None, frames_path = None,
                           training = False, # alias for shuffling!
                          options = None,
                          crop_list = None,
                          transform = None,
                          include_mask = False,
                          masks_path = None,
                          frames_of_interest = None,
                          transform_frames_of_interest = False,
                          extra_frames = 0,
                          extra_tasks = [],
                          extra_task_transform = None,
                          extra_args = {},):
        dataloader_options = VideoModel.default_dataloader_options.copy()
        
        if options is not None:
            dataloader_options.update(options)
            
        precrop = dataloader_options['precrop']
        crop = dataloader_options['crop']
        norm_vals = dataloader_options['norm_vals']
        num_workers = dataloader_options['num_workers']
        batch_size = dataloader_options['batch_size']
        pin_memory = dataloader_options['pin_memory']
        norm_vals = dataloader_options['norm_vals']
        balanced_sampling = dataloader_options['balanced_sampling']
        
        if transform is None:
            transform_list = []

            transform_list = [
                ToFloatTensorInZeroOne(),
                Resize((precrop, precrop)),
                CenterCrop((crop, crop)),
                Normalize(*norm_vals)
            ]
            
            transform = tv_transforms.Compose(transform_list)
        
        dataset = VideoClipDataset(
            id_to_video_files, segments, labels, frames_path = frames_path,
            transform = transform,
            crop_list = crop_list,
            include_mask = include_mask,
            masks_path = masks_path,
            frames_of_interest = frames_of_interest,
            transform_frames_of_interest = transform_frames_of_interest,
            extra_frames = extra_frames,
            extra_tasks = extra_tasks,
            extra_task_transform = extra_task_transform,
            extra_args = extra_args,
        )
        
        if not balanced_sampling:
            dataloader = DataLoader(
                dataset,
                shuffle = training,
                batch_size = batch_size,
                num_workers = num_workers,
                pin_memory = pin_memory
            )
        else:
            weights = torch.DoubleTensor(make_weights_for_balanced_classes(labels))
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            
            dataloader = DataLoader(
                dataset,
                sampler = sampler,
                batch_size = batch_size,
                num_workers = num_workers,
                pin_memory = pin_memory
            )
        
        return dataloader
    
    def compute_embeddings(self, dataloader, options):
        raise NotImplementedError
        
    def compute_embeddings(self, dataloader, options = None):
        '''
        Compute embeddings using the model.
        '''
        embedding_options = VideoSimCLRModel.default_embedding_options.copy()
        if options is not None:
            embedding_options.update(options)
        
        layer = self.model._modules.get('avgpool')
        if self.model_type in ['r3d-18', 'r2plus1d-18']:
            embedding_size = 512
        else:
            embedding_size = 576
        
        model = self.model
        model = model.eval()

        def get_vector_fast(img):
            img = img.to(self.device)

            my_embedding = torch.zeros((img.shape[0], embedding_size)).to(self.device)

            def copy_data(m, i, o):
                my_embedding.copy_(torch.flatten(o, 1).squeeze(0).data)

            h = layer.register_forward_hook(copy_data)

            model(img)

            h.remove()

            return my_embedding
        
        all_embeddings = []
        for img, label in dataloader if not embedding_options['verbose'] else tqdm(dataloader):
            embedding = get_vector_fast(img)
            all_embeddings += embedding.tolist()
            
        if embedding_options['mean_center']:
            emb_mean = np.mean(all_embeddings, axis=0)
            all_embeddings = [
                (emb - emb_mean)
                for emb in all_embeddings
            ]
        
        return np.vstack(all_embeddings).astype('float32')
    
    def compute_logits(
        self,
        model: torch.nn,
        dataloader_batch: any,
        device: str
    ) -> torch.tensor:
        inputs = dataloader_batch[0].to(device)
        
        return model(inputs)
    
    def get_labels(
        self,
        dataloader_batch: any,
        device: str
    ) -> torch.tensor:
        return dataloader_batch[1].to(device)

class SimCLRModel(nn.Module):
    def __init__(
        self, backbone, dim_in, feat_dim=128,
        extra_task_heads = [] # number of output classes for each output task head
    ):
        super(SimCLRModel, self).__init__()
        self.encoder = backbone
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        self.task_heads = nn.ModuleList()
        for n_task_classes in extra_task_heads:
            self.task_heads.append(nn.Linear(dim_in, n_task_classes))
    
    def forward(self, x, extra_task = None):
        if extra_task is None:
            feat = self.encoder(x)
            feat = nn.functional.normalize(self.head(feat), dim = 1)
            return feat
        else:
            feat = self.encoder(x)
            return self.task_heads[extra_task](feat)
    
class VideoSimCLRModel(DeepModel):
    default_training_options = {
        'loss': 'custom',
        'epochs': 25,
        'scheduler': 'step',
        'warmup': 0,
        'optimizer': 'sgd',
        'lr': .001,
        'sgd_momentum': 0.9,
    }
    
    default_embedding_options = {
        'mean_center': True,        # whether to zero-center all the embeddings
        'verbose': False,
    }
    
    def __init__(
        self,
        model_type = 'r3d-18',
        from_pretrained = 'pytorch',
        model_weights = None,
        device = 'cpu',
        num_classes = 1,
        num_frames = 4,
        feat_dim = 128,
        extra_task_heads = [],
        encoder_weights = None,
    ):
        default_training_options = DeepModel.default_training_options.copy()
        default_training_options.update(VideoSimCLRModel.default_training_options)
        
        self.default_training_options = default_training_options
        
        self.model_type = model_type
        self.from_pretrained = from_pretrained
        self.model_weights = model_weights
        self.device = device
        self.num_classes = num_classes
        self.extra_task_heads = extra_task_heads
        
        if model_type in ['r3d-18', 'r2plus1d-18']:
            if model_type == 'r3d-18':
                encoder = tv_models.video.r3d_18(pretrained=from_pretrained == 'pytorch')
            else:
                encoder = tv_models.video.r2plus1d_18(pretrained=from_pretrained == 'pytorch')
            
            if encoder_weights is not None:
                encoder = load_valid_weights_only(encoder, encoder_weights)
            
            num_ftrs = encoder.fc.in_features
            encoder.fc = nn.Identity()
            
            model = SimCLRModel(encoder, num_ftrs,
                                extra_task_heads = extra_task_heads)
            
            if model_weights is not None:
                model = load_valid_weights_only(model, model_weights)
        elif model_type in ['rubiks3d-large']:
            if from_pretrained == 'svl':
                ckpt = torch.load(
                    os.path.expanduser(model_weights),
                    map_location="cpu")
            
                encoder = RubiksNet(
                    tier = 'large', num_classes = ckpt['num_classes'],
                    num_frames = num_frames
                )
                
                encoder.load_state_dict(ckpt['model'])
                
                num_ftrs = encoder.new_fc.in_features
                encoder.new_fc = nn.Identity()

                model = SimCLRModel(encoder, num_ftrs,
                                    extra_task_heads = extra_task_heads)
            else:
                encoder = RubiksNet(
                    tier = 'large', num_classes = num_classes,
                    num_frames = num_frames
                )
                
                if encoder_weights is not None:
                    encoder = load_valid_weights_only(encoder, encoder_weights)
                
                num_ftrs = encoder.new_fc.in_features
                encoder.new_fc = nn.Identity()

                model = SimCLRModel(encoder, num_ftrs,
                                    extra_task_heads = extra_task_heads)
                
                if model_weights is not None:
                    model = load_valid_weights_only(model, model_weights)
        else:
            raise NotImplementedError('Model {} not supported'.format(model_type))
                
        model = model.to(device)
            
        self.model = model
        
    def to_video_model(
        model_type = 'r3d-18',
        from_pretrained = 'pytorch',
        model_weights = None,
        device = 'cpu',
        num_classes = 1,
        num_frames = 4,
    ):
        if model_type in ['r3d-18', 'r2plus1d-18']:
            if model_type == 'r3d-18':
                encoder = tv_models.video.r3d_18(pretrained=from_pretrained == 'pytorch')
            else:
                encoder = tv_models.video.r2plus1d_18(pretrained=from_pretrained == 'pytorch')
            
            num_ftrs = encoder.fc.in_features
            encoder.fc = nn.Identity()
            
            pretrained_model = SimCLRModel(encoder, num_ftrs)
            
            if model_weights is not None:
                pretrained_model = load_valid_weights_only(pretrained_model, model_weights)
        elif model_type in ['rubiks3d-large']:
            if from_pretrained == 'svl':
                ckpt = torch.load(
                    os.path.expanduser(model_weights),
                    map_location="cpu")
            
                encoder = RubiksNet(
                    tier = 'large', num_classes = ckpt['num_classes'],
                    num_frames = num_frames
                )
                
                encoder.load_state_dict(ckpt['model'])
                
                num_ftrs = encoder.new_fc.in_features
                encoder.new_fc = nn.Identity()

                pretrained_model = SimCLRModel(encoder, num_ftrs)
            else:
                encoder = RubiksNet(
                    tier = 'large', num_classes = num_classes,
                    num_frames = num_frames
                )
                
                num_ftrs = encoder.new_fc.in_features
                encoder.new_fc = nn.Identity()

                pretrained_model = SimCLRModel(encoder, num_ftrs)
                
                if model_weights is not None:
                    pretrained_model = load_valid_weights_only(pretrained_model, model_weights)
                
        video_model = VideoModel(model_type, from_pretrained = '',
                                 model_weights = None,
                                device = device, num_classes = num_classes,
                                num_frames = num_frames)

        model_dict = video_model.model.state_dict()
        pretrained_dict = pretrained_model.encoder.state_dict()

        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
        }

        model_dict.update(pretrained_dict)

        video_model.model.load_state_dict(model_dict)

        return video_model
        
    def save_weights(self, path: str) -> None:
        '''
        Save the weights of the model into path.
        
        Note that this does NOT save this whole class (for now).
        '''
        torch.save(self.model.state_dict(), path)
        
    def load_weights(self, path: str) -> None:
        '''
        Load the weights of the model from path.
        
        Note that this does NOT load this whole class (for now).
        '''
        self.model.load_state_dict(torch.load(path))
    
    def compute_embeddings(self, dataloader, options = None):
        '''
        Compute embeddings using the model.
        '''
        embedding_options = VideoSimCLRModel.default_embedding_options.copy()
        if options is not None:
            embedding_options.update(options)
        
        layer = self.model._modules.get('head')
        if self.model_type in ['r3d-18', 'r2plus1d-18']:
            embedding_size = 512
        else:
            embedding_size = 576
        
        model = self.model
        model = model.eval()

        def get_vector_fast(img):
            img = img.to(self.device)

            my_embedding = torch.zeros((img.shape[0], embedding_size)).to(self.device)

            def copy_data(m, i, o):
                my_embedding.copy_(torch.flatten(i[0], 1).squeeze(0).data)

            h = layer.register_forward_hook(copy_data)

            model(img)

            h.remove()

            return my_embedding
        
        all_embeddings = []
        for img, label in dataloader if not embedding_options['verbose'] else tqdm(dataloader):
            embedding = get_vector_fast(img)
            all_embeddings += embedding.tolist()
            
        if embedding_options['mean_center']:
            emb_mean = np.mean(all_embeddings, axis=0)
            all_embeddings = [
                (emb - emb_mean)
                for emb in all_embeddings
            ]
        
        return np.vstack(all_embeddings).astype('float32')
    
    def compute_logits(
        self,
        model: torch.nn,
        dataloader_batch: any,
        device: str,
        task: int = 0,
    ) -> torch.tensor:
        if task == 0:
            inputs = dataloader_batch[0]
            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
        
            return model(inputs)
        else:
            inputs = dataloader_batch[2 * (task)].to(device)
            return model(inputs, extra_task = task - 1)
    
    def get_labels(
        self,
        dataloader_batch: any,
        device: str,
        task: int = 0,
    ) -> torch.tensor:
        return dataloader_batch[2 * task + 1].to(device)
