'''
Wrapper around an abstract deep model to provide common functionality
(like the training loop).
'''
from abc import ABC, abstractmethod
from typing import Optional, Mapping, Dict, Sequence, Tuple, List
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm.auto import tqdm
import numpy as np
import warnings
import random
import time
import datetime
import copy
from metrics import StreamMetrics
from ipdb import set_trace as bp

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

class DeepModel(ABC):
    '''
    This is an abstract class that wraps a training loop.
    
    Downstream classes need to set an instance attributes self.default_training_options,
    self.device, and self.model upon instantiation, and implement compute_logits.
    '''
    
    # Default options for training.
    default_training_options = {
        'loss': 'bcewithlogits',     # Which loss to use. Can be 'bcewithlogits',
                                     # 'crossentropyloss', or 'custom'
                                     # 'crossentropyloss', or 'custom'
        'custom_criterion': None,    # Custom criterion class for a loss function
        'custom_loss': None,         # Custom function to compute the loss, takes in logits and labels
        'num_tasks': 1,              # If greater than one, compute and add up multiple losses
        'task_classes': None,        # If num_tasks > 1, number of classes for each task
        'loss_list': None,           # Name of each loss when num_tasks > 1
        'log_metrics': True,         # If True, log accuracy, F1, precision, recall, etc
        'pos_weight': 1.0,           # weight for the positive class (1.0 is neutral)
        'class_weights': [],         # class weights for cross entropy loss
        'clamp_labels': False,       # if True, clamp labels to 1.0 and 0.0 (sometimes useful
                                     # for training with probabilistic labels)
        'epochs': 25,                # number of epochs
        
        # scheduler parameters
        'scheduler': 'step',         # scheduler, can be linear or step
        'warmup': 0,                 # number of warmup steps
        'step_gamma': 0.1,           # for step schedule, gamma parameter
        'step_size': 7,              # for step schedule, how many epochs between steps
        
        # optimizer parameters
        'lr': 5e-5,                  # default learning rate
        'optimizer': 'sgd',          # default optimizer, options are 'adam' or 'sgd'
        'adam_eps': 1e-8,            # for the Adam optimizer, set epsilon
        'sgd_momentum': 0.9,         # momentum for SGD
        'clip_grad_norm': True,      # whether to clip the gradient norm
        'custom_param_groups': None, # function to get custom parameter groups, takes in
                                     # parameter generator and learning rate
                                    
        
        
        'return_best': None,         # if specified, return the model from the epoch that does
                                     # the best according to this validation set
                                     # i.e., if you have a validation set named 'val' then
                                     # return_best could be set to val
        'best_metric': 'f1',         # metric by which to measure which epoch is the "best"
        'seed': None,                # random seed
        'verbose': True,             # whether to print things out
        'print_every': 10,
    }
    
    # Instance variables that downstream classes should set
    model = None
    device = None
    
    @abstractmethod
    def compute_logits(
        self,
        model: torch.nn,
        dataloader_batch: any,
        device: str,
        task: Optional[int] = 0,
    ) -> torch.tensor:
        '''
        Compute logits from a dataloader batch using model, with execution on
        device. Expects a torch.nn as the model, a batch from the dataloader,
        and a device as specified by PyTorch (i.e., 'cpu' or 'cuda:0').
        '''
        pass
    
    @abstractmethod
    def get_labels(
        self,
        dataloader_batch: any,
        device: str,
        task: Optional[int] = 0,
    ) -> torch.tensor:
        '''
        Get the labels from the batch.
        '''
        pass
    
    @abstractmethod
    def compute_embeddings(
        self,
        dataloader: DataLoader,
        options: Optional[Mapping[str, any]] = None
    ) -> np.ndarray:
        '''
        Compute embeddings for (text) items in the DataLoader.
        '''
        pass
    
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
    
    def train(
        self,
        dataloader: DataLoader,
        validation: Sequence[Tuple[str, DataLoader]] = None,
        log_file: Optional[str] = None,
        options: Optional[Mapping[str, any]] = None
    ) -> torch.nn:
        '''
        Train a model using the points/labels in train_dataloader.
        
        Expects the model to be in instance attribute self.model (an instance of torch.nn),
        default training options to be in self.default_training_options, and the device
        to be in self.device. Updates self.model with the trained model.
        
        Each items in train_dataloader is expected to have the data point, an attention
        mask, and a label.
        
        If the validation parameter is specified, compute validation loss/metrics on
        those dataloaders as well.
        The validation parameter expects a sequence of pairs of (str, DataLoader), where
        the string will be printed out along with the DataLoader.
        
        For example: [("val1", val1_dataloader), ("val2", val2_dataloader)]
        
        Based on the options, the training loop can return the epoch where the model does
        the best on one of the validation sets.
        '''
        def format_time(elapsed):
            '''
            Takes a time in seconds and returns a string.
            
            TODO: move this somewhere else.
            '''

            # Format as hh:mm:ss
            return str(datetime.timedelta(seconds=elapsed))
        
        training_options = self.default_training_options.copy()
        
        if options is not None:
            training_options.update(options)
        
        print(training_options)
        
        if training_options['seed'] is not None:
            random.seed(training_options['seed'])
            np.random.seed(training_options['seed'])
            torch.manual_seed(training_options['seed'])
            torch.cuda.manual_seed_all(training_options['seed'])
            
        model = self.model
        if training_options['custom_param_groups'] is None:
            params = model.parameters()
        else:
            params = training_options['custom_param_groups'](
                model, training_options['lr'])
        
        if training_options['optimizer'] == 'adam':
            optimizer = AdamW(params,
                lr = training_options['lr'],
                eps = training_options['adam_eps']
            )
        elif training_options['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                params,
                lr = training_options['lr'],
                momentum = training_options['sgd_momentum']
            )
        else:
            raise NotImplementedError('Optimizer {} not implemented yet'.format(
                training_options['optimizer']
            ))
            
        verbose = training_options['verbose']
        epochs = training_options['epochs']
        
        total_steps = len(dataloader) * epochs
        
        if training_options['scheduler'] == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = training_options['warmup'],
                num_training_steps = total_steps
            )
        elif training_options['scheduler'] == 'step':
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=training_options['step_size'], 
                gamma=training_options['step_gamma']
            )
        else:
            raise NotImplementedError('Scheduler {} not implemented yet'.format(
                training_options['scheduler']
            ))
            
        if training_options['num_tasks'] > 1:
            assert(training_options['loss'] == 'custom')
            assert(training_options['custom_loss'] != None)
            
        if training_options['loss'] == 'bcewithlogits':
            if training_options['pos_weight'] != 1.0:
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight = torch.tensor([training_options['pos_weight']])
                ).to(self.device)
            else:
                criterion = nn.BCEWithLogitsLoss().to(self.device)
        elif training_options['loss'] == 'crossentropyloss':
            if training_options['class_weights'] != []:
                class_weights = torch.FloatTensor(
                    training_options['class_weights']).to(self.device)
                criterion = nn.CrossEntropyLoss(weight = class_weights).to(self.device)
            else:
                criterion = nn.CrossEntropyLoss().to(self.device)
        elif training_options['loss'] == 'custom':
            if training_options['num_tasks'] > 1:
                criterion_list = [crit.to(self.device) for crit in training_options['custom_criterion']] 
            else:
                criterion = training_options['custom_criterion'].to(self.device)
        else:
            raise NotImplementedError('Loss {} not implemented yet'.format(
                training_options['loss']
            ))
        
        total_t0 = time.time()
        
        best_loss = None
        best_score = None
        best_epoch = None
        best_model_wts = None
        
        if log_file is not None:
            log_f = open(log_file, 'w')
            
        phases = ['train', 'val'] if validation is not None else ['train']
        
        for epoch_i in range(1, epochs + 1):
            if verbose:
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
                print('Training...')
            if log_file is not None:
                print("", file=log_f)
                print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs), file=log_f)
                print('Training...', file=log_f)
            
            epoch_t0 = time.time()
            
            for phase in phases:
                if phase == 'train':
                    dataloaders = [('train', dataloader)]
                else:
                    dataloaders = validation
                    if verbose:
                        print("Beginning validation")
                    if log_file is not None:
                        print("Beginning validation", file=log_f)
                    
                for dl_name, cur_dataloader in dataloaders:
                    if phase != 'train':
                        if verbose:
                            print('Val set {}'.format(dl_name))
                        if log_file is not None:
                            print('Val set {}'.format(dl_name), file=log_f)
                    
                    # Measure how long the epoch takes.
                    t0 = time.time()

                    # Reset the total loss for this epoch.
                    if training_options['num_tasks'] > 1:
                        total_loss_list = [0 for i in range(training_options['num_tasks'])]
                    total_loss = 0              
                    
                    if training_options['num_tasks'] > 1:
                        stream_metrics_list = [
                            StreamMetrics([
                                 'acc', 'pre', 'rec', 'f1', 'tp', 'tn', 'fp', 'fn'
                            ], num_classes = n_classes) if n_classes > 1 else
                            StreamMetrics([
                                 'acc', 'pre', 'rec', 'f1', 'tp', 'tn', 'fp', 'fn'
                            ])
                            for _, n_classes in zip(
                                range(training_options['num_tasks']),
                                training_options['task_classes']
                            )
                        ]
                    else:
                        if self.num_classes > 1:
                            stream_metrics = StreamMetrics([
                                 'acc', 'pre', 'rec', 'f1', 'tp', 'tn', 'fp', 'fn'
                            ], num_classes = self.num_classes)
                        else:
                            stream_metrics = StreamMetrics([
                                 'acc', 'pre', 'rec', 'f1', 'tp', 'tn', 'fp', 'fn'
                            ])
                    
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    for step, batch in enumerate(cur_dataloader):
                        if step % training_options['print_every'] == 0 and not step == 0:
                            elapsed = format_time(time.time() - t0)
                            
                            if verbose:
                                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                                    step, len(cur_dataloader), elapsed))
                            if log_file is not None:
                                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                                    step, len(cur_dataloader), elapsed), file=log_f)

                        if phase == 'train':
                            model.zero_grad()
                        
                        if training_options['num_tasks'] > 1:
                            labels_list = [
                                self.get_labels(batch, self.device, task = i)
                                for i in range(training_options['num_tasks'])
                            ]
                        else:
                            labels = self.get_labels(batch, self.device)
                        
                        with torch.set_grad_enabled(phase == 'train'):
                            if training_options['num_tasks'] == 1:
                                logits = self.compute_logits(model, batch, self.device)
                            if training_options['loss'] == 'bcewithlogits':
                                if phase != 'train' or training_options['clamp_labels']:
                                    loss = criterion(
                                        logits.view(labels.shape), 
                                        (labels > 0.5).float()
                                    )
                                else:
                                    loss = criterion(logits.view(labels.shape), labels.float())
                            elif training_options['loss'] == 'crossentropyloss':
                                loss = criterion(logits, labels.long())
                            elif training_options['loss'] == 'custom':
                                if training_options['num_tasks'] > 1:
                                    loss_list = []
                                    logits_list = []
                                    for i in range(training_options['num_tasks']):
                                        cur_logits = self.compute_logits(
                                            model, batch, self.device, task = i)
                                        loss_list.append(training_options['custom_loss'][i](
                                            cur_logits, labels_list[i], criterion_list[i]
                                        ))
                                        logits_list.append(cur_logits.detach())
                                    loss = sum(loss_list)
                                else:
                                    loss = training_options['custom_loss'](logits, labels, criterion)
                            else:
                                # Will need to do something different to the logits for cross entropy
                                raise NotImplementedError('Loss {} not supported'.format(
                                    training_options['loss']
                                ))
                                
                            if phase == 'train':
                                loss.backward()
                                
                                if training_options['clip_grad_norm']:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                
                                optimizer.step() 
                                
                                if training_options['scheduler'] != 'step':
                                    scheduler.step()
                            if training_options['num_tasks'] == 1:
                                logits = logits.detach()

                            total_loss += loss.item()
                            if training_options['num_tasks'] > 1:
                                for i in range(training_options['num_tasks']):
                                    total_loss_list[i] += loss_list[i].item()
                        
                        if training_options['log_metrics']:
                            if (training_options['loss'] == 'custom' and 
                                training_options['num_tasks'] > 1):
                                loss_names = training_options['loss_list']
                            else:
                                stream_metrics_list = [stream_metrics]
                                loss_names = [training_options['loss']]
                                labels_list = [labels]
                                logits_list = [logits]
                            num_tasks = training_options['num_tasks']
                            for i in range(num_tasks):
                                loss_name = loss_names[i]
                                metric = stream_metrics_list[i]
                                cur_labels = labels_list[i]
                                cur_logits = logits_list[i]
                                
                                if loss_name == 'bcewithlogits':
                                    metric.update(
                                        torch.flatten((cur_labels.detach() > 0.5).int()),
                                        torch.flatten((logits > 0.0).int())
                                    )
                                elif loss_name == 'crossentropyloss':
                                    _, preds = torch.max(cur_logits, 1)
                                    metric.update(
                                        torch.flatten(cur_labels.detach().int()),
                                        torch.flatten(preds.int())
                                    )
                                        
                        
                    avg_loss = total_loss / len(cur_dataloader)
                    if training_options['num_tasks'] > 1:
                        avg_loss_list = [
                            total_loss_list[i] / len(cur_dataloader)
                            for i in range(training_options['num_tasks'])
                        ]
                    else:
                        avg_loss_list = [avg_loss]
                    
                    if training_options['num_tasks'] == 1:
                        stream_metrics_list = [stream_metrics]
                        task_classes = [self.num_classes]
                        loss_names = [training_options['loss']]
                    else:
                        task_classes = training_options['task_classes']
                        loss_names = training_options['loss_list']
                    for i in range(training_options['num_tasks']):
                        if training_options['num_tasks'] > 1:
                            if verbose:
                                print('Metrics for task {}'.format(i))
                            if log_file is not None:
                                print('Metrics for task {}'.format(i),
                                      file = log_f)
                        
                        if loss_names[i] != 'custom' and training_options['log_metrics']:
                            metrics = stream_metrics_list[i].compute()

                            if task_classes[i] == 1:
                                acc = metrics[0][1]
                                pre = metrics[1][1]
                                rec = metrics[2][1]
                                f1 = metrics[3][1]
                                tp = metrics[4][1]
                                tn = metrics[5][1]
                                fp = metrics[6][1]
                                fn = metrics[7][1]
                                if verbose:
                                    print("  {}\tLoss {:.4f}\tAcc {:.4f}\tPre: {:.4f}\tRec: {:.4f}\tF1: {:.4f}".format(
                                        dl_name, avg_loss_list[i], acc, pre, rec, f1
                                    ))
                                    print("  {}\tTP {}\tTN {}\tFP {}\tFN {}".format(dl_name, tp, tn, fp, fn))
                                if log_file is not None:
                                    print("  {}\tLoss {:.4f}\tAcc {:.4f}\tPre: {:.4f}\tRec: {:.4f}\tF1: {:.4f}".format(
                                        dl_name, avg_loss_list[i], acc, pre, rec, f1
                                    ), file = log_f)
                                    print("  {}\tTP {}\tTN {}\tFP {}\tFN {}".format(dl_name, tp, tn, fp, fn), file=log_f)
                            else:
                                classes = metrics[0][1]
                                acc = metrics[1][1]
                                pre = metrics[2][1]
                                rec = metrics[3][1]
                                f1 = metrics[4][1]
                                tp = metrics[5][1]
                                tn = metrics[6][1]
                                fp = metrics[7][1]
                                fn = metrics[8][1]
                                if verbose:
                                    print("  {}\tLoss {:.4f}\tAcc {:.4f}".format(
                                        dl_name, avg_loss_list[i], acc
                                    ))
                                    print("  Classes: {}\n  Pre: {}\n  Rec: {}\n  F1: {}\n".format(
                                        classes, pre, rec, f1
                                    ) + "  TP: {}\n  TN: {}\n  FP: {}\n  FN: {}\n".format(
                                        tp, tn, fp, fn
                                    ))
                                if log_file is not None:
                                    print("  {}\tLoss {:.4f}\tAcc {:.4f}".format(
                                        dl_name, avg_loss_list[i], acc
                                    ), file = log_f)
                                    print("  Classes: {}\n  Pre: {}\n  Rec: {}\n  F1: {}\n".format(
                                        classes, pre, rec, f1
                                    ) + "  TP: {}\n  TN: {}\n  FP: {}\n  FN: {}\n".format(
                                        tp, tn, fp, fn
                                    ), file = log_f)
                        else:
                            if verbose:
                                print("  {}\tLoss {:.4f}".format(
                                    dl_name, avg_loss_list[i]
                                ))
                            if log_file is not None:
                                print("  {}\tLoss {:.4f}".format(
                                    dl_name, avg_loss_list[i]
                                ), file = log_f)
                    if training_options['num_tasks'] > 1:
                        if verbose:
                            print("{}\tTotal Loss {:.4f}".format(
                                dl_name, avg_loss
                            ))
                        if log_file is not None:
                            print("{}\tTotal Loss {:.4f}".format(
                                dl_name, avg_loss
                            ), file = log_f)

                    if phase == 'train' and training_options['scheduler'] == 'step':
                        scheduler.step()
                        
                    if (training_options['return_best'] is not None and 
                        training_options['return_best'] == dl_name):
                        if training_options['num_tasks'] > 1:
                            print(
                                'WARNING: RETURN BEST NOT SUPPORTED FOR MULTIPLE TASKS',
                                file = sys.stderr
                            )
                        # Update best model/score
                        if training_options['best_metric'] == 'f1':
                            score = f1
                        elif training_options['best_metric'] == 'pre':
                            score = pre
                        elif training_options['best_metric'] == 'rec':
                            score = rec
                        elif training_options['best_metric'] == 'acc':
                            score = acc
                        else:
                            raise NotImplementedError("Metric {} not supported".format(
                                training_options['best_metric']
                            ))
                        
                        if best_score is None or score > best_score:
                            best_score = score
                            best_epoch = epoch_i
                            best_loss = loss
                            best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # DL time
                    dl_time = format_time(time.time() - t0)
                    if verbose:
                        print('DL {} took {}'.format(
                            dl_name, dl_time
                        ))
                    if log_file is not None:
                        print('DL {} took {}'.format(
                            dl_name, dl_time
                        ), file = log_f)
                    
                    if log_file is not None:
                        log_f.flush()
            
            # Epoch time
            epoch_time = format_time(time.time() - epoch_t0)
            if verbose:
                print('Epoch {} took {}'.format(
                    epoch_i, epoch_time
                ))
            if log_file is not None:
                print('Epoch {} took {}'.format(
                    epoch_i, epoch_time
                ), file = log_f)
        
        # Update with best model if necessary
        if training_options['return_best']:
            model.load_state_dict(best_model_wts)
            
            if verbose:
                print("Best Score: {}".format(best_score))
                print("Best epoch: {}".format(best_epoch))
                print("Best epoch loss: {}".format(best_loss))
            if log_file is not None:
                print("Best Score: {}".format(best_score), file=log_f)
                print("Best epoch: {}".format(best_epoch), file=log_f)
                print("Best epoch loss: {}".format(best_loss), file=log_f)
        
        self.model = model
        
        # Total time
        total_time = format_time(time.time() - total_t0)
        print('Total time took {}'.format(
            total_time
        ))
            
        if log_file is not None:
            log_f.close()
            
    def predict_proba(
        self,
        dataloader: DataLoader,
        verbose: Optional[bool] = False,
    ) -> np.ndarray:
        '''
        Generate logit predictions.
        '''
        results = []
        
        model = self.model
        model.eval()
        
        for batch in dataloader if not verbose else tqdm(dataloader):
            with torch.set_grad_enabled(False):
                logits = self.compute_logits(model, batch, self.device)
                
                results.append(logits.detach().cpu().numpy())
        
        return np.concatenate(results)
    
    def predict(
        self,
        dataloader: DataLoader,
        verbose: Optional[bool] = False,
    ) -> np.ndarray:
        '''
        Generate predictions.
        '''
        
        if self.num_classes == 1:
            results = self.predict_proba(dataloader, verbose = verbose)
        else:
            results = []
        
            model = self.model
            model.eval()

            for batch in dataloader if not verbose else tqdm(dataloader):
                with torch.set_grad_enabled(False):
                    logits = self.compute_logits(model, batch, self.device)
                    _, preds = torch.max(logits, 1)
                    
                    results.append(preds.detach().cpu().numpy())
                
            return np.concatenate(results).flatten()
        
        return np.where(results > 0.0, 1, -1).flatten()
