# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:19:13 2021

@author: alimi
"""
import torch
import pandas as pd
from torch import nn
import numpy as np
from PIL import Image
import os
from model.loss import CAMLoss
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import random
from torchvision.utils import save_image
from collections import defaultdict
from model.transformer_loss import isTransformer


OBJECT_CATEGORIES_PASCAL = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
OBJECT_CATEGORIES_COCO = [str(i) for i in range(91)]

TARGET_IMAGE_DIMENSIONS = [256,256]
TARGET_IMAGE_DIMENSIONS_TRANSFORMER = [224,224]
THRESHOLD = 0.5
PLOT_PRED_BBOX = False

def create_bbox_from_map(map, attentionMethod):
    bbox = {}
    # initialize
    transformer = isTransformer(attentionMethod)
    bbox['xmin'] = TARGET_IMAGE_DIMENSIONS[0]
    bbox['ymin'] = TARGET_IMAGE_DIMENSIONS[1]
    bbox['xmax'] = 0
    bbox['ymax'] = 0
    # update bbox
    for i in range(TARGET_IMAGE_DIMENSIONS[0]):
        for j in range(TARGET_IMAGE_DIMENSIONS[1]):
            if map[i,j,0] == 1:
                if i < bbox['ymin']:
                    bbox['ymin'] = i
                elif i > bbox['ymax']:
                    bbox['ymax'] = i
                if j < bbox['xmin']:
                    bbox['xmin'] = j
                elif j > bbox['xmax']:
                    bbox['xmax'] = j
    return bbox

def binarize_map(map, threshold):
    map_bin = map.copy()
    map_bin[map_bin > threshold] = 1
    map_bin[map_bin <= threshold] = 0
    return map_bin

def cast_bbox_to_int(bbox):
    for key in bbox:
        bbox[key] = int(bbox[key])
    return bbox

def rescale_bbox(target_image_dimensions, actual_image_dimension, bbox):
    # bbox coef in this order: xmin, ymin, xmax, ymax
    rescaling_factors = [target*1./actual for target, actual in zip(target_image_dimensions,actual_image_dimension)]
    rescaled_bbox = {}
    for key in bbox:
        if key in ['xmin', 'xmax']:
            rescaled_bbox[key] = bbox[key] * rescaling_factors[0]
        elif key in ['ymin', 'ymax']:
            rescaled_bbox[key] = bbox[key] * rescaling_factors[1]
    return rescaled_bbox

def create_map_from_bbox_list(bbox_list):
    map = np.zeros(TARGET_IMAGE_DIMENSIONS)
    for bbox in bbox_list:
        for i in range(TARGET_IMAGE_DIMENSIONS[0]):
            for j in range(TARGET_IMAGE_DIMENSIONS[1]):
                if bbox['xmin'] < j < bbox['xmax'] and bbox['ymin'] < i < bbox['ymax']:
                    map[i, j] = 1
    map = np.repeat(np.expand_dims(map, axis=-1), 3, axis=-1)
    return map

def compute_overlap(attention_map, bbox_map):
    # compute intersection and union
    intersection = 0
    union = 0
    for i in range(TARGET_IMAGE_DIMENSIONS[0]):
        for j in range(TARGET_IMAGE_DIMENSIONS[1]):
            if attention_map[i, j, 0] == 1 and bbox_map[i, j, 0] == 1:
                intersection += 1
            if attention_map[i, j, 0] == 1 or bbox_map[i, j, 0] == 1:
                union += 1
    # compute overlap
    overlap = intersection*1. / union
    return overlap

def create_rect(bbox, color):
    rect = patches.Rectangle((bbox['xmin'], bbox['ymin']), bbox['xmax'] - bbox['xmin'],
                             bbox['ymax'] - bbox['ymin'], linewidth=1, edgecolor=color,
                             facecolor='none', alpha=0.5)
    return rect


def evaluate(model, data_loader, device, lossInstance, attentionMethod, batchDirectory = '', print_images=False,
             print_attention_maps=False, use_bbox=True, dataset='pascal'):

    # set model to eval mode
    model.eval()
    
    # set seed to select random bbox when there is no TP
    random.seed(20)

    # create path to save images
    if print_images or print_attention_maps:
        path_save_images = os.path.join(batchDirectory,'images')
        if not os.path.exists(path_save_images):
            os.makedirs(path_save_images)

    # create dataframe to store all predictions and all ground truths
    df_gt = pd.DataFrame()
    df_pred = pd.DataFrame()

    # loop over entire test set and save stats
    stats = pd.DataFrame()
    # loop over dataset
    with torch.set_grad_enabled(False):
        for image_id, data in enumerate(data_loader, 0):
            # print every 10
            if image_id % 10 == 0:
                print(image_id)

            # open figure
            plt.figure()

            # pass to the network
            if use_bbox:
                inputs, labels, _ = data
            else:
                inputs, labels = data
            inputs = inputs.to(device)
            labels = labels[0].to(device)
            # get logit
            logits = model(inputs)
            m = nn.Sigmoid()
 
            # save gt and preds in df
            df_pred = df_pred.append(pd.DataFrame(m(logits).cpu().numpy()))
            df_gt = df_gt.append(pd.DataFrame(labels.cpu().numpy()).T)

            # get ground truth classes
            gt_class_ids = np.nonzero(np.array(labels.cpu()))[0]
            if dataset == 'pascal':
                gt_class_names = [OBJECT_CATEGORIES_PASCAL[id] for id in gt_class_ids]
            elif dataset == 'coco':
                gt_class_names = [OBJECT_CATEGORIES_COCO[id] for id in gt_class_ids]

            # get predicted class
            pred_class_id = int(np.argmax(logits.cpu()))
            if dataset == 'pascal':
                pred_class_name = OBJECT_CATEGORIES_PASCAL[pred_class_id]
            elif dataset == 'coco':
                pred_class_name = OBJECT_CATEGORIES_COCO[pred_class_id]

            # compute whether it is a TP
            TP = pred_class_name in gt_class_names

            # compute stats
            # compute unsup loss and attention maps
            target_category = None

            with torch.set_grad_enabled(True):
                consistency_loss, img, gradcam1, guidedbackprop, mask, mask_img, gradcam2 = \
                    lossInstance(inputs, target_category, visualize=True)
                BCE = nn.BCEWithLogitsLoss()(logits[0], labels)
            # postproc guidedbackprop
            guidedbackprop[guidedbackprop > 1] = 1
            # binarize attention maps
            gradcam1_bin = binarize_map(gradcam1, THRESHOLD)
            gradcam2_bin = binarize_map(gradcam2, THRESHOLD)
            guidedbackprop_bin = binarize_map(guidedbackprop, THRESHOLD)
            mask_bin = binarize_map(mask, THRESHOLD)

            # prepare GT bbox
            if use_bbox:
            # select correct gt bboxes
                bbox_list = []
                all_bboxes = data[2][0]['object']
                # select bbox of the correctly identified class if current samples is TP, otherwise select random bboxes
                if TP:
                    target_class = pred_class_name
                else:
                    target_class = all_bboxes[random.randint(0,len(all_bboxes)-1)]['name']
                # iterate to bboxes
                for bbox in all_bboxes:
                    if bbox['name'] == target_class:
                        # cast to int
                        bbox = cast_bbox_to_int(bbox['bndbox'])
                        # append to store bboxes
                        bbox_list.append(bbox)

                # rescale bbox
                image_size = data[2][0]['size']
                actual_image_dimension = [int(image_size['width']), int(image_size['height'])]
                bbox_list_rescaled = []
                for bbox in bbox_list:
                    bbox_list_rescaled.append(rescale_bbox(TARGET_IMAGE_DIMENSIONS, actual_image_dimension, bbox))

                # transform list of bbox to binary map
                bbox_map = create_map_from_bbox_list(bbox_list_rescaled)

            # transform attention map into bbox
            bbox_gb = create_bbox_from_map(guidedbackprop_bin, attentionMethod)
            bbox_gradcam1 = create_bbox_from_map(gradcam1_bin, attentionMethod)
            bbox_gradcam2 = create_bbox_from_map(gradcam2_binattentionMethod)

            # transform list of bbox to binary map
            bbox_gb_map = create_map_from_bbox_list([bbox_gb])


            # print image with corresponding ID
            if print_images:
                # preproc image
                image = np.moveaxis(np.array(data[0][0].cpu()), 0, -1)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))

                # save image
                plt.clf()
                fig, ax = plt.subplots()
                ax.imshow(image)
                # plot all gt bboxes on image
                if use_bbox:
                    for bbox in bbox_list_rescaled:
                        rect = create_rect(bbox, 'g')
                        ax.add_patch(rect)
                if PLOT_PRED_BBOX:
                    ## plot gb bbox
                    rect = create_rect(bbox_gb, 'b')
                    ax.add_patch(rect)
                    # plot gradcam1 bbox
                    rect = create_rect(bbox_gradcam1, 'b')
                    ax.add_patch(rect)
                    # plot gradcam1 bbox
                    rect = create_rect(bbox_gradcam2, 'r')
                    ax.add_patch(rect)
                # save to disk
                plt.axis('off')
                plt.savefig(os.path.join(path_save_images, str(image_id) + "_image.jpeg"), bbox_inches='tight',
                            pad_inches=0)
                plt.close()

            # print attention maps
            if print_attention_maps:
                # save Grad-CAM
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_GradCAM1.jpeg"), gradcam1)
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_GradCAM1_bin.jpeg"), gradcam1_bin)
                # save Grad-CAM 2
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_GradCAM2.jpeg"), gradcam2)
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_GradCAM2_bin.jpeg"), gradcam2_bin)
                # save guided backprop
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_GuidedBackprop.jpeg"), guidedbackprop)
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_GuidedBackprop_bin.jpeg"), guidedbackprop_bin)
                # save mask
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_mask.jpeg"), mask)
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_mask_bin.jpeg"), mask_bin)
                # save bbox maps
                if use_bbox:
                    matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_bbox_map.jpeg"), bbox_map)
                matplotlib.image.imsave(os.path.join(path_save_images, str(image_id) + "_bbox_gb_map.jpeg"), bbox_gb_map)

            # compute overlap attention maps with annotations
            if use_bbox:
                # Grad-CAM
                overlap_gradcam_bbox = compute_overlap(gradcam1_bin, bbox_map)
                # guided-backprop
                overlap_guidedbackprop_bbox = compute_overlap(guidedbackprop_bin, bbox_map)
                overlap_bbox_gb_bbox = compute_overlap(bbox_gb_map, bbox_map)
                # mask
                overlap_mask_bbox = compute_overlap(mask_bin, bbox_map)
                # compute overlap between guided and gradcam bbox
                overlap_gradcam_guidedbackprop = compute_overlap(gradcam1_bin, guidedbackprop_bin)
                overlap_gradcam_guidedbackprop_bbox = compute_overlap(gradcam1_bin, bbox_gb_map)


                # compute surface of the bounding box
                surface_bbox = np.count_nonzero(bbox_map)

                # store in dataframe
                stats = stats.append({'image_id':image_id,
                                    'gt_class_ids':gt_class_ids,
                                    'gt_class_names':gt_class_names,
                                    'pred_class_id':pred_class_id,
                                    'pred_class_name':pred_class_name,
                                    'TP':TP,
                                    'consistency':-consistency_loss[0],
                                    'BCE':float(BCE),
                                    'surface_bbox':surface_bbox,
                                    'overlap_gradcam_bbox':overlap_gradcam_bbox,
                                    'overlap_guidedbackprop_bbox':overlap_guidedbackprop_bbox,
                                    'overlap_bbox_gb_bbox': overlap_bbox_gb_bbox,
                                    'overlap_mask_bbox':overlap_mask_bbox,
                                    'overlap_gradcam_bbox/surface_bbox':overlap_gradcam_bbox/surface_bbox,
                                    'overlap_gradcam_guidedbackprop':overlap_gradcam_guidedbackprop,
                                    'overlap_gradcam_guidedbackprop_bbox':overlap_gradcam_guidedbackprop_bbox},
                                    ignore_index=True)
            else:
                # store in dataframe
                stats = stats.append({'image_id':image_id,
                                    'gt_class_ids':gt_class_ids,
                                    'gt_class_names':gt_class_names,
                                    'pred_class_id':pred_class_id,
                                    'pred_class_name':pred_class_name,
                                    'TP':TP,
                                    'consistency':-consistency_loss[0],
                                    'BCE':float(BCE)},
                                    ignore_index=True)

            # clear figures
            plt.close('all')

    # save dataframes
    stats.to_csv(os.path.join(batchDirectory, 'stats.csv'), index=False)
    df_gt.to_csv(os.path.join(batchDirectory, 'gt.csv'), index=False)
    df_pred.to_csv(os.path.join(batchDirectory, 'predictions.csv'), index=False)

    print("Finished Evaluation")

def visualizeTransformerMasking(model, data_loader, device, lossInstance, patch_size=28, batchDirectory = '', print_images=False):
    # set model to eval mode
    model.eval()
    if print_images:
        path_save_images = os.path.join(batchDirectory,'images')
        if not os.path.exists(path_save_images):
            os.makedirs(path_save_images)
    with torch.set_grad_enabled(False):
        num_different_preds = defaultdict(int)
        for image_id, data in enumerate(data_loader, 0):
            if image_id % 10 == 0:
                print(image_id)
            # visualize = True if image_id == 0 else False
            visualize = True
            inputs, labels, _ = data
            labels = labels[0].to(device)
            m = nn.Sigmoid()
            if visualize:
                save_image(inputs[0], f"{path_save_images}/orig.png")
            num_patches_x = inputs.shape[-2] // patch_size
            num_patches_y = inputs.shape[-1] // patch_size

            if inputs.shape[0] > 1:
                print("Warning: Batch size is greater than 1 for masking")
            masked = torch.zeros(num_patches_x * num_patches_y, *(inputs.shape[1:]))
            # Creating tensor of all individual patches
            for i in range(0, num_patches_x * patch_size, patch_size):
                for j in range(0, num_patches_y * patch_size, patch_size):
                    masked[i * num_patches_y // patch_size + j // patch_size,:,i:i+patch_size,j:j+patch_size] = inputs[0,:,i:i+patch_size,j:j+patch_size]
            masked = masked.to(device)
            # get logit
            logits = model(masked)
            true_class = torch.argmax(labels)
            attn_map = torch.exp(logits[:,true_class].view(num_patches_x, num_patches_y))
            attn_map = ((attn_map - attn_map.min()) / attn_map.max()) * 255.
            if visualize:
                save_image(inputs[0], f"{path_save_images}/{image_id}.png")
                im = Image.fromarray(attn_map.cpu().numpy()).convert('RGB')
                im.save(f"{path_save_images}/{image_id}_map.png")

                # for i in range(masked.shape[0]):
                #     save_image(masked[i], f"{path_save_images}/{image_id}_{i}.png")
            
            # Getting number of different predictions for the same base image based on different patches
        #     preds = torch.argmax(logits, dim=1)
        #     class_occurences = torch.bincount(preds)
        #     num_different_preds[torch.nonzero(class_occurences).shape[0]] += 1
        # print(num_different_preds)
        # torch.save(num_different_preds, os.path.join(batchDirectory,'num_different_preds.pt'))
                    

