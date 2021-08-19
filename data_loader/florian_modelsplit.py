import pims
import os
import sys
import pandas as pd
import json
from basic_functions import createExpFolderandCodeList
from ipdb import set_trace as bp
import math
import random
import collections
import numpy as np


experiment_id = "003"
MAX_TRAINING_SAMPLE_PER_CLASS = 3 # None 3
PATH_EXPERIMENTS = '/home/alimirz1/babul/' #'/mnt/data/eegml/fdubost/experiments'


# parameters
DATA_TYPES = {'train': ['92'], 'dev': ['97','317'], 'test': ['95','330']} #{'train': ['92'], 'dev': ['97','317'], 'test': ['95','330']} {'107': 'train', '114': 'dev'}
LENGTH_SEQUENCE = 16 #15 1 8 16
STRIDE = 4
EXP_VIDEO_TALBE = '331' # 256 309 331
BALANCED = True
BALANCE_ONLY_TRAIN = False

OVERLAP = False

def shuffle_segments(segments, labels):
    zipped_for_shuffling = list(zip(segments, labels))
    random.shuffle(zipped_for_shuffling)
    segments, labels = zip(*zipped_for_shuffling)

    return segments, labels

# PATHS
# path experiment

# save experiment and code




root_savepath = os.path.join(PATH_EXPERIMENTS, experiment_id)
# create exp folder
createExpFolderandCodeList(root_savepath)

# read video table
with open(os.path.join(PATH_EXPERIMENTS,EXP_VIDEO_TALBE,'video_table.json')) as json_file:
    video_table = json.load(json_file)

# initialize dict
segment_info = {}
for data_type in DATA_TYPES:
    segment_info[data_type] = {'segments': [],'labels': []}
segment_info['unlabeled'] = {'segments': [],'labels': []}

# iterate through videos in video_table
for video_id in range(len(video_table)):
    # get video meta data
    curr_video_meta = video_table[video_id]

    # compute number of segments to extract
    if OVERLAP:
        num_segments = curr_video_meta['num_frames'] - LENGTH_SEQUENCE*STRIDE
    else:
        num_segments = math.floor(curr_video_meta['num_frames'] / (LENGTH_SEQUENCE * STRIDE))

    # iterate over segments
    for segment_id in range(num_segments):
        # get start and end frames
        if OVERLAP:
            start_frame = segment_id
            end_frame = segment_id + LENGTH_SEQUENCE*STRIDE
        else:
            start_frame = segment_id * LENGTH_SEQUENCE * STRIDE
            end_frame = (segment_id + 1) * LENGTH_SEQUENCE * STRIDE - 1

        # find which data type it belongs to
        for data_type in DATA_TYPES:
            if curr_video_meta['exp_folder'] in DATA_TYPES[data_type]:
                data_type_curr_video = data_type
                break

        # add segment data to dict
        segment_info[data_type_curr_video]['segments'].append(['vid'+curr_video_meta['id'], [start_frame,end_frame,STRIDE]])
        segment_info[data_type_curr_video]['labels'].append(curr_video_meta['label'])

# rebalance dataset
if BALANCED:
    # save left out segment for unsupervised training
    left_out_segments = []
    left_out_labels = []

    # set split to balance
    if BALANCE_ONLY_TRAIN:
        splits_to_balance = ['train']
    else:
        splits_to_balance = DATA_TYPES.keys()

    # iterate over data splits
    for split in splits_to_balance:
        # extract data
        segments = segment_info[split]['segments']
        labels = segment_info[split]['labels']

        # find minimum number of samples per class
        classes = set(labels)
        min_nbr_samples_per_class = min([len(np.array(labels)[np.array(labels) == curr_class]) for curr_class in classes])
        # compute number of samples per class to extract
        if MAX_TRAINING_SAMPLE_PER_CLASS is not None and split =='train':
            number_sample_per_class_to_extract = MAX_TRAINING_SAMPLE_PER_CLASS
        else:
            number_sample_per_class_to_extract = min_nbr_samples_per_class

        # shuffle data
        random.seed(10)
        segments, labels = shuffle_segments(segments, labels)

        # create new list with balanced data
        new_segments = []
        new_labels = []

        # record number of sample per class
        current_nbr_samples_per_class = collections.defaultdict(int)
        for segment, label in zip(segments,labels):
            # check that the current class is not already oversampled
            if current_nbr_samples_per_class[label] < number_sample_per_class_to_extract:
                # fill lists
                new_segments.append(segment)
                new_labels.append(label)
                # indent number of sample per class
                current_nbr_samples_per_class[label] += 1
            # otherwise save it for unlabeled samples
            elif split == 'train':
                left_out_segments.append(segment)
                left_out_labels.append(label)

        # shuffle data
        new_segments, new_labels = shuffle_segments(new_segments, new_labels)

        # replace segments and labels in dict
        segment_info[split]['segments'] = new_segments
        segment_info[split]['labels'] = new_labels

    # use remaining segments for unlabeled data
    # shuffle
    left_out_segments, left_out_labels = shuffle_segments(left_out_segments, left_out_labels)
    # add to dict
    segment_info['unlabeled']['segments'] = left_out_segments
    segment_info['unlabeled']['labels'] = left_out_labels


# save json
with open(os.path.join(PATH_EXPERIMENTS,experiment_id,'model_splits.json'), 'w') as outfile:
    json.dump(segment_info, outfile)






