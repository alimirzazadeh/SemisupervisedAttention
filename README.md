# ATCON: Attention Consistency for Vision Models

Attention--or attribution--maps methods are methods designed to highlight regions of the model’s input that were discriminative for its predictions. However, different attention maps methods can highlight different regions of the input, with sometimes contradictory explanations for a prediction. This effect is exacerbated when the training set is small. This indicates that either the model learned incorrect representations or that the attention maps methods did not accurately estimate the model’s representations. We propose an unsupervised fine-tuning method that optimizes the consistency of attention maps and show that it improves both classification performance and the quality of attention maps. We propose an implementation for two state-of-the-art attention computation methods, Grad-CAM and Guided Backpropagation, which relies on an input masking technique. We also show results on Grad-CAM and Integrated Gradients in an ablation study. We evaluate this method on our own dataset of event detection in continuous video recordings of hospital patients aggregated and curated for this work. As a sanity check, we also evaluate the proposed method on PASCAL VOC and SVHN. With the proposed method, with small training sets, we achieve a 6.6 points lift of F1 score over the baselines on our video dataset, a 2.9 point lift of F1 score on PASCAL, and a 1.8 points lift of mean Intersection over Union over Grad-CAM for weakly supervised detection on PASCAL. Those improved attention maps may help clinicians better understand vision model predictions and ease the deployment of machine learning systems into clinical care.

For more information about ATCON, please read the following [paper](https://arxiv.org/pdf/2210.09705.pdf):

    Mirzazadeh, A., Dubost, F., Pike, M., Maniar, K., Zuo, M., Lee-Messer, C. and Rubin, D., 2022. 
    ATCON: Attention Consistency for Vision Models. WACV.

Please also cite this paper if you are using nnU-Net for your research!

# Description of the code #

## Batch Script ##

How to Observe Testing Loss throughout Training:
python3 visualizer/loss_visualizer.py

How to save a training iteration:
python3 project_saver.py


BATCH SCRIPT TO RUN:

```
#!/bin/bash
#
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
export BATCH_DIRECTORY=exp_88
export TO_LOAD_CHECKPOINT=False
export NUM_FIGURES_TO_CREATE=None
export TO_TRAIN=True
export TO_EVALUATE=True
export WHICH_TRAINING=alternating
export LEARNING_RATE=0.000005
export NUM_EPOCHS=50
export BATCH_SIZE=4
export RESOLUTION_MATCH=2
export SIMILARITY_METRIC=0
export ALPHA=8
export UNSUP_BATCH_SIZE=4
export FULLY_BALANCED=True
export USE_NEW_UNSUPERVISED=True
export UNSUP_DATASET_SIZE=None
export NUM_OUTPUT_CLASSES=20
export REFLECT_PADDING=True
export PER_BATCH_EVAL=None
export SAVE_RECURRING_CHECKPOINT=None
export NUM_IMAGES_PER_CLASS=2
export MASK_INTENSITY=8
ml python/3.9.0
ml opencv/4.5.2
python3 main.py $TO_LOAD_CHECKPOINT $NUM_FIGURES_TO_CREATE $TO_TRAIN $TO_EVALUATE $WHICH_TRAINING $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA $UNSUP_BATCH_SIZE $FULLY_BALANCED $USE_NEW_UNSUPERVISED $UNSUP_DATASET_SIZE $NUM_OUTPUT_CLASSES $REFLECT_PADDING $PER_BATCH_EVAL $SAVE_RECURRING_CHECKPOINT $NUM_IMAGES_PER_CLASS $MASK_INTENSITY
```

```
python3 main.py noloadCheckpoint noVisualLoss train notrackLoss supervised $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA
```

## Running Command in batch script ##

python3 main.py ```[loadCheckpoint/noLoadCheckpoint]``` ```[visualLoss/noVisualLoss]``` ```[train/noTrain]``` ```[trackLoss/noTrackLoss]``` ```[supervised/unsupervised/alternating]```

## Parameter Description ##

Warm vs Cold start
- Use ```loadCheckpoint``` to load in a checkpoint for warm start. Specify the path in the ```main.py``` file
- Use ```noLoadCheckpoint``` if you want don't want a warm start.

Visualizing Attention Maps
- Use ```visualLoss``` if you want to compare various attention maps for a certain batch of images
- Typically don't run ```visualLoss``` and ```train``` together

Training Model
- Use ```train``` to train the model on a certain number of epochs
- ```noTrain``` if you don't want to train the model at all

Type of training
- There are three types of training (supervised, unsupervised, and alternating)
- Only need to specify the ```alpha``` value for ```alternating```

Use ```trackLoss``` to track the loss throughout the training
