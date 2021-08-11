# Project Description

**Goal**: This is a semi-supervised learning approach to improving video classification performance for hospital video. <br>
**Current Progress**: Working on using this for video data, outperforms supervised benchmark on Pascal/CIFAR

# Description of Running Ali's code #

## Batch Script ##

How to Observe Testing Loss throughout Training:
python3 visualizer/loss_visualizer.py

How to save a training iteration:
python3 project_saver.py


BATCH SCRIPT TO RUN:

```
#!/bin/bash
#
#BATCH --job-name=exp77
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
export BATCH_DIRECTORY=${SLURM_JOB_NAME}
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
export LOGFILE="$BATCH_DIRECTORY/${SLURM_JOB_NAME}.out"
ml python/3.9.0
ml opencv/4.5.2
python3 main.py $TO_LOAD_CHECKPOINT $NUM_FIGURES_TO_CREATE $TO_TRAIN $TO_EVALUATE $WHICH_TRAINING $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA $UNSUP_BATCH_SIZE $FULLY_BALANCED $USE_NEW_UNSUPERVISED $UNSUP_DATASET_SIZE $NUM_OUTPUT_CLASSES $REFLECT_PADDING $PER_BATCH_EVAL $SAVE_RECURRING_CHECKPOINT $NUM_IMAGES_PER_CLASS $MASK_INTENSITY &> ${LOGFILE}
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
