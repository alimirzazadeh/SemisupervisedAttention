#!/bin/bash
#
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null
export BATCH_DIRECTORY=laso_sup4_rebuttal_continued2
export TO_LOAD_CHECKPOINT=/scratch/groups/rubin/alimirz1/saved_batches/laso_sup4_rebuttal_continued/saved_checkpoints/model_best_mAP.pt
export NUM_FIGURES_TO_CREATE=None
export TO_TRAIN=True
export TO_EVALUATE=False
export WHICH_TRAINING=supervised
export LEARNING_RATE=0.000008
export NUM_EPOCHS=1500
export BATCH_SIZE=8
export RESOLUTION_MATCH=2
export SIMILARITY_METRIC=0
export ALPHA=8
export UNSUP_BATCH_SIZE=4
export FULLY_BALANCED=True
export USE_NEW_UNSUPERVISED=False
export UNSUP_DATASET_SIZE=None
export NUM_OUTPUT_CLASSES=20
export REFLECT_PADDING=False
export PER_BATCH_EVAL=None
export SAVE_RECURRING_CHECKPOINT=None
export NUM_IMAGES_PER_CLASS=4
export MASK_INTENSITY=8
ml python/3.9.0
ml opencv/4.5.2
python3 main.py $TO_LOAD_CHECKPOINT $NUM_FIGURES_TO_CREATE $TO_TRAIN $TO_EVALUATE $WHICH_TRAINING $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA $UNSUP_BATCH_SIZE $FULLY_BALANCED $USE_NEW_UNSUPERVISED $UNSUP_DATASET_SIZE $NUM_OUTPUT_CLASSES $REFLECT_PADDING $PER_BATCH_EVAL $SAVE_RECURRING_CHECKPOINT $NUM_IMAGES_PER_CLASS $MASK_INTENSITY
