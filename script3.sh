#!/bin/bash
#
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -C GPU_MEM:16GB
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null
export BATCH_DIRECTORY=vit_s2_longer
export TO_LOAD_CHECKPOINT=False
export NUM_FIGURES_TO_CREATE=None
export TO_TRAIN=True
export TO_EVALUATE=True
export WHICH_TRAINING=supervised
export LEARNING_RATE=.000001
export NUM_EPOCHS=250
export BATCH_SIZE=4
export RESOLUTION_MATCH=2
export SIMILARITY_METRIC=0
export ALPHA=1
export UNSUP_BATCH_SIZE=4
export FULLY_BALANCED=True
export USE_NEW_UNSUPERVISED=False
export UNSUP_DATASET_SIZE=None
export NUM_OUTPUT_CLASSES=20
export REFLECT_PADDING=True
export PER_BATCH_EVAL=None
export SAVE_RECURRING_CHECKPOINT=None
export NUM_IMAGES_PER_CLASS=2
export MASK_INTENSITY=100
export ATTENTION_METHOD=8
export THETA=0.8
export REDIRECT_OUTPUT=True
export BATCH_DIRECTORY_PATH=/scratch/groups/rubin/mpike27/saved_batches/
export LOAD_CHECKPOINT_PATH=/scratch/groups/rubin/mpike27/saved_batches/vit_s2_longer/saved_checkpoints/model_best_mAP.pt
# export LOAD_CHECKPOINT_PATH=/scratch/groups/rubin/mpike27/saved_batches//saved_checkpoints/model_best_mAP.pt
export LOAD_FIGURE_COMPARISON_CHECKPOINT_PATH=/scratch/groups/rubin/mpike27/saved_batches/vit_u276/saved_checkpoints/model_best_mAP.pt
export PRINT_IMAGES=False 
export PRINT_ATTENTION_MAPS=False
export IG_STEPS=5
export RANDOMIZED_SPLIT=False
export MODEL_TYPE=3
export NUM_WORKERS=1
ml python/3.9.0
ml opencv/4.5.2
python3 main.py $TO_LOAD_CHECKPOINT $NUM_FIGURES_TO_CREATE $TO_TRAIN $TO_EVALUATE $WHICH_TRAINING $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA $UNSUP_BATCH_SIZE $FULLY_BALANCED $USE_NEW_UNSUPERVISED $UNSUP_DATASET_SIZE $NUM_OUTPUT_CLASSES $REFLECT_PADDING $PER_BATCH_EVAL $SAVE_RECURRING_CHECKPOINT $NUM_IMAGES_PER_CLASS $MASK_INTENSITY $ATTENTION_METHOD $THETA $REDIRECT_OUTPUT $BATCH_DIRECTORY_PATH $LOAD_CHECKPOINT_PATH $LOAD_FIGURE_COMPARISON_CHECKPOINT_PATH $PRINT_IMAGES $PRINT_ATTENTION_MAPS $IG_STEPS $RANDOMIZED_SPLIT $MODEL_TYPE $NUM_WORKERS
