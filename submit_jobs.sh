#!/bin/bash

# NUM_IMGS=(8)
# BATCH_DIRECTORY=vit_u
# FULLY_BALANCED=True
# ATTENTION_METHOD=8
# IG_STEPS=5
# for i in ${!NUM_IMGS[@]};
# do  
#     N=${NUM_IMGS[$i]}
#     DIR="${BATCH_DIRECTORY}${ATTENTION_METHOD}${N}6_2"
#     LOAD_CHECKPOINT_PATH="/scratch/groups/rubin/mpike27/saved_batches/vit_s"${N}"_longer/saved_checkpoints/model_best_mAP.pt"
#     echo $DIR
#     bash generalized_script.sh $DIR $FULLY_BALANCED $N $ATTENTION_METHOD $LOAD_CHECKPOINT_PATH $IG_STEPS
# done

NUM_IMGS=2
BATCH_DIRECTORY=vit_u
FULLY_BALANCED=True
ATTENTION_METHOD=8
N=2
DIR="${BATCH_DIRECTORY}${ATTENTION_METHOD}${N}6_cc"
LOAD_CHECKPOINT_PATH="/scratch/groups/rubin/mpike27/saved_batches/vit_s"${N}"_longer/saved_checkpoints/model_best_mAP.pt"
echo $DIR
bash generalized_script.sh $DIR $FULLY_BALANCED $N $ATTENTION_METHOD $LOAD_CHECKPOINT_PATH 1
DIR="${BATCH_DIRECTORY}${ATTENTION_METHOD}${N}6_ssim"
bash generalized_script.sh $DIR $FULLY_BALANCED $N $ATTENTION_METHOD $LOAD_CHECKPOINT_PATH 2
