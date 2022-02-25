#!/bin/bash

BATCH_DIRECTORY=u226_m10_multi
NUM_JOBS=4
FULLY_BALANCED=True
NUM_IMAGES_PER_CLASS=2
ATTENTION_METHOD=2
LOAD_CHECKPOINT_PATH=/scratch/groups/rubin/alimirz1/saved_batches/mAP_bench4_s2_aug/saved_checkpoints/model_best_mAP.pt
IG_STEPS=10
for (( c=1; c<=$NUM_JOBS; c++ ))
do 
    DIR="${BATCH_DIRECTORY}/${c}"
    # echo $DIR
    sbatch generalized_script.sh $DIR $FULLY_BALANCED $NUM_IMAGES_PER_CLASS $ATTENTION_METHOD $LOAD_CHECKPOINT_PATH $IG_STEPS
done