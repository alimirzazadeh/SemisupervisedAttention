#!/bin/bash
#
#BATCH --job-name=exp_test
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -e /scratch/groups/rubin/krish05m/saved_batches/exp_test.err
#SBATCH -o /scratch/users/alimirz1/saved_batches/exp_test.out
export BATCH_DIRECTORY=exp_test
export LEARNING_RATE=0.000001
export NUM_EPOCHS=50
export BATCH_SIZE=4
export RESOLUTION_MATCH=1
export SIMILARITY_METRIC=1
export ALPHA=8

mkdir /scratch/groups/rubin/krish05m/saved_batches/$BATCH_DIRECTORY
ml python/3.9.0
ml opencv/4.5.2


python3 main.py loadCheckpoint noVisualLoss train noTrackLoss alternating $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA
