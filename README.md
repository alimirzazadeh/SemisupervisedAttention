This is a semi-supervised learning approach to improving video classification performance.
For Hospital Video.

How To Run:
python3 main.py [loadCheckpoint/noLoadCheckpoint] [visualLoss/noVisualLoss] [train/noTrain] [trackLoss/noTrackLoss] [supervised/unsupervised/alternating]

How To Train:
run main with either Checkpoint, noVisualLoss, train, noTrackLoss

How to Run Model on the Testing Set and save figures:
run main with Load certain Checkpoint, visualLoss, noTrain, noTrackLoss

How to Observe Testing Loss throughout Training:
python3 visualizer/loss_visualizer.py

How to save a training iteration:
python3 project_saver.py


BATCH SCRIPT TO RUN:

```
#!/bin/bash
#
#BATCH --job-name=sz_pred_preproc
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -e /scratch/groups/rubin/krish05m/AttentionMap/saved_batches/exp14.err
#SBATCH -o /scratch/groups/rubin/krish05m/AttentionMap/saved_batches/exp14.out
export BATCH_DIRECTORY=exp14
export LEARNING_RATE=0.000005
export NUM_EPOCHS=50
export BATCH_SIZE=4
export RESOLUTION_MATCH=1
export SIMILARITY_METRIC=1
export ALPHA=8
mkdir /scratch/groups/rubin/krish05m/AttentionMap/saved_batches/$BATCH_DIRECTORY
ml python/3.9.0
ml opencv/4.5.2

python3 main.py noloadCheckpoint novisualoss train notrackloss alternating $BATCH_DIRECTORY $LEARNING_RATE $NUM_EPOCHS $BATCH_SIZE $RESOLUTION_MATCH $SIMILARITY_METRIC $ALPHA
```

```
