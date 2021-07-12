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
