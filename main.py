import torch.optim as optim
from train import train
from evaluate import evaluate
from metrics.UnsupervisedMetrics import visualizeLossPerformance
from visualizer.visualizer import visualizeImageBatch, show_cam_on_image
# from data_loader.new_pascal_runner import loadPascalData
# from data_loader.cifar_data_loader import loadCifarData
from data_loader.train_videossl import loadVideoData
from ipdb import set_trace as bp
import os
import numpy as np
from torch import nn
from model.model import r3d_18
import torchvision.transforms as transforms
import torchvision
import torch
import cv2
import sys
sys.path.append("./")
import json
import distutils.util


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if __name__ == '__main__':
    toLoadCheckpoint = bool(distutils.util.strtobool(sys.argv[1]))
    toTrain = bool(distutils.util.strtobool(sys.argv[3]))
    toEvaluate = bool(distutils.util.strtobool(sys.argv[4])) #True
    whichTraining = sys.argv[5] #alternating
    batchDirectoryFile = sys.argv[6]
    learning_rate = float(sys.argv[7])  # 0.00001
    numEpochs = int(sys.argv[8])  # 400
    batch_size = int(sys.argv[9])  # 4
    resolutionMatch = int(sys.argv[10])  # 2
    similarityMetric = int(sys.argv[11])  # 2
    alpha = float(sys.argv[12])  # 2
    unsup_batch_size = int(sys.argv[13]) #12
    fullyBalanced = bool(distutils.util.strtobool(sys.argv[14])) #True
    useNewUnsupervised=bool(distutils.util.strtobool(sys.argv[15])) #True
    numOutputClasses = int(sys.argv[17]) #20
    reflectPadding = bool(distutils.util.strtobool(sys.argv[18])) #True
    numImagesPerClass = int(sys.argv[21]) #2
    maskIntensity = int(sys.argv[22]) #8

    try:
        unsupDatasetSize=int(sys.argv[16]) #None
    except:
        unsupDatasetSize=None
    try:
        numFiguresToCreate=int(sys.argv[2]) #None
    except:
        numFiguresToCreate=None
    try:
        perBatchEval=int(sys.argv[19]) #None to do per epoch eval
    except:
        perBatchEval=None
    try:
        saveRecurringCheckpoint=int(sys.argv[20])
    except:
        saveRecurringCheckpoint=None

    #json contains the paths required to launch on sherlock
    with open('./zaman_launch.json') as f:
        zaman_json = json.load(f)

    #checks if on zaman, otherwise creates folder in the batchDirectory in home repo (usually when running on cpu)
    if os.path.isdir('/home/alimirz1'):
        batchDirectory = zaman_json['batch_directory_path'] + \
            batchDirectoryFile + '/'
    else:
        batchDirectory = batchDirectoryFile + '/'


    CHECK_FOLDER = os.path.isdir(batchDirectory)
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory)
    
    log = open(batchDirectory + "log.out", "a")
    sys.stdout = log
    sys.stderr = log
    
    print('############## Run Settings: ###############')

    print(sherlock_json)

    CHECK_FOLDER = os.path.isdir(batchDirectory + "saved_figs")
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory + "saved_figs")

    CHECK_FOLDER = os.path.isdir(batchDirectory + "saved_checkpoints")
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory + "saved_checkpoints")


    print('Loading Checkpoint: ', toLoadCheckpoint)
    print('Training: ', toTrain)
    print('Evaluating: ', toEvaluate)
    print('Training Mode: ', whichTraining)
    print('Batch Directory: ', batchDirectoryFile)
    print('Learning Rate: ', learning_rate)
    print('Number of Epochs: ', numEpochs)
    print('Batch Size: ', batch_size)
    print('Resolution Match Mode: ', resolutionMatch)
    print('Similarity Metric Mode: ', similarityMetric)
    print('Alpha: ', alpha)
    print('Unsupervised Batch Size: ', unsup_batch_size)
    print('Fully Balanced Supervised Dataset: ', fullyBalanced)
    print('Number of Images per Class in Supervised Dataset: ', numImagesPerClass)
    print('Using New Unsupervised Data', useNewUnsupervised)
    print('Number of output classes: ', numOutputClasses)
    print('Using Reflecting Padding: ', reflectPadding)
    print('UnsupervisedDatasetSize (Everything if None): ', unsupDatasetSize)
    print('Number of Figures to create: ', numFiguresToCreate)
    print('Evaluating Per How Many Batches (Per Epoch if None): ', perBatchEval)
    print('Saving Recurring Checkpoints (Only best checkpoints if None): ', saveRecurringCheckpoint)
    print('Mask Intensity: ', maskIntensity)

    print('########################################### \n\n')

    
    suptrainloader, unsuptrainloader, validloader, testloader = loadVideoData(
        batch_size=batch_size, unsup_batch_size=unsup_batch_size, useNewUnsupervised=useNewUnsupervised)

    CHECK_FOLDER = os.path.isdir(batchDirectory + "saved_figs")
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory + "saved_figs")
        print("Made Saved_Figs folder")

    CHECK_FOLDER = os.path.isdir(batchDirectory + "saved_checkpoints")
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory + "saved_checkpoints")
        print("Made Saved_Checkpoints folder")

    # model = models.resnet50(pretrained=True)
    #model =  torchvision.models.video.r3d_18(pretrained=True)
    model = r3d_18(pretrained=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.fc = nn.Linear(int(model.fc.in_features), 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None

    all_checkpoints = os.listdir('saved_checkpoints')
    epoch = 0

    print(model.fc.weight)

    def loadCheckpoint(path, model):
        checkpoint = torch.load(path, map_location=device)
        try:
            model.load_state_dict(checkpoint)
        except:
            model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            epoch = checkpoint['epoch']
        except:
            epoch = 0
        
    if sys.argv[1] != 'noLoadCheckpoint':
        whichCheckpoint = 6

        if os.path.isdir('/scratch/'):
            # PATH = '/scratch/users/alimirz1/saved_batches/...'
            # PATH = '/scratch/users/alimirz1/saved_batches/exp_11/saved_checkpoints/model_59.pt'
            # PATH = '/scratch/users/alimirz1/saved_batches/pre_gradFix/savingAfter15Sup/saved_checkpoints/model_14.pt'
            # PATH = '/scratch/users/alimirz1/saved_batches/hot_bench/saved_checkpoints/model_best.pt'
            PATH = '/scratch/users/alimirz1/saved_batches/hot_bench_2s_saved/saved_checkpoints/' + sys.argv[1]
        elif os.path.isdir('/home/alimirz1'):
            print('in here')
            #PATH = '/home/alimirz1/SemisupervisedAttention/saved_batches/exp_22/saved_checkpoints/model_best.pt'

            PATH = zaman_json['load_checkpoint_path']
            PATH2 = zaman_json['load_figure_comparison_checkpoint_path']
        else:
            # + all_checkpoints[whichCheckpoint]
            PATH = 'saved_checkpoints/7_31_21/model_best_alt.pt'
            PATH2 = 'saved_checkpoints/7_31_21/model_best_sup.pt'
        
        loadCheckpoint(PATH, model)
                # loss = checkpoint['loss']

    target_layer = model.layer4[-1]  # this is the layer before the pooling

    print(model.fc.weight)
    # model.conv1.padding_mode = 'reflect'
    # model.stem[0].padding_mode = 'reflect'
    # for x in model.layer1:
    #     x.conv1[0].padding_mode = 'reflect'
    #     x.conv2[0].padding_mode = 'reflect'
    # for x in model.layer2:
    #     x.conv1[0].padding_mode = 'reflect'
    #     x.conv2[0].padding_mode = 'reflect'
    # for x in model.layer3:
    #     x.conv1[0].padding_mode = 'reflect'
    #     x.conv2[0].padding_mode = 'reflect'
    # for x in model.layer4:
    #     x.conv1[0].padding_mode = 'reflect'
    #     x.conv2[0].padding_mode = 'reflect'

    use_cuda = torch.cuda.is_available()
    # load a few images from CIFAR and save
    if sys.argv[2] == 'visualLoss':
        from model.loss import CAMLoss
        
        
        CAMLossInstance = CAMLoss(
            model, target_layer, use_cuda, resolutionMatch, similarityMetric)
        dataiter = iter(validloader)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.eval()
        idx2label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        for i in range(15):
            images, labels = dataiter.next()
            images = images.to(device)
            labels = labels.to(device)
            # images.to("cpu")
            model.to(device)
            with torch.no_grad():
                # calculate outputs by running images through the network
                loadCheckpoint(PATH, model)
                outputs = model(images)
                loadCheckpoint(PATH2, model)
                outputs2 = model(images)
                # the class with the highest energy is what we choose as prediction
                val, predicted = torch.max(outputs.data, 1)
                mean = torch.mean(outputs.data, 1)
                val2, predicted2 = torch.max(outputs2.data, 1)
                mean2 = torch.mean(outputs2.data, 1)
                
                # for pred in range(predicted.shape[0]):
                #     running_corrects += labels[pred, int(predicted[pred])]
                    
                predicted = predicted.tolist()
                predicted2 = predicted2.tolist()
                print(predicted, predicted2)
                predictedNames = [idx2label[p] for p in predicted]
                labels = labels.tolist()
                actualLabels = labels == predicted
                predictedNames2 = [idx2label[p] for p in predicted]
                actualLabels2 = labels == predicted
                print("DIFFERING!", actualLabels, actualLabels2)
                print("\n\n Val: ", val, val2, "\n\n")
                print("\n\n Mean: ", mean, mean2, "\n\n")
                    
            
            if predicted == predicted2:
                loadCheckpoint(PATH, model)
                imgTitle = "which_0_epoch_" + str(epoch) + "_batchNum_" + str(i)
                visualizeLossPerformance(
                    CAMLossInstance, images, labels=actualLabels, imgTitle=imgTitle, imgLabels=predictedNames)
                loadCheckpoint(PATH2, model)
                imgTitle = "which_1_epoch_" + str(epoch) + "_batchNum_" + str(i)
                visualizeLossPerformance(
                    CAMLossInstance, images, labels=actualLabels2, imgTitle=imgTitle, imgLabels=predictedNames2)

    # visualizeImageBatch(images, labels)

    target_category = None

    # need to set params?

    # model.fc = nn.Linear(int(model.fc.in_features), 10)

    print("done")

    whichTraining = sys.argv[5]
    if whichTraining not in ['supervised', 'unsupervised', 'alternating']:
        print('invalid Training. will alternate')
        whichTraining = 'alternating'
    if sys.argv[3] == 'train':
        trackLoss = sys.argv[4] == 'trackLoss'
        print(trackLoss)
        train(model, numEpochs, suptrainloader, unsuptrainloader, validloader, optimizer, target_layer, target_category, use_cuda, resolutionMatch,
              similarityMetric, alpha, trackLoss=trackLoss, training=whichTraining, batchDirectory=batchDirectory, scheduler=scheduler, batch_size=batch_size)
        print("Training Complete. Evaluating on Test Set...")
        checkpoint = torch.load(batchDirectory + "saved_checkpoints/model_best.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(model, testloader, device, batchDirectory=batchDirectory)