import torch.optim as optim
from train import train
from evaluate import evaluate, visualizeTransformerMasking
from metrics.UnsupervisedMetrics import visualizeLossPerformance
from data_loader.new_pascal_runner import loadPascalData
import os
import numpy as np
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch
import cv2
import sys
import json
import distutils.util
from shutil import copyfile
import shutil
from model.layer_attention_loss import isLayerWiseAttention
from model.transformer_loss import isTransformer, TransformerLoss
from pytorch_pretrained_vit import ViT
sys.path.append("./")


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
    attentionMethod = int(sys.argv[23]) #0
    theta = float(sys.argv[24]) #0.8
    REDIRECT_OUTPUT = bool(distutils.util.strtobool(sys.argv[25]))
    batch_directory_path = sys.argv[26]
    load_checkpoint_path = sys.argv[27]
    load_figure_comparison_checkpoint_path = sys.argv[28]
    print_images = sys.argv[29]
    print_attention_maps = sys.argv[30]
    ig_steps = sys.argv[31]
    randomized_split = sys.argv[32]
    model_type = int(sys.argv[33])
    num_workers = int(sys.argv[34])

    
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
    with open('./sherlock_launch.json') as f:
        sherlock_json = json.load(f)

    #checks if on sherlock, otherwise creates folder in the batchDirectory in home repo (usually when running on cpu)
    if os.path.isdir('/scratch/'):
        batchDirectory = batch_directory_path + \
            batchDirectoryFile + '/'
    else:
        batchDirectory = batchDirectoryFile + '/'


    CHECK_FOLDER = os.path.isdir(batchDirectory)
    if CHECK_FOLDER:
        shutil.rmtree(batchDirectory)
    os.makedirs(batchDirectory)
    
    if REDIRECT_OUTPUT:
        log = open(batchDirectory + "log.out", "a")
        sys.stdout = log
        sys.stderr = log
    
    print('############## Run Settings: ###############')

    # print(sherlock_json)


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
    print('Attention Method: ', attentionMethod)
    print('Theta: ', theta)
    print('Redirect output: ', REDIRECT_OUTPUT)
    print('batch_directory_path: ', batch_directory_path)
    print('load_checkpoint_path: ', load_checkpoint_path)
    print('load_figure_comparison_checkpoint_path: ', load_figure_comparison_checkpoint_path)
    print('print_images: ', print_images)
    print('print_attention_maps: ', print_attention_maps)
    print('ig_steps: ', ig_steps)
    print('randomized_split: ', randomized_split)
    print('model_type: ', model_type)
    print('num_workers: ', num_workers)

    print('########################################### \n\n')

    copyfile("script3.sh", batchDirectory + "script3.sh")

    if model_type == 0:
        model = models.resnet50(pretrained=True)
    elif model_type == 1:
        model = models.densenet161(pretrained=False)
    elif model_type == 2:
        model = models.inception_v3(pretrained=False)
    elif model_type == 3:
        model = ViT('B_16', pretrained=True)

    image_size_resize = 224 if isTransformer(attentionMethod) else 256

    suptrainloader, unsuptrainloader, validloader, testloader, evaluationLoader = loadPascalData(
        numImagesPerClass, batch_size=batch_size, unsup_batch_size=unsup_batch_size, 
        fullyBalanced=fullyBalanced, useNewUnsupervised=useNewUnsupervised, 
        unsupDatasetSize=unsupDatasetSize, randomized=randomized_split, image_size_resize=image_size_resize, num_workers=num_workers)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    try:
        model.fc = nn.Linear(int(model.fc.in_features), numOutputClasses)
    except:
        model.classifier = nn.Linear(int(model.classifier.in_features), numOutputClasses)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None

    epoch = 0


    def loadCheckpoint(path, model):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            epoch = checkpoint['epoch']
        except:
            epoch = 0
        
    if toLoadCheckpoint:
        if os.path.isdir('/scratch/'):
            PATH = load_checkpoint_path
            ##wont load path2 unless numFiguresToCreate is not None
            PATH2 = load_figure_comparison_checkpoint_path
        else:
            PATH = 'saved_checkpoints/hot_bench_150s_model_best.pt'
            PATH2 = 'saved_checkpoints/hot_bench_150s_model_best.pt'

        # print(model.fc.weight)
        loadCheckpoint(PATH, model)
        ## Sanity check to make sure the loaded weights are different after loading chekcpoint
        print(model.fc.weight)


    ## WHICH LAYER FOR GRADCAM
    if model_type == 0:
        if isLayerWiseAttention(attentionMethod):
            target_layer = [model.layer4[-1].conv3, model.layer4[-2].conv3]
        else:
            target_layer = model.layer4[-1]  # this is the layer before the pooling
    elif model_type == 1:
        target_layer = model.features.denseblock4[-1]
    elif model_type == 2:
        target_layer = model.Mixed_7c.branch3x3dbl_3b
    else:
        target_layer = None
    
    if reflectPadding:
        def _freeze_norm_stats(net):
            try:
                for m in net.modules():
                    if isinstance(m, nn.Conv2d):
                        m.padding_mode = 'reflect'
                        # print('ha')
        
            except ValueError:  
                print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
                return
        model.apply(_freeze_norm_stats)

    use_cuda = torch.cuda.is_available()
    # use_cuda = False

    target_category = None

    if toTrain and whichTraining not in ['supervised', 'unsupervised', 'alternating', 'combining']:
        print('invalid Training. Choose between supervised, unsupervised, alternating')
        sys.exit()
    if toTrain:
        print('Beginning Training')
        train(model, numEpochs, suptrainloader, unsuptrainloader, validloader, optimizer, target_layer, target_category, use_cuda, resolutionMatch,
              similarityMetric, alpha, theta, training=whichTraining, batchDirectory=batchDirectory, batch_size=batch_size, 
              unsup_batch_size=unsup_batch_size, perBatchEval=perBatchEval, saveRecurringCheckpoint=saveRecurringCheckpoint, maskIntensity=maskIntensity, 
              attentionMethod=attentionMethod, ig_steps=ig_steps)
        print("Training Complete.")

    # if isTransformer(attentionMethod) and toEvaluate: #Temporary to visualize impact of masking on transformer preds
    #     lossInstance = TransformerLoss(model, use_cuda)
    #     visualizeTransformerMasking(model, evaluationLoader, device, lossInstance, batchDirectory=batchDirectory, print_images=True)
    #     exit()
    # elif isTransformer(attentionMethod): #Transformer Loss not implemented fully yet, so cannot evaluate or create figures
    #     print("Evaluation is not yet implemented for Visual Transformer")
    #     exit()

    # load a few images from CIFAR and save
    if numFiguresToCreate is not None or toEvaluate:
        # if isLayerWiseAttention(attentionMethod):
        #     from model.layer_attention_loss import LayerAttentionLoss
        #     lossInstance = LayerAttentionLoss(model, target_layer, use_cuda, maskIntensity, theta, attentionMethod)
        # else:
        from model.loss import CAMLoss        
        lossInstance = CAMLoss(
            model, target_layer, use_cuda, resolutionMatch, similarityMetric, maskIntensity, attentionMethod, ig_steps=ig_steps)
    
    if numFiguresToCreate is not None:
        dataiter = iter(validloader)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.eval()
        idx2label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        
        for i in range(numFiguresToCreate):
            images, labels = dataiter.next()
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # calculate outputs by running images through the network
                # print('1: ', torch.cuda.memory_allocated(0) / 1e9)
                if toLoadCheckpoint:
                    loadCheckpoint(PATH, model)
                outputs = model(images)
                if toLoadCheckpoint:
                    loadCheckpoint(PATH2, model)
                outputs2 = model(images)
                # the class with the highest energy is what we choose as prediction
                val, predicted = torch.max(outputs.data, 1)
                mean = torch.mean(outputs.data, 1)
                val2, predicted2 = torch.max(outputs2.data, 1)
                mean2 = torch.mean(outputs2.data, 1)
                
                predicted = predicted.tolist()
                predicted2 = predicted2.tolist()
                print(predicted, predicted2)

                predictedNames = [idx2label[p] for p in predicted]
                labels = labels.cpu().numpy()
                actualLabels = [labels[p, predicted[p]]
                                for p in range(len(predicted))]
                predictedNames2 = [idx2label[p] for p in predicted2]
                actualLabels2 = [labels[p, predicted2[p]]
                                for p in range(len(predicted2))]
                print("\n\n Val: ", val, val2, "\n\n")
                print("\n\n Mean: ", mean, mean2, "\n\n")
            
            ### To only create figures with differing predictions, wrap the following lines with if predicted != predicted2:
            if toLoadCheckpoint:
                loadCheckpoint(PATH, model)
            imgTitle = "which_0_epoch_" + str(epoch) + "_batchNum_" + str(i)
            visualizeLossPerformance(
                lossInstance, images, attentionMethod, labels=actualLabels, imgTitle=imgTitle, imgLabels=predictedNames, batchDirectory=batchDirectory)
            if toLoadCheckpoint:
                loadCheckpoint(PATH2, model)
            imgTitle = "which_1_epoch_" + str(epoch) + "_batchNum_" + str(i)
            visualizeLossPerformance(
                lossInstance, images, attentionMethod, labels=actualLabels2, imgTitle=imgTitle, imgLabels=predictedNames2, batchDirectory=batchDirectory)

    if toEvaluate:
        print("Evaluating on Test Set...")
        ##load the best checkpoint and evaulate it. 
        if toTrain:
            checkpoint = torch.load(batchDirectory + "saved_checkpoints/model_best.pt", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else: 
            loadCheckpoint(PATH, model)
        evaluate(model, evaluationLoader, device, lossInstance, attentionMethod, batchDirectory=batchDirectory,
                    print_images=print_images, print_attention_maps=print_attention_maps)
    print("Successfully Completed. Good bye!")