import sys
sys.path.append("./")


import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn


import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from data_loader.cifar_data_loader import loadCifarData
from data_loader.pascal_runner import loadPascalData
from visualizer.visualizer import visualizeImageBatch, show_cam_on_image
from metrics.UnsupervisedMetrics import visualizeLossPerformance
# from model.loss import calculateLoss
from train import train
import torch.optim as optim

if __name__ == '__main__':

    # learning_rate = 0.000001
    learning_rate = 0.001
    numEpochs = 200
    batch_size = 4
    
    
    print('Learning Rate: ', learning_rate)
    print('Number of Epochs: ', numEpochs)



    if os.path.isdir('/scratch/'):
        batchDirectory = '/scratch/users/alimirz1/saved_batches/' + sys.argv[6] + '/'
    else:
        batchDirectory = ''
    ## Load the CIFAR Dataset
    suptrainloader,unsuptrainloader, testloader = loadPascalData(batch_size=batch_size)


    CHECK_FOLDER = os.path.isdir(batchDirectory + "saved_figs")
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory + "saved_figs")
        print("Made Saved_Figs folder")
    
    CHECK_FOLDER = os.path.isdir(batchDirectory + "saved_checkpoints")
    if not CHECK_FOLDER:
        os.makedirs(batchDirectory + "saved_checkpoints")
        print("Made Saved_Checkpoints folder")

    

    model = models.resnet50(pretrained = True)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # lr = [1e-5, 5e-3]
    # optimizer = optim.SGD([   
    #     {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
    #     {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
    #     ])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = None
    
    
    
    all_checkpoints = os.listdir('saved_checkpoints')
    epoch = 0
    
    if sys.argv[1] == 'loadCheckpoint':
        whichCheckpoint = 6
        if len(all_checkpoints) > 0:
            
            if os.path.isdir('/scratch/'):
                # PATH = '/scratch/users/alimirz1/saved_batches/...'
                PATH = '/home/users/alimirz1/SemisupervisedAttention/saved_checkpoints/resnet50-19c8e357.pth'
            else:
                PATH = 'saved_checkpoints/' + all_checkpoints[whichCheckpoint]

            print('Loading Saved Model', PATH)
            
            if True:
                net_state_dict = model.state_dict()
                pretrained_dict34 = torch.load(PATH,map_location=device)
                pretrained_dict_1 = {k: v for k, v in pretrained_dict34.items() if k in net_state_dict}
                net_state_dict.update(pretrained_dict_1)
                model.load_state_dict(net_state_dict)             
            else:
                checkpoint = torch.load(PATH, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                # loss = checkpoint['loss']
        
    model.fc = nn.Linear(int(model.fc.in_features), 20)
    
    target_layer = model.layer4[-1] ##this is the layer before the pooling


    model.conv1.padding_mode = 'reflect'
    for x in model.layer1:
        x.conv2.padding_mode = 'reflect'
    for x in model.layer2:
        x.conv2.padding_mode = 'reflect'
    for x in model.layer3:
        x.conv2.padding_mode = 'reflect'
    for x in model.layer4:
        x.conv2.padding_mode = 'reflect'


    use_cuda = torch.cuda.is_available()
    # load a few images from CIFAR and save
    if sys.argv[2] == 'visualLoss':
        from model.loss import CAMLoss
        CAMLossInstance = CAMLoss(model, target_layer, use_cuda)
        dataiter = iter(testloader)

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.eval()
        
        # def findWord(arr, idx2label):
        #     for item in arr:
        #         ones = np.argwhere(item.numpy())
        #         ones = [i[0] for i in ones]
        #         print(idx2label[ones[0]])

        # f = open("imagenet_class_index.json",)
        # class_idx = json.load(f)
        # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        
        # def showImg(arr):
        #     fig, axs = plt.subplots(1,4)
        #     counter = 0
        #     for item in arr:
        #         bb = np.moveaxis(arr[counter].numpy(),0,-1)
        #         bb -= np.min(bb)
        #         bb = bb/ np.max(bb)
        #         axs[counter].imshow(bb)
        #         counter += 1
        #     plt.show()
        
        
        idx2label = ['aeroplane','bicycle', 'bird', 'boat', 'bottle', 'bus', 
                     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
                     'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        for i in range(3):
            images, labels = dataiter.next()
            images = images.to(device)
            labels = labels.to(device)
            # images.to("cpu")
            # model.to(device)
            with torch.no_grad():
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.tolist()
                # print(predicted)
                predictedNames = [idx2label[p] for p in predicted]
                labels = labels.numpy()
                actualLabels = [labels[p,predicted[p]] for p in range(len(predicted))]
                # print(predictedNames)
                print(actualLabels)
                # print(predictedNames)
            imgTitle = "epoch_" + str(epoch) + "_batchNum_" + str(i)

            visualizeLossPerformance(CAMLossInstance, images, labels=actualLabels, imgTitle=imgTitle, imgLabels=predictedNames)
        
    # visualizeImageBatch(images, labels)

    
    target_category = None
    
    #need to set params?
    
    
    
    # model.fc = nn.Linear(int(model.fc.in_features), 10)
    
    print("done")

    whichTraining = sys.argv[5]
    if whichTraining not in ['supervised', 'unsupervised', 'alternating']:
        print('invalid Training. will alternate')
        whichTraining = 'alternating'
    if sys.argv[3] == 'train':
        trackLoss = sys.argv[4] == 'trackLoss'
        print(trackLoss)
        train(model, numEpochs, suptrainloader, unsuptrainloader, testloader, optimizer, target_layer, target_category, use_cuda, trackLoss=trackLoss, training=whichTraining, batchDirectory=batchDirectory, scheduler=scheduler)
    
    
    
    

