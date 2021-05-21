# import sys
# sys.path.append("./")
from libs.pytorch_grad_cam.grad_cam import GradCAM
from libs.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel


import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from libs.pytorch_grad_cam.utils.image import deprocess_image, preprocess_image

if __name__ == '__main__':
    ## Load the CIFAR Dataset
    transform = transforms.ToTensor()

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    CHECK_FOLDER = os.path.isdir("saved_figs")
    if not CHECK_FOLDER:
        os.makedirs("saved_figs")
        print("Made Saved_Figs folder")

    # functions to show an image

    def visualize(images, labels):
        def imshow(img):
            # img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.savefig('./saved_figs/sampleImages.jpg')


        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


    def show_cam_on_image(img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.
        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :returns: The default image with the cam overlay.
        """
        # print(img.shape, mask.shape)
        # print(mask.shape)
        # print(mask[0])
        # mask = cv2.flip(mask, 0)
        # mask = cv2.flip(mask, 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        # print(heatmap.shape)
        # print(heatmap[0])
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise Exception("The input image should np.float32 in the range [0, 1]")

        cam = heatmap + img
        # cam = heatmap
        cam = cam / np.max(cam)
        hmp = heatmap / np.max(heatmap)
        return np.uint8(255 * hmp), np.uint8(255 * cam)



    # replace the classifier layer with CAM Image Generation

    model = models.resnet50(pretrained = True)

    target_layer = model.layer4[-1] ##this is the layer before the pooling


    # load a few images from CIFAR and save

    dataiter = iter(trainloader)

    images, labels = dataiter.next()

    print(images.shape)

    visualize(images, labels)



    # if len(input_tensor.shape) > 3:
    #     gbimgs = []
    #     for i in range(input_tensor.shape[0]):
    #         input_tensor_gb = input_tensor[i:i+1,:,:,:]
    #         gb = gb_model(input_tensor_gb, target_category=target_category)
    #         gb_visualization = deprocess_image(gb)
    #         gbimgs.append(gb_visualization)

    #     final_gb_frame = cv2.hconcat(gbimgs)
    #     cv2.imwrite('./saved_figs/sampleImage_GuidedBackprop.jpg', final_gb_frame)



    # # Now Grad CAM
    # if len(rgb_img.shape) > 3:
    #     imgs = []
    #     for i in range(rgb_img.shape[0]):
    #         thisImg = rgb_img[i,:,:,:]
    #         thisImg = np.transpose(thisImg, (1, 2, 0))
    #         thisGray = grayscale_cam[i, :, :]
    #         visualization = show_cam_on_image(thisImg, thisGray)
    #         imgs.append(visualization)
    #     final_frame = cv2.hconcat(imgs)
    #     cv2.imwrite('./saved_figs/sampleImage_GradCAM.jpg', final_frame)
    print("Labels: ", labels)
    input_tensor = images

    rgb_img = np.float32(input_tensor.numpy()) ##numpy version of input tensor, for visualization

    use_cuda = torch.cuda.is_available()
    target_category = None

    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)


    # print(grayscale_cam[0,:])


    #Now for guided backprop

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)


    if len(input_tensor.shape) > 3:
        fig, axs = plt.subplots(3, input_tensor.shape[0])
        gbimgs = []
        imgs = []
        hmps = []
        for i in range(input_tensor.shape[0]):



            thisImg = rgb_img[i,:,:,:]
            thisImg = np.moveaxis(thisImg, 0, -1)
            thisImg = cv2.resize(thisImg, (256, 256))
            axs[0,i].imshow(thisImg)
            axs[0,i].axis('off')
            thisImgPreprocessed = preprocess_image(thisImg, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            grayscale_cam = cam(input_tensor=thisImgPreprocessed, target_category=target_category)
            thisGray = grayscale_cam[0, :]
            hmp, visualization = show_cam_on_image(thisImg, thisGray)
            imgs.append(visualization)
            hmps.append(hmp)

            # input_tensor_gb = input_tensor[i:i+1,:,:,:]
            gb = gb_model(thisImgPreprocessed, target_category=target_category)
            gb_visualization = deprocess_image(gb)
            print("average gb value: ", np.average(gb_visualization))
            gbimgs.append(gb_visualization)
            
            hmp_correlate = thisGray
            hmp_correlate = (hmp_correlate - np.mean(hmp_correlate)) / np.std(hmp_correlate)
            
            gb_correlate = gb
            gb_correlate = (gb_correlate - np.mean(gb_correlate)) / np.std(gb_correlate)
            gb_correlate = np.abs(gb_correlate)
            gb_correlate = np.sum(gb_correlate, axis = 2)
            # gb_correlate = (gb_correlate - np.mean(gb_correlate)) / np.std(gb_correlate)
            
            axs[1,i].imshow(hmp_correlate)
            axs[1,i].set_title("Grad CAM",fontsize=6)
            axs[2,i].imshow(gb_correlate)
            axs[2,i].set_title("Backprop",fontsize=6)
            axs[1,i].axis('off')
            axs[2,i].axis('off')
            
            output = np.corrcoef(hmp_correlate.flatten(), gb_correlate.flatten())[0,1]
            print("The Pearson output loss is: ", output)
            output2 = np.correlate(hmp_correlate.flatten(), gb_correlate.flatten())[0]
            print("The Cross Corr output loss is: ", output2)
            
            axs[0,i].set_title("Pearson Corr: " + str(round(output,3)) + "\n Cross Corr: " + str(round(output2)),fontsize=8)

        final_gb_frame = cv2.hconcat(gbimgs)
        cv2.imwrite('./saved_figs/sampleImage_GuidedBackprop.jpg', final_gb_frame)
        final_frame = cv2.hconcat(imgs)
        cv2.imwrite('./saved_figs/sampleImage_GradCAM.jpg', final_frame)
        final_hmp_frame = cv2.hconcat(hmps)
        cv2.imwrite('./saved_figs/sampleImage_GradCAM_hmp.jpg', final_hmp_frame)



