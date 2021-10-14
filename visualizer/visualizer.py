# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:12:28 2021

@author: alimi
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import cv2

def visualizeImageBatch(images, labels, resnetLabels=""):
    def imshow(img, labels):
        # print(labels)
        # img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        npimg -= np.min(npimg)
        npimg = npimg / np.max(npimg)
        # print("Image Characteristics:")
        # print(np.max(npimg), np.min(npimg), np.mean(npimg))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        classes = str(labels)
        titleString = ' '.join('%5s' % classes)
        plt.title(titleString + '\n' + resnetLabels)
        # plt.savefig('./saved_figs/sampleImages.jpg')

    imshow(torchvision.utils.make_grid(images), labels)

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

