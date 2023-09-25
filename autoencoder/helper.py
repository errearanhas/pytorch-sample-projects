# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np 
import torch


def show_image(org_image, noisy_image, pred_image = None, permutations = False):
    
    org_image = org_image.to("cpu")
    noisy_image = noisy_image.to("cpu")

    if permutations:
        org_image = org_image.permute(1,2,0).squeeze()
        noisy_image = noisy_image.permute(1,2,0).squeeze()
        if pred_image is not None:
            pred_image = pred_image.permute(1,2,0).squeeze()
    
    if pred_image == None:
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.set_title('original_image')
        ax1.imshow(org_image, cmap = 'gray')
        
        ax2.set_title('noisy_image')
        ax2.imshow(noisy_image, cmap = 'gray')

    elif pred_image != None :
        pred_image = pred_image.to("cpu")
        
        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
        ax1.set_title('noisy_image')
        ax1.imshow(noisy_image, cmap = 'gray')
        
        ax2.set_title('original_image')
        ax2.imshow(org_image, cmap = 'gray')
        
        ax3.set_title('denoised_image')
        ax3.imshow(pred_image, cmap = 'gray')
        
        
class ToTensorForAE(object):
    
    def __call__(self, sample):
        
        labels, images = sample
        
        labels = labels.transpose((2,0,1))
        labels = torch.from_numpy(labels).float()
        
        images = images.transpose((2,0,1))
        images = torch.from_numpy(images).float()
        
        return labels, images


# =====================
# DEFINE AUGMENTATIONS
# =====================

def get_train_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

def get_valid_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
    ])