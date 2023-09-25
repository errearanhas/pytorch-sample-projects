import matplotlib.pyplot as plt 
import numpy as np 
import torch


def show_image(image, mask, pred_image = None, permutations = False):
    
    if permutations:
        image = image.permute(1,2,0).squeeze()
        mask = mask.permute(1,2,0).squeeze()
        if pred_image is not None:
            pred_image = pred_image.permute(1,2,0).squeeze()


    if pred_image is None:
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        ax1.set_title('IMAGE')
        ax1.imshow(image, cmap = 'gray')
        
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask, cmap = 'gray')
        
    elif pred_image is not None :
        
        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image, cmap = 'gray')
        
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask, cmap = 'gray')
        
        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image, cmap = 'gray')
        
        
