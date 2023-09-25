# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import time
import configparser

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations as A
import cv2

import utils

# =====================
# GET CONFIG VARIABLES
# =====================

#!git clone https://github.com/parth1620/GradCAM-Dataset.git

config = configparser.ConfigParser()
config.read('config.ini')

BatchSize = config.getint('TRAIN', 'BatchSize')
Epochs = config.getint('TRAIN', 'Epochs')
LearningRate = config.getfloat('TRAIN', 'LearningRate')
file = config.get('TRAIN', 'TrainFile')
device = config.get('TRAIN', 'Device')

dirname = os.path.dirname(__file__)
FILE_PATH = os.path.join(dirname, "GradCAM-Dataset", file)
DEVICE = device
BATCH_SIZE = BatchSize
EPOCHS = Epochs
LR = LearningRate

data = pd.read_csv(FILE_PATH)

# =================
# TRAIN/TEST SPLIT
# =================

# cucumber: 0, eggplant: 1, mushroom: 2

train_df, valid_df = train_test_split(data,
                                      test_size=0.2,
                                      random_state=42)

# =====================
# DEFINE AUGMENTATIONS
# =====================

train_augs = A.Compose([
    A.Rotate(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

valid_augs = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])


# ==============================
# DEFINE TRAIN AND VAL DATASETS
# ==============================

DATA_DIR = os.path.join(dirname, "GradCAM-Dataset/")

trainset = utils.ImageDataset(train_df, augs = train_augs, data_dir = DATA_DIR)
validset = utils.ImageDataset(valid_df, augs = valid_augs, data_dir = DATA_DIR)


n = 22
image, label = validset[n] # (c, h, w) -> (h, w, c)

class_list = ['cucumber', 'eggplant', 'mushroom']

plt.imshow(image.permute(1, 2, 0))
plt.title(class_list[label]);

print(f"No. of examples in the trainset {len(trainset)}")
print(f"No. of examples in the validset {len(validset)}")


# =================================
# DEFINE TRAIN AND VAL DATALOADERS
# =================================

trainloader = DataLoader(trainset,
                         batch_size = BATCH_SIZE,
                         shuffle = True,
                         )

validloader = DataLoader(validset,
                         batch_size = BATCH_SIZE,
                         shuffle = False)

print(f"No. of batches in trainloader : {len(trainloader)}")
print(f"No. of batches in validloader : {len(validloader)}")

# take just first batch as sample
for images, labels in trainloader:
    break

print(f"One batch image shape : {images.shape}")
print(f"One batch label shape : {labels.shape}")


# =============
# CREATE MODEL
# =============

class ImageModel(nn.Module):
    def __init__(self, n_classes=3):
        super(ImageModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5, 5), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (4, 4), stride = 2),

            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (5, 5), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (4, 4), stride = 2),

            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (5, 5), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (4, 4), stride = 2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 5), padding = 1),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size = (4, 4), stride = 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_classes)
        )

        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad


    def forward(self, images):

        x = self.feature_extractor(images) # activation maps

        h = x.register_hook(self.activations_hook) # attach a custom function (a hook) to the tensor

        x = self.maxpool(x)
        x = self.classifier(x)

        return x

    def get_activation_gradients(self): # a1, a2, a3, ..., ak
        return self.gradient

    def get_activation(self, x): # A1, A2, A3, ..., Ak
        return self.feature_extractor(x) # 64 * 8 * 8 (c, h, w)
    

model  = ImageModel()
model.to(DEVICE)


# ==============
# TRAINING LOOP
# ==============

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
criterion = torch.nn.CrossEntropyLoss()

best_valid_loss = np.Inf

for i in range(EPOCHS):
    train_loss = utils.train_fn(trainloader, model, optimizer, criterion)
    valid_loss = utils.eval_fn(validloader, model, criterion)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict, 'best_weights.pt')
        best_valid_loss = valid_loss
        print("SAVED_WEIGHTS_SUCCESS")

    print(f"EPOCH: {i + 1} TRAIN LOSS: {train_loss} VALID_LOSS: {valid_loss}")

    
# ============
# GET GRADCAM
# ============
    
# cucumber: 0, eggplant: 1, mushroom: 2
sample = 8
required_label_to_check = 0

image, label = validset[sample]
denorm_image = image.permute(1, 2, 0) * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

image = image.unsqueeze(0).to(DEVICE)
pred = model(image)

heatmap = utils.get_gradcam(model, image, pred[0][required_label_to_check], size=227)

utils.plot_heatmap(denorm_image, pred, heatmap)