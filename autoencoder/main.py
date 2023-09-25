# -*- coding: utf-8 -*-

import sys
import os

import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import helper
import configparser
import models


# =====================
# GET CONFIG VARIABLES
# =====================

config = configparser.ConfigParser()
config.read('config.ini')

BatchSize = config.getint('TRAIN', 'BatchSize')
Epochs = config.getint('TRAIN', 'Epochs')
LearningRate = config.getfloat('TRAIN', 'LearningRate')
file = config.get('TRAIN', 'TrainFile')
device = config.get('TRAIN', 'Device')

dirname = os.path.dirname("__file__")
FILE_PATH = os.path.join(dirname, file)
DEVICE = device
BATCH_SIZE = BatchSize
EPOCHS = Epochs
LR = LearningRate


# ==============================
# DEFINE DATASET AND DATALOADER
# ==============================

class MNIST_AE_Dataset(Dataset):
    
    def __init__(self, csv_file, noise_factor=0.2, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.noise_factor = noise_factor
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        img = self.data.iloc[idx]
        img = np.array(img).astype("float32")
        
        img = np.reshape(img, (28, 28, 1)) / 255.0 # channel as 1
        
        noisy_img = img + self.noise_factor * np.random.randn(*img.shape)
        noisy_img = np.clip(noisy_img, 0, 1)
        
        sample = (img, noisy_img)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    
trainset = MNIST_AE_Dataset(FILE_PATH, transform=helper.ToTensorForAE())
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# =============
# DEFINE MODEL
# =============

model = models.AutoEncoder()
model.to(DEVICE)

summary(model, input_size = (1, 28, 28))

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

summary(model, input_size = (1, 28, 28))

# ==============
# TRAINING LOOP
# ==============

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
criterion = nn.MSELoss()

for i in range(EPOCHS):
    
    train_loss = 0.0
    
    model.train()
    
    for batch in tqdm(trainloader):
        
        orig_image, noisy_image = batch
        
        orig_image = orig_image.to(DEVICE)
        noisy_image = noisy_image.to(DEVICE)

        denoised_image = model(noisy_image)
        loss = criterion(denoised_image, orig_image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(trainloader)
    
    print(f"Epoch: {i+1} - Train Loss: {avg_train_loss}")