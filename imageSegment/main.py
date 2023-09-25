# -*- coding: utf-8 -*-

import sys

import torch 
import cv2

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import albumentations as A

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper
import configparser


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
img_size = config.getint('TRAIN', 'ImageSize')
enc = config.get('TRAIN', 'Encoder')
weights = config.get('TRAIN', 'Weights')

dirname = os.path.dirname("__file__")
FILE_PATH = os.path.join(dirname, "Human-Segmentation-Dataset-master", file)
DEVICE = device
BATCH_SIZE = BatchSize
EPOCHS = Epochs
LR = LearningRate
IMAGE_SIZE = img_size
ENCODER = enc
WEIGHTS = weights


# =====================
# DEFINE AUGMENTATIONS
# =====================

def get_train_augs():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

def get_valid_augs():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ])


# ================================
# CREATE DATASETS AND DATALOADERS
# ================================

class SegmentationDataset(Dataset):
    
    def __init__(self, df, augmentations):
        
        self.df = df
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = row.images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = row.masks
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # (h, w, c)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]
            
        # (h, w, c) --> (c, h, w)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)      
        image = torch.Tensor(image) / 255.0
        
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        mask = torch.Tensor(mask) / 255.0
        mask = torch.round(mask)
        
        return image, mask


# =================================
# DEFINE TRAIN AND VAL DATALOADERS
# =================================

data = pd.read_csv(FILE_PATH)

train_df, valid_df = train_test_split(data,
                                      test_size=0.2,
                                      random_state=42)

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

trainloader = DataLoader(trainset, batch_size = BatchSize, shuffle = True)
validloader = DataLoader(validset, batch_size = BatchSize)


# =============
# CREATE MODEL
# =============

class SegmentationModel(nn.Module):
    
    def __init__(self):
        super(SegmentationModel, self).__init__()
        
        self.arc = smp.Unet(
            encoder_name = ENCODER,
            encoder_weights = WEIGHTS,
            in_channels = 3,
            classes = 1,
            activation = None
        )
        
    def forward(self, images, masks=None):

        logits = self.arc(images)

        if masks is not None:
            loss1 = DiceLoss(mode="binary")(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            loss = loss1 + loss2
            return logits, loss

        return logits
    

model = SegmentationModel()
model.to(DEVICE)


# ===============================
# CREATE TRAIN AND VAL FUNCTIONS
# ===============================

def train_fn(data_loader, model, optimizer):
    
    model.train()
    
    total_loss = 0.0
    
    for images, masks in tqdm(data_loader):
        
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def eval_fn(data_loader, model):
    
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, loss = model(images, masks)

            total_loss += loss.item()

        return total_loss / len(data_loader)
    
    
# ==============
# TRAINING LOOP
# ==============

optimizer = torch.optim.Adam(model.parameters(), lr = LR)

best_validation_loss = np.Inf

for i in range(EPOCHS):
    
    train_loss = train_fn(trainloader, model, optimizer)
    valid_loss = eval_fn(validloader, model)
    
    if valid_loss < best_validation_loss:
        torch.save(model.state_dict(), "best_model.pt")
        print(f"SAVED MODEL AT EPOCH {i+1}")
        best_validation_loss = valid_loss
        
    print(f"Epoch: {i+1}, Train loss: {train_loss}, Valid loss: {valid_loss}")
    

# ===============================
# USE BEST MODEL FOR PREDICTIONS
# ===============================
    

idx = 2

model.load_state_dict(torch.load("best_model.pt"))

image, mask = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0)) # (c, h, w) -> (1, c, h, w)
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0
pred_mask = pred_mask.detach().cpu().squeeze(0)