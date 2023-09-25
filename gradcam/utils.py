import torch 
from torch.utils.data import Dataset

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import configparser
import torch
from torch import nn

config = configparser.ConfigParser()
config.read('config.ini')

device = config.get('TRAIN', 'Device')
DEVICE = device

def plot_heatmap(denorm_image, pred, heatmap):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,20), ncols=3)

    classes = ['cucumber', 'eggplant', 'mushroom']
    ps = torch.nn.Softmax(dim = 1)(pred).cpu().detach().numpy()
    ax1.imshow(denorm_image)

    ax2.barh(classes, ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    ax3.imshow(denorm_image)
    ax3.imshow(heatmap, cmap='magma', alpha=0.7)
    
    plt.savefig("grad_cam.jpg")
    return


class ImageDataset(Dataset):

    def __init__(self, df, data_dir = None, augs = None,):
        self.df = df
        self.augs = augs
        self.data_dir = data_dir 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = self.data_dir + row.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = row.label 

        if self.augs:
            data = self.augs(image = img)
            img = data['image']

        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, label

    
def train_fn(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, labels in (dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images) # just logits (no softmax, etc), because cross entropy loss will receive logits and true labels
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_fn(dataloader, model, criterion):
    model.eval()
    total_loss = 0.0

    for images, labels in (dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()

    return total_loss / len(dataloader)

# get gradcam
def get_gradcam(model, image, label, size):
    label.backward()
    gradients = model.get_activation_gradients()
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3]) # a1, a2, ..., ak
    activations = model.get_activation(image).detach() # A1, A2, ..., Ak

    for i in range(activations.shape[1]): # number of channels
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
    heatmap = nn.ReLU()(heatmap) # get rid of unwanted regions
    heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (size, size))

    return heatmap