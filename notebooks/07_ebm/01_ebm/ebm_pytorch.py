import pathlib
import sys
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# define constants
IMAGE_SIZE=32
CHANNELS=1
STEP_SIZE=10
STEPS=60
NOISE=0.005
ALPHA=0.1
GRADIENT_CLIP=0.03
BATCH_SIZE=128
BUFFER_SIZE=8192
LEARNING_RATE = 1e-4
EPOCHS=60

def get_mnist_dataset():    
    datapath=pathlib.Path(r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data")
    transform = transforms.Compose([transforms.Pad(2),transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
    train_mnist = datasets.MNIST(str(datapath),train=True,download=True,transform=transform)
    test_mnist = datasets.MNIST(str(datapath),train=False,download=True, transform=transform)
    print(f"train mnist size : {len(train_mnist)}, test mnist size : {len(test_mnist)}")
    return train_mnist, test_mnist

def split_to_train_and_validation(train_mnist):
    val_size = int(len(train_mnist)*0.2)
    train_size = int(len(train_mnist)-val_size)
    train_ds, val_ds = random_split(train_mnist,[train_size,val_size])
    print(f"train_ds has {len(train_ds)}, val_ds has {len(val_ds)}")
    return train_ds, val_ds

def get_mnist_data_loaders(train_ds, val_ds, test_mnist):
    # loaders
    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_mnist,batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


def swish(x,beta=1.0):
    return x * F.sigmoid(beta*x)


class EnergyFunction(nn.Module):
    def __init__(self,out_size=2, out_channels=64) -> None:
        super(EnergyFunction,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1)
        self.conv2=nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.dense = nn.Linear(out_size*out_size*out_channels,64)
        self.dense2 = nn.Linear(64,1)
    
    def forward(self,x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = x.flatten(start_dim=1)
        x = swish(self.dense(x))
        return self.dense2(x)

def generate_samples(model, inp_images, steps, step_size, noise, device, return_imgs_per_step=False):
    imgs_per_step = []
    scores_per_step = []

    for _ in range(steps):
        inp_images = inp_images.detach()
        inp_images.requires_grad_(True)
        image_noises = torch.normal(0, noise, size = inp_images.size())
        image_noises = image_noises.to(device)
        noised_inp_images = inp_images + image_noises
        noised_inp_images = torch.clamp(noised_inp_images, -1.0, 1.0)               

        model.zero_grad()
        outscore = model(noised_inp_images)
        mean_outscore = torch.sum(outscore,dim=0)
        mean_outscore.backward()
        grads = torch.clamp(inp_images.grad,-1*GRADIENT_CLIP,GRADIENT_CLIP)
        inp_images = inp_images + step_size * grads
        inp_images = torch.clamp(inp_images, -1.0, 1.0)

        if return_imgs_per_step:
            imgs_per_step.append(inp_images.detach())
            scores_per_step.append(outscore.detach())
    if return_imgs_per_step:
        return torch.stack(imgs_per_step,dim=0), torch.stack(scores_per_step, dim=0)
    return inp_images

