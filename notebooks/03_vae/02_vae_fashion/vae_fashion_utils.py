import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import pathlib

def get_fashion_mnist_datasets():
    datasets_folder = pathlib.Path(r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data\fashion")
    transform = transforms.Compose([transforms.Pad(2),transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(str(datasets_folder), train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(str(datasets_folder), train=False, download=True, transform=transform)
    return train_ds, test_ds

def get_fashion_mnist_dataloaders(train_ds,test_ds,batch_size=32,shuffle=True):
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=shuffle)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=shuffle)
    return train_loader,test_loader

def display_fashion_images(images, n=10, size=(20, 3), cmap="gray_r", as_type=torch.float32, save_to=None):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    images = images.cpu()
    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].to(as_type).squeeze(), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()


# Sampling module
class Sampling(nn.Module):
  def __init__(self):
    super(Sampling,self).__init__()

  def forward(self,z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    epsilon = torch.randn_like(std)
    z = z_mean + epsilon * std
    return z
  

  # Encoder
class Encoder(nn.Module):
  def __init__(self, out_size=4, out_channels=128,embedding_size=2):
    super(Encoder,self).__init__()

    self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

    self.z_mean_fc = nn.Linear(out_size*out_size*out_channels,embedding_size)
    self.z_log_var_fc = nn.Linear(out_size*out_size*out_channels,embedding_size)

    self.relu = nn.ReLU()

  def forward(self, x):
    # pdb.set_trace()
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = torch.flatten(x, start_dim=1)
    z_mean = self.z_mean_fc(x)
    z_log_var = self.z_log_var_fc(x)
    return z_mean, z_log_var
  
  # Decoder
class Decoder(nn.Module):
  def __init__(self,inp_channels=128,inp_size=4,embedding_size=2):
    super(Decoder,self).__init__()

    self.inp_channels=inp_channels
    self.inp_size=inp_size
    self.fc1 = nn.Linear(embedding_size, inp_size * inp_size * inp_channels)
    self.convtrns1=nn.ConvTranspose2d(inp_channels, 128, kernel_size=4,stride=2, padding=1)
    self.convtrns2 = nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2, padding=1)
    self.convtrns3 = nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2, padding=1)

    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(32,1,kernel_size=3,padding=1)

  def forward(self,z):
    x = self.fc1(z)
    # pdb.set_trace()
    x = x.view(-1,self.inp_channels,self.inp_size,self.inp_size)
    x = self.relu(self.convtrns1(x))
    x = self.relu(self.convtrns2(x))
    x = self.relu(self.convtrns3(x))
    x = torch.sigmoid(self.conv(x))
    return x
  
  # VAE
class VAE(nn.Module):
  def __init__(self, encoder, sampling, decoder):
    super(VAE,self).__init__()
    self.encoder=encoder
    self.decoder=decoder
    self.sampling=sampling

  def forward(self,x):
    z_mean, z_log_var = self.encoder(x)
    z = self.sampling(z_mean,z_log_var)
    out = self.decoder(z)
    return out, z_mean, z_log_var
  

