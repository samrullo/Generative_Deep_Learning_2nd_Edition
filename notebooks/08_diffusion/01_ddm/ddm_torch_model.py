import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(x):
    """
    param x: (N,1) shape random normal noise
    """
    # produces 16 linearly spaced numbers between 0 and ln(1000)
    frequencies = torch.linspace(torch.log(torch.tensor(1.0)),torch.log(torch.tensor(1000.0)),16)
    angular_speeds = 2 * torch.pi * torch.exp(frequencies) * x
    
    # return single scalar as 32 dimensional vector
    return torch.cat((torch.sin(angular_speeds),torch.cos(angular_speeds)),dim=1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def ResidualBlock(n_channels):
    def apply(x):
        in_channels = x.shape[1]
        if n_channels==in_channels:
            residual = x
        else:
            residual = nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=1)(x)
        x=nn.BatchNorm2d(num_features=in_channels,affine=False)(x)
        x = nn.Conv2d(in_channels=in_channels,out_channels=n_channels, kernel_size=3, padding=1, stride=1)(x)
        x = Swish()(x)
        x = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, stride=1)(x)
        return x + residual
    return apply

def DownBlock(n_channels,block_depth):
    def apply(x):
        x, skips = x

        for _ in range(block_depth):
            x = ResidualBlock(n_channels)(x)
            skips.append(x)
        x = nn.AvgPool2d(kernel_size=2)(x)
        return x
    return apply

def UpBlock(n_channels, block_depth):
    def apply(x):
        x, skips = x
        x = nn.Upsample(scale_factor=2,mode="bilinear")(x)
        for _ in range(block_depth):
            x = torch.cat([x,skips.pop()],dim=1)
            x = ResidualBlock(n_channels)(x)
        return x
    return apply

class Unet(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=64, mode="nearest")

        self.skips=[]

        self.downblock1 = DownBlock(32,2)
        self.downblock2 = DownBlock(64,2)
        self.downblock3 = DownBlock(96,2)

        self.residual1 = ResidualBlock(128)
        self.residual2 = ResidualBlock(128)

        self.upblock1 = UpBlock(96,2)
        self.upblock2 = UpBlock(64, 2)
        self.upblock3 = UpBlock(32,2)
        
        self.conv_last = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
    
    def forward(self, x):
        noisy_images,noise_variances = x
        noise_embeddings = sinusoidal_embedding(noise_variances)
        noise_embeddings = noise_embeddings.unsqueeze(2).unsqueeze(3)
        noise_embeddings = self.upsample1(noise_embeddings)
        
        x = self.conv1(noisy_images)
        x = torch.cat([x, noise_embeddings], dim=1)

        x = self.downblock1([x, self.skips])
        x = self.downblock2([x, self.skips])
        x = self.downblock3([x, self.skips])

        x = self.residual1(x)
        x = self.residual2(x)

        x = self.upblock1([x, self.skips])
        x = self.upblock2([x, self.skips])
        x = self.upblock3([x, self.skips])

        x = self.conv_last(x)

        return x

