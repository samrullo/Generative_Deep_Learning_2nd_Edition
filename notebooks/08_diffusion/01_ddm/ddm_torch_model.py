import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
LOAD_MODEL = False

NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 20

# optimization
EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50

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

class Normalizer:
    def __init__(self) -> None:
        self.mean=None
        self.std=None
    
    def adapt(self,train):
        self.mean=train.mean(dim=[0,2,3])
        self.std=train.std(dim=[0,2,3])
    
    def normalize(self, x):
        normalize_transform = transforms.Normalize(mean=self.mean, std=self.std)
        return normalize_transform(x)
    

def offset_cosine_diffusion_schedule(diffusion_times):
    """
    param diffusion_times : (N,1) diffusion times
    """
    min_rate = 0.02
    max_rate = 0.95
    start_angle = torch.arccos(torch.tensor(max_rate))
    end_angle = torch.arccos(torch.tensor(min_rate))
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


class DiffusionModel(nn.Module):
    def __init__(
        self, in_channels, adapted_normalizer: Normalizer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.unet = Unet(in_channels)
        self.ema_unet = Unet(in_channels)
        self.normalizer = adapted_normalizer
    
    def denormalize(self,images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return torch.clamp(images, 0.0, 1.0)
    
    def denoise(self,noisy_images,pred_noises,noise_rates, signal_rates):
        return (noisy_images - noise_rates * pred_noises) / signal_rates
    
    def reverse_diffusion(self, initial_noise, diffusion_steps):        
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):            
            diffusion_times = torch.ones((num_images,1)) - step * step_size
            noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)
            pred_noises = self.predict_noises(current_images,noise_rates**2)
            pred_images = self.denoise(current_images,pred_noises,noise_rates,signal_rates)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = offset_cosine_diffusion_schedule(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images,3, IMAGE_SIZE, IMAGE_SIZE))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def predict_noises(self,noisy_images, noise_variances):
        if self.training:
            pred_noises = self.unet([noisy_images, noise_variances])
        else:
            pred_noises = self.ema_unet([noisy_images, noise_variances])
        return pred_noises

    def forward(self, images):
        images = self.normalizer.normalize(images)
        batch_size, n_channels, height, width = images.size()
        noises = torch.randn((batch_size, 1, 1, 1))
        diffusion_times = torch.rand(batch_size, 1)
        noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)
        noise_rates = noise_rates.unsqueeze(2).unsqueeze(3)
        signal_rates = signal_rates.unsqueeze(2).unsqueeze(3)
        noisy_images = signal_rates * images + noise_rates * noises
        return self.predict_noises(noisy_images,noises**2)


def get_flower_images_train_dataset():
    transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    train_data = datasets.ImageFolder(r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data\flower\flower_data\flower_data\train",
                                    transform=transform)
    return train_data

def convert_images_torch_to_numpy_for_display(images:torch.tensor):
    return images.permute(0,2,3,1).numpy()