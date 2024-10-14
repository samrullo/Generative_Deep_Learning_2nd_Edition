from typing import List,Tuple
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from ema_pytorch import EMA
from torchvision import datasets, transforms
from tqdm import tqdm
from notebooks.utils import display

IMAGE_SIZE = 64
BATCH_SIZE = 64
DATASET_REPETITIONS = 5
LOAD_MODEL = False

NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 20

# optimization
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50

class SinusoidalEmbedding(nn.Module):
    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        frequencies = torch.linspace(torch.log(torch.tensor(1.0)),torch.log(torch.tensor(1000.0)),16)
        frequencies = frequencies.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        frequencies = frequencies.to(device)
        self.frequencies = frequencies
    
    def forward(self,x):
        angular_speeds = 2 * torch.pi * torch.exp(self.frequencies) * x
        # return single scalar as 32 dimensional vector
        return torch.cat((torch.sin(angular_speeds),torch.cos(angular_speeds)),dim=1)

def sinusoidal_embedding_torch(x:torch.tensor):
    """
    Embed Nx1x1x1 noise variance to Nx32x1x1 tensor
    """
    frequencies = torch.exp(torch.linspace(torch.log(torch.tensor(1.0)),torch.log(torch.tensor(1000.0)),NOISE_EMBEDDING_SIZE//2))
    angular_speeds = 2 * torch.pi* frequencies * x
    return torch.cat((torch.sin(angular_speeds), torch.cos(angular_speeds)), dim=1)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.n_channels = out_channels
        if in_channels!=out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.n_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()        
        self.batch_norm = nn.BatchNorm2d(num_features=self.in_channels,affine=False)
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.swish = Swish()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)    

    
    def forward(self,x):        
        residual = self.shortcut(x)
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.swish(x)
        x = self.conv2(x)
        return x + residual


class Unet(nn.Module):
    def __init__(self, in_channels, down_blocks_io_channels:List[List[Tuple[int,int]]],up_blocks_io_channels:List[List[Tuple[int,int]]],mid_io_channels:List[Tuple[int,int]],last_in_channels:int, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=IMAGE_SIZE, mode="nearest")

        self.down_avg_poolings = nn.ModuleList([])
        self.down_blocks = nn.ModuleList([])
        for db_io_chans in down_blocks_io_channels:
            one_block_downs = nn.ModuleList([])
            for io_chan in db_io_chans:
                i_chan,o_chan=io_chan
                one_block_downs.append(ResidualBlock(i_chan, o_chan))
            self.down_blocks.append(one_block_downs)
            self.down_avg_poolings.append(nn.AvgPool2d(kernel_size=2))


        self.mid_blocks = nn.ModuleList([ResidualBlock(i_chan,o_chan) for i_chan,o_chan in mid_io_channels])        

        self.upsample_layers = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        for ub_io_chans in up_blocks_io_channels:
            one_block_ups = nn.ModuleList([])
            for io_chan in ub_io_chans:
                i_chan,o_chan=io_chan
                one_block_ups.append(ResidualBlock(i_chan,o_chan))
            self.up_blocks.append(one_block_ups)
            self.upsample_layers.append(nn.Upsample(scale_factor=2,mode="bilinear"))

        # self.upblock1 = UpBlock([(224,96),(192,96)])
        # self.upblock2 = UpBlock([(160,64),(128,64)])
        # self.upblock3 = UpBlock([(96,32),(64,32)])
        self.conv_last = nn.Conv2d(in_channels=last_in_channels, out_channels=3, kernel_size=1)
        nn.init.constant_(self.conv_last.weight, 0.0)
        if self.conv_last.bias is not None:
            nn.init.constant_(self.conv_last.bias, 0.0)

        self.sinusoidal_embedding = SinusoidalEmbedding(device)

        self.device = device
    
    def forward(self, x):
        noisy_images,noise_variances = x
        # noise_embeddings = sinusoidal_embedding(noise_variances,self.device)
        noise_embeddings = self.sinusoidal_embedding(noise_variances)
        noise_embeddings = self.upsample1(noise_embeddings)
        
        x = self.conv1(noisy_images)
        x = torch.cat([x, noise_embeddings], dim=1)

        skips=[]
        for i in range(len(self.down_blocks)):
            avg_pooling = self.down_avg_poolings[i]
            one_down_blocks = self.down_blocks[i]
            for dblock in one_down_blocks:
                x = dblock(x)
                skips.append(x)
            x = avg_pooling(x)

        for mid_block in self.mid_blocks:
            x = mid_block(x)

        for i in range(len(self.up_blocks)):
            upsample = self.upsample_layers[i]
            one_up_blocks = self.up_blocks[i]
            x = upsample(x)
            for ublock in one_up_blocks:    
                x = ublock(x)
                x = torch.cat((x,skips.pop()),dim=1)            
        x = self.conv_last(x)

        return x

class Normalizer:
    def __init__(self, mean:List[float]=None, std:List[float]=None) -> None:
        if mean is None:
            mean = [0.4353, 0.3773, 0.2871]
        if std is None:
            std = [0.2856, 0.2333, 0.2585]
        self.mean:List[float]=mean
        self.std:List[float]=std

    def denormalize(self, x):
        denormalize_transform = transforms.Normalize(
            mean=[-m/s for m, s in zip(self.mean, self.std)], 
            std=[1/s for s in self.std]
        )
        return denormalize_transform(x)

    
class OffsetCosineDiffusionSchedule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        min_rate = 0.02
        max_rate = 0.95
        self.start_angle = torch.arccos(torch.tensor(max_rate,requires_grad=False))
        self.end_angle = torch.arccos(torch.tensor(min_rate,requires_grad=False))
        self.cosine=torch.cos
        self.sine=torch.sin
    
    def forward(self,diffusion_times):        
        diffusion_angles = self.start_angle + diffusion_times * (self.end_angle - self.start_angle)
        signal_rates = self.cosine(diffusion_angles)
        noise_rates = self.sine(diffusion_angles)
        return noise_rates, signal_rates


class UniformSample(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,sample_shape):
        return torch.rand(sample_shape)

def unsqueeze_noise_or_signal_rates(noise_signal_rates:torch.tensor):
    return noise_signal_rates.unsqueeze(2).unsqueeze(3)



class DiffusionModel(nn.Module):
    def __init__(
        self, in_channels:int,down_blocks_io_channels:List[List[Tuple[int,int]]],up_blocks_io_channels:List[List[Tuple[int,int]]],mid_io_channels:List[Tuple[int,int]],last_in_channels:int, adapted_normalizer: Normalizer,device, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.unet = Unet(in_channels, down_blocks_io_channels,up_blocks_io_channels, mid_io_channels, last_in_channels, device)
        self.normalizer = adapted_normalizer
        self.device = device        
        self.diffusion_schedule = OffsetCosineDiffusionSchedule()
        self.uniform_sample=UniformSample()        
    
    def denormalize(self,images):       
        images = self.normalizer.denormalize(images)
        return torch.clamp(images, 0.0, 1.0)
    
    def denoise(self,noisy_images,pred_noises,noise_rates, signal_rates):
        return (noisy_images - noise_rates * pred_noises) / signal_rates
    
    def reverse_diffusion(self, initial_noise, diffusion_steps):        
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):            
            diffusion_times = torch.ones((num_images,1,1,1)) - step * step_size
            diffusion_times = diffusion_times.to(self.device)
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            #noise_rates = unsqueeze_noise_or_signal_rates(noise_rates)
            #signal_rates = unsqueeze_noise_or_signal_rates(signal_rates)
            noise_rates, signal_rates = noise_rates.to(self.device), signal_rates.to(self.device)
            pred_noises = self.predict_noises(current_images,noise_rates**2)
            pred_images = self.denoise(current_images,pred_noises,noise_rates,signal_rates)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            #next_noise_rates = unsqueeze_noise_or_signal_rates(next_noise_rates)
            #next_signal_rates = unsqueeze_noise_or_signal_rates(next_signal_rates)
            next_noise_rates, next_signal_rates = next_noise_rates.to(self.device), next_signal_rates.to(self.device)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images,3, IMAGE_SIZE, IMAGE_SIZE))
            initial_noise = initial_noise.to(self.device)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def predict_noises(self,noisy_images, noise_variances):        
        pred_noises = self.unet([noisy_images, noise_variances])
        return pred_noises    
    
    def forward(self, images, noises):        
        batch_size, n_channels, height, width = images.size()        
        diffusion_times = self.uniform_sample((batch_size, 1,1,1))
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)        
        noise_rates = noise_rates.to(self.device)
        signal_rates = signal_rates.to(self.device)
        noisy_images = signal_rates * images + noise_rates * noises
        return self.predict_noises(noisy_images,noise_rates**2)


def get_flower_images_train_dataset(should_normalize=True, mean:List[float]=None, std:List[float]=None):    
    if should_normalize:
        if mean is None:
            mean = [0.4353, 0.3773, 0.2871]
        if std is None:
            std = [0.2856, 0.2333, 0.2585]
        transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor(),transforms.Normalize(mean,std)])
    else:
        transform = transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        )
    train_data = datasets.ImageFolder(r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data\flower\flower_data\flower_data\train",
                                    transform=transform)
    return train_data

def convert_images_torch_to_numpy_for_display(images:torch.tensor):
    return images.permute(0,2,3,1).cpu().detach().numpy()

def training_loop(n_epochs, optimizer, model:DiffusionModel, loss_fn, train_loader,device,checkpoints_folder:pathlib.Path,images_folder:pathlib.Path):
    train_losses=[]
    min_train_loss = 1e8
    ema = EMA(model,beta=0.995,update_every=10)
    for epoch in tqdm(range(n_epochs), desc="Epoch loop"):
        loss_train=0
        model.train()
        for images, _ in tqdm(train_loader,desc="Train loader loop"):
            batch_size,n_channels,height, width = images.size()
            images = images.to(device)            
            noises = torch.randn((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE))
            noises = noises.to(device)
            pred_noises = model(images,noises)
            loss = loss_fn(noises,pred_noises)

            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            ema.update()
        loss_train_avg = loss_train / len(train_loader)        
        if loss_train_avg <= min_train_loss:
            torch.save(ema.state_dict(),str(checkpoints_folder/f"ddm_torch_checkpoints_{epoch}.pt"))
        min_train_loss = min(min_train_loss,loss_train_avg)        
        print(f"Epoch {epoch} train loss : {loss_train_avg}")        
        
        with torch.inference_mode():
            model.eval()
            ema.eval()
            generated_images = ema.ema_model.generate(10,20)
            generated_images_np = convert_images_torch_to_numpy_for_display(generated_images)
            display(generated_images_np,save_to=str(images_folder))        
    return train_losses