import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalEmbedding(nn.Module):
    def __init__(self, noise_embedding_size):
        super(SinusoidalEmbedding, self).__init__()
        self.noise_embedding_size = noise_embedding_size
        frequencies = torch.exp(torch.linspace(np.log(1.0), np.log(1000.0), self.noise_embedding_size // 2))
        angular_speeds = 2.0 * np.pi * frequencies
        self.register_buffer('angular_speeds', angular_speeds)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        grid_x, grid_y = torch.meshgrid([torch.arange(height), torch.arange(width)])
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # (2, H, W)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, 2, H, W)

        grid = grid.unsqueeze(1).expand(-1, self.noise_embedding_size // 2, -1, -1, -1)  # (B, NoiseEmbeddingSize//2, 2, H, W)
        embeddings = torch.cat((torch.sin(self.angular_speeds * grid), torch.cos(self.angular_speeds * grid)), dim=1)  # (B, NoiseEmbeddingSize, H, W)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(width)

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x

class DownBlock(nn.Module):
    def __init__(self, width, block_depth):
        super(DownBlock, self).__init__()
        self.block_depth = block_depth
        self.residual_blocks = nn.ModuleList([ResidualBlock(width) for _ in range(block_depth)])
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x, skips = x
        for i in range(self.block_depth):
            x = self.residual_blocks[i](x)
            skips.append(x)
        x = self.avg_pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth):
        super(UpBlock, self).__init__()
        self.block_depth = block_depth
        self.residual_blocks = nn.ModuleList([ResidualBlock(width) for _ in range(block_depth)])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skips):
        x = self.upsample(x)
        for i in range(self.block_depth):
            x = torch.cat((x, skips.pop()), dim=1)
            x = self.residual_blocks[i](x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, image_size, noise_embedding_size):
        super(DiffusionModel, self).__init__()
        self.image_size = image_size
        self.noise_embedding_size = noise_embedding_size

        self.sinusoidal_embedding = SinusoidalEmbedding(noise_embedding_size)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.concat = nn.Conv2d(64, 64, kernel_size=1)
        
        self.down_block1 = DownBlock(32, block_depth=2)
        self.down_block2 = DownBlock(64, block_depth=2)
        self.down_block3 = DownBlock(96, block_depth=2)
        
        self.residual_block1 = ResidualBlock(128)
        self.residual_block2 = ResidualBlock(128)
        
        self.up_block1 = UpBlock(96, block_depth=2)
        self.up_block2 = UpBlock(64, block_depth=2)
        self.up_block3 = UpBlock(32, block_depth=2)
        
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1)

        self.ema = 0.999
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def offset_cosine_diffusion_schedule(self, diffusion_times):
        min_signal_rate = 0.02
        max_signal_rate = 0.95
        start_angle = torch.acos(torch.tensor(max_signal_rate))
        end_angle = torch.acos(torch.tensor(min_signal_rate))

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates):
        x = self.conv1(noisy_images)

        noise_embedding = self.sinusoidal_embedding(noise_rates)
        noise_embedding = F.interpolate(noise_embedding, size=(self.image_size, self.image_size), mode='nearest')
        
        x = self.concat(torch.cat((x, noise_embedding), dim=1))

        skips = []

        x = self.down_block1([x, skips])
        x = self.down_block2([x, skips])
        x = self.down_block3([x, skips])

        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.up_block1(x, skips)
        x = self.up_block2(x, skips)
        x = self.up_block3(x, skips)

        x = self.conv2(x)
        
        return x

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.size(0)
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = torch.ones(num_images, 1, 1, 1) - step * step_size
            noise_rates, signal_rates = self.offset_cosine_diffusion_schedule(diffusion_times)
            pred_images = self.denoise(current_images, noise_rates, signal_rates)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.offset_cosine_diffusion_schedule(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * current_images
        return pred_images

    def forward(self, noisy_images, initial_noise, diffusion_steps):
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        return generated_images