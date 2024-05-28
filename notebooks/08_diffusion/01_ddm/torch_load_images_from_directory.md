# How to load images from direcotry with torch

In PyTorch, you can achieve similar functionality using `torchvision.datasets.ImageFolder` for loading data from a directory and `torch.utils.data.DataLoader` for batching and shuffling. Below is an example of how to do this:

1. **Set Up the Dataset:**

   Use `torchvision.datasets.ImageFolder` to load the images from the directory. This assumes that your directory structure is suitable for classification tasks, but you can modify it to fit your specific needs.

2. **Data Transformations:**

   Use `torchvision.transforms` to apply necessary transformations such as resizing, converting to tensor, etc.

3. **Create DataLoader:**

   Use `torch.utils.data.DataLoader` to handle batching and shuffling.

Here's the complete code:

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the image size
IMAGE_SIZE = (224, 224)  # or any other size you need

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),      # Resize images to the desired size
    transforms.ToTensor()               # Convert images to PyTorch tensors
])

# Load the dataset
train_data = datasets.ImageFolder(
    root=r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data\flower\flower_data\flower_data\train",
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=32,  # or any other batch size you need
    shuffle=True,
    num_workers=4,  # number of subprocesses to use for data loading
    seed=42  # you can use torch.manual_seed(42) instead
)

# Example of iterating over the data
for images, labels in train_loader:
    # Do something with the images and labels
    print(images.shape, labels.shape)
```

### Explanation:
1. **Transforms:** The `transforms.Resize(IMAGE_SIZE)` resizes the images to the specified size. `transforms.ToTensor()` converts the images to PyTorch tensors.
2. **ImageFolder:** `datasets.ImageFolder` loads images from the directory, assuming a certain directory structure. If your data doesn't have subdirectories for each class, you can create a custom dataset.
3. **DataLoader:** The `DataLoader` handles batching and shuffling of the dataset. `num_workers` can be adjusted based on your system's capability. For setting the seed, you can use `torch.manual_seed(42)` to ensure reproducibility.

If your images are not organized in subdirectories, you can create a custom dataset class. Here's an example:

```python
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),      # Resize images to the desired size
    transforms.ToTensor()               # Convert images to PyTorch tensors
])

# Load the dataset
train_data = CustomImageDataset(
    root_dir=r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data\flower\flower_data\flower_data\train",
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=32,  # or any other batch size you need
    shuffle=True,
    num_workers=4  # number of subprocesses to use for data loading
)

# Example of iterating over the data
for images in train_loader:
    # Do something with the images
    print(images.shape)
```

In this custom dataset, we don't handle labels since your original TensorFlow code had `labels=None`. If you need to handle labels, you can modify the `CustomImageDataset` class accordingly.