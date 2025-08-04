# dcgan_data_preprocessing.py

import os
import torch
from torchvision import transforms, datasets
import pickle
from torch.utils.data import Subset
import random

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
random.seed(42)  # For reproducible subsets

# Data parameters
img_size = 64  # DCGAN architecture is typically designed for 64x64 images
celeba_subset_size = 50000

real_faces_dir = '../img_align_celeba'

# Transformations for the dataset
# DCGAN uses Tanh, so we normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the FULL CelebA dataset
print(f"Loading FULL real faces dataset from: {real_faces_dir}")
full_dataset_real = datasets.ImageFolder(root=real_faces_dir, transform=transform)
print(f"Full CelebA dataset size: {len(full_dataset_real)}")

# Create a random subset
print(f"Creating a random subset of {celeba_subset_size} images...")
indices = list(range(len(full_dataset_real)))
random.shuffle(indices)
subset_indices = indices[:celeba_subset_size]
dataset_subset = Subset(full_dataset_real, subset_indices)
print(f"Subset created with {len(dataset_subset)} images.")

# Create save directory for the dataset
os.makedirs('./saved_dcgan_datasets', exist_ok=True)

# Save the subset dataset
print("Saving DCGAN dataset...")
with open('./saved_dcgan_datasets/dcgan_celeba_dataset.pkl', 'wb') as f:
    pickle.dump(dataset_subset, f)

# Save metadata
metadata = {
    'img_size': img_size,
    'dataset_samples': len(dataset_subset),
    'dataset_name': f'CelebA ({celeba_subset_size} subset)'
}

with open('./saved_dcgan_datasets/dcgan_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("DCGAN dataset preprocessing complete!")
print(f"Dataset: {metadata['dataset_name']}")
print(f"Samples: {metadata['dataset_samples']}")
print(f"Image Size: {metadata['img_size']}x{metadata['img_size']}")
print("Files saved in './saved_dcgan_datasets/' directory")