# cyclegan_data_preprocessing.py

import os
import torch
from torchvision import transforms, datasets
import pickle
from torch.utils.data import Subset # <-- MODIFICATION: Import Subset
import random                      # <-- MODIFICATION: Import random

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
random.seed(42) # For reproducible subsets

# Data parameters
img_size = 256
batch_size = 4
celeba_subset_size = 50000 
# Paths - UPDATE THESE FOR YOUR SETUP
real_faces_dir = '.././img_align_celeba'
# Point this to your curated Van Gogh portrait folder or the full dataset
painted_faces_dir = './van_gogh_portraits'

# Transformations for real faces (CelebA)
transform_real = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformations for painted faces (Van Gogh)
transform_painted = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- MODIFICATION START ---

# Load FULL datasets
print("Loading FULL real faces dataset (CelebA)...")
full_dataset_real = datasets.ImageFolder(root=real_faces_dir, transform=transform_real)
print(f"Full CelebA dataset size: {len(full_dataset_real)}")

# Create a random subset of the CelebA dataset
print(f"Creating a random subset of {celeba_subset_size} images from CelebA...")
indices = list(range(len(full_dataset_real)))
random.shuffle(indices)
subset_indices = indices[:celeba_subset_size]
dataset_real = Subset(full_dataset_real, subset_indices)

# --- MODIFICATION END ---

print("Loading painted faces dataset (Van Gogh)...")
dataset_painted = datasets.ImageFolder(root=painted_faces_dir, transform=transform_painted)

print(f"Using {len(dataset_real)} images for real faces (Domain A)")
print(f"Using {len(dataset_painted)} images for painted faces (Domain B)")

# Create save directory
os.makedirs('./saved_cyclegan_datasets', exist_ok=True)

# Save datasets
print("Saving CycleGAN datasets...")

# Save real faces dataset (Domain A - Real Faces)
with open('./saved_cyclegan_datasets/real_faces_dataset.pkl', 'wb') as f:
    pickle.dump(dataset_real, f)

# Save painted faces dataset (Domain B - Van Gogh Paintings)
with open('./saved_cyclegan_datasets/painted_faces_dataset.pkl', 'wb') as f:
    pickle.dump(dataset_painted, f)

# Save metadata
metadata = {
    'img_size': img_size,
    'batch_size': batch_size,
    'real_faces_samples': len(dataset_real),
    'painted_faces_samples': len(dataset_painted),
    'domain_a': f'Real Faces (CelebA, {celeba_subset_size} subset)', # <-- MODIFICATION: Updated description
    'domain_b': 'Painted Faces (Van Gogh)'
}

with open('./saved_cyclegan_datasets/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("CycleGAN dataset preprocessing complete!")
print(f"Real faces (Domain A): {len(dataset_real)} samples")
print(f"Painted faces (Domain B): {len(dataset_painted)} samples")
print("Files saved in './saved_cyclegan_datasets/' directory")
print("\nDataset structure:")
print(f"  Domain A ({metadata['domain_a']}) -> Domain B ({metadata['domain_b']})")
print(f"  Domain B ({metadata['domain_b']}) -> Domain A ({metadata['domain_a']})")