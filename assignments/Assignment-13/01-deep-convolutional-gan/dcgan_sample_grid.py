# dcgan_sample_grid.py

import os
import torch
import torchvision.utils as vutils
from PIL import Image
import pickle
from dcgan_load_datasets import load_dcgan_data  # Import your DCGAN data loader

# ----------------------------------------------------------
# Load the DCGAN data
dataloader, _ = load_dcgan_data(batch_size=32) # Load a batch
save_dir = "./DCGAN_sample_grids"
os.makedirs(save_dir, exist_ok=True)

# Grab the first batch and take the first 16 images
try:
    imgs, _ = next(iter(dataloader)) # DCGAN dataset might not have labels returned, or returns None/empty
except ValueError:
    # If the dataset returns only images (no labels), this will catch it
    imgs = next(iter(dataloader))

# Ensure we only take the first 16 images
imgs = imgs[:16]

# Create a 4x4 grid
# Assuming images are normalized to [-1, 1] as is common in GAN training
grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=True, value_range=(-1, 1))

# Convert the grid tensor to a PIL Image
# Grid is CxHxW, PIL needs HxWxC. Also, make sure values are uint8 [0, 255]
grid_np = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
grid_img = Image.fromarray(grid_np)

# Save the grid image
filename = f"{save_dir}/dcgan_4x4_samples.png"
grid_img.save(filename)
print("Saved sample grid image to:", filename)
