# cgan_data_preprocessing.py

import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, Subset
import pickle
import pandas as pd
from PIL import Image
import random

# --- Custom Dataset Class for CelebA with Attributes from CSV ---
# This class reads the CSV to get the correct label for each image.
class CelebAConditionalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_attribute='Bangs'):
        self.img_dir = img_dir
        self.transform = transform
        
        # Read the annotations CSV file
        print(f"Reading annotations CSV file from: {annotations_file}")
        self.attributes = pd.read_csv(annotations_file)
        
        # --- IMPORTANT ---
        # The column in your CSV with filenames like '000001.jpg'. 
        # Please double-check your CSV and change 'image_id' if needed.
        self.image_id_column = 'image_id'
        
        # Keep only the image filenames and the target attribute
        self.attributes = self.attributes[[self.image_id_column, target_attribute]]
        
        # Convert labels from [-1, 1] to a clean [0, 1] format
        self.attributes[target_attribute] = self.attributes[target_attribute].apply(lambda x: 1 if x == 1 else 0)
        
        print(f"Loaded {len(self.attributes)} image annotations.")
        print(f"Target Attribute: '{target_attribute}'")
        print("Label Distribution (1 = Bangs, 0 = Not Bangs):")
        print(self.attributes[target_attribute].value_counts())

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        # Get the image filename and the corresponding label
        img_filename = self.attributes.loc[idx, self.image_id_column]
        label = self.attributes.loc[idx, 'Bangs']
        
        # Construct the full path to the image
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if they exist
        if self.transform:
            image = self.transform(image)
        
        # Return the image and its label
        return image, torch.tensor([label], dtype=torch.float32)

# --- Main Preprocessing Logic ---
if __name__ == "__main__":
    # Configuration
    IMG_SIZE = 64
    SUBSET_SIZE = 100000
    TARGET_ATTRIBUTE = 'Bangs'
    random.seed(42)

    # --- PLEASE VERIFY THESE PATHS ---
    # Path to the FOLDER containing the actual .jpg image files
    celeba_image_dir = '../img_align_celeba/img_align_celeba' 
    # Path to the annotations CSV file
    annotations_file_path = '../list_attr_celeba.csv' 

    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the full conditional dataset from the CSV and image folder
    print("Creating full conditional dataset...")
    full_dataset = CelebAConditionalDataset(
        annotations_file=annotations_file_path,
        img_dir=celeba_image_dir,
        transform=transform,
        target_attribute=TARGET_ATTRIBUTE
    )
    
    # Create a random subset of 50,000 samples
    print(f"Creating a random subset of {SUBSET_SIZE} images...")
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:SUBSET_SIZE]
    dataset_subset = Subset(full_dataset, subset_indices)
    print(f"Subset created with {len(dataset_subset)} images.")

    # Create directory for saving the processed dataset
    os.makedirs('./saved_cgan_datasets', exist_ok=True)

    # Save the final dataset object to a file using pickle
    print("Saving Conditional GAN (CGAN) dataset...")
    with open(f'./saved_cgan_datasets/cgan_celeba_{TARGET_ATTRIBUTE.lower()}_dataset.pkl', 'wb') as f:
        pickle.dump(dataset_subset, f)

    # Save metadata for later use
    metadata = {
        'img_size': IMG_SIZE,
        'dataset_samples': len(dataset_subset),
        'dataset_name': f'CelebA Conditional ({SUBSET_SIZE} subset)',
        'condition': TARGET_ATTRIBUTE
    }
    with open('./saved_cgan_datasets/cgan_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\nCGAN dataset preprocessing complete!")
    print(f"Dataset saved for condition: '{metadata['condition']}'")
    print("Files saved in './saved_cgan_datasets/' directory")