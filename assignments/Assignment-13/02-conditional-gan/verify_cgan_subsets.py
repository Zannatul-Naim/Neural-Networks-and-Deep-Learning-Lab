# verify_cgan_subset.py

import pickle
import sys
import pandas as pd
import torch # We need torch for the Dataset class
from torch.utils.data import Dataset # We need this for the class definition
from PIL import Image # The class uses this
import os # The class uses this

print("--- Starting CGAN Subset Verification Script ---")

# --- MODIFIED: Added the required class definition ---
# Pickle needs this "blueprint" to understand how to load the saved object.
class CelebAConditionalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_attribute='Bangs'):
        self.img_dir = img_dir
        self.transform = transform
        self.attributes = pd.read_csv(annotations_file)
        self.image_id_column = 'image_id'
        self.attributes = self.attributes[[self.image_id_column, target_attribute]]
        self.attributes[target_attribute] = self.attributes[target_attribute].apply(lambda x: 1 if x == 1 else 0)

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        img_filename = self.attributes.loc[idx, self.image_id_column]
        label = self.attributes.loc[idx, 'Bangs']
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)
# --- End of added class definition ---


# --- Step 1: Define filenames ---
TARGET_ATTRIBUTE = 'Bangs'
# Note: I'm correcting the filename to singular 'subset' as in my previous code
# Make sure this matches the file you want to run.
dataset_filename = f'./saved_cgan_datasets/cgan_celeba_{TARGET_ATTRIBUTE.lower()}_dataset.pkl'

# --- Step 2: Load the saved dataset object ---
try:
    print(f"Loading dataset from: {dataset_filename}")
    with open(dataset_filename, 'rb') as f:
        dataset_subset = pickle.load(f)
except FileNotFoundError:
    print(f"\nERROR: Could not find the file {dataset_filename}.")
    print("Please make sure you have run 'cgan_data_preprocessing.py' successfully first.")
    sys.exit()

print("Dataset loaded successfully.")

# --- Step 3: Extract label information ---
try:
    original_dataset = dataset_subset.dataset
    subset_indices = dataset_subset.indices
    all_attributes = original_dataset.attributes
    subset_attributes = all_attributes.iloc[subset_indices]
    label_counts = subset_attributes[TARGET_ATTRIBUTE].value_counts()
    
    count_black_hair = label_counts.get(1, 0)
    count_not_black_hair = label_counts.get(0, 0)
    total_samples = len(subset_attributes)

except Exception as e:
    print(f"An error occurred during verification: {e}")
    print("The saved dataset file might be corrupt or in an unexpected format.")
    sys.exit()

# --- Step 4: Display the results ---
print("\n--- Verification Results ---")
print(f"Total images in the random subset: {total_samples}")
print("-" * 30)
print(f"Images WITH Bangs (Label 1): {count_black_hair}")
print(f"Images WITHOUT Bangs (Label 0): {count_not_black_hair}")
print("-" * 30)

percent_black_hair = (count_black_hair / total_samples) * 100
percent_not_black_hair = (count_not_black_hair / total_samples) * 100

print(f"Percentage with Bangs: {percent_black_hair:.2f}%")
print(f"Percentage without Bangs: {percent_not_black_hair:.2f}%")

# print("\n--- Comparison ---")
# print("The full dataset had approximately 15.60% images with Bangs.")
# print("Your random subset's percentage should be close to this value.")
# print("\n--- Verification Complete ---")