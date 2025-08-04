# dcgan_load_datasets.py

import torch
import pickle

def load_dcgan_data(batch_size=128):
    """Load preprocessed DCGAN dataset for CelebA faces"""

    # Load the dataset object
    with open('./saved_dcgan_datasets/dcgan_celeba_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    # Load metadata
    with open('./saved_dcgan_datasets/dcgan_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # Create the DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Use more workers for faster data loading
        pin_memory=True # Helps speed up data transfer to the GPU
    )

    print(f"DCGAN data loaded:")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Samples: {metadata['dataset_samples']}")
    print(f"  Image size: {metadata['img_size']}x{metadata['img_size']}")
    print(f"  Batch size: {batch_size}")

    return dataloader, metadata