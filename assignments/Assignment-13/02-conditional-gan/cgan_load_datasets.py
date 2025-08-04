# cgan_load_datasets.py
# The class definition is NOT needed here.

import torch
import pickle

def load_cgan_data(batch_size=128, attribute='Black_Hair'):
    dataset_filename = f'./saved_cgan_datasets/cgan_celeba_{attribute.lower()}_dataset.pkl'
    metadata_filename = './saved_cgan_datasets/cgan_metadata.pkl'
    
    print(f"Loading dataset from: {dataset_filename}")
    # The class definition will be provided by cgan_train.py when this is called.
    with open(dataset_filename, 'rb') as f:
        dataset = pickle.load(f)

    with open(metadata_filename, 'rb') as f:
        metadata = pickle.load(f)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    print("\nCGAN data loaded successfully:")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Condition: {metadata['condition']}")
    print(f"  Samples: {metadata['dataset_samples']}")
    print(f"  Image size: {metadata['img_size']}x{metadata['img_size']}")
    print(f"  Batch size: {batch_size}")

    return dataloader, metadata