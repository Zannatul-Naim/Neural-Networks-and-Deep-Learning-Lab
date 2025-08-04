# cyclegan_load_datasets.py

import torch
import pickle

def load_cyclegan_data():
    """Load preprocessed CycleGAN datasets for Real Faces <-> Painted Faces"""
    
    # Load real faces dataset (Domain A)
    with open('./saved_cyclegan_datasets/real_faces_dataset.pkl', 'rb') as f:
        dataset_real = pickle.load(f)
    
    # Load painted faces dataset (Domain B)
    with open('./saved_cyclegan_datasets/painted_faces_dataset.pkl', 'rb') as f:
        dataset_painted = pickle.load(f)
    
    # Load metadata
    with open('./saved_cyclegan_datasets/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Create dataloaders
    dataloader_real = torch.utils.data.DataLoader(
        dataset_real, 
        batch_size=metadata['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )
    
    dataloader_painted = torch.utils.data.DataLoader(
        dataset_painted, 
        batch_size=metadata['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    print(f"CycleGAN data loaded:")
    print(f"  Domain A (Real Faces): {metadata['real_faces_samples']} samples")
    print(f"  Domain B (Painted Faces): {metadata['painted_faces_samples']} samples")
    print(f"  Image size: {metadata['img_size']}x{metadata['img_size']}")
    print(f"  Batch size: {metadata['batch_size']}")
    
    return dataloader_real, dataloader_painted, metadata