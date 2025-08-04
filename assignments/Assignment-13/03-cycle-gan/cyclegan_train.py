# # cyclegan_train.py

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
# import itertools
# from cyclegan_load_datasets import load_cyclegan_data

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.manual_seed(42)
# np.random.seed(42)

# # --- (Model classes ResBlock, CycleGAN_Generator, CycleGAN_Discriminator are unchanged) ---
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(channels)
#         )
#     def forward(self, x):
#         return x + self.block(x)

# class CycleGAN_Generator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, n_res=9):
#         super().__init__()
#         # Encoder
#         model = [
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels, 64, 7, 1, 0, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#         ]
        
#         # Downsampling
#         for i in range(2):
#             mult = 2 ** i
#             model += [
#                 nn.Conv2d(64 * mult, 64 * mult * 2, 3, 2, 1, bias=False),
#                 nn.InstanceNorm2d(64 * mult * 2),
#                 nn.ReLU(inplace=True)
#             ]
        
#         # ResNet blocks
#         mult = 2 ** 2
#         for i in range(n_res):
#             model += [ResBlock(64 * mult)]
        
#         # Upsampling
#         for i in range(2):
#             mult = 2 ** (2 - i)
#             model += [
#                 nn.ConvTranspose2d(64 * mult, int(64 * mult / 2), 3, 2, 1, 1, bias=False),
#                 nn.InstanceNorm2d(int(64 * mult / 2)),
#                 nn.ReLU(inplace=True)
#             ]
        
#         # Output layer
#         model += [
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(64, out_channels, 7, 1, 0),
#             nn.Tanh()
#         ]
        
#         self.model = nn.Sequential(*model)
    
#     def forward(self, x):
#         return self.model(x)

# class CycleGAN_Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
        
#         def discriminator_block(in_filters, out_filters, normalize=True):
#             layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
#             if normalize:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
        
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels, 64, normalize=False),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(256, 512),
#             nn.Conv2d(512, 1, 4, 1, 1)
#         )
    
#     def forward(self, x):
#         return self.model(x)

# # Training function
# def train_cyclegan(G_AB, G_BA, D_A, D_B, dataloader_real, dataloader_painted, 
#                    num_epochs, device, save_interval=1): # MODIFIED: Changed default save_interval
#     print("--- Training CycleGAN: Real Faces <-> Painted Faces ---")
    
#     # Create save directory
#     save_dir = "./CycleGAN_generated_images"
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Loss functions
#     criterion_GAN = nn.MSELoss()
#     criterion_cycle = nn.L1Loss()
#     criterion_identity = nn.L1Loss()
    
#     # Optimizers
#     optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), 
#                            lr=0.0002, betas=(0.5, 0.999))
#     optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
#     # Loss weights
#     lambda_cycle = 10.0
#     lambda_identity = 5.0
    
#     # Loss tracking
#     g_losses, d_a_losses, d_b_losses = [], [], []
    
#     # Fixed samples for visualization (using 4 samples for a 2x2 grid of pairs)
#     try:
#         fixed_real = next(iter(dataloader_real))[0][:4].to(device)
#     except:
#         fixed_real = torch.randn(4, 3, 256, 256, device=device)
    
#     # Create infinite iterators
#     real_iter = iter(dataloader_real)
#     painted_iter = iter(dataloader_painted)
    
#     for epoch in range(num_epochs):
#         g_loss_sum, d_a_loss_sum, d_b_loss_sum = 0.0, 0.0, 0.0
#         num_batches = min(len(dataloader_real), len(dataloader_painted))
        
#         for batch_idx in range(num_batches):
#             # Get batch data
#             try:
#                 real_A, _ = next(real_iter)
#             except StopIteration:
#                 real_iter = iter(dataloader_real)
#                 real_A, _ = next(real_iter)
            
#             try:
#                 real_B, _ = next(painted_iter)
#             except StopIteration:
#                 painted_iter = iter(dataloader_painted)
#                 real_B, _ = next(painted_iter)
            
#             real_A = real_A.to(device)  # Real faces
#             real_B = real_B.to(device)  # Painted faces
#             batch_size = real_A.size(0)
            
#             # Adversarial ground truths
#             with torch.no_grad():
#                 dummy_out = D_A(real_A[:1])
#                 d_out_shape = dummy_out.shape[1:]
            
#             valid = torch.ones((batch_size, *d_out_shape), device=device)
#             fake = torch.zeros((batch_size, *d_out_shape), device=device)
            
#             # --- (The core training logic for losses and optimizers is unchanged) ---
#             # Train Generators
#             optimizer_G.zero_grad()
#             same_B = G_AB(real_B)
#             loss_identity_B = criterion_identity(same_B, real_B) * lambda_identity
#             fake_B = G_AB(real_A)
#             loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
#             recovered_A = G_BA(fake_B)
#             loss_cycle_A = criterion_cycle(recovered_A, real_A) * lambda_cycle
#             same_A = G_BA(real_A)
#             loss_identity_A = criterion_identity(same_A, real_A) * lambda_identity
#             fake_A = G_BA(real_B)
#             loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
#             recovered_B = G_AB(fake_A)
#             loss_cycle_B = criterion_cycle(recovered_B, real_B) * lambda_cycle
#             loss_G = ((loss_GAN_AB + loss_GAN_BA) / 2) + ((loss_cycle_A + loss_cycle_B) / 2) + ((loss_identity_A + loss_identity_B) / 2)
#             loss_G.backward()
#             optimizer_G.step()
            
#             # Train Discriminator A
#             optimizer_D_A.zero_grad()
#             loss_real_D_A = criterion_GAN(D_A(real_A), valid)
#             loss_fake_D_A = criterion_GAN(D_A(fake_A.detach()), fake)
#             loss_D_A = (loss_real_D_A + loss_fake_D_A) / 2
#             loss_D_A.backward()
#             optimizer_D_A.step()
            
#             # Train Discriminator B
#             optimizer_D_B.zero_grad()
#             loss_real_D_B = criterion_GAN(D_B(real_B), valid)
#             loss_fake_D_B = criterion_GAN(D_B(fake_B.detach()), fake)
#             loss_D_B = (loss_real_D_B + loss_fake_D_B) / 2
#             loss_D_B.backward()
#             optimizer_D_B.step()
            
#             g_loss_sum += loss_G.item()
#             d_a_loss_sum += loss_D_A.item()
#             d_b_loss_sum += loss_D_B.item()
        
#         avg_g_loss = g_loss_sum / num_batches
#         avg_d_a_loss = d_a_loss_sum / num_batches
#         avg_d_b_loss = d_b_loss_sum / num_batches
        
#         g_losses.append(avg_g_loss)
#         d_a_losses.append(avg_d_a_loss)
#         d_b_losses.append(avg_d_b_loss)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}] | G_loss: {avg_g_loss:.4f} | D_A_loss: {avg_d_a_loss:.4f} | D_B_loss: {avg_d_b_loss:.4f}')
        
#         # --- MODIFIED: Image saving logic for new visualization format ---
#         if (epoch + 1) % save_interval == 0:
#             G_AB.eval() # Set generator to evaluation mode
#             with torch.no_grad():
#                 # Generate painted versions of the fixed real faces
#                 generated_painted = G_AB(fixed_real)
                
#                 # Interleave the images: [real_1, painted_1, real_2, painted_2, ...]
#                 num_samples = fixed_real.size(0)
#                 comparison_images = torch.empty((num_samples * 2, *fixed_real.shape[1:]), device=device)
#                 comparison_images[0::2] = fixed_real          # Even indices get real images
#                 comparison_images[1::2] = generated_painted   # Odd indices get painted images
                
#                 # De-normalize images from [-1, 1] to [0, 1] for saving
#                 comparison_images_denorm = comparison_images * 0.5 + 0.5
                
#                 # Create a grid with 2 columns: Original | Generated
#                 grid = make_grid(comparison_images_denorm, nrow=2, padding=2, normalize=False)
#                 grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))
                
#                 # Save the image
#                 png_filename = os.path.join(save_dir, f"CycleGAN_comparison_epoch_{epoch+1:04d}.png")
#                 plt.imsave(png_filename, grid_np)
#                 print(f"  -> Saved comparison image: {png_filename}")
#             G_AB.train() # Set generator back to training mode
    
#     print("\n--- Training Finished ---")
#     # --- (Plotting logic at the end is unchanged) ---
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(g_losses, label='Generator Loss')
#     plt.legend()
#     plt.title("Generator Loss")
#     plt.subplot(1, 2, 2)
#     plt.plot(d_a_losses, label='D_A Loss')
#     plt.plot(d_b_losses, label='D_B Loss')
#     plt.legend()
#     plt.title("Discriminator Losses")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'cyclegan_training_summary.png'))
#     plt.close()
#     print(f"  -> Saved training loss plot")
    
#     return g_losses, d_a_losses, d_b_losses

# # Main execution
# if __name__ == "__main__":
#     # Load preprocessed data
#     dataloader_real, dataloader_painted, metadata = load_cyclegan_data()
    
#     # MODIFIED: Training parameters updated as requested
#     num_epochs = 5      # Set to 5 epochs for a quick run
#     save_interval = 1   # Save images after every epoch
    
#     # Create models
#     G_AB = CycleGAN_Generator().to(device)  # Real -> Painted
#     G_BA = CycleGAN_Generator().to(device)  # Painted -> Real
#     D_A = CycleGAN_Discriminator().to(device)  # Discriminator for Real faces
#     D_B = CycleGAN_Discriminator().to(device)  # Discriminator for Painted faces
    
#     print("Model Summary:")
#     print(f"  G_AB: Real Faces -> Painted Faces")
#     print(f"  G_BA: Painted Faces -> Real Faces")
#     print(f"  D_A: Discriminator for Real Faces")
#     print(f"  D_B: Discriminator for Painted Faces")
    
#     # MODIFIED: Pass the save_interval to the training function
#     g_losses, d_a_losses, d_b_losses = train_cyclegan(
#         G_AB, G_BA, D_A, D_B, dataloader_real, dataloader_painted, 
#         num_epochs, device, save_interval
#     )
    
#     # Save trained models
#     os.makedirs('./saved_cyclegan_models', exist_ok=True)
#     torch.save(G_AB.state_dict(), './saved_cyclegan_models/generator_real_to_painted.pth')
#     torch.save(G_BA.state_dict(), './saved_cyclegan_models/generator_painted_to_real.pth')
    
#     print("CycleGAN training complete!")
#     print("Models saved in './saved_cyclegan_models/' directory")
#     print("Generated images saved in './CycleGAN_generated_images/' directory")




# cyclegan_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import itertools
from cyclegan_load_datasets import load_cyclegan_data

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# --- (Model classes ResBlock, CycleGAN_Generator, CycleGAN_Discriminator are unchanged) ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class CycleGAN_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res=9):
        super().__init__()
        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(64 * mult, 64 * mult * 2, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(64 * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # ResNet blocks
        mult = 2 ** 2
        for i in range(n_res):
            model += [ResBlock(64 * mult)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(64 * mult, int(64 * mult / 2), 3, 2, 1, 1, bias=False),
                nn.InstanceNorm2d(int(64 * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class CycleGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_cyclegan(G_AB, G_BA, D_A, D_B, dataloader_real, dataloader_painted, 
                   num_epochs, device, save_interval=1):
    print("--- Training CycleGAN: Real Faces <-> Painted Faces ---")
    
    # Create save directory
    save_dir = "./CycleGAN_generated_images"
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), 
                           lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss weights
    lambda_cycle = 10.0
    lambda_identity = 5.0
    
    # Loss tracking
    g_losses, d_a_losses, d_b_losses = [], [], []
    
    # Fixed samples for visualization (using 4 samples for a 4x3 grid)
    try:
        fixed_real = next(iter(dataloader_real))[0][:4].to(device)
    except:
        fixed_real = torch.randn(4, 3, 256, 256, device=device)
    
    # Create infinite iterators
    real_iter = iter(dataloader_real)
    painted_iter = iter(dataloader_painted)
    
    for epoch in range(num_epochs):
        g_loss_sum, d_a_loss_sum, d_b_loss_sum = 0.0, 0.0, 0.0
        num_batches = min(len(dataloader_real), len(dataloader_painted))
        
        for batch_idx in range(num_batches):
            # Get batch data
            try:
                real_A, _ = next(real_iter)
            except StopIteration:
                real_iter = iter(dataloader_real)
                real_A, _ = next(real_iter)
            
            try:
                real_B, _ = next(painted_iter)
            except StopIteration:
                painted_iter = iter(dataloader_painted)
                real_B, _ = next(painted_iter)
            
            real_A = real_A.to(device)  # Real faces
            real_B = real_B.to(device)  # Painted faces
            batch_size = real_A.size(0)
            
            # Adversarial ground truths
            with torch.no_grad():
                dummy_out = D_A(real_A[:1])
                d_out_shape = dummy_out.shape[1:]
            
            valid = torch.ones((batch_size, *d_out_shape), device=device)
            fake = torch.zeros((batch_size, *d_out_shape), device=device)
            
            # Train Generators
            optimizer_G.zero_grad()
            same_B = G_AB(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * lambda_identity
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * lambda_cycle
            same_A = G_BA(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * lambda_identity
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * lambda_cycle
            loss_G = ((loss_GAN_AB + loss_GAN_BA) / 2) + ((loss_cycle_A + loss_cycle_B) / 2) + ((loss_identity_A + loss_identity_B) / 2)
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_real_D_A = criterion_GAN(D_A(real_A), valid)
            loss_fake_D_A = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_D_A + loss_fake_D_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_real_D_B = criterion_GAN(D_B(real_B), valid)
            loss_fake_D_B = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_D_B + loss_fake_D_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            g_loss_sum += loss_G.item()
            d_a_loss_sum += loss_D_A.item()
            d_b_loss_sum += loss_D_B.item()
        
        avg_g_loss = g_loss_sum / num_batches
        avg_d_a_loss = d_a_loss_sum / num_batches
        avg_d_b_loss = d_b_loss_sum / num_batches
        
        g_losses.append(avg_g_loss)
        d_a_losses.append(avg_d_a_loss)
        d_b_losses.append(avg_d_b_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | G_loss: {avg_g_loss:.4f} | D_A_loss: {avg_d_a_loss:.4f} | D_B_loss: {avg_d_b_loss:.4f}')
        
        # --- MODIFIED: 3-column visualization showing Original -> Painted -> Reconstructed ---
        if (epoch + 1) % save_interval == 0:
            G_AB.eval()
            G_BA.eval()
            with torch.no_grad():
                # Generate the full cycle: Real -> Painted -> Reconstructed Real
                generated_painted = G_AB(fixed_real)        # Real -> Painted
                reconstructed_real = G_BA(generated_painted) # Painted -> Real (reconstructed)
                
                # Create 3-column comparison: [original, painted, reconstructed] for each sample
                num_samples = fixed_real.size(0)
                comparison_images = torch.empty((num_samples * 3, *fixed_real.shape[1:]), device=device)
                
                # Fill the tensor with the pattern: [orig_1, painted_1, recon_1, orig_2, painted_2, recon_2, ...]
                comparison_images[0::3] = fixed_real           # Every 3rd index starting from 0: original
                comparison_images[1::3] = generated_painted    # Every 3rd index starting from 1: painted
                comparison_images[2::3] = reconstructed_real   # Every 3rd index starting from 2: reconstructed
                
                # De-normalize images from [-1, 1] to [0, 1] for saving
                comparison_images_denorm = comparison_images * 0.5 + 0.5
                
                # Create a grid with 3 columns: Original | Painted | Reconstructed
                grid = make_grid(comparison_images_denorm, nrow=3, padding=2, normalize=False)
                grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))
                
                # Save the image
                png_filename = os.path.join(save_dir, f"CycleGAN_3column_comparison_epoch_{epoch+1:04d}.png")
                plt.imsave(png_filename, grid_np)
                print(f"  -> Saved 3-column comparison: {png_filename}")
                
                # Calculate and display cycle consistency loss for monitoring
                cycle_loss_value = nn.L1Loss()(reconstructed_real, fixed_real).item()
                print(f"     Cycle consistency loss on fixed samples: {cycle_loss_value:.4f}")
                
            G_AB.train()
            G_BA.train()
    
    print("\n--- Training Finished ---")
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.legend()
    plt.title("Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(d_a_losses, label='D_A Loss (Real)')
    plt.plot(d_b_losses, label='D_B Loss (Painted)')
    plt.legend()
    plt.title("Discriminator Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cyclegan_training_summary.png'))
    plt.close()
    print(f"  -> Saved training loss plot")
    
    return g_losses, d_a_losses, d_b_losses

# Main execution
if __name__ == "__main__":
    # Load preprocessed data
    dataloader_real, dataloader_painted, metadata = load_cyclegan_data()
    
    # Training parameters
    num_epochs = 5      # Set to 5 epochs for a quick run
    save_interval = 1   # Save images after every epoch
    
    # Create models
    G_AB = CycleGAN_Generator().to(device)  # Real -> Painted
    G_BA = CycleGAN_Generator().to(device)  # Painted -> Real
    D_A = CycleGAN_Discriminator().to(device)  # Discriminator for Real faces
    D_B = CycleGAN_Discriminator().to(device)  # Discriminator for Painted faces
    
    print("Model Summary:")
    print(f"  G_AB: Real Faces -> Painted Faces")
    print(f"  G_BA: Painted Faces -> Real Faces")
    print(f"  D_A: Discriminator for Real Faces")
    print(f"  D_B: Discriminator for Painted Faces")
    
    # Train the model
    g_losses, d_a_losses, d_b_losses = train_cyclegan(
        G_AB, G_BA, D_A, D_B, dataloader_real, dataloader_painted, 
        num_epochs, device, save_interval
    )
    
    # Save trained models
    os.makedirs('./saved_cyclegan_models', exist_ok=True)
    torch.save(G_AB.state_dict(), './saved_cyclegan_models/generator_real_to_painted.pth')
    torch.save(G_BA.state_dict(), './saved_cyclegan_models/generator_painted_to_real.pth')
    torch.save(D_A.state_dict(), './saved_cyclegan_models/discriminator_real.pth')
    torch.save(D_B.state_dict(), './saved_cyclegan_models/discriminator_painted.pth')
    
    print("\nCycleGAN training complete!")
    print("Models saved in './saved_cyclegan_models/' directory")
    print("Generated images saved in './CycleGAN_generated_images/' directory")
    print("\nImage format: Each row shows [Original | Generated Painted | Reconstructed Original]")