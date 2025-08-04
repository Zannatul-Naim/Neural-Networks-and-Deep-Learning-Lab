# # dcgan_train.py

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
# from dcgan_load_datasets import load_dcgan_data

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.manual_seed(42)
# np.random.seed(42)

# # --- DCGAN Models (as provided) ---
# class DCGAN_G(nn.Module):
#     def __init__(self, z_dim=100, img_channels=3, features_g=64):
#         super().__init__()
#         self.gen = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, features_g*8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(features_g*8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(features_g*8, features_g*4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(features_g*4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(features_g*2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(features_g),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )
#     def forward(self, x):
#         return self.gen(x)

# class DCGAN_D(nn.Module):
#     def __init__(self, img_channels=3, features_d=64):
#         super().__init__()
#         self.disc = nn.Sequential(
#             nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(features_d*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(features_d*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(features_d*8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d*8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.disc(x).view(-1, 1).squeeze(1)

# # --- Training Function (as provided, with minor print clarification) ---
# def train_dcgan(generator, discriminator, dataloader, num_epochs, device, model_name="DCGAN", save_interval=5):
#     print(f"--- Training {model_name} ---")
#     save_dir = f"./{model_name}_generated_images"
#     os.makedirs(save_dir, exist_ok=True)
    
#     z_dim = 100
#     criterion = nn.BCELoss()
#     optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     g_losses, d_losses = [], []
#     fixed_noise = torch.randn(64, z_dim, 1, 1, device=device) # Generate 64 images for an 8x8 grid

#     for epoch in range(num_epochs):
#         for i, (real_imgs, _) in enumerate(dataloader):
#             real_imgs = real_imgs.to(device)
#             b_size = real_imgs.size(0)
            
#             # --- Train Discriminator ---
#             optimizerD.zero_grad()
#             # Real images
#             output_real = discriminator(real_imgs)
#             loss_real = criterion(output_real, torch.ones_like(output_real))
#             # Fake images
#             noise = torch.randn(b_size, z_dim, 1, 1, device=device)
#             fake_imgs = generator(noise)
#             output_fake = discriminator(fake_imgs.detach())
#             loss_fake = criterion(output_fake, torch.zeros_like(output_fake))
#             loss_D = loss_real + loss_fake
#             loss_D.backward()
#             optimizerD.step()
            
#             # --- Train Generator ---
#             optimizerG.zero_grad()
#             output = discriminator(fake_imgs)
#             loss_G = criterion(output, torch.ones_like(output))
#             loss_G.backward()
#             optimizerG.step()

#         print(f'Epoch [{epoch+1}/{num_epochs}] | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}')
#         g_losses.append(loss_G.item())
#         d_losses.append(loss_D.item())

#         if (epoch + 1) % save_interval == 0:
#             generator.eval()
#             with torch.no_grad():
#                 fake_imgs_grid = generator(fixed_noise).detach().cpu()
#                 grid = make_grid(fake_imgs_grid, padding=2, normalize=True)
#                 grid_np = np.transpose(grid.numpy(), (1, 2, 0))
#                 png_filename = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1:04d}.png")
#                 plt.imsave(png_filename, grid_np)
#                 print(f"  -> Saved generated image grid: {png_filename}")
#             generator.train()

#     return g_losses, d_losses


# # --- Main Execution Block ---
# if __name__ == "__main__":
#     # Hyperparameters
#     BATCH_SIZE = 64
#     NUM_EPOCHS = 25 # DCGAN needs a fair number of epochs to generate good images
#     Z_DIM = 100
#     SAVE_INTERVAL = 1 # Save images every 5 epochs
    
#     # Load data
#     dataloader, metadata = load_dcgan_data(batch_size=BATCH_SIZE)
    
#     # Create models
#     generator = DCGAN_G(z_dim=Z_DIM).to(device)
#     discriminator = DCGAN_D().to(device)

#     # Optional: Initialize weights
#     def weights_init(m):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             nn.init.normal_(m.weight.data, 0.0, 0.02)
#         elif classname.find('BatchNorm') != -1:
#             nn.init.normal_(m.weight.data, 1.0, 0.02)
#             nn.init.constant_(m.bias.data, 0)
#     generator.apply(weights_init)
#     discriminator.apply(weights_init)
    
#     print("Model Summary:")
#     print(generator)
#     print(discriminator)
    
#     # Train the model
#     g_losses, d_losses = train_dcgan(
#         generator, discriminator, dataloader, NUM_EPOCHS, device, save_interval=SAVE_INTERVAL
#     )
    
#     # Save the final trained generator model
#     os.makedirs('./saved_dcgan_models', exist_ok=True)
#     torch.save(generator.state_dict(), './saved_dcgan_models/dcgan_generator.pth')
#     print("Final generator model saved.")


# dcgan_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image # Import save_image
from dcgan_load_datasets import load_dcgan_data
import pandas as pd # Import pandas for CSV

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# --- DCGAN Models (as provided) ---
class DCGAN_G(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, features_g=64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g*8, features_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)

class DCGAN_D(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x).view(-1, 1).squeeze(1)

# --- Modified Training Function ---
def train_dcgan(generator, discriminator, dataloader, num_epochs, device, model_name="DCGAN", save_interval=5):
    print(f"--- Training {model_name} ---")
    save_dir = f"./{model_name}_generated_images"
    os.makedirs(save_dir, exist_ok=True)

    z_dim = 100
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss tracking per epoch
    epoch_losses_g, epoch_losses_d = [], []

    # Fixed noise for consistent image generation across epochs (4x4 grid = 16 images)
    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # --- Train Discriminator ---
            optimizerD.zero_grad()
            # Real images
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, torch.ones_like(output_real))
            # Fake images
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, torch.zeros_like(output_fake))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizerD.step()

            # --- Train Generator ---
            optimizerG.zero_grad()
            output = discriminator(fake_imgs)
            loss_G = criterion(output, torch.ones_like(output))
            loss_G.backward()
            optimizerG.step()

             # Accumulate losses for averaging
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            num_batches += 1

        # Calculate and store average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        epoch_losses_g.append(avg_g_loss)
        epoch_losses_d.append(avg_d_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}')

        # Save images at specified intervals
        if (epoch + 1) % save_interval == 0:
            generator.eval()
            with torch.no_grad():
                fake_imgs_grid = generator(fixed_noise).detach().cpu()
                # Create and save a 4x4 grid using torchvision.utils.save_image
                grid_img = make_grid(fake_imgs_grid, nrow=4, padding=2, normalize=True)
                png_filename = os.path.join(save_dir, f"{model_name}_grid_epoch_{epoch+1:04d}.png")
                save_image(grid_img, png_filename) # save_image handles saving the tensor grid
                print(f"  -> Saved 4x4 generated image grid: {png_filename}")
            generator.train()

    # --- Save Training History (Loss Plot and CSV) ---
    print("\n--- Saving Training History ---")
    # plt.figure(figsize=(12, 5))

    # Plot Generator and Discriminator Losses
    # plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_losses_g) + 1), epoch_losses_g, label='Generator Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(epoch_losses_d) + 1), epoch_losses_d, label='Discriminator Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Loss Ratio (G/D)
    # plt.subplot(1, 2, 2)
    loss_ratio = [g/d if d != 0 else 0 for g, d in zip(epoch_losses_g, epoch_losses_d)]
    # plt.plot(range(1, len(loss_ratio) + 1), loss_ratio, label='G/D Loss Ratio', color='green', linewidth=2)
    # plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
    # plt.xlabel('Epoch')
    # plt.ylabel('Ratio')
    # plt.title('Generator/Discriminator Loss Ratio')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved training history plot: {plot_filename}")

    # Save loss data as CSV
    loss_data = pd.DataFrame({
        'epoch': range(1, len(epoch_losses_g) + 1),
        'generator_loss': epoch_losses_g,
        'discriminator_loss': epoch_losses_d,
        'loss_ratio': loss_ratio
    })
    csv_filename = os.path.join(save_dir, f'{model_name}_training_losses.csv')
    loss_data.to_csv(csv_filename, index=False)
    print(f"  -> Saved training data: {csv_filename}")

    return epoch_losses_g, epoch_losses_d


# --- Main Execution Block ---
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    Z_DIM = 100
    SAVE_INTERVAL = 5 # Save images every 5 epochs (adjust as needed)
    MODEL_NAME = "DCGAN" # Define model name for saving

    # Load data
    dataloader, metadata = load_dcgan_data(batch_size=BATCH_SIZE)

    # Create models
    generator = DCGAN_G(z_dim=Z_DIM).to(device)
    discriminator = DCGAN_D().to(device)

    # Optional: Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print("Model Summary:")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Total parameters: {sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,}")

    # Train the model
    g_losses, d_losses = train_dcgan(
        generator, discriminator, dataloader, NUM_EPOCHS, device, model_name=MODEL_NAME, save_interval=SAVE_INTERVAL
    )

    # Save the final trained models
    os.makedirs('./saved_dcgan_models', exist_ok=True)
    torch.save(generator.state_dict(), f'./saved_dcgan_models/{MODEL_NAME.lower()}_generator.pth')
    torch.save(discriminator.state_dict(), f'./saved_dcgan_models/{MODEL_NAME.lower()}_discriminator.pth')
    print(f"\nFinal {MODEL_NAME} models saved to './saved_dcgan_models/'")
    print(f"Generated images and training history saved to './{MODEL_NAME}_generated_images/'")