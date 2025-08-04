# # cgan_train.py

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
# from cgan_load_datasets import load_cgan_data
# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image

# # --- Class definition for pickle ---
# class CelebAConditionalDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_attribute='Black_Hair'):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.attributes = pd.read_csv(annotations_file)
#         self.image_id_column = 'image_id'
#         self.attributes = self.attributes[[self.image_id_column, target_attribute]]
#         self.attributes[target_attribute] = self.attributes[target_attribute].apply(lambda x: 1 if x == 1 else 0)
#     def __len__(self): return len(self.attributes)
#     def __getitem__(self, idx):
#         img_filename = self.attributes.loc[idx, self.image_id_column]
#         label = self.attributes.loc[idx, 'Black_Hair']
#         img_path = os.path.join(self.img_dir, img_filename)
#         image = Image.open(img_path).convert("RGB")
#         if self.transform: image = self.transform(image)
#         return image, torch.tensor([label], dtype=torch.float32)

# # --- Device setup and Model definitions ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.manual_seed(42); np.random.seed(42)

# class CGAN_G(nn.Module):
#     def __init__(self, z_dim=100, n_classes=2, img_channels=3, features_g=64):
#         super().__init__()
#         self.label_emb = nn.Embedding(n_classes, n_classes)
#         self.gen = nn.Sequential(
#             nn.ConvTranspose2d(z_dim + n_classes, features_g*8, 4, 1, 0, bias=False), nn.BatchNorm2d(features_g*8), nn.ReLU(True),
#             nn.ConvTranspose2d(features_g*8, features_g*4, 4, 2, 1, bias=False), nn.BatchNorm2d(features_g*4), nn.ReLU(True),
#             nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features_g*2), nn.ReLU(True),
#             nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False), nn.BatchNorm2d(features_g), nn.ReLU(True),
#             nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False), nn.Tanh() )
#     def forward(self, noise, labels):
#         labels_squeezed = labels.squeeze(1).long()
#         label_input = self.label_emb(labels_squeezed).unsqueeze(2).unsqueeze(3)
#         x = torch.cat([noise, label_input], 1)
#         return self.gen(x)

# class CGAN_D(nn.Module):
#     def __init__(self, n_classes=2, img_channels=3, features_d=64):
#         super().__init__()
#         self.label_emb = nn.Embedding(n_classes, n_classes)
#         self.disc = nn.Sequential(
#             nn.Conv2d(img_channels + n_classes, features_d, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features_d*2), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False), nn.BatchNorm2d(features_d*4), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias=False), nn.BatchNorm2d(features_d*8), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(features_d*8, 1, 4, 1, 0, bias=False), nn.Sigmoid() )
#     def forward(self, img, labels):
#         labels_squeezed = labels.squeeze(1).long()
#         label_input = self.label_emb(labels_squeezed)
#         label_map = label_input.unsqueeze(2).unsqueeze(3).expand(-1, -1, img.size(2), img.size(3))
#         x = torch.cat([img, label_map], 1)
#         return self.disc(x).view(-1, 1).squeeze(1)

# def train_cgan(generator, discriminator, dataloader, num_epochs, device, model_name="cGAN", save_interval=5):
#     print(f"--- Training {model_name} ---")
#     save_dir = f"./{model_name}_generated_images"
#     os.makedirs(save_dir, exist_ok=True)
#     z_dim = 100
#     criterion = nn.BCELoss()
#     optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     g_losses, d_losses = [], []
#     fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
#     fixed_labels = torch.cat([torch.zeros(8), torch.ones(8)]).long().to(device)
#     for epoch in range(num_epochs):
#         for i, (real_imgs, labels) in enumerate(dataloader):
#             real_imgs, labels = real_imgs.to(device), labels.to(device)
#             b_size = real_imgs.size(0)
#             optimizerD.zero_grad()
#             output_real = discriminator(real_imgs, labels)
#             loss_real = criterion(output_real, torch.ones_like(output_real))
#             noise = torch.randn(b_size, z_dim, 1, 1, device=device)
#             fake_imgs = generator(noise, labels)
#             output_fake = discriminator(fake_imgs.detach(), labels)
#             loss_fake = criterion(output_fake, torch.zeros_like(output_fake))
#             loss_D = loss_real + loss_fake
#             loss_D.backward()
#             optimizerD.step()
#             optimizerG.zero_grad()
#             output = discriminator(fake_imgs, labels)
#             loss_G = criterion(output, torch.ones_like(output))
#             loss_G.backward()
#             optimizerG.step()
#         print(f'Epoch [{epoch+1}/{num_epochs}] | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}')
#         g_losses.append(loss_G.item()); d_losses.append(loss_D.item())
#         if (epoch + 1) % save_interval == 0:
#             generator.eval()
#             with torch.no_grad():
#                 fake_imgs_grid = generator(fixed_noise, fixed_labels.unsqueeze(1).float()).detach().cpu()
#                 grid = make_grid(fake_imgs_grid, nrow=8, padding=2, normalize=True)
#                 grid_np = np.transpose(grid.numpy(), (1, 2, 0))
#                 plt.imsave(os.path.join(save_dir, f"{model_name}_epoch_{epoch+1:04d}.png"), grid_np)
#                 print(f"  -> Saved conditional image grid: {os.path.join(save_dir, f'{model_name}_epoch_{epoch+1:04d}.png')}")
#             generator.train()
#     return g_losses, d_losses

# if __name__ == "__main__":
#     BATCH_SIZE, NUM_EPOCHS, Z_DIM, N_CLASSES, SAVE_INTERVAL = 128, 50, 100, 2, 5
#     dataloader, metadata = load_cgan_data(batch_size=BATCH_SIZE, attribute='Black_Hair')
#     generator = CGAN_G(z_dim=Z_DIM, n_classes=N_CLASSES).to(device)
#     discriminator = CGAN_D(n_classes=N_CLASSES).to(device)
    
#     def weights_init(m):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
#         elif classname.find('BatchNorm') != -1: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
#     generator.apply(weights_init); discriminator.apply(weights_init)
    
#     print("Model Summary:"); print(generator); print(discriminator)
    
#     g_losses, d_losses = train_cgan(generator, discriminator, dataloader, NUM_EPOCHS, device, save_interval=SAVE_INTERVAL)
    
#     os.makedirs('./saved_cgan_models', exist_ok=True)
#     torch.save(generator.state_dict(), './saved_cgan_models/cgan_generator_blackhair.pth')
#     print("Final conditional generator model saved.")


# cgan_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from cgan_load_datasets import load_cgan_data
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

# --- Class definition for pickle ---
class CelebAConditionalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_attribute='Bangs'):
        self.img_dir = img_dir
        self.transform = transform
        self.attributes = pd.read_csv(annotations_file)
        self.image_id_column = 'image_id'
        self.attributes = self.attributes[[self.image_id_column, target_attribute]]
        self.attributes[target_attribute] = self.attributes[target_attribute].apply(lambda x: 1 if x == 1 else 0)
    def __len__(self): return len(self.attributes)
    def __getitem__(self, idx):
        img_filename = self.attributes.loc[idx, self.image_id_column]
        label = self.attributes.loc[idx, 'Bangs']
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

# --- Device setup and Model definitions ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42); np.random.seed(42)

class CGAN_G(nn.Module):
    def __init__(self, z_dim=100, n_classes=2, img_channels=3, features_g=64):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim + n_classes, features_g*8, 4, 1, 0, bias=False), nn.BatchNorm2d(features_g*8), nn.ReLU(True),
            nn.ConvTranspose2d(features_g*8, features_g*4, 4, 2, 1, bias=False), nn.BatchNorm2d(features_g*4), nn.ReLU(True),
            nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features_g*2), nn.ReLU(True),
            nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False), nn.BatchNorm2d(features_g), nn.ReLU(True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False), nn.Tanh() )
    def forward(self, noise, labels):
        labels_squeezed = labels.squeeze(1).long()
        label_input = self.label_emb(labels_squeezed).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_input], 1)
        return self.gen(x)

class CGAN_D(nn.Module):
    def __init__(self, n_classes=2, img_channels=3, features_d=64):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels + n_classes, features_d, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False), nn.BatchNorm2d(features_d*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False), nn.BatchNorm2d(features_d*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias=False), nn.BatchNorm2d(features_d*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d*8, 1, 4, 1, 0, bias=False), nn.Sigmoid() )
    def forward(self, img, labels):
        labels_squeezed = labels.squeeze(1).long()
        label_input = self.label_emb(labels_squeezed)
        label_map = label_input.unsqueeze(2).unsqueeze(3).expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, label_map], 1)
        return self.disc(x).view(-1, 1).squeeze(1)

def create_labeled_subplot_grid(images_tensor, labels, label_names=["No Bangs", "Bangs"], save_path=None):
    """Create a matplotlib subplot grid with labeled images"""
    # Convert tensor to numpy and denormalize
    images_np = images_tensor.numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
    images_np = (images_np + 1) / 2.0  # Denormalize from [-1,1] to [0,1]
    
    # Create subplot grid (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    # fig.suptitle('Conditional GAN Generated Images', fontsize=16, fontweight='bold')
    
    # Plot each image in its subplot
    for i, (img, label) in enumerate(zip(images_np, labels)):
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(label_names[int(label)], fontsize=10)
        axes[row, col].axis('off')  # Remove axes for cleaner look
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)  # Make room for main title
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return None
    else:
        return fig

def train_cgan(generator, discriminator, dataloader, num_epochs, device, model_name="cGAN", save_interval=5):
    print(f"--- Training {model_name} ---")
    save_dir = f"./{model_name}_generated_images"
    os.makedirs(save_dir, exist_ok=True)
    z_dim = 100
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # Loss tracking
    g_losses, d_losses = [], []
    epoch_losses_g, epoch_losses_d = [], []
    # Fixed samples for visualization - 4x4 grid (16 total images)
    # Example: Generate 8 "No Bangs" (label 0) and 8 "Bangs" (label 1)
    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
    # Create labels for the 4x4 grid (e.g., alternating or grouped)
    # This example groups first 8 as 0 (No Bangs) and next 8 as 1 (Bangs)
    fixed_labels = torch.cat([torch.zeros(8), torch.ones(8)]).long().to(device)

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        for i, (real_imgs, labels) in enumerate(dataloader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            b_size = real_imgs.size(0)
            # Train Discriminator
            optimizerD.zero_grad()
            # Real images
            output_real = discriminator(real_imgs, labels)
            loss_real = criterion(output_real, torch.ones_like(output_real))
            # Fake images
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_imgs = generator(noise, labels)
            output_fake = discriminator(fake_imgs.detach(), labels)
            loss_fake = criterion(output_fake, torch.zeros_like(output_fake))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizerD.step()
            # Train Generator
            optimizerG.zero_grad()
            output = discriminator(fake_imgs, labels)
            loss_G = criterion(output, torch.ones_like(output))
            loss_G.backward()
            optimizerG.step()
            # Track losses
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            num_batches += 1
        # Calculate average losses for this epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        epoch_losses_g.append(avg_g_loss)
        epoch_losses_d.append(avg_d_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}')
        # Save images and update loss tracking
        if epoch == 0 or (epoch + 1) % save_interval == 0:
            generator.eval()
            with torch.no_grad():
                # Generate images using fixed noise and labels
                fake_imgs_grid = generator(fixed_noise, fixed_labels.unsqueeze(1).float()).detach().cpu()

                # Create a 4x4 grid using torchvision.utils.make_grid
                # Normalize images from [-1, 1] to [0, 1] for saving
                grid_img = make_grid(fake_imgs_grid, nrow=4, padding=2, normalize=True, value_range=(-1, 1))

                # Save the grid image
                filename = os.path.join(save_dir, f"{model_name}_grid_epoch_{epoch+1:04d}.png")
                # torchvision save requires a PIL Image or converting tensor appropriately.
                # torchvision.utils.save_image is the standard way.
                from torchvision.utils import save_image
                save_image(grid_img, filename, nrow=4, padding=2, normalize=False) # normalize=False because we already normalized in make_grid
                print(f"  -> Saved 4x4 grid image: {filename}")
            generator.train()
    # Plot and save training history
    print("\n--- Saving Training History ---")
    # plt.figure(figsize=(12, 5))
    # Plot losses
    # plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_losses_g) + 1), epoch_losses_g, label='Generator Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(epoch_losses_d) + 1), epoch_losses_d, label='Discriminator Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('cGAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Plot loss ratio (optional - helps see balance)
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
    # Save loss data as CSV for further analysis
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

if __name__ == "__main__":
    BATCH_SIZE, NUM_EPOCHS, Z_DIM, N_CLASSES, SAVE_INTERVAL = 64, 50, 100, 2, 5
    dataloader, metadata = load_cgan_data(batch_size=BATCH_SIZE, attribute='Bangs')
    generator = CGAN_G(z_dim=Z_DIM, n_classes=N_CLASSES).to(device)
    discriminator = CGAN_D(n_classes=N_CLASSES).to(device)
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)
    generator.apply(weights_init); discriminator.apply(weights_init)
    
    print("Model Summary:")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Total parameters: {sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,}")
    
    g_losses, d_losses = train_cgan(generator, discriminator, dataloader, NUM_EPOCHS, device, save_interval=SAVE_INTERVAL)
    
    # Save final model
    os.makedirs('./saved_cgan_models', exist_ok=True)
    torch.save(generator.state_dict(), './saved_cgan_models/cgan_generator_bangs_hair.pth')
    torch.save(discriminator.state_dict(), './saved_cgan_models/cgan_discriminator_bangs_hair.pth')
    print(f"\nFinal cGAN models saved to './saved_cgan_models/'")
    print(f"Generated images and training history saved to './cGAN_generated_images/'")
    print(f"Image format: 2x4 grid with labels (Top row: No Bangs, Bottom row: Bangs)")