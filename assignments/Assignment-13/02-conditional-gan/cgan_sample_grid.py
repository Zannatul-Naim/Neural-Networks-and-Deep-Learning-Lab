# cgan_sample_grid.py


import os, torch, torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import pandas as pd
import pickle
from cgan_load_datasets import load_cgan_data

# ---- same class for pickle
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

# ----------------------------------------------------------
dataloader, _ = load_cgan_data(batch_size=32, attribute='Bangs')
save_dir = "./CGAN_sample_grids"
os.makedirs(save_dir, exist_ok=True)

# grab 16 samples
imgs, labels = next(iter(dataloader))
imgs  = imgs[:16]
labels = labels[:16].squeeze().tolist()   # 0 or 1

# create 4×4 grid
grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=True, value_range=(-1, 1))
grid_np = grid.permute(1, 2, 0).numpy()
grid_img = Image.fromarray((grid_np * 255).astype('uint8'))

# draw labels
draw = ImageDraw.Draw(grid_img)
font = ImageFont.load_default()
cell_h, cell_w = grid_img.height // 4, grid_img.width // 4
for i, label in enumerate(labels):
    row, col = divmod(i, 4)
    x = col * cell_w + 5
    y = row * cell_h + 5
    text = "Bangs" if label else "Not Bangs"
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

grid_img.save(f"{save_dir}/conditional_bangs_4x4_labeled.png")
print("Saved:", f"{save_dir}/conditional_bangs_4x4_labeled.png")