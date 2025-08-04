import torchvision.utils as vutils
import os
from cyclegan_load_datasets import load_cyclegan_data  

# ----------------------------------------------------------
# Save 4×4 random grids to disk
# ----------------------------------------------------------
dataloader_real, dataloader_painted, metadata = load_cyclegan_data()
save_dir = "./CycleGAN_sample_grids"
os.makedirs(save_dir, exist_ok=True)

def save_grid(loader, title, out_path, n=16):
    batch, _ = next(iter(loader))
    imgs = batch[:n].cpu()
    grid = vutils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, out_path)
    print(f"Saved: {out_path}")

save_grid(dataloader_real,  "Real Faces",  f"{save_dir}/real_faces_4x4.png")
save_grid(dataloader_painted, "Painted Faces", f"{save_dir}/painted_faces_4x4.png")