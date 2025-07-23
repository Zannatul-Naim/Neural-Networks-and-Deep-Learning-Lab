import os
import zipfile
import requests
from tqdm import tqdm
from PIL import Image

base_dir = "datasets/widerface"

os.makedirs(base_dir, exist_ok=True)

def download(url, dest):
    if os.path.exists(dest):
        return

def unzip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def convert_to_yolo(txt_path, img_dir, label_dir):
    count_valid = 0
    count_missing = 0
    os.makedirs(label_dir, exist_ok=True)

    with open(txt_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(".jpg"):
            image_rel_path = line
            i += 1
            try:
                bbox_count = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            i += 1

            full_img_path = os.path.join(img_dir, image_rel_path)
            if not os.path.exists(full_img_path):
                i += bbox_count
                count_missing += 1
                continue

            try:
                img = Image.open(full_img_path)
                iw, ih = img.size
            except:
                i += bbox_count
                continue

            yolo_lines = []
            for _ in range(bbox_count):
                parts = lines[i].strip().split()
                i += 1
                if len(parts) < 4:
                    continue
                x, y, w, h = map(float, parts[:4])
                if w <= 1 or h <= 1:
                    continue
                cx = (x + w / 2) / iw
                cy = (y + h / 2) / ih
                nw = w / iw
                nh = h / ih
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if yolo_lines:
                label_path = os.path.join(label_dir, os.path.splitext(image_rel_path)[0] + ".txt")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                with open(label_path, "w") as out:
                    out.write("\n".join(yolo_lines))
                count_valid += 1
            else:
                count_missing += 1
        else:
            i += 1

if __name__ == "__main__":

    unzip("WIDER_train.zip", base_dir)
    unzip("WIDER_val.zip", base_dir)

    convert_to_yolo("wider_face_train_bbx_gt.txt",
                    os.path.join(base_dir, "WIDER_train", "images"),
                    os.path.join(base_dir, "WIDER_train", "labels"))

    convert_to_yolo("wider_face_val_bbx_gt.txt",
                    os.path.join(base_dir, "WIDER_val", "images"),
                    os.path.join(base_dir, "WIDER_val", "labels"))

    os.system(f"chmod -R u+w {base_dir}")

    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {base_dir}
                train: WIDER_train/images
                val: WIDER_val/images
                nc: 1
                names: ['face']
                """)