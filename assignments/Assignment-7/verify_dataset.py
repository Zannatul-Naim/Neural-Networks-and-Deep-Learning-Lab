
import os
from PIL import Image

def verify_dataset(image_dir, label_dir):
    total_images = 0
    missing_labels = 0
    empty_labels = 0
    bad_labels = 0

    for root, _, files in os.walk(image_dir):
        for file in files:
            if not file.endswith((".jpg")):
                continue

            total_images += 1
            image_path = os.path.join(root, file)
            rel_path = os.path.relpath(image_path, image_dir)
            label_path = os.path.join(label_dir, os.path.splitext(rel_path)[0] + ".txt")

            if not os.path.exists(label_path):
                missing_labels += 1
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines:
                empty_labels += 1
                continue

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    bad_labels += 1
                    break

                try:
                    class_id, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        raise ValueError
                except ValueError:
                    bad_labels += 1
                    break

    print(f"Total images: {total_images}")
    print(f"Missing labels: {missing_labels}")
    print(f"Empty labels: {empty_labels}")
    print(f"Invalid label files: {bad_labels}")

    if missing_labels or empty_labels or bad_labels:
        print("\nDataset has issues.")
    else:
        print("\nAll images and labell are valid!")

if __name__ == "__main__":

    image_root = "datasets/widerface/WIDER_train/images"
    label_root = "datasets/widerface/WIDER_train/labels"

    verify_dataset(image_root, label_root)
