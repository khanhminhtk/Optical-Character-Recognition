import os
import shutil
import glob
import random
from pathlib import Path

source_dir = './data_test/text_recognizer_data'
target_dir = './data_test/overfit_50'
labels_file = 'labels.txt'

os.makedirs(target_dir, exist_ok=True)

all_images = sorted(glob.glob(os.path.join(source_dir, '*.jpg')))
random.seed(42)
selected_images = random.sample(all_images, 50)

print(f"Copying 50 random images to {target_dir}...")
for img_path in selected_images:
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(target_dir, img_name))
    print(f"  Copied: {img_name}")

source_labels = os.path.join(source_dir, labels_file)
if os.path.exists(source_labels):
    selected_names = set(os.path.basename(p) for p in selected_images)
    
    with open(source_labels, 'r') as f:
        all_labels = f.readlines()
    
    filtered_labels = []
    for line in all_labels:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_name = os.path.basename(parts[0])
            if img_name in selected_names:
                filtered_labels.append(line)
    
    target_labels = os.path.join(target_dir, labels_file)
    with open(target_labels, 'w') as f:
        f.writelines(filtered_labels)
    
    print(f"\nCreated {target_labels} with {len(filtered_labels)} entries")
else:
    print(f"\nWARNING: {source_labels} not found")

print(f"\nDone! Dataset ready at {target_dir}")
print(f"Total images: {len(selected_images)}")
