import os
import shutil
import random
import yaml
from typing import List, Tuple, Dict, Optional


def _split_data_by_type(path_files: List[str]) -> Tuple[List[str], List[str]]:
    images = []
    labels = []
    image_exts = (".jpg", ".png")
    
    for f in path_files:
        f_lower = f.lower()
        if f_lower.endswith(image_exts):
            images.append(f)
        elif f_lower.endswith(".txt"):
            labels.append(f)
        else:
            print(f"[WARNING] Unknown file type: {f}")

    return images, labels


def _split_dataset(
    images: List[str], 
    labels: List[str], 
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Dict[str, Tuple[List[str], List[str]]]:
    
    image_names = {os.path.splitext(os.path.basename(img))[0]: img for img in images}
    label_names = {os.path.splitext(os.path.basename(lbl))[0]: lbl for lbl in labels}

    common_names = list(set(image_names.keys()) & set(label_names.keys()))
    
    if shuffle:
        random.seed(seed)
        random.shuffle(common_names)
    
    if max_samples is not None and max_samples < len(common_names):
        common_names = common_names[:max_samples]
    
    split_idx = int(len(common_names) * train_ratio)
    
    train_names = common_names[:split_idx]
    val_names = common_names[split_idx:]
    
    return {
        "train": (
            [image_names[n] for n in train_names],
            [label_names[n] for n in train_names]
        ),
        "val": (
            [image_names[n] for n in val_names],
            [label_names[n] for n in val_names]
        )
    }


def _copy_files(
    files: List[str], 
    src_folder: str, 
    dst_folder: str
) -> None:
    os.makedirs(dst_folder, exist_ok=True)
    for f in files:
        src_path = os.path.join(src_folder, f) if not os.path.isabs(f) else f
        dst_path = os.path.join(dst_folder, os.path.basename(f))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"[WARNING] File not found: {src_path}")


def _create_yaml_config(
    output_folder: str,
    class_names: List[str],
    yaml_name: str = "data.yaml"
) -> str:
    yaml_path = os.path.join(output_folder, yaml_name)
    
    data_yaml = {
        'path': os.path.abspath(output_folder),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"[INFO] Created {yaml_path}")
    return yaml_path


def dataset_builder(
    data_folder: str,
    output_folder: str,
    class_names: List[str] = ["text"],
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> str:
    """
    Build YOLO dataset từ folder chứa images và labels.
    
    Args:
        data_folder: Folder chứa cả images (.jpg, .png) và labels (.txt)
        output_folder: Folder output cho YOLO dataset
        class_names: Danh sách tên classes
        train_ratio: Tỷ lệ train (0.8 = 80% train, 20% val)
        shuffle: Có shuffle data không
        seed: Random seed
        max_samples: Giới hạn số lượng samples (None = tất cả)
        
    Returns:
        Path to data.yaml
    
    Output structure:
        output_folder/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/
    """
    print(f"[INFO] Building YOLO dataset...")
    print(f"[INFO] Source: {data_folder}")
    print(f"[INFO] Output: {output_folder}")
    
    train_images_dir = os.path.join(output_folder, "train", "images")
    train_labels_dir = os.path.join(output_folder, "train", "labels")
    val_images_dir = os.path.join(output_folder, "val", "images")
    val_labels_dir = os.path.join(output_folder, "val", "labels")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    all_files = os.listdir(data_folder)
    images, labels = _split_data_by_type(all_files)
    
    print(f"[INFO] Found {len(images)} images, {len(labels)} labels")
    
    splits = _split_dataset(images, labels, train_ratio, shuffle, seed, max_samples)
    
    train_images, train_labels = splits["train"]
    val_images, val_labels = splits["val"]
    
    print(f"[INFO] Train: {len(train_images)} samples")
    print(f"[INFO] Val: {len(val_images)} samples")

    print("[INFO] Copying train images...")
    _copy_files(train_images, data_folder, train_images_dir)
    print("[INFO] Copying train labels...")
    _copy_files(train_labels, data_folder, train_labels_dir)
    print("[INFO] Copying val images...")
    _copy_files(val_images, data_folder, val_images_dir)
    print("[INFO] Copying val labels...")
    _copy_files(val_labels, data_folder, val_labels_dir)
    
    yaml_path = _create_yaml_config(output_folder, class_names)
    
    print("[INFO] Dataset built successfully!")
    return yaml_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build YOLO dataset")
    parser.add_argument("--data", type=str, required=True, 
                        help="Data folder with images and labels")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder for YOLO dataset")
    parser.add_argument("--classes", type=str, nargs="+", default=["text"],
                        help="Class names")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle data")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples (for overfit test)")
    
    args = parser.parse_args()
    
    dataset_builder(
        data_folder=args.data,
        output_folder=args.output,
        class_names=args.classes,
        train_ratio=args.train_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_samples=args.max_samples
    )