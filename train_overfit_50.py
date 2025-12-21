import sys
import os
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import shutil
import random
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from infrastructure.ml.models.vit_ctc.model import ModelTextRecoginizer
from infrastructure.ml.models.vit_ctc.pos_patch_embed import PatchEmbedWithPos
from infrastructure.ml.models.vit_ctc.patch_embedding import PatchEmbedding
from infrastructure.ml.models.vit_ctc.position_embedding import PosEmbedding
from infrastructure.ml.models.vit_ctc.backbone import Backbone
from infrastructure.ml.models.vit_ctc.transformer import TransformerBlock
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, mobilenet_v3_small


def prepare_subset_data(source_dir, output_dir, num_samples=50):
    print("="*80)
    print("PREPARING SUBSET DATA")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    labels_file = os.path.join(source_dir, 'labels.txt')
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    valid_samples = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_path = parts[0]
            label = ' '.join(parts[1:])
            img_name = os.path.basename(img_path)
            local_img_path = os.path.join(source_dir, img_name)
            
            if os.path.exists(local_img_path):
                valid_samples.append((img_name, label, line))
    
    print(f"Found {len(valid_samples)} valid samples")
    
    # Force include '000013.jpg' if available (for debugging 'wines')
    target_img = '000013.jpg'
    forced_sample = None
    remaining_samples = []
    
    for s in valid_samples:
        if s[0] == target_img:
            forced_sample = s
        else:
            remaining_samples.append(s)
            
    num_to_sample = min(num_samples, len(valid_samples))
    if forced_sample:
        print(f"Force including {target_img}")
        selected = [forced_sample] + random.sample(remaining_samples, min(num_to_sample - 1, len(remaining_samples)))
    else:
        print(f"Warning: {target_img} not found in valid samples")
        selected = random.sample(valid_samples, num_to_sample)
    
    print(f"Selected {len(selected)} samples for training")
    
    new_labels = []
    for img_name, label, original_line in selected:
        src = os.path.join(source_dir, img_name)
        dst = os.path.join(output_dir, img_name)
        shutil.copy2(src, dst)
        new_labels.append(original_line)
    
    labels_out = os.path.join(output_dir, 'labels.txt')
    with open(labels_out, 'w') as f:
        f.writelines(new_labels)
        
    # Validating list of images
    selected_list_out = os.path.join(output_dir, 'selected_images.txt')
    with open(selected_list_out, 'w') as f:
        for img_name, _, _ in selected:
            f.write(f"{img_name}\n")
    
    print(f"Created selected images list: {selected_list_out}")
    
    print(f"Copied {len(selected)} images to {output_dir}")
    print(f"Created labels file: {labels_out}")
    print("="*80)
    print()
    
    return len(selected)


class OverfitDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        with open(self.data_dir / 'labels.txt', 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = ' '.join(parts[1:])
                img_name = os.path.basename(img_path)
                local_img_path = self.data_dir / img_name
                
                if local_img_path.exists():
                    self.samples.append((str(local_img_path), img_name, label))
        
        self.char_to_idx = {chr(i): i-ord('a') for i in range(ord('a'), ord('z')+1)}
        self.char_to_idx[' '] = 26
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, img_name, label = self.samples[idx]
        
        image = cv2.imread(img_path)
        label_lower = label.lower()
        label_indices = []
        for char in label_lower:
            if char in self.char_to_idx:
                label_indices.append(self.char_to_idx[char])
        
        if len(label_indices) == 0:
            label_indices = [26]
        
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        rows = 1
        cols = 40
        
        return image, label_tensor, rows, cols, img_name


def collate_fn(batch):
    images, labels, rows, cols, img_names = zip(*batch)
    
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)
    
    return list(images), targets, target_lengths, rows[0], cols[0], list(img_names)


def create_model():
    rows = 1
    cols = 40
    num_patches = rows * cols
    num_classes = 28
    
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = Backbone(model=mobilenet)
    patch_embedding = PatchEmbedding(
        backbone=backbone,
        num_patches=num_patches,
        rows=rows,
        cols=cols
    )
    backbone_output_dim = 1000
    pos_embedding = PosEmbedding(
        dim=backbone_output_dim,
        num_patches=num_patches
    )
    patch_embed = PatchEmbedWithPos(
        patch_embedding=patch_embedding,
        pos_embedding=pos_embedding
    )
    
    transformer = TransformerBlock(
        n_block=6,
        nhead=8,
        dim=backbone_output_dim,
        drop_out=0.1
    )
    
    model = ModelTextRecoginizer(
        patchembedwithpos=patch_embed,
        transformer=transformer,
        num_classes=num_classes,
        embedding_dim=backbone_output_dim,
        hidden_classifier=512,
        dropout=0.1
    )
    
    return model


def main():
    source_dir = './data_test/text_recognizer_data/'
    subset_dir = './temp/overfit_50/'
    checkpoint_dir = './temp/checkpoints_overfit/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    num_epochs = 200
    lr = 1e-3
    
    print("\n" + "="*80)
    print("OVERFIT TRAINING ON 50 IMAGES")
    print("="*80)
    print(f"Source dir:      {source_dir}")
    print(f"Subset dir:      {subset_dir}")
    print(f"Checkpoint dir:  {checkpoint_dir}")
    print(f"Device:          {device}")
    print(f"Batch size:      {batch_size}")
    print(f"Epochs:          {num_epochs}")
    print(f"Learning rate:   {lr}")
    print("="*80)
    print()
    
    num_samples = prepare_subset_data(source_dir, subset_dir, num_samples=50)
    
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    print("Loading dataset...")
    dataset = OverfitDataset(subset_dir)
    print(f"Dataset size: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss_fn = torch.nn.CTCLoss(blank=27, zero_infinity=True)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, targets, target_lengths, rows, cols, img_names) in enumerate(pbar):
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, rows, cols, return_sequence=True)
            
            batch_size = len(images)
            if outputs.dim() == 3:
                seq_len = outputs.size(1)
                outputs = outputs.permute(1, 0, 2)
            else:
                seq_len = 1
                outputs = outputs.unsqueeze(0)
            
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
            
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'images': ', '.join(img_names[:3])
            })
        
        avg_loss = epoch_loss / num_batches
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(checkpoint_dir, 'best_overfit.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved best model (loss: {avg_loss:.4f})")
        
        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(checkpoint_dir, f'overfit_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        if avg_loss < 0.01:
            print(f"\n{'='*80}")
            print(f"OVERFIT ACHIEVED! Loss < 0.01 at epoch {epoch+1}")
            print(f"{'='*80}")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*80)


if __name__ == "__main__":
    random.seed(42)
    main()
