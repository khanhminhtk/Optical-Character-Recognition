import sys
import os
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.ml.models.vit_ctc.model import ModelTextRecoginizer
from infrastructure.ml.models.vit_ctc.pos_patch_embed import PatchEmbedWithPos
from infrastructure.ml.models.vit_ctc.patch_embedding import PatchEmbedding
from infrastructure.ml.models.vit_ctc.position_embedding import PosEmbedding
from infrastructure.ml.models.vit_ctc.backbone import Backbone
from infrastructure.ml.models.vit_ctc.transformer import TransformerBlock
from infrastructure.ml.models.vit_ctc.trainer_factory import create_trainer, create_trainer_with_ctc
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TextRecognizerDataset(Dataset):
    def __init__(self, data_dir, labels_file='labels.txt', split='train', train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        
        with open(self.data_dir / labels_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = ' '.join(parts[1:])
                img_name = os.path.basename(img_path)
                local_img_path = self.data_dir / img_name
                
                if local_img_path.exists():
                    self.samples.append((str(local_img_path), label))
        
        total = len(self.samples)
        split_idx = int(total * train_ratio)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        self.char_to_idx = {chr(i): i-ord('a') for i in range(ord('a'), ord('z')+1)}
        self.char_to_idx[' '] = 26
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = cv2.imread(img_path)
        label_lower = label.lower()
        label_indices = []
        for char in label_lower:
            if char in self.char_to_idx:
                label_indices.append(self.char_to_idx[char])
        
        if len(label_indices) == 0:
            label_indices = [26]
        
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        rows = 4
        cols = 2
        
        return image, label_tensor, rows, cols


def collate_fn(batch):
    images, labels, rows, cols = zip(*batch)
    
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels)
    
    return list(images), targets, target_lengths, rows[0], cols[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train CTC text recognition model')
    
    parser.add_argument('--data_dir', type=str, default='./data_test/text_recognizer_data/',
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_ctc',
                        help='Checkpoint directory')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs_ctc',
                        help='TensorBoard directory')
    
    parser.add_argument('--img_height', type=int, default=224,
                        help='Image height')
    parser.add_argument('--img_width', type=int, default=224,
                        help='Image width')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--mlp_dim', type=int, default=512,
                        help='MLP dimension')
    parser.add_argument('--hidden_classifier', type=int, default=512,
                        help='Hidden classifier dimension')
    parser.add_argument('--num_classes', type=int, default=27,
                        help='Number of classes (26 chars + 1 blank for CTC)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=5.0,
                        help='Gradient clipping value')
    
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[0.5, 0.3, 0.2],
                        help='Loss weights [CTC, Focal, LabelSmoothing]')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing value')
    
    parser.add_argument('--use_ctc', action='store_true',
                        help='Use CTC loss')
    parser.add_argument('--ctc_blank', type=int, default=26,
                        help='CTC blank index')
    
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Scheduler type')
    
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use TensorBoard logging')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


def create_model(args):
    rows = 4
    cols = 2
    num_patches = rows * cols
    
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
        n_block=args.num_layers,
        nhead=args.num_heads,
        dim=backbone_output_dim,
        drop_out=args.dropout
    )
    
    model = ModelTextRecoginizer(
        patchembedwithpos=patch_embed,
        transformer=transformer,
        num_classes=args.num_classes,
        embedding_dim=backbone_output_dim,
        hidden_classifier=args.hidden_classifier,
        dropout=args.dropout
    )
    
    return model


def main():
    args = parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    print("\n" + "="*60)
    print("CTC Training Configuration")
    print("="*60)
    print(f"Data Dir:          {args.data_dir}")
    print(f"Checkpoint Dir:    {args.checkpoint_dir}")
    print(f"Tensorboard Dir:   {args.tensorboard_dir}")
    print(f"Image Size:        {args.img_height}x{args.img_width}")
    print(f"Patch Size:        {args.patch_size}")
    print(f"Embedding Dim:     {args.embedding_dim}")
    print(f"Num Layers:        {args.num_layers}")
    print(f"Num Heads:         {args.num_heads}")
    print(f"Num Classes:       {args.num_classes}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Epochs:            {args.num_epochs}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Device:            {device}")
    print(f"Use CTC:           {args.use_ctc}")
    print(f"CTC Blank:         {args.ctc_blank}")
    print("="*60)
    print()
    print("Creating model...")
    model = create_model(args)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    print("Creating trainer...")
    if args.use_ctc:
        trainer = create_trainer_with_ctc(
            model=model,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            loss_weights=args.loss_weights,
            ctc_blank=args.ctc_blank,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            use_scheduler=args.use_scheduler,
            scheduler_type=args.scheduler_type,
            checkpoint_dir=args.checkpoint_dir,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_val=args.gradient_clip
        )
    else:
        trainer = create_trainer(
            model=model,
            num_classes=args.num_classes,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            loss_weights=args.loss_weights[:2],  # Only Focal and LabelSmoothing
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            use_scheduler=args.use_scheduler,
            scheduler_type=args.scheduler_type,
            checkpoint_dir=args.checkpoint_dir,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_val=args.gradient_clip
        )
    
    print("Trainer created")
    
    print("\nLoading datasets...")
    train_dataset = TextRecognizerDataset(args.data_dir, split='train', train_ratio=0.8)
    val_dataset = TextRecognizerDataset(args.data_dir, split='val', train_ratio=0.8)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    print("\nStarting training...")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        use_ctc=args.use_ctc
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs: {args.tensorboard_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

