import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import sys

from text_recoginizer.model import ModelTextRecoginizer
from text_recoginizer.pos_patch_embed import PatchEmbedWithPos
from text_recoginizer.transformer import TransformerBlock
from text_recoginizer.trainer_factory import create_trainer, create_trainer_with_ctc


class TextRecognizerDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.data = []
        self._load_data()
    
    def _load_data(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        label = sample['label']
        rows = sample.get('rows', 1)
        cols = sample.get('cols', 1)
        return image, label, rows, cols


def collate_fn(batch):
    images, labels, rows, cols = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    return images, labels, rows, cols


def create_model(config):
    patch_embed = PatchEmbedWithPos(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        in_channels=config['in_channels'],
        embedding_dim=config['embedding_dim']
    )
    
    transformer = TransformerBlock(
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout']
    )
    
    model = ModelTextRecoginizer(
        patchembedwithpos=patch_embed,
        transformer=transformer,
        embedding_dim=config['embedding_dim'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        hidden_classifier=config['hidden_classifier']
    )
    
    return model


def train(args):
    print("=" * 80)
    print("Text Recognizer Training")
    print("=" * 80)
    
    config = {
        'image_size': (args.img_height, args.img_width),
        'patch_size': args.patch_size,
        'in_channels': 3,
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'mlp_dim': args.mlp_dim,
        'dropout': args.dropout,
        'num_classes': args.num_classes,
        'hidden_classifier': args.hidden_classifier,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'device': args.device
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    train_dataset = TextRecognizerDataset(args.data_dir, split='train')
    val_dataset = TextRecognizerDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    model = create_model(config)
    
    if args.use_ctc:
        trainer = create_trainer_with_ctc(
            model=model,
            num_classes=config['num_classes'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=str(device),
            loss_weights=args.loss_weights,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            ctc_blank=args.ctc_blank,
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
            num_classes=config['num_classes'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=str(device),
            loss_weights=args.loss_weights[:2],
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
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume, resume_training=True)
        else:
            print(f"\nWarning: Checkpoint not found at {args.resume}. Starting from scratch.")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        save_every=args.save_every
    )
    
    print("\n" + "=" * 80)
    print("Training finished!")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    if args.use_tensorboard:
        print(f"TensorBoard logs: {args.tensorboard_dir}")
        print(f"Run: tensorboard --logdir={args.tensorboard_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train Text Recognizer')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs',
                       help='Directory for TensorBoard logs')
    
    parser.add_argument('--img_height', type=int, default=32,
                       help='Input image height')
    parser.add_argument('--img_width', type=int, default=128,
                       help='Input image width')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='Patch size for patch embedding')
    
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--mlp_dim', type=int, default=512,
                       help='MLP hidden dimension')
    parser.add_argument('--hidden_classifier', type=int, default=512,
                       help='Hidden dimension for classifier')
    parser.add_argument('--num_classes', type=int, default=26,
                       help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    parser.add_argument('--loss_weights', type=float, nargs='+', 
                       default=[0.7, 0.3],
                       help='Loss weights [focal, label_smoothing] or [ctc, focal, label_smoothing]')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    
    parser.add_argument('--use_ctc', action='store_true',
                       help='Use CTC loss')
    parser.add_argument('--ctc_blank', type=int, default=0,
                       help='CTC blank label index')
    
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                       help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='Type of scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                       help='Use TensorBoard logging')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.use_tensorboard:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    train(args)


if __name__ == '__main__':
    main()
