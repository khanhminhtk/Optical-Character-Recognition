import sys
import os
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

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
    num_patches = (args.img_height // args.patch_size) * (args.img_width // args.patch_size)
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = Backbone(model=mobilenet)
    patch_embedding = PatchEmbedding(
        backbone=backbone,
        num_patches=num_patches,
        rows=args.img_height // args.patch_size,
        cols=args.img_width // args.patch_size
    )
    backbone_output_dim = 576
    pos_embedding = PosEmbedding(
        dim=backbone_output_dim,
        num_patches=num_patches
    )
    patch_embed = PatchEmbedWithPos(
        patch_embedding=patch_embedding,
        pos_embedding=pos_embedding
    )
    transformer_blocks = [
        TransformerBlock(
            n_block=1,
            nhead=args.num_heads,
            dim=backbone_output_dim,
            drop_out=args.dropout
        )
        for _ in range(args.num_layers)
    ]
    
    model = ModelTextRecoginizer(
        patch_embed=patch_embed,
        transformer_blocks=transformer_blocks,
        num_classes=args.num_classes,
        embedding_dim=backbone_output_dim, 
        hidden_dim=args.hidden_classifier,
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
    print("✓ Creating model...")
    model = create_model(args)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    print("✓ Creating trainer...")
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
    
    print("✓ Trainer created")
    print("\n⚠ Warning: Data loading not implemented!")
    print("Please implement your dataset and data loaders in this script.")
    print("\nExample:")
    print("  from your_module import YourDataset")
    print("  train_dataset = YourDataset(args.data_dir, split='train')")
    print("  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, ...)")
    print("  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, ...)")
    print("\nThen call:")
    print("  trainer.train(train_loader, val_loader, num_epochs=args.num_epochs)")
    
    # print("\n✓ Starting training...")
    # trainer.train(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=args.num_epochs,
    #     save_every=args.save_every
    # )
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs: {args.tensorboard_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

