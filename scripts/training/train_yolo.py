#!/usr/bin/env python3
"""
YOLO Text Detection Training Script
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.ml.models.yolo.config.trainconfig import TrainConfigs
from infrastructure.ml.models.yolo.trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLO text detection model')
    
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., yolo11n)')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Use pretrained weights (True/False)')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0, 1, cpu, etc.)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer name')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--augment', type=str, default='True',
                        help='Use data augmentation (True/False)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--save_period', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Convert string booleans
    pretrained = args.pretrained.lower() == 'true'
    augment = args.augment.lower() == 'true'
    
    # Create training configuration
    config = TrainConfigs(
        model_name=args.model_name,
        pretrained=pretrained,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        augment=augment,
        project=args.project,
        name=args.name,
        save_period=args.save_period
    )
    
    print("\n" + "="*60)
    print("YOLO Training Configuration")
    print("="*60)
    print(f"Model:          {config.model_name}")
    print(f"Pretrained:     {config.pretrained}")
    print(f"Epochs:         {config.epochs}")
    print(f"Image Size:     {config.imgsz}")
    print(f"Batch Size:     {config.batch_size}")
    print(f"Device:         {config.device}")
    print(f"Workers:        {config.workers}")
    print(f"Optimizer:      {config.optimizer}")
    print(f"Learning Rate:  {config.lr0}")
    print(f"Weight Decay:   {config.weight_decay}")
    print(f"Patience:       {config.patience}")
    print(f"Augmentation:   {config.augment}")
    print(f"Project:        {config.project}")
    print(f"Name:           {config.name}")
    print(f"Data YAML:      {args.data_yaml}")
    print("="*60)
    
    # Initialize trainer
    print(f"\n✓ Initializing YOLO Trainer...")
    trainer = Trainer(config)
    
    # Load model
    print(f"✓ Loading model: {args.model_name}")
    trainer.load_model(weights_path=None)
    
    # Train
    print(f"\n✓ Starting training on {args.data_yaml}")
    print(f"✓ Training will save to: {args.project}/{args.name}")
    print()
    
    trainer.train(data_yaml=args.data_yaml)
    
    # Training completed
    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)
    print(f"Results saved to: {args.project}/{args.name}")
    print(f"Best weights:     {args.project}/{args.name}/weights/best.pt")
    print(f"Last weights:     {args.project}/{args.name}/weights/last.pt")
    print("="*60)


if __name__ == "__main__":
    main()
