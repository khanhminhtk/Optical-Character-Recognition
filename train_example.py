"""
Professional Training Example for Text Recognizer
================================================

This example demonstrates how to use the professional trainer with:
- TQDM progress bars
- TensorBoard logging
- Early stopping
- Combined loss (Focal + Label Smoothing)
- Learning rate scheduling
- Gradient clipping
- Automatic checkpointing
"""

import torch
from torch.utils.data import DataLoader

# Import your model and dataset
from src.model.text_recoginizer.model import ModelTextRecoginizer
from src.model.text_recoginizer.trainer_factory import create_trainer
# from your_dataset import YourDataset  # Replace with your actual dataset

def main():
    # ====================================================================
    # 1. Setup Configuration
    # ====================================================================
    config = {
        'num_classes': 26,  # Number of character classes
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Loss configuration
        'loss_weights': [0.7, 0.3],  # [focal_weight, label_smoothing_weight]
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        
        # Scheduler configuration
        'use_scheduler': True,
        'scheduler_type': 'plateau',  # 'plateau' or 'cosine'
        
        # Training configuration
        'early_stopping_patience': 10,
        'gradient_clip_val': 1.0,
        'save_every': 5,
        
        # Directories
        'checkpoint_dir': './checkpoints',
        'tensorboard_dir': './runs',
        'use_tensorboard': True,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ====================================================================
    # 2. Create Model
    # ====================================================================
    # TODO: Replace with your actual model initialization
    # model = ModelTextRecoginizer(
    #     patchembedwithpos=your_patch_embed,
    #     transformer=your_transformer,
    #     embedding_dim=512,
    #     num_classes=config['num_classes'],
    #     dropout=0.2,
    #     hidden_classifier=512
    # )
    
    # ====================================================================
    # 3. Create DataLoaders
    # ====================================================================
    # TODO: Replace with your actual dataset
    # train_dataset = YourDataset(split='train')
    # val_dataset = YourDataset(split='val')
    # 
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )
    # 
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    # ====================================================================
    # 4. Create Professional Trainer
    # ====================================================================
    trainer = create_trainer(
        model=model,
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        loss_weights=config['loss_weights'],
        focal_alpha=config['focal_alpha'],
        focal_gamma=config['focal_gamma'],
        label_smoothing=config['label_smoothing'],
        use_scheduler=config['use_scheduler'],
        scheduler_type=config['scheduler_type'],
        checkpoint_dir=config['checkpoint_dir'],
        use_tensorboard=config['use_tensorboard'],
        tensorboard_dir=config['tensorboard_dir'],
        early_stopping_patience=config['early_stopping_patience'],
        gradient_clip_val=config['gradient_clip_val']
    )
    
    # ====================================================================
    # 5. Start Training
    # ====================================================================
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        save_every=config['save_every']
    )
    
    # ====================================================================
    # 6. View Results in TensorBoard
    # ====================================================================
    print("\n" + "=" * 80)
    print("📊 To view training progress in TensorBoard, run:")
    print(f"   tensorboard --logdir={config['tensorboard_dir']}")
    print("   Then open: http://localhost:6006")
    print("=" * 80)


def resume_training_example():
    """Example: Resume training from checkpoint"""
    
    # Create model and trainer (same as above)
    # ...
    
    # Load checkpoint and resume
    trainer.load_checkpoint(
        checkpoint_path='./checkpoints/latest_checkpoint.pth',
        resume_training=True
    )
    
    # Continue training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,  # Will continue from where it stopped
        save_every=5
    )


def inference_example():
    """Example: Load model for inference only"""
    
    # Create model and trainer
    # ...
    
    # Load only model weights (not training state)
    trainer.load_checkpoint(
        checkpoint_path='./checkpoints/best_model.pth',
        resume_training=False
    )
    
    # Use model for inference
    trainer.model.eval()
    with torch.no_grad():
        # Your inference code here
        pass


def train_with_ctc_example():
    """Example: Train with CTC loss for sequence recognition"""
    
    from src.model.text_recoginizer.trainer_factory import create_trainer_with_ctc
    
    config = {
        'num_classes': 27,  # 26 characters + 1 blank for CTC
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 50,
        
        # CTC-specific configuration
        'loss_weights': [0.5, 0.3, 0.2],  # [CTC, Focal, LabelSmoothing]
        'ctc_blank': 0,  # Index for blank label
        
        'checkpoint_dir': './checkpoints_ctc',
        'use_tensorboard': True,
    }
    
    # Create trainer with CTC support
    trainer = create_trainer_with_ctc(
        model=model,
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate'],
        loss_weights=config['loss_weights'],
        ctc_blank=config['ctc_blank'],
        checkpoint_dir=config['checkpoint_dir'],
        use_tensorboard=config['use_tensorboard']
    )
    
    # Important: Your DataLoader should provide input_lengths and target_lengths
    # Example batch format: (images, labels, rows, cols, input_lengths, target_lengths)
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs']
    )
    
    print("\n💡 Tips for CTC training:")
    print("  1. Model outputs should be logits (no softmax)")
    print("  2. DataLoader must provide input_lengths and target_lengths")
    print("  3. CTC blank index is typically 0 or num_classes-1")
    print("  4. Target sequences should not include blank labels")


if __name__ == '__main__':
    main()
    
    # Uncomment to test resume training
    # resume_training_example()
    
    # Uncomment to test inference
    # inference_example()
    
    # Uncomment to test CTC training
    # train_with_ctc_example()
