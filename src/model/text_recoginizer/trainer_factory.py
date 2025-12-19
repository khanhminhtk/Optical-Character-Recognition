import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.model.text_recoginizer.trainer import TrainerTextRecoginizer
from src.model.text_recoginizer.model import ModelTextRecoginizer
from src.model.text_recoginizer.loss import CombinedLoss, FocalLoss, LabelSmoothingLoss, CTCLoss
from src.model.text_recoginizer.earlystopping import EarlyStopping


def create_trainer(
    model: ModelTextRecoginizer,
    num_classes: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    loss_weights: list = [0.7, 0.3],
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    use_scheduler: bool = True,
    scheduler_type: str = 'plateau',
    checkpoint_dir: str = './checkpoints',
    use_tensorboard: bool = True,
    tensorboard_dir: str = './runs',
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 1.0
):
    print("\n" + "=" * 80)
    print("Creating Professional Trainer")
    print("=" * 80)
    
    focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    label_smooth_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing)
    combined_loss = CombinedLoss(
        losses=[focal_loss, label_smooth_loss],
        weights=loss_weights
    )
    print(f"Loss: Combined (Focal: {loss_weights[0]}, LabelSmoothing: {loss_weights[1]})")
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    
    scheduler = None
    if use_scheduler:
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=False,
                min_lr=1e-7
            )
            print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=1e-7
            )
            print(f"Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-7)")
        else:
            print(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
    else:
        print("Scheduler: None")
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
        mode='min'
    )
    
    trainer = TrainerTextRecoginizer(
        model=model,
        loss_fn=combined_loss,
        optimizer=optimizer,
        early_topping=early_stopping,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        use_tensorboard=use_tensorboard,
        tensorboard_dir=tensorboard_dir,
        gradient_clip_val=gradient_clip_val
    )
    
    print("=" * 80)
    trainer.get_model_summary()
    
    return trainer


def create_trainer_with_ctc(
    model: ModelTextRecoginizer,
    num_classes: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    loss_weights: list = [0.5, 0.3, 0.2],
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    ctc_blank: int = 0,
    use_scheduler: bool = True,
    scheduler_type: str = 'plateau',
    checkpoint_dir: str = './checkpoints',
    use_tensorboard: bool = True,
    tensorboard_dir: str = './runs',
    early_stopping_patience: int = 10,
    gradient_clip_val: float = 1.0
):
    print("\n" + "=" * 80)
    print("Creating Professional Trainer with CTC Loss")
    print("=" * 80)
    
    ctc_loss = CTCLoss(blank=ctc_blank, reduction='mean', zero_infinity=True)
    focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    label_smooth_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing)
    
    combined_loss = CombinedLoss(
        losses=[ctc_loss, focal_loss, label_smooth_loss],
        weights=loss_weights,
        apply_softmax=False
    )
    print(f"Loss: Combined (CTC: {loss_weights[0]}, Focal: {loss_weights[1]}, LabelSmoothing: {loss_weights[2]})")
    print(f"  CTC blank index: {ctc_blank}")
    print(f"  Note: Model outputs should be logits (no softmax)")
    print(f"  Note: Dataloader should provide input_lengths and target_lengths for CTC")
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    
    scheduler = None
    if use_scheduler:
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=False,
                min_lr=1e-7
            )
            print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=1e-7
            )
            print(f"Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-7)")
        else:
            print(f"Unknown scheduler type: {scheduler_type}")
    else:
        print("Scheduler: None")
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
        mode='min'
    )
    
    trainer = TrainerTextRecoginizer(
        model=model,
        loss_fn=combined_loss,
        optimizer=optimizer,
        early_topping=early_stopping,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        use_tensorboard=use_tensorboard,
        tensorboard_dir=tensorboard_dir,
        gradient_clip_val=gradient_clip_val
    )
    
    print("=" * 80)
    trainer.get_model_summary()
    
    return trainer
