import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, List
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

from infrastructure.ml.models.vit_ctc.model import ModelTextRecoginizer
from infrastructure.ml.models.vit_ctc.loss import CombinedLoss, FocalLoss, LabelSmoothingLoss, CTCLoss
from infrastructure.ml.models.vit_ctc.earlystopping import EarlyStopping
from infrastructure.ml.models.vit_ctc.metricstracker import MetricsTracker

class TrainerTextRecoginizer:
    def __init__(
            self,
            model: ModelTextRecoginizer,
            loss_fn: CombinedLoss,
            optimizer: torch.optim.Optimizer,
            early_topping: EarlyStopping,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            scheduler: Optional[Any] = None,
            checkpoint_dir: str = './checkpoints',
            use_tensorboard: bool = True,
            tensorboard_dir: str = './runs',
            gradient_clip_val: float = 1.0
        ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.gradient_clip_val = gradient_clip_val

        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        if use_tensorboard:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(tensorboard_dir, f'experiment_{timestamp}')
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
            print(f"Run: tensorboard --logdir={tensorboard_dir}")
        
        self.early_stopping = early_topping
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.current_epoch = 0
        
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
    
    def train_epoch(self, train_loader, epoch: int, use_ctc=False):
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', 
                   leave=False, dynamic_ncols=True)
        
        for batch_idx, batch_data in enumerate(pbar):
            if use_ctc:
                images, labels, target_lengths, rows, cols = batch_data
                target_lengths = target_lengths.to(self.device)
            else:
                images, labels, rows, cols = batch_data
                target_lengths = None
            
            labels = labels.to(self.device)
            outputs = self.model(images, rows, cols, return_sequence=use_ctc)
            
            if use_ctc:
                batch_size = len(images)
                if outputs.dim() == 3:
                    seq_len = outputs.size(1)
                    outputs = outputs.permute(1, 0, 2)
                else:
                    seq_len = 1
                    outputs = outputs.unsqueeze(0)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                loss = self.loss_fn(outputs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            else:
                loss = self.loss_fn(outputs, labels)
            
            loss.backward()

            if self.gradient_clip_val > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.gradient_clip_val
                )
            else:
                grad_norm = 0.0

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if not use_ctc:
                _, predicted = torch.max(outputs.data, 1)
                self.train_metrics.update(loss.item(), predicted, labels)
            else:
                self.train_metrics.update(loss.item(), None, None)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad_norm': f'{grad_norm:.2f}' if self.gradient_clip_val > 0 else 'N/A'
            })

            if self.writer and (batch_idx % 10 == 0):
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                if self.gradient_clip_val > 0:
                    self.writer.add_scalar('Train/GradNorm', grad_norm, global_step)
        
        metrics = self.train_metrics.compute()
        return metrics
    
    def validate(self, val_loader, epoch: int, use_ctc=False):
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', 
                   leave=False, dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_data in pbar:
                if use_ctc:
                    images, labels, target_lengths, rows, cols = batch_data
                    target_lengths = target_lengths.to(self.device)
                else:
                    images, labels, rows, cols = batch_data
                    target_lengths = None
                
                labels = labels.to(self.device)
                outputs = self.model(images, rows, cols, return_sequence=use_ctc)
                
                if use_ctc:
                    batch_size = len(images)
                    if outputs.dim() == 3:
                        seq_len = outputs.size(1)
                        outputs = outputs.permute(1, 0, 2)
                    else:
                        seq_len = 1
                        outputs = outputs.unsqueeze(0)
                    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                    loss = self.loss_fn(outputs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
                else:
                    loss = self.loss_fn(outputs, labels)
                
                if not use_ctc:
                    _, predicted = torch.max(outputs.data, 1)
                    self.val_metrics.update(loss.item(), predicted, labels)
                else:
                    self.val_metrics.update(loss.item(), None, None)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        metrics = self.val_metrics.compute()
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              save_every: int = 5, log_images: bool = False, use_ctc: bool = False):
        print("=" * 80)
        print(f"Starting Training on {self.device}")
        print("=" * 80)
        print(f"Total epochs: {num_epochs}")
        print(f"Early stopping patience: {self.early_stopping.patience}")
        print(f"Gradient clipping: {self.gradient_clip_val}")
        print(f"TensorBoard: {'Enabled' if self.use_tensorboard else 'Disabled'}")
        print(f"Use CTC: {use_ctc}")
        print("=" * 80)
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

                train_metrics = self.train_epoch(train_loader, epoch, use_ctc=use_ctc)
                self.train_losses.append(train_metrics['loss'])
                if not use_ctc:
                    self.train_accuracies.append(train_metrics['accuracy'])
                
                val_metrics = self.validate(val_loader, epoch, use_ctc=use_ctc)
                self.val_losses.append(val_metrics['loss'])
                if not use_ctc:
                    self.val_accuracies.append(val_metrics['accuracy'])
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                print(f"\nResults:")
                if not use_ctc:
                    print(f"  Train Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.2f}%")
                    print(f"  Val Loss:   {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.2f}%")
                else:
                    print(f"  Train Loss: {train_metrics['loss']:.4f}")
                    print(f"  Val Loss:   {val_metrics['loss']:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")

                if self.writer:
                    self.writer.add_scalars('Loss', {
                        'train': train_metrics['loss'],
                        'val': val_metrics['loss']
                    }, epoch)
                    if not use_ctc:
                        self.writer.add_scalars('Accuracy', {
                            'train': train_metrics['accuracy'],
                            'val': val_metrics['accuracy']
                        }, epoch)
                    self.writer.add_scalar('LearningRate', current_lr, epoch)

                    if 'per_class_accuracy' in val_metrics:
                        for cls, acc in val_metrics['per_class_accuracy'].items():
                            self.writer.add_scalar(f'PerClassAccuracy/class_{cls}', acc, epoch)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                        if current_lr != self.optimizer.param_groups[0]['lr']:
                            print(f"  ⚡ Learning rate reduced to {self.optimizer.param_groups[0]['lr']:.6f}")
                    else:
                        self.scheduler.step()
                
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch + 1, val_metrics)
                
                improved = False
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    improved = True
                
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    improved = True
                
                if improved:
                    self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
                    print(f"  New best model saved! (loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.2f}%)")
                
                if self.early_stopping(val_metrics['loss'], epoch):
                    print(f"\n Early stopping triggered!")
                    print(f"  Best epoch: {self.early_stopping.best_epoch + 1}")
                    print(f"  Best val loss: {self.early_stopping.best_score:.4f}")
                    break
                else:
                    if self.early_stopping.counter > 0:
                        print(f"  Early stopping counter: {self.early_stopping.counter}/{self.early_stopping.patience}")
                
                self.current_epoch = epoch + 1
            
            print("\n" + "=" * 80)
            print("Training Completed!")
            print("=" * 80)
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
            print(f"Total epochs trained: {self.current_epoch}")

            self._save_training_history()
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving current state...")
            self.save_checkpoint(self.current_epoch, {
                'loss': self.val_losses[-1] if self.val_losses else float('inf'),
                'accuracy': self.val_accuracies[-1] if self.val_accuracies else 0.0
            }, is_best=False)
            print("State saved successfully!")
        
        finally:
            if self.writer:
                self.writer.close()
                print("TensorBoard writer closed.")
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with complete training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'early_stopping_counter': self.early_stopping.counter,
            'early_stopping_best_score': self.early_stopping.best_score
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            print(f"  Saving best model to {path}")
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            print(f"  Saving checkpoint to {path}")
        
        torch.save(checkpoint, path)
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            self.current_epoch = checkpoint['epoch']
            
            if 'early_stopping_counter' in checkpoint:
                self.early_stopping.counter = checkpoint['early_stopping_counter']
                self.early_stopping.best_score = checkpoint['early_stopping_best_score']
            
            print(f"Full training state restored")
            print(f"   Resuming from epoch {checkpoint['epoch']}")
            print(f"   Best val loss: {self.best_val_loss:.4f}")
            print(f"   Best val accuracy: {self.best_val_accuracy:.2f}%")
        else:
            print(f"Model weights loaded (inference mode)")
        
        return checkpoint['epoch']
    
    def _save_training_history(self):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'total_epochs': self.current_epoch
        }
        
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")
    
    def get_model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "=" * 80)
        print("Model Summary")
        print("=" * 80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("=" * 80)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }