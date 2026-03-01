import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def plot_training_history(checkpoint_dir='./checkpoints', save_dir='./plots'):
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"Training history not found at: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['train_losses']) + 1))
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accs = history['train_accuracies']
    val_accs = history['val_accuracies']
    lrs = history['learning_rates']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, label='Val Loss', marker='s', linewidth=2, markersize=4)
    ax1.axhline(y=history['best_val_loss'], color='r', linestyle='--', 
                label=f'Best Val Loss: {history["best_val_loss"]:.4f}', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_accs, label='Train Accuracy', marker='o', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accs, label='Val Accuracy', marker='s', linewidth=2, markersize=4)
    ax2.axhline(y=history['best_val_accuracy'], color='r', linestyle='--',
                label=f'Best Val Acc: {history["best_val_accuracy"]:.2f}%', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(epochs, lrs, label='Learning Rate', marker='o', linewidth=2, 
             markersize=4, color='green')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    gap = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    ax4.plot(epochs, gap, label='Train-Val Loss Gap', marker='o', 
             linewidth=2, markersize=4, color='orange')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Gap', fontsize=12)
    ax4.set_title('Overfitting Indicator (Train-Val Gap)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total Epochs: {history['total_epochs']}")
    print(f"Best Validation Loss: {history['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Val Accuracy: {val_accs[-1]:.2f}%")
    print(f"Final Learning Rate: {lrs[-1]:.6f}")
    print("=" * 80)


def plot_comparison(checkpoint_dirs, labels, save_dir='./plots'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(range(len(checkpoint_dirs)))
    
    for idx, (checkpoint_dir, label) in enumerate(zip(checkpoint_dirs, labels)):
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        
        if not os.path.exists(history_path):
            print(f"Skipping {label}: history not found")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = list(range(1, len(history['val_losses']) + 1))
        axes[0].plot(epochs, history['val_losses'], label=label, 
                    linewidth=2, color=colors[idx])
        
        axes[1].plot(epochs, history['val_accuracies'], label=label,
                    linewidth=2, color=colors[idx])
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'experiment_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory containing training_history.json')
    parser.add_argument('--save_dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='List of checkpoint directories to compare')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Labels for comparison experiments')
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.labels or len(args.labels) != len(args.compare):
            print("Please provide labels for all experiments using --labels")
        else:
            plot_comparison(args.compare, args.labels, args.save_dir)
    else:
        plot_training_history(args.checkpoint_dir, args.save_dir)
