#!/bin/bash

# CTC Text Recognition Training Script
# Usage: ./scripts/training/train_ctc.sh [options]

set -e  # Exit on error

# Default configuration
DATA_DIR="./data_test/text_recognizer_data/"
CHECKPOINT_DIR="./checkpoints_ctc"
TENSORBOARD_DIR="./runs_ctc"
IMG_HEIGHT=224
IMG_WIDTH=224
PATCH_SIZE=8
EMBEDDING_DIM=256
NUM_HEADS=8
NUM_LAYERS=6
MLP_DIM=512
HIDDEN_CLASSIFIER=512
NUM_CLASSES=27
DROPOUT=0.1
BATCH_SIZE=4
NUM_EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-5
GRADIENT_CLIP=5.0
LOSS_WEIGHTS="0.5 0.3 0.2"
FOCAL_ALPHA=1.0
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1
USE_CTC=true
CTC_BLANK=26
USE_SCHEDULER=true
SCHEDULER_TYPE="plateau"
EARLY_STOPPING_PATIENCE=15
USE_TENSORBOARD=true
SAVE_EVERY=5
NUM_WORKERS=4
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --help)
            echo "CTC Text Recognition Training Script"
            echo ""
            echo "Usage: ./scripts/training/train_ctc.sh [options]"
            echo ""
            echo "Options:"
            echo "  --data_dir DIR         Data directory (default: ./data_test/text_recognizer_data/)"
            echo "  --checkpoint_dir DIR   Checkpoint directory (default: ./checkpoints_ctc)"
            echo "  --batch_size N         Batch size (default: 4)"
            echo "  --num_epochs N         Number of epochs (default: 100)"
            echo "  --lr RATE              Learning rate (default: 1e-4)"
            echo "  --device DEVICE        Device (default: cuda)"
            echo "  --cpu                  Use CPU instead of GPU"
            echo "  --help                 Show this help message"
            echo ""
            echo "Example:"
            echo "  ./scripts/training/train_ctc.sh --batch_size 8 --num_epochs 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "CTC Training Configuration"
echo "========================================="
echo "Data Dir:          $DATA_DIR"
echo "Checkpoint Dir:    $CHECKPOINT_DIR"
echo "Tensorboard Dir:   $TENSORBOARD_DIR"
echo "Image Size:        ${IMG_HEIGHT}x${IMG_WIDTH}"
echo "Batch Size:        $BATCH_SIZE"
echo "Epochs:            $NUM_EPOCHS"
echo "Learning Rate:     $LR"
echo "Device:            $DEVICE"
echo "Use CTC:           $USE_CTC"
echo "========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_ctc.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT"
    echo "Falling back to direct path..."
    TRAIN_SCRIPT="src/model/train.py"
fi

echo "Starting CTC training..."
echo ""

python "$TRAIN_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --tensorboard_dir "$TENSORBOARD_DIR" \
    --img_height $IMG_HEIGHT \
    --img_width $IMG_WIDTH \
    --patch_size $PATCH_SIZE \
    --embedding_dim $EMBEDDING_DIM \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --mlp_dim $MLP_DIM \
    --hidden_classifier $HIDDEN_CLASSIFIER \
    --num_classes $NUM_CLASSES \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --gradient_clip $GRADIENT_CLIP \
    --loss_weights $LOSS_WEIGHTS \
    --focal_alpha $FOCAL_ALPHA \
    --focal_gamma $FOCAL_GAMMA \
    --label_smoothing $LABEL_SMOOTHING \
    $([ "$USE_CTC" = true ] && echo "--use_ctc") \
    --ctc_blank $CTC_BLANK \
    $([ "$USE_SCHEDULER" = true ] && echo "--use_scheduler") \
    --scheduler_type $SCHEDULER_TYPE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    $([ "$USE_TENSORBOARD" = true ] && echo "--use_tensorboard") \
    --save_every $SAVE_EVERY \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Tensorboard: $TENSORBOARD_DIR"
echo "========================================="
