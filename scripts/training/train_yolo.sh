set -e 

MODEL_NAME="yolo11n"
PRETRAINED=true
DATA_YAML="data_test/yolo_data/data.yml"
EPOCHS=50
IMGSZ=640
BATCH_SIZE=4
DEVICE="0"
WORKERS=8
OPTIMIZER="AdamW"
LR0=0.001
WEIGHT_DECAY=0.0005
PATIENCE=10
AUGMENT=true
PROJECT="model_checkpoint/pt/yolo"
NAME="yolo_text_detection"
SAVE_PERIOD=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data)
            DATA_YAML="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --lr)
            LR0="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --help)
            echo "YOLO Text Detection Training Script"
            echo ""
            echo "Usage: ./scripts/training/train_yolo.sh [options]"
            echo ""
            echo "Options:"
            echo "  --model NAME       Model name (default: yolo11n)"
            echo "  --data PATH        Path to data.yaml (default: data/data.yaml)"
            echo "  --epochs N         Number of epochs (default: 100)"
            echo "  --batch N          Batch size (default: 16)"
            echo "  --imgsz N          Image size (default: 1024)"
            echo "  --device DEVICE    Device (default: 0, use 'cpu' for CPU)"
            echo "  --lr LR            Learning rate (default: 0.001)"
            echo "  --name NAME        Experiment name (default: yolo_text_detection)"
            echo "  --project PATH     Project directory (default: runs/detect)"
            echo "  --cpu              Use CPU instead of GPU"
            echo "  --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  ./scripts/training/train_yolo.sh --epochs 200 --batch 32 --name exp1"
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
echo "YOLO Training Configuration"
echo "========================================="
echo "Model:          $MODEL_NAME"
echo "Pretrained:     $PRETRAINED"
echo "Data YAML:      $DATA_YAML"
echo "Epochs:         $EPOCHS"
echo "Image Size:     $IMGSZ"
echo "Batch Size:     $BATCH_SIZE"
echo "Device:         $DEVICE"
echo "Workers:        $WORKERS"
echo "Optimizer:      $OPTIMIZER"
echo "Learning Rate:  $LR0"
echo "Weight Decay:   $WEIGHT_DECAY"
echo "Patience:       $PATIENCE"
echo "Augmentation:   $AUGMENT"
echo "Project:        $PROJECT"
echo "Name:           $NAME"
echo "Save Period:    $SAVE_PERIOD"
echo "========================================="
echo ""

if [ ! -f "$DATA_YAML" ]; then
    echo "Error: Data YAML file not found: $DATA_YAML"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_yolo.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

echo "Starting training..."
echo ""

python "$TRAIN_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --pretrained "$PRETRAINED" \
    --data_yaml "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --optimizer "$OPTIMIZER" \
    --lr0 "$LR0" \
    --weight_decay "$WEIGHT_DECAY" \
    --patience "$PATIENCE" \
    --augment "$AUGMENT" \
    --project "$PROJECT" \
    --name "$NAME" \
    --save_period "$SAVE_PERIOD"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Results: $PROJECT/$NAME"
echo "Best weights: $PROJECT/$NAME/weights/best.pt"
echo "Last weights: $PROJECT/$NAME/weights/last.pt"
echo "========================================="
