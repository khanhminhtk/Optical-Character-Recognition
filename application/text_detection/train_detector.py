"""
Text Detection Training Script
"""
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config, DataConfig, TrainConfig
from data.data_processor import YOLODataProcessor
from trainer.yolo_trainer import YOLOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO for text detection")
    
    # Config file (primary way)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Data args (can override config)
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for processed data")
    parser.add_argument("--xml-file", type=str, default=None,
                        help="XML annotation file name")
    
    # Training args (can override config)
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO model name (yolov8n, yolov8s, yolov8m, etc.)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Input image size")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, or device id)")
    
    # Project args
    parser.add_argument("--project", type=str, default=None,
                        help="Project directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")
    
    # Flags
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data preparation if already done")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config from YAML or use defaults
    if args.config:
        print(f"Loading config from: {args.config}")
        config = Config.from_yaml(args.config)
        # Merge with command line args (args override YAML)
        config.merge_with_args(args)
    else:
        # Create config from command line args only
        if not args.raw_data or not args.output_dir:
            print("Error: Either --config or both --raw-data and --output-dir are required")
            sys.exit(1)
            
        data_config = DataConfig(
            raw_data_path=args.raw_data,
            xml_file=args.xml_file or "words.xml",
            output_dir=args.output_dir,
        )
        
        train_config = TrainConfig(
            model_name=args.model or "yolov8s",
            epochs=args.epochs or 200,
            imgsz=args.imgsz or 1024,
            batch_size=args.batch_size or 16,
            device=args.device,
            project=args.project or "models",
            name=args.name or "yolov8_text_detect",
            weights=args.weights,
        )
        
        config = Config(data=data_config, train=train_config)
    
    # Print config summary
    print("\n" + "=" * 50)
    print("Configuration:")
    print("=" * 50)
    print(f"  Data path: {config.data.raw_data_path}")
    print(f"  Output dir: {config.data.output_dir}")
    print(f"  Model: {config.train.model_name}")
    print(f"  Epochs: {config.train.epochs}")
    print(f"  Image size: {config.train.imgsz}")
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  Device: {config.train.device or 'auto'}")
    print("=" * 50 + "\n")
    
    # Step 1: Prepare data
    if not args.skip_data_prep:
        print("=" * 50)
        print("Step 1: Preparing dataset...")
        print("=" * 50)
        
        data_processor = YOLODataProcessor(config.data)
        data_yaml_path = data_processor.prepare()
        
        print("\nDataset summary:")
        for key, value in data_processor.summary.items():
            print(f"  {key}: {value}")
    else:
        data_yaml_path = os.path.join(config.data.output_dir, 'data.yaml')
        print(f"Skipping data prep. Using existing: {data_yaml_path}")
    
    # Step 2: Train model
    print("\n" + "=" * 50)
    print("Step 2: Training model...")
    print("=" * 50)
    
    trainer = YOLOTrainer(config.train)
    trainer.load_model(weights_path=config.train.weights)
    trainer.train(data_yaml=data_yaml_path)
    
    # Step 3: Validate
    print("\n" + "=" * 50)
    print("Step 3: Validating model...")
    print("=" * 50)
    
    metrics = trainer.validate()
    print("\nValidation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nBest weights saved at: {trainer.best_weights_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()
