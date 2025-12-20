"""
Text Detection Inference Script
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.yolo_trainer import YOLOTrainer
from utils.visualize import DetectionVisualizer
from config.config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run text detection inference")
    
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Image path or directory")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--save", action="store_true",
                        help="Save results")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize trainer with default config
    config = TrainConfig()
    trainer = YOLOTrainer(config)
    trainer.load_model(weights_path=args.weights)
    
    # Initialize visualizer
    visualizer = DetectionVisualizer()
    
    # Get image paths
    source = Path(args.source)
    if source.is_dir():
        image_paths = list(source.glob("*.jpg")) + list(source.glob("*.png"))
    else:
        image_paths = [source]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Create save directory
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Process each image
    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        
        results = trainer.predict(str(img_path), conf=args.conf)
        
        save_path = None
        if args.save:
            save_path = os.path.join(args.save_dir, f"det_{img_path.name}")
        
        visualizer.visualize_yolo_results(
            str(img_path),
            results,
            conf_threshold=args.conf,
            show=not args.no_show,
            save_path=save_path
        )
    
    print("Done!")


if __name__ == "__main__":
    main()
