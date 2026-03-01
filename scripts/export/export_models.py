#!/usr/bin/env python3
"""
Export PyTorch models to ONNX format.

Usage:
    # Export all models
    python scripts/export/export_models.py
    
    # Export only YOLO
    python scripts/export/export_models.py --mode yolo
    
    # Export with custom paths
    python scripts/export/export_models.py --yolo-pt path/to/yolo.pt --yolo-onnx output/yolo.onnx
    
    # Custom input size
    python scripts/export/export_models.py --yolo-input-size 416 --text-input-size 32
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.ml.exporters.export_to_onnx import (
    export_yolo_onnx,
    export_text_recognizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX')
    parser.add_argument(
        '--mode',
        choices=['all', 'yolo', 'text'],
        default='all',
        help='Which models to export'
    )
    parser.add_argument(
        '--yolo-pt',
        type=str,
        default='model_checkpoint/pt/yolo/detect/train3/weights/best.pt',
        help='Path to YOLO PyTorch model'
    )
    parser.add_argument(
        '--text-pt',
        type=str,
        default='model_checkpoint/pt/text_recognizer/checkpoint_ctc_epoch_30.pth',
        help='Path to Text Recognizer PyTorch model'
    )
    parser.add_argument(
        '--yolo-onnx',
        type=str,
        default='model_checkpoint/onnx/yolo/yolo_detector.onnx',
        help='Output path for YOLO ONNX model'
    )
    parser.add_argument(
        '--text-onnx',
        type=str,
        default='model_checkpoint/onnx/text_recognizer/text_recognizer.onnx',
        help='Output path for Text Recognizer ONNX model'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=11,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        default=True,
        help='Enable dynamic batch size'
    )
    parser.add_argument(
        '--yolo-input-size',
        type=int,
        default=640,
        help='YOLO input size (height and width)'
    )
    parser.add_argument(
        '--text-input-size',
        type=int,
        default=224,
        help='Text recognizer input size (height and width)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    yolo_pt = PROJECT_ROOT / args.yolo_pt
    text_pt = PROJECT_ROOT / args.text_pt
    yolo_onnx = PROJECT_ROOT / args.yolo_onnx
    text_onnx = PROJECT_ROOT / args.text_onnx
    
    logger.info("=" * 60)
    logger.info("PyTorch to ONNX Export")
    logger.info("=" * 60)
    logger.info(f"Export mode: {args.mode}")
    logger.info(f"Opset version: {args.opset_version}")
    logger.info(f"Dynamic batch: {args.dynamic_batch}")
    
    success = True
    
    # Export YOLO
    if args.mode in ['all', 'yolo']:
        logger.info("\n" + "=" * 60)
        logger.info("Exporting YOLO Detector")
        logger.info("=" * 60)
        try:
            export_yolo_onnx(
                path_pt_yolo=str(yolo_pt),
                path_onnx_yolo=str(yolo_onnx),
                input_shape=(1, 3, args.yolo_input_size, args.yolo_input_size),
                opset_version=args.opset_version,
                dynamic_batch=args.dynamic_batch
            )
            logger.info("✓ YOLO export completed successfully")
        except Exception as e:
            logger.error(f"✗ YOLO export failed: {e}")
            success = False
    
    # Export Text Recognizer
    if args.mode in ['all', 'text']:
        logger.info("\n" + "=" * 60)
        logger.info("Exporting Text Recognizer")
        logger.info("=" * 60)
        try:
            export_text_recognizer(
                path_pt_text_recognizer=str(text_pt),
                path_onnx_text_recognizer=str(text_onnx),
                input_shape=(1, 3, args.text_input_size, args.text_input_size),
                opset_version=args.opset_version,
                dynamic_batch=args.dynamic_batch
            )
            logger.info("✓ Text Recognizer export completed successfully")
        except Exception as e:
            logger.error(f"✗ Text Recognizer export failed: {e}")
            success = False
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ All exports completed successfully")
        logger.info(f"ONNX models saved to: {PROJECT_ROOT / 'model_checkpoint/onnx'}")
    else:
        logger.error("✗ Some exports failed")
        sys.exit(1)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
