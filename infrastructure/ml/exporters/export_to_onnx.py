import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def export_yolo_onnx(
    path_pt_yolo: str,
    path_onnx_yolo: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    opset_version: int = 11,
    dynamic_batch: bool = True
) -> None:
    pt_path = Path(path_pt_yolo)
    if not pt_path.exists():
        raise FileNotFoundError(f"Model file not found: {path_pt_yolo}")
    
    onnx_path = Path(path_onnx_yolo)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Loading YOLO model from {path_pt_yolo}")
        model = torch.load(path_pt_yolo, map_location='cpu', weights_only=False)
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        model = model.float()
        model.eval()
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'images': {0: 'batch'},
                'output0': {0: 'batch'}
            }
        
        logger.info(f"Exporting to ONNX format: {path_onnx_yolo}")
        logger.info(f"Input shape: {input_shape}, Opset: {opset_version}")
        torch.onnx.export(
            model,
            input_tensor,
            path_onnx_yolo,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            dynamo=False
        )
        
        logger.info(f"Successfully exported YOLO model to {path_onnx_yolo}")
        
    except Exception as e:
        logger.error(f"Failed to export YOLO model: {str(e)}")
        raise RuntimeError(f"Export failed: {str(e)}") from e


def export_text_recognizer(
    path_pt_text_recognizer: str,
    path_onnx_text_recognizer: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 11,
    dynamic_batch: bool = True
) -> None:
    """
    Export text recognition model from PyTorch to ONNX format.
    
    Args:
        path_pt_text_recognizer: Path to PyTorch model (.pt/.pth file)
        path_onnx_text_recognizer: Output path for ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_batch: Whether to support dynamic batch size
    """
    pt_path = Path(path_pt_text_recognizer)
    if not pt_path.exists():
        raise FileNotFoundError(f"Model file not found: {path_pt_text_recognizer}")
    
    onnx_path = Path(path_onnx_text_recognizer)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Loading text recognizer model from {path_pt_text_recognizer}")
        checkpoint = torch.load(path_pt_text_recognizer, map_location='cpu', weights_only=False)
        
        # Extract model from checkpoint
        if isinstance(checkpoint, dict):
            # Try different common keys
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # State dict only - need to recreate model architecture
                logger.info("Loading model architecture for state_dict checkpoint")
                try:
                    from infrastructure.ml.models.ctc.model import ModelTextRecoginizer
                    from infrastructure.ml.models.ctc.pos_patch_embed import PatchEmbedWithPos
                    from infrastructure.ml.models.ctc.transformer import TransformerBlock
                    
                    # Recreate model with same config as training
                    # These should match your training config
                    patch_size = 16
                    img_h, img_w = input_shape[2], input_shape[3]
                    embedding_dim = 256
                    num_classes = 27  # 26 chars + blank
                    
                    patchembedwithpos = PatchEmbedWithPos(
                        patch_size=patch_size,
                        img_h=img_h,
                        img_w=img_w,
                        embedding_dim=embedding_dim
                    )
                    
                    transformer = TransformerBlock(
                        embedding_dim=embedding_dim,
                        num_heads=8,
                        num_layers=6,
                        dropout=0.2
                    )
                    
                    model = ModelTextRecoginizer(
                        patchembedwithpos=patchembedwithpos,
                        transformer=transformer,
                        embedding_dim=embedding_dim,
                        num_classes=num_classes,
                        dropout=0.2,
                        hidden_classifier=512
                    )
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Successfully loaded model state dict")
                except Exception as e:
                    raise ValueError(f"Cannot load model architecture: {str(e)}")
            elif 'state_dict' in checkpoint:
                raise ValueError("state_dict format not fully supported yet")
            else:
                raise ValueError(f"Cannot find model in checkpoint. Available keys: {list(checkpoint.keys())}")
        else:
            model = checkpoint
        
        # Convert to float32
        model = model.float()
        model.eval()
        
        # Disable inplace operations for better ONNX compatibility
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch'},
                'output': {0: 'batch'}
            }
        
        logger.info(f"Exporting to ONNX format: {path_onnx_text_recognizer}")
        logger.info(f"Input shape: {input_shape}, Opset: {opset_version}")
        
        # Use legacy export (disable dynamo) for better compatibility
        torch.onnx.export(
            model,
            input_tensor,
            path_onnx_text_recognizer,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            dynamo=False  # Disable new export mode, use legacy
        )
        
        logger.info(f"Successfully exported text recognizer to {path_onnx_text_recognizer}")
        
    except Exception as e:
        logger.error(f"Failed to export text recognizer: {str(e)}")
        raise RuntimeError(f"Export failed: {str(e)}") from e
