"""
Visualization utilities for text detection
"""
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from pathlib import Path


class DetectionVisualizer:
    """Visualize detection results"""
    
    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        thickness: int = 2,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        text_bg_color: Tuple[int, int, int] = (0, 255, 0)
    ):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.box_color = box_color
        self.text_color = text_color
        self.text_bg_color = text_bg_color
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        confidence: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw a single bounding box on image.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box (x1, y1, x2, y2)
            label: Label text
            confidence: Confidence score
            
        Returns:
            Image with drawn bbox
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), self.box_color, self.thickness)
        
        # Prepare text
        if confidence is not None:
            text = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"
        else:
            text = label
        
        if text:
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                text, self.font, self.font_scale, self.thickness
            )
            
            # Draw text background
            cv2.rectangle(
                image,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 5, y1),
                self.text_bg_color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image, text,
                (x1 + 2, y1 - 5),
                self.font, self.font_scale,
                self.text_color, self.thickness
            )
        
        return image
    
    def visualize_predictions(
        self,
        image_path: Union[str, Path],
        predictions: List[dict],
        conf_threshold: float = 0.5,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize YOLO predictions on image.
        
        Args:
            image_path: Path to input image
            predictions: List of prediction dicts with 'box', 'confidence', 'class'
            conf_threshold: Minimum confidence to display
            show: Whether to display image
            save_path: Path to save result
            
        Returns:
            Annotated image
        """
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for pred in predictions:
            conf = pred.get('confidence', 0)
            if conf < conf_threshold:
                continue
            
            box = pred['box']
            bbox = (box['x1'], box['y1'], box['x2'], box['y2'])
            label = pred.get('name', '')
            
            img = self.draw_bbox(img, bbox, label, conf)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        
        return img
    
    def visualize_yolo_results(
        self,
        image_path: Union[str, Path],
        yolo_results,
        conf_threshold: float = 0.5,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize YOLO results object directly.
        
        Args:
            image_path: Path to input image
            yolo_results: YOLO Results object
            conf_threshold: Minimum confidence threshold
            show: Whether to display
            save_path: Path to save
            
        Returns:
            Annotated image
        """
        predictions = json.loads(yolo_results[0].to_json())
        return self.visualize_predictions(
            image_path, predictions, conf_threshold, show, save_path
        )


def visualize_dataset_sample(
    image_path: str,
    label_path: str,
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize a sample from YOLO dataset.
    
    Args:
        image_path: Path to image
        label_path: Path to YOLO format label file
        class_names: List of class names
        figsize: Figure size
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:])
        
        # Convert to pixel coordinates
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if class_names:
            label_text = class_names[class_id]
            cv2.putText(img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Sample: {Path(image_path).name}")
    plt.show()
