from typing import Dict

import torch
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        
    def update(self, loss: float, predictions=None, targets=None):
        self.losses.append(loss)
        if predictions is not None and targets is not None:
            self.predictions.extend(predictions.cpu().numpy())
            self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        if len(self.predictions) > 0 and len(self.targets) > 0:
            predictions = np.array(self.predictions)
            targets = np.array(self.targets)
            accuracy = (predictions == targets).mean() * 100
            unique_classes = np.unique(targets)
            per_class_acc = {}
            for cls in unique_classes:
                mask = targets == cls
                if mask.sum() > 0:
                    per_class_acc[int(cls)] = (predictions[mask] == targets[mask]).mean() * 100
            
            return {
                'loss': np.mean(self.losses),
                'accuracy': accuracy,
                'per_class_accuracy': per_class_acc
            }
        else:
            return {
                'loss': np.mean(self.losses),
                'accuracy': 0.0,
                'per_class_accuracy': {}
            }