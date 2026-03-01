import os
from typing import Optional, Dict, Any, Union
from pathlib import Path

from ultralytics import YOLO

from configs.training.detection_config.config import TrainConfig


class YOLOTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self._model: Optional[YOLO] = None
        self._results = None
        self._metrics = None
    
    def load_model(self, weights_path: Optional[str] = None) -> "YOLOTrainer":
        if weights_path and os.path.exists(weights_path):
            self._model = YOLO(weights_path)
            print(f"Loaded model from {weights_path}")
        else:
            model_yaml = f"{self.config.model_name}.yaml"
            model_pt = f"{self.config.model_name}.pt"
            
            if self.config.pretrained:
                self._model = YOLO(model_yaml).load(model_pt)
                print(f"Loaded pretrained {self.config.model_name}")
            else:
                self._model = YOLO(model_yaml)
                print(f"Initialized {self.config.model_name} from scratch")
        
        return self
    
    def train(self, data_yaml: str, **kwargs) -> "YOLOTrainer":
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        train_args = {
            'data': data_yaml,
            'epochs': self.config.epochs,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch_size,
            'optimizer': self.config.optimizer,
            'lr0': self.config.lr0,
            'weight_decay': self.config.weight_decay,
            'augment': self.config.augment,
            'project': self.config.project,
            'name': self.config.name,
            'save_period': self.config.save_period,
        }
        
        if self.config.device:
            train_args['device'] = self.config.device
            
        train_args.update(kwargs)
        
        print(f"Starting training with config: epochs={train_args['epochs']}, imgsz={train_args['imgsz']}")
        self._results = self._model.train(**train_args)
        
        return self
    
    def validate(self, **kwargs) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        val_args = {
            'project': self.config.project,
            'name': f"{self.config.name}_val",
        }
        val_args.update(kwargs)
        
        self._metrics = self._model.val(**val_args)
        
        return {
            'mAP50': self._metrics.box.map50,
            'mAP50-95': self._metrics.box.map,
            'precision': self._metrics.box.mp,
            'recall': self._metrics.box.mr,
        }
    
    def predict(
        self, 
        source: Union[str, Path, list],
        conf: float = 0.25,
        save: bool = False,
        **kwargs
    ) -> list:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = self._model(source, conf=conf, save=save, **kwargs)
        return results
    
    def save(self, save_path: str) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._model.save(save_path)
        print(f"Model saved to {save_path}")
        return save_path
    
    def export(self, format: str = "onnx", **kwargs) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self._model.export(format=format, **kwargs)
    
    @property
    def model(self) -> Optional[YOLO]:
        return self._model
    
    @property
    def results(self):
        return self._results
    
    @property
    def metrics(self):
        return self._metrics
    
    @property
    def best_weights_path(self) -> Optional[str]:
        if self._results is None:
            return None
        return os.path.join(
            self.config.project,
            self.config.name,
            'weights',
            'best.pt'
        )
