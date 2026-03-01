import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import yaml

from ultralytics import YOLO

from infrastructure.ml.models.yolo.config.trainconfig import TrainConfigs


class Trainer:
    def __init__(self, config: TrainConfigs):
        self.config: TrainConfigs = config
        self._model: Optional[YOLO]
        self._results = None
        self._metrics = None

    def load_model(self, weights_path: str):
        if weights_path and os.path.exists(weights_path):
            self._model = YOLO(model=weights_path)
        else:
            model_yaml = f"{self.config.model_name}.yaml"
            model_pt = f"{self.config.model_name}.pt"
            if self.config.pretrained:
                self._model = YOLO(model=model_yaml).load(model_pt)
            else:
                self._model = YOLO(model_pt)

    def train(self, data_yaml: str):
        train_config = {
            "data": data_yaml,
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.imgsz,
            "device": self.config.device,
            "workers": self.config.workers,
            "optimizer": self.config.optimizer,
            "lr0": self.config.lr0,
            "patience": self.config.patience,
            "augment": self.config.augment
        }
        print(f"Starting training with config: epochs={train_config['epochs']}, imgsz={train_config['imgsz']}")
        self._results = self._model.train(**train_config)

    def validate(self, **kwargs):
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
        conf: float = 0.5,
        save: bool = False,
        device: str = "cpu",
        stream: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> list:
        results = self._model(source, conf=conf, save=save, device=device, stream=stream, verbose=verbose,**kwargs)
        return results
    
    def get_bboxes(self, results) -> list:
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])]
                }
                all_detections.append(detection)
        
        return all_detections
    
    def save(self, save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._model.save(save_path)
        print(f"Model saved to {save_path}")
        return save_path
    
    def export(self, format: str = "onnx", **kwargs) -> str:
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
        return os.path.join(
            self.config.project,
            self.config.name,
            'weights',
            'best.pt'
        )
    
def load_training_config(yaml_path: str):
    train_config = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key, item in config.items():
        if isinstance(item, list):
            for elem in item:
                if isinstance(elem, dict):
                    train_config.update(elem)
        elif isinstance(item, dict):
            train_config.update(item)
        else:
            train_config[key] = item

    return TrainConfigs.from_dict(dictionary=train_config)
