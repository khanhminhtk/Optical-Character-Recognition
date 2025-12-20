"""
Configuration for Text Detection Training
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union
import os
import yaml


@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_path: str = ""
    xml_file: str = "words.xml"
    output_dir: str = ""
    
    # Split ratios
    val_size: float = 0.2
    test_size: float = 0.125
    seed: int = 42
    shuffle: bool = True
    
    # Class config
    class_names: List[str] = field(default_factory=lambda: ["text"])
    
    # Filter config
    exclude_chars: List[str] = field(default_factory=lambda: ["é", "ñ"])
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)
    
    @property
    def xml_path(self) -> str:
        return os.path.join(self.raw_data_path, self.xml_file)


@dataclass
class TrainConfig:
    """Training configuration"""
    # Model
    model_name: str = "yolov8s"
    pretrained: bool = True
    weights: Optional[str] = None
    
    # Training params
    epochs: int = 200
    imgsz: int = 1024
    batch_size: int = 16
    
    # Optimizer
    optimizer: str = "auto"
    lr0: float = 0.01
    weight_decay: float = 0.0005
    
    # Augmentation
    augment: bool = True
    
    # Save
    project: str = "models"
    name: str = "yolov8_text_detect"
    save_period: int = 10
    
    # Device
    device: Optional[str] = None  # auto-detect if None


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary"""
        data_config = DataConfig(**config_dict.get("data", {}))
        train_config = TrainConfig(**config_dict.get("train", {}))
        return cls(data=data_config, train=train_config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file"""
        config_dict = {
            'data': {
                'raw_data_path': self.data.raw_data_path,
                'xml_file': self.data.xml_file,
                'output_dir': self.data.output_dir,
                'val_size': self.data.val_size,
                'test_size': self.data.test_size,
                'seed': self.data.seed,
                'shuffle': self.data.shuffle,
                'class_names': self.data.class_names,
                'exclude_chars': self.data.exclude_chars,
            },
            'train': {
                'model_name': self.train.model_name,
                'pretrained': self.train.pretrained,
                'weights': self.train.weights,
                'epochs': self.train.epochs,
                'imgsz': self.train.imgsz,
                'batch_size': self.train.batch_size,
                'optimizer': self.train.optimizer,
                'lr0': self.train.lr0,
                'weight_decay': self.train.weight_decay,
                'augment': self.train.augment,
                'project': self.train.project,
                'name': self.train.name,
                'save_period': self.train.save_period,
                'device': self.train.device,
            }
        }
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def merge_with_args(self, args) -> "Config":
        """Merge config with command line arguments (args override config)"""
        # Override data config
        if hasattr(args, 'raw_data') and args.raw_data:
            self.data.raw_data_path = args.raw_data
        if hasattr(args, 'output_dir') and args.output_dir:
            self.data.output_dir = args.output_dir
        if hasattr(args, 'xml_file') and args.xml_file:
            self.data.xml_file = args.xml_file
            
        # Override train config
        if hasattr(args, 'model') and args.model:
            self.train.model_name = args.model
        if hasattr(args, 'epochs') and args.epochs:
            self.train.epochs = args.epochs
        if hasattr(args, 'imgsz') and args.imgsz:
            self.train.imgsz = args.imgsz
        if hasattr(args, 'batch_size') and args.batch_size:
            self.train.batch_size = args.batch_size
        if hasattr(args, 'device') and args.device:
            self.train.device = args.device
        if hasattr(args, 'project') and args.project:
            self.train.project = args.project
        if hasattr(args, 'name') and args.name:
            self.train.name = args.name
        if hasattr(args, 'weights') and args.weights:
            self.train.weights = args.weights
            
        return self
